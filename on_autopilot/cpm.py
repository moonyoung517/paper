"""
cpm.py — Continual Prompting Module (CPM) for industrial MTS.

Key differences from the image-based CPM (Liu et al., 2024)
------------------------------------------------------------
1. **Unsupervised key generation** via Furthest Point Sampling (FPS) on
   unlabelled MTS streams — no contrastive labels required.
2. **Multi-channel normality standard** Kn_t built via greedy CoreSet
   sampling in the INR latent space Z, capturing inter-channel correlations.
3. **No explicit task boundary** — regime transition detected purely by
   cosine similarity falling below threshold τ_drift.

Memory bank entry: (k_t, p_t, Kn_t)
  k_t  : task key  — representative latent vector (FPS on Z)
  p_t  : prompt    — learnable task-specific bias (PromptModule weights)
  Kn_t : normality standard — CoreSet subset of Z_trn
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .models import PromptModule


# ---------------------------------------------------------------------------
# Furthest Point Sampling
# ---------------------------------------------------------------------------

def furthest_point_sampling(
    points: torch.Tensor, n_samples: int
) -> torch.Tensor:
    """
    Greedy Furthest Point Sampling (FPS).

    Iteratively selects the point with the maximum minimum distance to
    the already-selected set.  O(N · k) runtime.

    Parameters
    ----------
    points   : (N, D)
    n_samples: int  — number of points to select (k)

    Returns
    -------
    selected : (n_samples, D)
    """
    N, D = points.shape
    n_samples = min(n_samples, N)

    selected_idx = torch.zeros(n_samples, dtype=torch.long, device=points.device)
    # Initialise: pick a random seed
    selected_idx[0] = torch.randint(0, N, (1,)).item()

    # Distances from each point to the nearest selected point
    min_dists = torch.full((N,), float("inf"), device=points.device)

    for i in range(1, n_samples):
        last = points[selected_idx[i - 1]].unsqueeze(0)  # (1, D)
        dists = torch.cdist(points, last).squeeze(-1)     # (N,)
        min_dists = torch.minimum(min_dists, dists)
        selected_idx[i] = min_dists.argmax()

    return points[selected_idx]  # (n_samples, D)


# ---------------------------------------------------------------------------
# Greedy CoreSet Sampling
# ---------------------------------------------------------------------------

def coreset_sampling(
    embeddings: torch.Tensor, n_samples: int
) -> torch.Tensor:
    """
    Greedy CoreSet selection in embedding space (Roth et al., 2022).

    Iteratively adds the point with maximum distance to the current coreset
    until `n_samples` points are selected.

    Parameters
    ----------
    embeddings : (N, D)  — latent vectors Z_trn
    n_samples  : int     — coreset size

    Returns
    -------
    coreset : (n_samples, D)
    """
    N, D = embeddings.shape
    n_samples = min(n_samples, N)

    # Start with the embedding closest to the centroid
    centroid = embeddings.mean(0, keepdim=True)  # (1, D)
    init_idx = torch.cdist(embeddings, centroid).squeeze(-1).argmin().item()

    selected = [int(init_idx)]
    min_dists = torch.cdist(embeddings, embeddings[selected]).squeeze(-1)  # (N,)

    for _ in range(1, n_samples):
        # Point with max dist to current coreset
        new_idx = min_dists.argmax().item()
        selected.append(int(new_idx))
        new_dists = torch.cdist(
            embeddings, embeddings[[new_idx]]
        ).squeeze(-1)  # (N,)
        min_dists = torch.minimum(min_dists, new_dists)

    return embeddings[selected]  # (n_samples, D)


# ---------------------------------------------------------------------------
# CPM Memory Bank
# ---------------------------------------------------------------------------

class CPMMemoryBank:
    """
    Continual Prompting Module memory bank.

    Stores a list of (k_t, p_t, Kn_t) triplets, one per detected regime.

    Attributes
    ----------
    tau_drift   : float — cosine similarity threshold for drift detection
    key_dim     : int   — dimensionality of task keys k_t
    prompt_dim  : int   — dimensionality of prompt embeddings p_t
    fps_samples : int   — number of FPS samples for key generation
    coreset_size: int   — CoreSet size for normality standard Kn_t
    device      : torch.device
    """

    def __init__(
        self,
        key_dim: int,
        prompt_dim: int,
        tau_drift: float = 0.65,
        fps_samples: int = 32,
        coreset_size: int = 64,
        device: torch.device = torch.device("cpu"),
    ):
        self.key_dim = key_dim
        self.prompt_dim = prompt_dim
        self.tau_drift = tau_drift
        self.fps_samples = fps_samples
        self.coreset_size = coreset_size
        self.device = device

        # Bank: list of dicts with keys 'key', 'prompt', 'kn', 'theta'
        self._bank: List[dict] = []

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._bank)

    def is_empty(self) -> bool:
        return len(self._bank) == 0

    def generate_key(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Generate task key k_query from INR latent Z via Furthest Point Sampling.

        Parameters
        ----------
        Z : (N, D) or (B, W, D)  INR latent embeddings

        Returns
        -------
        k_query : (D,)  mean of FPS-selected points
        """
        if Z.dim() == 3:
            Z = Z.reshape(-1, Z.shape[-1])  # (B*W, D)
        pts = furthest_point_sampling(Z.detach(), self.fps_samples)  # (fps, D)
        return pts.mean(0)  # (D,)

    def detect_drift(self, k_query: torch.Tensor) -> Tuple[bool, int, float]:
        """
        Test whether the query key matches any stored key above τ_drift.

        Parameters
        ----------
        k_query : (D,)

        Returns
        -------
        drift    : bool  — True if no stored key exceeds τ_drift
        best_idx : int   — index of best matching entry (-1 if bank empty)
        s_max    : float — maximum cosine similarity found
        """
        if self.is_empty():
            return True, -1, 0.0

        keys = torch.stack([e["key"] for e in self._bank], dim=0)  # (M, D)
        sims = F.cosine_similarity(
            k_query.unsqueeze(0).expand(len(keys), -1), keys, dim=1
        )  # (M,)
        s_max, best_idx = sims.max(0)
        drift = s_max.item() < self.tau_drift
        return drift, int(best_idx.item()), float(s_max.item())

    def retrieve(self, k_query: torch.Tensor) -> Optional[dict]:
        """
        Retrieve the most similar bank entry for the query key.
        Returns None if the bank is empty.
        """
        if self.is_empty():
            return None
        _, best_idx, _ = self.detect_drift(k_query)
        return self._bank[best_idx]

    def retrieve_latest_theta(self) -> Optional[dict]:
        """Return the warm-start θ_{t-1} from the most recently stored entry."""
        if self.is_empty():
            return None
        return self._bank[-1].get("theta")

    def register(
        self,
        key: torch.Tensor,
        prompt: PromptModule,
        kn: torch.Tensor,
        theta_state: Optional[dict] = None,
    ) -> None:
        """
        Store a new (k_t, p_t, Kn_t, θ*) entry in the bank.

        Parameters
        ----------
        key         : (D,)     task key
        prompt      : PromptModule with learned weights (p_t)
        kn          : (C, D)   CoreSet normality standard
        theta_state : dict     state_dict snapshot of detector θ* (optional)
        """
        entry = {
            "key": key.detach().cpu(),
            "prompt": {k: v.detach().cpu() for k, v in prompt.state_dict().items()},
            "kn": kn.detach().cpu(),
            "theta": theta_state,
        }
        self._bank.append(entry)

    def build_kn(self, Z_trn: torch.Tensor) -> torch.Tensor:
        """
        Build normality standard Kn_t via CoreSet sampling of Z_trn.

        Parameters
        ----------
        Z_trn : (N, D) or (B, W, D)

        Returns
        -------
        kn : (coreset_size, D)
        """
        if Z_trn.dim() == 3:
            Z_trn = Z_trn.reshape(-1, Z_trn.shape[-1])
        return coreset_sampling(Z_trn.detach(), self.coreset_size)

    # ------------------------------------------------------------------
    # Anomaly scoring helper
    # ------------------------------------------------------------------

    def latent_distance(
        self, z_t: torch.Tensor, kn: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute min distance from z_t to normality standard Kn_t.

        Parameters
        ----------
        z_t : (B, D)  or (B, W, D)
        kn  : (C, D)

        Returns
        -------
        dist : (B,) or (B, W)
        """
        squeeze = z_t.dim() == 2
        if squeeze:
            z_t = z_t.unsqueeze(1)  # (B, 1, D)

        # (B, W, D) vs (C, D) → (B, W, C)
        dists = torch.cdist(z_t, kn.to(z_t.device))  # (B, W, C)
        min_dist = dists.min(dim=-1).values  # (B, W)

        return min_dist.squeeze(1) if squeeze else min_dist
