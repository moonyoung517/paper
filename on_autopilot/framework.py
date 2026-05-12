"""
framework.py — On-autoPilot: Unsupervised Continual Anomaly Detection.

Implements Algorithm 1 from the paper:

  Phase 1  — Task Identification
    Compute query key k_{query} via FPS on Z_t
    S_max = max_i cos(k_{query}, k_i^{CPM})
    If S_max < τ_drift  →  new regime detected

  Phase 2  — Self-Tuning (bi-level optimisation)
    Warm-start θ ← θ_{t-1}  (from CPM bank)
    Run Self-Tuning Engine: (θ*, a*) on D_t-normal

  Phase 3  — Knowledge Consolidation
    k_t    = FPS(Z_t)
    Kn_t   = CoreSet(Z_t)
    CPM.register(k_t, p_t, Kn_t, θ*)

Anomaly score (Section 4.5):
    S(t) = λ · ||x_t - x̂_t||²  +  (1-λ) · min_{m ∈ Kn_t} dist(z_t, m)

Saliency / root-cause (Section 4.5):
    ĝ = ∇_x S(t)             (gradient w.r.t. input)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augmentation import DifferentiableAugmentation
from .cpm import CPMMemoryBank
from .models import FrozenLLMEncoder, GroupBasedINR, PromptModule, WeightPredictor
from .self_tuning import SelfTuningEngine
from .simple_feat_extractor import SimpleFeatExtractor


class OnAutoPilot(nn.Module):
    """
    On-autoPilot: Frozen LLM-Guided Self-Tuning Framework for
    Unsupervised Continual Anomaly Detection in industrial MTS.

    Parameters
    ----------
    input_dim     : int   — number of sensor channels (d)
    seq_len       : int   — sliding window length (W)
    llm_hidden    : int   — LLM hidden dimension (default 768 for GPT-2 small)
    llm_extract_layer : int — which GPT-2 hidden layer to extract (default 5)
    inr_global_hidden : int — global SIREN hidden dim (default 64)
    inr_group_hidden  : int — per-group SIREN hidden dim (default 32)
    num_groups    : int   — number of channel groups for GroupBasedINR
    prompt_dim    : int   — prompt vector dimension
    tau_drift     : float — cosine-sim threshold for drift detection
    lambda_score  : float — λ weighting between recon and latent score
    fps_samples   : int   — FPS samples for key generation
    coreset_size  : int   — CoreSet size for normality standard
    inner_lr / outer_lr / inner_steps / outer_steps : self-tuning hyperparms
    device        : torch.device
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        llm_hidden: int = 768,
        use_llm: bool = True,
        llm_extract_layer: int = 5,
        inr_global_hidden: int = 64,
        inr_group_hidden: int = 32,
        num_groups: int = 4,
        prompt_dim: int = 64,
        tau_drift: float = 0.65,
        lambda_score: float = 0.5,
        fps_samples: int = 32,
        coreset_size: int = 64,
        inner_lr: float = 1e-3,
        outer_lr: float = 1e-3,
        inner_steps: int = 5,
        outer_steps: int = 50,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_score = lambda_score
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.use_llm = use_llm
        self.feat_dim = llm_hidden
        if prompt_dim != inr_global_hidden:
            raise ValueError("prompt_dim must equal inr_global_hidden for latent prompt injection")

        # ---- Core modules --------------------------------------------------
        if use_llm:
            self.llm_encoder = FrozenLLMEncoder(
                input_dim=input_dim,
                seq_len=seq_len,
                llm_hidden=llm_hidden,
                extract_layer=llm_extract_layer,
            )
        else:
            self.llm_encoder = SimpleFeatExtractor(
                input_dim=input_dim,
                seq_len=seq_len,
                output_dim=llm_hidden,
            )

        self.inr = GroupBasedINR(
            input_dim=input_dim,
            num_groups=num_groups,
            global_hidden=inr_global_hidden,
            group_hidden=inr_group_hidden,
        )

        self.weight_predictor = WeightPredictor(
            llm_dim=self.feat_dim,
            inr_hidden=inr_global_hidden,
        )

        self.prompt = PromptModule(prompt_dim=prompt_dim)

        # ---- CPM Memory Bank -----------------------------------------------
        self.cpm_bank = CPMMemoryBank(
            key_dim=inr_global_hidden,
            prompt_dim=prompt_dim,
            tau_drift=tau_drift,
            fps_samples=fps_samples,
            coreset_size=coreset_size,
            device=self.device,
        )

        # ---- Augmentation --------------------------------------------------
        self.augmentation = DifferentiableAugmentation(
            seq_len=seq_len, n_channels=input_dim
        )

        # ---- Self-Tuning Engine -------------------------------------------
        self.self_tuning_engine = SelfTuningEngine(
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            inner_steps=inner_steps,
            outer_steps=outer_steps,
            device=self.device,
        )

        self.to(self.device)

    # ------------------------------------------------------------------
    # Time coordinate helper
    # ------------------------------------------------------------------

    def _time_coords(self, batch_size: int) -> torch.Tensor:
        """Return normalised time coordinates t ∈ [0, 1] of shape (B, W, 1)."""
        t = torch.linspace(0, 1, self.seq_len, device=self.device)
        t = t.view(1, self.seq_len, 1).expand(batch_size, -1, -1)
        return t

    # ------------------------------------------------------------------
    # Forward pass — compute anomaly scores for a batch of windows
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        kn: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x  : (B, W, d) — input windows
        kn : (C, D) or None — normality standard Kn_t from CPM bank

        Returns
        -------
        scores : (B,)       anomaly score S(t)
        z      : (B, D)     mean INR latent
        x_hat  : (B, W, d)  INR reconstruction
        """
        x = x.to(self.device)
        B = x.shape[0]
        t = self._time_coords(B)

        # LLM feature amplification
        z_llm = self.llm_encoder(x)             # (B, W, llm_hidden)
        omega = self.weight_predictor(z_llm)    # (B, inr_hidden)
        omega = self.prompt(omega)              # (B, inr_hidden)

        # INR reconstruction
        x_hat = self.inr(t, omega=omega)        # (B, W, d)

        # INR latent (mean over W)
        z_seq = self.inr.get_latent(t, omega=omega)  # (B, W, D)
        z = z_seq.mean(1)                       # (B, D)

        # Reconstruction score
        recon_score = (x - x_hat).pow(2).mean(dim=[1, 2])  # (B,)

        # Latent-distance score
        if kn is not None:
            kn = kn.to(self.device)
            latent_score = self.cpm_bank.latent_distance(z, kn)  # (B,)
        else:
            latent_score = torch.zeros_like(recon_score)

        # Combined score (Eq. in Section 4.5)
        scores = self.lambda_score * recon_score + (1 - self.lambda_score) * latent_score

        return scores, z, x_hat

    # ------------------------------------------------------------------
    # Gradient saliency for explainable root-cause localisation
    # ------------------------------------------------------------------

    def compute_saliency(
        self,
        x: torch.Tensor,
        kn: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute channel-wise saliency ĝ = ∇_x S(t).

        Parameters
        ----------
        x  : (B, W, d)
        kn : (C, D) normality standard

        Returns
        -------
        g_hat : (B, d) mean |gradient| over time dimension
        """
        x = x.to(self.device).requires_grad_(True)
        scores, _, _ = self.forward(x, kn=kn)
        total_score = scores.sum()
        total_score.backward()
        g = x.grad.abs().mean(dim=1)  # (B, d) — average over W
        return g

    # ------------------------------------------------------------------
    # Algorithm 1: autonomous stream processing
    # ------------------------------------------------------------------

    def detect_and_adapt(
        self,
        x_stream: torch.Tensor,
        window_size: int = 100,
        stride: int = 1,
        adapt_buffer: int = 200,
    ) -> List[dict]:
        """
        On-autoPilot stream inference (Algorithm 1).

        Processes `x_stream` sample-by-sample (or in windows).
        Triggers self-tuning + CPM registration on detected regime transitions.

        Parameters
        ----------
        x_stream    : (T, d) — raw MTS time series
        window_size : int    — sliding window W (must match self.seq_len)
        stride      : int    — sliding stride
        adapt_buffer: int    — number of windows to accumulate before adapting

        Returns
        -------
        results : list of dicts with keys
                  't'     : time-step index
                  'score' : scalar anomaly score
                  'task'  : current task index (-1 on first window)
        """
        assert window_size == self.seq_len, (
            f"window_size {window_size} must match seq_len {self.seq_len}"
        )

        T, d = x_stream.shape
        results = []
        buffer_windows: List[torch.Tensor] = []

        current_task_idx = -1
        current_kn: Optional[torch.Tensor] = None

        t_ptr = 0
        while t_ptr + window_size <= T:
            xw = x_stream[t_ptr:t_ptr + window_size].unsqueeze(0)  # (1, W, d)

            # ---- Phase 1: Task Identification ----------------------------
            t_coords = self._time_coords(1)
            z_llm = self.llm_encoder(xw.to(self.device))
            omega = self.weight_predictor(z_llm)        # (1, D)
            z_seq = self.inr.get_latent(t_coords, omega=omega)  # (1, W, D)
            Z_flat = z_seq.reshape(-1, z_seq.shape[-1]) # (W, D)

            k_query = self.cpm_bank.generate_key(Z_flat)
            drift, best_idx, s_max = self.cpm_bank.detect_drift(k_query)

            if drift:
                # ---- Regime transition detected -------------------------
                # Collect buffer for adaptation
                buffer_windows.append(xw.squeeze(0))

                if len(buffer_windows) >= adapt_buffer:
                    x_buf = torch.stack(buffer_windows, dim=0)  # (N, W, d)
                    n_train = max(1, int(0.8 * x_buf.shape[0]))
                    x_train = x_buf[:n_train]
                    x_val = x_buf[n_train:] if n_train < x_buf.shape[0] else x_buf[-1:]
                    t_train = self._time_coords(x_train.shape[0])
                    t_val = self._time_coords(x_val.shape[0])

                    # New regime starts with a fresh task-specific prompt p_new.
                    p_new = PromptModule(prompt_dim=self.prompt.prompt.shape[0]).to(self.device)
                    self.augmentation.reset_parameters()

                    # ---- Phase 2: Self-Tuning ---------------------------
                    theta_warm = self.cpm_bank.retrieve_latest_theta()
                    self.self_tuning_engine.run(
                        inr=self.inr,
                        weight_pred=self.weight_predictor,
                        prompt=p_new,
                        llm_encoder=self.llm_encoder,
                        augmentation=self.augmentation,
                        x_train=x_train.to(self.device),
                        t_coords=t_train.to(self.device),
                        x_val=x_val.to(self.device),
                        t_val=t_val.to(self.device),
                        theta_init=theta_warm,
                    )

                    # ---- Phase 3: Knowledge Consolidation ---------------
                    with torch.no_grad():
                        z_llm_buf = self.llm_encoder(x_train.to(self.device))
                        omega_buf = self.weight_predictor(z_llm_buf)
                        omega_buf = p_new(omega_buf)
                        z_seq_buf = self.inr.get_latent(t_train.to(self.device), omega=omega_buf)
                        Z_buf = z_seq_buf.reshape(-1, z_seq_buf.shape[-1])

                    k_new = k_query.detach()
                    kn_new = self.cpm_bank.build_kn(Z_buf)
                    theta_snap = self.self_tuning_engine.snapshot_theta(
                        self.inr, self.weight_predictor
                    )
                    self.cpm_bank.register(k_new, p_new, kn_new, theta_snap)

                    # Activate the newly learned prompt for subsequent scoring.
                    self.prompt.load_state_dict(p_new.state_dict(), strict=False)

                    current_task_idx = len(self.cpm_bank) - 1
                    current_kn = kn_new
                    buffer_windows.clear()

            else:
                # Retrieve matched task
                current_task_idx = best_idx
                entry = self.cpm_bank._bank[best_idx]
                current_kn = entry["kn"].to(self.device) if entry["kn"] is not None else None
                if entry.get("prompt") is not None:
                    self.prompt.load_state_dict(entry["prompt"], strict=False)

            # ---- Anomaly scoring -----------------------------------------
            with torch.no_grad():
                score, _, _ = self.forward(xw.to(self.device), kn=current_kn)

            results.append({
                "t": t_ptr,
                "score": score.item(),
                "task": current_task_idx,
                "s_max": s_max,
            })

            t_ptr += stride

        return results
