"""
losses.py — Loss functions for On-autoPilot.

  1. sliced_wasserstein() — Efficient Monte Carlo Sliced Wasserstein Distance
     for measuring distributional shift between INR latent representations.
     Used in the outer loop of the Self-Tuning Engine.

  2. TSContrastiveLoss — Time-Series Self-Supervised Contrastive Loss (TS-SCL)
     that promotes temporally proximate windows as positives and augmented
     anomalous windows as negatives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sliced Wasserstein Distance
# ---------------------------------------------------------------------------

def sliced_wasserstein(
    Z_src: torch.Tensor,
    Z_tgt: torch.Tensor,
    n_projections: int = 128,
) -> torch.Tensor:
    """
    Monte Carlo Sliced Wasserstein Distance (SWD).

    Projects both distributions onto `n_projections` random unit vectors,
    sorts the projected values, and returns the mean squared difference
    between sorted projections.

    This is fully differentiable w.r.t. both Z_src and Z_tgt.

    Parameters
    ----------
    Z_src        : (N1, D) — source distribution (e.g., training latents)
    Z_tgt        : (N2, D) — target distribution (e.g., augmented latents)
    n_projections: int     — Monte Carlo samples (default 128)

    Returns
    -------
    swd : scalar Tensor — Sliced Wasserstein Distance
    """
    D = Z_src.shape[-1]
    device = Z_src.device

    # Sample random unit directions: (n_projections, D)
    directions = torch.randn(n_projections, D, device=device)
    directions = F.normalize(directions, dim=1)

    # Project: (N, n_projections)
    proj_src = Z_src @ directions.T  # (N1, n_projections)
    proj_tgt = Z_tgt @ directions.T  # (N2, n_projections)

    # Sort along sample dimension
    proj_src_sorted = proj_src.sort(dim=0).values   # (N1, n_projections)
    proj_tgt_sorted = proj_tgt.sort(dim=0).values   # (N2, n_projections)

    # Align lengths via linear interpolation if N1 ≠ N2
    if proj_src_sorted.shape[0] != proj_tgt_sorted.shape[0]:
        n = max(proj_src_sorted.shape[0], proj_tgt_sorted.shape[0])
        proj_src_sorted = F.interpolate(
            proj_src_sorted.T.unsqueeze(0), size=n, mode="linear", align_corners=True
        ).squeeze(0).T
        proj_tgt_sorted = F.interpolate(
            proj_tgt_sorted.T.unsqueeze(0), size=n, mode="linear", align_corners=True
        ).squeeze(0).T

    swd = (proj_src_sorted - proj_tgt_sorted).pow(2).mean()
    return swd


# ---------------------------------------------------------------------------
# Time-Series Self-Supervised Contrastive Loss (TS-SCL)
# ---------------------------------------------------------------------------

class TSContrastiveLoss(nn.Module):
    """
    Time-Series Contrastive Loss defining:
      - Positives z⁺ : latents from temporally nearby windows (within `pos_radius`)
      - Negatives z⁻ : latents computed from anomaly-augmented windows

    Loss per anchor z:
        L = max(0, ||z - z⁺||² - ||z - z⁻||² + ε)

    This is the hinge (margin-based) contrastive form.

    Parameters
    ----------
    margin : float — ε in the hinge loss (default 1.0)
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        z_anchor: torch.Tensor,
        z_pos: torch.Tensor,
        z_neg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_anchor : (B, D) — anchor latents (clean windows)
        z_pos    : (B, D) — positive latents (temporally proximate clean windows)
        z_neg    : (B, D) — negative latents (latents from augmented anomalous windows)

        Returns
        -------
        loss : scalar Tensor
        """
        d_pos = (z_anchor - z_pos).pow(2).sum(dim=-1)   # (B,)
        d_neg = (z_anchor - z_neg).pow(2).sum(dim=-1)   # (B,)
        loss = F.relu(d_pos - d_neg + self.margin)       # (B,)
        return loss.mean()


# ---------------------------------------------------------------------------
# Reconstruction loss
# ---------------------------------------------------------------------------

def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    MSE between original window x and INR reconstruction x_hat.

    Parameters
    ----------
    x     : (B, W, d)
    x_hat : (B, W, d)

    Returns
    -------
    loss : scalar Tensor
    """
    return F.mse_loss(x_hat, x)
