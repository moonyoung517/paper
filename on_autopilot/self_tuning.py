"""
self_tuning.py — Differentiable Self-Tuning Engine.

Implements the bi-level optimisation described in Section 4.2:

  Inner loop  (θ update):
    θ* = argmin_{θ} L_trn(θ; D_trn, D_aug)
       = L_recon + L_SCL  (supervised by augmented negatives)

  Outer loop  (a update):
        a* = argmin_{a} SWD(Z_trn ∪ Z_aug(a), Z_val)
       ← gradient through the inner loop via second-order grads

The outer loop uses Sliced Wasserstein Distance as a distribution-level
alignment signal against an unlabeled validation stream in INR latent
space.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .losses import TSContrastiveLoss, reconstruction_loss, sliced_wasserstein


class SelfTuningEngine:
    """
    Differentiable Self-Tuning Engine (bi-level optimisation).

    Parameters
    ----------
    inner_lr     : float — inner-loop learning rate for θ
    outer_lr     : float — outer-loop learning rate for augmentation params a
    inner_steps  : int   — number of inner gradient steps per outer update
    outer_steps  : int   — total outer-loop iterations
    scl_margin   : float — margin ε in TS-SCL
    device       : torch.device
    """

    def __init__(
        self,
        inner_lr: float = 1e-3,
        outer_lr: float = 1e-3,
        inner_steps: int = 5,
        outer_steps: int = 50,
        scl_margin: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.outer_steps = outer_steps
        self.scl_loss = TSContrastiveLoss(margin=scl_margin)
        self.device = device

    # ------------------------------------------------------------------
    # Inner loop: update detector θ on (D_trn, D_aug)
    # ------------------------------------------------------------------

    def inner_loop(
        self,
        inr: nn.Module,
        weight_pred: nn.Module,
        prompt: nn.Module,
        llm_encoder: nn.Module,
        augmentation: nn.Module,
        x_clean: torch.Tensor,
        x_pos: torch.Tensor,
        t_coords: torch.Tensor,
        create_graph: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One inner-loop pass.

        Computes L_trn = L_recon + L_SCL on (x_clean, x_aug) and
        returns the combined loss (with graph kept for outer-loop
        second-order derivatives when `create_graph=True`).

        Parameters
        ----------
        inr          : GroupBasedINR
        weight_pred  : WeightPredictor
        prompt       : PromptModule
        llm_encoder  : FrozenLLMEncoder
        augmentation : DifferentiableAugmentation
        x_clean      : (B, W, d) normal windows (D_trn)
        x_pos        : (B, W, d) temporally adjacent windows (positives for SCL)
        t_coords     : (B, W, 1) normalised time coordinates
        create_graph : keep second-order graph for outer-loop grad

        Returns
        -------
        loss     : scalar Tensor
        z_anchor : (B, D) latent used by outer loop
        """
        # --- Augmented windows (anomalous negatives) --------------------
        x_aug = augmentation(x_clean)  # (B, W, d)

        # --- LLM features for clean windows -----------------------------
        z_llm_clean = llm_encoder(x_clean)   # (B, W, llm_dim)
        omega_clean = weight_pred(z_llm_clean)  # (B, inr_hidden)
        omega_clean = prompt(omega_clean)

        # --- LLM features for augmented windows (negatives) -------------
        z_llm_aug = llm_encoder(x_aug)    # (B, W, llm_dim)
        omega_aug = weight_pred(z_llm_aug)  # (B, inr_hidden)
        omega_aug = prompt(omega_aug)

        # --- LLM features for positive windows --------------------------
        z_llm_pos = llm_encoder(x_pos)    # (B, W, llm_dim)
        omega_pos = weight_pred(z_llm_pos)  # (B, inr_hidden)
        omega_pos = prompt(omega_pos)

        # --- INR reconstruction on clean windows (with omega) -----------
        x_hat = inr(t_coords, omega=omega_clean)  # (B, W, d)
        l_recon = reconstruction_loss(x_clean, x_hat)

        # --- TS-SCL: get latents with omega ------------------------------
        z_anchor = inr.get_latent(t_coords, omega=omega_clean).mean(1)  # (B, D)

        with torch.no_grad():
            z_pos = inr.get_latent(t_coords, omega=omega_pos).mean(1)  # (B, D)

        # Augmented latent (negative): use omega from augmented input
        z_neg = inr.get_latent(t_coords, omega=omega_aug).mean(1)  # (B, D)

        l_scl = self.scl_loss(z_anchor, z_pos.detach(), z_neg)

        loss = l_recon + l_scl
        return loss, z_anchor

    # ------------------------------------------------------------------
    # Full bi-level run
    # ------------------------------------------------------------------

    def run(
        self,
        inr: nn.Module,
        weight_pred: nn.Module,
        prompt: nn.Module,
        llm_encoder: nn.Module,
        augmentation: nn.Module,
        x_train: torch.Tensor,
        t_coords: torch.Tensor,
        x_val: Optional[torch.Tensor] = None,
        t_val: Optional[torch.Tensor] = None,
        theta_init: Optional[dict] = None,
    ) -> None:
        """
        Run the full bi-level Self-Tuning Engine (Algorithm 1, Phase 2).

        Modifies `inr`, `weight_pred`, and `prompt` weights in-place.
        Modifies `augmentation` magnitude params in-place.

        Parameters
        ----------
        inr, weight_pred, prompt, llm_encoder, augmentation : nn.Module
        x_train    : (N, W, d) collected normal windows for this regime
        t_coords   : (N, W, 1) normalised time coordinates
        x_val      : (M, W, d) unlabeled validation windows for outer loop
        t_val      : (M, W, 1) coordinates for x_val
        theta_init : optional warm-start state_dict for inr + weight_pred
        """
        # Warm-start θ from previous regime
        if theta_init is not None:
            inr.load_state_dict(theta_init.get("inr", inr.state_dict()), strict=False)
            weight_pred.load_state_dict(
                theta_init.get("weight_pred", weight_pred.state_dict()), strict=False
            )

        if x_val is None or t_val is None:
            x_val = x_train
            t_val = t_coords

        inr.to(self.device)
        weight_pred.to(self.device)
        prompt.to(self.device)
        llm_encoder.to(self.device)
        augmentation.to(self.device)

        # Optimisers
        theta_params = (
            list(inr.parameters())
            + list(weight_pred.parameters())
            + list(prompt.parameters())
        )
        opt_theta = optim.Adam(theta_params, lr=self.inner_lr)
        opt_a = optim.Adam(list(augmentation.parameters()), lr=self.outer_lr)

        N = x_train.shape[0]
        M = x_val.shape[0]
        batch_size = min(32, N)
        val_batch_size = min(32, M)

        for outer_step in range(self.outer_steps):
            # Sample mini-batch
            idx = torch.randperm(N)[:batch_size]
            xb = x_train[idx].to(self.device)
            tb = t_coords[idx].to(self.device)

            # Positive: adjacent window (wrap-around)
            pos_idx = (idx + 1) % N
            xb_pos = x_train[pos_idx].to(self.device)

            # ---- Inner loop ------------------------------------------------
            for inner_step in range(self.inner_steps):
                opt_theta.zero_grad()
                # Keep graph only for the final inner step when outer loss is evaluated.
                create_graph = (outer_step % 5 == 0) and (inner_step == self.inner_steps - 1)
                inner_loss, _ = self.inner_loop(
                    inr, weight_pred, prompt, llm_encoder, augmentation,
                    xb, xb_pos, tb,
                    create_graph=create_graph,
                )
                inner_loss.backward(retain_graph=create_graph)
                opt_theta.step()

            # ---- Outer loop (every 5 inner steps) -------------------------
            if outer_step % 5 == 0:
                opt_a.zero_grad()

                # Compute omega for clean windows
                with torch.no_grad():
                    z_llm_clean = llm_encoder(xb)
                    omega_clean = weight_pred(z_llm_clean)
                    omega_clean = prompt(omega_clean)
                    z_clean = inr.get_latent(tb, omega=omega_clean).reshape(-1, inr.global_hidden)

                # Compute latent for augmented windows (with their own omega)
                x_aug = augmentation(xb)
                z_llm_aug = llm_encoder(x_aug)
                omega_aug = weight_pred(z_llm_aug)
                omega_aug = prompt(omega_aug)
                z_aug = inr.get_latent(tb, omega=omega_aug).reshape(-1, inr.global_hidden)

                # Unlabeled validation windows from current stream
                val_idx = torch.randperm(M)[:val_batch_size]
                xv = x_val[val_idx].to(self.device)
                tv = t_val[val_idx].to(self.device)
                with torch.no_grad():
                    z_llm_val = llm_encoder(xv)
                    omega_val = weight_pred(z_llm_val)
                    omega_val = prompt(omega_val)
                    z_val = inr.get_latent(tv, omega=omega_val).reshape(-1, inr.global_hidden)

                z_mix = torch.cat([z_clean.detach(), z_aug], dim=0)

                # Paper objective: minimise W(Z_trn ∪ Z_aug, Z_val)
                swd = sliced_wasserstein(z_mix, z_val.detach())
                outer_loss = swd
                outer_loss.backward()
                opt_a.step()

    def snapshot_theta(
        self, inr: nn.Module, weight_pred: nn.Module
    ) -> dict:
        """Return a CPU state_dict snapshot of θ (for warm-start storage)."""
        return {
            "inr": deepcopy({k: v.cpu() for k, v in inr.state_dict().items()}),
            "weight_pred": deepcopy(
                {k: v.cpu() for k, v in weight_pred.state_dict().items()}
            ),
        }
