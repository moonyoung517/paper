"""
augmentation.py — Differentiable Anomaly Synthesis Profile.

Six anomaly types drawn from industrial MTS literature:
  1. Mean Shift     — sustained level change in a channel
  2. Platform       — consecutive constant segment (sensor lock-up)
  3. Trend          — monotone linear drift
  4. Amplitude Shift — scaled signal variance
  5. Extremum       — isolated spike / drop
  6. Frequency Shift — resampled / stretched subsequence

Each type has a real-valued learnable magnitude parameter `a`
(positive = learnable strength), optimised by the outer loop of the
Self-Tuning Engine via Sliced Wasserstein distance.

Usage::
    aug = DifferentiableAugmentation(seq_len=100, n_channels=38)
    x_aug = aug(x)           # draw random type per sample
    x_aug = aug(x, aug_type=0)  # specific type
"""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_AUG_NAMES = [
    "mean_shift",
    "platform",
    "trend",
    "amplitude_shift",
    "extremum",
    "freq_shift",
]


class DifferentiableAugmentation(nn.Module):
    """
    Differentiable anomaly augmentation with 6 learnable magnitude params.

    Parameters
    ----------
    seq_len    : int  — window length W
    n_channels : int  — number of sensor channels d
    a_init     : float — initial magnitude for all types (default 0.5)
    """

    def __init__(
        self,
        seq_len: int,
        n_channels: int,
        a_init: float = 0.5,
    ):
        super().__init__()
        self.W = seq_len
        self.d = n_channels

        # One learnable magnitude per augmentation type (log parameterisation
        # ensures positivity while allowing unconstrained optimisation)
        self._log_a = nn.ParameterDict(
            {name: nn.Parameter(torch.tensor(a_init).log()) for name in _AUG_NAMES}
        )
        self._a_init = float(a_init)

    def reset_parameters(self, a_init: Optional[float] = None) -> None:
        """Re-initialise augmentation magnitudes for a new regime."""
        target = self._a_init if a_init is None else float(a_init)
        log_target = torch.tensor(target).log()
        with torch.no_grad():
            for p in self._log_a.values():
                p.copy_(log_target.to(device=p.device, dtype=p.dtype))

    # ------------------------------------------------------------------
    # Magnitude accessors (always positive)
    # ------------------------------------------------------------------

    def a(self, name: str) -> torch.Tensor:
        return self._log_a[name].exp()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        aug_type: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : (B, W, d)
        aug_type : int in [0, 5] or None (random per sample)

        Returns
        -------
        x_aug : (B, W, d)
        """
        B = x.shape[0]
        out = x.clone()
        dispatch = [
            self._mean_shift,
            self._platform,
            self._trend,
            self._amplitude_shift,
            self._extremum,
            self._freq_shift,
        ]

        for b in range(B):
            idx = aug_type if aug_type is not None else random.randrange(len(dispatch))
            out[b] = dispatch[idx](x[b])  # (W, d)

        return out

    # ------------------------------------------------------------------
    # Individual augmentation types
    # ------------------------------------------------------------------

    def _mean_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Shift: x + a * std(x) on a random channel subset."""
        W, d = x.shape
        sigma = x.std(0, keepdim=True) + 1e-6  # (1, d)
        mask = (torch.rand(d, device=x.device) < 0.5).float()  # channel mask
        return x + self.a("mean_shift") * sigma * mask

    def _platform(self, x: torch.Tensor) -> torch.Tensor:
        """Freeze a random window [t_start, t_start+L] on a random channel."""
        W, d = x.shape
        L = max(1, int(self.a("platform").item() * W * 0.3))
        t_start = random.randint(0, max(0, W - L - 1))
        c = random.randrange(d)
        out = x.clone()
        out[t_start:t_start + L, c] = x[t_start, c]
        return out

    def _trend(self, x: torch.Tensor) -> torch.Tensor:
        """Add linear drift a * t / W on a random channel."""
        W, d = x.shape
        t = torch.linspace(0, 1, W, device=x.device).unsqueeze(-1).expand(W, d)
        sigma = x.std(0, keepdim=True) + 1e-6
        return x + self.a("trend") * sigma * t

    def _amplitude_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Scale a random channel by (1 + a)."""
        W, d = x.shape
        c = random.randrange(d)
        scale = 1.0 + self.a("amplitude_shift")
        out = x.clone()
        out[:, c] = x[:, c] * scale
        return out

    def _extremum(self, x: torch.Tensor) -> torch.Tensor:
        """Inject a spike of height a * 4*std at a random time step."""
        W, d = x.shape
        t = random.randint(0, W - 1)
        c = random.randrange(d)
        sigma = x[:, c].std() + 1e-6
        out = x.clone()
        sign = 1 if random.random() < 0.5 else -1
        out[t, c] = x[t, c] + sign * self.a("extremum") * 4 * sigma
        return out

    def _freq_shift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Resample a random channel with stretch factor (1 + 0.5 * a)
        then align back to original length W via linear interpolation.
        """
        W, d = x.shape
        c = random.randrange(d)
        stretch = 1.0 + 0.5 * self.a("freq_shift")
        new_len = max(2, int(W * stretch.item()))
        src = x[:, c].unsqueeze(0).unsqueeze(0)       # (1, 1, W)
        stretched = F.interpolate(src, size=new_len, mode="linear", align_corners=True)
        back = F.interpolate(stretched, size=W, mode="linear", align_corners=True)
        out = x.clone()
        out[:, c] = back.squeeze()
        return out
