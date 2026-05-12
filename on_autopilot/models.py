"""
models.py — Neural network components for On-autoPilot.

Components
----------
FrozenLLMEncoder  : Frozen GPT-2 encoder; amplifies structural breaks.
SIRENLayer        : Single SIREN layer (sinusoidal activation).
GroupBasedINR     : Group-wise SIREN INR reconstructor for MTS.
WeightPredictor   : 1D-CNN that predicts time-variant INR weights ω.
"""

import math
import ssl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# SSL 인증서 검증 무시 (보안이 강한 회사 환경用)
ssl._create_default_https_context = ssl._create_unverified_context


# ---------------------------------------------------------------------------
# Frozen LLM Encoder
# ---------------------------------------------------------------------------

class FrozenLLMEncoder(nn.Module):
    """
    Frozen pre-trained GPT-2-family encoder.

    Maps x ∈ R^{W×d}  →  z_LLM ∈ R^{W×llm_hidden}.
    Features are extracted from `extract_layer` (paper: 5th layer).
    Cross-modal amplification is achieved via the LLM's attention mechanism
    over the channel-projected time-series tokens.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        llm_hidden: int = 768,
        n_layers: int = 6,
        extract_layer: int = 5,
    ):
        super().__init__()
        self.extract_layer = extract_layer
        self.llm_hidden = llm_hidden

        # Try to load a pretrained GPT-2-family backbone; fall back to a plain transformer.
        try:
            import os
            from transformers import AutoModel  # type: ignore

            # Allow override via environment variable
            # Usage: set HF_MODEL_PATH=d:\models\gpt2_local  (PowerShell)
            #        export HF_MODEL_PATH=/path/to/gpt2_local  (Linux)
            _env_path = os.environ.get("HF_MODEL_PATH", "")
            if _env_path:
                print(f"[FrozenLLMEncoder] Loading from HF_MODEL_PATH: {_env_path}")
                model_name = _env_path
                self.backbone = AutoModel.from_pretrained(model_name, local_files_only=True)
            else:
                model_name = "distilgpt2" if n_layers <= 6 else "gpt2"
                print(f"[FrozenLLMEncoder] Attempting to download: {model_name}")
                try:
                    # Try with internet (if available)
                    self.backbone = AutoModel.from_pretrained(model_name)
                except Exception as e:
                    # Fall back to transformer if download fails
                    print(f"[FrozenLLMEncoder] Download failed ({str(e)[:60]}...)")
                    print("[FrozenLLMEncoder] Using Transformer core (non-pretrained)")
                    raise
            backbone_hidden = int(self.backbone.config.hidden_size)
            self._gpt2 = True
            print(f"[FrozenLLMEncoder] ✓ Pretrained model loaded ({model_name})")
        except ImportError:
            print("[FrozenLLMEncoder] transformers not found → using Transformer core")
            backbone_hidden = llm_hidden
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=backbone_hidden,
                nhead=max(1, backbone_hidden // 64),
                dim_feedforward=backbone_hidden * 4,
                batch_first=True,
                dropout=0.0,
            )
            self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self._gpt2 = False
        except Exception as e:
            print(f"[FrozenLLMEncoder] ⚠️  {str(e)[:100]}")
            print("[FrozenLLMEncoder] Falling back to Transformer core (non-pretrained)")
            print()
            print("Solution: Download GPT-2 locally and set HF_MODEL_PATH environment variable:")
            print("  python download_gpt2.py --method transformers --save_dir ./gpt2_model")
            print("  set HF_MODEL_PATH=./gpt2_model")
            print("  python train.py")
            print()
            
            backbone_hidden = llm_hidden
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=backbone_hidden,
                nhead=max(1, backbone_hidden // 64),
                dim_feedforward=backbone_hidden * 4,
                batch_first=True,
                dropout=0.0,
            )
            self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self._gpt2 = False

        # Input projection: d → backbone hidden
        self.input_proj = nn.Linear(input_dim, backbone_hidden)

        # Output projection keeps the public interface z_LLM ∈ R^{W×llm_hidden}
        if backbone_hidden != llm_hidden:
            self.output_proj = nn.Linear(backbone_hidden, llm_hidden)
        else:
            self.output_proj = nn.Identity()

        # Freeze all backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, W, d)

        Returns
        -------
        z_LLM : (B, W, llm_hidden)
        """
        h = self.input_proj(x)  # (B, W, llm_hidden)

        if self._gpt2:
            out = self.backbone(inputs_embeds=h, output_hidden_states=True)
            z_backbone = out.hidden_states[self.extract_layer]  # (B, W, backbone_hidden)
        else:
            z_backbone = self.backbone(h)  # (B, W, backbone_hidden)

        return self.output_proj(z_backbone)  # (B, W, llm_hidden)


# ---------------------------------------------------------------------------
# SIREN building block
# ---------------------------------------------------------------------------

class SIRENLayer(nn.Module):
    """
    Fully-connected layer with sinusoidal activation.
    Weight initialisation follows Sitzmann et al. (2020).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        omega_0: float = 30.0,
        is_first: bool = False,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_dim, out_dim)
        self._init_weights(is_first, in_dim)

    def _init_weights(self, is_first: bool, in_dim: int) -> None:
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_dim
            else:
                bound = math.sqrt(6.0 / in_dim) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


# ---------------------------------------------------------------------------
# Group-based INR Reconstructor
# ---------------------------------------------------------------------------

class GroupBasedINR(nn.Module):
    """
    Scalable group-based SIREN architecture for MTS reconstruction.

    Architecture
    ------------
    • 3 global SIREN layers (dim=global_hidden) — capture inter-channel
      dependencies from normalised time coordinate t ∈ [0,1].
    • k group-specific 2-layer stacks (dim=group_hidden → group_dim) — model
      intra-channel details for each of the k variable groups.

    The spectral bias of SIREN inherently rejects high-frequency anomalous
    fluctuations, widening the reconstruction error gap for detection.
    """

    def __init__(
        self,
        input_dim: int,
        num_groups: int = 4,
        global_hidden: int = 64,
        group_hidden: int = 32,
        omega_0: float = 30.0,
    ):
        super().__init__()
        if input_dim % num_groups != 0:
            raise ValueError("`input_dim` must be divisible by `num_groups`")

        self.num_groups = num_groups
        self.group_dim = input_dim // num_groups
        self.global_hidden = global_hidden

        # --- Global layers: time coordinate t → shared representation ---
        self.global_net = nn.Sequential(
            SIRENLayer(1, global_hidden, omega_0, is_first=True),
            SIRENLayer(global_hidden, global_hidden, omega_0),
            SIRENLayer(global_hidden, global_hidden, omega_0),
        )

        # --- Group layers: shared repr → per-group reconstruction ---
        self.group_nets = nn.ModuleList(
            [
                nn.Sequential(
                    SIRENLayer(global_hidden, group_hidden, omega_0),
                    SIRENLayer(group_hidden, self.group_dim, omega_0),
                )
                for _ in range(num_groups)
            ]
        )

    def forward(self, t: torch.Tensor, omega: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        t : (B, W, 1)  normalised time coordinates in [0, 1]

        Returns
        -------
        x_hat : (B, W, d)  reconstructed MTS
        """
        global_feat = self.global_net(t)  # (B, W, global_hidden)
        if omega is not None:
            # Inject time-variant INR conditioning predicted from LLM features.
            global_feat = global_feat + omega.unsqueeze(1)
        groups = [g(global_feat) for g in self.group_nets]  # each (B, W, group_dim)
        return torch.cat(groups, dim=-1)  # (B, W, d)

    def get_latent(self, t: torch.Tensor, omega: torch.Tensor | None = None) -> torch.Tensor:
        """
        Return INR latent representation Z (shaped by spectral bias).

        Parameters
        ----------
        t : (B, W, 1)

        Returns
        -------
        Z : (B, W, global_hidden)
        """
        z = self.global_net(t)  # (B, W, global_hidden)
        if omega is not None:
            z = z + omega.unsqueeze(1)
        return z


# ---------------------------------------------------------------------------
# Weight Predictor (1D-CNN)
# ---------------------------------------------------------------------------

class WeightPredictor(nn.Module):
    """
    Lightweight 1D-CNN that maps LLM features z_LLM → time-variant INR
    weights ω, enabling instance-adaptive manifold mapping.
    """

    def __init__(self, llm_dim: int, inr_hidden: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(llm_dim, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, inr_hidden)

    def forward(self, z_llm: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_llm : (B, W, llm_dim)

        Returns
        -------
        omega : (B, inr_hidden)
        """
        h = self.cnn(z_llm.transpose(1, 2)).squeeze(-1)  # (B, 64)
        return self.fc(h)  # (B, inr_hidden)


# ---------------------------------------------------------------------------
# Task-specific Prompt Module
# ---------------------------------------------------------------------------

class PromptModule(nn.Module):
    """
    Learnable prompt p_t stored in the CPM memory bank.
    Acts as a task-specific bias injected into the INR latent space.
    """

    def __init__(self, prompt_dim: int):
        super().__init__()
        self.prompt = nn.Parameter(torch.zeros(prompt_dim))
        nn.init.normal_(self.prompt, std=0.01)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., prompt_dim) → z + prompt"""
        return z + self.prompt
