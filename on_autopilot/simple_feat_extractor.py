"""
On-autoPilot Framework with optional LLM encoder

LLM 사용 여부를 선택할 수 있도록 수정된 버전

use_llm=True  : 기존대로 LLM encoder 사용 (논문 방식)
use_llm=False : Raw data 또는 간단한 통계 특성 직접 사용 (비교용)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SimpleFeatExtractor(nn.Module):
    """
    LLM 대신 간단한 특성 추출기
    
    Raw data (B, W, d)를 입력받아 (B, W, d)를 출력
    - 그냥 data normalization만 수행
    - learnable 파라미터 없음
    """
    
    def __init__(self, input_dim: int, seq_len: int, output_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        # 선택사항: raw data를 다른 차원으로 projection
        # 지금은 간단하게 normalization만 수행
        self.use_projection = False
        
        if self.use_projection:
            self.proj = nn.Linear(input_dim, output_dim)
        else:
            # input_dim == output_dim으로 맞춰서 사용
            assert input_dim <= output_dim, "input_dim should be <= output_dim for SimpleFeatExtractor"
            if input_dim < output_dim:
                # Padding: zero padding으로 차원 맞추기
                self.pad_dim = output_dim - input_dim
            else:
                self.pad_dim = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, W, d)  raw MTS
        
        Returns
        -------
        z : (B, W, output_dim)  extracted features
        """
        B, W, d = x.shape
        
        # Normalize by channel (per-feature)
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, d)
        std = x.std(dim=1, keepdim=True) + 1e-8  # (B, 1, d)
        x_norm = (x - mean) / std  # (B, W, d)
        
        # Pad or project to output_dim
        if self.pad_dim > 0:
            padding = torch.zeros(B, W, self.pad_dim, device=x.device, dtype=x.dtype)
            z = torch.cat([x_norm, padding], dim=-1)  # (B, W, output_dim)
        else:
            z = x_norm  # (B, W, d)
        
        return z


def forward_with_optional_llm(
    x: torch.Tensor,
    use_llm: bool,
    llm_encoder: Optional[nn.Module],
    simple_feat_extractor: Optional[nn.Module],
    device: torch.device,
) -> torch.Tensor:
    """
    LLM 여부에 따라 특성 추출
    
    Parameters
    ----------
    x : (B, W, d)
    use_llm : bool
    llm_encoder : FrozenLLMEncoder 또는 None
    simple_feat_extractor : SimpleFeatExtractor 또는 None
    device : torch.device
    
    Returns
    -------
    z_feat : (B, W, feat_dim)
    """
    x = x.to(device)
    
    if use_llm:
        assert llm_encoder is not None, "use_llm=True but llm_encoder is None"
        return llm_encoder(x)  # (B, W, llm_hidden)
    else:
        assert simple_feat_extractor is not None, "use_llm=False but simple_feat_extractor is None"
        return simple_feat_extractor(x)  # (B, W, feat_dim)


# 테스트용 코드
if __name__ == "__main__":
    # Test SimpleFeatExtractor
    device = torch.device("cpu")
    extractor = SimpleFeatExtractor(input_dim=38, seq_len=100, output_dim=64)
    
    x = torch.randn(4, 100, 38)
    z = extractor(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {z.shape}")
    print(f"Expected:     (4, 100, 64)")
    assert z.shape == (4, 100, 64), "Shape mismatch!"
    print("✓ SimpleFeatExtractor works!")
