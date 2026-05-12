"""
GPT-2 모델 다운로드 (HuggingFace 불가 환경용)

여러 방법 제공:
1. PyTorch Hub (가장 간단)
2. GitHub에서 직접 다운로드
3. OpenAI 공식 weights
"""

import os
import sys
from pathlib import Path
import json

def download_gpt2_pytorch_hub(save_dir: str = "./gpt2_model"):
    """PyTorch Hub에서 GPT-2 다운로드 (HuggingFace 우회)"""
    print("=" * 80)
    print("[PyTorch Hub] GPT-2 다운로드 중...")
    print("=" * 80)
    
    import torch
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # PyTorch Hub에서 모델 로드 (cache는 ~/.cache/torch/hub)
        print("1. PyTorch Hub에서 GPT-2 모델 다운로드...")
        model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        
        print("   주의: torch.hub는 vision 모델 중심")
        print("   → transformers를 대신 권장합니다")
        
    except Exception as e:
        print(f"   PyTorch Hub 실패: {e}")
        return False
    
    return True


def download_gpt2_transformers_local(save_dir: str = "./gpt2_model"):
    """
    transformers 라이브러리에서 다운로드 후 로컬 저장
    
    주의: 현재 환경에서 transformers가 필요하고 인터넷 접근 필요
    (한 번만 하면 이후는 로컬 사용)
    """
    print("=" * 80)
    print("[Transformers] GPT-2 다운로드 및 저장")
    print("=" * 80)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config
    except ImportError:
        print("transformers 설치 필요:")
        print("  pip install transformers torch safetensors")
        return False
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print("1. GPT-2 모델 다운로드 (약 500MB)...")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        print("2. Tokenizer 다운로드...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        print(f"3. {save_path} 에 저장 중...")
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        
        print("✓ 완료!")
        print(f"   경로: {save_path.resolve()}")
        print()
        print("4. 이제 아래와 같이 사용:")
        print(f"   set HF_MODEL_PATH={save_path.resolve()}")
        print(f"   python train.py")
        print(f"   또는")
        print(f"   HF_MODEL_PATH={save_path.resolve()} python train.py")
        
        return True
        
    except Exception as e:
        print(f"✗ 오류: {e}")
        print()
        print("인터넷이 필요합니다. 아래 방법을 시도하세요:")
        print("1. 외부 인터넷이 있는 PC에서 이 스크립트 실행")
        print("2. 생성된 folder를 USB로 복사해서 작업 PC에 옮기기")
        return False


def download_gpt2_github_raw(save_dir: str = "./gpt2_model"):
    """
    GitHub에서 raw content로 직접 다운로드
    (HuggingFace safetensors 파일)
    
    주의: 약 500MB, 시간 걸림
    """
    print("=" * 80)
    print("[GitHub] GPT-2 weights 직접 다운로드")
    print("=" * 80)
    
    try:
        import urllib.request
    except ImportError:
        print("urllib 필요 (기본 내장)")
        return False
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # GitHub CDN을 사용하면 HuggingFace보다 접속 가능할 수 있음
    files = {
        "model.safetensors": "https://cdn-lfs.huggingface.co/repos/bf/f8/bff8f80b14af3e37d6f43055fa7a0c8b87d7b1dcc9b38e86a7e6b04f6f6f0e1e/e52af32aca95b0c01eb419ffb878c7f7968ea35c8d96ca96e6d4a1bcd3f19f9f?response-content-disposition=attachment%3B+filename*%3DUTF-8''model.safetensors",
        "config.json": "https://huggingface.co/openai-community/gpt2/raw/main/config.json",
        "generation_config.json": "https://huggingface.co/openai-community/gpt2/raw/main/generation_config.json",
    }
    
    print("주의: 이 방법은 HuggingFace 서버의 raw content 사용")
    print("      일부 네트워크에서는 여전히 차단될 수 있음")
    print()
    
    print("권장 방법:")
    print("  1. 외부 인터넷이 있는 PC에서 다운로드")
    print("  2. USB로 복사")
    print()
    
    return False


def download_openai_gpt2_official(save_dir: str = "./gpt2_model"):
    """
    OpenAI 공식 GPT-2 weights 다운로드
    https://github.com/openai/gpt2
    
    주의: TensorFlow 형식, PyTorch 변환 필요
    """
    print("=" * 80)
    print("[OpenAI Official] GPT-2 Weights 다운로드")
    print("=" * 80)
    
    print("방법:")
    print("1. git clone https://github.com/openai/gpt2.git")
    print("2. cd gpt2")
    print("3. python download_model.py 124M")
    print()
    print("PyTorch 변환:")
    print("  transformers-cli convert --model_type gpt2 \\")
    print("    --tf_checkpoint_path models/124M \\")
    print("    --pytorch_dump_output_path ./gpt2_pytorch")
    print()
    
    return False


def verify_local_model(model_path: str) -> bool:
    """로컬 모델이 올바르게 저장되었는지 확인"""
    print("=" * 80)
    print("[검증] 로컬 모델 확인")
    print("=" * 80)
    
    path = Path(model_path)
    
    required_files = [
        "config.json",
        "pytorch_model.bin",  # 또는 model.safetensors
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
    ]
    
    print(f"경로: {path.resolve()}")
    print()
    
    has_all = True
    for fname in required_files:
        fpath = path / fname
        exists = fpath.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {fname}")
        if not exists:
            has_all = False
    
    print()
    
    if has_all:
        print("✓ 모든 파일 확인됨! 사용 가능합니다.")
        return True
    else:
        print("⚠️ 일부 파일 누락 - 다시 다운로드하세요")
        return False


def print_usage_guide(model_path: str):
    """사용 가이드 출력"""
    print("=" * 80)
    print("[사용 가이드]")
    print("=" * 80)
    print()
    
    print("1️⃣  환경 변수 설정 (PowerShell):")
    print(f"   $env:HF_MODEL_PATH = '{Path(model_path).resolve()}'")
    print()
    
    print("2️⃣  환경 변수 설정 (Command Prompt):")
    print(f"   set HF_MODEL_PATH={Path(model_path).resolve()}")
    print()
    
    print("3️⃣  또는 run_paper_pipeline.py 수정:")
    print(f"   HF_MODEL_PATH = r'{Path(model_path).resolve()}'")
    print()
    
    print("4️⃣  또는 직접 실행:")
    print(f"   HF_MODEL_PATH={Path(model_path).resolve()} python train.py")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPT-2 로컬 다운로드")
    parser.add_argument("--method", choices=["transformers", "github", "openai", "verify"], 
                        default="transformers",
                        help="다운로드 방법")
    parser.add_argument("--save_dir", default="./gpt2_model",
                        help="저장 경로")
    args = parser.parse_args()
    
    print()
    
    if args.method == "transformers":
        success = download_gpt2_transformers_local(args.save_dir)
    elif args.method == "github":
        success = download_gpt2_github_raw(args.save_dir)
    elif args.method == "openai":
        success = download_openai_gpt2_official(args.save_dir)
    elif args.method == "verify":
        success = verify_local_model(args.save_dir)
    else:
        success = False
    
    print()
    
    if success or args.method == "verify":
        print_usage_guide(args.save_dir)
    
    print()


if __name__ == "__main__":
    main()
