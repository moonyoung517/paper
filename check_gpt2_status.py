"""
GPT-2 모델 상태 확인 및 대체 전략

설치된 모델 확인 → 없으면 현재 환경에서 최대한 활용
"""

import os
import sys
from pathlib import Path
from typing import Optional

def check_transformers_cache():
    """Transformers 캐시 확인"""
    print("=" * 80)
    print("[1] Transformers 캐시 확인")
    print("=" * 80)
    
    cache_home = os.environ.get("TRANSFORMERS_CACHE", "")
    if not cache_home:
        from transformers import utils
        cache_home = utils._TRANSFORMERS_CACHE
    
    print(f"캐시 위치: {cache_home}")
    print()
    
    cache_path = Path(cache_home)
    if cache_path.exists():
        models = list(cache_path.glob("models--*/"))
        print(f"캐시된 모델 수: {len(models)}")
        for model in sorted(models)[:10]:
            print(f"  - {model.name}")
        if len(models) > 10:
            print(f"  ... and {len(models) - 10} more")
    else:
        print("캐시 디렉토리 없음 (모델 미다운로드)")
    
    print()
    return cache_path if cache_path.exists() else None


def check_huggingface_hub():
    """HuggingFace Hub cache 확인"""
    print("=" * 80)
    print("[2] HuggingFace Hub 캐시 확인")
    print("=" * 80)
    
    hub_home = os.environ.get("HF_HOME", "")
    if not hub_home:
        from huggingface_hub import _is_offline_mode
        try:
            from pathlib import Path
            hub_home = str(Path.home() / ".cache" / "huggingface" / "hub")
        except:
            hub_home = ""
    
    print(f"Hub 위치: {hub_home}")
    print()
    
    if hub_home:
        hub_path = Path(hub_home)
        if hub_path.exists():
            models = list(hub_path.glob("models--*/"))
            print(f"캐시된 모델 수: {len(models)}")
            for model in sorted(models)[:10]:
                print(f"  - {model.name}")
            if len(models) > 10:
                print(f"  ... and {len(models) - 10} more")
        else:
            print("캐시 디렉토리 없음")
    
    print()
    return Path(hub_home) if hub_home and Path(hub_home).exists() else None


def check_env_variable():
    """환경 변수 확인"""
    print("=" * 80)
    print("[3] 환경 변수 확인")
    print("=" * 80)
    
    hf_path = os.environ.get("HF_MODEL_PATH", "")
    if hf_path:
        path = Path(hf_path)
        if path.exists():
            print(f"✓ HF_MODEL_PATH 설정됨: {hf_path}")
            config = path / "config.json"
            if config.exists():
                print(f"  ✓ config.json 존재 (모델 준비됨)")
                return True
            else:
                print(f"  ✗ config.json 없음")
        else:
            print(f"✗ HF_MODEL_PATH 경로 없음: {hf_path}")
    else:
        print("HF_MODEL_PATH 환경 변수 설정 안됨")
    
    print()
    return False


def check_local_folders():
    """로컬 폴더 확인"""
    print("=" * 80)
    print("[4] 로컬 폴더 확인")
    print("=" * 80)
    
    candidates = [
        Path("./gpt2_model"),
        Path("../gpt2_model"),
        Path("D:/gpt2_model"),
        Path("d:/models/gpt2"),
        Path.home() / ".cache" / "huggingface" / "hub" / "models--openai-community--gpt2",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            config = candidate / "config.json"
            has_model = (candidate / "pytorch_model.bin").exists()
            status = "✓ (완전함)" if has_model else "⚠ (불완전)"
            print(f"{status} {candidate.resolve()}")
            if config.exists():
                return str(candidate.resolve())
    
    print("로컬에서 모델 찾지 못함")
    print()
    return None


def suggest_strategy(model_path: Optional[str] = None):
    """전략 제안"""
    print("=" * 80)
    print("[전략]")
    print("=" * 80)
    print()
    
    if model_path:
        print("✓ Pretrained GPT-2 사용 가능!")
        print()
        print("다음 명령으로 테스트 실행:")
        print(f"  $env:HF_MODEL_PATH = '{model_path}'")
        print(f"  python run_paper_pipeline.py --model_type gpt2")
        return True
    else:
        print("⚠️  현재 pretrained GPT-2 모델을 다운로드할 수 없습니다.")
        print()
        print("원인: 회사 네트워크 보안 정책")
        print("  - HuggingFace 서버 차단")
        print("  - GitHub 다운로드 차단 (403 Forbidden)")
        print()
        print("해결책 (우선순위):")
        print()
        print("1. 외부 인터넷 PC에서 다운로드:")
        print("   python -c \"from transformers import AutoModel, AutoTokenizer;")
        print("   m = AutoModel.from_pretrained('gpt2');")
        print("   t = AutoTokenizer.from_pretrained('gpt2');")
        print("   m.save_pretrained('./gpt2_local');")
        print("   t.save_pretrained('./gpt2_local')\"")
        print()
        print("2. USB로 ./gpt2_local 폴더 복사")
        print()
        print("3. 작업 PC에서:")
        print("   $env:HF_MODEL_PATH = '작업폴더/gpt2_local'")
        print("   python run_paper_pipeline.py")
        print()
        print("4. 또는 현재 코드(non-pretrained)로 진행:")
        print("   python run_paper_pipeline.py  # Transformer fallback 사용")
        print()
        return False


def main():
    print()
    print("🔍 GPT-2 모델 상태 진단")
    print()
    
    has_env = check_env_variable()
    has_local = check_local_folders()
    
    if has_env:
        model_path = os.environ.get("HF_MODEL_PATH")
        suggest_strategy(model_path)
    elif has_local:
        suggest_strategy(has_local)
    else:
        check_transformers_cache()
        check_huggingface_hub()
        suggest_strategy(None)


if __name__ == "__main__":
    main()
