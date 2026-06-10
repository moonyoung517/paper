"""
Compare Freeze / Oracle / Ours on multi-domain streams.

This script is designed for quick reproducible comparisons under the current
codebase constraints:
  - One OnAutoPilot model uses a fixed input dimension.
  - Domains may have different channel counts.

To run mixed-domain streams, each domain is z-score normalized on its own train
split and then right-zero-padded to the maximum channel dimension.

Scenarios:
  - two_domain: stream = smd -> msl -> smd
  - five_domain: stream = smd -> msl -> hai -> psm -> skab -> msl -> psm -> hai -> skab -> smd

If required datasets are missing, the script exits with a clear error.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# Ensure local package import works when running from scripts/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from on_autopilot import OnAutoPilot
from on_autopilot.models import PromptModule


@dataclass
class DomainData:
    name: str
    train: np.ndarray
    test: np.ndarray
    labels: np.ndarray


def load_domain(root: Path, name: str) -> DomainData:
    base = root / name
    train_path = base / "train.npy"
    test_path = base / "test.npy"
    labels_path = base / "labels.npy"
    if not (train_path.exists() and test_path.exists() and labels_path.exists()):
        raise FileNotFoundError(
            f"Missing dataset files for '{name}' under {base}. "
            "Expected train.npy, test.npy, labels.npy"
        )

    train = np.load(train_path).astype(np.float32)
    test = np.load(test_path).astype(np.float32)
    labels = np.load(labels_path).astype(np.int32)

    mean = train.mean(0, keepdims=True)
    std = train.std(0, keepdims=True) + 1e-8
    train = (train - mean) / std
    test = (test - mean) / std

    return DomainData(name=name, train=train, test=test, labels=labels)


def pad_channels(x: np.ndarray, target_dim: int) -> np.ndarray:
    if x.shape[1] == target_dim:
        return x
    out = np.zeros((x.shape[0], target_dim), dtype=np.float32)
    out[:, : x.shape[1]] = x
    return out


def build_stream(
    domain_map: Dict[str, DomainData],
    sequence: List[str],
    max_test_points: int,
    target_dim: int,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, int, int]]]:
    tests: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    segments: List[Tuple[str, int, int]] = []
    cursor = 0

    for name in sequence:
        d = domain_map[name]
        n = min(max_test_points, len(d.test)) if max_test_points > 0 else len(d.test)
        x = pad_channels(d.test[:n], target_dim)
        y = d.labels[:n]
        tests.append(x)
        labels.append(y)
        segments.append((name, cursor, cursor + n))
        cursor += n

    return np.concatenate(tests, axis=0), np.concatenate(labels, axis=0), segments


def segment_aurocs(scores: np.ndarray, labels: np.ndarray, segments: List[Tuple[str, int, int]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for name, s, e in segments:
        ys = labels[s:e]
        ss = scores[s:e]
        if ys.min() == ys.max():
            continue
        out.setdefault(name, []).append(float(roc_auc_score(ys, ss)))
    return out


def mean_dict(values: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: float(np.mean(v)) for k, v in values.items() if len(v) > 0}


def visit_aurocs(scores: np.ndarray, labels: np.ndarray, segments: List[Tuple[str, int, int]]) -> List[dict]:
    visits: List[dict] = []
    seen: Dict[str, int] = {}
    for name, s, e in segments:
        seen[name] = seen.get(name, 0) + 1
        ys = labels[s:e]
        ss = scores[s:e]
        auroc = None
        if ys.min() != ys.max():
            auroc = float(roc_auc_score(ys, ss))
        visits.append(
            {
                "domain": name,
                "visit": int(seen[name]),
                "start": int(s),
                "end": int(e),
                "auroc": auroc,
            }
        )
    return visits


def robustness_summary(visits: List[dict], collapse_threshold: float) -> dict:
    vals = [v["auroc"] for v in visits if v["auroc"] is not None]
    if len(vals) == 0:
        return {
            "worst_case_auroc": None,
            "best_case_auroc": None,
            "range": None,
            "collapse_threshold": float(collapse_threshold),
            "collapse_count": None,
            "below_random_count": None,
        }

    collapse_count = sum(1 for x in vals if x < collapse_threshold)
    below_random_count = sum(1 for x in vals if x < 0.5)
    return {
        "worst_case_auroc": float(min(vals)),
        "best_case_auroc": float(max(vals)),
        "range": float(max(vals) - min(vals)),
        "collapse_threshold": float(collapse_threshold),
        "collapse_count": int(collapse_count),
        "below_random_count": int(below_random_count),
    }


def rolling_auroc_stats(
    scores: np.ndarray,
    labels: np.ndarray,
    window: int,
    step: int,
    collapse_threshold: float,
) -> dict:
    vals: List[float] = []
    if window <= 0 or step <= 0 or len(scores) < window:
        return {
            "count": 0,
            "worst_case_auroc": None,
            "best_case_auroc": None,
            "mean_auroc": None,
            "collapse_count": None,
            "below_random_count": None,
        }

    for s in range(0, len(scores) - window + 1, step):
        e = s + window
        y = labels[s:e]
        if y.min() == y.max():
            continue
        vals.append(float(roc_auc_score(y, scores[s:e])))

    if len(vals) == 0:
        return {
            "count": 0,
            "worst_case_auroc": None,
            "best_case_auroc": None,
            "mean_auroc": None,
            "collapse_count": None,
            "below_random_count": None,
        }

    return {
        "count": int(len(vals)),
        "worst_case_auroc": float(min(vals)),
        "best_case_auroc": float(max(vals)),
        "mean_auroc": float(np.mean(vals)),
        "collapse_count": int(sum(1 for x in vals if x < collapse_threshold)),
        "below_random_count": int(sum(1 for x in vals if x < 0.5)),
    }


def warm_register_domain(
    model: OnAutoPilot,
    train_x: np.ndarray,
    seq_len: int,
    stride: int,
    max_train_windows: int,
    device: torch.device,
    theta_init: dict | None,
) -> None:
    starts = list(range(0, max(1, len(train_x) - seq_len + 1), stride))
    if max_train_windows > 0:
        starts = starts[:max_train_windows]
    windows = np.stack([train_x[s : s + seq_len] for s in starts], axis=0)
    x_train = torch.from_numpy(windows).to(device)
    t_train = model._time_coords(x_train.shape[0])

    model.self_tuning_engine.run(
        inr=model.inr,
        weight_pred=model.weight_predictor,
        prompt=model.prompt,
        llm_encoder=model.llm_encoder,
        augmentation=model.augmentation,
        x_train=x_train,
        t_coords=t_train,
        theta_init=theta_init,
    )

    with torch.no_grad():
        z_llm = model.llm_encoder(x_train)
        omega = model.weight_predictor(z_llm)
        z_seq = model.inr.get_latent(t_train, omega=omega)
        z_flat = z_seq.reshape(-1, z_seq.shape[-1])

    k_t = model._key_from_raw_windows(x_train)
    kn_t = model.cpm_bank.build_kn(z_flat)
    theta_snap = model.self_tuning_engine.snapshot_theta(model.inr, model.weight_predictor)
    p_t = PromptModule(prompt_dim=model.prompt.prompt.shape[0]).to(device)
    p_t.load_state_dict(model.prompt.state_dict(), strict=False)
    model.cpm_bank.register(k_t, p_t, kn_t, theta_snap)


def score_stream_static(model: OnAutoPilot, x: np.ndarray, seq_len: int, stride: int, device: torch.device) -> np.ndarray:
    if model.cpm_bank.is_empty():
        raise RuntimeError("Static baseline requires at least one registered bank entry")

    entry = model.cpm_bank._bank[-1]
    kn = entry["kn"].to(device)

    starts = list(range(0, len(x) - seq_len + 1, stride))
    ts_scores = np.zeros(len(x), dtype=np.float32)
    x_t = torch.from_numpy(x.astype(np.float32)).to(device)

    with torch.no_grad():
        for s in starts:
            w = x_t[s : s + seq_len].unsqueeze(0)
            score, _, _ = model.forward(w, kn=kn)
            ts_scores[s : s + seq_len] = np.maximum(ts_scores[s : s + seq_len], float(score.item()))

    return ts_scores


def score_stream_ours(model: OnAutoPilot, x: np.ndarray, seq_len: int, stride: int, adapt_buffer: int, device: torch.device) -> np.ndarray:
    stream = torch.from_numpy(x.astype(np.float32)).to(device)
    results = model.detect_and_adapt(stream, window_size=seq_len, stride=stride, adapt_buffer=adapt_buffer)
    ts_scores = np.zeros(len(x), dtype=np.float32)
    for r in results:
        s = r["t"]
        e = min(s + seq_len, len(x))
        ts_scores[s:e] = np.maximum(ts_scores[s:e], float(r["score"]))
    return ts_scores


def score_stream_sda(
    model: OnAutoPilot,
    x: np.ndarray,
    seq_len: int,
    stride: int,
    adapt_buffer: int,
    device: torch.device,
) -> np.ndarray:
    """
    CID-unsafe SDA-style adaptation baseline.

    Uses the same boundary-free detection path as Ours but disables clean
    filtering by forcing Stage-1 subset to include the full buffered stream.
    """
    stream = torch.from_numpy(x.astype(np.float32)).to(device)
    results = model.detect_and_adapt(
        stream,
        window_size=seq_len,
        stride=stride,
        adapt_buffer=adapt_buffer,
        clean_ratio=1.0,
        min_clean_windows=1,
    )
    ts_scores = np.zeros(len(x), dtype=np.float32)
    for r in results:
        s = r["t"]
        e = min(s + seq_len, len(x))
        ts_scores[s:e] = np.maximum(ts_scores[s:e], float(r["score"]))
    return ts_scores


def score_stream_oracle(
    build_oracle_model: Callable[[], OnAutoPilot],
    init_state: dict,
    domain_map: Dict[str, DomainData],
    sequence: List[str],
    max_test_points: int,
    max_train_windows: int,
    seq_len: int,
    stride: int,
    target_dim: int,
    device: torch.device,
    reset_each_domain: bool,
    warm_start: bool,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, int, int]]]:
    model = build_oracle_model()
    model.load_state_dict(copy.deepcopy(init_state), strict=False)

    seg_scores: List[np.ndarray] = []
    seg_labels: List[np.ndarray] = []
    segments: List[Tuple[str, int, int]] = []
    cursor = 0

    for name in sequence:
        if reset_each_domain:
            model = build_oracle_model()
            model.load_state_dict(copy.deepcopy(init_state), strict=False)

        d = domain_map[name]
        train_x = pad_channels(d.train, target_dim)
        test_x = pad_channels(d.test, target_dim)
        n = min(max_test_points, len(test_x)) if max_test_points > 0 else len(test_x)
        test_x = test_x[:n]
        labels = d.labels[:n]

        theta_init = model.cpm_bank.retrieve_latest_theta() if warm_start else None
        warm_register_domain(
            model=model,
            train_x=train_x,
            seq_len=seq_len,
            stride=stride,
            max_train_windows=max_train_windows,
            device=device,
            theta_init=theta_init,
        )

        s = score_stream_static(model, test_x, seq_len=seq_len, stride=stride, device=device)
        seg_scores.append(s)
        seg_labels.append(labels)
        segments.append((name, cursor, cursor + n))
        cursor += n

    return np.concatenate(seg_scores), np.concatenate(seg_labels), segments


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare Freeze / Oracle / Ours on multi-domain streams")
    p.add_argument("--data_path", default="./data")
    p.add_argument("--scenario", choices=["two_domain", "five_domain"], default="two_domain")
    p.add_argument("--seq_len", type=int, default=100)
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--adapt_buffer", type=int, default=150)
    p.add_argument("--max_train_windows", type=int, default=120)
    p.add_argument("--max_test_points", type=int, default=12000)
    p.add_argument("--oracle_max_train_windows", type=int, default=0,
                   help="Oracle train windows per domain (0 uses max_train_windows)")
    p.add_argument("--llm_hidden", type=int, default=64)
    p.add_argument("--num_groups", type=int, default=4)
    p.add_argument("--tau_drift", type=float, default=0.65)
    p.add_argument("--inner_steps", type=int, default=2)
    p.add_argument("--outer_steps", type=int, default=12)
    p.add_argument("--oracle_inner_steps", type=int, default=4)
    p.add_argument("--oracle_outer_steps", type=int, default=20)
    p.add_argument("--inner_lr", type=float, default=1e-3)
    p.add_argument("--outer_lr", type=float, default=1e-3)
    p.add_argument("--coreset_size", type=int, default=64)
    p.add_argument("--fps_samples", type=int, default=32)
    p.add_argument("--use_llm", action="store_true",
                   help="Use frozen LLM encoder. Default is disabled (non-LLM feature extractor).")
    p.add_argument("--oracle_no_reset", action="store_true",
                   help="Reuse one oracle model across domains (default resets per domain).")
    p.add_argument("--oracle_warm_start", action="store_true",
                   help="Warm-start oracle from previous theta.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--collapse_threshold", type=float, default=0.55,
                   help="AUROC threshold used to count collapse events in robustness report")
    p.add_argument("--rolling_window", type=int, default=1000,
                   help="Window size for timeline worst-case AUROC analysis")
    p.add_argument("--rolling_step", type=int, default=250,
                   help="Step size for timeline worst-case AUROC analysis")
    return p.parse_args()


def make_sequence(scenario: str) -> List[str]:
    if scenario == "two_domain":
        return ["smd", "msl", "smd"]
    return ["smd", "msl", "hai", "psm", "skab", "msl", "psm", "hai", "skab", "smd"]


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    sequence = make_sequence(args.scenario)
    domains = sorted(set(sequence))

    data_root = Path(args.data_path)
    domain_map: Dict[str, DomainData] = {}
    for name in domains:
        domain_map[name] = load_domain(data_root, name)

    target_dim = max(d.train.shape[1] for d in domain_map.values())
    num_groups = args.num_groups
    while target_dim % num_groups != 0 and num_groups > 1:
        num_groups -= 1

    def build_model(inner_steps: int, outer_steps: int) -> OnAutoPilot:
        return OnAutoPilot(
            input_dim=target_dim,
            seq_len=args.seq_len,
            llm_hidden=args.llm_hidden,
            use_llm=args.use_llm,
            num_groups=num_groups,
            tau_drift=args.tau_drift,
            lambda_score=0.5,
            fps_samples=args.fps_samples,
            coreset_size=args.coreset_size,
            inner_lr=args.inner_lr,
            outer_lr=args.outer_lr,
            inner_steps=inner_steps,
            outer_steps=outer_steps,
            device=device,
        )

    def build_main_model() -> OnAutoPilot:
        return build_model(args.inner_steps, args.outer_steps)

    def build_oracle_model() -> OnAutoPilot:
        return build_model(args.oracle_inner_steps, args.oracle_outer_steps)

    # Build shared stream for Freeze and Ours
    stream_x, stream_y, stream_segments = build_stream(
        domain_map=domain_map,
        sequence=sequence,
        max_test_points=args.max_test_points,
        target_dim=target_dim,
    )

    # Warm-up on the first domain for fair initialization
    init_domain = sequence[0]
    init_train = pad_channels(domain_map[init_domain].train, target_dim)

    model_init = build_main_model()
    warm_register_domain(
        model=model_init,
        train_x=init_train,
        seq_len=args.seq_len,
        stride=args.stride,
        max_train_windows=args.max_train_windows,
        device=device,
        theta_init=None,
    )

    # Freeze baseline
    freeze_model = copy.deepcopy(model_init)
    freeze_scores = score_stream_static(
        model=freeze_model,
        x=stream_x,
        seq_len=args.seq_len,
        stride=args.stride,
        device=device,
    )

    # Ours baseline
    ours_model = copy.deepcopy(model_init)
    ours_scores = score_stream_ours(
        model=ours_model,
        x=stream_x,
        seq_len=args.seq_len,
        stride=args.stride,
        adapt_buffer=args.adapt_buffer,
        device=device,
    )

    # SDA baseline (CID-unsafe)
    sda_model = copy.deepcopy(model_init)
    sda_scores = score_stream_sda(
        model=sda_model,
        x=stream_x,
        seq_len=args.seq_len,
        stride=args.stride,
        adapt_buffer=args.adapt_buffer,
        device=device,
    )

    # Oracle baseline
    oracle_template = build_oracle_model()
    oracle_init_state = copy.deepcopy(oracle_template.state_dict())
    oracle_max_train_windows = args.oracle_max_train_windows if args.oracle_max_train_windows > 0 else args.max_train_windows

    oracle_scores, oracle_labels, oracle_segments = score_stream_oracle(
        build_oracle_model=build_oracle_model,
        init_state=oracle_init_state,
        domain_map=domain_map,
        sequence=sequence,
        max_test_points=args.max_test_points,
        max_train_windows=oracle_max_train_windows,
        seq_len=args.seq_len,
        stride=args.stride,
        target_dim=target_dim,
        device=device,
        reset_each_domain=not args.oracle_no_reset,
        warm_start=args.oracle_warm_start,
    )

    def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
        if y.min() == y.max():
            return float("nan")
        return float(roc_auc_score(y, s))

    freeze_visits = visit_aurocs(freeze_scores, stream_y, stream_segments)
    ours_visits = visit_aurocs(ours_scores, stream_y, stream_segments)
    sda_visits = visit_aurocs(sda_scores, stream_y, stream_segments)
    oracle_visits = visit_aurocs(oracle_scores, oracle_labels, oracle_segments)

    summary = {
        "scenario": args.scenario,
        "use_llm": bool(args.use_llm),
        "sequence": sequence,
        "target_dim": target_dim,
        "num_groups": num_groups,
        "oracle_config": {
            "reset_each_domain": bool(not args.oracle_no_reset),
            "warm_start": bool(args.oracle_warm_start),
            "oracle_inner_steps": int(args.oracle_inner_steps),
            "oracle_outer_steps": int(args.oracle_outer_steps),
            "oracle_max_train_windows": int(oracle_max_train_windows),
        },
        "global_auroc": {
            "freeze": safe_auroc(stream_y, freeze_scores),
            "ours": safe_auroc(stream_y, ours_scores),
            "sda": safe_auroc(stream_y, sda_scores),
            "oracle": safe_auroc(oracle_labels, oracle_scores),
        },
        "per_domain_auroc": {
            "freeze": mean_dict(segment_aurocs(freeze_scores, stream_y, stream_segments)),
            "ours": mean_dict(segment_aurocs(ours_scores, stream_y, stream_segments)),
            "sda": mean_dict(segment_aurocs(sda_scores, stream_y, stream_segments)),
            "oracle": mean_dict(segment_aurocs(oracle_scores, oracle_labels, oracle_segments)),
        },
        "visit_auroc": {
            "freeze": freeze_visits,
            "ours": ours_visits,
            "sda": sda_visits,
            "oracle": oracle_visits,
        },
        "robustness": {
            "freeze": robustness_summary(freeze_visits, args.collapse_threshold),
            "ours": robustness_summary(ours_visits, args.collapse_threshold),
            "sda": robustness_summary(sda_visits, args.collapse_threshold),
            "oracle": robustness_summary(oracle_visits, args.collapse_threshold),
        },
        "rolling_robustness": {
            "window": int(args.rolling_window),
            "step": int(args.rolling_step),
            "freeze": rolling_auroc_stats(
                freeze_scores, stream_y,
                window=args.rolling_window,
                step=args.rolling_step,
                collapse_threshold=args.collapse_threshold,
            ),
            "ours": rolling_auroc_stats(
                ours_scores, stream_y,
                window=args.rolling_window,
                step=args.rolling_step,
                collapse_threshold=args.collapse_threshold,
            ),
            "sda": rolling_auroc_stats(
                sda_scores, stream_y,
                window=args.rolling_window,
                step=args.rolling_step,
                collapse_threshold=args.collapse_threshold,
            ),
            "oracle": rolling_auroc_stats(
                oracle_scores, oracle_labels,
                window=args.rolling_window,
                step=args.rolling_step,
                collapse_threshold=args.collapse_threshold,
            ),
        },
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
