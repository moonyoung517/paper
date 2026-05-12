"""
train.py — Training and evaluation entry point for On-autoPilot.

Supports the following public MTS anomaly detection benchmarks:
  • SWaT  (Secure Water Treatment)
  • SMAP  (Soil Moisture Active Passive)
  • MSL   (Mars Science Laboratory)
  • SMD   (Server Machine Dataset)

Usage (offline simulation of a continual stream)
-------------------------------------------------
    python train.py \\
        --dataset swat \\
        --data_path ./data/swat \\
        --seq_len 100 \\
        --stride 1 \\
        --tau_drift 0.65 \\
        --outer_steps 100 \\
        --adapt_buffer 200 \\
        --device cuda

Dataset file convention
-----------------------
Each dataset directory should contain:
    train.npy   — (T_train, d)  unlabelled normal data
    test.npy    — (T_test,  d)  test data (may contain anomalies)
    labels.npy  — (T_test,)     binary anomaly labels (0/1 per timestamp)
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from on_autopilot import OnAutoPilot


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

class SlideWindowDataset(torch.utils.data.Dataset):
    """Sliding-window dataset for offline pre-processing (used in evaluation)."""

    def __init__(self, data: np.ndarray, seq_len: int, stride: int = 1):
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.stride = stride
        T = len(data)
        self.indices = list(range(0, T - seq_len + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = self.indices[idx]
        return torch.from_numpy(self.data[start:start + self.seq_len])


def load_dataset(
    data_path: str,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train / test / labels arrays for the specified dataset.

    Returns
    -------
    train_data : (T_train, d)
    test_data  : (T_test,  d)
    labels     : (T_test,)
    """
    base = os.path.join(data_path, dataset_name.lower())

    def _load(fname: str) -> np.ndarray:
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset file not found: {path}\n"
                "Download the dataset and place train.npy / test.npy / labels.npy "
                f"in {base}/"
            )
        return np.load(path)

    train_data = _load("train.npy")
    test_data = _load("test.npy")
    labels = _load("labels.npy")

    # Z-score normalisation (fit on train)
    mean = train_data.mean(0, keepdims=True)
    std = train_data.std(0, keepdims=True) + 1e-8
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    print(
        f"[{dataset_name.upper()}] "
        f"train={train_data.shape}  test={test_data.shape}  "
        f"anomaly_rate={labels.mean():.3f}"
    )
    return train_data, test_data, labels


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """
    Sweep thresholds and return (best F1, best threshold).
    Applies the standard point-adjustment strategy common in MTS benchmarks.
    """
    thresholds = np.percentile(scores, np.linspace(80, 99.9, 200))
    best_f1 = 0.0
    best_thr = thresholds[0]
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        preds = _point_adjust(preds, labels)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_f1, best_thr


def _point_adjust(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Point-adjust: if any point in an anomaly segment is detected,
    mark the whole segment as detected (standard in MTS-AD evaluation).
    """
    adjusted = preds.copy()
    in_anomaly = False
    seg_start = 0
    for i, label in enumerate(labels):
        if label == 1 and not in_anomaly:
            in_anomaly = True
            seg_start = i
        elif label == 0 and in_anomaly:
            # Check if segment was detected
            if adjusted[seg_start:i].any():
                adjusted[seg_start:i] = 1
            in_anomaly = False
    if in_anomaly and adjusted[seg_start:].any():
        adjusted[seg_start:] = 1
    return adjusted


def evaluate(
    model: OnAutoPilot,
    test_data: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
    stride: int,
    adapt_buffer: int,
    device: torch.device,
) -> dict:
    """
    Run model on test stream and compute AUROC, AP, best-F1.

    Returns
    -------
    dict with keys: auroc, ap, f1, threshold
    """
    model.eval()
    x_stream = torch.from_numpy(test_data.astype(np.float32)).to(device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = model.detect_and_adapt(
            x_stream,
            window_size=seq_len,
            stride=stride,
            adapt_buffer=adapt_buffer,
        )

    scores_raw = np.array([r["score"] for r in results])
    # Map window scores back to per-timestep scores (last-window rule)
    n_timesteps = len(labels)
    timestep_scores = np.full(n_timesteps, scores_raw[0])
    for i, r in enumerate(results):
        t = r["t"]
        end = min(t + seq_len, n_timesteps)
        timestep_scores[t:end] = np.maximum(timestep_scores[t:end], r["score"])

    auroc = roc_auc_score(labels, timestep_scores)
    ap = average_precision_score(labels, timestep_scores)
    f1, thr = best_f1_threshold(timestep_scores, labels)

    return {"auroc": auroc, "ap": ap, "f1": f1, "threshold": thr}


# ---------------------------------------------------------------------------
# Initial offline warm-up on training data
# ---------------------------------------------------------------------------

def warmup_on_train(
    model: OnAutoPilot,
    train_data: np.ndarray,
    seq_len: int,
    adapt_buffer: int,
    device: torch.device,
) -> None:
    """
    Run Algorithm 1 on the training stream to initialise CPM bank entries
    before test-time evaluation.
    """
    model.train()
    x_train = torch.from_numpy(train_data.astype(np.float32)).to(device)
    print("[Warm-up] Processing training stream …")
    model.detect_and_adapt(
        x_train,
        window_size=seq_len,
        stride=seq_len // 2,   # 50% overlap for faster warm-up
        adapt_buffer=adapt_buffer,
    )
    print(f"[Warm-up] CPM bank size after training: {len(model.cpm_bank)} task(s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="On-autoPilot training/eval")
    p.add_argument("--dataset", default="swat",
                   choices=["swat", "smap", "msl", "smd"],
                   help="Benchmark dataset name")
    p.add_argument("--data_path", default="./data",
                   help="Root directory containing dataset folders")
    p.add_argument("--seq_len", type=int, default=100,
                   help="Sliding window length W")
    p.add_argument("--stride", type=int, default=1,
                   help="Sliding window stride (test inference)")
    p.add_argument("--num_groups", type=int, default=4,
                   help="Number of sensor groups for GroupBasedINR")
    p.add_argument("--llm_hidden", type=int, default=128,
                   help="LLM embedding dimension (use 128 for speed, 768 for GPT-2)")
    p.add_argument("--tau_drift", type=float, default=0.65,
                   help="Cosine-sim threshold for regime transition")
    p.add_argument("--lambda_score", type=float, default=0.5,
                   help="Weighting between reconstruction and latent score")
    p.add_argument("--adapt_buffer", type=int, default=200,
                   help="Number of windows to accumulate before self-tuning")
    p.add_argument("--inner_lr", type=float, default=1e-3)
    p.add_argument("--outer_lr", type=float, default=1e-3)
    p.add_argument("--inner_steps", type=int, default=5)
    p.add_argument("--outer_steps", type=int, default=100)
    p.add_argument("--coreset_size", type=int, default=64)
    p.add_argument("--fps_samples", type=int, default=32)
    p.add_argument("--device", default="auto",
                   help="'auto', 'cpu', or 'cuda'")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Device ------------------------------------------------------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ---- Reproducibility ---------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Data --------------------------------------------------------------
    train_data, test_data, labels = load_dataset(args.data_path, args.dataset)
    input_dim = train_data.shape[1]

    # Ensure input_dim is divisible by num_groups
    num_groups = args.num_groups
    while input_dim % num_groups != 0:
        num_groups -= 1
    if num_groups != args.num_groups:
        print(f"[Warning] num_groups adjusted {args.num_groups} → {num_groups} "
              f"to divide input_dim={input_dim}")

    # ---- Model -------------------------------------------------------------
    model = OnAutoPilot(
        input_dim=input_dim,
        seq_len=args.seq_len,
        llm_hidden=args.llm_hidden,
        num_groups=num_groups,
        tau_drift=args.tau_drift,
        lambda_score=args.lambda_score,
        fps_samples=args.fps_samples,
        coreset_size=args.coreset_size,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        inner_steps=args.inner_steps,
        outer_steps=args.outer_steps,
        device=device,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: total={total_params:,}  trainable={trainable:,}")

    # ---- Warm-up on training stream ----------------------------------------
    warmup_on_train(model, train_data, args.seq_len, args.adapt_buffer, device)

    # ---- Evaluation on test stream -----------------------------------------
    print("[Test] Running anomaly detection on test stream …")
    metrics = evaluate(
        model, test_data, labels,
        seq_len=args.seq_len,
        stride=args.stride,
        adapt_buffer=args.adapt_buffer,
        device=device,
    )

    print("\n=== Results ===")
    print(f"  AUROC : {metrics['auroc']:.4f}")
    print(f"  AP    : {metrics['ap']:.4f}")
    print(f"  F1    : {metrics['f1']:.4f}  (threshold={metrics['threshold']:.4f})")
    print(f"  CPM bank entries: {len(model.cpm_bank)}")


if __name__ == "__main__":
    main()
