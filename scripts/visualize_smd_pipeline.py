"""
Visualize one SMD sample through the On-autoPilot pipeline.

This script shows:
1. Normal vs anomaly raw SMD windows
2. Normal vs anomaly LLM-encoded features
3. Normal vs anomaly INR latent features and reconstructions
4. How a task is registered into the CPM memory bank

Example:
  python scripts/visualize_smd_pipeline.py --output_dir outputs/smd_viz
  d:/GIT/others/paper/.venv/Scripts/python.exe scripts/visualize_smd_pipeline.py --output_dir outputs/smd_pipeline_viz --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from on_autopilot import OnAutoPilot
from on_autopilot.models import PromptModule


def build_windows(data: np.ndarray, seq_len: int, stride: int) -> torch.Tensor:
    starts = list(range(0, len(data) - seq_len + 1, stride))
    windows = [data[start : start + seq_len] for start in starts]
    return torch.from_numpy(np.stack(windows).astype(np.float32))


def load_smd(data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = data_root / "smd"
    train = np.load(base / "train.npy")
    test = np.load(base / "test.npy")
    labels = np.load(base / "labels.npy")
    return train, test, labels


def choose_anomaly_window(test: np.ndarray, labels: np.ndarray, seq_len: int) -> tuple[np.ndarray, int]:
    anomaly_idx = np.flatnonzero(labels > 0)
    if len(anomaly_idx) == 0:
        start = 0
    else:
        center = int(anomaly_idx[0])
        start = max(0, min(len(test) - seq_len, center - seq_len // 2))
    return test[start : start + seq_len], start


def choose_normal_window(test: np.ndarray, labels: np.ndarray, seq_len: int) -> tuple[np.ndarray, int]:
    usable = len(test) - seq_len + 1
    for start in range(0, max(0, usable)):
        if labels[start : start + seq_len].sum() == 0:
            return test[start : start + seq_len], start
    return test[:seq_len], 0


def save_heatmap(data: np.ndarray, title: str, out_path: Path, xlabel: str, ylabel: str) -> None:
    fig: Figure
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(data.T, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_side_by_side_heatmaps(
    left: np.ndarray,
    right: np.ndarray,
    left_title: str,
    right_title: str,
    super_title: str,
    out_path: Path,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    vmin = min(left.min(), right.min())
    vmax = max(left.max(), right.max())
    for ax, arr, title in zip(axes, [left, right], [left_title, right_title]):
        im = ax.imshow(arr.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(super_title)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_reconstruction_comparison(raw: np.ndarray, recon: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    vmin = min(raw.min(), recon.min())
    vmax = max(raw.max(), recon.max())

    axes[0].imshow(raw.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Raw SMD Window")
    axes[0].set_ylabel("Channel")

    axes[1].imshow(recon.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("INR Reconstruction")
    axes[1].set_ylabel("Channel")

    diff = np.abs(raw - recon)
    im = axes[2].imshow(diff.T, aspect="auto", cmap="magma")
    axes[2].set_title("Absolute Reconstruction Error")
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Channel")
    fig.colorbar(im, ax=axes[2], fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_dual_reconstruction_comparison(
    normal_raw: np.ndarray,
    normal_recon: np.ndarray,
    anomaly_raw: np.ndarray,
    anomaly_recon: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey="row")

    normal_diff = np.abs(normal_raw - normal_recon)
    anomaly_diff = np.abs(anomaly_raw - anomaly_recon)

    vmin = min(normal_raw.min(), normal_recon.min(), anomaly_raw.min(), anomaly_recon.min())
    vmax = max(normal_raw.max(), normal_recon.max(), anomaly_raw.max(), anomaly_recon.max())

    axes[0, 0].imshow(normal_raw.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Normal Raw")
    axes[0, 1].imshow(normal_recon.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Normal INR Recon")
    im0 = axes[0, 2].imshow(normal_diff.T, aspect="auto", cmap="magma")
    axes[0, 2].set_title("Normal Abs Error")

    axes[1, 0].imshow(anomaly_raw.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Anomaly Raw")
    axes[1, 1].imshow(anomaly_recon.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("Anomaly INR Recon")
    im1 = axes[1, 2].imshow(anomaly_diff.T, aspect="auto", cmap="magma")
    axes[1, 2].set_title("Anomaly Abs Error")

    axes[0, 0].set_ylabel("Channel")
    axes[1, 0].set_ylabel("Channel")
    for ax in axes[1, :]:
        ax.set_xlabel("Time Step")

    fig.colorbar(im0, ax=axes[0, 2], fraction=0.04, pad=0.02)
    fig.colorbar(im1, ax=axes[1, 2], fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def analyze_window(model: OnAutoPilot, window_np: np.ndarray, device: torch.device) -> dict:
    window = torch.from_numpy(window_np.astype(np.float32)).unsqueeze(0).to(device)
    t_sample = model._time_coords(1)

    with torch.no_grad():
        z_llm = model.llm_encoder(window)
        omega = model.weight_predictor(z_llm)
        z_query = model.inr.get_latent(t_sample, omega=omega)
        k_query = model.cpm_bank.generate_key(z_query.reshape(-1, z_query.shape[-1]))
        omega_prompted = model.prompt(omega)
        z_inr = model.inr.get_latent(t_sample, omega=omega_prompted)
        x_hat = model.inr(t_sample, omega=omega_prompted)
        score, z_mean, _ = model.forward(window)

    return {
        "raw": window.squeeze(0).cpu().numpy(),
        "llm": z_llm.squeeze(0).cpu().numpy(),
        "inr": z_inr.squeeze(0).cpu().numpy(),
        "recon": x_hat.squeeze(0).cpu().numpy(),
        "score": float(score.item()),
        "latent_norm": float(np.linalg.norm(z_mean.cpu().numpy())),
        "k_query": k_query.detach().cpu().numpy(),
    }


def save_memory_bank_plot(z_buf: np.ndarray, kn: np.ndarray, key: np.ndarray, out_path: Path) -> None:
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_buf)
    kn_2d = pca.transform(kn)
    key_2d = pca.transform(key.reshape(1, -1))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(z_2d[:, 0], z_2d[:, 1], s=10, alpha=0.25, label="Z_buf (all latent points)")
    ax.scatter(kn_2d[:, 0], kn_2d[:, 1], s=35, c="tab:red", label="Kn (coreset)")
    ax.scatter(key_2d[:, 0], key_2d[:, 1], s=90, c="black", marker="*", label="k_new")
    ax.set_title("CPM Memory Bank Registration View (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize SMD raw -> LLM -> INR -> CPM pipeline")
    parser.add_argument("--data_root", default="data", help="Root directory containing smd/")
    parser.add_argument("--output_dir", default="outputs/smd_pipeline_viz", help="Directory to save figures")
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--buffer_windows", type=int, default=32)
    parser.add_argument("--llm_hidden", type=int, default=64)
    parser.add_argument("--num_groups", type=int, default=2)
    parser.add_argument("--inner_steps", type=int, default=2)
    parser.add_argument("--outer_steps", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    train, test, labels = load_smd(Path(args.data_root))

    mean = train.mean(0, keepdims=True)
    std = train.std(0, keepdims=True) + 1e-8
    train = (train - mean) / std
    test = (test - mean) / std

    model = OnAutoPilot(
        input_dim=train.shape[1],
        seq_len=args.seq_len,
        llm_hidden=args.llm_hidden,
        inr_global_hidden=64,
        prompt_dim=64,
        num_groups=args.num_groups,
        inner_steps=args.inner_steps,
        outer_steps=args.outer_steps,
        device=device,
    )
    model.eval()

    normal_window_np, normal_start = choose_normal_window(test, labels, args.seq_len)
    anomaly_window_np, anomaly_start = choose_anomaly_window(test, labels, args.seq_len)

    normal = analyze_window(model, normal_window_np, device)
    anomaly = analyze_window(model, anomaly_window_np, device)

    save_side_by_side_heatmaps(
        normal["raw"],
        anomaly["raw"],
        f"Normal Raw (start={normal_start})",
        f"Anomaly Raw (start={anomaly_start})",
        "SMD Raw Windows: Normal vs Anomaly",
        out_dir / "01_raw_normal_vs_anomaly.png",
        "Time Step",
        "Channel",
    )
    save_side_by_side_heatmaps(
        normal["llm"],
        anomaly["llm"],
        "Normal LLM Features",
        "Anomaly LLM Features",
        "LLM Encoder Output: Normal vs Anomaly",
        out_dir / "02_llm_normal_vs_anomaly.png",
        "Time Step",
        "LLM Feature",
    )
    save_side_by_side_heatmaps(
        normal["inr"],
        anomaly["inr"],
        "Normal INR Latent",
        "Anomaly INR Latent",
        "INR Latent Representation: Normal vs Anomaly",
        out_dir / "03_inr_normal_vs_anomaly.png",
        "Time Step",
        "Latent Feature",
    )
    save_dual_reconstruction_comparison(
        normal["raw"],
        normal["recon"],
        anomaly["raw"],
        anomaly["recon"],
        out_dir / "04_reconstruction_normal_vs_anomaly.png",
    )

    train_windows = build_windows(train, args.seq_len, args.stride)[: args.buffer_windows]
    x_train = train_windows.to(device)
    n_train = max(1, int(0.8 * x_train.shape[0]))
    x_buf_train = x_train[:n_train]
    x_buf_val = x_train[n_train:] if n_train < x_train.shape[0] else x_train[-1:]
    t_train = model._time_coords(x_buf_train.shape[0])
    t_val = model._time_coords(x_buf_val.shape[0])

    p_new = PromptModule(prompt_dim=model.prompt.prompt.shape[0]).to(device)
    model.augmentation.reset_parameters()
    theta_warm = model.cpm_bank.retrieve_latest_theta()
    model.self_tuning_engine.run(
        inr=model.inr,
        weight_pred=model.weight_predictor,
        prompt=p_new,
        llm_encoder=model.llm_encoder,
        augmentation=model.augmentation,
        x_train=x_buf_train,
        t_coords=t_train,
        x_val=x_buf_val,
        t_val=t_val,
        theta_init=theta_warm,
    )

    with torch.no_grad():
        z_llm_buf = model.llm_encoder(x_buf_train)
        omega_buf = model.weight_predictor(z_llm_buf)
        omega_buf = p_new(omega_buf)
        z_seq_buf = model.inr.get_latent(t_train, omega=omega_buf)
        z_buf = z_seq_buf.reshape(-1, z_seq_buf.shape[-1])

    k_new = anomaly["k_query"]
    kn_new = model.cpm_bank.build_kn(z_buf)
    theta_snap = model.self_tuning_engine.snapshot_theta(model.inr, model.weight_predictor)
    model.cpm_bank.register(torch.from_numpy(k_new), p_new, kn_new, theta_snap)

    save_memory_bank_plot(
        z_buf.cpu().numpy(),
        kn_new.cpu().numpy(),
        np.asarray(k_new),
        out_dir / "05_memory_bank_registration.png",
    )

    memory_entry = model.cpm_bank._bank[-1]
    summary = {
        "normal_window": {
            "start_index": int(normal_start),
            "raw_shape": list(normal["raw"].shape),
            "llm_shape": list(normal["llm"].shape),
            "inr_latent_shape": list(normal["inr"].shape),
            "reconstruction_shape": list(normal["recon"].shape),
            "anomaly_score": normal["score"],
            "latent_norm": normal["latent_norm"],
        },
        "anomaly_window": {
            "start_index": int(anomaly_start),
            "raw_shape": list(anomaly["raw"].shape),
            "llm_shape": list(anomaly["llm"].shape),
            "inr_latent_shape": list(anomaly["inr"].shape),
            "reconstruction_shape": list(anomaly["recon"].shape),
            "anomaly_score": anomaly["score"],
            "latent_norm": anomaly["latent_norm"],
        },
        "memory_bank_registration": {
            "bank_size": len(model.cpm_bank),
            "key_shape": list(memory_entry["key"].shape),
            "prompt_keys": list(memory_entry["prompt"].keys()),
            "kn_shape": list(memory_entry["kn"].shape),
            "theta_keys": list(memory_entry["theta"].keys()) if memory_entry["theta"] is not None else [],
            "buffer_windows": int(x_buf_train.shape[0]),
        },
        "saved_files": [
            "01_raw_normal_vs_anomaly.png",
            "02_llm_normal_vs_anomaly.png",
            "03_inr_normal_vs_anomaly.png",
            "04_reconstruction_normal_vs_anomaly.png",
            "05_memory_bank_registration.png",
            "summary.json",
        ],
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Saved pipeline visualizations to: {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()