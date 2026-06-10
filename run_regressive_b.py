from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = ROOT / "scripts" / "compare_multidomain_baselines.py"
OUTPUT_PATH = ROOT / "outputs" / "aggressive_B_results.json"
SEEDS = [42, 123, 456]


def mean_std(values: list[float]) -> dict[str, float]:
    if len(values) == 0:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)),
    }


def run_seed(seed: int) -> dict:
    cmd = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--scenario", "two_domain",
        "--data_path", "./data",
        "--device", "cpu",
        "--seed", str(seed),
        "--max_train_windows", "32",
        "--max_test_points", "5000",
        "--outer_steps", "2",
        "--inner_steps", "1",
        "--oracle_outer_steps", "6",
        "--oracle_inner_steps", "3",
        "--oracle_max_train_windows", "64",
        "--stride", "10",
        "--adapt_buffer", "10",
        "--tau_drift", "0.9",
        "--collapse_threshold", "0.55",
        "--rolling_window", "1000",
        "--rolling_step", "100",
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    rr = payload["rolling_robustness"]
    return {
        "config": "aggressive_B",
        "seed": seed,
        "ours_collapse": float(rr["ours"]["collapse_count"]),
        "ours_below_random": float(rr["ours"]["below_random_count"]),
        "sda_collapse": float(rr["sda"]["collapse_count"]),
        "sda_below_random": float(rr["sda"]["below_random_count"]),
    }


def main() -> None:
    raw_rows = [run_seed(seed) for seed in SEEDS]

    ours_collapse = [row["ours_collapse"] for row in raw_rows]
    ours_below_random = [row["ours_below_random"] for row in raw_rows]
    sda_collapse = [row["sda_collapse"] for row in raw_rows]
    sda_below_random = [row["sda_below_random"] for row in raw_rows]

    result = {
        "metric_source": "rolling_robustness",
        "collapse_threshold": 0.55,
        "config": "aggressive_B",
        "seeds": SEEDS,
        "summary": {
            "ours_collapse": mean_std(ours_collapse),
            "ours_below_random": mean_std(ours_below_random),
            "sda_collapse": mean_std(sda_collapse),
            "sda_below_random": mean_std(sda_below_random),
        },
        "raw": raw_rows,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
