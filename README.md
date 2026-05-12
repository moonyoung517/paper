## Quick Setup (Current Code)

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) (Optional) Download pretrained GPT-2 if internet is available
python download_gpt2.py --method transformers --save_dir ./gpt2_model

# 3) Set model path (Windows PowerShell)
$env:HF_MODEL_PATH="./gpt2_model"

# 4) If only txt files are available, restore npy files first
python scripts/npy_txt_converter.py txt2npy --input data --recursive

# 5) Train / evaluate
python train.py --dataset smd --data_path ./data --device cpu
```

Notes:
- `train.py` reads `train.npy`, `test.npy`, `labels.npy` from each dataset folder.
- `.txt` files are optional for transport/inspection and must be converted back to `.npy` before running `train.py`.
- If pretrained GPT-2 cannot be downloaded, code still runs with the internal Transformer fallback.

### Pretrained GPT-2 On Another PC

If you want the same code to run on another PC with the pretrained GPT-2 path enabled:

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Restore dataset files if only txt files were copied
python scripts/npy_txt_converter.py txt2npy --input data --recursive

# 3) Download GPT-2 on an internet-enabled PC
python download_gpt2.py --method transformers --save_dir ./gpt2_model

# 4) Copy the project folder and gpt2_model folder to the target PC

# 5) Set the model path on the target PC (Windows PowerShell)
$env:HF_MODEL_PATH="./gpt2_model"
(export HF_MODEL_PATH="./gpt2_model" in mac)
# 6) Verify the model path if needed
python check_gpt2_status.py

# 7) Run training / evaluation
python train.py --dataset smd --data_path ./data --device cpu
```

If `HF_MODEL_PATH` is not set or the folder is missing, the code falls back to the internal non-pretrained Transformer encoder.




# On-autoPilot: Unsupervised Continual Anomaly Detection for Industrial MTS

> **"Frozen LLM-Guided Self-Tuning with Autonomous Regime Adaptation"**

Official PyTorch implementation accompanying the paper submission to AAAI 2026.

---

## Overview

Industrial multivariate time-series (MTS) streams are non-stationary: sensors drift, operating regimes shift, and anomaly patterns evolve — all without explicit task boundaries or ground-truth labels.

**On-autoPilot** addresses this with three tightly coupled components:

| Component | Role |
|---|---|
| **Frozen LLM Encoder** | Amplify structural breaks via GPT-2 attention (frozen weights) |
| **Group-based SIREN INR** | Reconstruct normal manifold; spectral bias inherently rejects anomalous fluctuations |
| **Differentiable Self-Tuning Engine** | Bi-level optimisation — inner loop adapts detector θ, outer loop sharpens anomaly synthesis profile *a* |
| **Continual Prompting Module (CPM)** | Detect regime transitions without task boundaries; store/retrieve (key, prompt, normality standard) triplets |

### Key Contributions

1. **Boundary-free regime detection** via cosine similarity on FPS-derived task keys (no explicit change-point labels required).
2. **Warm-start continual learning** — θ is initialised from the most recent CPM bank entry (θ_{t-1}), enabling rapid post-transition adaptation.
3. **Differentiable anomaly synthesis** — 6-type augmentation profile with learnable magnitudes, optimised end-to-end via Sliced Wasserstein Distance.
4. **Explainable root-cause localisation** via input-gradient saliency ĝ = ∇_x S(t).

---

## Repository Structure

```
paper/
├── on_autopilot/              # Core Python package
│   ├── __init__.py            # Public API
│   ├── models.py              # FrozenLLMEncoder, SIRENLayer, GroupBasedINR,
│   │                          #   WeightPredictor, PromptModule
│   ├── cpm.py                 # CPMMemoryBank, furthest_point_sampling,
│   │                          #   coreset_sampling
│   ├── augmentation.py        # DifferentiableAugmentation (6 anomaly types)
│   ├── losses.py              # sliced_wasserstein, TSContrastiveLoss
│   ├── self_tuning.py         # SelfTuningEngine (bi-level optimisation)
│   └── framework.py           # OnAutoPilot — Algorithm 1 entry point
├── train.py                   # Training / evaluation script
├── main.tex                   # AAAI 2026 paper source
├── main_iclr.tex              # ICLR version
└── references.bib             # Bibliography
```

---

## Installation

```bash
# Python >= 3.9 recommended
pip install -r requirements.txt
```

No additional build steps required — the `on_autopilot` package is used directly from the repository root.

---

## Datasets

Download the benchmark datasets and place them under `data/<dataset_name>/`:

| Dataset | Channels | Description | Download |
|---|---|---|---|
| **SWaT** | 51 | Secure Water Treatment plant | [iTrust](https://itrust.sutd.edu.sg/itrust-labs_datasets/) |
| **SMAP** | 25 | NASA soil moisture satellite | [NASA](https://github.com/khundman/telemanom) |
| **MSL** | 55 | NASA Mars rover telemetry | [NASA](https://github.com/khundman/telemanom) |
| **SMD** | 38 | Server machine cluster | [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) |

Each directory must contain:

```
data/
└── swat/
    ├── train.npy     # (T_train, d)  unlabelled normal windows
    ├── test.npy      # (T_test,  d)  evaluation stream
    └── labels.npy    # (T_test,)     binary anomaly labels (0/1)
```

---

## Quick Start

```bash
# SWaT benchmark (GPU)
python train.py \
    --dataset swat \
    --data_path ./data \
    --seq_len 100 \
    --tau_drift 0.65 \
    --outer_steps 100 \
    --adapt_buffer 200 \
    --device cuda

# SMD benchmark (CPU, small LLM hidden dim for speed)
python train.py \
    --dataset smd \
    --data_path ./data \
    --seq_len 100 \
    --llm_hidden 128 \
    --num_groups 2 \
    --device cpu

# Visualize one SMD normal window and one anomaly window through
# raw -> LLM -> INR -> CPM registration
python scripts/visualize_smd_pipeline.py --output_dir outputs/smd_pipeline_viz --device cpu
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `swat` | `swat` / `smap` / `msl` / `smd` |
| `--seq_len` | `100` | Sliding window length W |
| `--tau_drift` | `0.65` | Cosine-sim threshold for regime transition detection |
| `--lambda_score` | `0.5` | Balance between reconstruction and latent-distance score |
| `--adapt_buffer` | `200` | Windows to accumulate before triggering self-tuning |
| `--outer_steps` | `100` | Outer-loop iterations of bi-level optimisation |
| `--llm_hidden` | `128` | LLM embedding dim (128 for speed, 768 for full GPT-2) |
| `--num_groups` | `4` | Channel groups in GroupBasedINR |
| `--coreset_size` | `64` | Normality standard Kn_t size |
| `--fps_samples` | `32` | FPS samples for task key generation |

---

## Architecture Details

### Anomaly Scoring

$$S(t) = \lambda \cdot \|x_t - \hat{x}_t\|^2 + (1-\lambda) \cdot \min_{m \in K_n^t} \text{dist}(z_t,\, m)$$

- **Reconstruction term**: INR spectral bias rejects high-frequency anomalous patterns.
- **Latent-distance term**: Distance from current latent z_t to CoreSet normality standard K_n^t.

### Bi-level Self-Tuning

```
Inner loop:  θ* = argmin_θ  L_recon(θ) + L_SCL(θ; z_anchor, z_pos, z_neg(a))
Outer loop:  a* = argmin_a  SWD(Z_trn ∪ Z_aug(a), Z_val)
```

During each new regime adaptation, both task prompt `p_new` and augmentation magnitudes are re-initialized and then optimized.

### CPM Memory Bank Entry

Each detected regime stores a triplet **(k_t, p_t, K_n^t)**:
- **k_t** — task key (mean of FPS-selected latent points)
- **p_t** — task-specific prompt (learnable bias)
- **K_n^t** — normality standard (greedy CoreSet of Z_trn)
- **θ_t** — detector snapshot for warm-start at next transition

### Anomaly Augmentation Types

| ID | Type | Effect |
|---|---|---|
| 0 | Mean Shift | Sustained level change |
| 1 | Platform | Consecutive constant segment |
| 2 | Trend | Monotone linear drift |
| 3 | Amplitude Shift | Scaled signal variance |
| 4 | Extremum | Isolated spike / drop |
| 5 | Frequency Shift | Resampled / stretched subsequence |

All magnitudes are learnable parameters optimised by the outer loop.

---

## Programmatic API

```python
from on_autopilot import OnAutoPilot
import torch

model = OnAutoPilot(
    input_dim=51,       # SWaT: 51 channels
    seq_len=100,
    tau_drift=0.65,
    device=torch.device("cuda"),
)

# Anomaly score for a batch of windows
x = torch.randn(8, 100, 51)          # (B, W, d)
scores, z, x_hat = model(x)          # scores: (8,)

# Gradient saliency for root-cause localisation
saliency = model.compute_saliency(x) # (8, 51) — per-channel importance

# Stream-level inference with autonomous adaptation (Algorithm 1)
x_stream = torch.randn(10000, 51)    # (T, d)
results = model.detect_and_adapt(x_stream, window_size=100)
# results: list of {'t': int, 'score': float, 'task': int, 's_max': float}
```

---

## Evaluation Protocol

- **Metrics**: AUROC, Average Precision (AP), F1 (best threshold sweep)
- **Point-adjust**: Standard MTS-AD convention — if any point in an anomaly segment is detected, the full segment is credited.
- **Normalisation**: Z-score, fit on training split only.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{onAutoPilot2026,
  title     = {On-autoPilot: Unsupervised Continual Anomaly Detection
               for Industrial Multivariate Time Series},
  booktitle = {Proceedings of the 40th AAAI Conference on Artificial Intelligence},
  year      = {2026},
}
```

---

## License

This repository is released for research purposes only.
