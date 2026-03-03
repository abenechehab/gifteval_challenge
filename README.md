# TSFC — Time Series Forecasting with Learned Mixtures

A **learned mixture ensemble** of zero-shot foundation models (Chronos-2, TimesFM-2.5, Moirai-2) trained on the [GiftEvalPretrain](https://huggingface.co/datasets/theforecastingcompany/GiftEvalPretrain) dataset. Scalar (or instance-adaptive) per-model weights — plus an optional patch-transformer boosting head — are optimised by gradient descent on sampled training windows.

---

## Repository layout

```
tsfc/                     # installable package
├── data/
│   ├── gifteval.py       # GiftEvalDataset (train/val/test) + GiftEvalDataModule
│   ├── transforms.py     # normalization, freq encoding, feature extraction
│   └── utils.py          # NaN filling, freq string normalization
├── models/
│   ├── base.py           # ForecastResult dataclass + ForecastingModel ABC
│   ├── registry.py       # register / load_model / list_models
│   ├── chronos2.py       # Chronos-2 wrapper
│   ├── timesfm.py        # TimesFM-2.5 wrapper
│   ├── moirai.py         # Moirai-2 wrapper
│   └── mixture.py        # WeightedMixture + MetaWeighter + BoostingForecaster
├── train/
│   ├── loss.py           # MAE / MSE / MASE / CRPS losses
│   └── trainer.py        # cache-based mixture training loop (legacy)
└── eval/
    └── metrics.py        # MASE, CRPS, WQL, evaluate_dataset()

scripts/
├── explore_data.py           # inspect GiftEvalPretrain subsets
├── train_mixture_online.py   # *** MAIN TRAINING SCRIPT ***
├── cache_predictions.py      # (optional) pre-cache frozen model outputs to disk
├── train_mixture.py          # (legacy) train from pre-cached predictions
└── evaluate.py               # evaluate a single model or trained mixture checkpoint

tests/
└── test_data.py              # pytest unit tests
```

---

## Installation

### 1. Core package

```bash
uv sync
```

> **Python version:** 3.11 is pinned via `.python-version`.

### 2. Foundation model extras

Each backbone model is an optional extra defined in `pyproject.toml`. Install only the ones you need — they have conflicting transitive dependencies so do not install all three into the same solver call.

#### Chronos-2

```bash
uv sync --extra chronos
# installs: chronos-forecasting>=2.1.0
```

#### TimesFM-2.5

```bash
uv sync --extra timesfm
# installs: timesfm from git+https://github.com/google-research/timesfm.git
```

#### Moirai-2

```bash
uv sync --extra moirai
# installs: uni2ts>=1.2.0 (gluonts is pulled in as a transitive dependency)
```

> **Solver conflicts:** `uni2ts` pins `gluonts~=0.14.3` and `datasets~=2.17.1`, which can clash with other extras. If `uv` cannot resolve, install directly:
> ```bash
> uv run pip install "uni2ts>=1.2.0"
> ```

### 3. Dev tools

```bash
uv sync --extra dev
# installs: pytest, pytest-cov, ruff, pyright
```

---

## Workflow

```
explore_data.py  →  train_mixture_online.py  →  evaluate.py
```

`train_mixture_online.py` is the **primary training script**. It runs the frozen backbone models live during every training step — no pre-caching required. A full train / val / test split is used, all metrics are logged to TensorBoard and CSV, and individual backbone model baselines are compared automatically at the end.

---

## Scripts

All scripts use [tyro](https://brentyi.github.io/tyro/) for CLI parsing. Every argument maps 1-to-1 to a field in the script's `@dataclass` config. Pass `--help` to any script to see the full argument list and defaults.

---

### `scripts/train_mixture_online.py` ⭐ main script

Trains the `WeightedMixture` model with **on-the-fly backbone inference** — no pre-cached predictions are needed. The three data splits used are:

| Split | Content |
|---|---|
| **train** | Random windows sampled from the first `(1 − val_frac)` of each series |
| **val** | Single window whose target immediately follows the train boundary |
| **test** | Single window whose target follows the val window (completely unseen) |

At the end of training the script evaluates the mixture **and each individual backbone model** on the test split, prints a comparison table, logs everything to TensorBoard, and writes results to CSV.

**Outputs** (all under an auto-timestamped run directory):

```
logs/
└── YYYY-MM-DD_HH-MM-SS[_run_name]/
    ├── events.out.tfevents.*   # TensorBoard events
    ├── config.json             # full config snapshot
    └── results.csv             # final test metrics (mixture + per model)

checkpoints/
└── mixture_epoch<NNN>.pt       # saved after every epoch
```

**TensorBoard scalars logged:**

| Prefix | Frequency | Metrics |
|---|---|---|
| `train/` | every step and epoch | `loss`, `mse`, `mae`, `mase`, `grad_norm`, `lr`, `epoch_time_s` |
| `mixture/weight/` | every step | per-model batch-averaged softmax weight |
| `mixture/weight_epoch/`, `mixture/log_weight/` | every epoch (static mode) | softmax and raw log-weights |
| `param_norm/` | every epoch | L2 norm of each learnable parameter |
| `grad_norm/` | every step | per-parameter gradient L2 norm |
| `val/` | every `val_every_n_epochs` epochs | `loss`, `mse`, `mae`, `mase` |
| `test/mixture/` and `test/<model>/` | once, after training | `loss`, `mse`, `mae`, `mase` for mixture and each backbone |

```
uv run python scripts/train_mixture_online.py [OPTIONS]
```

**Data arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--subsets` | `list[str]` | `["traffic_hourly", "weather", "m5", "monash_m3_monthly"]` | GiftEvalPretrain subsets |
| `--context_length` | `int` | `512` | Context window length |
| `--prediction_length` | `int\|None` | `None` | Horizon; `None` = freq-based default per series |
| `--val_frac` | `float` | `0.2` | Fraction of each series held back for val/test |
| `--max_windows_per_series` | `int` | `10` | Max training windows per series |
| `--max_series_per_subset` | `int\|None` | `None` | Cap series per subset (debug) |
| `--batch_size` | `int` | `32` | DataLoader batch size |
| `--num_workers` | `int` | `0` | DataLoader workers (0 recommended to avoid model-loading conflicts) |

**Model arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model_names` | `list[str]` | `["chronos2", "timesfm25", "moirai2"]` | Frozen backbone models to mix |
| `--device` | `str` | `"cuda"` | Torch device for inference and mixture |

**Mixture arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--adaptive` | `bool` | `False` | Use `MetaWeighter` MLP for per-instance weights instead of global scalars |
| `--context_feature_dim` | `int` | `5` | Input dim for the meta-weighter (when `--adaptive`) |

**Boosting arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--boosting` | `bool` | `False` | Add a patch-transformer residual correction on top of the weighted blend |
| `--boosting_prediction_length` | `int` | `12` | Fixed output horizon for the booster head (must be ≥ `prediction_length` when set) |
| `--boosting_patch_size` | `int` | `16` | Context patch size fed to the transformer |
| `--boosting_d_model` | `int` | `64` | Transformer model dimension |
| `--boosting_n_heads` | `int` | `4` | Number of self-attention heads |
| `--boosting_n_layers` | `int` | `2` | Number of encoder layers |
| `--boosting_dropout` | `float` | `0.1` | Dropout probability |

**Training arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--loss` | `str` | `"mase"` | Training loss: `"mse"`, `"mae"`, or `"mase"` |
| `--lr` | `float` | `0.01` | Adam learning rate |
| `--epochs` | `int` | `50` | Number of training epochs |
| `--checkpoint_dir` | `Path` | `checkpoints/` | Where to save epoch checkpoints |
| `--seed` | `int` | `42` | Random seed |

**Logging arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--log_every_n_steps` | `int` | `10` | Console + TensorBoard step-level logging frequency |
| `--val_every_n_epochs` | `int` | `5` | Run full val evaluation every N epochs |
| `--log_dir` | `Path` | `logs/` | Parent directory for timestamped run directories |
| `--run_name` | `str\|None` | `None` | Optional suffix appended to the run directory name |

**Examples:**

```bash
# Quick smoke test on CPU — 2 epochs, 3 series per subset
uv run python scripts/train_mixture_online.py \
    --subsets weather m5 \
    --device cpu \
    --epochs 2 \
    --batch_size 8 \
    --max_series_per_subset 3

# Full run with static weights, MASE loss
uv run python scripts/train_mixture_online.py \
    --subsets traffic_hourly weather m5 monash_m3_monthly \
    --model_names chronos2 timesfm25 moirai2 \
    --loss mase \
    --lr 0.01 \
    --epochs 50 \
    --val_every_n_epochs 5 \
    --run_name static_mase

# Adaptive weighting (per-instance MetaWeighter MLP)
uv run python scripts/train_mixture_online.py \
    --subsets traffic_hourly weather m5 \
    --adaptive \
    --loss mase \
    --lr 0.005 \
    --epochs 50

# Static mixture + boosting head (patch-transformer residual correction)
uv run python scripts/train_mixture_online.py \
    --subsets traffic_hourly weather m5 \
    --loss mase \
    --boosting \
    --boosting_prediction_length 24 \
    --boosting_d_model 64 \
    --boosting_n_layers 2 \
    --epochs 50 \
    --run_name boosted

# Launch TensorBoard after (or during) training
tensorboard --logdir logs/
```

**Final output example:**

```
============================================================
FINAL TEST RESULTS
============================================================
Model                      Loss        MSE        MAE       MASE
------------------------------------------------------------
mixture              0.8712     0.6431     0.5210     0.8712
chronos2             0.9341     0.7122     0.5689     0.9341
timesfm25            0.9015     0.6834     0.5480     0.9015
moirai2              0.9208     0.7001     0.5601     0.9208
============================================================
CSV:         logs/2026-03-03_14-22-01_static_mase/results.csv
TensorBoard: tensorboard --logdir logs/2026-03-03_14-22-01_static_mase
============================================================
```

---

### `scripts/explore_data.py`

Inspect subset shapes, lengths, frequencies, and NaN rates before committing to a full run.

```bash
# List all 149 available subsets
uv run python scripts/explore_data.py --list_subsets

# Inspect three specific subsets, show 10 series each
uv run python scripts/explore_data.py \
    --subsets weather m5 traffic_hourly \
    --max_series 10
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--subsets` | `list[str]` | `["weather", "m5"]` | Subset names to inspect |
| `--max_series` | `int` | `5` | Series to print per subset |
| `--list_subsets` | `bool` | `False` | Print all available subset names and exit |
| `--context_length` | `int` | `512` | (informational only) |
| `--prediction_length` | `int\|None` | `None` | (informational only) |

---

### `scripts/evaluate.py`

Evaluate a single registered backbone model or a trained `WeightedMixture` checkpoint on the validation split. When evaluating a mixture checkpoint, pre-cached backbone predictions must be available.

Exactly one of `--model_name` or `--checkpoint` must be provided.

```bash
# Evaluate Chronos-2 directly
uv run python scripts/evaluate.py \
    --model_name chronos2 \
    --subsets weather m5 \
    --metric mase

# Evaluate a saved mixture checkpoint (requires cache)
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/mixture_epoch050.pt \
    --subsets weather m5 traffic_hourly \
    --metric mase \
    --output_file results/mixture_eval.json

# Evaluate a checkpoint trained with boosting
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/mixture_epoch050.pt \
    --boosting \
    --boosting_prediction_length 24 \
    --subsets weather m5
```

> **Note:** `--adaptive`, `--boosting`, and all `--boosting_*` flags must match the settings used during training so the checkpoint's `state_dict` maps correctly onto the reconstructed model.

---

### `scripts/cache_predictions.py` (optional)

Pre-run frozen backbone models over all configured windows and save predictions to disk. Only needed for the legacy `train_mixture.py` workflow or for `evaluate.py` mixture evaluation.

```bash
# Cache Chronos-2 and TimesFM on two subsets
uv run python scripts/cache_predictions.py \
    --models chronos2 timesfm25 \
    --subsets weather traffic_hourly

# Full cache run
uv run python scripts/cache_predictions.py \
    --models chronos2 timesfm25 moirai2 \
    --subsets traffic_hourly weather m5 monash_m3_monthly \
    --device cuda \
    --output_dir cache/predictions
```

**Cache layout:**

```
cache/predictions/
└── <model_name>/
    └── <subset_name>.pt        # torch.save'd dict
        key:   "{item_id}__{window_start}"
        value: {"point": np.ndarray (V, H), "quantiles": dict | None}
```

---

### `scripts/train_mixture.py` (legacy)

Cache-based training: reads backbone predictions from disk. Requires running `cache_predictions.py` first. For new experiments prefer `train_mixture_online.py`.

```bash
uv run python scripts/train_mixture.py \
    --subsets weather m5 \
    --loss mase \
    --epochs 50
```

---

## Running tests

```bash
uv run pytest
# with coverage
uv run pytest --cov=tsfc --cov-report=term-missing
```

Tests are offline (no HuggingFace downloads required) and cover frequency normalization, NaN filling, normalization round-trips, context feature extraction, `GiftEvalDataset` window building (train/val splits), `WeightedMixture` forward pass (static and adaptive), all loss functions, and all evaluation metrics.

---

## Linting and type-checking

```bash
uv run ruff check . --fix && uv run ruff format .
uv run pyright
```

---

## Key design notes

- **On-the-fly inference** — `train_mixture_online.py` runs frozen backbone models live inside each training step. Context is denormalized before passing to the model, and the raw-scale prediction is re-normalized before being fed to the mixture. No disk cache is needed.
- **Three-split protocol** — train / val / test windows are temporally non-overlapping: the train boundary is at `T_train = max(ctx+pred, (1−val_frac)·T)`; the val target immediately follows; the test target follows the val target. Shorter series that cannot fit all three windows are silently dropped from val/test.
- **Boosting head** — when `--boosting` is enabled, a `BoostingForecaster` (patch transformer, channel-independent) adds a learnable residual on top of the weighted blend. Its output head is zero-initialized, so training always starts from the mixture-only baseline.
- **Lazy model loading** — backbone weights are loaded on first `predict()` call; importing the package does not trigger downloads.
- **Instance normalization** — every context window is normalized with its own robust (median + MAD) statistics before being passed to models, then denormalized before metric computation. TimesFM normalizes internally; do not double-normalize.
- **Frequency handling** — all frequency strings are canonicalized through `pandas.tseries.frequencies.to_offset`. `"30T"` and `"30min"` are treated identically.
- **NaN policy** — context NaNs are filled by linear interpolation. Target windows with > 20 % NaN are skipped entirely.
