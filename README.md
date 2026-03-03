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
uv pip install -e .[chronos]
# installs: chronos-forecasting>=2.1.0
```

#### TimesFM-2.5

```bash
uv pip install -e .[timesfm]
# installs: timesfm from git+https://github.com/google-research/timesfm.git
```

#### Moirai-2

```bash
uv pip install -e .[moirai]
# installs: uni2ts>=1.2.0
```

### 3. Dev tools

```bash
uv pip install -e .[dev]
# installs: pytest, pytest-cov, ruff, pyright
```

---

## Scripts

`train_mixture_online.py` is the **primary training script**. It runs the frozen backbone models live during every training step — no pre-caching required. A full train / val / test split is used, all metrics are logged to TensorBoard and CSV, and individual backbone model baselines are compared automatically at the end.

All scripts use [tyro](https://brentyi.github.io/tyro/) for CLI parsing. Every argument maps 1-to-1 to a field in the script's `@dataclass` config. Pass `--help` to any script to see the full argument list and defaults.

---

### `scripts/train_mixture_online.py` ⭐ main script

Trains the `WeightedMixture` model with **on-the-fly backbone inference** — no pre-cached predictions are needed. The three data splits used are:

| Split | Content |
|---|---|
| **train** | Random windows sampled from the first `(1 − val_frac)` of each series |
| **val** | Single window whose target immediately follows the train boundary |
| **test** | Single window whose target follows the val window |

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

* Main file: `tsfc/models/mixture.py`

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

## Linting and type-checking

```bash
uv run ruff check . --fix && uv run ruff format .
uv run pyright
```

---

By: Abdelhakim Benechehab.
