# TSFC — Time Series Forecasting with Learned Mixtures

A **learned mixture ensemble** of zero-shot foundation models (Chronos-2, TimesFM-2.5, Moirai-2) trained on the [GiftEvalPretrain](https://huggingface.co/datasets/theforecastingcompany/GiftEvalPretrain) dataset. Scalar (or instance-adaptive) per-model weights are optimised by gradient descent on sampled training windows. The data pipeline and model interfaces are intentionally general to support training custom architectures from scratch in a later phase.

---

## Repository layout

```
tsfc/                     # installable package
├── data/
│   ├── gifteval.py       # GiftEvalDataset + GiftEvalDataModule
│   ├── transforms.py     # normalization, freq encoding, feature extraction
│   └── utils.py          # NaN filling, freq string normalization
├── models/
│   ├── base.py           # ForecastResult dataclass + ForecastingModel ABC
│   ├── registry.py       # register / load_model / list_models
│   ├── chronos2.py       # Chronos-2 wrapper
│   ├── timesfm.py        # TimesFM-2.5 wrapper
│   ├── moirai.py         # Moirai-2 wrapper
│   └── mixture.py        # WeightedMixture + MetaWeighter
├── train/
│   ├── loss.py           # MAE / MSE / MASE / CRPS losses
│   └── trainer.py        # mixture training loop + TrainingConfig
└── eval/
    └── metrics.py        # MASE, CRPS, WQL, evaluate_dataset()

scripts/
├── explore_data.py       # inspect GiftEvalPretrain subsets
├── cache_predictions.py  # pre-cache frozen model outputs to disk
├── train_mixture.py      # train the WeightedMixture model
└── evaluate.py           # evaluate a single model or trained mixture

configs/
└── mixture.yaml          # reference config (not auto-loaded; values shown as defaults)

tests/
└── test_data.py          # pytest unit tests
```

---

## Installation

### 1. Core package

The core install brings in `torch`, `numpy`, and `tyro` only — no model weights are downloaded.

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

> **Solver conflicts:** `uni2ts` pins `gluonts~=0.14.3` and `datasets~=2.17.1`, which can clash with other extras. If `uv` cannot resolve, install directly into the venv:
> ```bash
> uv run pip install "uni2ts>=1.2.0"
> ```

### 3. Dev tools (linting, type-checking, tests)

```bash
uv sync --extra dev
# installs: pytest, pytest-cov, ruff, pyright
```

---

## Workflow

The intended execution order is:

```
explore_data.py  →  cache_predictions.py  →  train_mixture.py  →  evaluate.py
```

---

## Scripts

All scripts use [tyro](https://brentyi.github.io/tyro/) for CLI parsing. Every argument maps 1-to-1 to a field in the script's `@dataclass` config. Pass `--help` to any script to see the full argument list and defaults.

---

### `scripts/explore_data.py`

Inspect subset shapes, lengths, frequencies, and NaN rates before committing to a full run.

```
uv run python scripts/explore_data.py [OPTIONS]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--subsets` | `list[str]` | `["weather", "m5"]` | Subset names to inspect |
| `--max_series` | `int` | `5` | Series to print per subset |
| `--list_subsets` | `bool` | `False` | Print all 149 available subset names and exit |
| `--context_length` | `int` | `512` | (informational only) |
| `--prediction_length` | `int\|None` | `None` | (informational only) |
| `--cache_dir` | `Path\|None` | `None` | HuggingFace dataset cache directory |

**Examples:**

```bash
# List all 149 available subsets
uv run python scripts/explore_data.py --list_subsets

# Inspect three specific subsets, show 10 series each
uv run python scripts/explore_data.py \
    --subsets weather m5 traffic_hourly \
    --max_series 10
```

---

### `scripts/cache_predictions.py`

Run frozen backbone models over all configured windows and save predictions to disk. This only needs to be done **once** per (model, subset, context_length) combination.

```
uv run python scripts/cache_predictions.py [OPTIONS]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--models` | `list[str]` | `["chronos2", "timesfm25", "moirai2"]` | Registered model names to cache |
| `--subsets` | `list[str]` | `["traffic_hourly", "weather", "m5", "monash_m3_monthly"]` | Subsets to run |
| `--context_length` | `int` | `512` | Context length passed to each model |
| `--prediction_length` | `int\|None` | `None` | Horizon; `None` = freq-based default per series |
| `--output_dir` | `Path` | `cache/predictions` | Root cache directory |
| `--max_series_per_subset` | `int\|None` | `None` | Cap series per subset (debug) |
| `--max_windows_per_series` | `int` | `10` | Windows sampled per series |
| `--device` | `str` | `"cuda"` | Torch device for inference |
| `--overwrite` | `bool` | `False` | Re-cache even if file already exists |
| `--split` | `str` | `"train"` | Which split to cache (`"train"` or `"val"`) |

**Cache layout on disk:**

```
cache/predictions/
└── <model_name>/
    └── <subset_name>.pt        # torch.save'd dict
        key:   "{item_id}__{window_start}"
        value: {"point": np.ndarray (V, H), "quantiles": dict | None}
```

**Examples:**

```bash
# Cache all three models on two subsets (quick test with 3 series each)
uv run python scripts/cache_predictions.py \
    --models chronos2 timesfm25 \
    --subsets weather traffic_hourly \
    --max_series_per_subset 3

# Full cache run on CPU
uv run python scripts/cache_predictions.py \
    --models chronos2 timesfm25 moirai2 \
    --subsets traffic_hourly weather m5 monash_m3_monthly \
    --device cpu \
    --output_dir cache/predictions

# Re-cache a subset that changed
uv run python scripts/cache_predictions.py \
    --models chronos2 \
    --subsets m5 \
    --overwrite
```

---

### `scripts/train_mixture.py`

Train the `WeightedMixture` model. Backbone predictions must already be cached. Only the mixture weights (3 scalars for the static variant, or the meta-weighter MLP for the adaptive variant) are updated via gradient descent.

```
uv run python scripts/train_mixture.py [OPTIONS]
```

**Data arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--subsets` | `list[str]` | `["traffic_hourly", "weather", "m5", "monash_m3_monthly"]` | Training subsets |
| `--context_length` | `int` | `512` | Context window length |
| `--prediction_length` | `int\|None` | `None` | Horizon; `None` = freq-based default |
| `--val_frac` | `float` | `0.2` | Fraction of each series held out for validation |
| `--max_windows_per_series` | `int` | `10` | Max training windows per series |
| `--batch_size` | `int` | `64` | DataLoader batch size |
| `--num_workers` | `int` | `4` | DataLoader worker processes |

**Model arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model_names` | `list[str]` | `["chronos2", "timesfm25", "moirai2"]` | Models to mix (must match cache) |
| `--cache_dir` | `Path` | `cache/predictions` | Location of cached predictions |
| `--adaptive` | `bool` | `False` | Use MetaWeighter MLP instead of global scalars |
| `--context_feature_dim` | `int` | `5` | Input dim for the meta-weighter (if `--adaptive`) |

**Training arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--loss` | `str` | `"mase"` | Loss function: `"mae"`, `"mase"`, or `"crps"` |
| `--lr` | `float` | `0.01` | Adam learning rate |
| `--epochs` | `int` | `50` | Number of training epochs |
| `--device` | `str` | `"cuda"` | Torch device |
| `--checkpoint_dir` | `Path` | `checkpoints/` | Where to save epoch checkpoints |
| `--log_every_n_steps` | `int` | `10` | Log training loss every N steps |
| `--seed` | `int` | `42` | Random seed |

**Examples:**

```bash
# Minimal run (static weights, MAE loss)
uv run python scripts/train_mixture.py \
    --subsets weather m5 \
    --loss mae \
    --epochs 20

# Adaptive weighting with MASE loss
uv run python scripts/train_mixture.py \
    --subsets traffic_hourly weather m5 monash_m3_monthly \
    --adaptive \
    --loss mase \
    --lr 0.005 \
    --epochs 50

# CPU run for debugging
uv run python scripts/train_mixture.py \
    --subsets weather \
    --device cpu \
    --epochs 2 \
    --batch_size 8
```

Checkpoints are saved to `checkpoints/mixture_epoch<NNN>.pt` after every epoch. Each file contains `{epoch, model_state, optimizer_state, train_loss, val_loss}`.

---

### `scripts/evaluate.py`

Evaluate either a single registered backbone model or a trained `WeightedMixture` checkpoint on the validation split.

```
uv run python scripts/evaluate.py [OPTIONS]
```

Exactly one of `--model_name` or `--checkpoint` must be provided.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--subsets` | `list[str]` | `["weather", "m5"]` | Subsets to evaluate on |
| `--model_name` | `str\|None` | `None` | Backbone to evaluate directly (`"chronos2"`, `"timesfm25"`, `"moirai2"`) |
| `--checkpoint` | `Path\|None` | `None` | Path to a `*.pt` mixture checkpoint |
| `--model_names_in_checkpoint` | `list[str]` | `["chronos2", "timesfm25", "moirai2"]` | Models the checkpoint was trained with |
| `--cache_dir` | `Path` | `cache/predictions` | Pre-cached backbone predictions (needed for mixture eval) |
| `--context_length` | `int` | `512` | Context length for the val dataset |
| `--prediction_length` | `int\|None` | `None` | Horizon; `None` = freq-based default |
| `--max_series_per_subset` | `int\|None` | `None` | Cap series per subset |
| `--metric` | `str` | `"mase"` | Metric: `"mase"`, `"mae"`, or `"wql"` |
| `--device` | `str` | `"cuda"` | Torch device |
| `--output_file` | `Path\|None` | `None` | Save JSON results to this path |
| `--adaptive` | `bool` | `False` | Whether the checkpoint used adaptive weighting |
| `--context_feature_dim` | `int` | `5` | Meta-weighter feature dim (if `--adaptive`) |

**Examples:**

```bash
# Evaluate Chronos-2 directly on two subsets
uv run python scripts/evaluate.py \
    --model_name chronos2 \
    --subsets weather m5 \
    --metric mase

# Evaluate a trained mixture checkpoint
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/mixture_epoch050.pt \
    --subsets weather m5 traffic_hourly \
    --metric mase \
    --output_file results/mixture_eval.json

# Evaluate with WQL metric and save results
uv run python scripts/evaluate.py \
    --model_name timesfm25 \
    --subsets monash_m3_monthly \
    --metric wql \
    --output_file results/timesfm_wql.json
```

---

## Running tests

```bash
uv run pytest
# or with coverage
uv run pytest --cov=tsfc --cov-report=term-missing
```

Tests are offline (no HuggingFace downloads required) and cover:
- Frequency normalization and lookup tables
- NaN filling (1D and 2D)
- Robust normalization / denormalization round-trip
- Context feature extraction
- `GiftEvalDataset` window building (train and val splits) with synthetic data
- `WeightedMixture` forward pass (static and adaptive)
- All loss functions
- All evaluation metrics

---

## Linting and type-checking

```bash
uv run ruff check . --fix && uv run ruff format .
uv run pyright
```

---

## Key design notes

- **Lazy model loading** — backbone weights are loaded on first `predict()` call, so importing the package does not trigger any downloads.
- **Instance normalization** — every context window is normalized with its own robust (median + MAD) statistics before being passed to models, then denormalized before metric computation. TimesFM normalizes internally; do not double-normalize.
- **Caching is required for training** — `train_mixture.py` reads cached predictions from disk. Run `cache_predictions.py` first.
- **Frequency handling** — all frequency strings are canonicalized through `pandas.tseries.frequencies.to_offset` before lookup. `"30T"` and `"30min"` are treated identically.
- **NaN policy** — context NaNs are filled by linear interpolation. Target windows with >20% NaN are skipped entirely.
