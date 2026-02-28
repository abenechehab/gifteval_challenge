# CLAUDE.md — Instructions for Claude Code

This file is the primary entrypoint for Claude Code working on this project.
Read it fully before writing any code.

---

## Project Summary

We are building a **time series forecasting system** on the GiftEvalPretrain dataset.

**Phase 1 goal**: A learned mixture of existing zero-shot foundation models (Chronos-2, TimesFM, Moirai-2), where scalar (or instance-adaptive) weights are optimized by gradient descent on training windows sampled from GiftEvalPretrain.

**Design principle**: Keep code general. The data pipeline and model interfaces should support training custom architectures from scratch in a later phase, not just the mixture approach.

**Detailed specs in**:
- `TASK.md` — full task description, architecture, training strategy
- `DATA.md` — dataset schema, loading patterns, windowing, normalization
- `MODELS.md` — each model's API, wrapper skeletons, caching strategy

---

## Immediate Priorities

Work in this order:

### 1. Data utilities (`tsfc/data/`)

This is the most important part. Build:

- `tsfc/data/gifteval.py` — `GiftEvalDataset` (PyTorch Dataset) + `GiftEvalDataModule` (Lightning DataModule)
  - Load GiftEvalPretrain subsets via HuggingFace `datasets` library
  - Support streaming for large subsets
  - Window sampling: random windows of `(context_length + prediction_length)` per series
  - Train/val split: use last `val_frac=0.2` of each series by time for validation
  - Handle NaN via linear interpolation; skip windows with >20% NaN in target
  - Return: `dict` with `context` (Tensor), `target` (Tensor), `mask`, `freq`, `dataset_name`, `item_id`, `loc`, `scale`

- `tsfc/data/transforms.py` — normalization, freq encoding, feature extraction
  - Instance normalization (robust: median + MAD scale)
  - `freq_to_period(freq: str) -> int` — for MASE denominator
  - `freq_to_timesfm_token(freq: str) -> int`
  - `extract_context_features(context: Tensor) -> Tensor` — stats for meta-weighter

- `tsfc/data/utils.py` — misc helpers (NaN filling, freq string normalization)

### 2. Model wrappers (`tsfc/models/`)

Build thin wrappers matching the `ForecastingModel` abstract interface in `MODELS.md`:
- `tsfc/models/base.py` — `ForecastResult` dataclass + `ForecastingModel` ABC
- `tsfc/models/chronos2.py` — `Chronos2Model` using `chronos.Chronos2Pipeline`
- `tsfc/models/timesfm.py` — `TimesFMModel` using `timesfm.TimesFM_2p5_200M_torch`
- `tsfc/models/moirai.py` — `MoiraiModel` using `uni2ts.model.moirai2.Moirai2Forecast`
- `tsfc/models/registry.py` — `register`, `load_model`, `list_models`

Important: models must be loadable **lazily** (on first use) so that the data pipeline can run without loading all model weights.

### 3. Prediction caching (`scripts/cache_predictions.py`)

Backbone models are **frozen**. Cache their predictions once before mixture training.  
See caching strategy in `MODELS.md` → "Caching Frozen Model Predictions".

Output format per item: `{"point": np.ndarray (n_variates, horizon), "quantiles": dict}`

### 4. Mixture model + training (`tsfc/models/mixture.py`, `tsfc/train/`)

- `WeightedMixture` module — see skeleton in `MODELS.md`
- `train/loss.py` — MAE, MASE, CRPS losses
- `train/trainer.py` — training loop that:
  1. Loads cached predictions for each batch item
  2. Stacks them into `(B, n_models, n_variates, horizon)` tensor
  3. Applies `WeightedMixture.forward()` 
  4. Computes loss vs ground truth
  5. Backprops through weights only

### 5. Evaluation (`tsfc/eval/metrics.py`)

Implement:
- `mase(forecast, target, context, freq)` — primary metric
- `crps_from_samples(samples, target)` — for probabilistic outputs
- `wql(quantile_forecasts, target, quantile_levels)` — quantile loss
- `evaluate_dataset(model, dataset, metric)` — aggregate over a full subset

---

## Coding Standards

- **Python 3.11+**, type hints everywhere
- **No global state** — pass configs/devices explicitly
- **Dataclasses** for configs: `@dataclass class TrainingConfig: ...`
- Lazy model loading — wrap in `@functools.cached_property` or load in `__init__` only when explicitly called
- All file I/O through `pathlib.Path`
- Use `logging` (not `print`) for progress/debugging
- Write docstrings for all public functions and classes
- Unit tests in `tests/` using `pytest`

---

## Key Design Decisions

### Instance Normalization

Always normalize context before passing to models, then denormalize predictions:

```python
# In data pipeline:
loc, scale = robust_stats(context)   # median, IQR
context_norm = (context - loc) / scale

# After prediction:
prediction_denorm = prediction * scale + loc
```

Exception: TimesFM has `normalize_inputs=True` built in — do NOT double-normalize for it.

### Frequency Handling

GiftEvalPretrain uses pandas freq strings. Normalize them before any lookups:
```python
import pandas as pd
# Canonical form:
freq_str = pd.tseries.frequencies.to_offset(raw_freq_str).freqstr
```

Moirai needs a GluonTS-compatible frequency; most pandas freq strings work directly.
Chronos-2 needs the freq for timestamp generation.
TimesFM needs a 0/1/2 token.

### Handling Multivariate Series

GiftEvalPretrain's `target` column is `list[list[float]]` — the outer dimension is variates.

- For Chronos-2: run all variates jointly (it supports multivariate natively).
- For TimesFM: run each variate separately, then stack.
- For Moirai-2: set `target_dim=n_variates` and pass the full target matrix.

For the mixture, aggregate across variates before computing weights, or compute variate-level weights (simpler: use variate-averaged metrics).

### Caching Key Convention

```
{dataset_name}__{item_id}__{window_start_idx}
```

E.g.: `"m5__FOODS_3_001_CA_1__512"`

---

## Config File

```yaml
# configs/mixture.yaml
data:
  subsets:           # list of GiftEvalPretrain subset names
    - traffic_hourly
    - weather
    - m5
    - monash_m3_monthly
  context_length: 512
  prediction_length: null   # null = use freq-based default
  val_frac: 0.2
  max_windows_per_series: 10
  batch_size: 64

models:
  names: [chronos2, timesfm25, moirai2]
  cache_dir: cache/predictions/

mixture:
  adaptive: false   # if true, use MetaWeighter MLP
  context_feature_dim: 5

training:
  loss: mase        # or "mae", "crps"
  lr: 0.01
  epochs: 50
  device: cuda

eval:
  metrics: [mase, crps]
```

---

## Common Pitfalls

1. **NaN propagation**: Always fill NaN before passing to models. Log how many windows were skipped.
2. **Frequency mismatch**: `"30T"` and `"30min"` are the same — normalize with `pd.tseries.frequencies.to_offset`.
3. **Memory**: `buildings_900k` has 1.8M rows and 1.76 TB total — always use streaming for large subsets.
4. **Moirai's GluonTS dependency**: Moirai requires `gluonts.dataset.common.ListDataset`. Don't confuse it with HuggingFace datasets.
5. **Chronos-2 DataFrame format**: The `predict_df` API requires columns `["id", "ds", "target"]` at minimum, with `id` being a string identifier per series.
6. **TimesFM context truncation**: If `context_length > max_context`, the model silently truncates. Clip yourself to be safe.
7. **Denormalization before metrics**: MASE is computed on original scale. Always denormalize before metric computation.
