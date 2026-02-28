# DATA.md — GiftEvalPretrain Loading & Format Guide

## Dataset Overview

**Source**: `theforecastingcompany/GiftEvalPretrain` on Hugging Face  
**Size**: ~3.49M rows (train split), ~1.76 TB total  
**Format**: Parquet, 149 subsets (one per dataset source)  
**Purpose**: Large-scale pre-training collection for universal time series forecasting

---

## Schema

Each row in the dataset represents a **single time series** with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `dataset_name` | string | Source dataset identifier (e.g. `"BEIJING_SUBWAY_30MIN"`) |
| `item_id` | string | Unique series identifier within the dataset |
| `start` | timestamp | Start timestamp of the series (UTC) |
| `freq` | string | Pandas-compatible frequency string (e.g. `"30T"`, `"H"`, `"D"`, `"W"`, `"M"`) |
| `target` | list[list[float]] | **Multivariate target**: outer list = variates, inner list = time steps. Univariate series → length-1 outer list. |
| `past_feat_dynamic_real` | list[list[float]] | Past-only covariates (same shape as target). May be empty list. |

### Key Detail: `target` Structure

```python
# Univariate: target[0] is the time series values
target = [[1.0, 2.0, 3.0, 4.0, ...]]

# Multivariate (e.g. PEMS traffic): target[i] = i-th variate
target = [
    [55.0, 123.0, 207.0, ...],  # variate 0
    [26.0,  43.0,  58.0, ...],  # variate 1
]
```

---

## Loading the Data

### Basic — Load a Single Subset

```python
from datasets import load_dataset

# Load one subset (e.g. M5 sales)
ds = load_dataset(
    "theforecastingcompany/GiftEvalPretrain",
    name="m5",
    split="train"
)
df = ds.to_pandas()
```

### Load All Subsets

```python
from datasets import load_dataset, get_dataset_config_names

configs = get_dataset_config_names("theforecastingcompany/GiftEvalPretrain")
# configs is a list of 149 subset names

datasets = {}
for name in configs:
    datasets[name] = load_dataset(
        "theforecastingcompany/GiftEvalPretrain",
        name=name,
        split="train"
    ).to_pandas()
```

### Streaming Mode (Recommended for Large Subsets)

For large subsets like `buildings_900k` (1.8M rows), use streaming:

```python
ds = load_dataset(
    "theforecastingcompany/GiftEvalPretrain",
    name="buildings_900k",
    split="train",
    streaming=True
)
for batch in ds.iter(batch_size=1000):
    # process batch
    ...
```

---

## Data Utilities to Implement

### 1. Frequency Handling

```python
# Frequency string → seasonal period (for MASE denominator)
FREQ_TO_PERIOD = {
    "T": 1440,    # minute → daily
    "30T": 48,    # 30-min → daily  
    "H": 24,      # hourly → daily
    "D": 7,       # daily → weekly
    "W": 52,      # weekly → yearly
    "M": 12,      # monthly → yearly
    "Q": 4,       # quarterly → yearly
    "A": 1,       # annual
    "Y": 1,
}

# Frequency → TimesFM frequency token {0, 1, 2}
FREQ_TO_TIMESFM_TOKEN = {
    "T": 0, "30T": 0, "H": 0,  # high freq
    "D": 0, "W": 1,              # medium
    "M": 2, "Q": 2, "A": 2,    # low freq
}
```

### 2. Window Sampler

For training, sample windows of `(context_length + prediction_length)` from each series:

```python
class WindowSampler:
    """
    Sample (context, target) windows from a time series for training.
    
    Returns:
        context: np.ndarray of shape (n_variates, context_length)
        target:  np.ndarray of shape (n_variates, prediction_length)
    """
    def __init__(self, context_length: int, prediction_length: int, stride: int = 1):
        ...
    
    def sample(self, series: np.ndarray) -> Iterator[tuple]:
        T = series.shape[-1]
        total = self.context_length + self.prediction_length
        starts = range(0, T - total + 1, self.stride)
        for s in starts:
            yield series[..., s:s + self.context_length], series[..., s + self.context_length:s + total]
```

### 3. Normalization

Each series should be normalized before being fed to models. Use **instance normalization** (normalize per window):

```python
def instance_normalize(context: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Normalize context by its mean and std. Return normalized context + stats
    for denormalization of predictions.
    
    Uses robust stats: median + IQR-based scale to handle outliers.
    """
    loc = np.median(context, axis=-1, keepdims=True)
    scale = np.maximum(np.abs(context - loc).mean(axis=-1, keepdims=True), 1e-8)
    return (context - loc) / scale, loc, scale
```

### 4. Missing Value Handling

GiftEvalPretrain contains series with missing values (NaN). Handle as follows:

- **For zero-shot model inference**: fill NaN with linear interpolation, then flag positions to exclude from loss.
- **For target (ground truth)**: if `> 20%` of the prediction window is NaN, skip the window entirely.
- **Mask**: always compute and store a boolean mask of observed positions.

```python
def handle_missing(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        filled: series with NaN replaced by linear interpolation
        mask: boolean array, True where values are observed (not NaN)
    """
    mask = ~np.isnan(series)
    filled = series.copy()
    if not mask.all():
        x = np.arange(len(series))
        filled = np.interp(x, x[mask], series[mask])
    return filled, mask
```

### 5. GluonTS-Compatible Format

Moirai and other models expect data in GluonTS format:

```python
from gluonts.dataset.pandas import PandasDataset

def series_row_to_gluonts(row: dict) -> dict:
    """Convert a GiftEvalPretrain row to GluonTS entry dict."""
    target = np.array(row["target"])  # (n_variates, T)
    if target.shape[0] == 1:
        target = target[0]  # univariate: shape (T,)
    return {
        "start": row["start"],
        "target": target,
        "freq": row["freq"],
        "item_id": row["item_id"],
        "feat_dynamic_real": np.array(row["past_feat_dynamic_real"]) if row["past_feat_dynamic_real"] else None,
    }
```

### 6. PyTorch Dataset

```python
class GiftEvalDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapping GiftEvalPretrain.
    
    Args:
        subset_names: list of dataset subset names to include
        context_length: number of past steps as input
        prediction_length: number of future steps to predict
        max_series_per_subset: cap series per subset (for fast iteration)
        split: "train" or "val" (val = last 20% of each series by time)
    
    Returns per item:
        {
            "context": Tensor (n_variates, context_length),
            "target": Tensor (n_variates, prediction_length),
            "mask": Tensor bool (prediction_length,),
            "freq": str,
            "dataset_name": str,
            "item_id": str,
            "loc": float,   # normalization location
            "scale": float, # normalization scale
        }
    """
```

---

## Recommended Subset Selection for Development

To iterate quickly, start with a **representative subset** of datasets:

```python
FAST_DEV_SUBSETS = [
    # Hourly
    "traffic_hourly",       # 862 series, hourly
    "pedestrian_counts",    # 66 series, hourly
    # Daily
    "nn5_daily_with_missing",  # 111 series, daily
    "weather",              # 3020 series, daily
    # Weekly / Monthly
    "nn5_weekly",           # 111 series, weekly
    "monash_m3_monthly",    # 1428 series, monthly
    "tourism_monthly",      # 366 series, monthly
    # Misc
    "m5",                   # 30500 series (retail), daily
    "fred_md",              # 107 macroeconomic series, monthly
]
```

For full training, include all 149 subsets.

---

## Prediction Length Convention

GiftEvalPretrain does not specify a canonical prediction length per series. Use dataset-specific defaults from GIFT-Eval benchmark, or define globally:

```python
FREQ_TO_DEFAULT_HORIZON = {
    "T":   60,    # 1 hour
    "30T": 48,    # 24 hours
    "H":   24,    # 1 day
    "D":   30,    # 30 days
    "W":   8,     # 8 weeks
    "M":   12,    # 1 year
    "Q":   4,
    "A":   1,
}
```

---

## Data Volume Estimates

| Category | Approx. Series Count | Typical Length |
|----------|---------------------|----------------|
| High-freq (≤1H) | ~500k+ | 1k–100k steps |
| Daily | ~200k | 200–5000 steps |
| Weekly | ~300k | 50–500 steps |
| Monthly | ~50k | 30–200 steps |
| Other | varies | varies |

For the mixture model, you do NOT need all data — a well-sampled subset is sufficient for learning weights.

---

## Caching Strategy

Since frozen model inference is expensive, cache predictions to disk before training the mixture:

```python
# Cache format: HDF5 file per (model_name, subset_name)
# Key: f"{item_id}__{window_start}"
# Value: dict with "point_forecast" and optionally "quantile_forecast"

import h5py

cache_path = Path("cache") / f"{model_name}_{subset_name}.h5"
with h5py.File(cache_path, "w") as f:
    for key, pred in predictions.items():
        f.create_dataset(key, data=pred)
```

Alternatively, use `torch.save` with a dictionary for simplicity in small-scale experiments.
