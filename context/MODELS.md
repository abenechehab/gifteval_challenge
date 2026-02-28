# MODELS.md — Foundation Model API Reference

This document describes how to use each zero-shot foundation model and how to wrap them in the unified `ForecastingModel` interface for the mixture ensemble.

---

## Unified Interface

All model wrappers must implement the following abstract base class:

```python
# tsfc/models/base.py
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

@dataclass
class ForecastResult:
    """
    Standardized forecast output.
    
    point_forecast: np.ndarray of shape (n_variates, horizon)
        The median or mean point forecast.
    quantile_forecasts: dict[float, np.ndarray] | None
        Optional quantile forecasts, e.g. {0.1: array(...), 0.5: ..., 0.9: ...}
        Each value has shape (n_variates, horizon).
    """
    point_forecast: np.ndarray
    quantile_forecasts: dict | None = None


class ForecastingModel(ABC):
    """Abstract base class for all forecasting model wrappers."""
    
    name: str  # model identifier, e.g. "chronos2", "timesfm25", "moirai2"

    @abstractmethod
    def predict(
        self,
        context: np.ndarray,          # (n_variates, context_length) — NaN-free
        prediction_length: int,
        freq: str,                    # pandas freq string
        past_covariates: np.ndarray | None = None,  # (n_cov, context_length)
    ) -> ForecastResult:
        """
        Run zero-shot inference and return a ForecastResult.
        All returned values are in the original (unnormalized) scale.
        """
        ...

    def batch_predict(
        self,
        contexts: list[np.ndarray],
        prediction_lengths: list[int],
        freqs: list[str],
    ) -> list[ForecastResult]:
        """Default implementation: loop over items. Override for efficiency."""
        return [
            self.predict(ctx, pl, fr)
            for ctx, pl, fr in zip(contexts, prediction_lengths, freqs)
        ]
```

---

## Model 1: Chronos-2

**Paper**: Chronos-2: From Univariate to Universal Forecasting (arXiv 2510.15821)  
**Repo**: https://github.com/amazon-science/chronos-forecasting  
**HuggingFace**: `amazon/chronos-2`  
**Install**: `pip install chronos-forecasting>=2.1.0`

### Capabilities

| Feature | Supported |
|---------|-----------|
| Univariate | ✅ |
| Multivariate | ✅ (via group attention) |
| Past covariates | ✅ |
| Known future covariates | ✅ |
| Probabilistic output | ✅ (quantiles) |
| Zero-shot | ✅ |

### Architecture

Chronos-2 is a T5-based encoder-decoder with a novel **group attention** mechanism that allows it to jointly model related time series (targets + covariates). Input time series are scaled, tokenized into discrete bins, and processed by the transformer. The decoder outputs quantile estimates directly.

### Minimal Inference Example

```python
import pandas as pd
from chronos import Chronos2Pipeline

# Load the model (do this once and cache the pipeline)
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda",      # or "cpu"
)

# --- Univariate forecast ---
context_df = pd.DataFrame({
    "id":     ["series_0"] * 100,
    "ds":     pd.date_range("2020-01-01", periods=100, freq="H"),
    "target": [float(i) for i in range(100)],
})

pred_df = pipeline.predict_df(
    context_df,
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="ds",
    target_columns=["target"],
)
# pred_df columns: id, ds (future timestamps), mean, 0.1, 0.5, 0.9
```

### With Covariates

```python
# context_df must include covariate columns
context_df = pd.DataFrame({
    "id": [...],
    "ds": [...],
    "target": [...],
    "temperature": [...],  # past covariate
})

# future_df: only covariate columns, for the forecast horizon
future_df = pd.DataFrame({
    "id": [...],
    "ds": [...],           # future timestamps
    "temperature": [...],  # known future values
})

pred_df = pipeline.predict_df(
    context_df,
    future_df=future_df,
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="ds",
    target_columns=["target"],
    past_feat_dynamic_real_columns=["temperature"],  # past-only
    # known_feat_dynamic_real_columns=["temperature"],  # or known future
)
```

### Wrapper Skeleton

```python
# tsfc/models/chronos2.py
import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline
from .base import ForecastingModel, ForecastResult

class Chronos2Model(ForecastingModel):
    name = "chronos2"

    def __init__(self, model_id="amazon/chronos-2", device="cuda", dtype="bfloat16"):
        self.pipeline = Chronos2Pipeline.from_pretrained(
            model_id, device_map=device, torch_dtype=dtype
        )

    def predict(self, context, prediction_length, freq, past_covariates=None):
        n_variates, T = context.shape
        
        # Build context DataFrame
        dates = pd.date_range("2000-01-01", periods=T, freq=freq)
        rows = []
        for v in range(n_variates):
            rows += [{"id": f"v{v}", "ds": d, "target": context[v, t]}
                     for t, d in enumerate(dates)]
        context_df = pd.DataFrame(rows)
        
        pred_df = self.pipeline.predict_df(
            context_df,
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="ds",
            target_columns=["target"],
        )
        
        # Extract results per variate
        point = np.stack([
            pred_df[pred_df["id"] == f"v{v}"]["0.5"].values
            for v in range(n_variates)
        ])  # (n_variates, prediction_length)
        
        quantiles = {
            q: np.stack([
                pred_df[pred_df["id"] == f"v{v}"][str(q)].values
                for v in range(n_variates)
            ])
            for q in [0.1, 0.5, 0.9]
        }
        
        return ForecastResult(point_forecast=point, quantile_forecasts=quantiles)
```

### Notes

- **Batch size**: `Chronos2Pipeline` handles batching internally. For large datasets, pass many series at once as a DataFrame.
- **Model size**: `amazon/chronos-2` is the base model (≈300M params). A small 28M-param variant also exists.
- **Memory**: ~4GB VRAM for base model with bfloat16.
- **Speed**: >300 forecasts/second on A10G GPU.

---

## Model 2: TimesFM

**Paper**: A Decoder-Only Foundation Model for Time-Series Forecasting (ICML 2024)  
**Repo**: https://github.com/google-research/timesfm  
**HuggingFace**: `google/timesfm-2.5-200m-pytorch`  
**Install**: `pip install timesfm` (or from source for v2.5)

### Capabilities

| Feature | Supported |
|---------|-----------|
| Univariate | ✅ |
| Multivariate | ❌ (univariate only; run separately per variate) |
| Past covariates | ⚠️ (via XReg external regressors; optional extra install) |
| Known future covariates | ⚠️ (via XReg) |
| Probabilistic output | ✅ (10 quantile heads, uncalibrated) |
| Zero-shot | ✅ |

### Checkpoints

| Checkpoint | Context | Params | Notes |
|------------|---------|--------|-------|
| `google/timesfm-1.0-200m` (JAX) | 512 | 200M | Original |
| `google/timesfm-2.0-500m-pytorch` | 2048 | 500M | +25% accuracy |
| `google/timesfm-2.5-200m-pytorch` | 1024 | 200M | Latest, PyTorch-native |

**Recommended**: `timesfm-2.5-200m-pytorch` — PyTorch, fast, latest API.

### Minimal Inference Example (v2.5 API)

```python
import torch
import numpy as np
import timesfm

torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

# Batch inference: list of 1D arrays (variable length OK)
inputs = [
    np.linspace(0, 1, 100),        # series 1
    np.sin(np.linspace(0, 20, 67)), # series 2
]

point_forecast, quantile_forecast = model.forecast(
    horizon=24,
    inputs=inputs,
)
# point_forecast.shape: (2, 24)
# quantile_forecast.shape: (2, 24, 10)  # 10th to 90th quantile, plus mean
```

### Older API (v2.0, for reference)

```python
import timesfm

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="gpu",
        per_core_batch_size=32,
        horizon_len=128,
        num_layers=50,
        context_len=2048,
        use_positional_embedding=False,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
    ),
)

# Frequency token: 0=high-freq, 1=medium, 2=low-freq
freq_tokens = [0]  # per series
point, quantiles = tfm.forecast(
    inputs=[series_array],
    freq=freq_tokens,
)
```

### Wrapper Skeleton

```python
# tsfc/models/timesfm.py
import numpy as np
import timesfm
import torch
from .base import ForecastingModel, ForecastResult

FREQ_TO_TOKEN = {"T": 0, "30T": 0, "H": 0, "D": 0, "W": 1, "M": 2, "Q": 2, "A": 2}

class TimesFMModel(ForecastingModel):
    name = "timesfm25"

    def __init__(self):
        torch.set_float32_matmul_precision("high")
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
        self.model.compile(timesfm.ForecastConfig(
            max_context=1024, max_horizon=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            fix_quantile_crossing=True,
        ))

    def predict(self, context, prediction_length, freq, past_covariates=None):
        n_variates = context.shape[0]
        inputs = [context[v] for v in range(n_variates)]  # list of 1D arrays
        
        point, q_forecasts = self.model.forecast(
            horizon=prediction_length,
            inputs=inputs,
        )
        # point: (n_variates, prediction_length)
        # q_forecasts: (n_variates, prediction_length, 10)
        
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        quantiles = {q: q_forecasts[:, :, i] for i, q in enumerate(quantile_levels)}
        
        return ForecastResult(
            point_forecast=point,
            quantile_forecasts=quantiles
        )
```

### Notes

- TimesFM is **univariate-only**; run each variate independently.
- `normalize_inputs=True` means you do NOT need to pre-normalize (the model handles it).
- For GIFT-Eval, TimesFM-2.0 is already on the leaderboard — use 2.5 for best results.
- Context is **truncated** to `max_context` if longer.

---

## Model 3: Moirai-2 (Salesforce)

**Paper**: Unified Training of Universal Time Series Forecasting Transformers (ICML 2024)  
**Repo**: https://github.com/SalesforceAIResearch/uni2ts  
**HuggingFace**: `Salesforce/moirai-2.0-R-{small,base,large}`  
**Install**: `pip install git+https://github.com/SalesforceAIResearch/uni2ts.git`  
(Or: `pip install uni2ts`)

### Capabilities

| Feature | Supported |
|---------|-----------|
| Univariate | ✅ |
| Multivariate | ✅ |
| Past covariates | ✅ (`past_feat_dynamic_real`) |
| Known future covariates | ❌ |
| Probabilistic output | ✅ (mixture of distributions, sample-based) |
| Zero-shot | ✅ |
| GIFT-Eval rank | #1 MASE (non-leaking models) |

### Model Sizes

| Size | Params | Notes |
|------|--------|-------|
| small | ~14M | Fast, good quality |
| base | ~91M | Better accuracy |
| large | ~311M | Best accuracy |

### Minimal Inference Example

```python
import torch
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

SIZE = "small"  # or "base", "large"
PDT = 24        # prediction length
CTX = 500       # context length
PSZ = "auto"    # patch size: auto, 8, 16, 32, 64, 128
BSZ = 32        # batch size

# Load pretrained module
model = Moirai2Forecast(
    module=Moirai2Module.from_pretrained(f"Salesforce/moirai-2.0-R-{SIZE}"),
    prediction_length=PDT,
    context_length=CTX,
    target_dim=1,                       # 1 for univariate
    feat_dynamic_real_dim=0,            # no known future covariates
    past_feat_dynamic_real_dim=0,       # no past covariates
    patch_size=PSZ,
    num_samples=100,                    # probabilistic samples
)
model = model.to("cuda")

# Build GluonTS dataset from pandas DataFrame
# df must have a DatetimeIndex and one column per variate
df = pd.DataFrame({"target": range(600)}, index=pd.date_range("2020-01-01", periods=600, freq="H"))
ds = PandasDataset({"series_0": df})

# Split into context + test horizon
_, test_template = split(ds, offset=-PDT)
test_data = test_template.generate_instances(prediction_length=PDT, windows=1)

# Run inference
predictor = model.create_predictor(batch_size=BSZ)
forecasts = list(predictor.predict(test_data.input))
# Each forecast: gluonts.model.forecast.SampleForecast with .samples shape (100, PDT)
```

### With Covariates

```python
# target_dim = number of target variates
# past_feat_dynamic_real_dim = number of past covariate channels
model = Moirai2Forecast(
    module=Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"),
    prediction_length=24,
    context_length=500,
    target_dim=1,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=3,  # 3 past covariate channels
    patch_size="auto",
    num_samples=100,
)

# The GluonTS dataset must include "past_feat_dynamic_real" key
# shaped (n_past_covariates, T_context)
```

### Wrapper Skeleton

```python
# tsfc/models/moirai.py
import numpy as np
import torch
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
from .base import ForecastingModel, ForecastResult

class MoiraiModel(ForecastingModel):
    name = "moirai2"

    def __init__(self, size="small", context_length=500, num_samples=100, device="cuda"):
        self.context_length = context_length
        self.num_samples = num_samples
        self.size = size
        self.device = device
        self._module = Moirai2Module.from_pretrained(f"Salesforce/moirai-2.0-R-{size}")

    def predict(self, context, prediction_length, freq, past_covariates=None):
        n_variates, T = context.shape
        past_cov_dim = past_covariates.shape[0] if past_covariates is not None else 0

        model = Moirai2Forecast(
            module=self._module,
            prediction_length=prediction_length,
            context_length=min(T, self.context_length),
            target_dim=n_variates,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=past_cov_dim,
            patch_size="auto",
            num_samples=self.num_samples,
        ).to(self.device)

        # Build GluonTS dataset entry
        date_idx = pd.date_range("2000-01-01", periods=T + prediction_length, freq=freq)
        
        # For prediction, we only have context; build dataset with context + dummy future
        # Moirai uses the last `context_length` steps
        entry = {
            "start": date_idx[0],
            "target": context[0] if n_variates == 1 else context,
            "item_id": "series",
        }
        if past_covariates is not None:
            entry["past_feat_dynamic_real"] = past_covariates

        # Use GluonTS dataset + split
        ds = [entry]
        predictor = model.create_predictor(batch_size=1)
        
        # Create test instance: we provide full context, ask for prediction_length future
        from gluonts.dataset.common import ListDataset
        test_ds = ListDataset([entry], freq=freq)
        forecasts = list(predictor.predict(test_ds))
        
        samples = forecasts[0].samples  # (num_samples, prediction_length)
        point = np.median(samples, axis=0)[np.newaxis]  # (1, prediction_length)
        
        quantiles = {}
        for q in [0.1, 0.5, 0.9]:
            quantiles[q] = np.quantile(samples, q, axis=0)[np.newaxis]
        
        return ForecastResult(point_forecast=point, quantile_forecasts=quantiles)
```

### Notes

- Moirai requires **GluonTS** as its inference backbone — `pip install gluonts`.
- The `Moirai2Forecast` object is created per-call with different `prediction_length` and `context_length`; the underlying `Moirai2Module` weights are shared and should be loaded once.
- `num_samples=100` is standard for CRPS evaluation; reduce to 20 for speed during dev.
- For batch inference efficiency, group series of the same frequency together and use a larger `batch_size`.

---

## Model Registry

```python
# tsfc/models/registry.py
from typing import Callable
from .base import ForecastingModel

_REGISTRY: dict[str, Callable[[], ForecastingModel]] = {}

def register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator

def list_models() -> list[str]:
    return list(_REGISTRY.keys())

def load_model(name: str, **kwargs) -> ForecastingModel:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list_models()}")
    return _REGISTRY[name](**kwargs)

# Usage:
# @register("chronos2")
# class Chronos2Model(ForecastingModel): ...
```

---

## Mixture Model

```python
# tsfc/models/mixture.py
import torch
import torch.nn as nn
import numpy as np
from .base import ForecastResult

class WeightedMixture(nn.Module):
    """
    Learnable convex combination of frozen model predictions.
    
    Args:
        model_names: list of model identifiers (must match cached prediction keys)
        adaptive: if True, use a meta-network to predict instance-wise weights
        context_feature_dim: input dim for meta-network (if adaptive)
    """
    def __init__(self, model_names: list[str], adaptive: bool = False, context_feature_dim: int = 16):
        super().__init__()
        self.model_names = model_names
        self.n_models = len(model_names)
        self.adaptive = adaptive
        
        if adaptive:
            # Small MLP: context features → model weights
            self.meta_net = nn.Sequential(
                nn.Linear(context_feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.n_models),
            )
        else:
            # Global learnable log-weights
            self.log_weights = nn.Parameter(torch.zeros(self.n_models))

    def get_weights(self, context_features=None):
        if self.adaptive and context_features is not None:
            return torch.softmax(self.meta_net(context_features), dim=-1)
        return torch.softmax(self.log_weights, dim=0)

    def forward(self, model_predictions: dict[str, torch.Tensor], context_features=None):
        """
        Args:
            model_predictions: dict mapping model_name → Tensor (B, n_variates, horizon)
            context_features: Tensor (B, context_feature_dim) for adaptive weighting
        
        Returns:
            mixture_forecast: Tensor (B, n_variates, horizon)
        """
        preds = torch.stack(
            [model_predictions[name] for name in self.model_names], dim=1
        )  # (B, n_models, n_variates, horizon)
        
        w = self.get_weights(context_features)  # (n_models,) or (B, n_models)
        
        if self.adaptive:
            # w: (B, n_models) → (B, n_models, 1, 1)
            w = w.unsqueeze(-1).unsqueeze(-1)
        else:
            # w: (n_models,) → (1, n_models, 1, 1)
            w = w.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        return (preds * w).sum(dim=1)  # (B, n_variates, horizon)


def extract_context_features(context: torch.Tensor) -> torch.Tensor:
    """
    Extract simple statistical features from context for the meta-network.
    
    Args:
        context: (B, n_variates, context_length)
    
    Returns:
        features: (B, n_features)
    """
    x = context[:, 0, :]  # use first variate for univariate case: (B, T)
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True) + 1e-8
    x_norm = (x - mean) / std
    
    features = torch.cat([
        mean,                                      # trend level
        std,                                       # volatility
        x_norm[:, -1:] - x_norm[:, 0:1],          # overall trend direction
        x_norm[:, -8:].mean(dim=-1, keepdim=True), # recent mean
        x_norm.std(dim=-1, keepdim=True),          # normalized std
    ], dim=-1)  # (B, 5) — extend as needed
    
    return features
```

---

## Caching Frozen Model Predictions

Since backbone models are **frozen**, pre-compute and cache their outputs before training the mixture. This avoids running inference repeatedly during training.

```python
# scripts/cache_predictions.py
"""
Pre-cache predictions from all frozen models.

Usage:
    python scripts/cache_predictions.py \
        --models chronos2 timesfm25 moirai2 \
        --subsets traffic_hourly m5 weather \
        --context_length 512 \
        --output_dir cache/
"""
import argparse
import torch
from pathlib import Path
from tsfc.data.gifteval import GiftEvalDataset
from tsfc.models.registry import load_model

def cache_model_predictions(model_name, subsets, context_length, prediction_length, output_dir):
    model = load_model(model_name)
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for subset in subsets:
        cache_file = output_dir / f"{subset}.pt"
        if cache_file.exists():
            print(f"Skip {model_name}/{subset} (already cached)")
            continue
        
        dataset = GiftEvalDataset(
            subset_names=[subset],
            context_length=context_length,
            prediction_length=prediction_length,
        )
        
        predictions = {}
        for i, item in enumerate(dataset):
            ctx = item["context"].numpy()  # (n_variates, context_length)
            freq = item["freq"]
            key = f"{item['item_id']}__{i}"
            
            result = model.predict(ctx, prediction_length, freq)
            predictions[key] = {
                "point": result.point_forecast,
                "quantiles": result.quantile_forecasts,
            }
        
        torch.save(predictions, cache_file)
        print(f"Saved {len(predictions)} predictions → {cache_file}")
```

---

## Environment Setup

```bash
# Create environment
conda create -n tsfc python=3.11 -y
conda activate tsfc

# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas datasets huggingface_hub h5py tqdm

# Model-specific
pip install chronos-forecasting>=2.1.0           # Chronos-2
pip install timesfm                               # TimesFM 2.5
pip install git+https://github.com/SalesforceAIResearch/uni2ts.git  # Moirai-2
pip install gluonts                               # required by Moirai

# Optional: training utilities
pip install lightning wandb hydra-core
```

---

## Quick Sanity Check

```python
# scripts/smoke_test.py — verify all models load and produce outputs
import numpy as np
from tsfc.models.registry import load_model

context = np.random.randn(1, 128).astype(np.float32)  # (1 variate, 128 steps)
freq = "H"
horizon = 24

for model_name in ["chronos2", "timesfm25", "moirai2"]:
    model = load_model(model_name)
    result = model.predict(context, horizon, freq)
    assert result.point_forecast.shape == (1, horizon), f"Bad shape for {model_name}"
    print(f"✓ {model_name}: point_forecast shape = {result.point_forecast.shape}")
```
