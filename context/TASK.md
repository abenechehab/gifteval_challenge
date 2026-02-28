# Time Series Forecasting Challenge

## Objective

Train a forecasting model on the [GiftEvalPretrain](https://huggingface.co/datasets/theforecastingcompany/GiftEvalPretrain) dataset. The initial approach is a **learned mixture model** that ensembles existing zero-shot foundation models by optimizing per-model weights via gradient descent. The code architecture should remain **general** — it should not be tightly coupled to the mixture idea so that training models from scratch on this data is also supported later.

---

## Phase 1: Weighted Mixture Ensemble (Current Goal)

### High-level Idea

1. For each training sample (context + ground truth horizon), run inference on all registered zero-shot foundation models.
2. Each model produces a point forecast (and optionally quantile forecasts) over the horizon.
3. A trainable weight vector `w` (one scalar per model, softmax-normalized) defines the final prediction as a convex combination of the individual model forecasts.
4. Minimize a loss (e.g. MAE or MSE, or CRPS for probabilistic) between the weighted mixture forecast and the ground truth using gradient descent through the weights only — the backbone models are **frozen**.
5. Optionally: make weights **instance-adaptive** — i.e. predict weights from context features using a small meta-network.

### Why This Is Interesting

- The weights can be interpreted as model-level expertise. If one model is consistently better on high-frequency data, it should get higher weight.
- Gradient descent through a convex combination is differentiable and cheap since backbones are frozen.
- Instance-adaptive weights (via a lightweight meta-network) can potentially outperform fixed global weights.

### Suggested Weight Parameterization

```python
# Static global weights
log_w = nn.Parameter(torch.zeros(n_models))  # shape: (n_models,)
w = torch.softmax(log_w, dim=0)

# Instance-adaptive: predict weights from context statistics
class MetaWeighter(nn.Module):
    def forward(self, context_features):  # (B, n_features)
        return torch.softmax(self.mlp(context_features), dim=-1)  # (B, n_models)
```

---

## Phase 2: Train from Scratch (Future Extension)

The data pipeline and utilities built in Phase 1 should support training a custom forecasting model from scratch. Candidate architectures to explore:

- **PatchTST / TimesNet / iTransformer**: Standard transformer-based univariate models.
- **Patch-based encoder**: Similar to Moirai/TimesFM — divide context into non-overlapping patches, encode with a transformer, decode future patches.
- **Simple NBEATS-style**: Doubly-residual stacking with trend/seasonality basis expansion. Very strong baseline.

The data module should output standard `(context, target, freq, metadata)` tuples that work for any of these.

---

## Evaluation Metrics

Use the same metrics as GIFT-Eval leaderboard:
- **MASE** (Mean Absolute Scaled Error) — primary metric; scale-free.
- **CRPS** (Continuous Ranked Probability Score) — for probabilistic forecasts; use when quantile outputs are available.
- **WQL** (Weighted Quantile Loss) — alternative probabilistic metric.

Compute metrics per dataset, then aggregate across datasets with equal weight (GIFT-Eval style).

---

## Repository Layout

```
tsfc/                        # main package
├── data/
│   ├── __init__.py
│   ├── gifteval.py          # GiftEvalPretrain loading & formatting
│   ├── transforms.py        # normalization, windowing, freq encoding
│   ├── datamodule.py        # PyTorch Lightning DataModule
│   └── utils.py             # freq → period mapping, missing-value handling
├── models/
│   ├── __init__.py
│   ├── registry.py          # model registry (name → loader/predictor)
│   ├── chronos2.py          # Chronos-2 wrapper
│   ├── timesfm.py           # TimesFM wrapper
│   ├── moirai.py            # Moirai-2 wrapper
│   ├── mixture.py           # WeightedMixture + MetaWeighter
│   └── base.py              # abstract ForecastingModel interface
├── train/
│   ├── __init__.py
│   ├── trainer.py           # training loop (PyTorch Lightning or plain)
│   └── loss.py              # MAE, MASE, CRPS losses
├── eval/
│   ├── __init__.py
│   └── metrics.py           # MASE, CRPS, WQL computation
├── configs/
│   └── mixture.yaml         # hydra/simple config for mixture training
├── scripts/
│   ├── explore_data.py      # quick data exploration script
│   ├── cache_predictions.py # pre-cache frozen model outputs to disk
│   └── train_mixture.py     # main training entry point
├── tests/
│   └── test_data.py
├── MODELS.md                # detailed model API reference (this repo)
├── DATA.md                  # data format and loading instructions
└── README.md
```

---

## Priority Order

1. **Data utilities** (`tsfc/data/`) — most important. Must load GiftEvalPretrain efficiently, handle missing values, build windowed train/val splits.
2. **Model wrappers** (`tsfc/models/`) — thin adapters around Chronos-2, TimesFM, Moirai so they share a unified interface.
3. **Caching** (`scripts/cache_predictions.py`) — since backbone models are frozen, cache their predictions once to disk (HDF5 or `.pt` tensors) to decouple training from slow inference.
4. **Mixture model + training loop** (`tsfc/models/mixture.py`, `tsfc/train/`).
5. **Evaluation** (`tsfc/eval/metrics.py`) and comparison vs individual baselines.
