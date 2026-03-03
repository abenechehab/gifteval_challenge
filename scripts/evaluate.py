"""Evaluate a forecasting model (or mixture) on GiftEvalPretrain subsets.

Usage
-----
    # Evaluate Chronos-2 directly
    uv run python scripts/evaluate.py --model_name chronos2 --subsets weather m5

    # Evaluate a saved mixture checkpoint
    uv run python scripts/evaluate.py \\
        --checkpoint checkpoints/mixture_epoch050.pt \\
        --subsets weather m5
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

from tsfc.data.gifteval import GiftEvalDataset
from tsfc.eval.metrics import evaluate_dataset
from tsfc.models.base import ForecastingModel, ForecastResult
from tsfc.models.mixture import WeightedMixture

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mixture wrapper that loads cached predictions at evaluation time
# ---------------------------------------------------------------------------


class CachedMixtureModel(ForecastingModel):
    """ForecastingModel that wraps a trained WeightedMixture using cached predictions."""

    name = "cached_mixture"

    def __init__(
        self,
        mixture: WeightedMixture,
        model_names: list[str],
        cache_dir: Path,
        dataset_name: str,
    ) -> None:
        self.mixture = mixture
        self.model_names = model_names
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name
        # Cache files are keyed by item_id__window_start; we can't know them here,
        # so this wrapper is used only from evaluate_dataset which has item metadata.
        self._cache: dict[str, dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        for m in self.model_names:
            cache_file = self.cache_dir / m / f"{self.dataset_name}.pt"
            if cache_file.exists():
                self._cache[m] = torch.load(cache_file, map_location="cpu")
            else:
                logger.warning("Cache file missing: %s", cache_file)
                self._cache[m] = {}

    def predict(
        self,
        context: np.ndarray[Any, np.dtype[np.floating[Any]]],
        prediction_length: int,
        freq: str,
        past_covariates: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    ) -> ForecastResult:
        raise NotImplementedError(
            "CachedMixtureModel.predict() is not implemented for single-item evaluation. "
            "Use evaluate_with_cache() instead."
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    """Configuration for the evaluation script."""

    subsets: list[str] = field(default_factory=lambda: ["weather", "m5"])
    """GiftEvalPretrain subsets to evaluate on."""

    model_name: str | None = None
    """Name of a registered single model to evaluate (e.g. ``"chronos2"``).
    Mutually exclusive with ``--checkpoint``."""

    checkpoint: Path | None = None
    """Path to a ``WeightedMixture`` checkpoint (``*.pt`` from training).
    Mutually exclusive with ``--model_name``."""

    model_names_in_checkpoint: list[str] = field(
        default_factory=lambda: ["chronos2", "timesfm25", "moirai2"]
    )
    """Model names used when the checkpoint was trained (needed to reconstruct the mixture)."""

    cache_dir: Path = Path("cache/predictions")
    """Directory containing pre-cached backbone predictions."""

    context_length: int = 512
    prediction_length: int | None = None
    max_series_per_subset: int | None = None
    metric: str = "mase"
    """Metric to evaluate: ``"mase"``, ``"mae"``, or ``"wql"``."""

    device: str = "cuda"
    output_file: Path | None = None
    """If set, write JSON results to this file."""

    adaptive: bool = False
    context_feature_dim: int = 5

    # Boosting head — must match the settings used during training
    boosting: bool = False
    boosting_prediction_length: int = 64
    boosting_patch_size: int = 16
    boosting_d_model: int = 64
    boosting_n_heads: int = 4
    boosting_n_layers: int = 2
    boosting_dropout: float = 0.1


def main(cfg: EvalConfig) -> None:
    """Run evaluation and print / save results."""
    if cfg.model_name is None and cfg.checkpoint is None:
        logger.error("Must specify either --model_name or --checkpoint.")
        raise SystemExit(1)
    if cfg.model_name is not None and cfg.checkpoint is not None:
        logger.error("Specify either --model_name or --checkpoint, not both.")
        raise SystemExit(1)

    all_results: dict[str, Any] = {}

    for subset in cfg.subsets:
        logger.info("Evaluating on subset %r …", subset)

        dataset = GiftEvalDataset(
            subset_names=[subset],
            context_length=cfg.context_length,
            prediction_length=cfg.prediction_length,
            split="val",
            max_series_per_subset=cfg.max_series_per_subset,
        )

        if cfg.model_name is not None:
            from tsfc.models.registry import load_model

            model: ForecastingModel = load_model(cfg.model_name, device=cfg.device)
            results = evaluate_dataset(model, dataset, metric=cfg.metric)
        else:
            assert cfg.checkpoint is not None
            # Load mixture checkpoint
            ckpt = torch.load(cfg.checkpoint, map_location="cpu")
            mixture = WeightedMixture(
                model_names=cfg.model_names_in_checkpoint,
                adaptive=cfg.adaptive,
                context_feature_dim=cfg.context_feature_dim,
                boosting=cfg.boosting,
                context_length=cfg.context_length,
                prediction_length=cfg.boosting_prediction_length,
                boosting_patch_size=cfg.boosting_patch_size,
                boosting_d_model=cfg.boosting_d_model,
                boosting_n_heads=cfg.boosting_n_heads,
                boosting_n_layers=cfg.boosting_n_layers,
                boosting_dropout=cfg.boosting_dropout,
            )
            mixture.load_state_dict(ckpt["model_state"])
            mixture.eval()
            logger.info("Loaded mixture from epoch %d", ckpt.get("epoch", "?"))
            # Evaluate using cached predictions
            results = _evaluate_mixture_with_cache(
                mixture=mixture,
                dataset=dataset,
                model_names=cfg.model_names_in_checkpoint,
                cache_dir=cfg.cache_dir,
                subset=subset,
                metric=cfg.metric,
                device=cfg.device,
                adaptive=cfg.adaptive,
            )

        all_results[subset] = results
        logger.info(
            "  %s MASE=%.4f ± %.4f  (n=%d)",
            subset,
            results["mean"],
            results["std"],
            results["n_items"],
        )

    # Aggregate across subsets
    means = [r["mean"] for r in all_results.values() if not np.isnan(r["mean"])]
    logger.info(
        "=== Overall mean %s: %.4f (across %d subsets) ===",
        cfg.metric.upper(),
        float(np.mean(means)),
        len(means),
    )

    if cfg.output_file is not None:
        cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
        with cfg.output_file.open("w") as fh:
            json.dump(all_results, fh, indent=2)
        logger.info("Results saved to %s", cfg.output_file)


def _evaluate_mixture_with_cache(
    mixture: WeightedMixture,
    dataset: GiftEvalDataset,
    model_names: list[str],
    cache_dir: Path,
    subset: str,
    metric: str,
    device: str,
    adaptive: bool,
) -> dict[str, Any]:
    """Evaluate the mixture using pre-cached backbone predictions."""
    from tsfc.data.transforms import denormalize, extract_context_features
    from tsfc.eval.metrics import mae, mase

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    mixture = mixture.to(dev)

    # Load all cache files for this subset
    cache: dict[str, dict[str, Any]] = {}
    for m in model_names:
        cache_file = cache_dir / m / f"{subset}.pt"
        if cache_file.exists():
            cache[m] = torch.load(cache_file, map_location="cpu")
        else:
            logger.warning("Cache file missing: %s", cache_file)
            cache[m] = {}

    scores: list[float] = []

    for i in range(len(dataset)):
        item = dataset[i]
        item_id: str = item["item_id"]
        window_start: int = item["window_start"]
        ctx_norm = item["context"].numpy()
        loc = item["loc"].numpy()
        scale = item["scale"].numpy()
        tgt_raw = item["target_raw"].numpy()
        mask_np = item["mask"].numpy()
        freq: str = item["freq"]
        item["prediction_length"]

        cache_key = f"{item_id}__{window_start}"

        # Collect per-model cached predictions
        model_preds: dict[str, torch.Tensor] = {}
        skip = False
        for m in model_names:
            if cache_key not in cache.get(m, {}):
                skip = True
                break
            point = cache[m][cache_key]["point"]
            model_preds[m] = torch.from_numpy(np.asarray(point)).float().unsqueeze(0).to(dev)

        if skip:
            continue

        ctx_t = torch.from_numpy(ctx_norm).float().unsqueeze(0).to(dev)  # (1, V, T)

        ctx_feats: torch.Tensor | None = None
        if adaptive:
            ctx_feats = extract_context_features(ctx_t)

        with torch.no_grad():
            forecast_t = mixture(model_preds, ctx_feats, context=ctx_t)  # (1, V, H)

        # Denormalize forecast
        forecast_np = denormalize(
            forecast_t.squeeze(0).cpu().numpy(),
            loc,
            scale,
        )  # (V, H)

        ctx_raw = denormalize(ctx_norm, loc, scale)

        if metric == "mase":
            score = mase(forecast_np, tgt_raw, ctx_raw, freq, mask_np)
        elif metric == "mae":
            score = mae(forecast_np, tgt_raw, mask_np)
        else:
            score = mae(forecast_np, tgt_raw, mask_np)

        scores.append(score)

    arr = np.array(scores)
    return {
        "mean": float(arr.mean()) if len(arr) > 0 else float("nan"),
        "std": float(arr.std()) if len(arr) > 0 else float("nan"),
        "n_items": len(scores),
        "per_item": scores,
    }


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
