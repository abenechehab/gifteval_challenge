"""Pre-cache frozen model predictions to disk.

Usage
-----
    uv run python scripts/cache_predictions.py \\
        --models chronos2 timesfm25 moirai2 \\
        --subsets traffic_hourly m5 weather \\
        --context_length 512 \\
        --output_dir cache/predictions

All model weights remain frozen; only the cache files are written.
Cache layout::

    output_dir/<model_name>/<subset_name>.pt
        key:   "{item_id}__{window_start}"
        value: {"point": np.ndarray (V, H), "quantiles": dict | None}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro

from tsfc.data.gifteval import GiftEvalDataset
from tsfc.data.transforms import denormalize
from tsfc.models.registry import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for the prediction caching script."""

    models: list[str] = field(default_factory=lambda: ["chronos2", "timesfm25", "moirai2"])
    """Model names to cache (must be registered in the model registry)."""

    subsets: list[str] = field(
        default_factory=lambda: [
            "traffic_hourly",
            "weather",
            "m5",
            "monash_m3_monthly",
        ]
    )
    """GiftEvalPretrain subset names to cache."""

    context_length: int = 512
    """Context length fed to each model."""

    prediction_length: int | None = None
    """Prediction horizon; None → use freq-based default per series."""

    output_dir: Path = Path("cache/predictions")
    """Root directory for cached prediction files."""

    max_series_per_subset: int | None = None
    """Cap series per subset (useful for debugging)."""

    max_windows_per_series: int = 10
    """Number of windows sampled per series."""

    device: str = "cuda"
    """Torch device for model inference."""

    overwrite: bool = False
    """If True, overwrite existing cache files."""

    split: str = "train"
    """Which dataset split to cache (``"train"`` or ``"val"``)."""


def cache_one_model(cfg: CacheConfig, model_name: str) -> None:
    """Cache predictions for *model_name* across all configured subsets."""
    model = load_model(model_name, device=cfg.device)
    model_dir = cfg.output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    for subset in cfg.subsets:
        cache_file = model_dir / f"{subset}.pt"
        if cache_file.exists() and not cfg.overwrite:
            logger.info("Skip %s/%s (already cached). Use --overwrite to redo.", model_name, subset)
            continue

        logger.info("Caching %s predictions for subset %r …", model_name, subset)
        dataset = GiftEvalDataset(
            subset_names=[subset],
            context_length=cfg.context_length,
            prediction_length=cfg.prediction_length,
            split=cfg.split,
            max_windows_per_series=cfg.max_windows_per_series,
            max_series_per_subset=cfg.max_series_per_subset,
        )

        predictions: dict[str, dict[str, object]] = {}
        skipped = 0

        for i in range(len(dataset)):
            item = dataset[i]
            ctx_norm = item["context"].numpy()  # (V, ctx_len)
            loc = item["loc"].numpy()  # (V, 1)
            scale = item["scale"].numpy()  # (V, 1)
            freq: str = item["freq"]
            item_id: str = item["item_id"]
            window_start: int = item["window_start"]
            pred_len: int = item["prediction_length"]

            # Denormalize context before passing to model
            ctx_raw = denormalize(ctx_norm, loc, scale)

            cache_key = f"{item_id}__{window_start}"

            try:
                result = model.predict(ctx_raw, pred_len, freq)
            except Exception:
                logger.exception(
                    "Prediction failed for %s/%s window %d; skipping.",
                    subset,
                    item_id,
                    window_start,
                )
                skipped += 1
                continue

            predictions[cache_key] = {
                "point": result.point_forecast,  # np.ndarray (V, H)
                "quantiles": result.quantile_forecasts,  # dict | None
            }

            if (i + 1) % 100 == 0:
                logger.info("  %d/%d windows cached …", i + 1, len(dataset))

        torch.save(predictions, cache_file)
        logger.info(
            "Saved %d predictions → %s  (%d skipped)",
            len(predictions),
            cache_file,
            skipped,
        )


def main(cfg: CacheConfig) -> None:
    """Entry point: cache predictions for all configured models and subsets."""
    logger.info("Starting prediction caching with config: %s", cfg)
    for model_name in cfg.models:
        logger.info("=== Model: %s ===", model_name)
        cache_one_model(cfg, model_name)
    logger.info("Done.")


if __name__ == "__main__":
    main(tyro.cli(CacheConfig))
