"""Main entry point for training the WeightedMixture model.

Usage
-----
    uv run python scripts/train_mixture.py \\
        --subsets traffic_hourly weather m5 \\
        --context_length 512 \\
        --loss mase \\
        --lr 0.01 \\
        --epochs 50
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro

sys.path.insert(0, str(Path(__file__).parent.parent))

from tsfc.data.gifteval import GiftEvalDataModule, GiftEvalDataModuleConfig
from tsfc.models.mixture import WeightedMixture
from tsfc.train.trainer import TrainingConfig, train_mixture

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TrainMixtureConfig:
    """Top-level configuration for mixture training."""

    # Data
    subsets: list[str] = field(
        default_factory=lambda: [
            "traffic_hourly",
            "weather",
            "m5",
            "monash_m3_monthly",
        ]
    )
    context_length: int = 512
    prediction_length: int | None = None
    val_frac: float = 0.2
    max_windows_per_series: int = 10
    batch_size: int = 64
    num_workers: int = 4

    # Models
    model_names: list[str] = field(
        default_factory=lambda: ["chronos2", "timesfm25", "moirai2"]
    )
    cache_dir: Path = Path("cache/predictions")

    # Mixture
    adaptive: bool = False
    context_feature_dim: int = 5

    # Training
    loss: str = "mase"
    lr: float = 0.01
    epochs: int = 50
    device: str = "cuda"
    checkpoint_dir: Path = Path("checkpoints")
    log_every_n_steps: int = 10

    # Misc
    seed: int = 42


def main(cfg: TrainMixtureConfig) -> None:
    """Build the mixture model and run the training loop."""
    torch.manual_seed(cfg.seed)
    logger.info("Starting mixture training with config: %s", cfg)

    # Data
    dm_config = GiftEvalDataModuleConfig(
        subset_names=cfg.subsets,
        context_length=cfg.context_length,
        prediction_length=cfg.prediction_length,
        val_frac=cfg.val_frac,
        max_windows_per_series=cfg.max_windows_per_series,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    dm = GiftEvalDataModule(dm_config)
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # Model
    mixture = WeightedMixture(
        model_names=cfg.model_names,
        adaptive=cfg.adaptive,
        context_feature_dim=cfg.context_feature_dim,
    )
    logger.info(
        "WeightedMixture: %d models, adaptive=%s", len(cfg.model_names), cfg.adaptive
    )

    # Training config
    train_cfg = TrainingConfig(
        model_names=cfg.model_names,
        cache_dir=cfg.cache_dir,
        loss=cfg.loss,
        lr=cfg.lr,
        epochs=cfg.epochs,
        device=cfg.device,
        adaptive=cfg.adaptive,
        context_feature_dim=cfg.context_feature_dim,
        log_every_n_steps=cfg.log_every_n_steps,
        checkpoint_dir=cfg.checkpoint_dir,
    )

    history = train_mixture(mixture, train_loader, val_loader, train_cfg)

    logger.info("Training complete.")
    logger.info("Final train loss: %.4f", history["train_loss"][-1])
    if history["val_loss"]:
        logger.info("Final val loss: %.4f", history["val_loss"][-1])


if __name__ == "__main__":
    main(tyro.cli(TrainMixtureConfig))
