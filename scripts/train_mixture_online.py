"""Train WeightedMixture with on-the-fly frozen model inference.

This script merges ``cache_predictions.py`` and ``train_mixture.py``: instead
of loading pre-cached model predictions, the frozen backbone models run live
during each training step.  Three data splits are used:

* **train** – windows sampled from the first ``(1 - val_frac)`` of each series.
* **val**   – single window ending at the train/val boundary.
* **test**  – single window immediately after the val window (fresh target data).

A TensorBoard SummaryWriter is created inside a timestamped ``logs/`` directory.
All metrics (loss, MSE, MAE, MASE) are logged per step and per epoch for train,
periodically for val, and once for test.  Results are also written to CSV.

Usage
-----
    uv run python scripts/train_mixture_online.py \\
        --subsets traffic_hourly weather m5 \\
        --model_names chronos2 timesfm25 moirai2 \\
        --loss mase \\
        --lr 0.01 \\
        --epochs 50 \\
        --val_every_n_epochs 5
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from torch.utils.tensorboard.writer import SummaryWriter

import tsfc.models  # type: ignore – triggers @register decorators  # noqa: F401
from tsfc.data.gifteval import GiftEvalDataModule, GiftEvalDataModuleConfig
from tsfc.data.transforms import denormalize, extract_context_features
from tsfc.models.base import ForecastingModel
from tsfc.models.mixture import WeightedMixture
from tsfc.models.registry import load_model
from tsfc.train.loss import mae_loss, mase_loss, mse_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Top-level configuration for online mixture training."""

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
    max_series_per_subset: int | None = None
    batch_size: int = 32
    num_workers: int = 0  # 0 avoids multiprocessing contention with model loading

    # Models (frozen)
    model_names: list[str] = field(default_factory=lambda: ["chronos2", "timesfm25", "moirai2"])
    device: str = "cuda"

    # Mixture
    adaptive: bool = False
    context_feature_dim: int = 5

    # Boosting head (patch-transformer residual correction)
    boosting: bool = False
    boosting_prediction_length: int = 12
    boosting_patch_size: int = 16
    boosting_d_model: int = 64
    boosting_n_heads: int = 4
    boosting_n_layers: int = 2
    boosting_dropout: float = 0.1

    # Training
    loss: str = "mase"  # "mse" | "mae" | "mase"
    lr: float = 0.01
    epochs: int = 50
    checkpoint_dir: Path = Path("checkpoints")

    # Logging
    log_every_n_steps: int = 10
    val_every_n_epochs: int = 5
    log_dir: Path = Path("logs")
    run_name: str | None = None  # auto-generated from timestamp when None

    # Misc
    seed: int = 42

    def __post_init__(self):
            if self.boosting and self.prediction_length is not None:
                assert self.boosting_prediction_length >= self.prediction_length, (
                    "When boosting is enabled, boosting_prediction_length must be at least prediction_length."
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_run_dir(log_dir: Path, run_name: str | None) -> Path:
    """Create and return a timestamped run directory under *log_dir*."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{timestamp}_{run_name}" if run_name else timestamp
    run_dir = log_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def compute_loss_value(
    name: str,
    forecast: torch.Tensor,
    target: torch.Tensor,
    context: torch.Tensor,
    freqs: list[str],
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the named loss (mse / mae / mase)."""
    if name == "mse":
        return mse_loss(forecast, target, mask)
    if name == "mae":
        return mae_loss(forecast, target, mask)
    if name == "mase":
        return mase_loss(forecast, target, context, freqs, mask)
    return mae_loss(forecast, target, mask)


@torch.no_grad() # type: ignore
def compute_all_metrics(
    forecast: torch.Tensor,
    target: torch.Tensor,
    context: torch.Tensor,
    freqs: list[str],
    mask: torch.Tensor,
) -> dict[str, float]:
    """Return MSE, MAE, and MASE (all in normalised space)."""
    return {
        "mse": mse_loss(forecast, target, mask).item(),
        "mae": mae_loss(forecast, target, mask).item(),
        "mase": mase_loss(forecast, target, context, freqs, mask).item(),
    }


def run_batch_inference(
    models: dict[str, ForecastingModel],
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Run on-the-fly frozen inference for all models on one batch.

    For each item ``i`` the raw context is reconstructed from the normalised
    context plus the per-variate ``loc`` / ``scale`` stored in the batch.
    Each model predicts in raw scale; the result is re-normalised before
    stacking so the mixture sees normalised predictions (consistent with the
    normalised target).

    Returns
    -------
    stacked : dict model_name → Tensor ``(B', V, H)`` on *device*
        Only items where **all** models succeeded are included.
    valid_indices : list[int]
        Positions in the original batch that appear in *stacked*.
    """
    ctx_norms: np.ndarray[Any, np.dtype[np.floating[Any]]] = batch["context"].cpu().numpy()  # (B, V, ctx_len)
    locs: np.ndarray[Any, np.dtype[np.floating[Any]]] = batch["loc"].cpu().numpy()  # (B, V, 1)
    scales: np.ndarray[Any, np.dtype[np.floating[Any]]] = batch["scale"].cpu().numpy()  # (B, V, 1)
    pred_lens: list[int] = batch["prediction_length"]  # list[int] from collate
    freqs: list[str] = batch["freq"]
    B = ctx_norms.shape[0]
    max_pred_len: int = batch["target"].shape[-1]

    model_preds: dict[str, list[np.ndarray[Any, np.dtype[np.floating[Any]]]]] = {name: [] for name in models}
    valid_indices: list[int] = []

    for i in range(B):
        ctx_raw_i = denormalize(ctx_norms[i], locs[i], scales[i])  # (V, ctx_len)
        pred_len_i: int = pred_lens[i]
        freq_i: str = freqs[i]

        item_preds: dict[str, np.ndarray[Any, np.dtype[np.floating[Any]]]] = {}
        all_ok = True

        for name, model in models.items():
            try:
                result = model.predict(ctx_raw_i, pred_len_i, freq_i)
                pred_raw = result.point_forecast  # (V, pred_len_i)
                pred_norm = (pred_raw - locs[i]) / scales[i]  # (V, pred_len_i)
                # Pad to max_pred_len (matches padding in collate_fn for target)
                if pred_len_i < max_pred_len:
                    pad_width = max_pred_len - pred_len_i
                    pred_norm = np.concatenate(
                        [pred_norm, np.zeros((pred_norm.shape[0], pad_width), dtype=np.float32)],
                        axis=-1,
                    )
                item_preds[name] = pred_norm.astype(np.float32)
            except Exception:
                logger.warning(
                    "Model %r failed for item %d (freq=%s, pred_len=%d); skipping batch item.",
                    name,
                    i,
                    freq_i,
                    pred_len_i,
                    exc_info=True,
                )
                all_ok = False
                break

        if all_ok:
            valid_indices.append(i)
            for name in models:
                model_preds[name].append(item_preds[name])

    if not valid_indices:
        return {}, []

    stacked: dict[str, torch.Tensor] = {
        name: torch.from_numpy(np.stack(preds)).to(device)
        for name, preds in model_preds.items()
    }
    return stacked, valid_indices


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


@torch.no_grad() # type: ignore
def evaluate_loader(
    mixture: WeightedMixture,
    models: dict[str, ForecastingModel],
    loader: Any,
    device: torch.device,
    adaptive: bool,
    loss_name: str,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Evaluate on a full DataLoader.

    Returns
    -------
    mixture_metrics : dict with keys ``loss``, ``mse``, ``mae``, ``mase``
    model_metrics   : dict model_name → same keys
    """
    mixture.eval()

    mix_buckets: dict[str, list[float]] = {"loss": [], "mse": [], "mae": [], "mase": []}
    per_model_buckets: dict[str, dict[str, list[float]]] = {
        name: {"loss": [], "mse": [], "mae": [], "mase": []} for name in models
    }

    for batch in loader:
        stacked, valid_indices = run_batch_inference(models, batch, device)
        if not valid_indices:
            continue

        context = batch["context"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        freqs: list[str] = batch["freq"]

        ctx_valid = context[valid_indices]
        tgt_valid = target[valid_indices]
        mask_valid = mask[valid_indices]
        freqs_valid = [freqs[i] for i in valid_indices]

        ctx_feats = extract_context_features(ctx_valid) if adaptive else None
        forecast = mixture(stacked, ctx_feats, context=ctx_valid)  # (B', V, H)

        # Mixture metrics
        mix_loss = compute_loss_value(loss_name, forecast, tgt_valid, ctx_valid, freqs_valid, mask_valid)
        mix_metrics = compute_all_metrics(forecast, tgt_valid, ctx_valid, freqs_valid, mask_valid)
        mix_buckets["loss"].append(mix_loss.item())
        for k, v in mix_metrics.items():
            mix_buckets[k].append(v)

        # Per-model metrics (no mixture weighting)
        for name in models:
            model_pred = stacked[name]  # (B', V, H) — already normalised
            m_loss = compute_loss_value(loss_name, model_pred, tgt_valid, ctx_valid, freqs_valid, mask_valid)
            m_metrics = compute_all_metrics(model_pred, tgt_valid, ctx_valid, freqs_valid, mask_valid)
            per_model_buckets[name]["loss"].append(m_loss.item())
            for k, v in m_metrics.items():
                per_model_buckets[name][k].append(v)

    def _avg(lst: list[float]) -> float:
        return float(np.mean(lst)) if lst else float("nan")

    mixture_metrics = {k: _avg(v) for k, v in mix_buckets.items()}
    model_metrics = {
        name: {k: _avg(v) for k, v in bkts.items()}
        for name, bkts in per_model_buckets.items()
    }
    return mixture_metrics, model_metrics


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def save_results_csv(
    run_dir: Path,
    mixture_metrics: dict[str, float],
    model_metrics: dict[str, dict[str, float]],
    split: str = "test",
) -> Path:
    """Write final metrics to ``<run_dir>/results.csv`` and return its path."""
    csv_path = run_dir / "results.csv"
    rows: list[dict[str, str]] = []

    for metric, value in mixture_metrics.items():
        rows.append({"split": split, "model": "mixture", "metric": metric, "value": f"{value:.6f}"})

    for model_name, metrics in model_metrics.items():
        for metric, value in metrics.items():
            rows.append(
                {"split": split, "model": model_name, "metric": metric, "value": f"{value:.6f}"}
            )

    fieldnames = ["split", "model", "metric", "value"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Results saved to %s", csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    mixture: WeightedMixture,
    models: dict[str, ForecastingModel],
    train_loader: Any,
    val_loader: Any,
    cfg: Config,
    device: torch.device,
    writer: SummaryWriter,
    run_dir: Path,
) -> dict[str, list[float]]:
    """Main training loop.

    Returns a history dict with ``train_loss``, ``train_mse``, ``train_mae``,
    ``train_mase``, and ``val_*`` equivalents (list of per-epoch averages).
    """
    mixture = mixture.to(device)
    optimizer = torch.optim.Adam(mixture.parameters(), lr=cfg.lr) # type: ignore

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_mse": [],
        "train_mae": [],
        "train_mase": [],
        "val_loss": [],
        "val_mse": [],
        "val_mae": [],
        "val_mase": [],
    }

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        mixture.train()

        epoch_buckets: dict[str, list[float]] = {
            "loss": [], "mse": [], "mae": [], "mase": []
        }
        t0 = time.perf_counter()

        for step, batch in enumerate(train_loader):
            stacked, valid_indices = run_batch_inference(models, batch, device)
            if not valid_indices:
                logger.debug("All inference failed for step %d – skipping.", step)
                continue

            context = batch["context"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            freqs: list[str] = batch["freq"]

            ctx_valid = context[valid_indices]
            tgt_valid = target[valid_indices]
            mask_valid = mask[valid_indices]
            freqs_valid = [freqs[i] for i in valid_indices]

            ctx_feats: torch.Tensor | None = None
            if cfg.adaptive:
                ctx_feats = extract_context_features(ctx_valid)

            forecast = mixture(stacked, ctx_feats, context=ctx_valid)  # (B', V, H)

            loss = compute_loss_value(cfg.loss, forecast, tgt_valid, ctx_valid, freqs_valid, mask_valid)

            optimizer.zero_grad()
            loss.backward()

            # Gradient norms
            total_grad_norm = 0.0
            for p_name, param in mixture.named_parameters():
                if param.grad is not None:
                    g_norm = param.grad.detach().norm(2).item()
                    total_grad_norm += g_norm ** 2
                    writer.add_scalar(f"grad_norm/{p_name}", g_norm, global_step)
            total_grad_norm = total_grad_norm ** 0.5

            optimizer.step()

            # Step-level metrics (no grad needed)
            step_metrics = compute_all_metrics(
                forecast.detach(), tgt_valid, ctx_valid, freqs_valid, mask_valid
            )
            loss_val = loss.item()

            for k in epoch_buckets:
                epoch_buckets[k].append(step_metrics[k] if k != "loss" else loss_val)

            # Batch-averaged mixture weights
            with torch.no_grad():
                batch_weights = mixture.get_weights(
                    ctx_feats, batch_size=len(valid_indices)
                ).mean(dim=0)  # (n_models,)

            # TensorBoard – step level
            writer.add_scalar("train/loss_step", loss_val, global_step)
            writer.add_scalar("train/mse_step", step_metrics["mse"], global_step)
            writer.add_scalar("train/mae_step", step_metrics["mae"], global_step)
            writer.add_scalar("train/mase_step", step_metrics["mase"], global_step)
            writer.add_scalar("train/grad_norm", total_grad_norm, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            for mi, model_name in enumerate(cfg.model_names):
                writer.add_scalar(
                    f"mixture/weight/{model_name}", batch_weights[mi].item(), global_step
                )

            global_step += 1

            if (step + 1) % cfg.log_every_n_steps == 0:
                logger.info(
                    "Epoch %d | step %d | loss=%.4f  mse=%.4f  mae=%.4f  mase=%.4f | grad_norm=%.4f",
                    epoch,
                    step + 1,
                    loss_val,
                    step_metrics["mse"],
                    step_metrics["mae"],
                    step_metrics["mase"],
                    total_grad_norm,
                )

        elapsed = time.perf_counter() - t0

        # Epoch averages – train
        def _avg(lst: list[float]) -> float:
            return float(np.mean(lst)) if lst else float("nan")

        avg_train = {k: _avg(v) for k, v in epoch_buckets.items()}
        history["train_loss"].append(avg_train["loss"])
        history["train_mse"].append(avg_train["mse"])
        history["train_mae"].append(avg_train["mae"])
        history["train_mase"].append(avg_train["mase"])

        writer.add_scalar("train/loss_epoch", avg_train["loss"], epoch)
        writer.add_scalar("train/mse_epoch", avg_train["mse"], epoch)
        writer.add_scalar("train/mae_epoch", avg_train["mae"], epoch)
        writer.add_scalar("train/mase_epoch", avg_train["mase"], epoch)
        writer.add_scalar("train/epoch_time_s", elapsed, epoch)

        # Static weights per epoch
        if not cfg.adaptive and hasattr(mixture, "log_weights"):
            with torch.no_grad():
                static_w = torch.softmax(mixture.log_weights, dim=0)
                for mi, model_name in enumerate(cfg.model_names):
                    writer.add_scalar(
                        f"mixture/weight_epoch/{model_name}", static_w[mi].item(), epoch
                    )
                    writer.add_scalar(
                        f"mixture/log_weight/{model_name}",
                        mixture.log_weights[mi].item(),
                        epoch,
                    )

        # Parameter norms
        for p_name, param in mixture.named_parameters():
            writer.add_scalar(f"param_norm/{p_name}", param.detach().norm(2).item(), epoch)

        logger.info(
            "Epoch %d/%d  loss=%.4f  mse=%.4f  mae=%.4f  mase=%.4f  (%.1fs)",
            epoch,
            cfg.epochs,
            avg_train["loss"],
            avg_train["mse"],
            avg_train["mae"],
            avg_train["mase"],
            elapsed,
        )

        # Validation (every val_every_n_epochs epochs)
        if epoch % cfg.val_every_n_epochs == 0 or epoch == cfg.epochs:
            logger.info("Running validation at epoch %d…", epoch)
            val_mix, _ = evaluate_loader(mixture, models, val_loader, device, cfg.adaptive, cfg.loss)
            history["val_loss"].append(val_mix["loss"])
            history["val_mse"].append(val_mix["mse"])
            history["val_mae"].append(val_mix["mae"])
            history["val_mase"].append(val_mix["mase"])

            writer.add_scalar("val/loss", val_mix["loss"], epoch)
            writer.add_scalar("val/mse", val_mix["mse"], epoch)
            writer.add_scalar("val/mae", val_mix["mae"], epoch)
            writer.add_scalar("val/mase", val_mix["mase"], epoch)

            logger.info(
                "  val  loss=%.4f  mse=%.4f  mae=%.4f  mase=%.4f",
                val_mix["loss"],
                val_mix["mse"],
                val_mix["mae"],
                val_mix["mase"],
            )
            mixture.train()

        # Checkpoint
        ckpt_path = cfg.checkpoint_dir / f"mixture_epoch{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": mixture.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": avg_train["loss"],
            },
            ckpt_path,
        )

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    logger.info("Config: %s", cfg)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Timestamped run directory
    run_dir = make_run_dir(cfg.log_dir, cfg.run_name)
    logger.info("Run directory: %s", run_dir)

    # Save config to JSON
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)
    logger.info("Config saved to %s", config_path)

    writer = SummaryWriter(log_dir=str(run_dir), comment=cfg.run_name or "")
    logger.info("TensorBoard writer: %s", run_dir)

    # Load frozen backbone models
    logger.info("Loading backbone models: %s", cfg.model_names)
    models: dict[str, ForecastingModel] = {}
    for name in cfg.model_names:
        logger.info("  Loading %r …", name)
        models[name] = load_model(name, device=cfg.device)
    logger.info("All backbone models loaded.")

    # Data
    dm_config = GiftEvalDataModuleConfig(
        subset_names=cfg.subsets,
        context_length=cfg.context_length,
        prediction_length=cfg.prediction_length,
        val_frac=cfg.val_frac,
        max_windows_per_series=cfg.max_windows_per_series,
        max_series_per_subset=cfg.max_series_per_subset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    dm = GiftEvalDataModule(dm_config)
    dm.setup(None)  # build all three splits
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    logger.info(
        "Dataset sizes – train: %d  val: %d  test: %d",
        len(dm._train_dataset),  # type: ignore[arg-type]
        len(dm._val_dataset),  # type: ignore[arg-type]
        len(dm._test_dataset),  # type: ignore[arg-type]
    )

    # Mixture model
    mixture = WeightedMixture(
        model_names=cfg.model_names,
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
    logger.info(
        "WeightedMixture: %d models, adaptive=%s, boosting=%s",
        len(cfg.model_names),
        cfg.adaptive,
        cfg.boosting,
    )

    # Training
    _ = train(
        mixture=mixture,
        models=models,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        writer=writer,
        run_dir=run_dir,
    )

    # -----------------------------------------------------------------------
    # Test evaluation: mixture + individual models
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Running test evaluation…")
    test_mix_metrics, test_model_metrics = evaluate_loader(
        mixture, models, test_loader, device, cfg.adaptive, cfg.loss
    )

    # Log test metrics to TensorBoard (at step=0 for the test scalars)
    for metric, value in test_mix_metrics.items():
        writer.add_scalar(f"test/mixture/{metric}", value, 0)
    for model_name, metrics in test_model_metrics.items():
        for metric, value in metrics.items():
            writer.add_scalar(f"test/{model_name}/{metric}", value, 0)

    writer.close()

    # Save CSV
    csv_path = save_results_csv(run_dir, test_mix_metrics, test_model_metrics, split="test")

    # -----------------------------------------------------------------------
    # Print final summary
    # -----------------------------------------------------------------------
    sep = "=" * 60
    logger.info(sep)
    logger.info("FINAL TEST RESULTS")
    logger.info(sep)

    header = f"{'Model':<20} {'Loss':>10} {'MSE':>10} {'MAE':>10} {'MASE':>10}"
    logger.info(header)
    logger.info("-" * len(header))

    def _fmt(d: dict[str, float]) -> str:
        return (
            f"{d['loss']:>10.4f} {d['mse']:>10.4f} {d['mae']:>10.4f} {d['mase']:>10.4f}"
        )

    logger.info("%-20s %s", "mixture", _fmt(test_mix_metrics))
    for model_name in cfg.model_names:
        logger.info("%-20s %s", model_name, _fmt(test_model_metrics[model_name]))

    logger.info(sep)
    logger.info("CSV results: %s", csv_path)
    logger.info("TensorBoard:  tensorboard --logdir %s", run_dir)
    logger.info(sep)

    # Also print to stdout for easy reading
    print(f"\n{sep}")
    print("FINAL TEST RESULTS")
    print(sep)
    print(header)
    print("-" * len(header))
    print(f"{'mixture':<20} {_fmt(test_mix_metrics)}")
    for model_name in cfg.model_names:
        print(f"{model_name:<20} {_fmt(test_model_metrics[model_name])}")
    print(sep)
    print(f"CSV:         {csv_path}")
    print(f"TensorBoard: tensorboard --logdir {run_dir}")
    print(sep)


if __name__ == "__main__":
    main(tyro.cli(Config))
