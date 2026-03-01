"""Training loop for the WeightedMixture model."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from tsfc.data.transforms import extract_context_features
from tsfc.models.mixture import WeightedMixture
from tsfc.train.loss import mae_loss, mase_loss, mse_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Hyperparameters for :func:`train_mixture`."""

    model_names: list[str] = field(default_factory=lambda: ["chronos2", "timesfm25", "moirai2"])
    cache_dir: Path = field(default_factory=lambda: Path("cache/predictions"))
    loss: str = "mase"  # "mae" | "mase" | "crps"
    lr: float = 0.01
    epochs: int = 50
    device: str = "cuda"
    adaptive: bool = False
    context_feature_dim: int = 5
    log_every_n_steps: int = 10
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    tensorboard_log_dir: Path | None = None  # set to enable TensorBoard logging


# ---------------------------------------------------------------------------
# Cache loader
# ---------------------------------------------------------------------------


def load_cached_predictions(
    cache_dir: Path,
    model_names: list[str],
    dataset_name: str,
    item_id: str,
    window_start: int,
) -> dict[str, torch.Tensor] | None:
    """Load pre-cached model predictions for a single window.

    Returns a dict mapping model name → Tensor ``(V, H)``, or ``None`` if
    any model's cache file is missing.

    Cache layout (written by ``scripts/cache_predictions.py``)::

        cache_dir/<model_name>/<dataset_name>.pt
            key: f"{item_id}__{window_start}"
            value: {"point": np.ndarray (V, H), "quantiles": {...}}
    """
    result: dict[str, torch.Tensor] = {}
    cache_key = f"{item_id}__{window_start}"

    for model_name in model_names:
        cache_file = cache_dir / model_name / f"{dataset_name}.pt"
        if not cache_file.exists():
            logger.warning("Cache file not found: %s", cache_file)
            return None
        cache: dict[str, Any] = torch.load(cache_file, map_location="cpu")
        if cache_key not in cache:
            logger.debug("Cache key %r not found in %s", cache_key, cache_file)
            return None
        point = cache[cache_key]["point"]  # np.ndarray (V, H)
        result[model_name] = torch.from_numpy(point).float()

    return result


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_mixture(
    mixture: WeightedMixture,
    train_loader: Any,  # DataLoader
    val_loader: Any | None = None,
    config: TrainingConfig | None = None,
) -> dict[str, list[float]]:
    """Train the :class:`~tsfc.models.mixture.WeightedMixture` on cached predictions.

    The backbone model predictions must already be cached to disk by
    ``scripts/cache_predictions.py``.  Only the mixture weights (and
    optionally the meta-weighter MLP) are updated.

    Parameters
    ----------
    mixture    : :class:`~tsfc.models.mixture.WeightedMixture`
    train_loader : PyTorch DataLoader yielding batches from :class:`~tsfc.data.gifteval.GiftEvalDataset`.
    val_loader : Optional validation DataLoader.
    config     : :class:`TrainingConfig`  (defaults used if ``None``).

    Returns
    -------
    dict with ``"train_loss"`` and ``"val_loss"`` lists (one entry per epoch).
    """
    if config is None:
        config = TrainingConfig()

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    mixture = mixture.to(device)

    optimizer = torch.optim.Adam(mixture.parameters(), lr=config.lr)  # type: ignore

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard writer (None when tensorboard is not installed or not configured)
    writer = None
    if config.tensorboard_log_dir is not None:
        config.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(config.tensorboard_log_dir))
        logger.info("TensorBoard logging to %s", config.tensorboard_log_dir)

    global_step = 0  # incremented every training step

    for epoch in range(1, config.epochs + 1):
        mixture.train()
        epoch_losses: list[float] = []
        t0 = time.perf_counter()

        for step, batch in enumerate(train_loader):
            context: torch.Tensor = batch["context"].to(device)  # (B, V, ctx_len)
            target: torch.Tensor = batch["target"].to(device)  # (B, V, H)
            mask: torch.Tensor = batch["mask"].to(device)  # (B, H)
            freqs: list[str] = batch["freq"]
            dataset_names: list[str] = batch["dataset_name"]
            item_ids: list[str] = batch["item_id"]
            window_starts: list[int] = batch["window_start"]

            # Load cached predictions for every item in the batch
            model_preds: dict[str, list[torch.Tensor]] = {m: [] for m in config.model_names}
            valid_indices: list[int] = []

            for i in range(context.shape[0]):
                preds = load_cached_predictions(
                    cache_dir=config.cache_dir,
                    model_names=config.model_names,
                    dataset_name=dataset_names[i],
                    item_id=item_ids[i],
                    window_start=window_starts[i],
                )
                if preds is None:
                    continue
                valid_indices.append(i)
                for m in config.model_names:
                    model_preds[m].append(preds[m])

            if not valid_indices:
                continue  # all cache misses — skip batch

            # Stack per-model predictions: (B', V, H)
            stacked: dict[str, torch.Tensor] = {
                m: torch.stack(model_preds[m], dim=0).to(device) for m in config.model_names
            }

            ctx_valid = context[valid_indices]
            tgt_valid = target[valid_indices]
            mask_valid = mask[valid_indices]
            freqs_valid = [freqs[i] for i in valid_indices]

            # Optional: extract context features for adaptive weighting
            ctx_feats: torch.Tensor | None = None
            if config.adaptive:
                ctx_feats = extract_context_features(ctx_valid)  # (B', 5)

            # Forward pass through mixture
            forecast = mixture(stacked, ctx_feats)  # (B', V, H)

            # Compute loss
            if config.loss == "mase":
                loss = mase_loss(forecast, tgt_valid, ctx_valid, freqs_valid, mask_valid)
            elif config.loss == "mae":
                loss = mae_loss(forecast, tgt_valid, mask_valid)
            elif config.loss == "mse":
                loss = mse_loss(forecast, tgt_valid, mask_valid)
            else:
                loss = mae_loss(forecast, tgt_valid, mask_valid)

            optimizer.zero_grad()
            loss.backward()

            # --- gradient norms (before optimizer step) ---
            total_grad_norm = 0.0
            for name, param in mixture.named_parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.detach().norm(2).item()
                    total_grad_norm += param_grad_norm**2
                    if writer is not None:
                        writer.add_scalar(f"grad_norm/{name}", param_grad_norm, global_step)
            total_grad_norm = total_grad_norm**0.5

            optimizer.step()

            # --- mixture weights (batch-averaged) ---
            with torch.no_grad():
                batch_weights = mixture.get_weights(ctx_feats, batch_size=len(valid_indices)).mean(
                    dim=0
                )  # (n_models,)

            epoch_losses.append(loss.item())

            if writer is not None:
                writer.add_scalar("train/loss_step", loss.item(), global_step)
                writer.add_scalar("train/grad_norm", total_grad_norm, global_step)
                writer.add_scalar(
                    "train/lr",
                    optimizer.param_groups[0]["lr"],
                    global_step,
                )
                for i, model_name in enumerate(config.model_names):
                    writer.add_scalar(
                        f"mixture/weight/{model_name}",
                        batch_weights[i].item(),
                        global_step,
                    )

            global_step += 1

            if (step + 1) % config.log_every_n_steps == 0:
                logger.info(
                    "Epoch %d | step %d | loss=%.4f | grad_norm=%.4f",
                    epoch,
                    step + 1,
                    float(torch.tensor(epoch_losses).mean()),
                    total_grad_norm,
                )

        avg_train = float(torch.tensor(epoch_losses).mean()) if epoch_losses else float("nan")
        history["train_loss"].append(avg_train)

        # Validation
        avg_val = float("nan")
        if val_loader is not None:
            mixture.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    context = batch["context"].to(device)
                    target = batch["target"].to(device)
                    mask = batch["mask"].to(device)
                    freqs = batch["freq"]
                    dataset_names = batch["dataset_name"]
                    item_ids = batch["item_id"]
                    window_starts = batch["window_start"]

                    model_preds = {m: [] for m in config.model_names}
                    valid_indices = []

                    for i in range(context.shape[0]):
                        preds = load_cached_predictions(
                            cache_dir=config.cache_dir,
                            model_names=config.model_names,
                            dataset_name=dataset_names[i],
                            item_id=item_ids[i],
                            window_start=window_starts[i],
                        )
                        if preds is None:
                            continue
                        valid_indices.append(i)
                        for m in config.model_names:
                            model_preds[m].append(preds[m])

                    if not valid_indices:
                        continue

                    stacked = {
                        m: torch.stack(model_preds[m], dim=0).to(device) for m in config.model_names
                    }
                    ctx_valid = context[valid_indices]
                    tgt_valid = target[valid_indices]
                    mask_valid = mask[valid_indices]
                    freqs_valid = [freqs[i] for i in valid_indices]

                    ctx_feats = None
                    if config.adaptive:
                        ctx_feats = extract_context_features(ctx_valid)

                    forecast = mixture(stacked, ctx_feats)
                    loss = mae_loss(forecast, tgt_valid, mask_valid)
                    val_losses.append(loss.item())

            avg_val = float(torch.tensor(val_losses).mean()) if val_losses else float("nan")
            history["val_loss"].append(avg_val)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  (%.1fs)",
            epoch,
            config.epochs,
            avg_train,
            avg_val,
            elapsed,
        )

        if writer is not None:
            writer.add_scalar("train/loss_epoch", avg_train, epoch)
            writer.add_scalar("train/epoch_time_s", elapsed, epoch)
            if not torch.isnan(torch.tensor(avg_val)):
                writer.add_scalar("val/loss_epoch", avg_val, epoch)

            # Static mode: log raw log_weights and softmax weights per epoch
            if not config.adaptive and hasattr(mixture, "log_weights"):
                with torch.no_grad():
                    static_w = torch.softmax(mixture.log_weights, dim=0)
                    for i, model_name in enumerate(config.model_names):
                        writer.add_scalar(
                            f"mixture/weight_epoch/{model_name}",
                            static_w[i].item(),
                            epoch,
                        )
                        writer.add_scalar(
                            f"mixture/log_weight/{model_name}",
                            mixture.log_weights[i].item(),
                            epoch,
                        )

            # Parameter norms (weights magnitude)
            for name, param in mixture.named_parameters():
                writer.add_scalar(f"param_norm/{name}", param.detach().norm(2).item(), epoch)

        # Save checkpoint
        ckpt_path = config.checkpoint_dir / f"mixture_epoch{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": mixture.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": avg_train,
                "val_loss": avg_val,
            },
            ckpt_path,
        )

    if writer is not None:
        writer.close()

    return history
