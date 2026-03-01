"""GiftEvalPretrain dataset loading, windowing, and PyTorch Lightning DataModule."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tsfc.data.transforms import freq_to_default_horizon, robust_normalize
from tsfc.data.utils import fill_nan_linear, nan_fraction, normalize_freq

logger = logging.getLogger(__name__)

# Maximum allowed NaN fraction in the target window before the window is skipped.
_MAX_NAN_FRAC: float = 0.20

# Subset names suitable for fast iteration / development.
FAST_DEV_SUBSETS: list[str] = [
    "traffic_hourly",
    "pedestrian_counts",
    "nn5_daily_with_missing",
    "weather",
    "nn5_weekly",
    "monash_m3_monthly",
    "tourism_monthly",
    "m5",
    "fred_md",
]


# ---------------------------------------------------------------------------
# Window index entry (one per sampled window)
# ---------------------------------------------------------------------------


@dataclass
class _WindowEntry:
    """Lightweight descriptor of a single training / validation window."""

    series: np.ndarray[
        Any, np.dtype[np.floating[Any]]
    ]  # (n_variates, T_slice)  — the relevant time slice
    start_idx: int  # index within *series* where context begins
    freq: str
    dataset_name: str
    item_id: str
    prediction_length: int
    context_length: int


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class GiftEvalDataset(Dataset[dict[str, Any]]):
    """PyTorch Dataset wrapping GiftEvalPretrain windows.

    Parameters
    ----------
    subset_names : list[str]
        HuggingFace config names to load (e.g. ``["m5", "weather"]``).
    context_length : int
        Number of past time steps fed to the model.
    prediction_length : int | None
        Number of future steps to forecast.  ``None`` → use freq-based default.
    split : "train" | "val"
        * ``"train"`` — windows sampled from the first ``(1 - val_frac)``
          of each series.
        * ``"val"`` — single evaluation window from the end of each series.
    val_frac : float
        Fraction of each series reserved for validation (applied by time).
    max_windows_per_series : int
        Maximum number of training windows sampled from a single series.
    max_series_per_subset : int | None
        Cap the number of series loaded per subset (useful during development).
    streaming : bool
        Use HuggingFace streaming mode for large subsets (avoids OOM).
    cache_dir : Path | None
        Local directory for HuggingFace dataset cache.
    seed : int
        Random seed for window sampling.
    """

    def __init__(
        self,
        subset_names: list[str],
        context_length: int = 512,
        prediction_length: int | None = None,
        split: str = "train",
        val_frac: float = 0.2,
        max_windows_per_series: int = 10,
        max_series_per_subset: int | None = None,
        streaming: bool = False,
        cache_dir: Path | None = None,
        seed: int = 42,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.split = split
        self.val_frac = val_frac
        self.max_windows_per_series = max_windows_per_series
        self.seed = seed

        rng = random.Random(seed)
        self._windows: list[_WindowEntry] = []

        for subset in subset_names:
            rows = self._load_subset(subset, streaming, cache_dir, max_series_per_subset)
            for row in rows:
                entries = self._build_windows(row, rng)
                self._windows.extend(entries)

        logger.info(
            "GiftEvalDataset[%s]: %d windows from %d subset(s)",
            split,
            len(self._windows),
            len(subset_names),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_subset(
        self,
        subset: str,
        streaming: bool,
        cache_dir: Path | None,
        max_series: int | None,
    ) -> list[dict[str, Any]]:
        """Load a single HuggingFace subset and return a list of row dicts."""
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required. Install it with: uv add datasets"
            ) from exc

        kwargs: dict[str, Any] = {
            "path": "theforecastingcompany/GiftEvalPretrain",
            "name": subset,
            "split": "train",
            "streaming": streaming,
        }
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)

        logger.info("Loading subset %r (streaming=%s)…", subset, streaming)
        ds = load_dataset(**kwargs)

        rows: list[dict[str, Any]] = []
        for i, row in enumerate(ds):
            if max_series is not None and i >= max_series:
                break
            rows.append(row)

        logger.info("  → %d series loaded from %r", len(rows), subset)
        return rows

    def _build_windows(
        self,
        row: dict[str, Any],
        rng: random.Random,
    ) -> list[_WindowEntry]:
        """Produce window entries for a single series row."""
        raw_target: list[list[float]] = row["target"]
        freq: str = normalize_freq(row["freq"])
        dataset_name: str = row["dataset_name"]
        item_id: str = row["item_id"]

        # Convert to (n_variates, T) float32 array
        series = np.array(raw_target, dtype=np.float32)
        if series.ndim == 1:
            series = series[np.newaxis, :]  # → (1, T)

        _, T = series.shape  # n_variates, T

        pred_len = self.prediction_length
        if pred_len is None:
            pred_len = freq_to_default_horizon(freq)

        total_len = self.context_length + pred_len

        if total_len > T:
            logger.debug(
                "Skipping %s/%s: length %d < required %d", dataset_name, item_id, T, total_len
            )
            return []

        # Train/val boundary (by time index)
        T_train = max(total_len, int((1.0 - self.val_frac) * T))

        entries: list[_WindowEntry] = []

        if self.split == "val":
            # Single evaluation window: context ends at T_train, target follows
            start_idx = T_train - self.context_length
            if start_idx < 0:
                return []
            target_slice = series[:, T_train : T_train + pred_len]
            if target_slice.shape[1] < pred_len:
                return []
            # Check NaN fraction in target
            if nan_fraction(target_slice) > _MAX_NAN_FRAC:
                logger.debug("Skipping val window for %s/%s: too many NaNs", dataset_name, item_id)
                return []
            entries.append(
                _WindowEntry(
                    series=series,
                    start_idx=start_idx,
                    freq=freq,
                    dataset_name=dataset_name,
                    item_id=item_id,
                    prediction_length=pred_len,
                    context_length=self.context_length,
                )
            )
        else:
            # Training: sample random windows from [0, T_train - total_len]
            max_start = T_train - total_len
            if max_start < 0:
                return []
            n_windows = min(self.max_windows_per_series, max_start + 1)
            starts = rng.sample(range(max_start + 1), n_windows)
            for start_idx in starts:
                target_slice = series[:, start_idx + self.context_length : start_idx + total_len]
                if nan_fraction(target_slice) > _MAX_NAN_FRAC:
                    continue
                entries.append(
                    _WindowEntry(
                        series=series,
                        start_idx=start_idx,
                        freq=freq,
                        dataset_name=dataset_name,
                        item_id=item_id,
                        prediction_length=pred_len,
                        context_length=self.context_length,
                    )
                )

        return entries

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._windows[idx]
        ctx_len = entry.context_length
        pred_len = entry.prediction_length
        s = entry.start_idx

        ctx_raw = entry.series[:, s : s + ctx_len]  # (V, ctx_len)
        tgt_raw = entry.series[:, s + ctx_len : s + ctx_len + pred_len]  # (V, pred_len)

        # Fill NaN in context; compute target mask
        ctx_filled, _ = fill_nan_linear(ctx_raw)
        tgt_filled, tgt_observed = fill_nan_linear(tgt_raw)

        # Aggregate mask: True iff ALL variates are observed at that step
        # shape: (pred_len,)
        mask = tgt_observed.all(axis=0)

        # Instance normalization (on filled context)
        ctx_norm, loc, scale = robust_normalize(ctx_filled)  # each (V,1)

        # Normalize target with same stats (for loss computation)
        tgt_norm = (tgt_filled - loc) / scale

        return {
            "context": torch.from_numpy(ctx_norm).float(),  # (V, ctx_len)
            "target": torch.from_numpy(tgt_norm).float(),  # (V, pred_len)
            "target_raw": torch.from_numpy(tgt_filled).float(),  # (V, pred_len) un-normed
            "mask": torch.from_numpy(mask),  # (pred_len,)
            "freq": entry.freq,
            "dataset_name": entry.dataset_name,
            "item_id": entry.item_id,
            "loc": torch.from_numpy(loc).float(),  # (V, 1)
            "scale": torch.from_numpy(scale).float(),  # (V, 1)
            "prediction_length": pred_len,
            "window_start": s,
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate that handles variable prediction lengths and string fields.

    When ``prediction_length=None`` is used, different series may have different
    horizon lengths (set by their frequency default).  Tensors whose last
    dimension equals ``prediction_length`` are padded to the longest horizon in
    the batch; the ``mask`` is extended with ``False`` for padding positions so
    the loss ignores them.
    """
    string_keys = {"freq", "dataset_name", "item_id"}
    int_keys = {"prediction_length", "window_start"}
    # Keys whose trailing dimension is prediction_length and may vary per item.
    pad_keys = {"target", "target_raw", "mask"}

    max_pred_len: int = max(item["prediction_length"] for item in batch)

    out: dict[str, Any] = {}
    for key in batch[0]:
        if key in string_keys or key in int_keys:
            out[key] = [item[key] for item in batch]
        elif key in pad_keys:
            padded: list[torch.Tensor] = []
            for item in batch:
                t: torch.Tensor = item[key]  # type: ignore[assignment]
                pad = max_pred_len - t.shape[-1]
                if pad > 0:
                    # bool tensors (mask) padded with False; float tensors with 0
                    fill = False if t.dtype == torch.bool else 0.0
                    t = torch.nn.functional.pad(t, (0, pad), value=fill)  # type: ignore[arg-type]
                padded.append(t)
            out[key] = torch.stack(padded)
        else:
            out[key] = torch.stack([item[key] for item in batch])  # type: ignore[arg-type]
    return out


# ---------------------------------------------------------------------------
# DataModule config
# ---------------------------------------------------------------------------


@dataclass
class GiftEvalDataModuleConfig:
    """Configuration for :class:`GiftEvalDataModule`."""

    subset_names: list[str] = field(default_factory=lambda: FAST_DEV_SUBSETS)
    context_length: int = 512
    prediction_length: int | None = None
    val_frac: float = 0.2
    max_windows_per_series: int = 10
    max_series_per_subset: int | None = None
    batch_size: int = 64
    num_workers: int = 4
    streaming: bool = False
    cache_dir: Path | None = None
    seed: int = 42


# ---------------------------------------------------------------------------
# PyTorch Lightning DataModule
# ---------------------------------------------------------------------------


class GiftEvalDataModule:
    """PyTorch Lightning–compatible DataModule for GiftEvalPretrain.

    Parameters
    ----------
    config : GiftEvalDataModuleConfig
    """

    def __init__(self, config: GiftEvalDataModuleConfig | None = None) -> None:
        self.config = config or GiftEvalDataModuleConfig()
        self._train_dataset: GiftEvalDataset | None = None
        self._val_dataset: GiftEvalDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Instantiate train and validation datasets."""
        cfg = self.config
        shared: dict[str, Any] = dict(
            subset_names=cfg.subset_names,
            context_length=cfg.context_length,
            prediction_length=cfg.prediction_length,
            val_frac=cfg.val_frac,
            max_windows_per_series=cfg.max_windows_per_series,
            max_series_per_subset=cfg.max_series_per_subset,
            streaming=cfg.streaming,
            cache_dir=cfg.cache_dir,
            seed=cfg.seed,
        )
        if stage in (None, "fit"):
            self._train_dataset = GiftEvalDataset(split="train", **shared)
        if stage in (None, "fit", "validate"):
            self._val_dataset = GiftEvalDataset(split="val", **shared)

    def train_dataloader(self) -> DataLoader[dict[str, Any]]:
        """Return the training DataLoader."""
        if self._train_dataset is None:
            self.setup("fit")
        assert self._train_dataset is not None
        return DataLoader(
            self._train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader[dict[str, Any]]:
        """Return the validation DataLoader."""
        if self._val_dataset is None:
            self.setup("validate")
        assert self._val_dataset is not None
        return DataLoader(
            self._val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
