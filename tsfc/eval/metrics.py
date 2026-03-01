"""Evaluation metrics: MASE, CRPS, WQL, and dataset-level aggregation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from tsfc.data.transforms import freq_to_period

if TYPE_CHECKING:
    from tsfc.data.gifteval import GiftEvalDataset
    from tsfc.models.base import ForecastingModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metrics (all operate on numpy arrays)
# ---------------------------------------------------------------------------


def mae(forecast: np.ndarray[Any, np.dtype[np.floating[Any]]], target: np.ndarray[Any, np.dtype[np.floating[Any]]], mask: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None) -> float:
    """Mean Absolute Error, optionally masked.

    Parameters
    ----------
    forecast : ndarray ``(n_variates, H)``
    target   : ndarray ``(n_variates, H)``
    mask     : bool ndarray ``(H,)`` | None  — True where observed.
    """
    err = np.abs(forecast - target)  # (V, H)
    if mask is not None:
        err = err[:, mask]
    return float(err.mean())


def mase(
    forecast: np.ndarray[Any, np.dtype[np.floating[Any]]],
    target: np.ndarray[Any, np.dtype[np.floating[Any]]],
    context: np.ndarray[Any, np.dtype[np.floating[Any]]],
    freq: str,
    mask: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
) -> float:
    """Mean Absolute Scaled Error (MASE).

    Scale is the in-sample naïve seasonal MAE computed from *context*.

    Parameters
    ----------
    forecast : ndarray ``(n_variates, H)``  — point forecast, original scale.
    target   : ndarray ``(n_variates, H)``  — ground truth, original scale.
    context  : ndarray ``(n_variates, T)``  — historical context, original scale.
    freq     : str  — pandas frequency string (determines seasonal period).
    mask     : bool ndarray ``(H,)`` | None  — True where observed.
    """
    period = freq_to_period(freq)
    _, T = context.shape

    if period >= T:
        # Fall back to period=1 if context is too short
        period = 1

    # Naïve seasonal denominator: mean |x_t - x_{t-period}| over context
    naive_err = np.abs(context[:, period:] - context[:, :-period])  # (V, T-period)
    scale = naive_err.mean(axis=-1, keepdims=True) + 1e-8  # (V, 1)

    abs_err = np.abs(forecast - target)  # (V, H)
    if mask is not None:
        abs_err = abs_err[:, mask]
        scale_val = scale.mean()
    else:
        scale_val = scale.mean()

    return float((abs_err / scale_val).mean())


def crps_from_samples(
    samples: np.ndarray[Any, np.dtype[np.floating[Any]]],
    target: np.ndarray[Any, np.dtype[np.floating[Any]]],
    mask: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
) -> float:
    """Continuous Ranked Probability Score estimated from samples.

    Uses the energy-form estimator:
        CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|

    Parameters
    ----------
    samples : ndarray ``(n_samples, n_variates, H)``
    target  : ndarray ``(n_variates, H)``
    mask    : bool ndarray ``(H,)`` | None
    """
    n_samples = samples.shape[0]

    # E|X - y| term: (n_samples, V, H) → (V, H)
    spread = np.abs(samples - target[np.newaxis]).mean(axis=0)

    # E|X - X'| term via pair sampling (O(n^2) for small n_samples)
    # Equivalent to 2 * mean(|samples_i - samples_j|) for i<j
    idx_i, idx_j = np.triu_indices(n_samples, k=1)
    pair_diffs = np.abs(samples[idx_i] - samples[idx_j]).mean(axis=0)  # (V, H)

    crps_grid = spread - 0.5 * pair_diffs  # (V, H)

    if mask is not None:
        crps_grid = crps_grid[:, mask]
    return float(crps_grid.mean())


def wql(
    quantile_forecasts: dict[float, np.ndarray[Any, np.dtype[np.floating[Any]]]],
    target: np.ndarray[Any, np.dtype[np.floating[Any]]],
    quantile_levels: list[float] | None = None,
    mask: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
) -> float:
    """Weighted Quantile Loss (WQL).

    WQL = (2 / |Q|) * sum_q  mean_t  q * max(y - yhat_q, 0) + (1-q) * max(yhat_q - y, 0)

    Parameters
    ----------
    quantile_forecasts : dict level → ndarray ``(n_variates, H)``
    target             : ndarray ``(n_variates, H)``
    quantile_levels    : list of quantile levels to use (default: all keys in dict).
    mask               : bool ndarray ``(H,)`` | None
    """
    levels = quantile_levels if quantile_levels is not None else sorted(quantile_forecasts)
    losses: list[float] = []

    for q in levels:
        yhat = quantile_forecasts[q]  # (V, H)
        diff = target - yhat
        ql = np.maximum(q * diff, (q - 1.0) * diff)  # (V, H)
        if mask is not None:
            ql = ql[:, mask]
        losses.append(float(ql.mean()))

    return float(2.0 * np.mean(losses))


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------


def evaluate_dataset(
    model: ForecastingModel,
    dataset: GiftEvalDataset,
    metric: str = "mase",
    max_items: int | None = None,
) -> dict[str, Any]:
    """Evaluate *model* on every item of *dataset* and return aggregated metrics.

    Parameters
    ----------
    model     : :class:`~tsfc.models.base.ForecastingModel`
    dataset   : :class:`~tsfc.data.gifteval.GiftEvalDataset`
    metric    : ``"mase"``, ``"mae"``, or ``"wql"``
    max_items : If set, evaluate only the first *max_items* windows.

    Returns
    -------
    dict with keys ``"mean"``, ``"std"``, ``"n_items"``, ``"per_item"`` (list of floats).
    """
    from tsfc.data.transforms import denormalize

    # metric_fn: Callable[..., float]
    if metric == "mase" or metric == "mae" or metric == "wql":
        pass
    else:
        raise ValueError(f"Unknown metric {metric!r}. Choose from: mase, mae, wql")

    scores: list[float] = []
    n = min(len(dataset), max_items) if max_items is not None else len(dataset)

    for i in range(n):
        item = dataset[i]

        ctx_norm: np.ndarray[Any, np.dtype[np.floating[Any]]] = item["context"].numpy()  # (V, ctx_len)
        tgt_raw: np.ndarray[Any, np.dtype[np.floating[Any]]] = item["target_raw"].numpy()  # (V, H)
        loc: np.ndarray[Any, np.dtype[np.floating[Any]]] = item["loc"].numpy()  # (V, 1)
        scale: np.ndarray[Any, np.dtype[np.floating[Any]]] = item["scale"].numpy()  # (V, 1)
        mask: np.ndarray[Any, np.dtype[np.floating[Any]]] = item["mask"].numpy()  # (H,)
        freq: str = item["freq"]
        pred_len: int = item["prediction_length"]

        # Denormalize context for passing to model
        ctx_raw = denormalize(ctx_norm, loc, scale)

        try:
            result = model.predict(ctx_raw, pred_len, freq)
        except Exception:
            logger.exception("Prediction failed for item %d; skipping.", i)
            continue

        point = result.point_forecast  # (V, H)

        if metric == "mase":
            score = mase(point, tgt_raw, ctx_raw, freq, mask)
        elif metric == "mae":
            score = mae(point, tgt_raw, mask)
        elif metric == "wql":
            if result.quantile_forecasts is None:
                logger.warning("No quantile forecasts for item %d; skipping WQL.", i)
                continue
            score = wql(result.quantile_forecasts, tgt_raw, mask=mask)
        else:
            score = 0.0

        scores.append(score)

    arr = np.array(scores)
    return {
        "mean": float(arr.mean()) if len(arr) > 0 else float("nan"),
        "std": float(arr.std()) if len(arr) > 0 else float("nan"),
        "n_items": len(scores),
        "per_item": scores,
    }
