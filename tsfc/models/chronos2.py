"""Chronos-2 model wrapper.

Install: ``uv add chronos-forecasting>=2.1.0``
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from tsfc.models.base import ForecastingModel, ForecastResult
from tsfc.models.registry import register

logger = logging.getLogger(__name__)

_QUANTILE_LEVELS: list[float] = [0.1, 0.5, 0.9]


@register("chronos2")
class Chronos2Model(ForecastingModel):
    """Zero-shot wrapper around ``chronos.Chronos2Pipeline``.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier (default ``"amazon/chronos-2"``).
    device : str
        Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
    dtype : str
        Torch dtype for the model weights (``"bfloat16"`` recommended).
    num_samples : int
        Number of forecast samples used to compute quantiles.
    """

    name = "chronos2"

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        device: str = "cuda",
        dtype: str = "bfloat16",
        num_samples: int = 20,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.num_samples = num_samples
        self._pipeline: Any = None  # loaded lazily

    def _load(self) -> None:
        """Lazy-load the Chronos-2 pipeline (runs once)."""
        if self._pipeline is not None:
            return
        try:
            import chronos  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "chronos-forecasting is not installed. "
                "Run: uv add chronos-forecasting>=2.1.0"
            ) from exc
        logger.info("Loading Chronos-2 pipeline from %r onto %r …", self.model_id, self.device)
        self._pipeline = chronos.Chronos2Pipeline.from_pretrained(
            self.model_id,
            device_map=self.device,
            torch_dtype=self.dtype,
        )

    def predict(
        self,
        context: np.ndarray,
        prediction_length: int,
        freq: str,
        past_covariates: np.ndarray | None = None,
    ) -> ForecastResult:
        """Run Chronos-2 inference.

        Parameters
        ----------
        context : ndarray of shape ``(n_variates, T)``
        prediction_length : int
        freq : str
            Pandas frequency string — used to build the timestamp index.
        past_covariates : ndarray of shape ``(n_cov, T)`` | None

        Returns
        -------
        ForecastResult
        """
        self._load()

        n_variates, T = context.shape
        dates = pd.date_range("2000-01-01", periods=T, freq=freq)

        # Build context DataFrame: one group per variate
        rows: list[dict[str, Any]] = []
        for v in range(n_variates):
            for t, d in enumerate(dates):
                rows.append({"id": f"v{v}", "ds": d, "target": float(context[v, t])})
        context_df = pd.DataFrame(rows)

        cov_cols: list[str] = []
        if past_covariates is not None:
            n_cov = past_covariates.shape[0]
            for c in range(n_cov):
                col = f"cov{c}"
                cov_cols.append(col)
                for v in range(n_variates):
                    mask = context_df["id"] == f"v{v}"
                    context_df.loc[mask, col] = past_covariates[c]  # type: ignore[index]

        predict_kwargs: dict[str, Any] = dict(
            prediction_length=prediction_length,
            quantile_levels=_QUANTILE_LEVELS,
            id_column="id",
            timestamp_column="ds",
            target_columns=["target"],
        )
        if cov_cols:
            predict_kwargs["past_feat_dynamic_real_columns"] = cov_cols

        pred_df: Any = self._pipeline.predict_df(context_df, **predict_kwargs)

        point = np.stack(
            [pred_df[pred_df["id"] == f"v{v}"]["0.5"].to_numpy() for v in range(n_variates)]
        )  # (n_variates, prediction_length)

        quantile_forecasts: dict[float, np.ndarray] = {
            q: np.stack(
                [
                    pred_df[pred_df["id"] == f"v{v}"][str(q)].to_numpy()
                    for v in range(n_variates)
                ]
            )
            for q in _QUANTILE_LEVELS
        }

        return ForecastResult(
            point_forecast=point,
            quantile_forecasts=quantile_forecasts,
        )
