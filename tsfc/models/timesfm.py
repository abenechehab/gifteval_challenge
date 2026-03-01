"""TimesFM-2.5 model wrapper.

Install: ``uv add timesfm>=1.0.0``
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from tsfc.models.base import ForecastingModel, ForecastResult
from tsfc.models.registry import register

logger = logging.getLogger(__name__)

# Quantile levels emitted by TimesFM (continuous quantile head: 9 quantiles)
_QUANTILE_LEVELS: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@register("timesfm25")
class TimesFMModel(ForecastingModel):
    """Zero-shot wrapper around ``timesfm.TimesFM_2p5_200M_torch``.

    TimesFM is univariate-only; multivariate series are processed variate-by-variate.
    The model has built-in input normalisation (``normalize_inputs=True``), so the
    caller must **not** pre-normalise the context.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    max_context : int
        Maximum context length (series are clipped to this length).
    max_horizon : int
        Maximum prediction horizon the model supports.
    """

    name = "timesfm25"

    def __init__(
        self,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        max_context: int = 1024,
        max_horizon: int = 256,
        device: str = "cuda",
    ) -> None:
        self.model_id = model_id
        self.max_context = max_context
        self.max_horizon = max_horizon
        self.device = device
        self._model: Any = None  # loaded lazily

    def _load(self) -> None:
        """Lazy-load TimesFM (runs once)."""
        if self._model is not None:
            return

        import timesfm
        import torch

        torch.set_float32_matmul_precision("high")
        logger.info("Loading TimesFM-2.5 from %r …", self.model_id)
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_id)
        model.compile(
            timesfm.ForecastConfig(
                max_context=self.max_context,
                max_horizon=self.max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        self._model = model

    def predict(
        self,
        context: np.ndarray[Any, np.dtype[np.floating[Any]]],
        prediction_length: int,
        freq: str,
        past_covariates: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    ) -> ForecastResult:
        """Run TimesFM inference for each variate independently.

        Parameters
        ----------
        context : ndarray of shape ``(n_variates, T)``
            Must be NaN-free but should be the **original** (un-normalised) scale —
            TimesFM normalises internally.
        """
        self._load()

        n_variates, T = context.shape

        # Clip to max_context to avoid silent truncation
        clipped = context[:, max(0, T - self.max_context) :]

        # TimesFM expects a list of 1-D arrays (variable length OK)
        inputs = [clipped[v] for v in range(n_variates)]

        point_raw, q_raw = self._model.forecast(
            horizon=prediction_length,
            inputs=inputs,
        )
        # point_raw: (n_variates, prediction_length)
        # q_raw:     (n_variates, prediction_length, n_quantiles)

        point = np.asarray(point_raw)  # (n_variates, H)
        q_arr = np.asarray(q_raw)  # (n_variates, H, n_q)

        quantile_forecasts: dict[float, np.ndarray[Any, np.dtype[np.floating[Any]]]] = {
            q: q_arr[:, :, i] for i, q in enumerate(_QUANTILE_LEVELS)
        }

        return ForecastResult(
            point_forecast=point,
            quantile_forecasts=quantile_forecasts,
        )
