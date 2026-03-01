"""Moirai-2 model wrapper (Salesforce uni2ts).

Install:
    uv add gluonts>=0.14.3,<0.15
    uv add uni2ts>=1.2.0
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


@register("moirai2")
class MoiraiModel(ForecastingModel):
    """Zero-shot wrapper around ``uni2ts.model.moirai2.Moirai2Forecast``.

    The underlying ``Moirai2Module`` weights are loaded once and reused across
    calls.  A fresh ``Moirai2Forecast`` predictor is constructed per call
    because Moirai's ``prediction_length`` / ``context_length`` are baked into
    the forecast object.

    Parameters
    ----------
    size : str
        Model size: ``"small"`` (14 M), ``"base"`` (91 M), or ``"large"`` (311 M).
    context_length : int
        Maximum context length passed to Moirai (series are clipped if longer).
    num_samples : int
        Number of probabilistic samples drawn per forecast.
    device : str
        Torch device (``"cuda"`` or ``"cpu"``).
    """

    name = "moirai2"

    def __init__(
        self,
        size: str = "small",
        context_length: int = 500,
        num_samples: int = 100,
        device: str = "cuda",
    ) -> None:
        self.size = size
        self.context_length = context_length
        self.num_samples = num_samples
        self.device = device
        self._module: Any = None  # uni2ts.Moirai2Module — loaded lazily

    def _load(self) -> None:
        """Lazy-load the Moirai2Module weights (runs once)."""
        if self._module is not None:
            return
        try:
            from uni2ts.model.moirai2 import Moirai2Module  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("uni2ts is not installed. Run: uv add uni2ts>=1.2.0") from exc

        repo = f"Salesforce/moirai-2.0-R-{self.size}"
        logger.info("Loading Moirai-2 (%s) from %r …", self.size, repo)
        self._module = Moirai2Module.from_pretrained(repo)

    def predict(
        self,
        context: np.ndarray[Any, np.dtype[np.floating[Any]]],
        prediction_length: int,
        freq: str,
        past_covariates: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    ) -> ForecastResult:
        """Run Moirai-2 inference.

        Parameters
        ----------
        context : ndarray of shape ``(n_variates, T)``
            NaN-free context.
        prediction_length : int
        freq : str
            Pandas frequency string.
        past_covariates : ndarray of shape ``(n_cov, T)`` | None

        Returns
        -------
        ForecastResult
            ``point_forecast`` is the per-variate median of ``num_samples`` samples.
        """
        self._load()

        try:
            from gluonts.dataset.common import ListDataset  # type: ignore[import-untyped]
            from uni2ts.model.moirai2 import Moirai2Forecast  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("gluonts and uni2ts must both be installed.") from exc

        n_variates, T = context.shape
        past_cov_dim = past_covariates.shape[0] if past_covariates is not None else 0
        ctx_len = min(T, self.context_length)

        model = Moirai2Forecast(
            module=self._module,
            prediction_length=prediction_length,
            context_length=ctx_len,
            target_dim=n_variates,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=past_cov_dim,
            # patch_size="auto",
            # num_samples=self.num_samples,
        ).to(self.device)

        # Build a GluonTS ListDataset entry
        start = pd.Timestamp("2000-01-01")
        target_data: Any = context[0] if n_variates == 1 else context  # (T,) or (V, T)

        entry: dict[str, Any] = {
            "start": start,
            "target": target_data,
            "item_id": "series",
            "freq": freq,
        }
        if past_covariates is not None:
            entry["past_feat_dynamic_real"] = past_covariates

        test_ds = ListDataset([entry], freq=freq, one_dim_target=(n_variates == 1))
        predictor = model.create_predictor(batch_size=1)
        forecasts = list(predictor.predict(test_ds))

        # samples: (num_samples, prediction_length) for univariate,
        #          (num_samples, n_variates, prediction_length) for multivariate
        # raw_samples: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.asarray(forecasts[0].samples)
        raw_samples: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.asarray(
            forecasts[0].forecast_array
        )

        if raw_samples.ndim == 2:
            # Univariate: expand to (1, num_samples, H) → (V=1, num_samples, H)
            raw_samples = raw_samples[np.newaxis, :, :]  # (1, S, H)
        else:
            # Multivariate: (S, V, H) → (V, S, H)
            raw_samples = raw_samples.transpose(1, 0, 2)

        # (V, H)
        point = np.median(raw_samples, axis=1)

        quantile_forecasts: dict[float, np.ndarray[Any, np.dtype[np.floating[Any]]]] = {
            q: np.quantile(raw_samples, q, axis=1) for q in _QUANTILE_LEVELS
        }

        return ForecastResult(
            point_forecast=point,
            quantile_forecasts=quantile_forecasts,
        )
