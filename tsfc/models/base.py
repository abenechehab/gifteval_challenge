"""Abstract base class and result dataclass shared by all model wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ForecastResult:
    """Standardised forecast output returned by every model wrapper.

    Attributes
    ----------
    point_forecast : ndarray of shape ``(n_variates, horizon)``
        Median / mean point forecast.
    quantile_forecasts : dict[float, ndarray] | None
        Optional quantile forecasts keyed by quantile level (e.g. 0.1, 0.5, 0.9).
        Each value has shape ``(n_variates, horizon)``.
    """

    point_forecast: np.ndarray[Any, np.dtype[np.floating[Any]]]
    quantile_forecasts: dict[float, np.ndarray[Any, np.dtype[np.floating[Any]]]] | None = field(default=None)


class ForecastingModel(ABC):
    """Abstract base class for all forecasting model wrappers.

    Subclasses must define a class-level ``name`` attribute and implement
    :meth:`predict`.  Lazy loading is strongly recommended: do not load
    model weights in ``__init__`` unless they are required immediately.
    """

    name: str  # model identifier, e.g. "chronos2", "timesfm25", "moirai2"

    @abstractmethod
    def predict(
        self,
        context: np.ndarray[Any, np.dtype[np.floating[Any]]],
        prediction_length: int,
        freq: str,
        past_covariates: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None,
    ) -> ForecastResult:
        """Run zero-shot inference and return a :class:`ForecastResult`.

        Parameters
        ----------
        context : ndarray of shape ``(n_variates, context_length)``
            NaN-free context window (caller must pre-fill NaNs).
        prediction_length : int
            Number of future steps to forecast.
        freq : str
            Pandas-compatible frequency string (e.g. ``"h"``, ``"D"``).
        past_covariates : ndarray of shape ``(n_cov, context_length)`` | None
            Optional past-only covariate channels.

        Returns
        -------
        ForecastResult
            All values in the **original** (un-normalised) scale.
        """

    def batch_predict(
        self,
        contexts: list[np.ndarray[Any, np.dtype[np.floating[Any]]]],
        prediction_lengths: list[int],
        freqs: list[str],
    ) -> list[ForecastResult]:
        """Predict for a list of items.

        Default implementation loops over items.  Override for efficiency.
        """
        return [
            self.predict(ctx, pl, fr)
            for ctx, pl, fr in zip(contexts, prediction_lengths, freqs, strict=True)
        ]
