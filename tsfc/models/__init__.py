"""Forecasting model wrappers and registry."""

from tsfc.models.base import ForecastingModel, ForecastResult
from tsfc.models.registry import list_models, load_model, register

__all__ = [
    "ForecastResult",
    "ForecastingModel",
    "register",
    "load_model",
    "list_models",
]
