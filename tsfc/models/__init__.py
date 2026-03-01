"""Forecasting model wrappers and registry."""

# Import wrappers so their @register(...) decorators execute and populate the
# registry.
import tsfc.models.chronos2 as _
import tsfc.models.moirai as _  # noqa: F811
import tsfc.models.timesfm as _  # noqa: F401,F811
from tsfc.models.base import ForecastingModel, ForecastResult
from tsfc.models.registry import list_models, load_model, register

__all__ = [
    "ForecastResult",
    "ForecastingModel",
    "register",
    "load_model",
    "list_models",
]
