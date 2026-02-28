"""Forecasting model wrappers and registry."""

from tsfc.models.base import ForecastingModel, ForecastResult
from tsfc.models.registry import list_models, load_model, register

# Import wrappers so their @register(...) decorators execute and populate the
# registry. Each import is guarded so that a missing optional dependency
# (chronos-forecasting, timesfm, uni2ts) does not break the rest of the package.
try:
    import tsfc.models.chronos2 as _  # noqa: F401
except ImportError:
    pass

try:
    import tsfc.models.timesfm as _  # noqa: F401
except ImportError:
    pass

try:
    import tsfc.models.moirai as _  # noqa: F401
except ImportError:
    pass

__all__ = [
    "ForecastResult",
    "ForecastingModel",
    "register",
    "load_model",
    "list_models",
]
