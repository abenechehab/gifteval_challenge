"""Global model registry: register, load, and list forecasting models."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from tsfc.models.base import ForecastingModel

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Callable[..., ForecastingModel]] = {}


def register(name: str) -> Callable[[type[ForecastingModel]], type[ForecastingModel]]:
    """Class decorator that registers a :class:`ForecastingModel` subclass.

    Usage
    -----
    .. code-block:: python

        @register("chronos2")
        class Chronos2Model(ForecastingModel):
            ...
    """

    def decorator(cls: type[ForecastingModel]) -> type[ForecastingModel]:
        if name in _REGISTRY:
            logger.warning("Overwriting existing model registration for %r", name)
        _REGISTRY[name] = cls
        return cls

    return decorator


def list_models() -> list[str]:
    """Return sorted list of all registered model names."""
    return sorted(_REGISTRY)


def load_model(name: str, **kwargs: Any) -> ForecastingModel:
    """Instantiate and return the model registered under *name*.

    Parameters
    ----------
    name : str
        Registered model identifier (e.g. ``"chronos2"``).
    **kwargs :
        Forwarded to the model's ``__init__``.

    Raises
    ------
    ValueError
        If *name* is not registered.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available models: {list_models()}")
    logger.info("Loading model %r with kwargs %s", name, kwargs)
    return _REGISTRY[name](**kwargs)
