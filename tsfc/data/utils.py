"""Miscellaneous helpers: NaN filling and frequency string normalization."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def normalize_freq(raw_freq: str) -> str:
    """Return the canonical pandas frequency string for *raw_freq*.

    Examples
    --------
    >>> normalize_freq("30T")
    '30min'
    >>> normalize_freq("H")
    'h'
    """
    offset = pd.tseries.frequencies.to_offset(raw_freq)
    if offset is None:
        raise ValueError(f"Cannot parse frequency string: {raw_freq!r}")
    return offset.freqstr


def _fill_1d(arr: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> tuple[np.ndarray[Any, np.dtype[np.floating[Any]]], np.ndarray[Any, np.dtype[np.floating[Any]]]]:
    """Linear-interpolation fill for a 1-D array.

    Returns
    -------
    filled : ndarray  — NaN replaced by linear interpolation.
    mask   : bool ndarray  — True where the original value was observed.
    """
    mask = ~np.isnan(arr)
    filled = arr.copy()
    if mask.all():
        return filled, mask
    if not mask.any():
        logger.warning("Encountered an all-NaN series; filling with zeros.")
        filled[:] = 0.0
        return filled, mask
    xs = np.arange(arr.shape[0])
    filled = np.interp(xs, xs[mask], arr[mask])
    return filled, mask


def fill_nan_linear(series: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> tuple[np.ndarray[Any, np.dtype[np.floating[Any]]], np.ndarray[Any, np.dtype[np.floating[Any]]]]:
    """Fill NaN values with linear interpolation for 1-D or 2-D arrays.

    Parameters
    ----------
    series : ndarray of shape ``(T,)`` or ``(n_variates, T)``

    Returns
    -------
    filled : same shape as *series*, NaN replaced.
    mask   : bool array, same shape — True where values were observed.
    """
    if series.ndim == 1:
        return _fill_1d(series)

    n = series.shape[0]
    filled = np.empty_like(series)
    mask = np.empty_like(series, dtype=bool)
    for i in range(n):
        filled[i], mask[i] = _fill_1d(series[i])
    return filled, mask


def nan_fraction(arr: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> float:
    """Return the fraction of NaN values in *arr*."""
    return float(np.isnan(arr).mean())
