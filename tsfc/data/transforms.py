"""Normalization, frequency encoding, and context feature extraction."""

from __future__ import annotations

import numpy as np
import torch

from tsfc.data.utils import normalize_freq

# ---------------------------------------------------------------------------
# Frequency look-up tables
# ---------------------------------------------------------------------------

# Canonical freq string → seasonal period (used as MASE denominator)
_FREQ_PERIOD: dict[str, int] = {
    "min": 1440,
    "30min": 48,
    "h": 24,
    "D": 7,
    "W": 52,
    "W-SUN": 52,
    "ME": 12,
    "MS": 12,
    "M": 12,
    "QE": 4,
    "QS": 4,
    "Q": 4,
    "QE-DEC": 4,
    "YE": 1,
    "YS": 1,
    "A": 1,
    "Y": 1,
    "BA": 1,
    "BAS": 1,
}

# Canonical freq string → TimesFM frequency token {0, 1, 2}
# 0 = high-freq (≤daily), 1 = weekly, 2 = low-freq (monthly+)
_FREQ_TFM_TOKEN: dict[str, int] = {
    "min": 0,
    "30min": 0,
    "h": 0,
    "D": 0,
    "W": 1,
    "W-SUN": 1,
    "ME": 2,
    "MS": 2,
    "M": 2,
    "QE": 2,
    "QS": 2,
    "Q": 2,
    "QE-DEC": 2,
    "YE": 2,
    "YS": 2,
    "A": 2,
    "Y": 2,
    "BA": 2,
    "BAS": 2,
}

# Canonical freq string → default prediction horizon (GIFT-Eval convention)
_FREQ_HORIZON: dict[str, int] = {
    "min": 60,
    "30min": 48,
    "h": 24,
    "D": 30,
    "W": 8,
    "W-SUN": 8,
    "ME": 12,
    "MS": 12,
    "M": 12,
    "QE": 4,
    "QS": 4,
    "Q": 4,
    "QE-DEC": 4,
    "YE": 1,
    "YS": 1,
    "A": 1,
    "Y": 1,
    "BA": 1,
    "BAS": 1,
}


def _lookup(table: dict[str, int], freq: str, default: int) -> int:
    """Try exact match then prefix match in *table*."""
    canonical = normalize_freq(freq)
    if canonical in table:
        return table[canonical]
    for key, val in table.items():
        if canonical.startswith(key):
            return val
    return default


def freq_to_period(freq: str) -> int:
    """Return the seasonal period for *freq* (used as MASE denominator).

    Falls back to ``1`` for unknown frequencies.
    """
    return _lookup(_FREQ_PERIOD, freq, 1)


def freq_to_timesfm_token(freq: str) -> int:
    """Return TimesFM frequency token (0=high, 1=medium, 2=low) for *freq*."""
    return _lookup(_FREQ_TFM_TOKEN, freq, 0)


def freq_to_default_horizon(freq: str) -> int:
    """Return the default prediction horizon for *freq* following GIFT-Eval convention."""
    return _lookup(_FREQ_HORIZON, freq, 12)


# ---------------------------------------------------------------------------
# Instance normalization
# ---------------------------------------------------------------------------


def robust_normalize(
    context: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize *context* using robust statistics (median + MAD-based scale).

    Parameters
    ----------
    context : ndarray of shape ``(n_variates, T)``

    Returns
    -------
    normalized : ``(n_variates, T)``  — normalized context.
    loc        : ``(n_variates, 1)``  — per-variate median.
    scale      : ``(n_variates, 1)``  — per-variate MAD-based scale (≥ 1e-8).
    """
    loc = np.median(context, axis=-1, keepdims=True)
    scale = np.maximum(
        np.abs(context - loc).mean(axis=-1, keepdims=True),
        1e-8,
    )
    return (context - loc) / scale, loc, scale


def denormalize(
    forecast: np.ndarray,
    loc: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Reverse instance normalization: ``forecast * scale + loc``."""
    return forecast * scale + loc


# ---------------------------------------------------------------------------
# Context feature extraction (for meta-weighter)
# ---------------------------------------------------------------------------


def extract_context_features(context: torch.Tensor) -> torch.Tensor:
    """Extract five statistical features from *context* for the meta-weighter.

    Parameters
    ----------
    context : Tensor of shape ``(B, n_variates, context_length)``

    Returns
    -------
    features : Tensor of shape ``(B, 5)``
        [mean, std, trend_direction, recent_mean, norm_std]
    """
    x = context[:, 0, :]  # first variate: (B, T)
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True) + 1e-8
    x_norm = (x - mean) / std

    tail = min(8, x_norm.shape[-1])
    features = torch.cat(
        [
            mean,  # overall level
            std,  # volatility
            x_norm[:, -1:] - x_norm[:, 0:1],  # trend direction
            x_norm[:, -tail:].mean(dim=-1, keepdim=True),  # recent mean
            x_norm.std(dim=-1, keepdim=True),  # normalised std
        ],
        dim=-1,
    )
    return features  # (B, 5)
