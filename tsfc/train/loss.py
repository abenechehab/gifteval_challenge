"""Differentiable loss functions for training the mixture model."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tsfc.data.transforms import freq_to_period


def mae_loss(
    forecast: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean Absolute Error loss.

    Parameters
    ----------
    forecast : Tensor ``(B, V, H)``
    target   : Tensor ``(B, V, H)``
    mask     : bool Tensor ``(B, H)`` | None  — True where observed.
    """
    err = (forecast - target).abs()  # (B, V, H)
    if mask is not None:
        # Expand mask to (B, 1, H) so it broadcasts over V
        m = mask.unsqueeze(1).float()
        return (err * m).sum() / (m.sum() * err.shape[1] + 1e-8)
    return err.mean()


def mse_loss(
    forecast: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean Squared Error loss.

    Parameters
    ----------
    forecast : Tensor ``(B, V, H)``
    target   : Tensor ``(B, V, H)``
    mask     : bool Tensor ``(B, H)`` | None
    """
    err = F.mse_loss(forecast, target, reduction="none")  # (B, V, H)
    if mask is not None:
        m = mask.unsqueeze(1).float()
        return (err * m).sum() / (m.sum() * err.shape[1] + 1e-8)
    return err.mean()


def mase_loss(
    forecast: torch.Tensor,
    target: torch.Tensor,
    context: torch.Tensor,
    freqs: list[str],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MASE loss (differentiable, scale-free).

    Scale denominator is computed from the context naïve seasonal forecast
    and treated as a constant (no gradient flows through it).

    Parameters
    ----------
    forecast : Tensor ``(B, V, H)``
    target   : Tensor ``(B, V, H)``
    context  : Tensor ``(B, V, T)``
    freqs    : list[str] of length B — pandas frequency per sample.
    mask     : bool Tensor ``(B, H)`` | None
    """
    B, V, _ = forecast.shape
    abs_err = (forecast - target).abs()  # (B, V, H)

    # Compute per-sample MASE denominator from context (no grad)
    denominators: list[torch.Tensor] = []
    for b in range(B):
        period = freq_to_period(freqs[b])
        ctx_b = context[b]  # (V, T)
        T = ctx_b.shape[-1]
        if period >= T:
            period = 1
        naive_diff = (ctx_b[:, period:] - ctx_b[:, :-period]).abs()  # (V, T-period)
        denom = naive_diff.mean().detach() + 1e-8  # scalar
        denominators.append(denom)

    denom_t = torch.stack(denominators).view(B, 1, 1)  # (B, 1, 1)
    scaled_err = abs_err / denom_t  # (B, V, H)

    if mask is not None:
        m = mask.unsqueeze(1).float()
        return (scaled_err * m).sum() / (m.sum() * V + 1e-8)
    return scaled_err.mean()


def crps_loss(
    quantile_preds: dict[float, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pinball / quantile loss summed over quantile levels (approximates CRPS).

    Parameters
    ----------
    quantile_preds : dict level → Tensor ``(B, V, H)``
    target         : Tensor ``(B, V, H)``
    mask           : bool Tensor ``(B, H)`` | None
    """
    losses: list[torch.Tensor] = []
    for q, yhat in quantile_preds.items():
        diff = target - yhat
        ql = torch.max(q * diff, (q - 1.0) * diff)  # (B, V, H)
        if mask is not None:
            m = mask.unsqueeze(1).float()
            losses.append((ql * m).sum() / (m.sum() * ql.shape[1] + 1e-8))
        else:
            losses.append(ql.mean())
    return torch.stack(losses).mean()
