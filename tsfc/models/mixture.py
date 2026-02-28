"""Weighted mixture of frozen model predictions.

Provides both a **static** (global scalar weights) and an **adaptive**
(instance-conditioned MLP weights) variant of the mixture.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MetaWeighter(nn.Module):
    """Lightweight MLP that predicts per-instance mixture weights.

    Parameters
    ----------
    n_features : int
        Dimension of the input context-feature vector.
    n_models : int
        Number of models to weight.
    hidden_dim : int
        Hidden layer width.
    """

    def __init__(self, n_features: int, n_models: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_models),
        )

    def forward(self, context_features: torch.Tensor) -> torch.Tensor:
        """Return softmax-normalised weights.

        Parameters
        ----------
        context_features : Tensor of shape ``(B, n_features)``

        Returns
        -------
        Tensor of shape ``(B, n_models)``
        """
        return torch.softmax(self.mlp(context_features), dim=-1)


class WeightedMixture(nn.Module):
    """Learnable convex combination of frozen backbone predictions.

    Parameters
    ----------
    model_names : list[str]
        Ordered list of model identifiers; must match keys in the predictions
        dict passed to :meth:`forward`.
    adaptive : bool
        If ``True``, a :class:`MetaWeighter` MLP produces per-instance weights.
        If ``False``, a single global log-weight vector is learned.
    context_feature_dim : int
        Input dimension for the meta-weighter (used only when ``adaptive=True``).
    hidden_dim : int
        Hidden dimension of the meta-weighter MLP.
    """

    def __init__(
        self,
        model_names: list[str],
        adaptive: bool = False,
        context_feature_dim: int = 5,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.model_names = model_names
        self.n_models = len(model_names)
        self.adaptive = adaptive

        if adaptive:
            self.meta_net = MetaWeighter(context_feature_dim, self.n_models, hidden_dim)
        else:
            # Global learnable log-weights (initialised to uniform)
            self.log_weights = nn.Parameter(torch.zeros(self.n_models))

    def get_weights(
        self,
        context_features: torch.Tensor | None = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Return mixture weights.

        Returns
        -------
        Tensor of shape ``(B, n_models)``
        """
        if self.adaptive:
            if context_features is None:
                raise ValueError("context_features must be provided when adaptive=True")
            return self.meta_net(context_features)  # (B, n_models)
        # Broadcast global weights to batch
        w = torch.softmax(self.log_weights, dim=0)  # (n_models,)
        return w.unsqueeze(0).expand(batch_size, -1)  # (B, n_models)

    def forward(
        self,
        model_predictions: dict[str, torch.Tensor],
        context_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the weighted mixture forecast.

        Parameters
        ----------
        model_predictions : dict mapping model name → Tensor ``(B, V, H)``
            One entry per model in ``self.model_names``.
        context_features : Tensor ``(B, n_features)`` | None
            Required when ``adaptive=True``.

        Returns
        -------
        Tensor of shape ``(B, V, H)`` — the weighted mixture forecast.
        """
        # Stack: (B, n_models, V, H)
        preds = torch.stack(
            [model_predictions[name] for name in self.model_names],
            dim=1,
        )
        B = preds.shape[0]
        w = self.get_weights(context_features, batch_size=B)  # (B, n_models)
        # Reshape for broadcasting: (B, n_models, 1, 1)
        w = w.unsqueeze(-1).unsqueeze(-1)
        return (preds * w).sum(dim=1)  # (B, V, H)
