"""Weighted mixture of frozen model predictions, with optional boosting head.

Provides three composable components:

* :class:`MetaWeighter` – lightweight MLP for per-instance mixture weights.
* :class:`BoostingForecaster` – small patch-based transformer that adds a
  learnable residual correction on top of the weighted blend.
* :class:`WeightedMixture` – combines the above and exposes a single
  ``forward()`` interface.
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


class BoostingForecaster(nn.Module):
    """Channel-independent patch-based transformer for residual boosting.

    The normalised context is divided into non-overlapping patches which are
    linearly projected and encoded by a small TransformerEncoder.  All patch
    representations are flattened and projected to ``prediction_length`` future
    steps.  The output is an **additive** correction on top of the weighted
    mixture blend.

    The output head is zero-initialised so that training starts from the pure
    mixture baseline and the booster gradually learns to correct residuals.

    Parameters
    ----------
    context_length    : int   — number of input time steps.
    prediction_length : int   — number of output steps to produce.
    patch_size        : int   — time steps per patch (context is truncated to a
                                multiple of *patch_size*).
    d_model           : int   — transformer model dimension.
    n_heads           : int   — number of self-attention heads.
    n_layers          : int   — number of TransformerEncoder layers.
    dropout           : float — dropout probability inside the encoder.
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        patch_size: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.prediction_length = prediction_length
        self.patch_size = patch_size
        self.n_patches = context_length // patch_size  # truncate to full patches

        # Per-patch linear projection: patch_size → d_model
        self.patch_proj = nn.Linear(patch_size, d_model)

        # Learnable positional embedding (one vector per patch)
        self.pos_emb = nn.Embedding(self.n_patches, d_model)

        # Small transformer encoder (Pre-LayerNorm for training stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Flatten all patch representations → prediction_length
        self.head = nn.Linear(self.n_patches * d_model, prediction_length)

        # Zero-init head so the booster starts as a no-op
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Compute additive forecast correction.

        Parameters
        ----------
        context : Tensor ``(B, V, ctx_len)`` — normalised context.

        Returns
        -------
        Tensor ``(B, V, prediction_length)``
        """
        B, V, _ = context.shape

        # Truncate to a multiple of patch_size
        T_use = self.n_patches * self.patch_size
        x = context[:, :, :T_use]  # (B, V, T_use)

        # Channel-independent: fold variates into the batch dimension
        x = x.reshape(B * V, self.n_patches, self.patch_size)  # (B*V, n_patches, patch_size)

        # Patch embedding + positional encoding
        x = self.patch_proj(x)  # (B*V, n_patches, d_model)
        pos_ids = torch.arange(self.n_patches, device=x.device)
        x = x + self.pos_emb(pos_ids)  # (B*V, n_patches, d_model)

        # Transformer encoding
        x = self.encoder(x)  # (B*V, n_patches, d_model)

        # Flatten and project
        x = x.reshape(B * V, -1)   # (B*V, n_patches * d_model)
        x = self.head(x)            # (B*V, prediction_length)

        return x.reshape(B, V, self.prediction_length)  # (B, V, prediction_length)


class WeightedMixture(nn.Module):
    """Learnable convex combination of frozen backbone predictions, with
    an optional patch-transformer boosting head.

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
    boosting : bool
        If ``True``, a :class:`BoostingForecaster` adds a learnable residual
        correction to the weighted blend.  The raw normalised context must then
        be passed as ``context`` to :meth:`forward`.
    context_length : int
        Length of the input context window; required when ``boosting=True``.
    prediction_length : int
        Fixed prediction horizon for the booster head; required when
        ``boosting=True``.  For batches whose actual horizon ``H`` differs,
        the booster output is truncated (``H < prediction_length``) or zero-
        padded (``H > prediction_length``) to match.
    boosting_patch_size : int
        Patch size for :class:`BoostingForecaster`.
    boosting_d_model : int
        Transformer model dimension for :class:`BoostingForecaster`.
    boosting_n_heads : int
        Number of attention heads for :class:`BoostingForecaster`.
    boosting_n_layers : int
        Number of encoder layers for :class:`BoostingForecaster`.
    boosting_dropout : float
        Dropout probability for :class:`BoostingForecaster`.
    """

    def __init__(
        self,
        model_names: list[str],
        adaptive: bool = False,
        context_feature_dim: int = 5,
        hidden_dim: int = 64,
        boosting: bool = False,
        context_length: int = 512,
        prediction_length: int = 24,
        boosting_patch_size: int = 16,
        boosting_d_model: int = 64,
        boosting_n_heads: int = 4,
        boosting_n_layers: int = 2,
        boosting_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_names = model_names
        self.n_models = len(model_names)
        self.adaptive = adaptive
        self.boosting = boosting

        if adaptive:
            self.meta_net = MetaWeighter(context_feature_dim, self.n_models, hidden_dim)
        else:
            # Global learnable log-weights (initialised to uniform)
            self.log_weights = nn.Parameter(torch.zeros(self.n_models))

        if boosting:
            self.booster = BoostingForecaster(
                context_length=context_length,
                prediction_length=prediction_length,
                patch_size=boosting_patch_size,
                d_model=boosting_d_model,
                n_heads=boosting_n_heads,
                n_layers=boosting_n_layers,
                dropout=boosting_dropout,
            )

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
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the (optionally boosted) weighted mixture forecast.

        Parameters
        ----------
        model_predictions : dict mapping model name → Tensor ``(B, V, H)``
            One entry per model in ``self.model_names``.
        context_features : Tensor ``(B, n_features)`` | None
            Required when ``adaptive=True``.
        context : Tensor ``(B, V, ctx_len)`` | None
            Normalised context window. Required when ``boosting=True``.

        Returns
        -------
        Tensor of shape ``(B, V, H)`` — the (boosted) weighted mixture forecast.
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
        blended = (preds * w).sum(dim=1)  # (B, V, H)

        if self.boosting:
            if context is None:
                raise ValueError("context must be provided when boosting=True")
            boost = self.booster(context)  # (B, V, prediction_length)
            H = blended.shape[-1]
            P = boost.shape[-1]
            if P >= H:
                # Booster trained for longer horizon: take first H steps
                blended = blended + boost[:, :, :H]
            else:
                # Booster trained for shorter horizon: zero-pad remainder
                pad = blended.new_zeros(B, blended.shape[1], H - P)
                blended = blended + torch.cat([boost, pad], dim=-1)

        return blended
