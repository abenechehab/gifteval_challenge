"""Unit tests for tsfc/data/ utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tsfc.data.transforms import (
    denormalize,
    extract_context_features,
    freq_to_default_horizon,
    freq_to_period,
    freq_to_timesfm_token,
    robust_normalize,
)
from tsfc.data.utils import fill_nan_linear, nan_fraction, normalize_freq

# ---------------------------------------------------------------------------
# tsfc.data.utils
# ---------------------------------------------------------------------------


class TestNormalizeFreq:
    def test_hourly(self) -> None:
        assert normalize_freq("H") == "h"

    def test_30min(self) -> None:
        # Both "30T" and "30min" should normalise to the same value
        assert normalize_freq("30T") == normalize_freq("30min")

    def test_monthly(self) -> None:
        result = normalize_freq("M")
        assert result in {"ME", "MS", "M"}  # varies by pandas version

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_freq("INVALID_FREQ_XYZ")


class TestFillNanLinear:
    def test_no_nan_1d(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        filled, mask = fill_nan_linear(arr)
        np.testing.assert_array_equal(filled, arr)
        assert mask.all()

    def test_interior_nan_1d(self) -> None:
        arr = np.array([1.0, np.nan, 3.0])
        filled, mask = fill_nan_linear(arr)
        assert filled[1] == pytest.approx(2.0)
        assert not mask[1]
        assert mask[0] and mask[2]

    def test_all_nan_1d(self) -> None:
        arr = np.full(5, np.nan)
        filled, mask = fill_nan_linear(arr)
        assert not mask.any()
        np.testing.assert_array_equal(filled, np.zeros(5))

    def test_2d_shape_preserved(self) -> None:
        arr = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
        filled, mask = fill_nan_linear(arr)
        assert filled.shape == arr.shape
        assert mask.shape == arr.shape


class TestNanFraction:
    def test_no_nan(self) -> None:
        assert nan_fraction(np.array([1.0, 2.0, 3.0])) == 0.0

    def test_all_nan(self) -> None:
        assert nan_fraction(np.full(4, np.nan)) == 1.0

    def test_half_nan(self) -> None:
        arr = np.array([1.0, np.nan, 3.0, np.nan])
        assert nan_fraction(arr) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# tsfc.data.transforms
# ---------------------------------------------------------------------------


class TestFreqLookups:
    @pytest.mark.parametrize(
        "freq, expected_period",
        [
            ("H", 24),
            ("D", 7),
            ("W", 52),
            ("M", 12),
            ("Q", 4),
        ],
    )
    def test_freq_to_period(self, freq: str, expected_period: int) -> None:
        assert freq_to_period(freq) == expected_period

    @pytest.mark.parametrize(
        "freq, expected_token",
        [
            ("H", 0),
            ("D", 0),
            ("W", 1),
            ("M", 2),
            ("Q", 2),
        ],
    )
    def test_freq_to_timesfm_token(self, freq: str, expected_token: int) -> None:
        assert freq_to_timesfm_token(freq) == expected_token

    def test_freq_to_default_horizon_hourly(self) -> None:
        assert freq_to_default_horizon("H") == 24

    def test_freq_to_default_horizon_daily(self) -> None:
        assert freq_to_default_horizon("D") == 30


class TestRobustNormalize:
    def test_shape_preserved(self) -> None:
        ctx = np.random.randn(3, 100).astype(np.float32)
        norm, loc, scale = robust_normalize(ctx)
        assert norm.shape == ctx.shape
        assert loc.shape == (3, 1)
        assert scale.shape == (3, 1)

    def test_scale_positive(self) -> None:
        ctx = np.random.randn(2, 50).astype(np.float32)
        _, _, scale = robust_normalize(ctx)
        assert (scale > 0).all()

    def test_constant_series_no_div_zero(self) -> None:
        ctx = np.ones((1, 50), dtype=np.float32)
        norm, _, scale = robust_normalize(ctx)
        assert np.isfinite(norm).all()
        assert scale[0, 0] == pytest.approx(1e-8)

    def test_denormalize_roundtrip(self) -> None:
        ctx = np.random.randn(2, 64).astype(np.float32)
        norm, loc, scale = robust_normalize(ctx)
        recovered = denormalize(norm, loc, scale)
        np.testing.assert_allclose(recovered, ctx, atol=1e-5)


class TestExtractContextFeatures:
    def test_output_shape(self) -> None:
        ctx = torch.randn(8, 2, 128)
        feats = extract_context_features(ctx)
        assert feats.shape == (8, 5)

    def test_finite(self) -> None:
        ctx = torch.randn(4, 1, 64)
        feats = extract_context_features(ctx)
        assert torch.isfinite(feats).all()


# ---------------------------------------------------------------------------
# Basic GiftEvalDataset smoke test (offline — uses synthetic data)
# ---------------------------------------------------------------------------


class TestGiftEvalDatasetSynthetic:
    """Tests that mock the HuggingFace dataset loading."""

    def _make_fake_row(
        self,
        n_variates: int = 1,
        T: int = 200,
        freq: str = "H",
        dataset_name: str = "test_ds",
        item_id: str = "item_0",
    ) -> dict[str, object]:
        target = (np.random.randn(n_variates, T) * 10 + 5).tolist()
        return {
            "dataset_name": dataset_name,
            "item_id": item_id,
            "start": "2020-01-01",
            "freq": freq,
            "target": target,
            "past_feat_dynamic_real": [],
        }

    def test_build_windows_train(self) -> None:
        import random

        from tsfc.data.gifteval import GiftEvalDataset

        ds = GiftEvalDataset.__new__(GiftEvalDataset)
        ds.context_length = 64
        ds.prediction_length = 16
        ds.split = "train"
        ds.val_frac = 0.2
        ds.max_windows_per_series = 5
        ds.seed = 0
        ds._windows = []

        row = self._make_fake_row(T=200)
        rng = random.Random(0)
        entries = ds._build_windows(row, rng)
        assert len(entries) > 0
        assert len(entries) <= 5

    def test_build_windows_val(self) -> None:
        import random

        from tsfc.data.gifteval import GiftEvalDataset

        ds = GiftEvalDataset.__new__(GiftEvalDataset)
        ds.context_length = 64
        ds.prediction_length = 16
        ds.split = "val"
        ds.val_frac = 0.2
        ds.max_windows_per_series = 5
        ds.seed = 0
        ds._windows = []

        row = self._make_fake_row(T=200)
        rng = random.Random(0)
        entries = ds._build_windows(row, rng)
        assert len(entries) == 1  # val = single window

    def test_series_too_short_skipped(self) -> None:
        import random

        from tsfc.data.gifteval import GiftEvalDataset

        ds = GiftEvalDataset.__new__(GiftEvalDataset)
        ds.context_length = 512
        ds.prediction_length = 24
        ds.split = "train"
        ds.val_frac = 0.2
        ds.max_windows_per_series = 5
        ds.seed = 0
        ds._windows = []

        row = self._make_fake_row(T=50)  # too short
        rng = random.Random(0)
        entries = ds._build_windows(row, rng)
        assert entries == []


# ---------------------------------------------------------------------------
# WeightedMixture
# ---------------------------------------------------------------------------


class TestWeightedMixture:
    def test_static_weights_output_shape(self) -> None:
        from tsfc.models.mixture import WeightedMixture

        model = WeightedMixture(["m1", "m2", "m3"], adaptive=False)
        B, V, H = 4, 2, 12
        preds = {
            "m1": torch.randn(B, V, H),
            "m2": torch.randn(B, V, H),
            "m3": torch.randn(B, V, H),
        }
        out = model(preds)
        assert out.shape == (B, V, H)

    def test_adaptive_weights_output_shape(self) -> None:
        from tsfc.models.mixture import WeightedMixture

        model = WeightedMixture(["m1", "m2"], adaptive=True, context_feature_dim=5)
        B, V, H = 3, 1, 8
        preds = {
            "m1": torch.randn(B, V, H),
            "m2": torch.randn(B, V, H),
        }
        feats = torch.randn(B, 5)
        out = model(preds, feats)
        assert out.shape == (B, V, H)

    def test_weights_sum_to_one(self) -> None:
        from tsfc.models.mixture import WeightedMixture

        model = WeightedMixture(["m1", "m2", "m3"], adaptive=False)
        w = model.get_weights(batch_size=1)
        assert w.sum().item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


class TestLossFunctions:
    def test_mae_loss_zero(self) -> None:
        from tsfc.train.loss import mae_loss

        t = torch.randn(2, 1, 12)
        loss = mae_loss(t, t)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_mae_loss_masked(self) -> None:
        from tsfc.train.loss import mae_loss

        forecast = torch.ones(2, 1, 4)
        target = torch.zeros(2, 1, 4)
        mask = torch.tensor([[True, False, True, False], [True, True, False, False]])
        loss = mae_loss(forecast, target, mask)
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_mase_loss_positive(self) -> None:
        from tsfc.train.loss import mase_loss

        B, V, H, T = 2, 1, 8, 64
        forecast = torch.randn(B, V, H)
        target = torch.zeros(B, V, H)
        context = torch.randn(B, V, T)
        loss = mase_loss(forecast, target, context, ["H", "D"])
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


class TestEvalMetrics:
    def test_mae_perfect(self) -> None:
        from tsfc.eval.metrics import mae

        arr = np.random.randn(2, 12).astype(np.float32)
        assert mae(arr, arr) == pytest.approx(0.0, abs=1e-6)

    def test_mase_positive(self) -> None:
        from tsfc.eval.metrics import mase

        ctx = np.random.randn(1, 64).astype(np.float32)
        forecast = np.random.randn(1, 12).astype(np.float32)
        target = np.zeros((1, 12), dtype=np.float32)
        score = mase(forecast, target, ctx, "H")
        assert score >= 0.0

    def test_wql_non_negative(self) -> None:
        from tsfc.eval.metrics import wql

        target = np.random.randn(1, 10).astype(np.float32)
        q_forecasts = {
            0.1: target - 1.0,
            0.5: target,
            0.9: target + 1.0,
        }
        score = wql(q_forecasts, target)
        assert score >= 0.0

    def test_crps_from_samples_shape(self) -> None:
        from tsfc.eval.metrics import crps_from_samples

        samples = np.random.randn(50, 1, 12).astype(np.float32)
        target = np.zeros((1, 12), dtype=np.float32)
        score = crps_from_samples(samples, target)
        assert isinstance(score, float)
        assert score >= 0.0
