"""Quick exploratory analysis of GiftEvalPretrain subsets.

Usage
-----
    uv run python scripts/explore_data.py --subsets weather m5 traffic_hourly
    uv run python scripts/explore_data.py --list_subsets
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ExploreConfig:
    """Configuration for the data exploration script."""

    subsets: list[str] = field(default_factory=lambda: ["weather", "m5"])
    """Subset names to explore."""

    max_series: int = 5
    """Number of series to print stats for per subset."""

    list_subsets: bool = False
    """If True, print all available subset names and exit."""

    context_length: int = 512
    prediction_length: int | None = None

    cache_dir: Path | None = None
    """HuggingFace cache directory."""


def list_all_subsets() -> None:
    """Print all available GiftEvalPretrain subset names."""
    try:
        from datasets import get_dataset_config_names  # type: ignore[import-untyped]
    except ImportError as err:
        logger.error("datasets package not installed. Run: uv add datasets")
        raise SystemExit(1) from err

    names = get_dataset_config_names("theforecastingcompany/GiftEvalPretrain")
    print(f"Found {len(names)} subsets:")
    for name in sorted(names):
        print(f"  {name}")


def explore_subset(name: str, max_series: int, cache_dir: Path | None) -> None:
    """Load and print summary statistics for subset *name*."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as err:
        logger.error("datasets package not installed. Run: uv add datasets")
        raise SystemExit(1) from err

    from tsfc.data.utils import nan_fraction, normalize_freq

    logger.info("Loading subset %r …", name)
    kwargs: dict[str, object] = {
        "path": "theforecastingcompany/GiftEvalPretrain",
        "name": name,
        "split": "train",
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    ds = load_dataset(**kwargs) # type: ignore
    total = len(ds)  # type: ignore[arg-type]
    print(f"\n{'='*60}")
    print(f"Subset: {name!r}  ({total} series)")
    print(f"{'='*60}")

    lengths: list[int] = []
    nan_fracs: list[float] = []
    freq_seen: set[str] = set()
    n_variates_seen: set[int] = set()

    n_show = min(max_series, total)
    for i, row in enumerate(ds):
        target = np.array(row["target"], dtype=np.float32)
        if target.ndim == 1:
            target = target[np.newaxis, :]

        V, T = target.shape
        nf = float(nan_fraction(target))
        lengths.append(T)
        nan_fracs.append(nf)
        n_variates_seen.add(V)
        freq_seen.add(normalize_freq(row["freq"]))

        if i < n_show:
            print(
                f"  [{i:4d}] item_id={row['item_id']!r:30s}  "
                f"shape=({V},{T:6d})  freq={normalize_freq(row['freq']):8s}  "
                f"nan%={100*nf:.1f}"
            )

    print(f"\n  Lengths : min={min(lengths)}, median={int(np.median(lengths))}, max={max(lengths)}")
    print(f"  NaN frac: mean={100*float(np.mean(nan_fracs)):.2f}%")
    print(f"  Freqs   : {sorted(freq_seen)}")
    print(f"  #variates: {sorted(n_variates_seen)}")


def main(cfg: ExploreConfig) -> None:
    """Run exploration."""
    if cfg.list_subsets:
        list_all_subsets()
        return

    for subset in cfg.subsets:
        explore_subset(subset, cfg.max_series, cfg.cache_dir)


if __name__ == "__main__":
    main(tyro.cli(ExploreConfig))
