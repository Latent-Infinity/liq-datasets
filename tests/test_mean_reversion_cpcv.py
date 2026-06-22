from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from liq.datasets.mean_reversion.labels import TripleBarrierConfig, TripleBarrierLabel, build_labels
from liq.datasets.mean_reversion.labels.assertions import EmbargoViolation, assert_cpcv_embargo
from liq.datasets.mean_reversion.labels.cpcv import (
    CPCVConfig,
    CPCVPath,
    embargo_width,
    generate_paths,
)
from liq.datasets.mean_reversion.labels.dto import AnchorVolSourceDTO, MeanReversionAnchorDTO

REAL_FIXTURE = Path(__file__).resolve().parents[2] / "liq-experiments" / "fixtures" / "real"


def _labels(count: int) -> list[TripleBarrierLabel]:
    return [
        TripleBarrierLabel(
            anchor_event_id=f"anchor-{index:03d}",
            outcome="positive" if index % 2 == 0 else "negative",
            first_touch_index=1,
            barrier_touched="reversion" if index % 2 == 0 else "continuation",
        )
        for index in range(count)
    ]


def _write_anchor_fixture(
    paths_dir: Path, bars: pl.DataFrame, index: int
) -> tuple[Path, Path, Path]:
    anchor_ts = bars[index, "timestamp"]
    anchor = MeanReversionAnchorDTO(
        symbol="NVDA",
        anchor_ts=anchor_ts,
        direction="up",
        anchor_event_id=f"real-anchor-{index}",
        scan_run_id="real-fixture-run",
        scan_query_version="query-v1",
        metric_version="midrange-excursion-v1",
        resolved_universe_version="fixture-real",
        quality_flags=(),
        excursion_units=Decimal("2.0"),
        midrange_now=Decimal(str(bars[index, "close"])),
        midrange_base=Decimal(str(bars[index - 1, "close"])),
        reversion_target=Decimal(str(bars[index + 1, "low"])),
        vol_t=Decimal("1"),
        anchor_vol_source=AnchorVolSourceDTO(
            estimator="range_mean",
            lookback=20,
            min_periods=20,
            calendar_policy="bar_count",
            availability_ts=anchor_ts,
        ),
        regime_at_anchor=None,
    )
    anchor_row = {
        "timestamp": anchor.anchor_ts,
        "symbol": anchor.symbol,
        "anchor_ts": anchor.anchor_ts,
        "direction": anchor.direction,
        "anchor_event_id": anchor.anchor_event_id,
        "scan_run_id": anchor.scan_run_id,
        "scan_query_version": anchor.scan_query_version,
        "metric_version": anchor.metric_version,
        "resolved_universe_version": anchor.resolved_universe_version,
        "quality_flags": json.dumps(list(anchor.quality_flags), sort_keys=True),
        "excursion_units": str(anchor.excursion_units),
        "midrange_now": str(anchor.midrange_now),
        "midrange_base": str(anchor.midrange_base),
        "reversion_target": str(anchor.reversion_target),
        "vol_t": str(anchor.vol_t),
        "anchor_vol_source": anchor.anchor_vol_source.model_dump_json(),
        "regime_at_anchor": "",
        "schema_version": 2,
    }
    anchors_path = paths_dir / f"anchors-{index}.parquet"
    bars_path = paths_dir / "bars.parquet"
    manifest_path = paths_dir / "meta.json"
    pl.DataFrame([anchor_row]).write_parquet(anchors_path)
    bars.write_parquet(bars_path)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "table_name": "mean_reversion_anchors",
                "fields": list(anchor_row),
                "field_types": {
                    "timestamp": "Datetime(time_unit='us', time_zone='UTC')",
                    "symbol": "String",
                    "anchor_ts": "Datetime(time_unit='us', time_zone='UTC')",
                    "direction": "String",
                    "anchor_event_id": "String",
                    "scan_run_id": "String",
                    "scan_query_version": "String",
                    "metric_version": "String",
                    "resolved_universe_version": "String",
                    "quality_flags": "String",
                    "excursion_units": "String",
                    "midrange_now": "String",
                    "midrange_base": "String",
                    "reversion_target": "String",
                    "vol_t": "String",
                    "anchor_vol_source": "String",
                    "regime_at_anchor": "String",
                    "schema_version": "Int64",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return anchors_path, bars_path, manifest_path


def test_generate_paths_emits_each_unique_test_fold_combination() -> None:
    config = CPCVConfig(n_splits=6, n_test_splits=2, embargo_bars=0)

    paths = list(generate_paths(_labels(24), config))

    assert len(paths) == math.comb(6, 2)
    assert len({path.path_id for path in paths}) == len(paths)
    assert paths[0].test_folds == (0, 1)
    assert paths[-1].test_folds == (4, 5)


def test_generate_paths_is_deterministic_and_covers_each_test_fold_size() -> None:
    labels = _labels(25)
    config = CPCVConfig(n_splits=5, n_test_splits=2, embargo_bars=1)

    first = list(generate_paths(labels, config))
    second = list(generate_paths(labels, config))

    assert first == second
    assert all(len(path.test_indices) == 10 for path in first)
    assert all(set(path.train_indices).isdisjoint(path.test_indices) for path in first)


def test_generate_paths_purges_embargoed_train_indices() -> None:
    labels = _labels(12)
    config = CPCVConfig(n_splits=3, n_test_splits=1, embargo_bars=1)

    first_path = next(generate_paths(labels, config))

    assert first_path.test_indices == (0, 1, 2, 3)
    assert first_path.train_indices == (5, 6, 7, 8, 9, 10, 11)


@pytest.mark.parametrize(
    ("l_vol", "l_base", "h", "expected"),
    [
        (1, 2, 3, 3),
        (20, 10, 5, 20),
        (4, 30, 12, 30),
        (9, 8, 40, 40),
    ],
)
def test_embargo_width_is_max_of_lookbacks_and_horizon(
    l_vol: int, l_base: int, h: int, expected: int
) -> None:
    assert embargo_width(l_vol, l_base, h) == expected


def test_embargo_width_rejects_non_positive_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        embargo_width(0, 1, 1)


def test_assert_cpcv_embargo_passes_for_generated_paths() -> None:
    labels = _labels(30)
    paths = list(generate_paths(labels, CPCVConfig(n_splits=5, n_test_splits=1, embargo_bars=3)))

    assert_cpcv_embargo(paths, labels, L_vol=1, L_base=2, H=3)


def test_assert_cpcv_embargo_passes_for_real_fixture_labels(tmp_path: Path) -> None:
    bars = pl.read_parquet(REAL_FIXTURE / "NVDA.parquet").with_columns(
        pl.col("date").cast(pl.Datetime("us", "UTC")).alias("timestamp"),
        pl.lit("NVDA").alias("symbol"),
    )
    labels: list[TripleBarrierLabel] = []
    for index in range(20, 32):
        anchor_path, bars_path, manifest_path = _write_anchor_fixture(tmp_path, bars, index)
        labels.extend(
            build_labels(
                anchor_path,
                bars_path,
                TripleBarrierConfig(H=3, continuation_range_units=Decimal("1000")),
                manifest_path=manifest_path,
            )
        )
    paths = list(generate_paths(labels, CPCVConfig(n_splits=4, n_test_splits=1, embargo_bars=3)))

    assert len(labels) == 12
    assert_cpcv_embargo(paths, labels, L_vol=1, L_base=2, H=3)


def test_assert_cpcv_embargo_raises_for_too_narrow_gap() -> None:
    labels = _labels(10)
    contaminated = CPCVPath(
        path_id="cpcv:manual",
        train_indices=(0, 1, 2, 3, 4),
        test_indices=(6, 7),
        test_folds=(1,),
    )

    with pytest.raises(EmbargoViolation, match="cpcv:manual"):
        assert_cpcv_embargo([contaminated], labels, L_vol=1, L_base=2, H=3)


def test_generate_paths_rejects_invalid_split_config() -> None:
    labels = _labels(4)

    with pytest.raises(ValueError, match="n_test_splits"):
        list(generate_paths(labels, CPCVConfig(n_splits=3, n_test_splits=3, embargo_bars=0)))
    with pytest.raises(ValueError, match="at least n_splits"):
        list(generate_paths(labels, CPCVConfig(n_splits=5, n_test_splits=1, embargo_bars=0)))


@pytest.mark.parametrize(
    ("l_vol", "l_base", "h"),
    [
        (1, 1, 1),
        (10_000, 1, 1),
        (1, 10_000, 1),
        (1, 1, 10_000),
        (7, 11, 13),
        (987, 610, 377),
    ],
)
def test_embargo_width_is_max_across_broad_positive_cases(l_vol: int, l_base: int, h: int) -> None:
    assert embargo_width(l_vol, l_base, h) == max(l_vol, l_base, h)


def test_assert_cpcv_embargo_raises_for_out_of_range_index() -> None:
    labels = _labels(5)
    invalid = CPCVPath(
        path_id="cpcv:bad-index",
        train_indices=(0, 1, 100),  # 100 is outside [0, 5)
        test_indices=(3,),
        test_folds=(0,),
    )

    with pytest.raises(EmbargoViolation, match="outside label range"):
        assert_cpcv_embargo([invalid], labels, L_vol=1, L_base=1, H=1)
