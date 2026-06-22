from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from liq.datasets.mean_reversion.labels import (
    AnchorSchemaParityViolation,
    TripleBarrierConfig,
    TripleBarrierLabel,
    build_labels,
    classify_barrier_path,
)
from liq.datasets.mean_reversion.labels.dto import AnchorVolSourceDTO, MeanReversionAnchorDTO

REAL_FIXTURE = Path(__file__).resolve().parents[2] / "liq-experiments" / "fixtures" / "real"


def _source(ts: datetime) -> AnchorVolSourceDTO:
    return AnchorVolSourceDTO(
        estimator="range_mean",
        lookback=20,
        min_periods=20,
        calendar_policy="bar_count",
        availability_ts=ts,
    )


def _anchor(
    *,
    anchor_ts: datetime,
    direction: str,
    midrange_now: Decimal,
    reversion_target: Decimal,
    vol_t: Decimal,
) -> MeanReversionAnchorDTO:
    return MeanReversionAnchorDTO(
        symbol="NVDA",
        anchor_ts=anchor_ts,
        direction=direction,
        anchor_event_id=f"anchor-{direction}-{anchor_ts.date().isoformat()}",
        scan_run_id="scan-run",
        scan_query_version="query-v1",
        metric_version="midrange-excursion-v1",
        resolved_universe_version="fixture-real",
        quality_flags=(),
        excursion_units=Decimal("2.0"),
        midrange_now=midrange_now,
        midrange_base=Decimal("100"),
        reversion_target=reversion_target,
        vol_t=vol_t,
        anchor_vol_source=_source(anchor_ts),
        regime_at_anchor=None,
    )


def _anchor_frame(anchor: MeanReversionAnchorDTO) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
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
                "regime_at_anchor": anchor.regime_at_anchor or "",
                "schema_version": 2,
            }
        ]
    )


def _write_manifest(path: Path, *, field_types: dict[str, str] | None = None) -> None:
    fields = list(
        _anchor_frame(
            _anchor(
                anchor_ts=datetime(2022, 1, 3, tzinfo=UTC),
                direction="up",
                midrange_now=Decimal("100"),
                reversion_target=Decimal("99"),
                vol_t=Decimal("1"),
            )
        ).columns
    )
    payload = {
        "schema_version": 2,
        "table_name": "mean_reversion_anchors",
        "fields": fields,
        "field_types": field_types
        or {
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
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    (
        "direction",
        "upper_path",
        "lower_path",
        "expected_outcome",
        "expected_index",
        "expected_barrier",
    ),
    [
        ("up", [101.0, 102.0], [99.0, 98.0], "positive", 1, "reversion"),
        ("up", [103.0, 104.0], [100.5, 99.0], "negative", 1, "continuation"),
        ("up", [101.0, 102.0], [100.5, 100.25], "timeout", None, "timeout"),
        ("down", [101.0, 102.0], [99.0, 98.0], "positive", 1, "reversion"),
        ("down", [100.5, 101.0], [97.0, 96.0], "negative", 1, "continuation"),
        ("down", [99.5, 99.75], [98.5, 98.25], "timeout", None, "timeout"),
    ],
)
def test_classify_barrier_path_decision_matrix(
    direction: str,
    upper_path: list[float],
    lower_path: list[float],
    expected_outcome: str,
    expected_index: int | None,
    expected_barrier: str,
) -> None:
    label = classify_barrier_path(
        anchor_event_id="anchor-1",
        direction=direction,
        reversion_barrier=Decimal("99") if direction == "up" else Decimal("101"),
        continuation_barrier=Decimal("103") if direction == "up" else Decimal("97"),
        upper_path=upper_path,
        lower_path=lower_path,
    )

    assert label == TripleBarrierLabel(
        anchor_event_id="anchor-1",
        outcome=expected_outcome,
        first_touch_index=expected_index,
        barrier_touched=expected_barrier,
    )


def test_first_touch_wins_when_both_barriers_are_reachable() -> None:
    label = classify_barrier_path(
        anchor_event_id="anchor-1",
        direction="up",
        reversion_barrier=Decimal("99"),
        continuation_barrier=Decimal("103"),
        upper_path=[101.0, 104.0],
        lower_path=[99.0, 98.0],
    )

    assert label.outcome == "positive"
    assert label.first_touch_index == 1
    assert label.barrier_touched == "reversion"


def test_range_unit_barriers_adapt_per_anchor() -> None:
    config = TripleBarrierConfig(H=2, continuation_range_units=Decimal("1.5"))
    small = _anchor(
        anchor_ts=datetime(2022, 1, 3, tzinfo=UTC),
        direction="up",
        midrange_now=Decimal("100"),
        reversion_target=Decimal("99"),
        vol_t=Decimal("2"),
    )
    large = _anchor(
        anchor_ts=datetime(2022, 1, 3, tzinfo=UTC),
        direction="up",
        midrange_now=Decimal("100"),
        reversion_target=Decimal("99"),
        vol_t=Decimal("4"),
    )

    assert config.continuation_barrier(small) == Decimal("103.0")
    assert config.continuation_barrier(large) == Decimal("106.0")


def test_build_labels_is_deterministic_and_emits_one_label_per_real_fixture_anchor(
    tmp_path: Path,
) -> None:
    bars = pl.read_parquet(REAL_FIXTURE / "NVDA.parquet").with_columns(
        pl.col("date").cast(pl.Datetime("us", "UTC")).alias("timestamp"),
        pl.lit("NVDA").alias("symbol"),
    )
    anchor_ts = bars[20, "timestamp"]
    anchor = _anchor(
        anchor_ts=anchor_ts,
        direction="up",
        midrange_now=Decimal(str(bars[20, "close"])),
        reversion_target=Decimal(str(bars[21, "low"])),
        vol_t=Decimal("1"),
    )
    anchors_path = tmp_path / "anchors.parquet"
    bars_path = tmp_path / "bars.parquet"
    manifest_path = tmp_path / "meta.json"
    _anchor_frame(anchor).write_parquet(anchors_path)
    bars.write_parquet(bars_path)
    _write_manifest(manifest_path)

    config = TripleBarrierConfig(H=3, continuation_range_units=Decimal("1000"))
    first = list(build_labels(anchors_path, bars_path, config, manifest_path=manifest_path))
    second = list(build_labels(anchors_path, bars_path, config, manifest_path=manifest_path))

    assert first == second
    assert len(first) == 1
    assert first[0].anchor_event_id == anchor.anchor_event_id
    assert first[0].outcome == "positive"


def test_schema_parity_mismatch_raises(tmp_path: Path) -> None:
    anchor = _anchor(
        anchor_ts=datetime(2022, 1, 3, tzinfo=UTC),
        direction="up",
        midrange_now=Decimal("100"),
        reversion_target=Decimal("99"),
        vol_t=Decimal("1"),
    )
    anchors_path = tmp_path / "anchors.parquet"
    bars_path = tmp_path / "bars.parquet"
    manifest_path = tmp_path / "meta.json"
    _anchor_frame(anchor).write_parquet(anchors_path)
    pl.DataFrame(
        {
            "symbol": ["NVDA"],
            "timestamp": [anchor.anchor_ts],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
        }
    ).write_parquet(bars_path)
    _write_manifest(manifest_path, field_types={"anchor_event_id": "Int64"})

    with pytest.raises(AnchorSchemaParityViolation, match="anchor_event_id"):
        list(
            build_labels(
                anchors_path, bars_path, TripleBarrierConfig(H=1), manifest_path=manifest_path
            )
        )


def test_down_direction_continuation_barrier_subtracts_range_units() -> None:
    anchor = _anchor(
        anchor_ts=datetime(2022, 1, 3, tzinfo=UTC),
        direction="down",
        midrange_now=Decimal("100"),
        reversion_target=Decimal("101"),
        vol_t=Decimal("4"),
    )

    assert TripleBarrierConfig(H=2, continuation_range_units=Decimal("1.5")).continuation_barrier(
        anchor
    ) == Decimal("94.0")


def test_classify_barrier_path_rejects_invalid_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        classify_barrier_path(
            anchor_event_id="anchor-1",
            direction="sideways",
            reversion_barrier=Decimal("99"),
            continuation_barrier=Decimal("103"),
            upper_path=[100.0],
            lower_path=[99.5],
        )


def test_build_labels_uses_default_manifest_path_and_date_column(tmp_path: Path) -> None:
    anchor_ts = datetime(2022, 1, 3, tzinfo=UTC)
    anchor = _anchor(
        anchor_ts=anchor_ts,
        direction="down",
        midrange_now=Decimal("100"),
        reversion_target=Decimal("101"),
        vol_t=Decimal("1"),
    )
    anchors_path = tmp_path / "anchors.parquet"
    bars_path = tmp_path / "bars.parquet"
    _anchor_frame(anchor).write_parquet(anchors_path)
    _write_manifest(tmp_path / "meta.json")
    pl.DataFrame(
        {
            "symbol": ["NVDA", "NVDA"],
            "date": [anchor_ts, datetime(2022, 1, 4, tzinfo=UTC)],
            "high": [100.0, 101.5],
            "low": [99.0, 98.0],
            "close": [100.0, 100.5],
        }
    ).write_parquet(bars_path)

    labels = list(build_labels(anchors_path, bars_path, TripleBarrierConfig(H=1)))

    assert labels[0].outcome == "positive"


def test_build_labels_rejects_manifest_without_field_types(tmp_path: Path) -> None:
    manifest_path = tmp_path / "meta.json"
    manifest_path.write_text(json.dumps({"fields": []}) + "\n", encoding="utf-8")
    anchor = _anchor(
        anchor_ts=datetime(2022, 1, 3, tzinfo=UTC),
        direction="up",
        midrange_now=Decimal("100"),
        reversion_target=Decimal("99"),
        vol_t=Decimal("1"),
    )
    anchors_path = tmp_path / "anchors.parquet"
    bars_path = tmp_path / "bars.parquet"
    _anchor_frame(anchor).write_parquet(anchors_path)
    pl.DataFrame(
        {"symbol": ["NVDA"], "timestamp": [anchor.anchor_ts], "high": [101.0], "low": [99.0]}
    ).write_parquet(bars_path)

    with pytest.raises(AnchorSchemaParityViolation, match="field_types"):
        list(
            build_labels(
                anchors_path, bars_path, TripleBarrierConfig(H=1), manifest_path=manifest_path
            )
        )


def test_build_labels_rejects_missing_anchor_timestamp(tmp_path: Path) -> None:
    anchor = _anchor(
        anchor_ts=datetime(2022, 1, 3, tzinfo=UTC),
        direction="up",
        midrange_now=Decimal("100"),
        reversion_target=Decimal("99"),
        vol_t=Decimal("1"),
    )
    anchors_path = tmp_path / "anchors.parquet"
    bars_path = tmp_path / "bars.parquet"
    manifest_path = tmp_path / "meta.json"
    _anchor_frame(anchor).write_parquet(anchors_path)
    _write_manifest(manifest_path)
    pl.DataFrame(
        {
            "symbol": ["NVDA"],
            "timestamp": [datetime(2022, 1, 4, tzinfo=UTC)],
            "high": [101.0],
            "low": [99.0],
        }
    ).write_parquet(bars_path)

    with pytest.raises(ValueError, match=anchor.anchor_event_id):
        list(
            build_labels(
                anchors_path, bars_path, TripleBarrierConfig(H=1), manifest_path=manifest_path
            )
        )


def test_build_labels_rejects_bars_without_time_column(tmp_path: Path) -> None:
    anchor = _anchor(
        anchor_ts=datetime(2022, 1, 3, tzinfo=UTC),
        direction="up",
        midrange_now=Decimal("100"),
        reversion_target=Decimal("99"),
        vol_t=Decimal("1"),
    )
    anchors_path = tmp_path / "anchors.parquet"
    bars_path = tmp_path / "bars.parquet"
    manifest_path = tmp_path / "meta.json"
    _anchor_frame(anchor).write_parquet(anchors_path)
    _write_manifest(manifest_path)
    pl.DataFrame({"symbol": ["NVDA"], "high": [101.0], "low": [99.0]}).write_parquet(bars_path)

    with pytest.raises(ValueError, match="timestamp or date"):
        list(
            build_labels(
                anchors_path, bars_path, TripleBarrierConfig(H=1), manifest_path=manifest_path
            )
        )


def test_labels_package_does_not_import_liq_scan() -> None:
    import sys

    before = {name for name in sys.modules if name == "liq.scan" or name.startswith("liq.scan.")}
    __import__("liq.datasets.mean_reversion.labels")
    after = {name for name in sys.modules if name == "liq.scan" or name.startswith("liq.scan.")}

    assert after == before
