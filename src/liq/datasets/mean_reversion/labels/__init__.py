"""Mean-reversion triple-barrier labels from persisted anchor artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from liq.datasets.mean_reversion.labels.dto import AnchorVolSourceDTO, MeanReversionAnchorDTO

LabelOutcome = Literal["positive", "negative", "timeout"]
BarrierTouched = Literal["reversion", "continuation", "timeout"]

_EXPECTED_FIELD_TYPES = {
    "anchor_event_id": "String",
    "anchor_ts": "Datetime(time_unit='us', time_zone='UTC')",
    "anchor_vol_source": "String",
    "direction": "String",
    "excursion_units": "String",
    "metric_version": "String",
    "midrange_base": "String",
    "midrange_now": "String",
    "quality_flags": "String",
    "regime_at_anchor": "String",
    "resolved_universe_version": "String",
    "reversion_target": "String",
    "scan_query_version": "String",
    "scan_run_id": "String",
    "schema_version": "Int64",
    "symbol": "String",
    "timestamp": "Datetime(time_unit='us', time_zone='UTC')",
    "vol_t": "String",
}


class AnchorSchemaParityViolation(RuntimeError):
    """Raised when persisted anchor schema metadata no longer matches the DTO mirror."""


class TripleBarrierConfig(BaseModel):
    """Configuration for range-unit continuation and time barriers."""

    model_config = ConfigDict(frozen=True)

    H: int = Field(gt=0)
    continuation_range_units: Decimal = Field(default=Decimal("1.0"), gt=Decimal("0"))

    def continuation_barrier(self, anchor: MeanReversionAnchorDTO) -> Decimal:
        offset = anchor.vol_t * self.continuation_range_units
        if anchor.direction == "up":
            return anchor.midrange_now + offset
        return anchor.midrange_now - offset


class TripleBarrierLabel(BaseModel):
    """One triple-barrier outcome linked to a source anchor."""

    model_config = ConfigDict(frozen=True)

    anchor_event_id: str
    outcome: LabelOutcome
    first_touch_index: int | None
    barrier_touched: BarrierTouched


def _decimal(value: object) -> Decimal:
    return Decimal(str(value))


def _timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    raise ValueError("anchor_ts must be a datetime")


def _direction(value: object) -> Literal["up", "down"]:
    if value == "up":
        return "up"
    if value == "down":
        return "down"
    raise ValueError("direction must be 'up' or 'down'")


def _regime_label(value: object) -> Literal["trend", "chop", "indeterminate"] | None:
    if value == "trend":
        return "trend"
    if value == "chop":
        return "chop"
    if value == "indeterminate":
        return "indeterminate"
    return None


def _parse_anchor(row: Mapping[str, object]) -> MeanReversionAnchorDTO:
    flags_raw = row.get("quality_flags") or "[]"
    source_raw = row.get("anchor_vol_source") or "{}"
    return MeanReversionAnchorDTO(
        symbol=str(row["symbol"]),
        anchor_ts=_timestamp(row["anchor_ts"]),
        direction=_direction(row["direction"]),
        anchor_event_id=str(row["anchor_event_id"]),
        scan_run_id=str(row["scan_run_id"]),
        scan_query_version=str(row["scan_query_version"]),
        metric_version=str(row["metric_version"]),
        resolved_universe_version=str(row["resolved_universe_version"]),
        quality_flags=tuple(str(flag) for flag in json.loads(str(flags_raw))),
        excursion_units=_decimal(row["excursion_units"]),
        midrange_now=_decimal(row["midrange_now"]),
        midrange_base=_decimal(row["midrange_base"]),
        reversion_target=_decimal(row["reversion_target"]),
        vol_t=_decimal(row["vol_t"]),
        anchor_vol_source=AnchorVolSourceDTO.model_validate_json(str(source_raw)),
        regime_at_anchor=_regime_label(row.get("regime_at_anchor")),
    )


def _manifest_path(anchor_parquet_path: Path) -> Path:
    parent_meta = anchor_parquet_path.parent / "meta.json"
    if parent_meta.exists():
        return parent_meta
    return anchor_parquet_path.with_name("meta.json")


def _assert_schema_parity(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    field_types = payload.get("field_types")
    if not isinstance(field_types, dict):
        raise AnchorSchemaParityViolation("anchor manifest missing field_types")
    for name, expected in _EXPECTED_FIELD_TYPES.items():
        observed = field_types.get(name)
        if observed != expected:
            raise AnchorSchemaParityViolation(
                f"anchor schema mismatch for {name}: expected {expected!r}, observed {observed!r}"
            )


def classify_barrier_path(
    *,
    anchor_event_id: str,
    direction: str,
    reversion_barrier: Decimal,
    continuation_barrier: Decimal,
    upper_path: Sequence[float],
    lower_path: Sequence[float],
) -> TripleBarrierLabel:
    """Classify a path using first-touch triple-barrier semantics."""
    for index, (upper, lower) in enumerate(zip(upper_path, lower_path, strict=True), start=1):
        high = _decimal(upper)
        low = _decimal(lower)
        if direction == "up":
            if low <= reversion_barrier:
                return TripleBarrierLabel(
                    anchor_event_id=anchor_event_id,
                    outcome="positive",
                    first_touch_index=index,
                    barrier_touched="reversion",
                )
            if high >= continuation_barrier:
                return TripleBarrierLabel(
                    anchor_event_id=anchor_event_id,
                    outcome="negative",
                    first_touch_index=index,
                    barrier_touched="continuation",
                )
        elif direction == "down":
            if high >= reversion_barrier:
                return TripleBarrierLabel(
                    anchor_event_id=anchor_event_id,
                    outcome="positive",
                    first_touch_index=index,
                    barrier_touched="reversion",
                )
            if low <= continuation_barrier:
                return TripleBarrierLabel(
                    anchor_event_id=anchor_event_id,
                    outcome="negative",
                    first_touch_index=index,
                    barrier_touched="continuation",
                )
        else:
            raise ValueError("direction must be 'up' or 'down'")
    return TripleBarrierLabel(
        anchor_event_id=anchor_event_id,
        outcome="timeout",
        first_touch_index=None,
        barrier_touched="timeout",
    )


def _bar_timestamp_column(frame: pl.DataFrame) -> str:
    if "timestamp" in frame.columns:
        return "timestamp"
    if "date" in frame.columns:
        return "date"
    raise ValueError("bars parquet must contain timestamp or date column")


def _forward_bars(bars: pl.DataFrame, anchor: MeanReversionAnchorDTO, horizon: int) -> pl.DataFrame:
    ts_column = _bar_timestamp_column(bars)
    symbol_bars = bars.filter(pl.col("symbol") == anchor.symbol).sort(ts_column)
    timestamps = symbol_bars[ts_column].to_list()
    try:
        anchor_index = timestamps.index(anchor.anchor_ts)
    except ValueError as exc:
        raise ValueError(f"anchor timestamp not found for {anchor.anchor_event_id}") from exc
    return symbol_bars.slice(anchor_index + 1, horizon)


def build_labels(
    anchor_parquet_path: Path,
    bars_parquet_path: Path,
    config: TripleBarrierConfig,
    *,
    manifest_path: Path | None = None,
) -> Iterator[TripleBarrierLabel]:
    """Build one triple-barrier label per persisted anchor."""
    resolved_manifest = manifest_path or _manifest_path(anchor_parquet_path)
    _assert_schema_parity(resolved_manifest)
    anchors = [
        _parse_anchor(row) for row in pl.read_parquet(anchor_parquet_path).iter_rows(named=True)
    ]
    bars = pl.read_parquet(bars_parquet_path)
    for anchor in anchors:
        forward = _forward_bars(bars, anchor, config.H)
        yield classify_barrier_path(
            anchor_event_id=anchor.anchor_event_id,
            direction=anchor.direction,
            reversion_barrier=anchor.reversion_target,
            continuation_barrier=config.continuation_barrier(anchor),
            upper_path=[float(value) for value in forward["high"].to_list()],
            lower_path=[float(value) for value in forward["low"].to_list()],
        )


__all__ = [
    "AnchorSchemaParityViolation",
    "TripleBarrierConfig",
    "TripleBarrierLabel",
    "build_labels",
    "classify_barrier_path",
]
