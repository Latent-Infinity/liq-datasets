"""DTOs mirrored from persisted mean-reversion anchor artifacts."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AnchorDirection = Literal["up", "down"]
RegimeLabel = Literal["trend", "chop", "indeterminate"]


class AnchorVolSourceDTO(BaseModel):
    """Mirror of liq-scan anchor volatility provenance schema v2."""

    model_config = ConfigDict(frozen=True)

    estimator: str
    lookback: int = Field(gt=0)
    min_periods: int = Field(gt=0)
    calendar_policy: str
    availability_ts: datetime


class MeanReversionAnchorDTO(BaseModel):
    """Mirror of liq-scan mean-reversion anchor schema v2."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    anchor_ts: datetime
    direction: AnchorDirection
    anchor_event_id: str
    scan_run_id: str
    scan_query_version: str
    metric_version: str
    resolved_universe_version: str
    quality_flags: tuple[str, ...] = ()
    excursion_units: Decimal
    midrange_now: Decimal
    midrange_base: Decimal
    reversion_target: Decimal
    vol_t: Decimal
    anchor_vol_source: AnchorVolSourceDTO
    regime_at_anchor: RegimeLabel | None = None


__all__ = [
    "AnchorDirection",
    "AnchorVolSourceDTO",
    "MeanReversionAnchorDTO",
    "RegimeLabel",
]
