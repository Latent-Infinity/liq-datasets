# liq-datasets

Dataset contracts and windowing utilities for the LIQ Stack.

Responsibilities:
- SequenceConfig and FeatureSchema contracts
- Windowing for SSL and supervised datasets
- Holdout splits with embargo and audit logs

This library does not perform feature engineering or model training.

## Failure Modes (Phase 6 hardening)

| Failure | Exception | Trigger | Remediation |
|---------|-----------|---------|-------------|
| Unsorted timestamps | `ValueError` | Input timestamps not in ascending order | Sort timestamps before constructing splits |
| Timezone-naive datetimes | `ValueError` | Datetime boundaries missing timezone info | Use `datetime.timezone.utc` or `zoneinfo` for all boundaries |
| Timezone mismatch | `ValueError` | Boundary timezone differs from index timezone | Align all timezones before split construction |
| Empty resolved range | `ValueError` | Boundary pair produces no data points after `bisect` | Widen window or check data coverage |
| Lookahead overflow | `ValueError` | `label_lookahead_bars` exceeds train/validate/test window | Reduce lookahead or increase window sizes |
| No complete splits | `ValueError` | Walk-forward parameters produce zero valid splits for data length | Increase data length, reduce window/step sizes |
| Invalid boundary type | `TypeError` | Boundary is not `slice` or `tuple[datetime, datetime]` | Use supported boundary types |

All errors are raised at construction or `to_bar_slices()` call time (fail-fast). No silent data truncation occurs.
