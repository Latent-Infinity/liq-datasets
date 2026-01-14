"""Dataset bundle models and hashing helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _stable_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def compute_hash(data: dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(data).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetBundle:
    """Container for datasets plus metadata and audit logs."""

    X: np.ndarray
    y: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_hash: str | None = None
    config_hash: str | None = None
    holdout_audit: dict[str, Any] | None = None
