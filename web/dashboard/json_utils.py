"""JSON helpers safe for browser ``JSON.parse`` (no NaN/Infinity)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf and numpy scalars with JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj
