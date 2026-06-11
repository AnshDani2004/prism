"""Helpers for narrowing pandas scalar types under mypy --strict."""

from __future__ import annotations

from datetime import date, datetime
from typing import cast

import pandas as pd


def as_str(val: object) -> str:
    return str(val)


def as_int(val: object) -> int:
    return int(cast("int | float | str", val))


def as_float(val: object) -> float:
    return float(cast("int | float | str", val))


def as_timestamp(val: object) -> pd.Timestamp:
    if isinstance(val, pd.Timestamp):
        return val
    return pd.Timestamp(cast("str | datetime | date", val))
