"""Shared validation helpers."""

import pandas as pd


def assert_no_duplicates(df: pd.DataFrame, keys: list[str], label: str) -> None:
    """Raise ValueError if duplicate key combinations exist."""
    dupes = df.duplicated(subset=keys, keep=False)
    if dupes.any():
        n = int(dupes.sum())
        raise ValueError(f"{label}: found {n} rows with duplicate keys {keys}")


def assert_column_in_range(
    df: pd.DataFrame,
    column: str,
    low: float,
    high: float,
    label: str,
) -> None:
    """Raise ValueError if column values fall outside [low, high]."""
    if df.empty:
        return
    out_of_range = (df[column] < low) | (df[column] > high)
    if out_of_range.any():
        n = int(out_of_range.sum())
        raise ValueError(f"{label}: {n} rows in '{column}' outside [{low}, {high}]")
