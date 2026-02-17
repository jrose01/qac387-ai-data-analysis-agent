from __future__ import annotations

from typing import List
import pandas as pd


def summarize_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Compute descriptive statistics for numeric columns."""
    if not numeric_cols:
        return pd.DataFrame(
            columns=["column", "count", "mean", "std", "min", "p25", "median", "p75", "max"]
        )

    summary = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    summary = summary.rename(columns={"50%": "median", "25%": "p25", "75%": "p75"})
    summary.insert(0, "column", summary.index)
    summary.reset_index(drop=True, inplace=True)
    return summary


def summarize_categorical(df: pd.DataFrame, cat_cols: List[str], top_k: int = 10) -> pd.DataFrame:
    """Compute descriptive statistics for categorical columns."""
    rows = []
    for c in cat_cols:
        series = df[c].astype("string")
        n = int(series.shape[0])
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        top = series.value_counts(dropna=True).head(top_k)

        rows.append(
            {
                "column": c,
                "count": n,
                "missing": n_missing,
                "unique": n_unique,
                "top_values": "; ".join([f"{idx} ({val})" for idx, val in top.items()]),
            }
        )
    return pd.DataFrame(rows)


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a missingness table.

    Returns a DataFrame with columns:
    column, missing_rate, missing_count
    sorted by missing_rate descending.
    """
    missing_rate = df.isna().mean()
    missing_count = df.isna().sum()

    out = (
        pd.DataFrame(
            {
                "column": missing_rate.index.astype(str),
                "missing_rate": missing_rate.values.astype(float),
                "missing_count": missing_count.values.astype(int),
            }
        )
        .sort_values("missing_rate", ascending=False, ignore_index=True)
    )
    return out


def correlations(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Compute correlations for numeric columns."""
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    return df[numeric_cols].corr()
