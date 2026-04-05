from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple, cast
import pandas as pd
from math import atanh, tanh, sqrt
from regex import sub
from scipy import stats


# output formatting helper
from src.utils.tool_result_utils import make_tool_result, ToolResult


def summarize_numeric(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    column: Optional[str] = None,
) -> pd.DataFrame:
    """Compute descriptive statistics for numeric columns.

    Accepts either:
      - numeric_cols=[...]
      - column="..." (single column)
    """
    if numeric_cols is not None and column is not None:
        raise ValueError("Provide only one of: 'numeric_cols' or 'column'.")

    if numeric_cols is None:
        if column is None:
            raise ValueError("Provide either 'numeric_cols' or 'column'.")
        numeric_cols = [column]

    if not numeric_cols:
        return pd.DataFrame(
            columns=[
                "column",
                "count",
                "mean",
                "std",
                "min",
                "p25",
                "median",
                "p75",
                "max",
            ]
        )

    missing = [c for c in numeric_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Numeric column(s) not found: {missing}")

    summary = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    summary = summary.rename(columns={"50%": "median", "25%": "p25", "75%": "p75"})
    summary.insert(0, "column", summary.index.astype(str))
    summary.reset_index(drop=True, inplace=True)
    return summary


def summarize_categorical(
    df: pd.DataFrame,
    cat_cols: List[str] | None = None,
    column: str | None = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """Compute descriptive statistics for categorical columns.

    Accepts either:
      - column="species"
      - cat_cols=["species", "island"]
    """
    if cat_cols is None:
        if column is None:
            raise ValueError("Provide either 'column' or 'cat_cols'.")
        cat_cols = [column]

    rows = []
    for c in cat_cols:
        if c not in df.columns:
            raise ValueError(f"Column not found: '{c}'")
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
    """Create a missingness table (column, missing_rate, missing_count)."""
    missing_rate = df.isna().mean()
    missing_count = df.isna().sum()

    out = pd.DataFrame(
        {
            "column": missing_rate.index.astype(str),
            "missing_rate": missing_rate.values.astype(float),
            "missing_count": missing_count.values.astype(int),
        }
    ).sort_values("missing_rate", ascending=False, ignore_index=True)
    return out


def pearson_correlation(
    df: pd.DataFrame,
    x: str,
    y: str,
    ci_level: float = 0.95,
    min_n_recommendation: int = 30,
) -> ToolResult:
    """
    Compute Pearson correlation statistics between two numeric variables.

    Returns:
        r, r2, p_value (two-sided), n,
        CI for r via Fisher z transform,
        plus explanatory methods note.
    """

    if x not in df.columns:
        raise ValueError(f"Column not found: {x}")
    if y not in df.columns:
        raise ValueError(f"Column not found: {y}")

    # Pairwise complete cases
    pair_df = df[[x, y]].copy()
    pair_df[x] = pd.to_numeric(pair_df[x], errors="coerce")
    pair_df[y] = pd.to_numeric(pair_df[y], errors="coerce")
    pair_df = pair_df.dropna()
    n = int(len(pair_df))

    if n < 10:
        raise ValueError(
            "Need at least 10 complete observations to compute CI and p-value."
        )

    r, p_value = cast(
        Tuple[float, float],
        stats.pearsonr(pair_df[x].to_numpy(), pair_df[y].to_numpy()),
    )
    r = float(r)
    p_value = float(p_value)
    r2 = r * r

    # Fisher z confidence interval
    eps = 1e-12
    r_clip = max(min(r, 1 - eps), -1 + eps)

    z = atanh(r_clip)
    se = 1.0 / sqrt(n - 3)

    alpha = 1.0 - ci_level
    zcrit = float(stats.norm.ppf(1 - alpha / 2))
    z_lo = z - zcrit * se
    z_hi = z + zcrit * se

    ci_low = float(tanh(z_lo))
    ci_high = float(tanh(z_hi))

    methods_note = (
        "Methods: Rows with missing or non-numeric values were dropped "
        "(pairwise complete cases). Pearson r and two-sided p-value "
        "computed using scipy.stats.pearsonr. Confidence interval "
        "computed via Fisher z-transform with SE=1/sqrt(n-3). "
        f"A common rule of thumb is n ≥ {min_n_recommendation} "
        "for more stable confidence interval estimates."
    )

    text = (
        f"Pearson correlation between '{x}' and '{y}': "
        f"r = {r:.4f} ({int(ci_level * 100)}% CI [{ci_low:.4f}, {ci_high:.4f}]), "
        f"r² = {r2:.4f}, p = {p_value:.4g}, n = {n}.\n\n"
        f"{methods_note}"
    )

    return make_tool_result(
        name="pearson_correlation",
        text=text,
        artifact_paths=[],
        structured={
            "x": x,
            "y": y,
            "n": n,
            "r": r,
            "r2": r2,
            "ci_level": ci_level,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_value,
            "min_n_recommendation": min_n_recommendation,
        },
    )
