from __future__ import annotations

from typing import Tuple, List, Dict, Any
import pandas as pd


def basic_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a basic JSON-serializable profile of the dataset."""
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "n_missing_total": int(df.isna().sum().sum()),
        "missing_by_col": df.isna().sum().to_dict(),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
    }


def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify and split numeric vs categorical columns.

    Build0 logic:
    - numeric = pandas number dtypes
    - categorical = everything else
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, cat_cols
