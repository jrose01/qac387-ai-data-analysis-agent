from __future__ import annotations

from typing import Any, Dict
import json
import pandas as pd


def assert_json_safe(obj: Any, context: str = "") -> None:
    """Raise a TypeError if obj cannot be serialized to JSON."""
    try:
        json.dumps(obj)
    except TypeError as e:
        msg = "Object not JSON-serializable"
        if context:
            msg += f" ({context})"
        raise TypeError(msg) from e


def target_check(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """
    Return basic information about a target column:
    - existence
    - dtype
    - missingness
    - (if numeric) basic stats
    - (if categorical) top values
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    s = df[target]
    out: Dict[str, Any] = {
        "target": str(target),
        "dtype": str(s.dtype),
        "n_rows": int(len(s)),
        "n_missing": int(s.isna().sum()),
        "missing_rate": float(s.isna().mean()),
    }

    if pd.api.types.is_numeric_dtype(s):
        desc = s.describe()
        out["numeric_summary"] = {
            "count": float(desc.get("count", float("nan"))),
            "mean": float(desc.get("mean", float("nan"))),
            "std": float(desc.get("std", float("nan"))),
            "min": float(desc.get("min", float("nan"))),
            "p25": float(desc.get("25%", float("nan"))),
            "median": float(desc.get("50%", float("nan"))),
            "p75": float(desc.get("75%", float("nan"))),
            "max": float(desc.get("max", float("nan"))),
        }
    else:
        top = s.astype("string").value_counts(dropna=True).head(10)
        out["top_values"] = [{"value": str(k), "count": int(v)} for k, v in top.items()]

    return out
