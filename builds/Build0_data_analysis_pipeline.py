"""
Build 0: Data Analysis Pipeline (SOLUTION KEY for grading)

This file provides:
- Completed code for BLANK 1 ... BLANK 10
- Full implementations of:
    * missingness_table(df)
    * multiple_linear_regression(df, outcome, predictors=None)
    * Note: The regression function uses statsmodels OLS instead of numpy lstsq for
    * better handling of categorical predictors and missing data.

HOW TO RUN (example): You can copy and paste this command (all one line) in your **terminal** after
replacing Target_Column, Outcome_Column, Predictor1, Predictor2 with actual column names from your dataset:

NOTE: This only works if you have cloned the repository exactly as is.
If not, you will have to adjust the path to the python code and data file to match your folder structure.

python builds/Build0_data_analysis_pipeline_assignment_1.py --data data/penguins.csv --target Target_Column --outcome Outcome_Column --predictors Predictor1,Predictor2 --report_dir reports/

This will run the full pipeline and save outputs to the specified report directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# -----------------------------
# Utilities
# -----------------------------


def ensure_dirs(reports: Path) -> None:
    """Create output folders."""
    # BLANK 1
    (reports / "figures").mkdir(parents=True, exist_ok=True)


def read_data(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame with basic error handling."""
    # BLANK 2
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    # BLANK 3
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataframe is empty.")
    return df


def basic_profile(df: pd.DataFrame) -> dict:
    """Return a basic JSON-serializable profile of the dataset."""
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        # BLANK 4
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "n_missing_total": int(df.isna().sum().sum()),
        "missing_by_col": df.isna().sum().to_dict(),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
    }


def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify and split numeric vs categorical columns."""
    # BLANK 5
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Treat everything else as categorical (for Build0 simplicity)
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, cat_cols


# -----------------------------
# Summaries
# -----------------------------


def summarize_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Compute descriptive statistics for numeric columns."""
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

    # BLANK 6
    summary = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T

    summary = summary.rename(columns={"50%": "median", "25%": "p25", "75%": "p75"})
    summary.insert(0, "column", summary.index)
    summary.reset_index(drop=True, inplace=True)
    return summary


def summarize_categorical(
    df: pd.DataFrame, cat_cols: List[str], top_k: int = 10
) -> pd.DataFrame:
    """Compute descriptive statistics for categorical columns."""
    rows = []
    for c in cat_cols:
        series = df[c].astype("string")
        n = int(series.shape[0])
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))

        # BLANK 7
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


# -----------------------------
# Missingness
# -----------------------------


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a missingness table.

    Returns a DataFrame with columns:
    column, missing_rate, missing_count
    sorted by missing_rate descending.
    """
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


def _is_numeric_series(s: pd.Series) -> bool:
    """Helper: check numeric dtype."""
    return pd.api.types.is_numeric_dtype(s)


def multiple_linear_regression(
    df: pd.DataFrame, outcome: str, predictors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Fit multiple linear regression using statsmodels OLS (formula interface).

    - outcome must be numeric
    - predictors optional:
        if None -> all numeric columns except outcome
        if provided -> uses numeric predictors as-is and wraps non-numeric predictors in C(...)
    - Missing data: statsmodels will drop rows with any missing values in outcome/predictors
        (listwise deletion), matching typical OLS behavior.
    """

    # -------------------------
    # 1) Validate inputs
    # -------------------------
    if outcome not in df.columns:
        raise ValueError(f"Outcome column '{outcome}' not found in DataFrame.")

    if not _is_numeric_series(df[outcome]):
        raise ValueError(
            f"Outcome column '{outcome}' must be numeric for OLS regression."
        )

    if predictors is None:
        # Default: all numeric columns except the outcome
        predictors = [
            c
            for c in df.select_dtypes(include=["number"]).columns.tolist()
            if c != outcome
        ]

    if len(predictors) == 0:
        raise ValueError("No predictors provided.")

    for p in predictors:
        if p not in df.columns:
            raise ValueError(f"Predictor column '{p}' not found in DataFrame.")
        if p == outcome:
            raise ValueError("Outcome cannot be included as a predictor.")

    # -------------------------
    # 2) Build formula
    # -------------------------
    terms: List[str] = []
    for p in predictors:
        if _is_numeric_series(df[p]):
            terms.append(p)
        else:
            # Treat strings/categories as categorical predictors
            terms.append(f"C({p})")

    formula = f"{outcome} ~ " + " + ".join(terms)

    # -------------------------
    # 3) Fit model
    # -------------------------
    model = smf.ols(formula=formula, data=df).fit()

    # -------------------------
    # 4) Package results (JSON-safe)
    # -------------------------
    params = {k: float(v) for k, v in model.params.items()}
    pvals = {k: float(v) for k, v in model.pvalues.items()}
    bse = {k: float(v) for k, v in model.bse.items()}

    ci = model.conf_int()
    conf_int = {idx: [float(ci.loc[idx, 0]), float(ci.loc[idx, 1])] for idx in ci.index}

    results: Dict[str, Any] = {
        "method": "statsmodels_ols",
        "formula": formula,
        "outcome": outcome,
        "predictors": predictors,
        "n_obs": int(model.nobs),
        "df_model": float(model.df_model),
        "df_resid": float(model.df_resid),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "f_stat": float(model.fvalue) if model.fvalue is not None else None,
        "f_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else None,
        "coefficients": params,
        "p_values": pvals,
        "std_err": bse,
        "conf_int_95": conf_int,
    }

    return results


def correlations(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Compute correlations for numeric columns."""
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    # BLANK 8
    corr = df[numeric_cols].corr()
    return corr


# -----------------------------
# Plots
# -----------------------------


def plot_missingness(miss_df: pd.DataFrame, out_path: Path, top_n: int = 30) -> None:
    """Plot missing data in a horizontal bar chart."""
    plot_df = miss_df.head(top_n).iloc[::-1]
    plt.figure()
    # BLANK 9
    plt.barh(plot_df["column"], plot_df["missing_rate"])
    plt.xlabel("Missing rate")
    plt.title(f"Top {min(top_n, len(miss_df))} columns by missingness")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_corr_heatmap(corr: pd.DataFrame, out_path: Path) -> None:
    """Create a heatmap of correlations."""
    if corr.empty:
        return
    plt.figure()
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
    plt.title("Correlation heatmap (numeric columns)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_histograms(
    df: pd.DataFrame, numeric_cols: List[str], fig_dir: Path, max_cols: int = 12
) -> None:
    """Plot histograms for numeric columns."""
    for c in numeric_cols[:max_cols]:
        series = df[c].dropna()
        if series.empty:
            continue
        plt.figure()
        plt.hist(series, bins=30)
        plt.title(f"Histogram: {c}")
        plt.xlabel(c)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(fig_dir / f"hist_{c}.png", dpi=200)
        plt.close()


from pathlib import Path
from typing import List, Optional


from pathlib import Path
from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt


def plot_bar_charts(
    df: pd.DataFrame,
    # Router/LLM-friendly args
    x: Optional[str] = None,  # categorical column (preferred)
    y: Optional[str] = None,  # ignored for bar charts (kept to avoid tool-call crashes)
    # Back-compat args
    cat_cols: Optional[List[str]] = None,
    column: Optional[str] = None,
    # Output control
    fig_dir: Optional[Path] = None,
    max_cols: int = 12,
    top_k: int = 20,
) -> None:
    """Save bar charts for categorical columns (top_k categories).

    Accepts any ONE of:
      - x="species"            (router style)
      - column="species"       (older style)
      - cat_cols=["species","island"]

    'y' is accepted for compatibility with (x, y) tool suggestions, but is not used.
    """

    # ---- Resolve output directory ----
    if fig_dir is None:
        fig_dir = Path("figures")
    else:
        fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Normalize column arguments ----
    provided = [arg is not None for arg in (cat_cols, column, x)]
    if sum(provided) > 1:
        raise ValueError("Provide only one of: 'cat_cols', 'column', or 'x'.")

    if cat_cols is None:
        if column is not None:
            cat_cols = [column]
        elif x is not None:
            cat_cols = [x]
        else:
            raise ValueError("Provide one of: 'cat_cols', 'column', or 'x'.")

    # ---- Plot ----
    for c in cat_cols[:max_cols]:
        if c not in df.columns:
            raise ValueError(f"Column not found: '{c}'")

        plt.figure()
        counts = df[c].astype("string").value_counts(dropna=True).head(top_k)
        counts.plot(kind="bar")
        plt.title(f"Top {min(top_k, len(counts))} values: {c}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"bar_{c}.png", dpi=200)
        plt.close()


# -----------------------------
# Simple model check
# -----------------------------


def assert_json_safe(obj, context: str = "") -> None:
    """Assert that an object can be serialized to JSON."""
    try:
        json.dumps(obj)
    except TypeError as e:
        raise AssertionError(
            f"Object is not JSON-serializable{': ' + context if context else ''}.\n"
            f"Hint: Convert Pandas / NumPy types to native Python types like "
            f"(str, int, float, list, dict).\n"
            f"Original error: {e}"
        )


def target_check(df: pd.DataFrame, target: str) -> Optional[dict]:
    """Look at a target column and return basic information about it."""
    if target not in df.columns:
        print(f"Column '{target}' not found.")
        return None

    y = df[target]

    results: Dict[str, Any] = {}
    results["target"] = str(target)
    results["dtype"] = str(y.dtype)
    results["missing_rate"] = float(y.isna().mean())
    results["n_unique"] = int(y.nunique(dropna=True))

    if y.dtype.kind in "if":
        results["mean"] = float(y.mean())
        results["std"] = float(y.std())
        results["min"] = float(y.min())
        results["max"] = float(y.max())
    else:
        top = y.astype(str).value_counts().head(5)
        results["top_values"] = {str(k): int(v) for k, v in top.items()}

    assert_json_safe(results, context=f"target_check output for column '{target}'")
    return results


# ----------------
# Main pipeline
# ----------------


def main():
    """
    The `main()` function is the entry point of this script.

    Why do we define a main function?
    --------------------------------
    - It clearly separates *what the program does* from helper functions.
    - It allows this file to be imported into another Python file without
    automatically running the analysis.
    - It makes the code easier to test, reuse, and later turn into tools
    or agent workflows.

    When this file is run from the command line, Python will call `main()`.
    """

    # ----------------------------------------
    # 1. Set up command-line arguments
    # ----------------------------------------
    # argparse lets users control the program from the terminal
    # without editing the code itself.
    parser = argparse.ArgumentParser()

    # Required argument: path to the dataset
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file")

    # Column to treat as a target for summary checks
    parser.add_argument("--target", type=str, default=None, help="target column")

    # Outcome variable for regression
    parser.add_argument(
        "--outcome", type=str, default=None, help="Optional outcome for regression"
    )

    # Predictors for regression (comma-separated string)
    parser.add_argument(
        "--predictors",
        type=str,
        default=None,
        help="Comma-separated predictors for regression (e.g., 'age,fare')",
    )

    # Directory where outputs will be saved
    parser.add_argument(
        "--report_dir", type=str, default="reports", help="Output directory"
    )

    # Parse all command-line arguments into a single object
    args = parser.parse_args()

    # ----------------------------------------
    # 2. Prepare output directories
    # ----------------------------------------
    report_dir = Path(args.report_dir)

    # Ensure the output directory (and subfolders) exist
    ensure_dirs(report_dir)

    # ----------------------------------------
    # 3. Load data and identify column types
    # ----------------------------------------
    df = read_data(Path(args.data))

    # Split columns into numeric vs categorical
    numeric_cols, cat_cols = split_columns(df)

    # ----------------------------------------
    # 4. Generate summary outputs
    # ----------------------------------------
    profile = basic_profile(df)
    miss_df = missingness_table(df)
    num_summary = summarize_numeric(df, numeric_cols)
    cat_summary = summarize_categorical(df, cat_cols)
    corr = correlations(df, numeric_cols)

    # ----------------------------------------
    # 5. Save tabular outputs to disk
    # ----------------------------------------
    (report_dir / "data_profile.json").write_text(json.dumps(profile, indent=2))
    miss_df.to_csv(report_dir / "missingness_by_column.csv", index=False)
    num_summary.to_csv(report_dir / "summary_numeric.csv", index=False)
    cat_summary.to_csv(report_dir / "summary_categorical.csv", index=False)

    # Only save correlations if at least one exists
    if not corr.empty:
        corr.to_csv(report_dir / "correlations.csv")

    # ----------------------------------------
    # 6. Generate and save plots
    # ----------------------------------------
    plot_missingness(miss_df, report_dir / "figures" / "missingness.png")
    plot_corr_heatmap(corr, report_dir / "figures" / "corr_heatmap.png")
    plot_histograms(df, numeric_cols, report_dir / "figures")
    plot_bar_charts(df, cat_cols, report_dir / "figures")

    # ----------------------------------------
    # 7. Target variable checks
    # ----------------------------------------
    # Only run this section if --target was provided
    if args.target:
        target_info = target_check(df, args.target)
        (report_dir / "target_overview.json").write_text(
            json.dumps(target_info, indent=2)
        )

    # ----------------------------------------
    # 8. Regression analysis
    # ----------------------------------------
    # Only runs if --outcome is provided
    if args.outcome:
        preds: Optional[List[str]] = None

        # If predictors were provided, convert the comma-separated string
        # into a clean Python list
        if args.predictors:
            # BLANK 10: parse comma-separated predictors into a list of cleaned names
            preds = [p.strip() for p in args.predictors.split(",") if p.strip()]

        # Run the regression
        reg_results = multiple_linear_regression(
            df, outcome=args.outcome, predictors=preds
        )

        # Ensure the output can be safely saved as JSON
        assert_json_safe(reg_results, context="multiple_linear_regression output")

        # Save regression results
        (report_dir / "regression_results.json").write_text(
            json.dumps(reg_results, indent=2)
        )

    # ----------------------------------------
    # 9. Final user message
    # ----------------------------------------
    print(f"Build0 pipeline complete. Outputs saved to: {report_dir.resolve()}")


# ------------------------------------------------------------
# This conditional ensures that main() only runs when this file
# is executed directly (not when it is imported as a module).
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
