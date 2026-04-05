from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union, Any

import pandas as pd
import matplotlib.pyplot as plt

# output formatting helper
from src.utils.tool_result_utils import ToolResult, make_tool_result


def plot_missingness(
    miss_df: pd.DataFrame,
    out_path: Union[str, Path],
    top_n: int = 30,
) -> ToolResult:
    """Plot missing data in a horizontal bar chart."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = miss_df.head(top_n).iloc[::-1]
    plt.figure()
    plt.barh(plot_df["column"], plot_df["missing_rate"])
    plt.xlabel("Missing rate")
    plt.title(f"Top {min(top_n, len(miss_df))} columns by missingness")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    summary_text = f"Saved missingness plot for top {min(top_n, len(miss_df))} column(s) to {out_path}."

    out = {
        "n_columns_plotted": int(min(top_n, len(miss_df))),
        "artifact_paths": [str(out_path)],
    }

    return make_tool_result(
        name="plot_missingness",
        text=summary_text,
        artifact_paths=[str(out_path)],
        structured=out,
    )


def plot_corr_heatmap(
    df: pd.DataFrame,
    numeric_cols: Optional[list[str]] = None,
    out_path: Optional[str | Path] = None,
    report_dir: Optional[str | Path] = None,
    missing: str = "drop",
) -> ToolResult:
    """
    Create a correlation heatmap for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Full input dataframe.
    numeric_cols : list[str] | None
        Specific numeric columns to include. If None, all numeric columns are used.
    out_path : str | Path | None
        Optional filename or full path for the saved figure.
    report_dir : str | Path | None
        Optional output directory used when out_path is relative or omitted.
    missing : str
        How to handle missing data before computing correlations.
        Supported:
        - "drop": drop rows with missing data in selected numeric columns
        - "pairwise": let pandas compute pairwise correlations

    Returns
    -------
    ToolResult:
        with:
        - "text": summary message
        - "artifact_paths": list of saved file paths
    """
    # If numeric_cols not provided, use all numeric columns
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Basic validation
    if not numeric_cols:
        return make_tool_result(
            name="plot_corr_heatmap",
            text="No numeric columns were available for a correlation heatmap.",
            artifact_paths=[],
            structured={},
        )

    missing_cols = [col for col in numeric_cols if col not in df.columns]
    if missing_cols:
        return make_tool_result(
            name="plot_corr_heatmap",
            text=f"These requested columns were not found in the dataframe: {missing_cols}",
            artifact_paths=[],
            structured={},
        )

    # Keep only numeric columns requested
    work_df = df[numeric_cols].copy()

    # Missing-data handling
    if missing == "drop":
        work_df = work_df.dropna()
        corr = work_df.corr(numeric_only=True)
    elif missing == "pairwise":
        corr = work_df.corr(numeric_only=True)
    else:
        return make_tool_result(
            name="plot_corr_heatmap",
            text=f"Unsupported missing option: {missing}. Use 'drop' or 'pairwise'.",
            artifact_paths=[],
            structured={},
        )

    if corr.empty:
        return make_tool_result(
            name="plot_corr_heatmap",
            text="Correlation matrix was empty after filtering/handling missing data.",
            artifact_paths=[],
            structured={},
        )

    # Resolve output path
    if out_path is None:
        if report_dir is not None:
            out_path = Path(report_dir) / "correlation_heatmap.png"
        else:
            out_path = Path("correlation_heatmap.png")
    else:
        out_path = Path(out_path)
        if not out_path.is_absolute() and report_dir is not None:
            out_path = Path(report_dir) / out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Plot
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)

    x_labels = corr.columns.tolist()
    y_labels = corr.index.tolist()
    plt.xticks(range(len(x_labels)), x_labels, rotation=90, fontsize=7)
    plt.yticks(range(len(y_labels)), y_labels, fontsize=7)

    # Add cell labels only when number of variables is manageable
    if corr.shape[0] <= 20:
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                value = corr.iloc[i, j]
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6)

    plt.title("Correlation heatmap (numeric columns)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return make_tool_result(
        name="plot_corr_heatmap",
        text=(
            f"Saved correlation heatmap for {len(numeric_cols)} numeric column(s): "
            f"{', '.join(numeric_cols)}"
        ),
        artifact_paths=[str(out_path)],
        structured={
            "numeric_columns": numeric_cols,
            "n_numeric_columns": len(numeric_cols),
            "artifact_paths": [str(out_path)],
        },
    )


def plot_histograms(
    df: pd.DataFrame,
    numeric_cols: List[str],
    fig_dir: Union[str, Path],
    max_cols: int = 12,
) -> ToolResult:
    """Save histograms for up to max_cols numeric columns.

    Backward compatible:
      - existing callers pass fig_dir
    Agent-friendly:
      - returns artifact_paths + text
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    for c in numeric_cols[:max_cols]:
        if c not in df.columns:
            continue
        plt.figure()
        df[c].dropna().hist(bins=30)
        plt.title(f"Histogram: {c}")
        plt.tight_layout()
        out_path = fig_dir / f"hist_{c}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        written.append(str(out_path))

    return make_tool_result(
        name="plot_histograms",
        text=f"Saved {len(written)} histogram(s) to {fig_dir}",
        artifact_paths=written,
        structured={
            "numeric_columns_requested": numeric_cols,
            "n_histograms_saved": len(written),
            "artifact_paths": written,
        },
    )


def plot_bar_charts(
    df: pd.DataFrame,
    # Router/LLM-friendly args
    x: Optional[str] = None,  # categorical column (preferred)
    y: Optional[str] = None,  # ignored for bar charts (kept to avoid tool-call crashes)
    # Back-compat args
    cat_cols: Optional[List[str]] = None,
    column: Optional[str] = None,
    # Output control
    fig_dir: Optional[Union[str, Path]] = None,
    max_cols: int = 12,
    top_k: int = 20,
) -> ToolResult:
    """Save bar charts of category COUNTS for categorical columns (top_k categories).

    Accepts any ONE of:
      - x="species"            (router style)
      - column="species"       (older style)
      - cat_cols=["species","island"]

    'y' is accepted for compatibility with (x, y) tool suggestions, but is not used.
    Returns artifact_paths + text (agent-friendly); safe for callers that ignore returns.
    """
    if fig_dir is None:
        fig_dir = Path("figures")
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

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

    written: List[str] = []
    for c in cat_cols[:max_cols]:
        if c not in df.columns:
            raise ValueError(f"Column not found: '{c}'")

        plt.figure()
        counts = df[c].astype("string").value_counts(dropna=True).head(top_k)
        counts.plot(kind="bar")
        plt.title(f"Top {min(top_k, len(counts))} values: {c}")
        plt.tight_layout()
        out_path = fig_dir / f"bar_{c}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        written.append(str(out_path))

    return make_tool_result(
        name="plot_bar_charts",
        text=f"Saved {len(written)} bar chart(s) (counts) to {fig_dir}",
        artifact_paths=written,
        structured={
            "categorical_columns_plotted": cat_cols[:max_cols],
            "top_k": top_k,
            "n_bar_charts_saved": len(written),
            "artifact_paths": written,
        },
    )


def plot_cat_num_boxplot(
    df: pd.DataFrame,
    categorical_column: str,
    numerical_column: str,
    # Backward compatible output controls:
    out_path: Optional[Union[str, Path]] = None,
    fig_dir: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    missing: str = "drop",
) -> ToolResult:
    """Boxplot of a numeric column grouped by a categorical column.

    Backward compatible:
      - older callers may pass out_path
      - Build2 runner tends to inject fig_dir
      - some routers may prefer out_dir

    Missing-data policy:
      - drop (default): drop rows missing either variable
      - raise: error if any missing
    """
    cat_col = categorical_column
    num_col = numerical_column

    if cat_col not in df.columns:
        raise ValueError(f"Column not found: {cat_col}")
    if num_col not in df.columns:
        raise ValueError(f"Column not found: {num_col}")

    # Resolve output directory / path
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        target_dir = fig_dir or out_dir or Path("figures")
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"boxplot_{cat_col}_vs_{num_col}.png"

    sub = df[[cat_col, num_col]].copy()
    n_before = len(sub)

    if missing == "drop":
        sub = sub.dropna(subset=[cat_col, num_col])
    elif missing == "raise":
        if sub[[cat_col, num_col]].isna().any().any():
            raise ValueError(f"Missing values found in '{cat_col}' and/or '{num_col}'.")
    else:
        raise ValueError(f"Unknown missing policy: {missing}")

    n_after = len(sub)
    dropped = n_before - n_after

    if sub.empty:
        raise ValueError("No non-missing rows remain for requested columns.")

    sub[num_col] = pd.to_numeric(sub[num_col], errors="coerce")
    sub = sub.dropna(subset=[num_col])
    n_after_missing = n_after
    n_after_coercion = len(sub)
    dropped_after_coercion = n_after_missing - n_after_coercion

    if sub.empty:
        raise ValueError("Numeric column could not be coerced to numeric.")

    groups: List[Any] = []
    labels: List[str] = []
    for g, grp in sub.groupby(cat_col):
        vals = grp[num_col].values
        if len(vals) > 0:
            groups.append(vals)
            labels.append(str(g))

    if not groups:
        raise ValueError("No valid groups to plot after filtering.")

    plt.figure()
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.xlabel(cat_col)
    plt.ylabel(num_col)
    plt.title(f"{num_col} by {cat_col}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return make_tool_result(
        name="plot_cat_num_boxplot",
        text=(
            f"Saved boxplot to {out_path}. "
            f"Dropped {dropped} row(s) due to missing values "
            f"(policy='{missing}') and {dropped_after_coercion} row(s) "
            f"during numeric coercion."
        ),
        artifact_paths=[str(out_path)],
        structured={
            "categorical_column": cat_col,
            "numerical_column": num_col,
            "missing_policy": missing,
            "n_rows_before": n_before,
            "n_rows_after_missing_filter": n_after,
            "n_rows_after_numeric_coercion": n_after_coercion,
            "n_dropped_missing": dropped,
            "n_dropped_numeric_coercion": dropped_after_coercion,
            "n_groups_plotted": len(groups),
            "artifact_paths": [str(out_path)],
        },
    )
