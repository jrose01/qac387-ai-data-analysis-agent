"""
Create "src" package with reusable functions to serve as tools for future agent builds

"""

from .io_utils import ensure_dirs, read_data
from .profiling import basic_profile, split_columns
from .summaries import (
    missingness_table,
    summarize_numeric,
    summarize_categorical,
    correlations,
)
from .plotting import (
    plot_missingness,
    plot_corr_heatmap,
    plot_histograms,
    plot_bar_charts,
)
from .checks import assert_json_safe, target_check
from .modeling import multiple_linear_regression

__all__ = [
    "ensure_dirs",
    "read_data",
    "basic_profile",
    "split_columns",
    "missingness_table",
    "summarize_numeric",
    "summarize_categorical",
    "correlations",
    "plot_missingness",
    "plot_corr_heatmap",
    "plot_histograms",
    "plot_bar_charts",
    "assert_json_safe",
    "target_check",
    "multiple_linear_regression",
]
