from __future__ import annotations

from typing import Optional, List, Dict, Any
import pandas as pd
import statsmodels.formula.api as smf

from src.utils.tool_result_utils import ToolResult, make_tool_result


def multiple_linear_regression(
    df: pd.DataFrame,
    outcome: str,
    predictors: Optional[List[str]] = None,
) -> ToolResult:
    """
    Fit multiple linear regression using statsmodels OLS (formula interface)
    for numeric outcome and numeric/categorical predictors.
    """

    # --- Validate inputs ---
    if outcome not in df.columns:
        raise ValueError(f"Outcome column '{outcome}' not found in dataframe.")

    if predictors is None or len(predictors) == 0:
        raise ValueError("You must specify at least one predictor.")

    missing_preds = [p for p in predictors if p not in df.columns]
    if missing_preds:
        raise ValueError(f"Predictor(s) not found: {missing_preds}")

    # --- Build formula ---
    terms: List[str] = []
    for p in predictors:
        if pd.api.types.is_numeric_dtype(df[p]):
            terms.append(p)
        else:
            terms.append(f"C({p})")

    if not terms:
        raise ValueError("No predictors provided after processing.")

    formula = f"{outcome} ~ " + " + ".join(terms)

    # --- Prepare data ---
    model_df = df[[outcome] + predictors].dropna()

    if model_df.shape[0] < 3:
        raise ValueError("Not enough complete rows to fit regression (need >= 3).")

    # --- Fit model ---
    fitted = smf.ols(formula=formula, data=model_df).fit()

    # --- Confidence intervals ---
    ci = fitted.conf_int()
    ci.columns = ["ci_lower", "ci_upper"]

    coef_table: Dict[str, Dict[str, float]] = {}
    for term in fitted.params.index:
        coef_table[str(term)] = {
            "coefficient": float(fitted.params[term]),
            "std_error": float(fitted.bse[term]),
            "t_value": float(fitted.tvalues[term]),
            "p_value": float(fitted.pvalues[term]),
            "ci_lower": float(ci.loc[term, "ci_lower"]),
            "ci_upper": float(ci.loc[term, "ci_upper"]),
        }

    # --- Structured output ---
    out: Dict[str, Any] = {
        "outcome": str(outcome),
        "predictors": [str(p) for p in predictors],
        "n_rows_used": int(model_df.shape[0]),
        "formula": str(formula),
        "r_squared": float(fitted.rsquared),
        "adj_r_squared": float(fitted.rsquared_adj),
        "f_statistic": float(fitted.fvalue) if fitted.fvalue is not None else None,
        "f_pvalue": float(fitted.f_pvalue) if fitted.f_pvalue is not None else None,
        "df_model": float(fitted.df_model),
        "df_resid": float(fitted.df_resid),
        "coefficients": coef_table,
    }

    # --- Human-readable coefficient lines ---
    coef_lines = []
    for term, vals in coef_table.items():
        coef_lines.append(
            f"- {term}: b = {vals['coefficient']:.4f}, "
            f"SE = {vals['std_error']:.4f}, "
            f"t = {vals['t_value']:.4f}, "
            f"p = {vals['p_value']:.4g}, "
            f"95% CI [{vals['ci_lower']:.4f}, {vals['ci_upper']:.4f}]"
        )

    coef_text = "\n".join(coef_lines)

    # --- Human-readable summary ---
    summary_text = (
        f"Fitted multiple linear regression.\n"
        f"Outcome: {outcome}\n"
        f"Predictors: {', '.join(predictors)}\n"
        f"Rows used: {model_df.shape[0]}\n"
        f"Formula: {formula}\n"
        f"R-squared: {fitted.rsquared:.4f}\n"
        f"Adjusted R-squared: {fitted.rsquared_adj:.4f}\n"
        f"F-statistic: {fitted.fvalue:.4f}\n"
        f"Degrees of freedom: model={fitted.df_model:.0f}, residual={fitted.df_resid:.0f}\n"
        f"Model p-value: {fitted.f_pvalue:.4g}\n\n"
        f"Coefficients:\n{coef_text}"
    )

    return make_tool_result(
        name="multiple_linear_regression",
        text=summary_text,
        structured=out,
    )
