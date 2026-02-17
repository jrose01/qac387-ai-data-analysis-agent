from __future__ import annotations

from typing import Optional, List, Dict, Any
import pandas as pd
import statsmodels.formula.api as smf


def multiple_linear_regression(
    df: pd.DataFrame,
    outcome: str,
    predictors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Fit multiple linear regression using statsmodels OLS (formula interface).

    - outcome must exist
    - predictors optional:
        if None -> all columns except outcome (you may want to restrict in your course)
    - drops rows with missing values in outcome/predictors
    - returns a JSON-safe dictionary of key results
    """
    if outcome not in df.columns:
        raise ValueError(f"Outcome column '{outcome}' not found in dataframe.")

    if predictors is None:
        predictors = [c for c in df.columns if c != outcome]

    missing_preds = [p for p in predictors if p not in df.columns]
    if missing_preds:
        raise ValueError(f"Predictor(s) not found: {missing_preds}")

    # Build formula: numeric predictors as-is, non-numeric wrapped as C(...)
    terms: List[str] = []
    for p in predictors:
        if pd.api.types.is_numeric_dtype(df[p]):
            terms.append(p)
        else:
            terms.append(f"C({p})")

    if not terms:
        raise ValueError("No predictors provided after processing.")

    formula = f"{outcome} ~ " + " + ".join(terms)

    model_df = df[[outcome] + predictors].dropna()
    if model_df.shape[0] < 3:
        raise ValueError("Not enough complete rows to fit regression (need >= 3).")

    fitted = smf.ols(formula=formula, data=model_df).fit()

    out: Dict[str, Any] = {
        "outcome": str(outcome),
        "predictors": [str(p) for p in predictors],
        "n_rows_used": int(model_df.shape[0]),
        "formula": str(formula),
        "r_squared": float(fitted.rsquared),
        "adj_r_squared": float(fitted.rsquared_adj),
        "params": {str(k): float(v) for k, v in fitted.params.items()},
        "pvalues": {str(k): float(v) for k, v in fitted.pvalues.items()},
        "stderr": {str(k): float(v) for k, v in fitted.bse.items()},
    }
    return out
