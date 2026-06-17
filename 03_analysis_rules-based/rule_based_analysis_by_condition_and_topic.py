#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

from rule_based_analysis_by_condition import (
    build_metrics_dataframe,
    CONDITION_A,
    CONDITION_B,
)

VALID_CHOSEN_TOPICS = {"Breakfast", "Watches", "Vacation"}


def detect_metric_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    exclude = {"id", "timestamp", "date", "condition", "chosen_topic"}
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    continuous, binary = [], []
    for col in numeric_cols:
        vals = set(df[col].dropna().unique().tolist())
        if vals.issubset({0, 1}):
            binary.append(col)
        else:
            continuous.append(col)
    return continuous, binary


def fit_models(df: pd.DataFrame) -> pd.DataFrame:
    continuous_cols, binary_cols = detect_metric_types(df)
    rows = []

    for metric in continuous_cols:
        tmp = df[["condition", "chosen_topic", metric]].dropna().copy()
        if len(tmp) < 10 or tmp[metric].nunique() < 2:
            continue

        formula = f"Q('{metric}') ~ C(condition) + C(chosen_topic)"
        try:
            model = smf.ols(formula, data=tmp).fit()
            coef_name = f"C(condition)[T.{CONDITION_B}]"
            coef = model.params.get(coef_name, np.nan)
            pval = model.pvalues.get(coef_name, np.nan)

            rows.append({
                "metric": metric,
                "metric_type": "continuous",
                "model_family": "OLS",
                "n_used": int(model.nobs),
                "condition_b_vs_a_coef": float(coef) if pd.notna(coef) else np.nan,
                "condition_b_vs_a_pvalue": float(pval) if pd.notna(pval) else np.nan,
                "r_squared": float(model.rsquared),
            })
        except Exception as e:
            rows.append({
                "metric": metric,
                "metric_type": "continuous",
                "model_family": "OLS",
                "n_used": len(tmp),
                "error": str(e),
            })

    for metric in binary_cols:
        tmp = df[["condition", "chosen_topic", metric]].dropna().copy()
        if len(tmp) < 10 or tmp[metric].nunique() < 2:
            continue

        formula = f"Q('{metric}') ~ C(condition) + C(chosen_topic)"
        try:
            model = smf.logit(formula, data=tmp).fit(disp=False)
            coef_name = f"C(condition)[T.{CONDITION_B}]"
            coef = model.params.get(coef_name, np.nan)
            pval = model.pvalues.get(coef_name, np.nan)

            rows.append({
                "metric": metric,
                "metric_type": "binary",
                "model_family": "Logit",
                "n_used": int(model.nobs),
                "condition_b_vs_a_coef": float(coef) if pd.notna(coef) else np.nan,
                "condition_b_vs_a_odds_ratio": float(np.exp(coef)) if pd.notna(coef) else np.nan,
                "condition_b_vs_a_pvalue": float(pval) if pd.notna(pval) else np.nan,
                "pseudo_r_squared": float(model.prsquared),
            })
        except Exception as e:
            rows.append({
                "metric": metric,
                "metric_type": "binary",
                "model_family": "Logit",
                "n_used": len(tmp),
                "error": str(e),
            })

    return pd.DataFrame(rows)


def main() -> None:
    here = Path(__file__).resolve().parent
    dialogs_path = here.parent / "dialogs.json"

    df = build_metrics_dataframe(dialogs_path)
    df = df[df["chosen_topic"].isin(VALID_CHOSEN_TOPICS)].copy()

    df["condition"] = pd.Categorical(
        df["condition"],
        categories=[CONDITION_A, CONDITION_B],
        ordered=True,
    )
    df["chosen_topic"] = pd.Categorical(
        df["chosen_topic"],
        categories=["Breakfast", "Watches", "Vacation"],
        ordered=True,
    )

    df.to_csv(here / "rule_based_metrics_by_conversation_condition_topic.csv", index=False)

    (
        df.groupby(["condition", "chosen_topic"])
        .agg(
            n_conversations=("id", "count"),
            mean_exchanges=("num_exchanges", "mean"),
            mean_feedback=("feedback", "mean"),
            mean_user_words=("total_user_words", "mean"),
            mean_offtopic_markers=("user_divergence_marker_count", "mean"),
        )
        .reset_index()
        .to_csv(here / "rule_based_condition_topic_descriptives.csv", index=False)
    )

    fit_models(df).to_csv(here / "rule_based_condition_topic_models.csv", index=False)

    print("Saved:")
    print(" - rule_based_metrics_by_conversation_condition_topic.csv")
    print(" - rule_based_condition_topic_descriptives.csv")
    print(" - rule_based_condition_topic_models.csv")


if __name__ == "__main__":
    main()