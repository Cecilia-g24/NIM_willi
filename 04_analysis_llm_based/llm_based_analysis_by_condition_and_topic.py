#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from llm_based_analysis_by_condition_tqdm import (
    CONDITION_A,
    CONDITION_B,
    VALID_CHOSEN_TOPICS,
    build_metrics_dataframe,
)


def detect_metric_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    exclude = {"id", "timestamp", "date", "condition", "feedback", "chosen_topic"}
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
        except Exception as e:  # noqa: BLE001
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
        except Exception as e:  # noqa: BLE001
            rows.append({
                "metric": metric,
                "metric_type": "binary",
                "model_family": "Logit",
                "n_used": len(tmp),
                "error": str(e),
            })

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="LLM-based conversation analysis controlling for topic.")
    parser.add_argument("--dialogs", type=Path, default=here.parent / "dialogs.json", help="Path to dialogs.json")
    parser.add_argument("--cache", type=Path, default=here / "llm_annotation_cache.jsonl", help="Path to annotation cache")
    parser.add_argument("--model", default="gpt-5.4-mini", help="Model ID; only used for dialogs not already cached")
    parser.add_argument("--max-dialogs", type=int, default=None, help="Optional cap for test runs")
    parser.add_argument("--max-retries", type=int, default=5, help="API retries per dialog")
    parser.add_argument("--retry-base-seconds", type=float, default=2.0, help="Base seconds for exponential backoff")
    parser.add_argument("--request-pause-seconds", type=float, default=0.0, help="Optional pause between requests")
    parser.add_argument("--overwrite-cache", action="store_true", help="Ignore any existing cache and annotate again")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent

    df = build_metrics_dataframe(
        args.dialogs,
        cache_path=args.cache,
        model=args.model,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        request_pause_seconds=args.request_pause_seconds,
        overwrite_cache=args.overwrite_cache,
        max_dialogs=args.max_dialogs,
    )
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

    df.to_csv(here / "llm_metrics_by_conversation_condition_topic.csv", index=False)

    (
        df.groupby(["condition", "chosen_topic"])
        .agg(
            n_conversations=("id", "count"),
            mean_engagement=("llm_engagement_score", "mean"),
            mean_disclosure=("llm_mean_disclosure", "mean"),
            mean_response_quality=("llm_assistant_response_quality_mean", "mean"),
            mean_offtopic_score=("llm_mean_offtopic_score", "mean"),
            mean_frustration=("llm_frustration_score", "mean"),
            mean_naturalness=("llm_naturalness_score", "mean"),
        )
        .reset_index()
        .to_csv(here / "llm_condition_topic_descriptives.csv", index=False)
    )

    fit_models(df).to_csv(here / "llm_condition_topic_models.csv", index=False)

    print("Saved:")
    print(" - llm_metrics_by_conversation_condition_topic.csv")
    print(" - llm_condition_topic_descriptives.csv")
    print(" - llm_condition_topic_models.csv")


if __name__ == "__main__":
    main()
