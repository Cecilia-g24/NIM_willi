from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_ind


# -----------------------------------------------------------------------------
# Default file paths for the project structure shown by the user.
#
# Expected structure:
# PROJECT_ROBOT_CHATS/
#   data/data_clean/dialogs_full.json
#   02_visitor_fraction/
#       visitor_fraction.py   <-- this script
#       NIM Besucherzahlen insgesamt_flat.xlsx
#
# Because this script lives inside 02_visitor_fraction, the project root is the
# parent directory of this file's directory.
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DIALOG_JSON = PROJECT_ROOT / "data" / "data_clean" / "dialogs_full.json"
VISITOR_EXCEL = SCRIPT_DIR / "NIM Besucherzahlen insgesamt_flat.xlsx"
DAILY_FRACTION_CSV = SCRIPT_DIR / "daily_robot_interaction_fractions.csv"
BOXPLOT_PNG = SCRIPT_DIR / "daily_robot_interaction_fractions_boxplot.png"

# Experimental conditions used in the dialogs JSON.
CONDITION_A = "Condition A (Willi)"
CONDITION_B = "Condition B (WV-34)"
VALID_EXPERIMENTAL_CONDITIONS = {CONDITION_A, CONDITION_B}
CONDITION_DISPLAY_NAMES = {
    CONDITION_A: "Condition A (Willi)",
    CONDITION_B: "Condition B (WV-34)",
}


def load_visitor_counts(excel_file: Path) -> pd.DataFrame:
    """Load daily museum visitor counts from the flattened Excel export.

    The source file is arranged as repeated pairs of columns, where each pair
    contains a date column and a visitor-count column. Closed days are encoded
    as the string "geschlossen" and are converted to 0 visitors.
    """
    raw_visitors = pd.read_excel(excel_file, sheet_name=0)
    visitor_tables: list[pd.DataFrame] = []

    for column_index in range(0, len(raw_visitors.columns), 2):
        subset = raw_visitors.iloc[1:, [column_index, column_index + 1]].copy()
        subset.columns = ["date", "visitors"]
        visitor_tables.append(subset)

    visitors = pd.concat(visitor_tables).dropna(subset=["date"])
    visitors["visitors"] = visitors["visitors"].mask(visitors["visitors"] == "geschlossen", 0)
    visitors["visitors"] = pd.to_numeric(visitors["visitors"], errors="coerce").fillna(0)
    visitors["date"] = pd.to_datetime(visitors["date"], errors="coerce").dt.normalize()

    visitors = visitors.dropna(subset=["date"]).drop_duplicates("date")
    visitors["visitors"] = visitors["visitors"].astype(int)
    return visitors.sort_values("date").reset_index(drop=True)


def load_dialogs_from_json(json_file: Path) -> pd.DataFrame:
    """Load dialogs directly from dialogs.json.

    The JSON is expected to be a list of dialog objects with at least:
      - timestamp
      - condition
      - id (optional but useful for debugging)
    """
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("dialogs.json must contain a list of dialog objects.")

    dialogs = pd.DataFrame(data)
    required_columns = {"timestamp", "condition"}
    missing = required_columns - set(dialogs.columns)
    if missing:
        raise ValueError(f"dialogs.json is missing required fields: {sorted(missing)}")

    dialogs = dialogs[dialogs["condition"].isin(VALID_EXPERIMENTAL_CONDITIONS)].copy()
    dialogs["date"] = pd.to_datetime(dialogs["timestamp"], errors="coerce").dt.normalize()
    dialogs = dialogs.dropna(subset=["date"])

    return dialogs.sort_values("date").reset_index(drop=True)


def load_condition_schedule_from_json(json_file: Path) -> pd.DataFrame:
    """Infer which condition was active on each date.

    This assumes a between-days design: each date belongs to exactly one
    experimental condition. If a date contains more than one condition in the
    JSON, the script raises an error because the visitor denominator would no
    longer be interpretable at the day level.
    """
    dialogs = load_dialogs_from_json(json_file)

    condition_counts = dialogs.groupby("date")["condition"].nunique()
    ambiguous_dates = condition_counts[condition_counts > 1].index.tolist()
    if ambiguous_dates:
        formatted_dates = ", ".join(pd.Timestamp(date).strftime("%Y-%m-%d") for date in ambiguous_dates[:10])
        raise ValueError(
            "Found dates with more than one experimental condition in dialogs.json. "
            f"Examples: {formatted_dates}"
        )

    schedule = dialogs[["date", "condition"]].drop_duplicates("date")
    return schedule.sort_values("date").reset_index(drop=True)


def load_daily_interactions_from_json(json_file: Path) -> pd.DataFrame:
    """Count recorded interactions per day and condition directly from JSON."""
    dialogs = load_dialogs_from_json(json_file)
    return dialogs.groupby(["date", "condition"]).size().reset_index(name="interactions")


def calculate_daily_interaction_fractions(
    json_file: Path,
    visitor_excel: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Create a day-level dataset with zero-interaction days preserved.

    Logic:
      1. Load all visitor days from the museum file.
      2. Infer the active condition for each experimental date from dialogs.json.
      3. Left-join daily interaction counts onto that date-condition schedule.
      4. Fill missing interaction counts with 0.

    This preserves valid museum days with zero conversations, instead of silently
    dropping them.
    """
    visitors = load_visitor_counts(visitor_excel)
    condition_schedule = load_condition_schedule_from_json(json_file)
    daily_interactions = load_daily_interactions_from_json(json_file)

    day_level = pd.merge(condition_schedule, visitors, on="date", how="left")
    day_level = pd.merge(day_level, daily_interactions, on=["date", "condition"], how="left")

    day_level["interactions"] = day_level["interactions"].fillna(0).astype(int)
    day_level["visitors"] = day_level["visitors"].fillna(0).astype(int)

    # Closed days (0 visitors) cannot contribute a meaningful rate.
    day_level = day_level[day_level["visitors"] > 0].copy()
    day_level["interaction_fraction"] = day_level["interactions"] / day_level["visitors"]

    day_level = day_level.sort_values(["date", "condition"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    day_level.to_csv(output_csv, index=False)
    return day_level


def calculate_pooled_condition_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pooled interaction rates per condition.

    This answers:
        total interactions in condition / total visitors exposed to that condition
    """
    pooled = (
        df.groupby("condition", as_index=False)
        .agg(
            total_interactions=("interactions", "sum"),
            total_visitors=("visitors", "sum"),
            n_days=("date", "count"),
            mean_daily_fraction=("interaction_fraction", "mean"),
            median_daily_fraction=("interaction_fraction", "median"),
        )
        .sort_values("condition")
        .reset_index(drop=True)
    )
    pooled["pooled_interaction_fraction"] = pooled["total_interactions"] / pooled["total_visitors"]
    return pooled


def print_fraction_summary(df: pd.DataFrame) -> None:
    """Print both day-level and pooled condition summaries."""
    group_a = df.loc[df["condition"] == CONDITION_A, "interaction_fraction"]
    group_b = df.loc[df["condition"] == CONDITION_B, "interaction_fraction"]

    stats_a = group_a.describe()
    stats_b = group_b.describe()

    # Nonparametric comparison of daily fractions.
    _, u_p_value = mannwhitneyu(group_a, group_b, alternative="two-sided")
    # Welch t-test for daily fractions.
    _, t_p_value = ttest_ind(group_a, group_b, equal_var=False, nan_policy="omit")

    print("DAY-LEVEL INTERACTION FRACTIONS (INCLUDING ZERO-INTERACTION DAYS)")
    print(f"{'Condition':<22} | {'Mean Fraction':<15} | {'Std Dev':<10} | {'N (Days)':<10}")
    print("-" * 70)
    print(
        f"{CONDITION_DISPLAY_NAMES[CONDITION_A]:<22} | "
        f"{stats_a['mean']:<15.4f} | {stats_a['std']:<10.4f} | {int(stats_a['count']):<10}"
    )
    print(
        f"{CONDITION_DISPLAY_NAMES[CONDITION_B]:<22} | "
        f"{stats_b['mean']:<15.4f} | {stats_b['std']:<10.4f} | {int(stats_b['count']):<10}"
    )
    print("-" * 70)
    print(f"Mann-Whitney U p-value: {u_p_value:.4f}")
    print(f"Welch's T-Test p-value:  {t_p_value:.4f}")

    if u_p_value < 0.05:
        print("\nCONCLUSION (day-level): There IS a statistically significant difference.")
    else:
        print("\nCONCLUSION (day-level): There is NO statistically significant difference.")

    pooled = calculate_pooled_condition_rates(df)
    pooled_display = pooled.copy()
    pooled_display["condition_name"] = pooled_display["condition"].replace(CONDITION_DISPLAY_NAMES)

    print("\nPOOLED CONDITION-LEVEL RATES")
    print(
        f"{'Condition':<22} | {'Total Interactions':<18} | {'Total Visitors':<15} | "
        f"{'Pooled Fraction':<15} | {'N (Days)':<10}"
    )
    print("-" * 95)
    for _, row in pooled_display.iterrows():
        print(
            f"{row['condition_name']:<22} | "
            f"{int(row['total_interactions']):<18} | "
            f"{int(row['total_visitors']):<15} | "
            f"{row['pooled_interaction_fraction']:<15.4f} | "
            f"{int(row['n_days']):<10}"
        )


def save_fraction_plot(df: pd.DataFrame, output_file: Path) -> None:
    plot_data = df.copy()
    plot_data["Condition Name"] = plot_data["condition"].replace(CONDITION_DISPLAY_NAMES)

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.boxplot(
        x="Condition Name",
        y="interaction_fraction",
        hue="Condition Name",
        data=plot_data,
        palette="Set2",
        showfliers=False,
        legend=False,
    )
    sns.stripplot(x="Condition Name", y="interaction_fraction", data=plot_data, color=".3", alpha=0.5)

    plt.title("Distribution of Daily Interaction Fractions by Robot Type", fontsize=14)
    plt.ylabel("Interaction Fraction (Interactions / Total Visitors)", fontsize=12)
    plt.xlabel("Robot Condition", fontsize=12)
    plt.ylim(bottom=0)
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.close()


def main() -> None:
    fractions = calculate_daily_interaction_fractions(
        json_file=DIALOG_JSON,
        visitor_excel=VISITOR_EXCEL,
        output_csv=DAILY_FRACTION_CSV,
    )
    print_fraction_summary(fractions)
    save_fraction_plot(fractions, BOXPLOT_PNG)

    print("\nSaved day-level data to:", DAILY_FRACTION_CSV)
    print("Saved boxplot to:", BOXPLOT_PNG)


if __name__ == "__main__":
    main()
