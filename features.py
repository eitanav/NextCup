"""Build the modeling table: one row per drink event, with answer columns and features.

Shared by the EDA notebook (M2) and the model (M3), so both see the same cleaning decisions.
The decisions themselves are documented in eda_plan.md, section 3.
"""

from pathlib import Path

import pandas as pd

DATA_FILE = Path(__file__).parent / "data" / "caffeine_events.csv"
DAY = ["person_id", "recall_day"]

# A single drink above this is almost surely a whole thermos or a whole day reported once
# (one brewed coffee of 3,600 mg is about 38 cups). Keep the row, cap the amount.
MG_CAP = 1000


def load_events(path: Path = DATA_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def merge_same_time(df: pd.DataFrame) -> pd.DataFrame:
    """Drinks one person logged at the same minute become one drink event.

    Otherwise the first of them gets a "next drink in 0 hours" answer, which is not a real gap.
    The merged event keeps the category and item of its most caffeinated part.
    """
    df = df.drop_duplicates().sort_values("caffeine_mg", ascending=False)
    return df.groupby(DAY + ["time"], as_index=False, sort=False).agg(
        weekday=("weekday", "first"),
        hour=("hour", "first"),
        category=("category", "first"),
        item=("item", "first"),
        caffeine_mg=("caffeine_mg", "sum"),
        occasion=("occasion", "first"),
        age=("age", "first"),
        sex=("sex", "first"),
        n_items=("item", "size"),
    )


def add_answers_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the two answer columns and the features known at the moment of the drink."""
    df = df.sort_values(DAY + ["hour"]).reset_index(drop=True)
    df["caffeine_mg"] = df["caffeine_mg"].clip(upper=MG_CAP)
    day = df.groupby(DAY)

    # Features: only what is known when the user logs this drink.
    df["drinks_before"] = day.cumcount()
    df["mg_before"] = day["caffeine_mg"].cumsum() - df["caffeine_mg"]
    df["hours_since_prev"] = df["hour"] - day["hour"].shift(1)  # NaN for the day's first drink

    # Answers: built from the person's next drink the same day.
    next_hour = day["hour"].shift(-1)
    df["has_next"] = next_hour.notna().astype(int)
    df["hours_to_next"] = next_hour - df["hour"]  # NaN when there is no next drink
    return df


def build_table(path: Path = DATA_FILE) -> pd.DataFrame:
    return add_answers_and_features(merge_same_time(load_events(path)))
