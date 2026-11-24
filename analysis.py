"""Shared descriptive statistics and held-out baselines for NextCup."""
from pathlib import Path

import pandas as pd

DATA_FILE = Path(__file__).resolve().parent / "data" / "next_coffee_events.csv"
LABEL = "has_next_coffee"
WAIT = "minutes_to_next_coffee"
FEATURES = [
    "minute_of_day", "weekday", "coffee_mg", "event_caffeine_mg",
    "day_caffeine_mg_to_now", "prior_coffee_events", "prior_caffeine_events",
    "minutes_since_prior_coffee",
]


def load_events() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE, dtype={WAIT: "Int64", "minutes_since_prior_coffee": "Int64"})


def baseline_scores(events: pd.DataFrame) -> dict:
    train = events.loc[events.split.eq("train")]
    test = events.loc[events.split.eq("test")]
    if set(train.person_id) & set(test.person_id):
        raise ValueError("Participant appears in both train and test.")
    majority = int(train[LABEL].mode().iloc[0])
    median = float(train[WAIT].median())
    return {
        "majority": majority,
        "accuracy": float(test[LABEL].eq(majority).mean()),
        "median_minutes": median,
        "mae_minutes": float(test[WAIT].sub(median).abs().mean()),
        "positive_test_events": int(test[LABEL].sum()),
        "test_events": len(test),
        "test_people": test.person_id.nunique(),
    }
