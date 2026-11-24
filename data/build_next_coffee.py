"""Build a labeled next-coffee dataset from NHANES XPT files and the USDA food-code map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

DATA = Path(__file__).resolve().parent
CDC = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"
FILES = {name: CDC + name for name in ("DR1IFF_J.xpt", "DR2IFF_J.xpt", "DEMO_J.xpt")}
MAPPING = "WWEIA1718_foodcat_FNDDS.xlsx"
FILES[MAPPING] = "https://www.ars.usda.gov/ARSUserFiles/80400530/apps/" + MAPPING
DAY = ["person_id", "recall_day"]
WEEKDAYS = dict(enumerate("Sun Mon Tue Wed Thu Fri Sat".split(), 1))
CATEGORIES = {7302: "coffee", 7304: "tea", 7102: "soda", 7202: "soda",
              7104: "sport_energy", 7206: "sport_energy"}
COLUMNS = [
    "person_id", "recall_day", "split", "weekday", "age_years", "sex",
    "clock_time", "minute_of_day", "coffee_mg", "event_caffeine_mg",
    "day_caffeine_mg_to_now", "prior_coffee_events", "prior_caffeine_events",
    "minutes_since_prior_coffee", "same_minute_items", "high_dose_flag",
    "has_next_coffee", "minutes_to_next_coffee",
]
HIGH_DOSE_MG = 1000


def ensure_sources(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = raw_dir / name
        if not target.exists():
            temp = target.with_suffix(".part")
            with urlopen(url, timeout=90) as response:
                temp.write_bytes(response.read())
            temp.replace(target)


def load_recall(raw_dir: Path, day: int) -> pd.DataFrame:
    p = f"DR{day}"
    aliases = {
        "SEQN": "person_id", f"{p}ILINE": "item_line", f"{p}DRSTZ": "recall_status",
        f"{p}DAY": "weekday_code", f"{p}_020": "seconds", f"{p}IFDCD": "food_code",
        f"{p}ICAFF": "caffeine_mg",
    }
    frame = pd.read_sas(raw_dir / f"{p}IFF_J.xpt")
    return frame[list(aliases)].rename(columns=aliases).assign(recall_day=day)


def build_source(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    demography = pd.read_sas(raw_dir / "DEMO_J.xpt")
    demography = demography[["SEQN", "RIDAGEYR", "RIAGENDR"]].rename(
        columns={"SEQN": "person_id", "RIDAGEYR": "age_years", "RIAGENDR": "sex"}
    )
    codes = pd.read_excel(raw_dir / MAPPING, usecols=["food_code", "category_number"])
    source = pd.concat([load_recall(raw_dir, d) for d in (1, 2)], ignore_index=True)
    report = {
        "raw_items": int(len(source)),
        "raw_items_by_recall": {str(k): int(v) for k, v in source.recall_day.value_counts().items()},
        "missing_caffeine_raw": int(source.caffeine_mg.isna().sum()),
        "unreliable_items": int(source.recall_status.ne(1).sum()),
        "unreliable_person_days": int(source.loc[source.recall_status.ne(1), DAY].drop_duplicates().shape[0]),
        "repeated_item_keys": int(source.duplicated(DAY + ["item_line"]).sum()),
    }
    if report["repeated_item_keys"]:
        raise ValueError("Repeated source item keys. Review upstream records before rebuilding.")

    reliable = source.loc[source.recall_status.eq(1)]
    joined = reliable.merge(demography, on="person_id", how="left", validate="many_to_one")
    report["missing_demographics_reliable_items"] = int(joined[["age_years", "sex"]].isna().any(axis=1).sum())
    report["underage_items_reliable"] = int(joined.age_years.lt(18).sum())
    adults = joined.loc[joined.age_years.ge(18) & joined.sex.isin([1, 2])]
    joined = adults.merge(codes, on="food_code", how="left", validate="many_to_one")
    report["unmapped_food_items_adults"] = int(joined.category_number.isna().sum())
    if report["unmapped_food_items_adults"]:
        raise ValueError("Review unmapped adult food codes before labeling the last coffee of a day.")
    drinks = joined.loc[joined.category_number.isin(CATEGORIES)].copy()
    drinks["category"] = drinks.category_number.map(CATEGORIES)
    invalid = (
        ~drinks.seconds.between(0, 86399)
        | ~drinks.weekday_code.isin(WEEKDAYS)
        | ~drinks.caffeine_mg.ge(0)
    )
    bad_days = drinks.loc[invalid, DAY].drop_duplicates()
    report["invalid_selected_beverage_items"] = int(invalid.sum())
    report["excluded_person_days_with_invalid_beverage"] = len(bad_days)
    if len(bad_days):
        drinks = drinks.merge(bad_days.assign(exclude_day=True), on=DAY, how="left")
        drinks = drinks.loc[drinks.exclude_day.isna()].drop(columns="exclude_day")

    ints = DAY + ["item_line", "food_code", "age_years"]
    drinks[ints] = drinks[ints].astype("int64")
    drinks["weekday"] = drinks.weekday_code.map(WEEKDAYS)
    drinks["sex"] = drinks.sex.map({1: "male", 2: "female"})
    return drinks, report


def make_coffee_table(drinks: pd.DataFrame, min_mg: float = 10) -> tuple[pd.DataFrame, dict]:
    eligible = drinks.loc[drinks.caffeine_mg.ge(min_mg)].copy()
    # SAS transports some numeric zeros as a tiny positive value.
    eligible["minute_of_day"] = (eligible.seconds.round().astype("int64") // 60)
    eligible["coffee_mg"] = eligible.caffeine_mg.where(eligible.category.eq("coffee"), 0)
    eligible["coffee_items"] = eligible.category.eq("coffee").astype(int)
    events = eligible.groupby(DAY + ["minute_of_day"], as_index=False).agg(
        weekday=("weekday", "first"), age_years=("age_years", "first"), sex=("sex", "first"),
        coffee_mg=("coffee_mg", "sum"), event_caffeine_mg=("caffeine_mg", "sum"),
        same_minute_items=("item_line", "size"), coffee_items=("coffee_items", "sum"),
    )
    by_day = events.groupby(DAY, sort=False)
    events["prior_caffeine_events"] = by_day.cumcount()
    events["day_caffeine_mg_to_now"] = by_day.event_caffeine_mg.cumsum()
    coffee = events.loc[events.coffee_items.gt(0)].copy()
    by_day = coffee.groupby(DAY, sort=False)
    coffee["prior_coffee_events"] = by_day.cumcount()
    coffee["minutes_since_prior_coffee"] = by_day.minute_of_day.diff().astype("Int64")
    next_minute = by_day.minute_of_day.shift(-1)
    coffee["has_next_coffee"] = next_minute.notna().astype(int)
    coffee["minutes_to_next_coffee"] = (next_minute - coffee.minute_of_day).astype("Int64")
    coffee["high_dose_flag"] = coffee.event_caffeine_mg.ge(HIGH_DOSE_MG).astype(int)
    coffee["clock_time"] = (
        (coffee.minute_of_day // 60).astype(str).str.zfill(2) + ":"
        + (coffee.minute_of_day % 60).astype(str).str.zfill(2)
    )
    for column in ["coffee_mg", "event_caffeine_mg", "day_caffeine_mg_to_now"]:
        coffee[column] = coffee[column].round(3)
    # Allocate by a stable hash of the public participant ID to keep both recalls together.
    people = coffee.person_id.unique()
    split = {
        person: ("test" if hashlib.sha256(f"NextCup-v1:{person}".encode()).digest()[0] < 51 else "train")
        for person in people
    }
    coffee["split"] = coffee.person_id.map(split)

    report = {
        "threshold_mg_per_item": min_mg,
        "eligible_beverage_items": len(eligible),
        "eligible_caffeine_events": len(events),
        "merged_caffeine_events": int(events.same_minute_items.gt(1).sum()),
        "eligible_coffee_items": int(eligible.coffee_items.sum()),
        "merged_coffee_events": int(coffee.same_minute_items.gt(1).sum()),
        "coffee_rows": len(coffee),
        "coffee_person_days": by_day.ngroups,
        "coffee_people": len(people),
        "next_coffees": int(coffee.has_next_coffee.sum()),
        "coffee_on_half_hour": int(coffee.minute_of_day.mod(30).eq(0).sum()),
        "high_dose_events": int(coffee.high_dose_flag.sum()),
        "max_event_caffeine_mg": float(coffee.event_caffeine_mg.max()),
    }
    return coffee[COLUMNS].reset_index(drop=True), report


def validate(table: pd.DataFrame) -> None:
    key = DAY + ["minute_of_day"]
    expected_gap = table.groupby(DAY).minute_of_day.shift(-1) - table.minute_of_day
    missing = table.isna().sum()
    first_coffee = table.prior_coffee_events.eq(0)
    split_count = table.groupby("person_id").split.nunique()
    required = [col for col in COLUMNS if col not in ["minutes_since_prior_coffee", "minutes_to_next_coffee"]]
    if (
        table.empty or table.duplicated(key).any() or missing[required].any()
        or not expected_gap.astype("Int64").equals(table.minutes_to_next_coffee)
        or not table.has_next_coffee.eq(expected_gap.notna()).all()
        or not expected_gap.dropna().gt(0).all()
        or not table.minutes_since_prior_coffee.isna().eq(first_coffee).all()
        or not table.minute_of_day.between(0, 1439).all()
        or not table.age_years.between(18, 80).all()
        or not table.day_caffeine_mg_to_now.ge(table.event_caffeine_mg).all()
        or not table.event_caffeine_mg.ge(table.coffee_mg).all()
        or not split_count.eq(1).all()
        or set(table.split) != {"train", "test"}
    ):
        raise ValueError("Validation failed: keys, label, missingness, amounts, or split are inconsistent.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DATA / "raw")
    parser.add_argument("--output-dir", type=Path, default=DATA)
    args = parser.parse_args()
    ensure_sources(args.raw_dir)
    drinks, report = build_source(args.raw_dir)
    table, cohort = make_coffee_table(drinks)
    validate(table)
    sensitivity = [make_coffee_table(drinks, threshold)[1] for threshold in (1, 20)]
    report.update({
        "cohort": cohort, "sensitivity": sensitivity,
        "source_sha256": {name: hashlib.sha256((args.raw_dir / name).read_bytes()).hexdigest() for name in FILES},
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "next_coffee_events.csv", index=False, lineterminator="\n")
    (args.output_dir / "data_quality.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{len(table):,} coffee events from {cohort['coffee_people']:,} people")


if __name__ == "__main__":
    main()
