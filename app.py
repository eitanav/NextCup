"""NextCup M2 Streamlit dashboard on the cleaned NHANES coffee events."""
import altair as alt
import pandas as pd
import streamlit as st

from analysis import LABEL, WAIT, baseline_scores, load_events

st.set_page_config(page_title="NextCup | חקירת נתונים", page_icon="☕", layout="wide")

st.markdown(
    """
    <style>
      [data-testid="stMainBlockContainer"] {max-width: 1320px; padding-top: 3.5rem;}
      .stMarkdown, h1, h2, h3, p, li {direction: rtl; text-align: right;}
      h1 {color: #123C34; letter-spacing: -.05em;}
      [data-testid="stMetric"] {border-radius: 13px; background: #F3F7F4;}
      [data-testid="stMetricLabel"] p {font-weight: 600;}
      [data-testid="stAppDeployButton"] {display:none;}
      [data-testid="stMetricValue"], [data-testid="stDataFrame"] {direction: ltr;}
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {direction: rtl;}
      .insight-box {padding: 15px 20px; background: #E5F1E9; border-radius: 12px;
        border-right: 4px solid #367754; color: #153F34; direction: rtl; text-align: right;
        margin: .15rem 0 .8rem; font-size: 1.04rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

WEEK = "Sun Mon Tue Wed Thu Fri Sat".split()
DAYS = dict(zip(WEEK, ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"]))
MIN_GROUP = 20


@st.cache_data
def data() -> pd.DataFrame:
    return load_events()


def summary_line(rows: pd.DataFrame) -> str:
    morning = rows.loc[rows.minute_of_day.between(360, 599)]
    afternoon = rows.loc[rows.minute_of_day.between(840, 1079)]
    if min(len(morning), len(afternoon)) < MIN_GROUP:
        return "להשוואת בוקר מול אחר הצהריים נדרשים לפחות 20 אירועי קפה בכל חלון."
    return (
        f"גילינו שב־{morning[LABEL].mean():.1%} מאירועי הקפה מ־06:00 עד 09:59 דווח על"
        f" עוד קפה באותו יום, לעומת {afternoon[LABEL].mean():.1%} מ־14:00 עד 17:59."
        " שעת הכוס הנוכחית נותנת נקודת פתיחה לחיזוי."
    )


def rate_chart(rows: pd.DataFrame) -> alt.Chart:
    plot = (rows.assign(block=rows.minute_of_day // 120 * 2)
            .groupby("block")[LABEL].agg(events="size", rate="mean")
            .reindex(range(0, 24, 2)).rename_axis("hour").reset_index())
    plot.loc[plot.events.fillna(0).lt(MIN_GROUP), "rate"] = None
    plot["events"] = plot.events.fillna(0).astype(int)
    return alt.Chart(plot).mark_line(point=alt.OverlayMarkDef(size=65, filled=True, color="#206A53"), strokeWidth=3,
                                     color="#206A53", invalid="break-paths-show-domains").encode(
        x=alt.X("hour:Q", title="תחילת חלון של שעתיים", scale=alt.Scale(domain=[0, 23], nice=False),
                axis=alt.Axis(values=list(range(0, 24, 2)), labelExpr="format(datum.value, '02d') + ':00'")),
        y=alt.Y("rate:Q", title="שיעור המשך",
                scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("hour:Q", title="שעת התחלה"), alt.Tooltip("events:Q", title="אירועים"),
                 alt.Tooltip("rate:Q", title="שיעור", format=".1%")],
    ).properties(height=245)


def wait_chart(rows: pd.DataFrame) -> alt.Chart:
    counts = rows[WAIT].dropna().floordiv(120).value_counts().reindex(range(12), fill_value=0)
    data = pd.DataFrame({
        "window": [f"{2*i}–{2*i+2}" for i in range(12)],
        "events": counts.to_list(),
    })
    return alt.Chart(data).mark_bar(color="#AC603A", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("window:N", title="שעות עד הקפה הבא", sort=data.window.to_list(),
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("events:Q", title="מספר אירועים"),
        tooltip=[alt.Tooltip("window:N", title="טווח שעות"), alt.Tooltip("events:Q", title="אירועים")],
    ).properties(height=245)


def main() -> None:
    full = data()
    st.sidebar.header("סינון הנתונים")
    age = st.sidebar.slider("גיל (80 = 80 ומעלה)", 18, 80, (18, 80))
    selected_days = st.sidebar.multiselect("יום בשבוע", WEEK, default=WEEK, format_func=DAYS.get)
    st.sidebar.caption("התוויות מחושבות לפני הסינון, על כל יום דיווח.")
    rows = full.loc[full.age_years.between(*age) & full.weekday.isin(selected_days)]

    st.title("NextCup")
    st.markdown("### מתי מגיע הקפה הבא?")
    if rows.empty:
        st.info("אין אירועי קפה בסינון הזה. אפשר להרחיב את טווח הגילים או לבחור ימים נוספים.")
        return

    st.markdown(f'<div class="insight-box">{summary_line(rows)}</div>', unsafe_allow_html=True)
    median = rows[WAIT].median()
    median_label = "—" if pd.isna(median) else f"{median / 60:.1f}"
    first, second, third, fourth = st.columns(4)
    first.metric("שיעור אירועים שאחריהם דווח עוד קפה", f"{rows[LABEL].mean():.1%}", border=True)
    second.metric("אירועי קפה", f"{len(rows):,}", border=True)
    third.metric("משתתפים", f"{rows.person_id.nunique():,}", border=True)
    fourth.metric("חציון שעות לקפה הבא", median_label, border=True)

    left, right = st.columns(2)
    with left:
        st.subheader("האם דווח עוד קפה באותו יום?")
        st.altair_chart(rate_chart(rows), width="stretch")
        st.caption("כל נקודה מסכמת חלון של שעתיים שבו יש לפחות 20 אירועי קפה.")
    with right:
        st.subheader("כמה זמן עבר עד הקפה הבא?")
        st.altair_chart(wait_chart(rows), width="stretch")
        st.caption("הגרף והחציון כוללים את האירועים שאחריהם דווח עוד קפה.")

    st.subheader("קווי הבסיס ל־M3")
    scores = baseline_scores(full)
    one, two = st.columns(2)
    one.metric("דיוק סיווג · מנבאים תמיד שלא ידווח עוד קפה", f"{scores['accuracy']:.1%}", border=True)
    two.metric("טעות מוחלטת ממוצעת · מנבאים חציון קבוע", f"{scores['mae_minutes'] / 60:.2f} שעות", border=True)
    st.caption(
        f"קטגוריית הרוב וחציון של {scores['median_minutes'] / 60:g} שעות נקבעו באימון."
        f" הדיוק נמדד על {scores['test_events']:,} אירועי מבחן"
        f" והטעות על {scores['positive_test_events']:,} אירועים חיוביים מתוכם."
        " המדדים קבועים כדי לאפשר השוואה למודל באותו מבחן."
    )

    st.subheader("דוגמה מהנתונים")
    sample_cols = ["weekday", "clock_time", "age_years", "coffee_mg", "prior_coffee_events", LABEL, WAIT]
    example = rows.sort_values(["minute_of_day", "person_id", "recall_day"]).iloc[::max(1, len(rows) // 7)]
    example = example[sample_cols].head(7).copy()
    example["weekday"] = example.weekday.map(DAYS)
    example[LABEL] = example[LABEL].map({1: "כן", 0: "לא"})
    # Show an empty cell (not "None") when there was no later coffee that day.
    example[WAIT] = example[WAIT].map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
    st.dataframe(
        example,
        hide_index=True,
        width="stretch",
        column_config=dict(zip(sample_cols, [
            "יום", "שעת קפה", "גיל", "מ״ג קפאין מקפה", "אירועי קפה מוקדמים יותר",
            "עוד קפה היום", "דקות עד הקפה הבא",
        ])),
    )
    st.caption(
        "מקור הנתונים הוא NHANES 2017–2018. המדדים מתארים את אירועי הדיווח במדגם, ללא משקלי סקר."
        " NextCup הוא כלי לימודי ולא ייעוץ רפואי."
    )


main()

