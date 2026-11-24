# NextCup · Context for Claude

NextCup is an Ariel University Vibe Coding project. M2 describes when adults reported the next caffeinated coffee in NHANES 2017–2018. M3 will predict the same outcome after a user logs a coffee.

- The production entry point is `streamlit run app.py`. It reads `data/next_coffee_events.csv` through `analysis.py`.
- The CSV contains 5,623 coffee episodes, 2,704 adult participants and 4,443 participant-days. An episode is a reported minute with a USDA 7302 coffee item containing at least 10 mg caffeine. Simultaneous eligible beverage items are aggregated. The canonical builder and provenance are in `data/build_next_coffee.py` and `data/data_quality.json`.
- `has_next_coffee` is the primary classification target. `minutes_to_next_coffee` is a conditional regression target for positive rows. Preserve positive gaps and the boundary at midnight of the same recall day.
- Never feed labels, IDs, recall index, split or end-of-day summaries into a model. Read `analysis.FEATURES` for inputs. Keep both recalls of each person in the same group. Tune and preprocess inside training folds by participant. Use the fixed test once for final reporting.
- Majority-class and median-time baselines were fitted on train. Held-out accuracy is 78.07% and positive-only MAE is 184.48 minutes. Only 21.0% of pooled events have later coffee. Report minority recall/precision alongside accuracy.
- The main rate is 22.3% for coffee at 06:00–09:59 vs 14.8% at 14:00–17:59. These rates are descriptive, unweighted and sensitive to time remaining before midnight.
- Survey sleep variables refer to usual habits. The next-coffee dataset has no same-night sleep outcome or post-dose alertness labels. Guard the distinction between a behavior prediction, study-level sleep evidence, and actual individual outcomes.
- Why coffee only (team decision, Nov 2026): coffee is about half of caffeinated drinks but ~70% of the caffeine. The app will log coffee only, so the M3 model must use only features computable from a coffee log (time, weekday, coffee mg, prior coffee); the all-caffeine history columns are for analysis. The older all-caffeine EDA is in git history (commit c8f2e51).
- `README.md` (Hebrew) is the project spec. `artifact/nextcup.html` is the M1 web app and still describes all caffeinated drinks.
- Keep the interface in Hebrew and repository code in English. Keep raw downloads out of Git. Not medical advice: say so wherever we show predictions about the body. Verify clean requirements, both charts, filter behavior and the README screenshot after dashboard edits.
