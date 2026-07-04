# FIFA World Cup 2026 — Live Dashboard

A local Streamlit dashboard that turns your Colab ETL notebook into a
reusable pipeline. Add new data to `raw_data/` any time — the dashboard
detects the change and recomputes everything automatically.

## 1. Setup (one time)

```bash
cd worldcup_dashboard
python3 -m venv venv
source venv/bin/activate          # Mac/Linux
pip install -r requirements.txt
```

## 2. Add your data

Copy your dataset CSVs into `raw_data/`:

```
raw_data/
  matches.csv
  teams.csv
  venues.csv
  tournament_stages.csv
  referees.csv
  match_team_stats.csv
  match_events.csv
  squads_and_players.csv
  match_lineups.csv
```

(Same files you were downloading from Kaggle in the Colab notebook —
just save them locally instead of to Google Drive.)

## 3. Run it

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## How the auto-update works

Every time the app reruns (page refresh, filter change, or the
"Refresh data now" button), `pipeline.py` computes a fingerprint of
`raw_data/` — every file's name, size, and last-modified time, hashed
together. That fingerprint is passed into every `@st.cache_data`
function as the `version` argument.

- **Fingerprint unchanged** → Streamlit reuses cached results instantly.
- **Fingerprint changed** (you added a match, updated a CSV, added a
  new file) → cache invalidates and the whole pipeline recomputes on
  the next rerun.

So "automatic" here means: *the moment your data changes, the very
next interaction with the app shows fresh numbers* — no re-running a
separate script, no manual cache-clearing.

### Want it to update even with no one touching the app?

Two options, in order of effort:

1. **Auto-refresh toggle (already built in):** the sidebar has an
   "Auto-refresh every 30s" toggle using `streamlit-autorefresh`. It
   silently reruns the app on a timer, so if you're editing CSVs in
   another window, the dashboard catches up within 30 seconds — no
   click required.
2. **A cron job / watcher that writes new CSVs into `raw_data/`:** if
   your data source is an API or a scheduled export rather than manual
   file edits, have that job write directly into `raw_data/` on its
   own schedule. The pipeline will pick it up next time the app runs
   or auto-refreshes — you don't need to touch `pipeline.py`.

## What's in the advanced version

- **Professional visual theme** — navy/gold stadium-scoreboard styling: a
  hero header banner, card-style KPIs, styled tabs, and insight cards
  instead of plain bullet lists.
- **Tournament Stages** — pick Group Stage, Round of 16, etc. from a
  dropdown. Shows a progression funnel across all stages, match results
  per stage, which teams advanced vs. got eliminated, and full group
  standings tables (Played/Won/Drawn/Lost/GF/GA/GD/Points) with a
  points-by-group chart. This directly answers "who made it to Ro32,"
  "what were the results," "who qualified for Ro16."
- **Deeper Head-to-Head** — pick two teams and get: their direct match
  history (if they've played), a multi-metric radar, a stacked win/
  draw/loss bar, and a grouped bar comparing goals/xG/possession/shots/
  points side by side — not just win rate.
- **Team Intelligence** — Attack/Defense/Control/Finishing ratings
  (0–100 normalized) + Overall Rating, radar chart, ranking bar chart,
  and an Attack-vs-Finishing bubble chart.
- **Style Map** — PCA + K-Means clustering into playing styles, cluster
  size breakdown, Attack-vs-Defense quadrant chart, Goals-vs-xG
  "who's clinical" scatter with trendline.
- **Match Excitement** — standardized index combining goals, xG, shots,
  fouls, and closeness; shows both most AND least exciting matches,
  plus a distribution histogram.
- **Goal Timing** — polar goal-distribution chart, cumulative timeline,
  stoppage-time callout, team×time heatmap, first-half vs second-half
  grouped bars.
- **Overview** — goals over time, goals/match by stage, goals
  distribution histogram, outcome-split donut, venue averages,
  confederation comparisons.
- **Advanced, specific insights** — biggest win margin, highest-scoring
  match, % of matches decided by one goal, draw rate, home vs away
  advantage, biggest upset (by Elo gap), clean sheet leader, most
  clinical finisher, most-carded team, best confederation, and whether
  high possession actually correlates with winning — computed fresh
  from whatever columns your data actually has.
- **Isolated tab errors** — if one tab hits a data issue (e.g. a
  missing column), it shows a friendly message and technical details
  in that tab only; the rest of the dashboard keeps working instead of
  going blank.
- **Players** (only appears if `squads_and_players.csv` and
  `match_lineups.csv` are present) — value vs. contribution, discipline
  by position, top scorer leaderboard.

All of this recomputes automatically the same way the basic KPIs do —
change your raw CSVs, refresh, done.

## Extending the pipeline

`src/pipeline.py` mirrors your notebook's stages:

| Notebook section | Function |
|---|---|
| Master Match Table + Feature Engineering | `build_master_matches` |
| Phase 3: Team Analytics Table | `build_team_summary` |
| Team Match Mart | `build_team_match_mart` |
| Event mart (goals/cards/subs) | `build_event_mart` |
| Tournament KPIs | `build_kpis` |
| Tournament Insights | `build_insights` |

If you add a new metric in the notebook, add it to the matching
function here — no need to touch `app.py` unless you want to display
it. Each function is defensive about missing columns (e.g. `home_xg`),
same pattern your notebook already used for the district shapefile
detection in your other project.

## Deploying (optional)

To share this like your R/Shiny "Departure for Development" dashboard:
push this folder to GitHub and deploy on **Streamlit Community Cloud**
(streamlit.io/cloud) — free, connects directly to a GitHub repo. If
your raw data changes often, commit updated CSVs to the repo and the
cloud app picks them up on the next deploy/rerun the same way it does
locally.
