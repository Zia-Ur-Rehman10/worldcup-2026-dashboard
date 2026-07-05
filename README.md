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

## Handling a live, in-progress tournament

Now that the real World Cup 2026 is underway, `matches.csv` contains a
mix of played matches and future fixtures (with `NaN` scores) — and
knockout matches can be decided by penalty shootouts. The pipeline
accounts for both:

- **Upcoming fixtures** are tracked separately (`is_played` flag) and
  excluded from every team stat, standings table, and style-cluster
  calculation — they're never silently treated as 0-0 draws. The
  Overview KPIs show an "Upcoming" count, and the Tournament Stages tab
  shows played-vs-scheduled counts per stage.
- **Penalty shootouts** are read from `home_penalty_score` /
  `away_penalty_score` — a knockout match level after normal time is
  correctly credited to the shootout winner, not recorded as a draw.
- **Stage order** is derived from `stage_id`, not match dates — dates
  aren't reliable for phase ordering (rescheduling, weather delays,
  etc. can put an earlier-dated match in a later phase).

Two extra files sometimes appear in the Kaggle dataset —
`matches_detailed.csv` and `player_stats.csv`. They're loaded but not
currently used by any chart (the pipeline only reads the specific
files it needs). If you want player ratings/shot data from
`player_stats.csv` wired into the Players tab once Kaggle populates it
with real values, that's a small addition — just ask.

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

## Deployed

Live at: **https://fifa-worldcup-2026-dashboard.streamlit.app/**

It's deployed via Streamlit Community Cloud, connected to
`github.com/Zia-Ur-Rehman10/worldcup-2026-dashboard`. Any push to
`main` auto-redeploys within about a minute.

## Fully automated updates (no manual git push)

Your data source is the Kaggle dataset
[mominullptr/fifa-world-cup-2026-dataset](https://www.kaggle.com/datasets/mominullptr/fifa-world-cup-2026-dataset) —
so instead of Colab → Drive → manual download → paste into `raw_data/`,
`scripts/fetch_data.py` pulls it directly via the official Kaggle API,
and GitHub Actions runs that on a schedule and pushes updates automatically.

### How it works

1. **`scripts/fetch_data.py`** — authenticates with Kaggle, downloads
   and unzips the dataset, then copies every CSV into `raw_data/` —
   but only overwrites a file if its contents actually changed, so you
   don't get noise commits every time the schedule fires.

2. **`scripts/validate_data.py`** — runs the *entire* dashboard
   pipeline against the freshly-fetched data before anything gets
   committed. If the upstream dataset changes a column name, drops a
   file, or introduces bad data — anything that would break the live
   dashboard — this catches it and prints exactly what broke and
   where.

3. **`.github/workflows/update_data.yml`** — runs on a schedule (daily
   at 06:00 UTC by default): fetch → validate → only if validation
   passes, commit and push. If validation fails, the workflow **rolls
   back to the last known-good `raw_data/`** and fails loudly (you'll
   get a GitHub notification) instead of silently pushing data that
   would break your live dashboard.

4. **Streamlit Community Cloud** already watches your repo's `main`
   branch and redeploys on every push — so once the Action pushes new
   (validated) data, your live dashboard updates within about a minute.

This closes the loop on the exact failure mode you hit manually:
before, a schema change from Kaggle would go straight to production
and break the dashboard. Now it gets caught and blocked automatically.

### One-time setup: Kaggle API token

You're already using this exact method in your Colab notebook — you
just need to give GitHub Actions the same token.

1. Go to **[kaggle.com/settings](https://www.kaggle.com/settings)** →
   scroll to the **API** section → click **Create New Token** (or, if
   you're rotating the one that was previously exposed, click
   **Expire Token** first, then create a new one). Kaggle will show
   you a token string starting with `KGAT_...` — copy it.
2. On GitHub, go to your repo → **Settings → Secrets and variables →
   Actions → New repository secret**. Add exactly **one** secret:
   - Name: `KAGGLE_API_TOKEN`
   - Value: your `KGAT_...` token
3. To test locally first (optional but recommended before relying on
   the schedule):
   ```bash
   export KAGGLE_API_TOKEN=your_token_here
   pip install -r scripts/requirements.txt
   python scripts/fetch_data.py
   python scripts/validate_data.py
   ```
   You should see each CSV reported as "updated" or "no changes,"
   followed by `VALIDATION PASSED`.

**Security note:** never commit a Kaggle token directly into a
notebook or any file in the repo — that's exactly what happened
previously and the exposed token should be expired via "Expire Token"
on the Kaggle settings page above. GitHub Secrets (used here) are
encrypted and never appear in logs or the repo itself.

### Running it

- **On schedule**: happens automatically once the secret is set —
  nothing more to do.
- **On demand**: GitHub repo → **Actions** tab → "Auto-update World Cup
  data" → **Run workflow**, to trigger it immediately instead of
  waiting for the schedule.
- **Change the frequency**: edit the `cron:` line in
  `.github/workflows/update_data.yml` (cron times are always UTC).
- **If it fails**: check the **Actions** tab → click the failed run →
  read the `validate_data.py` output. It tells you exactly which
  column or file changed. Your live dashboard keeps running on the
  last good data the whole time — a failed update never takes the
  site down.

### Before the secret is added

The workflow will run on schedule but fail at the fetch step with an
authentication error — this is expected and won't affect your live
dashboard or existing data, it just means new data won't be pulled in
until the secret above is configured.
