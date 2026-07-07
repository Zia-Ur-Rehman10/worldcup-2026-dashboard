"""
FIFA World Cup 2026 — Data Pipeline
====================================
This is the local, non-Colab version of your notebook's ETL logic.
Every function is wrapped in @st.cache_data and keyed on `version` —
a fingerprint of the raw_data folder. When you add or edit a CSV in
raw_data/, the fingerprint changes, Streamlit invalidates the cache,
and every stage below recomputes automatically on the next rerun.

You never need to "run the pipeline" separately — the Streamlit app
calls these functions directly.
"""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ============================================================
# 0. DATA VERSIONING — the thing that makes auto-update work
# ============================================================

def get_data_version(raw_path: Path) -> str:
    """
    Fingerprints the raw_data folder: filename + size + last-modified
    time of every CSV/XLSX. If ANY of that changes (new file added,
    existing file overwritten with new rows, etc.), this hash changes.

    Pass this string into every cached loader below as the `version`
    argument. Streamlit's cache key includes it, so a changed fingerprint
    = automatic recompute. An unchanged fingerprint = instant cache hit.
    """
    raw_path = Path(raw_path)
    fingerprint = []
    for f in sorted(raw_path.glob("*")):
        if f.suffix.lower() in (".csv", ".xlsx"):
            stat = f.stat()
            fingerprint.append(f"{f.name}:{stat.st_size}:{stat.st_mtime}")
    joined = "|".join(fingerprint)
    return hashlib.md5(joined.encode()).hexdigest()


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_raw_data(raw_path: str, version: str) -> dict:
    raw_path = Path(raw_path)
    dataframes = {}

    for file in raw_path.glob("*.csv"):
        dataframes[file.stem] = pd.read_csv(file)

    for file in raw_path.glob("*.xlsx"):
        dataframes[file.stem] = pd.read_excel(file)

    required = [
        "matches", "teams", "venues", "tournament_stages", "referees",
        "match_team_stats", "match_events"
    ]
    missing = [r for r in required if r not in dataframes]
    if missing:
        raise FileNotFoundError(
            f"Missing expected file(s) in raw_data/: {', '.join(missing)}.csv"
        )

    return dataframes


# ============================================================
# 2. MASTER MATCH TABLE + FEATURE ENGINEERING
# ============================================================

@st.cache_data(show_spinner=False)
def build_master_matches(_dataframes: dict, version: str) -> pd.DataFrame:
    d = _dataframes
    matches = d["matches"]
    teams = d["teams"]
    venues = d["venues"]
    stages = d["tournament_stages"]
    referees = d["referees"]

    # `venues.csv` and `referees.csv` can both carry a `country` column
    # (venue country vs. referee's home country) — rename up front so a
    # merge never silently collides them into ambiguous country_x/country_y.
    venues = venues.rename(columns={"country": "venue_country"}) if "country" in venues.columns else venues
    referees = referees.rename(columns={
        c: f"referee_{c}" for c in ["country", "avg_cards_per_game", "name"] if c in referees.columns
    })

    home_lookup = teams.rename(columns={
        "team_id": "home_team_id", "team_name": "home_team_name",
        "fifa_code": "home_fifa_code", "group_letter": "home_group",
        "confederation": "home_confederation",
        "fifa_ranking_pre_tournament": "home_fifa_rank",
        "elo_rating": "home_elo", "manager_name": "home_manager",
    })
    away_lookup = teams.rename(columns={
        "team_id": "away_team_id", "team_name": "away_team_name",
        "fifa_code": "away_fifa_code", "group_letter": "away_group",
        "confederation": "away_confederation",
        "fifa_ranking_pre_tournament": "away_fifa_rank",
        "elo_rating": "away_elo", "manager_name": "away_manager",
    })

    mm = (
        matches
        .merge(home_lookup, on="home_team_id", how="left")
        .merge(away_lookup, on="away_team_id", how="left")
        .merge(venues, on="venue_id", how="left")
        .merge(stages, on="stage_id", how="left")
        .merge(referees, on="referee_id", how="left")
    )

    # --- Two distinct reasons a team name can be missing here, handled
    # differently: (1) the fixture genuinely has no team assigned yet —
    # a future knockout slot like "Winner of R16 Match 93" where
    # home_team_id/away_team_id are null in the source data because that
    # earlier match hasn't been played. That's normal — label it "TBD".
    # (2) home_team_id/away_team_id IS set but didn't match anything in
    # teams.csv (e.g. a stale teams.csv missing a newer team_id) — a real
    # data problem, labeled "Unknown Team" so it's visibly different from
    # a normal TBD bracket slot, and doesn't crash the string
    # concatenation below (`NaN + str` would raise a TypeError). ---
    for side in ["home", "away"]:
        id_col, name_col = f"{side}_team_id", f"{side}_team_name"
        if name_col in mm.columns and mm[name_col].isna().any():
            missing = mm[name_col].isna()
            mm.loc[missing, name_col] = np.where(mm.loc[missing, id_col].isna(), "TBD", "Unknown Team")

    # --- Played vs. upcoming: a real, live tournament's matches.csv mixes
    # completed results with future fixtures (NaN scores). Every stat below
    # must be computed only from matches that have actually been played. ---
    mm["is_played"] = mm["home_score"].notna() & mm["away_score"].notna()
    if "status" in mm.columns:
        # Belt-and-suspenders: also respect an explicit status column if
        # the dataset provides one, in case scores are backfilled early.
        mm["is_played"] = mm["is_played"] & (mm["status"] != "Postponed")

    # --- Penalty shootout awareness: a knockout match level after 90/120
    # minutes is NOT a draw if it was decided on penalties. ---
    has_penalties = "home_penalty_score" in mm.columns and "away_penalty_score" in mm.columns
    if has_penalties:
        mm["decided_on_penalties"] = mm["home_penalty_score"].notna() & mm["away_penalty_score"].notna()
    else:
        mm["decided_on_penalties"] = False

    # --- Basic match features (computed for played matches; NaN/False for
    # upcoming fixtures rather than silently coded as 0-0 draws) ---
    mm["total_goals"] = mm["home_score"] + mm["away_score"]
    mm["goal_difference"] = (mm["home_score"] - mm["away_score"]).abs()

    def resolve_outcome(row):
        """Returns (home_win, away_win, draw, winner_name) accounting for
        penalty shootouts. Unplayed fixtures return all-False / None."""
        if not row["is_played"]:
            return False, False, False, None
        if row["home_score"] > row["away_score"]:
            return True, False, False, row["home_team_name"]
        if row["home_score"] < row["away_score"]:
            return False, True, False, row["away_team_name"]
        # Scores level — check for a penalty shootout before calling it a draw
        if row["decided_on_penalties"]:
            if row["home_penalty_score"] > row["away_penalty_score"]:
                return True, False, False, row["home_team_name"]
            elif row["away_penalty_score"] > row["home_penalty_score"]:
                return False, True, False, row["away_team_name"]
        return False, False, True, "Draw"

    outcomes = mm.apply(resolve_outcome, axis=1, result_type="expand")
    outcomes.columns = ["home_win", "away_win", "draw", "winner"]
    mm[["home_win", "away_win", "draw", "winner"]] = outcomes

    # --- xG features (guarded — some datasets omit xG until later stages) ---
    if "home_xg" in mm.columns and "away_xg" in mm.columns:
        mm["total_xg"] = mm["home_xg"] + mm["away_xg"]
        mm["xg_difference"] = (mm["home_xg"] - mm["away_xg"]).abs()
        mm["goal_minus_xg"] = mm["total_goals"] - mm["total_xg"]

    # --- Team strength ---
    if "home_fifa_rank" in mm.columns:
        mm["ranking_difference"] = mm["home_fifa_rank"] - mm["away_fifa_rank"]
    if "home_elo" in mm.columns:
        mm["elo_difference"] = mm["home_elo"] - mm["away_elo"]

        def favorite(row):
            if row.home_elo > row.away_elo:
                return row.home_team_name
            elif row.away_elo > row.home_elo:
                return row.away_team_name
            return "Equal"

        mm["favorite"] = mm.apply(favorite, axis=1)

    def goal_category(g):
        if pd.isna(g):
            return "Not Played"
        if g == 0:
            return "0 Goals"
        elif g <= 2:
            return "Low"
        elif g <= 4:
            return "Medium"
        return "High"

    mm["goal_category"] = mm["total_goals"].apply(goal_category)

    if "is_knockout" in mm.columns:
        mm["knockout_match"] = mm["is_knockout"].map({True: "Yes", False: "No"})

    mm["home_clean_sheet"] = mm["is_played"] & (mm["away_score"] == 0)
    mm["away_clean_sheet"] = mm["is_played"] & (mm["home_score"] == 0)

    def format_scoreline(row):
        if not row["is_played"]:
            return f"{row['home_team_name']} vs {row['away_team_name']} (Upcoming)"
        line = f"{row['home_team_name']} {int(row['home_score'])}\u2013{int(row['away_score'])} {row['away_team_name']}"
        if row["decided_on_penalties"]:
            line += f" (pens {int(row['home_penalty_score'])}\u2013{int(row['away_penalty_score'])})"
        return line

    mm["scoreline"] = mm.apply(format_scoreline, axis=1)

    return mm


# ============================================================
# 3. TEAM SUMMARY
# ============================================================

@st.cache_data(show_spinner=False)
def build_team_summary(master_matches: pd.DataFrame, teams: pd.DataFrame, version: str) -> pd.DataFrame:
    mm = master_matches
    has_xg = "home_xg" in mm.columns

    home_agg = {
        "home_matches": ("match_id", "count"),
        "home_goals": ("home_score", "sum"),
        "home_goals_conceded": ("away_score", "sum"),
        "home_wins": ("home_win", "sum"),
        "home_draws": ("draw", "sum"),
    }
    away_agg = {
        "away_matches": ("match_id", "count"),
        "away_goals": ("away_score", "sum"),
        "away_goals_conceded": ("home_score", "sum"),
        "away_wins": ("away_win", "sum"),
        "away_draws": ("draw", "sum"),
    }
    if has_xg:
        home_agg["home_xg"] = ("home_xg", "sum")
        away_agg["away_xg"] = ("away_xg", "sum")

    home_stats = mm.groupby("home_team_name").agg(**home_agg).reset_index()
    home_stats.rename(columns={"home_team_name": "team"}, inplace=True)

    away_stats = mm.groupby("away_team_name").agg(**away_agg).reset_index()
    away_stats.rename(columns={"away_team_name": "team"}, inplace=True)

    ts = home_stats.merge(away_stats, on="team", how="outer").fillna(0)

    ts["matches"] = ts["home_matches"] + ts["away_matches"]
    ts["goals"] = ts["home_goals"] + ts["away_goals"]
    ts["goals_conceded"] = ts["home_goals_conceded"] + ts["away_goals_conceded"]
    ts["wins"] = ts["home_wins"] + ts["away_wins"]
    ts["draws"] = ts["home_draws"] + ts["away_draws"]
    ts["losses"] = ts["matches"] - ts["wins"] - ts["draws"]
    ts["goal_difference"] = ts["goals"] - ts["goals_conceded"]

    ts["goals_per_match"] = ts["goals"] / ts["matches"]
    ts["goals_conceded_per_match"] = ts["goals_conceded"] / ts["matches"]
    ts["win_rate"] = ts["wins"] / ts["matches"] * 100

    if has_xg:
        ts["total_xg"] = ts["home_xg"] + ts["away_xg"]
        ts["goal_efficiency"] = (ts["goals"] / ts["total_xg"]).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)
    else:
        ts["goal_efficiency"] = 0

    ts = ts.merge(teams, left_on="team", right_on="team_name", how="left")

    ts["performance_score"] = (
        ts["wins"] * 3 + ts["draws"]
        + ts["goal_difference"] * 0.30
        + ts["goal_efficiency"] * 0.40
    )

    ts = ts.sort_values("performance_score", ascending=False).reset_index(drop=True)
    return ts


# ============================================================
# 4. TEAM MATCH MART (long format: one row per team per match)
# ============================================================

@st.cache_data(show_spinner=False)
def build_team_match_mart(master_matches: pd.DataFrame, match_stats: pd.DataFrame,
                           teams: pd.DataFrame, version: str) -> pd.DataFrame:
    mm = master_matches

    base_cols = ["match_id", "date", "stage_name", "stadium_name", "city"]
    has_xg = "home_xg" in mm.columns
    has_rank = "home_fifa_rank" in mm.columns
    has_elo = "home_elo" in mm.columns

    def make_view(prefix_team, prefix_opp, venue_type):
        cols = base_cols + [f"{prefix_team}_team_name", f"{prefix_opp}_team_name",
                             f"{prefix_team}_score", f"{prefix_opp}_score"]
        rename = {f"{prefix_team}_team_name": "team", f"{prefix_opp}_team_name": "opponent",
                  f"{prefix_team}_score": "goals_for", f"{prefix_opp}_score": "goals_against"}
        if has_xg:
            cols += [f"{prefix_team}_xg", f"{prefix_opp}_xg"]
            rename.update({f"{prefix_team}_xg": "xg_for", f"{prefix_opp}_xg": "xg_against"})
        if has_rank:
            cols += [f"{prefix_team}_fifa_rank", f"{prefix_opp}_fifa_rank"]
            rename.update({f"{prefix_team}_fifa_rank": "team_rank",
                           f"{prefix_opp}_fifa_rank": "opponent_rank"})
        if has_elo:
            cols += [f"{prefix_team}_elo", f"{prefix_opp}_elo"]
            rename.update({f"{prefix_team}_elo": "team_elo", f"{prefix_opp}_elo": "opponent_elo"})

        view = mm[cols].copy().rename(columns=rename)
        view["venue_type"] = venue_type
        return view

    home_view = make_view("home", "away", "Home")
    away_view = make_view("away", "home", "Away")

    tmm = pd.concat([home_view, away_view], axis=0).sort_values(["team", "date"]).reset_index(drop=True)

    # Merge in per-team match stats (possession, shots, corners, fouls...)
    team_stats = match_stats.merge(teams[["team_id", "team_name"]], on="team_id", how="left")
    tmm = (
        tmm.merge(team_stats.drop(columns=["team_id"]),
                  left_on=["match_id", "team"], right_on=["match_id", "team_name"], how="left")
        .drop(columns=["team_name"])
    )

    if "shots_on_target" in tmm.columns and "total_shots" in tmm.columns:
        tmm["shot_accuracy"] = tmm["shots_on_target"] / tmm["total_shots"]
        tmm["shot_conversion"] = tmm["goals_for"] / tmm["total_shots"]
        clean_cols = ["shot_accuracy", "shot_conversion"]
        if "xg_for" in tmm.columns:
            tmm["xg_efficiency"] = tmm["goals_for"] / tmm["xg_for"]
            clean_cols.append("xg_efficiency")
        tmm[clean_cols] = tmm[clean_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    tmm["goal_difference"] = tmm["goals_for"] - tmm["goals_against"]
    tmm["result"] = np.select(
        [tmm["goals_for"] > tmm["goals_against"], tmm["goals_for"] < tmm["goals_against"]],
        ["Win", "Loss"], default="Draw",
    )

    # --- Opponent-side stats (for dominance metrics: possession diff, shot diff, etc.) ---
    opp_stat_cols = [c for c in ["possession_pct", "total_shots", "shots_on_target",
                                   "corners", "fouls", "offsides", "saves", "xg_for", "goals_for"]
                     if c in tmm.columns]
    if opp_stat_cols and "opponent" in tmm.columns:
        opponent_stats = tmm[["match_id", "team"] + opp_stat_cols].copy()
        opponent_stats.rename(columns={"team": "opponent", **{
            "possession_pct": "opp_possession", "total_shots": "opp_shots",
            "shots_on_target": "opp_shots_on_target", "corners": "opp_corners",
            "fouls": "opp_fouls", "offsides": "opp_offsides", "saves": "opp_saves",
            "xg_for": "opp_xg", "goals_for": "opp_goals",
        }}, inplace=True)
        tmm = tmm.merge(opponent_stats, on=["match_id", "opponent"], how="left")

    return tmm


# ============================================================
# 5. EVENT MART (goals, cards, subs — for timelines)
# ============================================================

def _parse_minute(value) -> float:
    """Handles both plain minutes ('45') and stoppage-time notation
    ('90+6' -> 96.0). Real match-event data frequently uses '+' notation
    for added time, which otherwise silently turns the whole column into
    text and breaks every numeric comparison downstream."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if "+" in s:
        parts = s.split("+")
        try:
            return float(parts[0]) + float(parts[1])
        except (ValueError, IndexError):
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


@st.cache_data(show_spinner=False)
def build_event_mart(match_events: pd.DataFrame, teams: pd.DataFrame, version: str) -> pd.DataFrame:
    events = match_events.copy()
    if "team_id" in events.columns:
        events = events.merge(teams[["team_id", "team_name"]], on="team_id", how="left")
    if "minute" in events.columns:
        events["minute"] = events["minute"].apply(_parse_minute)
    return events


# ============================================================
# 6. TOURNAMENT KPIs + INSIGHTS
# ============================================================

@st.cache_data(show_spinner=False)
def build_kpis(master_matches: pd.DataFrame, teams: pd.DataFrame, venues: pd.DataFrame,
                version: str) -> dict:
    mm = master_matches
    completed = mm[mm["is_played"]] if "is_played" in mm.columns else mm
    upcoming = mm[~mm["is_played"]] if "is_played" in mm.columns else mm.iloc[0:0]

    kpi = {}
    kpi["Matches Played"] = len(completed)
    kpi["Upcoming Matches"] = len(upcoming)
    kpi["Goals"] = int(completed["home_score"].sum() + completed["away_score"].sum())
    kpi["Goals per Match"] = round(kpi["Goals"] / max(kpi["Matches Played"], 1), 2)
    kpi["Teams"] = teams["team_name"].nunique()
    kpi["Venues"] = venues["stadium_name"].nunique() if "stadium_name" in venues.columns else venues.shape[0]

    if "home_xg" in completed.columns:
        kpi["Average xG"] = round(
            (completed["home_xg"].sum() + completed["away_xg"].sum()) / max(kpi["Matches Played"], 1), 2
        )
    else:
        kpi["Average xG"] = None

    return kpi


@st.cache_data(show_spinner=False)
def build_insights(team_summary: pd.DataFrame, event_mart: pd.DataFrame, version: str) -> list:
    insights = []

    if not team_summary.empty:
        top_team = team_summary.sort_values("goals", ascending=False).iloc[0]
        insights.append(
            f"{top_team['team']} is the tournament's highest-scoring team with {int(top_team['goals'])} goals."
        )
        best_poss = team_summary.sort_values("win_rate", ascending=False).iloc[0]
        insights.append(
            f"{best_poss['team']} has the best win rate so far ({best_poss['win_rate']:.1f}%)."
        )

    if "event_type" in event_mart.columns and "minute" in event_mart.columns:
        goal_events = event_mart[event_mart["event_type"] == "Goal"]
        late_goals = goal_events[goal_events["minute"] >= 76].shape[0]
        insights.append(f"{late_goals} goals have been scored after the 75th minute.")

    return insights


# ============================================================
# 7. TEAM INTELLIGENCE (normalized 0-100 Attack/Defense/Control/Finishing)
# ============================================================

@st.cache_data(show_spinner=False)
def build_team_intelligence(team_match_mart: pd.DataFrame, version: str) -> pd.DataFrame:
    tmm = team_match_mart.copy()

    if "opp_possession" not in tmm.columns:
        return pd.DataFrame()

    tmm["possession_dominance"] = tmm["possession_pct"] - tmm.get("opp_possession", 0)
    tmm["shot_dominance"] = tmm["total_shots"] - tmm.get("opp_shots", 0)
    tmm["xg_dominance"] = tmm.get("xg_for", 0) - tmm.get("opp_xg", 0)
    tmm["clinical_finishing"] = tmm.get("goals_for", 0) - tmm.get("xg_for", 0)

    agg = {
        "Goals": ("goals_for", "mean"),
        "Goals_Conceded": ("goals_against", "mean"),
        "Shots": ("total_shots", "mean"),
        "Shots_On_Target": ("shots_on_target", "mean"),
        "Possession": ("possession_pct", "mean"),
        "Saves": ("saves", "mean"),
        "Shot_Accuracy": ("shot_accuracy", "mean"),
        "Shot_Conversion": ("shot_conversion", "mean"),
        "Clinical_Finishing": ("clinical_finishing", "mean"),
        "Possession_Dominance": ("possession_dominance", "mean"),
        "Shot_Dominance": ("shot_dominance", "mean"),
    }
    if "xg_for" in tmm.columns:
        agg["xG"] = ("xg_for", "mean")
        agg["Opponent_xG"] = ("opp_xg", "mean")

    ti = tmm.groupby("team").agg(**agg)

    def normalize(series):
        scaler = MinMaxScaler(feature_range=(0, 100))
        return scaler.fit_transform(series.fillna(series.mean()).values.reshape(-1, 1)).flatten()

    for col in ti.columns:
        ti[col + "_N"] = normalize(ti[col])

    ti["Attack_Index"] = (
        ti.get("Goals_N", 0) + ti.get("xG_N", 0) + ti["Shots_N"] + ti["Shots_On_Target_N"]
    ) / (4 if "xG_N" in ti.columns else 3)

    ti["Defense_Index"] = (
        (100 - ti["Goals_Conceded_N"]) + (100 - ti.get("Opponent_xG_N", 100)) + ti["Saves_N"]
    ) / 3

    ti["Control_Index"] = (
        ti["Possession_N"] + ti["Possession_Dominance_N"] + ti["Shot_Dominance_N"]
    ) / 3

    ti["Finishing_Index"] = (
        ti["Shot_Accuracy_N"] + ti["Shot_Conversion_N"] + ti["Clinical_Finishing_N"]
    ) / 3

    ti["Overall_Rating"] = (
        ti["Attack_Index"] * 0.30 + ti["Defense_Index"] * 0.30
        + ti["Control_Index"] * 0.20 + ti["Finishing_Index"] * 0.20
    )

    return ti.sort_values("Overall_Rating", ascending=False).reset_index()


# ============================================================
# 8. TEAM SUMMARY V2 + PLAYING STYLE CLUSTERS (PCA + K-Means)
# ============================================================

@st.cache_data(show_spinner=False)
def build_team_summary_v2(team_match_mart: pd.DataFrame, teams: pd.DataFrame, version: str) -> pd.DataFrame:
    tmm = team_match_mart
    has_xg = "xg_for" in tmm.columns

    agg = {
        "Matches": ("match_id", "count"),
        "Wins": ("result", lambda x: (x == "Win").sum()),
        "Draws": ("result", lambda x: (x == "Draw").sum()),
        "Losses": ("result", lambda x: (x == "Loss").sum()),
        "Goals": ("goals_for", "sum"),
        "Goals_Conceded": ("goals_against", "sum"),
        "Possession": ("possession_pct", "mean"),
    }
    if has_xg:
        agg["xG"] = ("xg_for", "sum")
    if "total_shots" in tmm.columns:
        agg["Shots"] = ("total_shots", "sum")
        agg["Shots_On_Target"] = ("shots_on_target", "sum")
    if "corners" in tmm.columns:
        agg["Corners"] = ("corners", "sum")
    if "fouls" in tmm.columns:
        agg["Fouls"] = ("fouls", "sum")

    tsv2 = tmm.groupby("team").agg(**agg).reset_index()

    tsv2["Points"] = tsv2["Wins"] * 3 + tsv2["Draws"]
    tsv2["Goal_Difference"] = tsv2["Goals"] - tsv2["Goals_Conceded"]
    tsv2["Goals_per_Match"] = tsv2["Goals"] / tsv2["Matches"]
    tsv2["Goals_Conceded_per_Match"] = tsv2["Goals_Conceded"] / tsv2["Matches"]
    tsv2["Win_Rate"] = tsv2["Wins"] / tsv2["Matches"] * 100

    if has_xg:
        tsv2["xG_per_Match"] = tsv2["xG"] / tsv2["Matches"]
        tsv2["Goals_minus_xG"] = tsv2["Goals"] - tsv2["xG"]

    if "Shots" in tsv2.columns:
        tsv2["Shot_Accuracy"] = tsv2["Shots_On_Target"] / tsv2["Shots"]
        tsv2["Shot_Conversion"] = tsv2["Goals"] / tsv2["Shots"]

    tsv2 = tsv2.merge(teams, left_on="team", right_on="team_name", how="left")

    # Standardized composite indices
    scale_cols = [c for c in ["Goals_per_Match", "xG_per_Match", "Shot_Accuracy",
                                "Goals_Conceded_per_Match", "Possession", "Shots",
                                "Corners", "Win_Rate"] if c in tsv2.columns]
    if len(scale_cols) >= 4:
        scaler = StandardScaler()
        z = scaler.fit_transform(tsv2[scale_cols].fillna(0))
        z_df = pd.DataFrame(z, columns=[f"{c}_Z" for c in scale_cols], index=tsv2.index)
        tsv2 = pd.concat([tsv2, z_df], axis=1)

        attack_parts = [c for c in ["Goals_per_Match_Z", "xG_per_Match_Z", "Shot_Accuracy_Z"] if c in tsv2.columns]
        tsv2["Attack_Index"] = tsv2[attack_parts].mean(axis=1) if attack_parts else 0
        tsv2["Defense_Index"] = -tsv2.get("Goals_Conceded_per_Match_Z", 0)
        control_parts = [c for c in ["Possession_Z", "Shots_Z", "Corners_Z"] if c in tsv2.columns]
        tsv2["Control_Index"] = tsv2[control_parts].mean(axis=1) if control_parts else 0
        tsv2["Overall_Index"] = (
            tsv2["Attack_Index"] * 0.40 + tsv2["Defense_Index"] * 0.35 + tsv2["Control_Index"] * 0.25
        )

        # PCA + K-Means playing style clusters (needs enough teams)
        cluster_features = [c for c in ["Goals_per_Match", "Goals_Conceded_per_Match", "Possession",
                                          "Shots", "Shot_Accuracy", "xG_per_Match", "Fouls",
                                          "Corners", "Win_Rate"] if c in tsv2.columns]
        n_teams = tsv2.shape[0]
        if len(cluster_features) >= 3 and n_teams >= 5:
            X = tsv2[cluster_features].fillna(0)
            X_scaled = StandardScaler().fit_transform(X)

            pca = PCA(n_components=2, random_state=42)
            pcs = pca.fit_transform(X_scaled)
            tsv2["PC1"], tsv2["PC2"] = pcs[:, 0], pcs[:, 1]

            n_clusters = min(5, n_teams)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            tsv2["Cluster"] = kmeans.fit_predict(X_scaled)

            # Label clusters by their attack/defense/control profile rather than
            # a fixed mapping, since cluster order isn't stable run to run.
            cluster_means = tsv2.groupby("Cluster")[["Attack_Index", "Defense_Index"]].mean()
            style_labels = {}
            for c in cluster_means.index:
                a, d = cluster_means.loc[c, "Attack_Index"], cluster_means.loc[c, "Defense_Index"]
                if a > 0.3 and d > 0.3:
                    style_labels[c] = "Elite Teams"
                elif a > 0.3:
                    style_labels[c] = "Attacking Teams"
                elif d > 0.3:
                    style_labels[c] = "Defensive Teams"
                elif a < -0.3 and d < -0.3:
                    style_labels[c] = "Struggling Teams"
                else:
                    style_labels[c] = "Balanced Competitors"
            tsv2["Playing_Style"] = tsv2["Cluster"].map(style_labels)

    return tsv2.sort_values("Points", ascending=False).reset_index(drop=True)


# ============================================================
# 9. MATCH EXCITEMENT INDEX
# ============================================================

@st.cache_data(show_spinner=False)
def build_match_summary(master_matches: pd.DataFrame, match_stats: pd.DataFrame, version: str) -> pd.DataFrame:
    ms = master_matches.copy()

    agg = {"Total_Shots": ("total_shots", "sum")}
    if "fouls" in match_stats.columns:
        agg["Total_Fouls"] = ("fouls", "sum")
    if "corners" in match_stats.columns:
        agg["Total_Corners"] = ("corners", "sum")

    stats_agg = match_stats.groupby("match_id").agg(**agg).reset_index()
    ms = ms.merge(stats_agg, on="match_id", how="left")

    ms["Total_Goals"] = ms["home_score"] + ms["away_score"]
    ms["Goal_Difference"] = (ms["home_score"] - ms["away_score"]).abs()
    if "home_xg" in ms.columns:
        ms["Total_xG"] = ms["home_xg"] + ms["away_xg"]

    metric_cols = [c for c in ["Total_Goals", "Total_xG", "Total_Shots",
                                 "Goal_Difference", "Total_Fouls"] if c in ms.columns]
    if len(metric_cols) < 2:
        return ms

    scaler = StandardScaler()
    z = scaler.fit_transform(ms[metric_cols].fillna(0))
    for i, c in enumerate(metric_cols):
        ms[f"{c}_Z"] = z[:, i]

    ms["Excitement_Index"] = (
        0.35 * ms.get("Total_Goals_Z", 0)
        + 0.25 * ms.get("Total_xG_Z", 0)
        + 0.20 * ms.get("Total_Shots_Z", 0)
        + 0.10 * ms.get("Total_Fouls_Z", 0)
        - 0.10 * ms.get("Goal_Difference_Z", 0)
    )

    return ms.sort_values("Excitement_Index", ascending=False)


# ============================================================
# 10. GOAL TIMING ANALYSIS
# ============================================================

@st.cache_data(show_spinner=False)
def build_goal_timing(event_mart: pd.DataFrame, version: str) -> dict:
    if "event_type" not in event_mart.columns or "minute" not in event_mart.columns:
        return {}

    goal_events = event_mart[event_mart["event_type"] == "Goal"].copy()
    if goal_events.empty:
        return {}

    bins = [0, 15, 30, 45, 60, 75, 90, 130]
    labels = ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"]
    goal_events["Time_Window"] = pd.cut(goal_events["minute"], bins=bins, labels=labels, include_lowest=True)

    timing_dist = goal_events.groupby("Time_Window", observed=True).size().reset_index(name="Goals")

    result = {"timing_distribution": timing_dist, "goal_events": goal_events}

    if "team_name" in goal_events.columns:
        goal_events["Half"] = np.where(goal_events["minute"] <= 45, "First Half", "Second Half")
        half_pivot = (
            goal_events.groupby(["team_name", "Half"]).size().reset_index(name="Goals")
            .pivot(index="team_name", columns="Half", values="Goals").fillna(0).reset_index()
        )
        if "First Half" in half_pivot.columns and "Second Half" in half_pivot.columns:
            half_pivot["Improvement"] = half_pivot["Second Half"] - half_pivot["First Half"]
        result["half_comparison"] = half_pivot.sort_values(
            "Improvement" if "Improvement" in half_pivot.columns else half_pivot.columns[1],
            ascending=False,
        )

        heatmap = (
            goal_events.groupby(["team_name", "Time_Window"], observed=True).size()
            .reset_index(name="Goals")
            .pivot(index="team_name", columns="Time_Window", values="Goals").fillna(0)
        )
        result["team_timing_heatmap"] = heatmap

    return result


# ============================================================
# 11. PLAYER SUMMARY (requires squads_and_players + match_lineups)
# ============================================================

@st.cache_data(show_spinner=False)
def build_player_summary(dataframes: dict, event_mart: pd.DataFrame, master_matches: pd.DataFrame, version: str) -> pd.DataFrame:
    if "squads_and_players" not in dataframes:
        return pd.DataFrame()

    players = dataframes["squads_and_players"].copy()

    # --- Bio enrichment: age from date_of_birth, if present ---
    if "date_of_birth" in players.columns:
        dob = pd.to_datetime(players["date_of_birth"], errors="coerce")
        # Age as of the tournament (using today's date is fine either way —
        # this is just for relative comparisons between players).
        players["age"] = ((pd.Timestamp.now() - dob).dt.days / 365.25).round(1)

    # --- Preferred path: player_stats.csv is the authoritative source for
    # real per-tournament totals (goals, assists, cards, minutes...). Fall
    # back to deriving from match_lineups/match_events only if it's absent
    # (e.g. an older dataset snapshot). ---
    if "player_stats" in dataframes:
        stats = dataframes["player_stats"].copy()
        rename_map = {
            "matches_played": "Matches", "matches_started": "Starts",
            "minutes_played": "Minutes", "goals": "Goals", "assists": "Assists",
            "yellow_cards": "Yellow_Cards", "red_cards": "Red_Cards",
            "penalty_goals": "Penalty_Goals", "own_goals": "Own_Goals",
            "clean_sheets": "Clean_Sheets", "saves": "Saves",
            "goals_conceded": "Goals_Conceded", "average_rating": "Average_Rating",
        }
        stats = stats.rename(columns={k: v for k, v in rename_map.items() if k in stats.columns})
        # Avoid duplicate player_name/position columns from both sources
        stats_cols = [c for c in stats.columns if c in ("player_id",) or c not in players.columns]
        ps = players.merge(stats[stats_cols], on="player_id", how="left")
    else:
        lineups = dataframes.get("match_lineups")
        if lineups is None:
            return pd.DataFrame()

        player_minutes = lineups.groupby("player_id").agg(
            Matches=("match_id", "count"),
            Starts=("is_starting_xi", "sum"),
            Minutes=("minutes_played", "sum"),
        ).reset_index()

        def count_event(event_type, out_name):
            if "event_type" not in event_mart.columns or "player_id" not in event_mart.columns:
                return pd.DataFrame(columns=["player_id", out_name])
            return (
                event_mart[event_mart["event_type"] == event_type]
                .groupby("player_id").size().reset_index(name=out_name)
            )

        ps = (
            players
            .merge(player_minutes, on="player_id", how="left")
            .merge(count_event("Goal", "Goals"), on="player_id", how="left")
            .merge(count_event("Assist", "Assists"), on="player_id", how="left")
            .merge(count_event("Yellow Card", "Yellow_Cards"), on="player_id", how="left")
            .merge(count_event("Red Card", "Red_Cards"), on="player_id", how="left")
        )

    numeric_cols = ["Matches", "Starts", "Minutes", "Goals", "Assists", "Yellow_Cards", "Red_Cards",
                     "Penalty_Goals", "Own_Goals", "Clean_Sheets", "Saves", "Goals_Conceded"]
    for c in numeric_cols:
        if c not in ps.columns:
            ps[c] = 0
    ps[numeric_cols] = ps[numeric_cols].fillna(0)

    ps["Goals_per_90"] = np.where(ps["Minutes"] > 0, ps["Goals"] * 90 / ps["Minutes"], 0)
    ps["Assists_per_90"] = np.where(ps["Minutes"] > 0, ps["Assists"] * 90 / ps["Minutes"], 0)
    ps["Goal_Contributions"] = ps["Goals"] + ps["Assists"]
    ps["Goal_Contributions_per_90"] = np.where(
        ps["Minutes"] > 0, ps["Goal_Contributions"] * 90 / ps["Minutes"], 0
    )

    if "market_value_eur" in ps.columns:
        ps["Market_Value_M"] = ps["market_value_eur"] / 1_000_000
        ps["Value_Efficiency"] = np.where(
            ps["Market_Value_M"] > 0, ps["Goal_Contributions"] / ps["Market_Value_M"], 0
        )

    # --- Player of the Match awards, if the dataset tracks them ---
    if "player_of_the_match_id" in master_matches.columns:
        potm_counts = (
            master_matches["player_of_the_match_id"].dropna()
            .value_counts().rename_axis("player_id").reset_index(name="POTM_Awards")
        )
        ps = ps.merge(potm_counts, on="player_id", how="left")
        ps["POTM_Awards"] = ps["POTM_Awards"].fillna(0).astype(int)
    else:
        ps["POTM_Awards"] = 0

    return ps


# ============================================================
# 13. ADVANCED / UNIQUE TOURNAMENT INSIGHTS
# ============================================================

@st.cache_data(show_spinner=False)
def build_advanced_insights(master_matches: pd.DataFrame, team_summary_v2: pd.DataFrame,
                              team_match_mart: pd.DataFrame, event_mart: pd.DataFrame,
                              version: str) -> list:
    mm = master_matches
    insights = []

    if mm.empty:
        return insights

    # Biggest win margin
    if "goal_difference" in mm.columns and "winner" in mm.columns:
        blowout = mm.loc[mm["goal_difference"].idxmax()]
        insights.append(
            f"🔨 Biggest margin of victory: **{blowout['winner']}** won "
            f"{blowout.get('scoreline', '')} — a {int(blowout['goal_difference'])}-goal gap."
        )

    # Highest-scoring match
    if "total_goals" in mm.columns:
        thriller = mm.loc[mm["total_goals"].idxmax()]
        insights.append(
            f"🎯 Highest-scoring match: {thriller.get('scoreline', '')} "
            f"({int(thriller['total_goals'])} goals combined)."
        )

    # Share of matches decided by a single goal
    if "goal_difference" in mm.columns:
        decided_mm = mm[mm["draw"] == False] if "draw" in mm.columns else mm
        if len(decided_mm) > 0:
            one_goal_pct = (decided_mm["goal_difference"] == 1).mean() * 100
            insights.append(f"⚖️ {one_goal_pct:.0f}% of decisive matches were settled by a single goal.")

    # Draw rate
    if "draw" in mm.columns:
        draw_pct = mm["draw"].mean() * 100
        insights.append(f"🤝 {draw_pct:.0f}% of matches so far have ended in a draw.")

    # Home advantage check
    if "home_win" in mm.columns and "away_win" in mm.columns:
        home_rate = mm["home_win"].mean() * 100
        away_rate = mm["away_win"].mean() * 100
        if home_rate > away_rate:
            insights.append(f"🏟️ Home teams are winning {home_rate:.0f}% of matches vs. {away_rate:.0f}% for away teams.")
        else:
            insights.append(f"✈️ Away teams are outperforming hosts: {away_rate:.0f}% away wins vs. {home_rate:.0f}% home wins.")

    # Biggest upset by ranking/elo gap where the lower-rated team won
    if "elo_difference" in mm.columns and "winner" in mm.columns:
        mm_valid = mm.dropna(subset=["elo_difference"])
        upsets = mm_valid[
            ((mm_valid["elo_difference"] > 0) & (mm_valid["away_win"] == True)) |
            ((mm_valid["elo_difference"] < 0) & (mm_valid["home_win"] == True))
        ]
        if not upsets.empty:
            upsets = upsets.copy()
            upsets["upset_gap"] = upsets["elo_difference"].abs()
            biggest_upset = upsets.loc[upsets["upset_gap"].idxmax()]
            insights.append(
                f"😱 Biggest upset: **{biggest_upset['winner']}** beat the Elo-favorite "
                f"in {biggest_upset.get('scoreline', '')}."
            )

    # Clean sheet leader
    if "home_clean_sheet" in mm.columns:
        cs_home = mm[mm["home_clean_sheet"]]["home_team_name"].value_counts()
        cs_away = mm[mm["away_clean_sheet"]]["away_team_name"].value_counts()
        cs_total = cs_home.add(cs_away, fill_value=0).sort_values(ascending=False)
        if not cs_total.empty:
            insights.append(f"🧤 {cs_total.index[0]} has the most clean sheets ({int(cs_total.iloc[0])}).")

    # Most clinical finisher (Goals - xG) from team_summary_v2
    if "Goals_minus_xG" in team_summary_v2.columns and not team_summary_v2.empty:
        clinical = team_summary_v2.loc[team_summary_v2["Goals_minus_xG"].idxmax()]
        insights.append(
            f"🎯 {clinical['team']} is over-performing their xG by "
            f"{clinical['Goals_minus_xG']:.1f} goals — most clinical in the tournament."
        )

    # Discipline: most cards
    if "event_type" in event_mart.columns and "team_name" in event_mart.columns:
        cards = event_mart[event_mart["event_type"].isin(["Yellow Card", "Red Card"])]
        if not cards.empty:
            worst = cards["team_name"].value_counts().idxmax()
            worst_count = cards["team_name"].value_counts().max()
            insights.append(f"🟨 {worst} is the tournament's most-carded team ({int(worst_count)} cards).")

    # Confederation strength
    if "confederation" in team_summary_v2.columns and "Win_Rate" in team_summary_v2.columns:
        confed_perf = team_summary_v2.groupby("confederation")["Win_Rate"].mean().sort_values(ascending=False)
        if not confed_perf.empty:
            insights.append(
                f"🌍 {confed_perf.index[0]} has the best average win rate by confederation "
                f"({confed_perf.iloc[0]:.0f}%)."
            )

    # Possession doesn't always win
    if "Possession" in team_match_mart.columns and "result" in team_match_mart.columns:
        high_poss_win_rate = team_match_mart[team_match_mart["possession_pct"] > 55]["result"].eq("Win").mean() * 100 \
            if "possession_pct" in team_match_mart.columns else None
        if high_poss_win_rate is not None and not np.isnan(high_poss_win_rate):
            insights.append(
                f"⚽ Teams with over 55% possession won {high_poss_win_rate:.0f}% of those matches."
            )

    return insights


# ============================================================
# 14. GROUP STAGE STANDINGS
# ============================================================

@st.cache_data(show_spinner=False)
def build_group_standings(master_matches: pd.DataFrame, version: str) -> pd.DataFrame:
    mm = master_matches
    if "stage_name" not in mm.columns or "home_group" not in mm.columns:
        return pd.DataFrame()

    group_matches = mm[mm["stage_name"].str.contains("Group", case=False, na=False)].copy()
    if group_matches.empty:
        return pd.DataFrame()

    group_matches["group"] = group_matches["home_group"].fillna(group_matches.get("away_group"))

    rows = []
    for _, m in group_matches.iterrows():
        rows.append({
            "group": m["group"], "team": m["home_team_name"],
            "played": 1, "won": int(m["home_score"] > m["away_score"]),
            "drawn": int(m["home_score"] == m["away_score"]),
            "lost": int(m["home_score"] < m["away_score"]),
            "gf": m["home_score"], "ga": m["away_score"],
        })
        rows.append({
            "group": m["group"], "team": m["away_team_name"],
            "played": 1, "won": int(m["away_score"] > m["home_score"]),
            "drawn": int(m["home_score"] == m["away_score"]),
            "lost": int(m["away_score"] < m["home_score"]),
            "gf": m["away_score"], "ga": m["home_score"],
        })

    long_df = pd.DataFrame(rows)
    standings = long_df.groupby(["group", "team"]).sum(numeric_only=True).reset_index()
    standings["gd"] = standings["gf"] - standings["ga"]
    standings["points"] = standings["won"] * 3 + standings["drawn"]
    standings = standings.sort_values(
        ["group", "points", "gd", "gf"], ascending=[True, False, False, False]
    ).reset_index(drop=True)
    standings["rank_in_group"] = standings.groupby("group").cumcount() + 1

    return standings


# ============================================================
# 15. STAGE-BY-STAGE TOURNAMENT PROGRESSION
# ============================================================

@st.cache_data(show_spinner=False)
def build_stage_progression(master_matches: pd.DataFrame, version: str) -> dict:
    mm = master_matches
    if "stage_name" not in mm.columns:
        return {}

    # Order stages by stage_id — that's what actually encodes tournament
    # phase order. Match dates are NOT reliable for this: rescheduling,
    # weather delays, or knockout legs played out of strict date sequence
    # can all make "earliest match date per stage" lie about real order.
    if "stage_id" in mm.columns:
        stage_order_map = mm.groupby("stage_name")["stage_id"].min().sort_values()
        stage_order = stage_order_map.index.tolist()
    elif "date" in mm.columns:
        stage_dates = mm.groupby("stage_name")["date"].min().sort_values()
        stage_order = stage_dates.index.tolist()
    else:
        return {}

    teams_by_stage = {}
    for stage in stage_order:
        stage_mm = mm[mm["stage_name"] == stage]
        teams_by_stage[stage] = (
            set(stage_mm["home_team_name"]).union(stage_mm["away_team_name"])
            - {"TBD", "Unknown Team"}
        )

    progression_rows = []
    for i, stage in enumerate(stage_order):
        participants = teams_by_stage[stage]
        next_stage = stage_order[i + 1] if i + 1 < len(stage_order) else None
        advancing = teams_by_stage[next_stage] if next_stage else set()
        eliminated = participants - advancing if next_stage else set()

        stage_mm = mm[mm["stage_name"] == stage]
        completed = stage_mm["is_played"].all() if "is_played" in stage_mm.columns else True
        matches_played = int(stage_mm["is_played"].sum()) if "is_played" in stage_mm.columns else len(stage_mm)

        progression_rows.append({
            "stage": stage,
            "order": i,
            "teams": len(participants),
            "matches": len(stage_mm),
            "matches_played": matches_played,
            "goals": int(stage_mm["total_goals"].sum()) if "total_goals" in stage_mm.columns else None,
            "advancing_teams": len(advancing) if next_stage else None,
            "eliminated_teams": sorted(eliminated) if next_stage else [],
            "participant_teams": sorted(participants),
            "advancing_team_list": sorted(advancing) if next_stage else sorted(participants),
            "is_completed": bool(completed),
        })

    return {
        "stage_order": stage_order,
        "progression": progression_rows,
        "matches_by_stage": {s: mm[mm["stage_name"] == s] for s in stage_order},
    }


# ============================================================
# 16. VENUE GEOGRAPHY & ELEVATION ANALYSIS
# ============================================================

@st.cache_data(show_spinner=False)
def build_venue_analysis(master_matches: pd.DataFrame, dataframes: dict, version: str) -> pd.DataFrame:
    venues = dataframes.get("venues")
    if venues is None or "latitude" not in venues.columns:
        return pd.DataFrame()

    mm = master_matches[master_matches.get("is_played", True) == True] \
        if "is_played" in master_matches.columns else master_matches

    if "stadium_name" not in mm.columns:
        return pd.DataFrame()

    agg = {"Matches": ("match_id", "count"), "Goals": ("total_goals", "sum")}
    if "total_xg" in mm.columns:
        agg["xG"] = ("total_xg", "sum")

    venue_stats = mm.groupby("stadium_name").agg(**agg).reset_index()
    venue_stats["Goals_per_Match"] = venue_stats["Goals"] / venue_stats["Matches"]

    venue_cols = [c for c in ["stadium_name", "venue_country", "city", "capacity",
                                "latitude", "longitude", "elevation_meters"] if c in venues.columns
                  or c == "stadium_name"]
    venues_renamed = venues.rename(columns={"country": "venue_country"}) if "country" in venues.columns else venues
    venue_stats = venue_stats.merge(
        venues_renamed[[c for c in venue_cols if c in venues_renamed.columns]],
        on="stadium_name", how="left"
    )

    return venue_stats.sort_values("Goals_per_Match", ascending=False).reset_index(drop=True)


# ============================================================
# 17. REFEREE DISCIPLINE ANALYSIS
# ============================================================

@st.cache_data(show_spinner=False)
def build_referee_analysis(master_matches: pd.DataFrame, event_mart: pd.DataFrame, version: str) -> pd.DataFrame:
    mm = master_matches[master_matches.get("is_played", True) == True] \
        if "is_played" in master_matches.columns else master_matches

    ref_col = "referee_name" if "referee_name" in mm.columns else ("name" if "name" in mm.columns else None)
    if ref_col is None or "referee_id" not in mm.columns:
        return pd.DataFrame()

    ref_stats = mm.groupby(ref_col).agg(
        Matches=("match_id", "count"),
        Goals_Officiated=("total_goals", "sum"),
    ).reset_index().rename(columns={ref_col: "referee_name"})

    if "referee_avg_cards_per_game" in mm.columns:
        expected = mm.groupby(ref_col)["referee_avg_cards_per_game"].mean().reset_index()
        expected.columns = ["referee_name", "Expected_Cards_per_Game"]
        ref_stats = ref_stats.merge(expected, on="referee_name", how="left")

    if "event_type" in event_mart.columns and "match_id" in event_mart.columns:
        cards = event_mart[event_mart["event_type"].isin(["Yellow Card", "Red Card"])]
        match_to_ref = mm[["match_id", ref_col]].rename(columns={ref_col: "referee_name"})
        cards_with_ref = cards.merge(match_to_ref, on="match_id", how="left")
        actual_cards = cards_with_ref.groupby("referee_name").size().reset_index(name="Actual_Cards")
        ref_stats = ref_stats.merge(actual_cards, on="referee_name", how="left")
        ref_stats["Actual_Cards"] = ref_stats["Actual_Cards"].fillna(0)
        ref_stats["Actual_Cards_per_Game"] = ref_stats["Actual_Cards"] / ref_stats["Matches"]

    return ref_stats.sort_values("Matches", ascending=False).reset_index(drop=True)


# ============================================================
# 18. PRE-MATCH FACTORS (squad value, rest days, host advantage)
# ============================================================

@st.cache_data(show_spinner=False)
def build_match_factors(dataframes: dict, version: str) -> pd.DataFrame:
    mpf = dataframes.get("match_prediction_features")
    if mpf is None:
        return pd.DataFrame()

    mpf = mpf.copy()
    if "home_squad_total_value_eur" in mpf.columns and "away_squad_total_value_eur" in mpf.columns:
        mpf["value_difference_m"] = (mpf["home_squad_total_value_eur"] - mpf["away_squad_total_value_eur"]) / 1_000_000
        mpf["higher_value_team_won"] = np.select(
            [mpf["value_difference_m"] > 0, mpf["value_difference_m"] < 0],
            [mpf.get("match_result") == "H", mpf.get("match_result") == "A"],
            default=np.nan,
        )
    if "home_rest_days" in mpf.columns and "away_rest_days" in mpf.columns:
        mpf["rest_days_difference"] = mpf["home_rest_days"] - mpf["away_rest_days"]
    if "home_is_host" in mpf.columns:
        mpf["host_team_playing"] = (mpf["home_is_host"] == 1) | (mpf.get("away_is_host", 0) == 1)

    return mpf


# ============================================================
# 19. ONE-CALL PIPELINE RUNNER
# ============================================================

def run_pipeline(raw_path: str):
    """
    Call this once from app.py. Returns a dict of every processed
    table + kpis + insights. Handles caching/versioning internally —
    the caller doesn't need to think about it.
    """
    version = get_data_version(raw_path)
    dataframes = load_raw_data(raw_path, version)

    master_matches = build_master_matches(dataframes, version)

    # Team-level stats, standings, and clustering must only ever see matches
    # that have actually been played — not future fixtures with NaN scores.
    # `master_matches` (full schedule, played + upcoming) is still returned
    # separately below for the Tournament Stages / fixtures view.
    played_matches = master_matches[master_matches["is_played"]].copy() \
        if "is_played" in master_matches.columns else master_matches

    team_summary = build_team_summary(played_matches, dataframes["teams"], version)
    team_match_mart = build_team_match_mart(
        played_matches, dataframes["match_team_stats"], dataframes["teams"], version
    )
    event_mart = build_event_mart(dataframes["match_events"], dataframes["teams"], version)
    kpis = build_kpis(master_matches, dataframes["teams"], dataframes["venues"], version)
    insights = build_insights(team_summary, event_mart, version)

    # Advanced analytics layer
    team_intelligence = build_team_intelligence(team_match_mart, version)
    team_summary_v2 = build_team_summary_v2(team_match_mart, dataframes["teams"], version)
    match_summary = build_match_summary(played_matches, dataframes["match_team_stats"], version)
    goal_timing = build_goal_timing(event_mart, version)
    player_summary = build_player_summary(dataframes, event_mart, master_matches, version)
    advanced_insights = build_advanced_insights(played_matches, team_summary_v2, team_match_mart, event_mart, version)
    group_standings = build_group_standings(played_matches, version)
    stage_progression = build_stage_progression(master_matches, version)
    venue_analysis = build_venue_analysis(master_matches, dataframes, version)
    referee_analysis = build_referee_analysis(master_matches, event_mart, version)
    match_factors = build_match_factors(dataframes, version)

    return {
        "version": version,
        "dataframes": dataframes,
        "master_matches": master_matches,
        "team_summary": team_summary,
        "team_match_mart": team_match_mart,
        "event_mart": event_mart,
        "kpis": kpis,
        "insights": insights,
        "team_intelligence": team_intelligence,
        "team_summary_v2": team_summary_v2,
        "match_summary": match_summary,
        "goal_timing": goal_timing,
        "player_summary": player_summary,
        "advanced_insights": advanced_insights,
        "group_standings": group_standings,
        "stage_progression": stage_progression,
        "venue_analysis": venue_analysis,
        "referee_analysis": referee_analysis,
        "match_factors": match_factors,
    }
