"""
FIFA World Cup 2026 — Live Analytics Dashboard
================================================
Run with:  streamlit run app.py
"""

from pathlib import Path
import sys
import traceback
import inspect

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent / "src"))
from pipeline import run_pipeline  # noqa: E402

st.set_page_config(page_title="FIFA World Cup 2026 Dashboard", layout="wide", page_icon="⚽")

RAW_PATH = Path(__file__).parent / "raw_data"

# ------------------------------------------------------------------
# Version-adaptive st.plotly_chart wrapper.
# Newer Streamlit uses width="stretch"; older Streamlit only knows
# use_container_width=True and treats any unrecognized keyword as a
# legacy Plotly config override (hence the "keyword arguments have been
# deprecated... use `config` instead" warning on every single chart).
# This checks the installed version once and always passes the
# argument that version actually supports.
# ------------------------------------------------------------------
_original_plotly_chart = st.plotly_chart
_plotly_chart_params = set(inspect.signature(_original_plotly_chart).parameters)


def _plotly_chart_compat(fig, *args, **kwargs):
    if "width" in _plotly_chart_params:
        kwargs.setdefault("width", "stretch")
    else:
        kwargs.pop("width", None)
        kwargs.setdefault("use_container_width", True)
    return _original_plotly_chart(fig, *args, **kwargs)


st.plotly_chart = _plotly_chart_compat

# ============================================================
# THEME — navy / gold, stadium-scoreboard inspired
# ============================================================
NAVY = "#0B2545"
NAVY_MID = "#13315C"
BLUE = "#1F6FB2"
GOLD = "#D4AF37"
GREEN = "#1E8449"
RED = "#B3261E"
BG = "#F4F7FB"

PLOTLY_TEMPLATE = "plotly_white"
COLOR_SEQUENCE = [BLUE, GOLD, GREEN, NAVY_MID, "#8E44AD", "#E67E22", "#16A085", RED]
px.defaults.color_discrete_sequence = COLOR_SEQUENCE
px.defaults.template = PLOTLY_TEMPLATE

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.block-container {{
    padding-top: 1.5rem;
    max-width: 1350px;
}}

/* Hero header banner */
.hero-banner {{
    background: linear-gradient(120deg, {NAVY} 0%, {NAVY_MID} 55%, {BLUE} 100%);
    border-radius: 14px;
    padding: 2rem 2.25rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 20px rgba(11, 37, 69, 0.25);
    border-bottom: 4px solid {GOLD};
}}
.hero-title {{
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 2.6rem;
    letter-spacing: 0.5px;
    color: #FFFFFF;
    margin: 0;
    line-height: 1.1;
}}
.hero-subtitle {{
    color: #C9D6E8;
    font-size: 1rem;
    margin-top: 0.35rem;
}}

/* KPI cards */
.kpi-card {{
    background: #FFFFFF;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    border-left: 5px solid {BLUE};
    box-shadow: 0 2px 8px rgba(11, 37, 69, 0.08);
    height: 100%;
}}
.kpi-label {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6B7A90;
    font-weight: 600;
    margin-bottom: 0.2rem;
}}
.kpi-value {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: {NAVY};
    line-height: 1;
}}

/* Section eyebrow headers */
.section-eyebrow {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: {NAVY};
    border-bottom: 3px solid {GOLD};
    display: inline-block;
    padding-bottom: 0.15rem;
    margin: 0.3rem 0 1rem 0;
}}

/* Insight cards */
.insight-card {{
    background: #FFFFFF;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.55rem;
    border-left: 4px solid {GOLD};
    box-shadow: 0 1px 4px rgba(11, 37, 69, 0.06);
    font-size: 0.95rem;
    color: #1A1A2E;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: #E9EEF6;
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1rem;
    font-weight: 600;
    color: {NAVY_MID};
}}
.stTabs [aria-selected="true"] {{
    background-color: {NAVY} !important;
    color: #FFFFFF !important;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {NAVY};
}}
section[data-testid="stSidebar"] * {{
    color: #E9EEF6 !important;
}}
section[data-testid="stSidebar"] .stButton button {{
    background-color: {GOLD};
    color: {NAVY};
    font-weight: 700;
    border: none;
}}

/* Badges */
.badge-live {{
    background-color: {GOLD}; color: {NAVY}; padding: 2px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
}}
.badge-done {{
    background-color: {GREEN}; color: white; padding: 2px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
}}
</style>
""", unsafe_allow_html=True)


def kpi_card(col, label, value):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title):
    st.markdown(f'<div class="section-eyebrow">{title}</div>', unsafe_allow_html=True)


def safe_section(title, render_fn):
    """Runs a tab's content in isolation — an error here won't blank out
    the rest of the dashboard, it just reports itself and moves on."""
    try:
        render_fn()
    except Exception as e:
        st.error(f"This section hit a snag: {e}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## ⚽ World Cup 2026")

try:
    from streamlit_autorefresh import st_autorefresh
    enable_auto = st.sidebar.toggle("Auto-refresh every 30s", value=False)
    if enable_auto:
        st_autorefresh(interval=30_000, key="data_refresh")
except ImportError:
    pass

if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

# ============================================================
# LOAD DATA
# ============================================================
if not RAW_PATH.exists() or not any(RAW_PATH.glob("*.csv")):
    st.error("No CSV files found in `raw_data/`. Add your dataset files there and refresh.")
    st.stop()

try:
    data = run_pipeline(str(RAW_PATH))
except Exception as e:
    st.error(f"Pipeline error: {e}")
    with st.expander("Technical details"):
        st.code(traceback.format_exc())
    st.stop()

mm = data["master_matches"]
team_summary = data["team_summary"]
team_summary_v2 = data["team_summary_v2"]
team_match_mart = data["team_match_mart"]
team_intelligence = data["team_intelligence"]
event_mart = data["event_mart"]
match_summary = data["match_summary"]
goal_timing = data["goal_timing"]
player_summary = data["player_summary"]
kpis = data["kpis"]
advanced_insights = data["advanced_insights"]
group_standings = data["group_standings"]
stage_progression = data["stage_progression"]

has_style = "Playing_Style" in team_summary_v2.columns

# ============================================================
# HERO HEADER
# ============================================================
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">FIFA World Cup 2026 — Analytics Dashboard</div>
    <div class="hero-subtitle">Live tournament intelligence · teams, matches, form &amp; style</div>
</div>
""", unsafe_allow_html=True)

kc = st.columns(5)
kpi_card(kc[0], "Matches Played", kpis["Matches Played"])
kpi_card(kc[1], "Total Goals", kpis["Goals"])
kpi_card(kc[2], "Goals / Match", kpis["Goals per Match"])
kpi_card(kc[3], "Teams", kpis["Teams"])
kpi_card(kc[4], "Avg xG", kpis["Average xG"] if kpis["Average xG"] is not None else "—")

st.write("")

if advanced_insights:
    with st.expander("📌 Tournament Insights", expanded=True):
        cols = st.columns(2)
        for i, insight in enumerate(advanced_insights):
            with cols[i % 2]:
                st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

st.write("")

# ============================================================
# TABS
# ============================================================
tab_names = ["Overview", "Tournament Stages", "Team Intelligence", "Style Map",
             "Match Excitement", "Goal Timing", "Head-to-Head", "Matches"]
if not player_summary.empty:
    tab_names.append("Players")

tabs = st.tabs(tab_names)
tab_map = dict(zip(tab_names, tabs))


# ================================= OVERVIEW =================================
def render_overview():
    section_header("Tournament Pulse")
    c1, c2 = st.columns(2)
    with c1:
        daily = mm.groupby("date").agg(Goals=("total_goals", "sum")).reset_index()
        st.plotly_chart(px.line(daily, x="date", y="Goals", markers=True,
                                 title="Goals Over Time"), width='stretch')
    with c2:
        if "stage_name" in mm.columns:
            stage_goals = mm.groupby("stage_name").agg(
                Goals=("total_goals", "sum"), Matches=("match_id", "count")
            ).reset_index()
            stage_goals["Goals per Match"] = stage_goals["Goals"] / stage_goals["Matches"]
            st.plotly_chart(px.bar(stage_goals, x="stage_name", y="Goals per Match", color="Goals",
                                    text_auto=".2f", title="Goals per Match by Stage"),
                             width='stretch')

    c3, c4 = st.columns(2)
    with c3:
        if "total_goals" in mm.columns:
            st.plotly_chart(px.histogram(mm, x="total_goals", nbins=int(mm["total_goals"].max()) + 1,
                                          title="Distribution of Goals per Match",
                                          color_discrete_sequence=[BLUE]),
                             width='stretch')
    with c4:
        if "home_win" in mm.columns and "away_win" in mm.columns and "draw" in mm.columns:
            outcome_counts = pd.DataFrame({
                "Outcome": ["Home Win", "Draw", "Away Win"],
                "Count": [mm["home_win"].sum(), mm["draw"].sum(), mm["away_win"].sum()],
            })
            st.plotly_chart(px.pie(outcome_counts, names="Outcome", values="Count", hole=0.45,
                                    title="Match Outcome Split",
                                    color_discrete_sequence=[BLUE, GOLD, GREEN]),
                             width='stretch')

    if "stadium_name" in mm.columns:
        venue_goals = (
            mm.groupby("stadium_name").agg(Goals=("total_goals", "sum"), Matches=("match_id", "count"))
            .reset_index()
        )
        venue_goals["Goals per Match"] = venue_goals["Goals"] / venue_goals["Matches"]
        st.plotly_chart(
            px.bar(venue_goals.sort_values("Goals per Match"), x="Goals per Match", y="stadium_name",
                   orientation="h", color="Goals per Match", title="Average Goals per Stadium"),
            width='stretch',
        )

    if "confederation" in team_summary_v2.columns and "Goals" in team_summary_v2.columns:
        confed = team_summary_v2.groupby("confederation").agg(
            Goals=("Goals", "sum"), Teams=("team", "count"), Win_Rate=("Win_Rate", "mean")
        ).reset_index()
        c5, c6 = st.columns(2)
        with c5:
            st.plotly_chart(px.bar(confed.sort_values("Goals", ascending=False), x="confederation",
                                    y="Goals", color="Goals", title="Goals by Confederation"),
                             width='stretch')
        with c6:
            st.plotly_chart(px.bar(confed.sort_values("Win_Rate", ascending=False), x="confederation",
                                    y="Win_Rate", color="Win_Rate", title="Average Win Rate by Confederation"),
                             width='stretch')

    section_header("Team Standings")
    st.dataframe(
        team_summary[["team", "matches", "wins", "draws", "losses", "goals",
                       "goals_conceded", "goal_difference", "win_rate", "performance_score"]]
        .sort_values("performance_score", ascending=False),
        width='stretch', hide_index=True,
    )


with tab_map["Overview"]:
    safe_section("Overview", render_overview)


# ============================= TOURNAMENT STAGES =============================
def render_stages():
    section_header("Tournament Progression")

    if not stage_progression:
        st.info("Needs `stage_name` and `date` columns in your matches data.")
        return

    stage_order = stage_progression["stage_order"]
    progression = {p["stage"]: p for p in stage_progression["progression"]}

    # Progression funnel across all stages
    funnel_df = pd.DataFrame([
        {"Stage": s, "Teams": progression[s]["teams"]} for s in stage_order
    ])
    st.plotly_chart(
        px.funnel(funnel_df, x="Teams", y="Stage", title="Team Progression Through the Tournament"),
        width='stretch',
    )

    selected_stage = st.selectbox("Explore a stage", stage_order, index=len(stage_order) - 1)
    p = progression[selected_stage]

    badge = '<span class="badge-done">Completed</span>' if p["is_completed"] else '<span class="badge-live">Ongoing</span>'
    st.markdown(f"### {selected_stage} &nbsp; {badge}", unsafe_allow_html=True)

    sc = st.columns(4)
    kpi_card(sc[0], "Teams Involved", p["teams"])
    kpi_card(sc[1], "Matches Played", p["matches"])
    kpi_card(sc[2], "Goals Scored", p["goals"] if p["goals"] is not None else "—")
    kpi_card(sc[3], "Advancing", p["advancing_teams"] if p["advancing_teams"] is not None else "Final stage")

    stage_matches = stage_progression["matches_by_stage"][selected_stage]
    show_cols = [c for c in ["date", "home_team_name", "away_team_name", "scoreline",
                               "winner", "stadium_name"] if c in stage_matches.columns]

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Match results**")
        st.dataframe(stage_matches[show_cols].sort_values("date"), width='stretch', hide_index=True)
    with c2:
        if p["eliminated_teams"]:
            st.markdown("**Eliminated after this stage**")
            st.dataframe(pd.DataFrame({"Team": p["eliminated_teams"]}), width='stretch', hide_index=True)
        st.markdown("**Advancing / participating teams**")
        st.dataframe(pd.DataFrame({"Team": p["advancing_team_list"]}), width='stretch', hide_index=True)

    # Group standings only shown for the Group Stage
    if "Group" in selected_stage and not group_standings.empty:
        st.divider()
        section_header("Group Standings")
        groups = sorted(group_standings["group"].dropna().unique())
        g_cols = st.columns(min(4, len(groups)) or 1)
        for i, g in enumerate(groups):
            g_table = group_standings[group_standings["group"] == g][
                ["rank_in_group", "team", "played", "won", "drawn", "lost", "gf", "ga", "gd", "points"]
            ].rename(columns={"rank_in_group": "#"})
            with g_cols[i % len(g_cols)]:
                st.markdown(f"**Group {g}**")
                st.dataframe(g_table, width='stretch', hide_index=True)

        st.plotly_chart(
            px.bar(group_standings.sort_values("points", ascending=False), x="team", y="points",
                   color="group", title="Points by Team, Colored by Group"),
            width='stretch',
        )


with tab_map["Tournament Stages"]:
    safe_section("Tournament Stages", render_stages)


# ============================= TEAM INTELLIGENCE =============================
def render_team_intelligence():
    section_header("Team Intelligence Ratings")
    st.caption("Attack, Defense, Control, and Finishing are each scaled 0–100 across all teams, "
               "then blended into an Overall Rating (30% Attack, 30% Defense, 20% Control, 20% Finishing).")

    if team_intelligence.empty:
        st.info("Needs `possession_pct`/opponent stats in match_team_stats.csv to compute this.")
        return

    top_n = st.slider("Show top N teams", 5, min(30, len(team_intelligence)), min(15, len(team_intelligence)))
    top = team_intelligence.head(top_n)

    st.dataframe(
        top[["team", "Attack_Index", "Defense_Index", "Control_Index",
             "Finishing_Index", "Overall_Rating"]].round(1),
        width='stretch', hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(top.sort_values("Overall_Rating"), x="Overall_Rating", y="team",
                                orientation="h", color="Overall_Rating", title="Overall Rating Ranking"),
                         width='stretch')
    with c2:
        fig = go.Figure()
        for _, row in top.head(8).iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row["Attack_Index"], row["Defense_Index"], row["Control_Index"], row["Finishing_Index"]],
                theta=["Attack", "Defense", "Control", "Finishing"],
                fill="toself", name=row["team"],
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                           template=PLOTLY_TEMPLATE, title="Intelligence Radar — Top 8")
        st.plotly_chart(fig, width='stretch')

    st.plotly_chart(
        px.scatter(top, x="Attack_Index", y="Finishing_Index", size="Overall_Rating",
                   color="Defense_Index", text="team", title="Attack vs Finishing (bubble = Overall Rating)"),
        width='stretch',
    )


with tab_map["Team Intelligence"]:
    safe_section("Team Intelligence", render_team_intelligence)


# ================================= STYLE MAP =================================
def render_style_map():
    section_header("Football Style Map")

    if "PC1" not in team_summary_v2.columns:
        st.info("Need at least 5 teams with complete stats to compute style clusters.")
    else:
        st.caption("9 performance metrics per team reduced to 2 dimensions via PCA, then grouped "
                   "into playing styles via K-Means.")
        fig = px.scatter(
            team_summary_v2, x="PC1", y="PC2",
            color="Playing_Style" if has_style else None,
            text="fifa_code" if "fifa_code" in team_summary_v2.columns else "team",
            hover_name="team", title="Playing Style Clusters",
        )
        fig.update_traces(textposition="top center", marker=dict(size=14))
        fig.update_layout(height=650)
        st.plotly_chart(fig, width='stretch')

        if has_style:
            style_counts = team_summary_v2["Playing_Style"].value_counts().reset_index()
            style_counts.columns = ["Playing Style", "Teams"]
            st.plotly_chart(px.bar(style_counts, x="Playing Style", y="Teams", color="Playing Style",
                                    title="Teams per Playing Style"), width='stretch')

    if "Attack_Index" in team_summary_v2.columns:
        section_header("Attack vs Defense Landscape")
        fig2 = px.scatter(
            team_summary_v2, x="Attack_Index", y="Defense_Index",
            size="Goals", color="Playing_Style" if has_style else "confederation",
            text="fifa_code" if "fifa_code" in team_summary_v2.columns else "team",
            hover_name="team", title="Team Performance Landscape",
        )
        fig2.add_vline(x=team_summary_v2["Attack_Index"].mean(), line_dash="dash", line_color="gray")
        fig2.add_hline(y=team_summary_v2["Defense_Index"].mean(), line_dash="dash", line_color="gray")
        fig2.update_traces(textposition="top center")
        fig2.update_layout(height=600)
        st.plotly_chart(fig2, width='stretch')
        st.caption("Right = stronger attack. Higher = stronger defense. Bubble size = total goals.")

    if "xG" in team_summary_v2.columns and "Goals" in team_summary_v2.columns:
        section_header("Goals vs Expected Goals")
        try:
            fig3 = px.scatter(
                team_summary_v2, x="xG", y="Goals",
                size="Shots" if "Shots" in team_summary_v2.columns else None,
                color="confederation" if "confederation" in team_summary_v2.columns else None,
                text="fifa_code" if "fifa_code" in team_summary_v2.columns else "team",
                trendline="ols", hover_name="team", title="Goals vs xG",
            )
        except Exception:
            fig3 = px.scatter(
                team_summary_v2, x="xG", y="Goals",
                size="Shots" if "Shots" in team_summary_v2.columns else None,
                color="confederation" if "confederation" in team_summary_v2.columns else None,
                text="fifa_code" if "fifa_code" in team_summary_v2.columns else "team",
                hover_name="team", title="Goals vs xG",
            )
        max_val = max(team_summary_v2["xG"].max(), team_summary_v2["Goals"].max())
        fig3.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                       line=dict(dash="dash", color="gray"))
        fig3.update_layout(height=600)
        st.plotly_chart(fig3, width='stretch')

        most_clinical = team_summary_v2.loc[team_summary_v2["Goals_minus_xG"].idxmax()]
        most_wasteful = team_summary_v2.loc[team_summary_v2["Goals_minus_xG"].idxmin()]
        c1, c2 = st.columns(2)
        c1.success(f"**Most clinical:** {most_clinical['team']} — "
                   f"{most_clinical['Goals_minus_xG']:.2f} goals more than xG predicted.")
        c2.warning(f"**Most wasteful:** {most_wasteful['team']} — "
                   f"{abs(most_wasteful['Goals_minus_xG']):.2f} goals fewer than xG predicted.")


with tab_map["Style Map"]:
    safe_section("Style Map", render_style_map)


# ============================== MATCH EXCITEMENT ==============================
def render_excitement():
    section_header("Most Exciting Matches")
    st.caption("Excitement Index blends goals, xG, shots, and fouls (positive) with goal difference "
               "(negative — closer games score higher) into one standardized score.")

    if "Excitement_Index" not in match_summary.columns:
        st.info("Not enough match-stat columns to compute an excitement index.")
        return

    top_matches = match_summary.head(15).copy()
    top_matches["Match"] = top_matches["home_team_name"] + " vs " + top_matches["away_team_name"]
    bottom_matches = match_summary.tail(10).copy()
    bottom_matches["Match"] = bottom_matches["home_team_name"] + " vs " + bottom_matches["away_team_name"]

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(top_matches, x="Excitement_Index", y="Match", orientation="h",
                     color="Excitement_Index", text="scoreline" if "scoreline" in top_matches.columns else None,
                     title="Most Exciting Matches", color_continuous_scale="Blues")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=550)
        st.plotly_chart(fig, width='stretch')
    with c2:
        fig = px.bar(bottom_matches, x="Excitement_Index", y="Match", orientation="h",
                     color="Excitement_Index", text="scoreline" if "scoreline" in bottom_matches.columns else None,
                     title="Least Exciting Matches", color_continuous_scale="Greys")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=550)
        st.plotly_chart(fig, width='stretch')

    st.plotly_chart(px.histogram(match_summary, x="Excitement_Index", nbins=20,
                                  title="Distribution of Excitement Across All Matches",
                                  color_discrete_sequence=[BLUE]), width='stretch')

    display_cols = ["home_team_name", "away_team_name", "Excitement_Index"]
    if "scoreline" in match_summary.columns:
        display_cols.insert(2, "scoreline")
    st.dataframe(match_summary[display_cols].head(10).round(2), width='stretch', hide_index=True)


with tab_map["Match Excitement"]:
    safe_section("Match Excitement", render_excitement)


# ================================= GOAL TIMING =================================
def render_goal_timing():
    section_header("When Do Goals Happen?")
    if not goal_timing:
        st.info("Needs `event_type` and `minute` columns in match_events.csv.")
        return

    c1, c2 = st.columns(2)
    with c1:
        timing = goal_timing["timing_distribution"]
        fig = px.bar_polar(timing, r="Goals", theta="Time_Window", color="Goals",
                           color_continuous_scale="Viridis", title="Goal Distribution Throughout a Match")
        st.plotly_chart(fig, width='stretch')
    with c2:
        goals_by_min = goal_timing["goal_events"].groupby("minute").size().reset_index(name="Goals")
        goals_by_min["Cumulative"] = goals_by_min["Goals"].cumsum()
        fig = px.area(goals_by_min, x="minute", y="Cumulative", markers=True,
                     title="Cumulative Goals Over Match Time", color_discrete_sequence=[BLUE])
        st.plotly_chart(fig, width='stretch')

    late_goals = goal_timing["goal_events"][goal_timing["goal_events"]["minute"] >= 90].shape[0]
    total_goals = goal_timing["goal_events"].shape[0]
    st.info(f"⏱️ {late_goals} of {total_goals} goals ({late_goals/total_goals*100:.1f}%) came in "
            f"minute 90 or later — stoppage-time drama.")

    if "team_timing_heatmap" in goal_timing:
        section_header("Team Goal Timing Heatmap")
        heatmap = goal_timing["team_timing_heatmap"]
        fig = px.imshow(heatmap, color_continuous_scale="YlOrRd", aspect="auto", text_auto=True,
                       title="Which Teams Score When")
        fig.update_layout(height=max(500, 20 * len(heatmap)), xaxis_title="Match Time", yaxis_title="Team")
        st.plotly_chart(fig, width='stretch')

    if "half_comparison" in goal_timing and "Improvement" in goal_timing["half_comparison"].columns:
        section_header("First Half vs Second Half")
        hc = goal_timing["half_comparison"].head(15)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=hc["team_name"], x=hc["First Half"], name="First Half",
                             orientation="h", marker_color=BLUE))
        fig.add_trace(go.Bar(y=hc["team_name"], x=hc["Second Half"], name="Second Half",
                             orientation="h", marker_color=GOLD))
        fig.update_layout(barmode="group", title="First Half vs Second Half Goal Production",
                          height=max(400, 28 * len(hc)))
        st.plotly_chart(fig, width='stretch')


with tab_map["Goal Timing"]:
    safe_section("Goal Timing", render_goal_timing)


# ================================= HEAD-TO-HEAD =================================
def render_head_to_head():
    section_header("Compare Two Teams")
    teams_list = sorted(team_summary_v2["team"].unique()) if not team_summary_v2.empty else []
    if len(teams_list) < 2:
        st.info("Need at least 2 teams with computed stats.")
        return

    c1, c2 = st.columns(2)
    team_1 = c1.selectbox("Team A", teams_list, index=0)
    team_2 = c2.selectbox("Team B", teams_list, index=min(1, len(teams_list) - 1))

    row_1 = team_summary_v2[team_summary_v2["team"] == team_1].iloc[0]
    row_2 = team_summary_v2[team_summary_v2["team"] == team_2].iloc[0]

    # --- Head-to-head history, if they've played each other ---
    h2h_matches = mm[
        ((mm["home_team_name"] == team_1) & (mm["away_team_name"] == team_2)) |
        ((mm["home_team_name"] == team_2) & (mm["away_team_name"] == team_1))
    ] if {"home_team_name", "away_team_name"}.issubset(mm.columns) else pd.DataFrame()

    if not h2h_matches.empty:
        st.markdown(f"**Head-to-head history: {team_1} vs {team_2}**")
        show_cols = [c for c in ["date", "stage_name", "scoreline", "winner"] if c in h2h_matches.columns]
        st.dataframe(h2h_matches[show_cols], width='stretch', hide_index=True)
    else:
        st.caption(f"{team_1} and {team_2} haven't played each other yet in this dataset.")

    # --- Radar comparison ---
    radar_metrics = [c for c in ["Attack_Index", "Defense_Index", "Control_Index",
                                   "Win_Rate", "Shot_Accuracy"] if c in team_summary_v2.columns]
    c3, c4 = st.columns(2)
    with c3:
        if radar_metrics:
            fig = go.Figure()
            for row in [row_1, row_2]:
                # Normalize win rate / shot accuracy onto a comparable 0-100-ish scale with the indices
                vals = []
                for m in radar_metrics:
                    v = row[m]
                    vals.append(v)
                fig.add_trace(go.Scatterpolar(r=vals, theta=radar_metrics, fill="toself", name=row["team"]))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                              title=f"{team_1} vs {team_2} — Profile", height=500)
            st.plotly_chart(fig, width='stretch')

    with c4:
        outcomes = pd.DataFrame({
            "Team": [team_1, team_2],
            "Wins": [row_1.get("Wins", 0), row_2.get("Wins", 0)],
            "Draws": [row_1.get("Draws", 0), row_2.get("Draws", 0)],
            "Losses": [row_1.get("Losses", 0), row_2.get("Losses", 0)],
        })
        fig = px.bar(outcomes.melt(id_vars="Team", var_name="Result", value_name="Count"),
                    x="Team", y="Count", color="Result", barmode="stack",
                    title="Results Breakdown", color_discrete_map={"Wins": GREEN, "Draws": GOLD, "Losses": RED})
        st.plotly_chart(fig, width='stretch')

    # --- Side-by-side grouped bar of raw metrics ---
    compare_metrics = [c for c in ["Goals", "Goals_Conceded", "xG", "Possession",
                                     "Shots", "Corners", "Points"] if c in team_summary_v2.columns]
    if compare_metrics:
        section_header("Metric-by-Metric Comparison")
        compare_long = pd.DataFrame({
            "Metric": compare_metrics * 2,
            "Team": [team_1] * len(compare_metrics) + [team_2] * len(compare_metrics),
            "Value": [row_1[m] for m in compare_metrics] + [row_2[m] for m in compare_metrics],
        })
        fig = px.bar(compare_long, x="Metric", y="Value", color="Team", barmode="group",
                    title=f"{team_1} vs {team_2} — Key Metrics",
                    color_discrete_map={team_1: BLUE, team_2: GOLD})
        st.plotly_chart(fig, width='stretch')

    show_cols = [c for c in ["team", "Goals", "Goals_Conceded", "xG", "Possession",
                               "Shots", "Shot_Accuracy", "Win_Rate", "Points"]
                 if c in team_summary_v2.columns]
    compare_table = team_summary_v2[team_summary_v2["team"].isin([team_1, team_2])][show_cols]
    st.dataframe(compare_table.set_index("team").T, width='stretch')


with tab_map["Head-to-Head"]:
    safe_section("Head-to-Head", render_head_to_head)


# ================================= MATCHES =================================
def render_matches():
    section_header("Browse Matches")
    teams_list_all = sorted(set(mm["home_team_name"]).union(mm["away_team_name"]))
    c1, c2 = st.columns(2)
    picked_team = c1.selectbox("Filter by team", ["All"] + teams_list_all)
    stage_options = ["All"] + sorted(mm["stage_name"].dropna().unique().tolist()) if "stage_name" in mm.columns else ["All"]
    picked_stage = c2.selectbox("Filter by stage", stage_options)

    view = mm.copy()
    if picked_team != "All":
        view = view[(view["home_team_name"] == picked_team) | (view["away_team_name"] == picked_team)]
    if picked_stage != "All":
        view = view[view["stage_name"] == picked_stage]

    show_cols = [c for c in ["date", "stage_name", "scoreline", "stadium_name",
                               "total_goals", "winner"] if c in view.columns]
    st.dataframe(view[show_cols].sort_values("date"), width='stretch', hide_index=True)

    c3, c4 = st.columns(2)
    with c3:
        if "total_goals" in view.columns and not view.empty:
            st.plotly_chart(px.histogram(view, x="total_goals", title="Goals per Match (filtered view)",
                                          color_discrete_sequence=[BLUE]), width='stretch')
    with c4:
        if "event_type" in event_mart.columns:
            counts = event_mart["event_type"].value_counts().reset_index()
            counts.columns = ["Event Type", "Count"]
            st.plotly_chart(px.bar(counts, x="Event Type", y="Count", title="Event Type Breakdown"),
                             width='stretch')


with tab_map["Matches"]:
    safe_section("Matches", render_matches)


# ================================= PLAYERS =================================
if "Players" in tab_map:
    def render_players():
        section_header("Player Value vs Contribution")
        ps = player_summary.copy()

        max_minutes = int(ps["Minutes"].max()) if ps["Minutes"].notna().any() and ps["Minutes"].max() > 0 else 1
        min_minutes = st.slider("Minimum minutes played", 0, max_minutes, min(180, max_minutes))
        ps_filtered = ps[ps["Minutes"] >= min_minutes]

        if ps_filtered.empty:
            st.info("No players meet this minutes threshold yet.")
            return

        c1, c2 = st.columns(2)
        with c1:
            if "Market_Value_M" in ps.columns:
                fig = px.scatter(
                    ps_filtered, x="Market_Value_M", y="Goal_Contributions_per_90",
                    size="Minutes", color="position" if "position" in ps.columns else None,
                    hover_name="player_name", text="player_name",
                    title="Player Value vs Tournament Contribution",
                )
                fig.update_traces(textposition="top center")
                st.plotly_chart(fig, width='stretch')
        with c2:
            if "Yellow_Cards" in ps.columns:
                cards = ps_filtered.groupby("position" if "position" in ps.columns else "player_name")[
                    ["Yellow_Cards", "Red_Cards"]].sum().reset_index()
                fig = px.bar(cards, x=cards.columns[0], y=["Yellow_Cards", "Red_Cards"], barmode="group",
                            title="Discipline by Position",
                            color_discrete_map={"Yellow_Cards": GOLD, "Red_Cards": RED})
                st.plotly_chart(fig, width='stretch')

        section_header("Top Scorers & Contributors")
        top_players = ps_filtered.sort_values("Goal_Contributions", ascending=False).head(20)
        st.plotly_chart(px.bar(top_players.head(10), x="player_name", y="Goal_Contributions",
                                color="Goals" if "Goals" in top_players.columns else None,
                                title="Top 10 Goal Contributors"), width='stretch')

        show_cols = [c for c in ["player_name", "position", "Goals", "Assists",
                                   "Goal_Contributions", "Goals_per_90", "Minutes"] if c in top_players.columns]
        st.dataframe(top_players[show_cols].round(2), width='stretch', hide_index=True)

    with tab_map["Players"]:
        safe_section("Players", render_players)
