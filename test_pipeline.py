"""Quick smoke test: run pipeline functions without needing `streamlit run`."""
import sys
import types

# Mock streamlit's cache_data decorator so pipeline.py imports cleanly
# outside of a real Streamlit session.
fake_st = types.ModuleType("streamlit")
fake_st.cache_data = lambda *a, **k: (lambda f: f) if not callable(a[0] if a else None) else a[0]
def cache_data_decorator(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    def wrapper(f):
        return f
    return wrapper
fake_st.cache_data = cache_data_decorator
sys.modules["streamlit"] = fake_st

sys.path.append("src")
from pipeline import run_pipeline

result = run_pipeline("raw_data")

print("KPIs:", result["kpis"])
print("\nTeam summary:\n", result["team_summary"][["team", "goals", "win_rate", "performance_score"]])
print("\nMaster matches shape:", result["master_matches"].shape)
print("\nTeam match mart shape:", result["team_match_mart"].shape)
print("\nInsights:", result["insights"])
print("\nData version:", result["version"])

print("\n--- Team Intelligence ---")
print(result["team_intelligence"][["team", "Attack_Index", "Defense_Index", "Overall_Rating"]].round(1) if not result["team_intelligence"].empty else "EMPTY")

print("\n--- Team Summary V2 (style clusters) ---")
tsv2 = result["team_summary_v2"]
cols = [c for c in ["team", "Points", "Overall_Index", "Playing_Style"] if c in tsv2.columns]
print(tsv2[cols] if cols else "MISSING COLS", tsv2.columns.tolist())

print("\n--- Match Excitement ---")
ms = result["match_summary"]
print(ms[["home_team_name","away_team_name","Excitement_Index"]].head() if "Excitement_Index" in ms.columns else "NO EXCITEMENT INDEX")

print("\n--- Goal Timing ---")
gt = result["goal_timing"]
print(list(gt.keys()))

print("\n--- Player Summary ---")
psum = result["player_summary"]
print(psum[["player_name","Goals","Assists","Goal_Contributions_per_90"]].sort_values("Goal_Contributions_per_90", ascending=False).head() if not psum.empty else "EMPTY")

print("\n--- Advanced Insights ---")
for i in result["advanced_insights"]:
    print("-", i)

print("\n--- Group Standings ---")
gs = result["group_standings"]
print(gs if not gs.empty else "EMPTY")

print("\n--- Stage Progression ---")
sp = result["stage_progression"]
print("Stage order:", sp.get("stage_order"))
for p in sp.get("progression", []):
    print(f"  {p['stage']}: {p['teams']} teams, {p['matches']} matches, advancing={p['advancing_teams']}")

print("\nALL GOOD")
