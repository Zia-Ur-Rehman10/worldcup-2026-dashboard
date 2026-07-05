"""
Pre-commit data validation — runs the ENTIRE dashboard pipeline against
whatever is currently in raw_data/ and fails loudly if anything breaks.

This exists so that a schema change in the upstream Kaggle dataset
(renamed column, new required file, a data quality issue, etc.) causes
the GitHub Action to fail and skip the commit — leaving your last
known-good data (and your live dashboard) untouched — instead of
silently pushing broken data that crashes the deployed app.

Exit code 0  = pipeline ran successfully end-to-end, safe to commit.
Exit code 1  = something broke; do NOT commit/push this data.
"""

from pathlib import Path
import sys
import types
import traceback

# Mock streamlit's cache_data decorator so pipeline.py can run outside
# an actual Streamlit session (same trick as test_pipeline.py).
fake_st = types.ModuleType("streamlit")


def _cache_data_decorator(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def wrapper(f):
        return f
    return wrapper


fake_st.cache_data = _cache_data_decorator
sys.modules["streamlit"] = fake_st

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
RAW_PATH = REPO_ROOT / "raw_data"


def main():
    print(f"Validating data in: {RAW_PATH}")

    try:
        from pipeline import run_pipeline
        result = run_pipeline(str(RAW_PATH))
    except Exception:
        print("\n" + "=" * 60)
        print("VALIDATION FAILED — the pipeline could not process this data.")
        print("=" * 60)
        traceback.print_exc()
        print("\nThe fetched data will NOT be committed. Your live dashboard")
        print("is unaffected — it's still running on the last good data.")
        print("\nMost likely cause: the upstream dataset changed its column")
        print("names, removed a required file, or changed a data type.")
        print("Compare the traceback above against src/pipeline.py to find")
        print("the mismatch, or share this output for help fixing it.")
        return 1

    # Basic sanity checks beyond "it didn't crash" — catch silent data
    # quality problems too.
    problems = []

    if result["master_matches"].empty:
        problems.append("master_matches is empty — no matches were loaded at all.")

    if result["team_summary"].empty:
        problems.append("team_summary is empty — no completed matches to compute standings from.")

    kpis = result["kpis"]
    if kpis.get("Teams", 0) == 0:
        problems.append("KPI 'Teams' is 0 — teams.csv may be empty or failed to merge.")

    if problems:
        print("\n" + "=" * 60)
        print("VALIDATION FAILED — pipeline ran but produced suspicious output:")
        print("=" * 60)
        for p in problems:
            print(f"  - {p}")
        print("\nThe fetched data will NOT be committed.")
        return 1

    print("\n" + "=" * 60)
    print("VALIDATION PASSED")
    print("=" * 60)
    print(f"  Matches loaded    : {len(result['master_matches'])}")
    print(f"  Matches played    : {kpis['Matches Played']}")
    print(f"  Upcoming matches  : {kpis.get('Upcoming Matches', 'n/a')}")
    print(f"  Teams             : {kpis['Teams']}")
    print(f"  Total goals       : {kpis['Goals']}")
    print("\nSafe to commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
