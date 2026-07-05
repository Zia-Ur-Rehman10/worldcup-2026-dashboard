"""
Automated Data Fetcher — pulls the FIFA World Cup 2026 dataset directly
from Kaggle, skipping the Colab -> Drive -> manual download step entirely.

Dataset: https://www.kaggle.com/datasets/mominullptr/fifa-world-cup-2026-dataset

Requires a Kaggle API token, provided as an environment variable:
    KAGGLE_API_TOKEN

Locally, you can instead run `kaggle auth login` once, or save the
token to ~/.kaggle/access_token yourself, and the Kaggle client will
read it automatically without needing the environment variable.
"""

from pathlib import Path
import os
import shutil
import sys
import tempfile

import pandas as pd

DATASET = "mominullptr/fifa-world-cup-2026-dataset"
RAW_PATH = Path(__file__).parent.parent / "raw_data"


def _prepare_kaggle_credentials():
    """Must run BEFORE the kaggle package is imported — importing it
    triggers an authentication check immediately, so credentials need
    to already be in place by then.

    Supports two methods:
      1. KAGGLE_CREDENTIALS_JSON — the full contents of a working
         ~/.kaggle/credentials.json (OAuth access+refresh token pair).
         This is the method confirmed to work via `kaggle auth login`.
      2. KAGGLE_API_TOKEN — a simpler single-token value, written to
         ~/.kaggle/access_token. Kept as a fallback in case your Kaggle
         account uses the older, simpler token method.
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    creds_json = os.environ.get("KAGGLE_CREDENTIALS_JSON")
    if creds_json:
        (kaggle_dir / "credentials.json").write_text(creds_json.strip())
        return

    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        (kaggle_dir / "access_token").write_text(token.strip())


def download_dataset(tmp_dir: Path) -> Path:
    """Downloads and unzips the Kaggle dataset into tmp_dir. Returns the
    folder containing the extracted CSVs."""
    _prepare_kaggle_credentials()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()  # reads ~/.kaggle/credentials.json (OAuth) or
                         # ~/.kaggle/access_token, whichever is present

    print(f"Downloading dataset: {DATASET}")
    api.dataset_download_files(DATASET, path=str(tmp_dir), unzip=True, quiet=False)

    return tmp_dir


def files_are_equal(path_a: Path, path_b: Path) -> bool:
    if not path_b.exists():
        return False
    try:
        df_a = pd.read_csv(path_a)
        df_b = pd.read_csv(path_b)
        return df_a.equals(df_b)
    except Exception:
        # Fall back to a raw byte comparison if either isn't a clean CSV read
        return path_a.read_bytes() == path_b.read_bytes()


def sync_to_raw_data(source_dir: Path) -> bool:
    """Copies every CSV found in source_dir into raw_data/, only
    overwriting files that actually changed. Returns True if anything
    was updated."""
    RAW_PATH.mkdir(parents=True, exist_ok=True)
    any_changes = False

    csv_files = list(source_dir.rglob("*.csv"))
    if not csv_files:
        print("  No CSV files found in the downloaded dataset — check the dataset contents.")
        return False

    for src_file in csv_files:
        dest_file = RAW_PATH / src_file.name
        if files_are_equal(src_file, dest_file):
            print(f"  {src_file.name}: no changes.")
            continue

        shutil.copy2(src_file, dest_file)
        print(f"  {src_file.name}: updated.")
        any_changes = True

    return any_changes


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            extracted_dir = download_dataset(tmp_path)
        except (Exception, SystemExit) as e:
            print(f"Download failed: {e}")
            print("Make sure KAGGLE_CREDENTIALS_JSON (or KAGGLE_API_TOKEN) is set "
                  "as an env var locally, or as a GitHub repo secret when run via Actions.")
            print("CHANGES_DETECTED=false")
            return 1

        any_changes = sync_to_raw_data(extracted_dir)

    print("CHANGES_DETECTED=" + ("true" if any_changes else "false"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
