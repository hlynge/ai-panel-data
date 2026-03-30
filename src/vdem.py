"""
vdem.py — load Varieties of Democracy (V-Dem) data.

V-Dem provides 500+ democracy and governance indicators for 202 countries
from 1789 to the present.  There is no public REST API; data is distributed
as a large CSV / RDS file that must be downloaded from:

    https://v-dem.net/data/the-v-dem-dataset/

This module:
  1. Checks whether a local file path is provided in config.yaml.
  2. If not, attempts a direct download from V-Dem's distribution URL
     (the URL pattern is versioned; set VDEM_DOWNLOAD_URL below if it changes).
  3. Subsets to the configured indicators and year range.
  4. Returns a tidy country-year DataFrame.

Tip for R users
───────────────
  If you already have the `vdemdata` R package installed, you can export the
  data from R with:
      library(vdemdata)
      write.csv(vdem, "data/raw/vdem.csv", row.names = FALSE)
  and then point config.yaml → vdem.local_file to that path.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .utils import get_logger, save_raw

logger = get_logger(__name__)

# Latest known direct-download URL (update if V-Dem releases a new version)
# V-Dem v15 (2025):
VDEM_DOWNLOAD_URL = (
    "https://v-dem.net/media/datasets/V-Dem-CY-Full+Others-v16.csv.zip"
)

# Fallback: Harvard Dataverse permanent DOI landing page
# Users can download from: https://doi.org/10.7910/DVN/T9SDEW
VDEM_DATAVERSE_URL = "https://doi.org/10.7910/DVN/T9SDEW"


# Core identifier columns always included
VDEM_ID_COLS = ["country_name", "country_text_id", "country_id", "year", "COWcode"]


def _download_vdem(dest_dir: Path) -> Optional[Path]:
    """
    Try to download the V-Dem CSV zip to *dest_dir*.  Returns the local path
    of the extracted CSV, or None if the download fails.
    """
    zip_path = dest_dir / "vdem.csv.zip"
    csv_path = dest_dir / "vdem.csv"

    if csv_path.exists():
        logger.info("Using cached V-Dem CSV: %s", csv_path)
        return csv_path

    logger.info("Attempting V-Dem download from %s …", VDEM_DOWNLOAD_URL)
    try:
        resp = requests.get(
            VDEM_DOWNLOAD_URL,
            timeout=300,
            stream=True,
            headers={"User-Agent": "ai-panel-data/1.0"},
        )
        resp.raise_for_status()
        with open(zip_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
        logger.info("Downloaded V-Dem zip → %s", zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError("No CSV found inside the zip archive.")
            zf.extract(csv_names[0], dest_dir)
            extracted = dest_dir / csv_names[0]
            extracted.rename(csv_path)

        zip_path.unlink(missing_ok=True)
        logger.info("V-Dem CSV extracted → %s", csv_path)
        return csv_path

    except Exception as exc:
        logger.warning(
            "V-Dem download failed: %s\n"
            "─────────────────────────────────────────────────────────\n"
            "Please download the V-Dem dataset manually:\n"
            "  1. Go to  https://v-dem.net/data/the-v-dem-dataset/\n"
            "  2. Download the 'Country-Year: V-Dem Full+Others' CSV.\n"
            "  3. Unzip and set  vdem.local_file  in config.yaml.\n"
            "─────────────────────────────────────────────────────────",
            exc,
        )
        return None


def load_vdem(
    indicators: dict[str, str],
    start_year: int = 2010,
    end_year: int = 2024,
    local_file: Optional[str | Path] = None,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """
    Load V-Dem data and return a tidy country-year DataFrame.

    Parameters
    ----------
    indicators  : Mapping of V-Dem column name → friendly output column name.
    start_year  : First year to include.
    end_year    : Last year to include.
    local_file  : Path to a locally downloaded V-Dem CSV (or None to auto-download).
    raw_dir     : Directory for caching downloads.

    Returns
    -------
    DataFrame with columns: iso3 (= country_text_id), year, <indicator_names…>
    """
    dest_dir = Path(raw_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Resolve CSV path
    if local_file and Path(local_file).exists():
        csv_path = Path(local_file)
        logger.info("Loading V-Dem from local file: %s", csv_path)
    else:
        csv_path = _download_vdem(dest_dir)

    if csv_path is None:
        logger.error("V-Dem data unavailable; skipping.")
        return pd.DataFrame()

    # Determine columns to load (ID cols + requested indicators)
    vdem_cols = indicators.keys()
    cols_to_load = list(set(VDEM_ID_COLS) | set(vdem_cols))

    logger.info("Reading V-Dem CSV (this may take a moment for the full dataset)…")
    try:
        # Read only the columns we need to save memory
        df = pd.read_csv(
            csv_path,
            usecols=lambda c: c in cols_to_load,
            low_memory=True,
        )
    except Exception as exc:
        logger.warning("Column-selective read failed (%s); reading full CSV…", exc)
        df = pd.read_csv(csv_path, low_memory=True)
        present = [c for c in cols_to_load if c in df.columns]
        df = df[present]

    # Year filter
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()

    # Rename indicator columns to friendly names
    rename = {k: v for k, v in indicators.items() if k in df.columns}
    missing = [k for k in indicators if k not in df.columns]
    if missing:
        logger.warning("V-Dem columns not found (may need a newer version): %s", missing)
    df.rename(columns=rename, inplace=True)

    # Standardise country identifier
    # country_text_id is already ISO3 in V-Dem (e.g. "USA", "GBR")
    if "country_text_id" in df.columns:
        df.rename(columns={"country_text_id": "iso3"}, inplace=True)

    # Keep only final columns
    keep_cols = ["iso3", "year", "country_name"] + list(rename.values())
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].drop_duplicates(subset=["iso3", "year"]).reset_index(drop=True)

    # Convert indicator columns to float
    ind_cols = list(rename.values())
    for col in ind_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    save_raw(df, "vdem_raw", raw_dir)
    logger.info("V-Dem: %d country-year rows, %d indicators", len(df), len(ind_cols))
    return df


def fetch_all_vdem(
    config: dict,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Entry point called by the main pipeline."""
    vdem_cfg = config.get("vdem", {})
    if not vdem_cfg.get("enabled", True):
        logger.info("V-Dem disabled in config.")
        return pd.DataFrame()

    return load_vdem(
        indicators=vdem_cfg.get("indicators", {}),
        start_year=config["pipeline"]["start_year"],
        end_year=config["pipeline"]["end_year"],
        local_file=vdem_cfg.get("local_file"),
        raw_dir=raw_dir,
    )
