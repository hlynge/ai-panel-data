"""
oecd_api.py — thin wrapper around the OECD SDMX APIs.

We use the *new* OECD SDMX 2.1 API (sdmx.oecd.org) as primary, because it
supports proper startPeriod/endPeriod filtering which keeps response sizes
manageable. The old API (stats.oecd.org) is kept as a fallback but returns
the full dataset which can be very large and slow to parse.

New API reference:
  https://www.oecd.org/en/data/insights/data-explainers/2024/09/api.html
"""

from __future__ import annotations

import io
import time
from typing import Optional

import pandas as pd

from .utils import fetch_url, get_logger

logger = get_logger(__name__)

# ── Endpoints ─────────────────────────────────────────────────────────────────

NEW_API_BASE = "https://sdmx.oecd.org/public/rest/data"
OLD_API_BASE = "https://stats.oecd.org/SDMX-JSON/data"

# Pause between successive API calls — OECD rate-limits at ~20 req/min
_MIN_REQUEST_INTERVAL = 3.0   # seconds
_last_call: float = 0.0


def _rate_limit() -> None:
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_call = time.time()


# ── New API (primary) ─────────────────────────────────────────────────────────

# Known dataflow IDs for the new OECD SDMX 2.1 API
DATAFLOWS = {
    "msti":       "OECD.STI.STP,DSD_MSTI@DF_MSTI,1.3",
    "pats":       "OECD.STI.STP,DSD_PATS@DF_PATS_IPC,1.0",
    "ict":        "OECD.STI.DEP,DSD_ICT_HH_IND@DF_HH,1.1",
    "berd":       "OECD.STI.STP,DSD_RDS_BERD@DF_BERD_INDU,1.0",
    # OECD AI-specific patent statistics (OECD's own AI patent definition,
    # based on CPC codes + text mining — see OECD AI Papers No. 30, 2024)
    # Agency: OECD.STI.PIE  Dimensions (12):
    #   PATENT_AUTHORITIES . FREQ . MEASURE . UNIT_MEASURE . DATE_TYPE .
    #   REF_AREA . PARTNER_AREA . AGENT_ROLE . COOPERATION_TYPE . WIPO .
    #   OECD_TECHNOLOGY_PATENT . TIME_PERIOD
    "ai_patents": "OECD.STI.PIE,DSD_PATENTS@DF_PATENTS_OECDSPECIFIC,1.0",
}


def fetch_dataset_new(
    dataflow: str,
    key: str = "all",
    start_period: Optional[int] = None,
    end_period: Optional[int] = None,
    fmt: str = "csvfilewithlabels",
    timeout: int = 300,
) -> pd.DataFrame:
    """
    Download a dataset from the new OECD SDMX 2.1 API.

    Parameters
    ----------
    dataflow     : Full dataflow reference, e.g.
                   "OECD.STI.STP,DSD_MSTI@DF_MSTI,2.0"
                   or a short alias from DATAFLOWS dict ("msti", "pats", etc.)
    key          : Dimension filter key. Use "all" for all observations.
    start_period : First year (YYYY).
    end_period   : Last year (YYYY).
    fmt          : "csvfilewithlabels" returns flat CSV with codes + labels.
    timeout      : Request timeout in seconds.

    Returns
    -------
    DataFrame as returned by the API.
    """
    # Allow short aliases
    dataflow = DATAFLOWS.get(dataflow, dataflow)

    url = f"{NEW_API_BASE}/{dataflow}/{key}"
    params: dict = {"format": fmt}
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period

    logger.info("Fetching OECD [new API]: %s  key=%s  %s–%s",
                dataflow.split(",")[1], key, start_period, end_period)
    _rate_limit()
    resp = fetch_url(url, params=params, timeout=timeout)

    # The new API returns a CSV; parse it
    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    logger.info("  → %d rows, %d cols", *df.shape)
    return df


# ── Old API (fallback) ────────────────────────────────────────────────────────

def fetch_dataset_old(
    dataset_id: str,
    filter_expr: str = "all",
    agency: str = "OECD",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    timeout: int = 300,
) -> pd.DataFrame:
    """
    Download a dataset from the old OECD SDMX-JSON API.

    Note: this returns the *full* dataset without server-side filtering,
    so it can be slow for large datasets. Prefer fetch_dataset_new() where
    possible.
    """
    url = f"{OLD_API_BASE}/{dataset_id}/{filter_expr}/{agency}"
    params = {"format": "csv"}
    if start_year:
        params["startTime"] = start_year
    if end_year:
        params["endTime"] = end_year

    logger.info("Fetching OECD [old API]: %s / %s", dataset_id, filter_expr)
    _rate_limit()
    resp = fetch_url(url, params=params, timeout=timeout)
    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    logger.info("  → %d rows", len(df))
    return df


# ── Normalise column names ────────────────────────────────────────────────────

def normalise_new_api(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names from the new OECD CSV export.

    The new API uses columns like:
      REF_AREA        → country_code
      TIME_PERIOD     → year
      OBS_VALUE       → value
      MEASURE / VAR   → kept as-is (varies by dataset)
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        cu = c.upper()
        if cu in ("REF_AREA", "REFERENCE_AREA"):
            rename[c] = "country_code"
        elif cu == "TIME_PERIOD":
            rename[c] = "year"
        elif cu == "OBS_VALUE":
            rename[c] = "value"
        elif cu == "OBS_STATUS":
            rename[c] = "flag"
    df.rename(columns=rename, inplace=True)

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df


def normalise_old_api(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names from the old OECD CSV export.

    Common columns:
      COUNTRY | COU | LOCATION → country_code
      TIME                     → year
      Value                    → value
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        cl = c.upper()
        if cl in ("COUNTRY", "COU", "LOCATION"):
            rename[c] = "country_code"
        elif cl == "TIME":
            rename[c] = "year"
        elif cl == "VALUE":
            rename[c] = "value"
        elif cl in ("FLAG CODES", "FLAG_CODES"):
            rename[c] = "flag"
    df.rename(columns=rename, inplace=True)

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df
