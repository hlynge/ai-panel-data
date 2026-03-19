"""
oecd_api.py — thin wrapper around the OECD SDMX-JSON / CSV API.

The OECD exposes two API generations:
  • Old API  : https://stats.oecd.org/SDMX-JSON/data/  (stable, widely used)
  • New API  : https://sdmx.oecd.org/public/rest/data/  (SDMX 2.1, 2024+)

We primarily use the *old* API because dataset IDs are well-documented and
stable. The new API is used for MSTI and other datasets that migrated there.

Reference:
  https://data.oecd.org/api/sdmx-json-documentation/
"""

from __future__ import annotations

import io
import time
from typing import Optional

import pandas as pd

from .utils import fetch_url, get_logger

logger = get_logger(__name__)

# ── Endpoints ─────────────────────────────────────────────────────────────────

OLD_API_BASE = "https://stats.oecd.org/SDMX-JSON/data"
NEW_API_BASE = "https://sdmx.oecd.org/public/rest/data"

# Pause between successive API calls to be a polite client (OECD rate-limits
# at ~20 req/min on the new API; the old API is more permissive).
_MIN_REQUEST_INTERVAL = 2.0   # seconds
_last_call: float = 0.0


def _rate_limit() -> None:
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_call = time.time()


# ── Old API ───────────────────────────────────────────────────────────────────

def fetch_dataset_old(
    dataset_id: str,
    filter_expr: str = "all",
    agency: str = "OECD",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Download a dataset from the *old* OECD SDMX-JSON API and return a
    tidy pandas DataFrame.

    Parameters
    ----------
    dataset_id  : OECD dataset identifier, e.g. "MSTI_PUB", "PATS_IPC".
    filter_expr : Dot-separated dimension filter.  Use "all" for everything.
                  Example for patents in class G06N:
                      "G06N.PCT_APPLICATIONS.../..."
    agency      : Data-providing agency (default "OECD").
    start_year  : First year to request (inclusive).
    end_year    : Last year to request (inclusive).

    Returns
    -------
    DataFrame with at least columns: COU (country), TIME (year), Value.
    """
    url = f"{OLD_API_BASE}/{dataset_id}/{filter_expr}/{agency}"
    params = {"format": "csv"}
    if start_year:
        params["startTime"] = start_year
    if end_year:
        params["endTime"] = end_year

    logger.info("Fetching OECD [old API]: %s / %s", dataset_id, filter_expr)
    _rate_limit()
    resp = fetch_url(url, params=params)
    df = pd.read_csv(io.StringIO(resp.text))
    logger.info("  → %d rows", len(df))
    return df


# ── New API (SDMX 2.1) ────────────────────────────────────────────────────────

def fetch_dataset_new(
    dataflow: str,
    key: str = "all",
    start_period: Optional[int] = None,
    end_period: Optional[int] = None,
    fmt: str = "csvfilewithlabels",
) -> pd.DataFrame:
    """
    Download a dataset from the *new* OECD SDMX 2.1 API.

    Parameters
    ----------
    dataflow    : Full dataflow reference, e.g.
                  "OECD.STI.STP,DSD_MSTI@DF_MSTI,2.0"
    key         : Dimension key, e.g. "AUS+BEL.GERD_GDPB.PC_GDP"
                  Use "all" for all observations.
    start_period: First year (YYYY).
    end_period  : Last year (YYYY).
    fmt         : Response format.  "csvfilewithlabels" gives a flat CSV with
                  both codes and human-readable labels.

    Returns
    -------
    DataFrame as returned by the API.
    """
    url = f"{NEW_API_BASE}/{dataflow}/{key}"
    params: dict = {"format": fmt}
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period

    logger.info("Fetching OECD [new API]: %s", dataflow)
    _rate_limit()
    resp = fetch_url(url, params=params)
    df = pd.read_csv(io.StringIO(resp.text))
    logger.info("  → %d rows", len(df))
    return df


# ── Convenience: normalise column names ───────────────────────────────────────

def normalise_old_api(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names from the old OECD CSV export to lowercase snake_case
    and rename common dimension columns for consistency.

    Common columns in old API output:
      "COUNTRY" | "COU"    → country_code
      "TIME"               → year
      "Value"              → value
      "Flag Codes"         → flag
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

    # year → integer where possible
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # value → float
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df
