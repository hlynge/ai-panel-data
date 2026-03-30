"""
utils.py — shared helpers used across all source modules.

Covers:
  - Logging setup
  - Rate-limited HTTP requests
  - Country-code harmonisation (ISO alpha-3)
  - Config loading
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml

# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a consistent format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str | Path = "config.yaml") -> dict:
    """Load and return the YAML config file as a nested dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

_DEFAULT_HEADERS = {
    "User-Agent": "ai-panel-data/1.0 (academic research project)"
}

def fetch_url(
    url: str,
    params: Optional[dict] = None,
    retries: int = 3,
    backoff: float = 2.0,
    timeout: int = 120,
    **kwargs,
) -> requests.Response:
    """
    GET *url* with automatic retries and exponential back-off.

    Parameters
    ----------
    url      : Full URL to fetch.
    params   : Optional query parameters dict.
    retries  : Number of attempts before raising.
    backoff  : Base seconds to wait between retries (doubles each time).
    timeout  : Request timeout in seconds.

    Returns
    -------
    requests.Response with status 200.
    """
    logger = get_logger("utils.fetch_url")
    session = requests.Session()
    session.headers.update(_DEFAULT_HEADERS)

    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            wait = backoff * (2 ** (attempt - 1))
            logger.warning("Attempt %d/%d failed: %s. Retrying in %.0fs…", attempt, retries, exc, wait)
            time.sleep(wait)


# ── Country-code harmonisation ────────────────────────────────────────────────

try:
    import country_converter as coco
    _CC = coco.CountryConverter()
    _COCO_AVAILABLE = True
except ImportError:
    _COCO_AVAILABLE = False

# Manual overrides for codes that country_converter sometimes misses
_MANUAL_ISO3 = {
    "Kosovo": "XKX",
    "TFYR Macedonia": "MKD",
    "North Macedonia": "MKD",
    "Taiwan": "TWN",
    "Hong Kong, China": "HKG",
    "Macao, China": "MAC",
    "Czech Republic": "CZE",
    "Slovak Republic": "SVK",
    "Korea": "KOR",
    "Korea, Rep.": "KOR",
    "Iran, Islamic Rep.": "IRN",
    "Venezuela, RB": "VEN",
    "Egypt, Arab Rep.": "EGY",
    "Russian Federation": "RUS",
    "Kyrgyz Republic": "KGZ",
    "Lao PDR": "LAO",
    "Brunei Darussalam": "BRN",
    "Congo, Dem. Rep.": "COD",
    "Congo, Rep.": "COG",
    "Cote d'Ivoire": "CIV",
    "Gambia, The": "GMB",
    "Yemen, Rep.": "YEM",
    "Syrian Arab Republic": "SYR",
    "West Bank and Gaza": "PSE",
    "Micronesia, Fed. Sts.": "FSM",
    "St. Kitts and Nevis": "KNA",
    "St. Lucia": "LCA",
    "St. Vincent and the Grenadines": "VCT",
    "Eswatini": "SWZ",
    "Türkiye": "TUR",
    "Turkey": "TUR",
}


def to_iso3(name_or_code: str) -> Optional[str]:
    """
    Convert a country name or 2-letter code to ISO 3166-1 alpha-3.

    Returns None if no match is found (rather than raising).
    """
    if pd.isna(name_or_code) or not name_or_code:
        return None

    # Already looks like an ISO3 code (3 uppercase letters)
    code = str(name_or_code).strip()
    if code in _MANUAL_ISO3.values():
        return code

    # Check manual overrides first
    if code in _MANUAL_ISO3:
        return _MANUAL_ISO3[code]

    if _COCO_AVAILABLE:
        result = _CC.convert(code, to="ISO3", not_found=None)
        if result and result != "not found":
            return result

    return None


def harmonise_country_column(
    df: pd.DataFrame,
    col: str = "country",
    iso3_col: str = "iso3",
) -> pd.DataFrame:
    """
    Add an *iso3_col* column to *df* by converting *col* via `to_iso3`.

    Rows where conversion fails are kept but get NaN in *iso3_col*.
    """
    df = df.copy()
    df[iso3_col] = df[col].apply(to_iso3)
    n_failed = df[iso3_col].isna().sum()
    if n_failed:
        failed = df.loc[df[iso3_col].isna(), col].unique().tolist()
        get_logger("utils").warning(
            "%d rows could not be matched to an ISO3 code: %s", n_failed, failed[:20]
        )
    return df


# ── Persistence helpers ───────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_raw(df: pd.DataFrame, name: str, raw_dir: str | Path = "data/raw") -> Path:
    """Save *df* as a parquet file in *raw_dir* for caching / reproducibility."""
    ensure_dir(raw_dir)
    path = Path(raw_dir) / f"{name}.parquet"
    df.to_parquet(path, index=False)
    get_logger("utils").info("Saved raw data → %s (%d rows)", path, len(df))
    return path
