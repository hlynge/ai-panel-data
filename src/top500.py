"""
top500.py — supercomputer infrastructure by country-year from the Top500 list.

The Top500 list (https://www.top500.org) ranks the 500 most powerful
supercomputers worldwide, published biannually (June and November).

This module downloads the official Top500 XML data files — one per list
release, freely accessible without authentication.

Download URL pattern:
  https://www.top500.org/lists/top500/{YEAR}/{MONTH:02d}/download/TOP500_{YEAR}{MONTH:02d}_all.xml

Each XML file contains one <top500:site> element per system with:
  <top500:country>      country name (full English name)
  <top500:r-max>        measured performance in GFlop/s (float)

For each biannual list the module records per country:
  - Number of systems in the Top500
  - Sum of Rmax (converted to PFlop/s)
  - Max Rmax (single fastest system)

Biannual values (June + November) are collapsed to an annual figure
by taking the maximum across the two releases.

Variables produced
──────────────────
  top500_n_systems       : count of systems in the Top500 list
  top500_total_rmax      : sum of Rmax (PFlop/s) of all country systems
  top500_max_rmax        : Rmax of the single fastest system in the country

Coverage: up to ~30 countries, 2010–2024.  Data are heavily dominated by
the United States and China.

Notes
─────
  • Rmax is stored in GFlop/s in the XML; we convert to PFlop/s (÷ 1e6).
  • XML files are cached locally under raw_dir to avoid re-downloads.
  • Requests are spaced 0.5 s apart to be polite to the server.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests

from .utils import get_logger, save_raw, to_iso3

logger = get_logger(__name__)


_BASE    = "https://www.top500.org"
_HEADERS = {"User-Agent": "ai-panel-data/1.0 (academic research)"}
_SLEEP   = 0.5   # seconds between requests

# GFlop/s → PFlop/s conversion
_GFLOPS_TO_PFLOPS = 1e-6


# ── XML parsing ───────────────────────────────────────────────────────────────

_SITE_RE    = re.compile(r"<top500:site>([\s\S]*?)</top500:site>")
_COUNTRY_RE = re.compile(r"<top500:country>(.*?)</top500:country>")
_RMAX_RE    = re.compile(r"<top500:r-max>([\d.eE+-]+)</top500:r-max>")


def _parse_xml(xml_text: str) -> list[dict]:
    """
    Parse a Top500 XML file and return one dict per system.

    Each dict has keys: "country" (str), "rmax_pflops" (float | None).
    """
    results = []
    for site_m in _SITE_RE.finditer(xml_text):
        site = site_m.group(1)

        c_m = _COUNTRY_RE.search(site)
        r_m = _RMAX_RE.search(site)

        country = c_m.group(1).strip() if c_m else None
        if not country:
            continue

        rmax_gflops = float(r_m.group(1)) if r_m else None
        rmax_pflops = rmax_gflops * _GFLOPS_TO_PFLOPS if rmax_gflops is not None else None

        results.append({"country": country, "rmax_pflops": rmax_pflops})

    return results


# ── Download helpers ──────────────────────────────────────────────────────────

def _xml_url(year: int, month: int) -> str:
    return f"{_BASE}/lists/top500/{year}/{month:02d}/download/TOP500_{year}{month:02d}_all.xml"


def _load_xml(year: int, month: int, cache_dir: Path) -> list[dict]:
    """
    Return parsed system list for one Top500 release.

    Uses a local cache file to avoid re-downloading on repeated runs.
    """
    cache_file = cache_dir / f"top500_{year}{month:02d}.xml"

    if cache_file.exists():
        logger.info("    Using cached XML: %s", cache_file.name)
        xml_text = cache_file.read_text(encoding="utf-8")
    else:
        url = _xml_url(year, month)
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=60)
            if resp.status_code == 404:
                logger.info("    No list for %d/%02d (404); skipping.", year, month)
                return []
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("    Download failed for %d/%02d: %s", year, month, exc)
            return []

        xml_text = resp.text
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(xml_text, encoding="utf-8")
        time.sleep(_SLEEP)

    return _parse_xml(xml_text)


# ── Aggregation ───────────────────────────────────────────────────────────────

def _aggregate(rows: list[dict], year: int) -> pd.DataFrame:
    """
    Aggregate system-level rows to a country-level DataFrame.

    Returns columns: iso3, year, _n_systems, _total_rmax, _max_rmax
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["rmax_pflops"] = pd.to_numeric(df["rmax_pflops"], errors="coerce")
    df["iso3"] = df["country"].apply(to_iso3)
    df = df[df["iso3"].notna()].copy()

    agg = (
        df.groupby("iso3")
        .agg(
            _n_systems  = ("iso3",        "count"),
            _total_rmax = ("rmax_pflops", "sum"),
            _max_rmax   = ("rmax_pflops", "max"),
        )
        .reset_index()
    )
    agg["year"] = year
    return agg


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_top500(
    start_year: int = 2010,
    end_year:   int = 2024,
    raw_dir:    str | Path = "data/raw",
) -> pd.DataFrame:
    """
    Return Top500 supercomputer metrics by country-year.

    Parameters
    ----------
    start_year : First year (inclusive).
    end_year   : Last year (inclusive).
    raw_dir    : Directory for raw cache files.

    Returns
    -------
    Wide DataFrame: iso3, year, top500_n_systems, top500_total_rmax, top500_max_rmax
    """
    raw_dir   = Path(raw_dir)
    cache_dir = raw_dir / "top500_xml"
    frames: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        year_frames: list[pd.DataFrame] = []

        for month in (6, 11):
            logger.info("  Fetching Top500 %d/%02d…", year, month)
            rows = _load_xml(year, month, cache_dir)
            if rows:
                df_list = _aggregate(rows, year)
                if not df_list.empty:
                    year_frames.append(df_list)
                    logger.info(
                        "    %d/%02d: %d systems, %d countries",
                        year, month, len(rows), len(df_list),
                    )

        if year_frames:
            combined = pd.concat(year_frames, ignore_index=True)
            # Collapse biannual → annual by taking maximum per country
            annual = (
                combined.groupby(["iso3", "year"])
                .agg(
                    top500_n_systems  = ("_n_systems",  "max"),
                    top500_total_rmax = ("_total_rmax", "max"),
                    top500_max_rmax   = ("_max_rmax",   "max"),
                )
                .reset_index()
            )
            frames.append(annual)

    if not frames:
        logger.warning("Top500: no data retrieved for %d–%d.", start_year, end_year)
        return pd.DataFrame()

    df_out = pd.concat(frames, ignore_index=True)
    df_out = df_out.sort_values(["iso3", "year"]).reset_index(drop=True)
    save_raw(df_out, "top500_raw", raw_dir)

    logger.info(
        "Top500: %d country-year rows, %d countries",
        len(df_out), df_out["iso3"].nunique(),
    )
    return df_out


def fetch_all_top500(
    config: dict,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Entry point called by the main pipeline."""
    t5_cfg = config.get("top500", {})
    if not t5_cfg.get("enabled", True):
        logger.info("Top500 disabled in config.")
        return pd.DataFrame()

    start = config["pipeline"]["start_year"]
    end   = config["pipeline"]["end_year"]

    return fetch_top500(start_year=start, end_year=end, raw_dir=raw_dir)
