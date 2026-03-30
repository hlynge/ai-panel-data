"""
imf.py — pull IMF World Economic Outlook data via the IMF DataMapper API.

The IMF DataMapper REST API provides free, programmatic access to WEO data.

Endpoint:
  https://www.imf.org/external/datamapper/api/v1/{indicator}

Documentation:
  https://www.imf.org/external/datamapper/api/help
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import fetch_url, get_logger, save_raw

logger = get_logger(__name__)

IMF_API_BASE = "https://www.imf.org/external/datamapper/api/v1"


def fetch_imf_indicator(
    indicator: str,
    friendly_name: str,
    start_year: int = 2010,
    end_year: int = 2024,
) -> pd.DataFrame:
    """
    Fetch a single WEO indicator for all countries and return a tidy DataFrame.

    Parameters
    ----------
    indicator     : IMF DataMapper indicator code, e.g. "NGDPDPC".
    friendly_name : Column name in the output DataFrame.
    start_year    : First year to include.
    end_year      : Last year to include.

    Returns
    -------
    DataFrame with columns: iso3, year, <friendly_name>
    """
    url = f"{IMF_API_BASE}/{indicator}"
    # IMF's Akamai CDN blocks most custom User-Agents; mimic curl
    resp = fetch_url(url, timeout=60, headers={"User-Agent": "curl/8.0"})
    data = resp.json()

    # Response structure: {"values": {indicator: {iso3: {year: value}}}}
    values_block = data.get("values", {}).get(indicator, {})
    if not values_block:
        logger.warning("No values returned for IMF indicator %s", indicator)
        return pd.DataFrame()

    rows = []
    for iso3, year_dict in values_block.items():
        for year_str, val in year_dict.items():
            try:
                yr = int(year_str)
            except ValueError:
                continue
            if start_year <= yr <= end_year:
                rows.append({"iso3": iso3, "year": yr, friendly_name: val})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df[friendly_name] = pd.to_numeric(df[friendly_name], errors="coerce")
    df["year"] = df["year"].astype(int)
    return df.sort_values(["iso3", "year"]).reset_index(drop=True)


def fetch_all_imf(
    config: dict,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """
    Fetch all configured IMF indicators and merge them into a wide DataFrame.

    Called by the main pipeline.
    """
    imf_cfg = config.get("imf", {})
    if not imf_cfg.get("enabled", True):
        logger.info("IMF disabled in config.")
        return pd.DataFrame()

    indicators: dict[str, str] = imf_cfg.get("indicators", {})
    start = config["pipeline"]["start_year"]
    end = config["pipeline"]["end_year"]

    logger.info("Fetching %d IMF indicators (%d–%d)…", len(indicators), start, end)

    frames = []
    for code, name in indicators.items():
        try:
            df = fetch_imf_indicator(code, name, start_year=start, end_year=end)
            if not df.empty:
                frames.append(df)
                logger.info("  ✓ %s (%s)", name, code)
        except Exception as exc:
            logger.warning("  ✗ Could not fetch %s (%s): %s", name, code, exc)

    if not frames:
        logger.error("No IMF data retrieved.")
        return pd.DataFrame()

    df_merged = frames[0]
    for df_next in frames[1:]:
        df_merged = df_merged.merge(df_next, on=["iso3", "year"], how="outer")

    df_merged = df_merged.sort_values(["iso3", "year"]).reset_index(drop=True)
    save_raw(df_merged, "imf_raw", raw_dir)
    logger.info(
        "IMF: %d country-year rows, %d indicators",
        len(df_merged),
        len(df_merged.columns) - 2,
    )
    return df_merged
