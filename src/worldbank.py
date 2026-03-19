"""
worldbank.py — pull World Bank WDI data via the wbgapi package.

wbgapi is the official World Bank Python client (pip install wbgapi).
It provides a clean, pandas-native interface to the World Development
Indicators and the World Governance Indicators.

Documentation: https://pypi.org/project/wbgapi/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import get_logger, save_raw

logger = get_logger(__name__)


def fetch_worldbank(
    indicators: dict[str, str],
    start_year: int = 2010,
    end_year: int = 2024,
    raw_dir: str | Path = "data/raw",
    economy_filter: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Download World Bank WDI indicators and return a wide country-year DataFrame.

    Parameters
    ----------
    indicators     : Mapping of WB indicator code → friendly column name.
                     E.g. {"NY.GDP.PCAP.KD": "gdp_per_capita_const2015usd"}
    start_year     : First year.
    end_year       : Last year.
    raw_dir        : Directory for caching raw downloads.
    economy_filter : Optional list of ISO3 codes to restrict the pull.
                     None (default) fetches all economies.

    Returns
    -------
    Wide DataFrame with columns: iso3, year, <indicator_names…>
    """
    try:
        import wbgapi as wb
    except ImportError as exc:
        raise ImportError(
            "wbgapi is required for World Bank data. "
            "Install it with:  pip install wbgapi"
        ) from exc

    wb_codes = list(indicators.keys())
    friendly = indicators            # code → friendly name
    time_range = range(start_year, end_year + 1)
    economy = economy_filter or "all"

    logger.info(
        "Fetching %d World Bank indicators (%d–%d)…", len(wb_codes), start_year, end_year
    )

    frames = []
    for code in wb_codes:
        friendly_name = friendly[code]
        try:
            df = wb.data.DataFrame(
                code,
                time=time_range,
                economy=economy,
                labels=False,
                numericTimeKeys=True,
            )
            # wbgapi returns a DataFrame: index = economy (ISO3), columns = years
            df = df.stack().reset_index()
            df.columns = ["iso3", "year", friendly_name]
            df["year"] = df["year"].astype(int)
            frames.append(df)
            logger.info("  ✓ %s (%s)", friendly_name, code)
        except Exception as exc:
            logger.warning("  ✗ Could not fetch %s (%s): %s", friendly_name, code, exc)

    if not frames:
        logger.error("No World Bank data retrieved.")
        return pd.DataFrame()

    # Merge all indicators on iso3 + year
    df_merged = frames[0]
    for df_next in frames[1:]:
        df_merged = df_merged.merge(df_next, on=["iso3", "year"], how="outer")

    # Sort and clean
    df_merged = df_merged.sort_values(["iso3", "year"]).reset_index(drop=True)

    # Convert numeric columns
    num_cols = [c for c in df_merged.columns if c not in ("iso3", "year")]
    df_merged[num_cols] = df_merged[num_cols].apply(pd.to_numeric, errors="coerce")

    save_raw(df_merged, "worldbank_raw", raw_dir)
    logger.info("World Bank: %d country-year rows, %d indicators", len(df_merged), len(num_cols))
    return df_merged


def fetch_all_worldbank(
    config: dict,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Entry point called by the main pipeline."""
    wb_cfg = config.get("worldbank", {})
    if not wb_cfg.get("enabled", True):
        logger.info("World Bank disabled in config.")
        return pd.DataFrame()

    indicators = wb_cfg.get("indicators", {})
    start = config["pipeline"]["start_year"]
    end = config["pipeline"]["end_year"]
    country_filter = config.get("countries", {}).get("filter") or None

    return fetch_worldbank(
        indicators=indicators,
        start_year=start,
        end_year=end,
        raw_dir=raw_dir,
        economy_filter=country_filter if country_filter else None,
    )
