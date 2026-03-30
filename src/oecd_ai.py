"""
oecd_ai.py — fetch AI-relevant indicators from OECD data sources.

All datasets now use the new OECD SDMX 2.1 API (sdmx.oecd.org) which
supports server-side year filtering, keeping responses fast and small.

Datasets covered
────────────────
  1. MSTI       — R&D expenditure & researcher counts (DSD_MSTI@DF_MSTI)
  2. AI Patents — OECD AI-specific patents (DSD_PATENTS@DF_PATENTS_OECDSPECIFIC)
                  Uses OECD's official AI patent definition (CPC + text mining,
                  see OECD AI Papers No. 30, 2024). Two series:
                    • Triadic AI patent families (IP5 offices)
                    • AI patent applications via WIPO PCT
  3. ICT        — household internet/broadband access (DSD_ICT_HH_IND@DF_HH)
  4. BERD       — business R&D by industry (DSD_RDS_BERD@DF_BERD_INDU)

Note: Stanford HAI AI Index download was removed — the direct download URL
returns HTML (login wall) rather than the Excel file, making it unreliable
as a programmatic source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional  # kept for type hints in fetch_* signatures

import pandas as pd

from .oecd_api import fetch_dataset_new, normalise_new_api, DATAFLOWS
from .utils import get_logger, save_raw

logger = get_logger(__name__)


# ── 1.  R&D indicators (MSTI) ─────────────────────────────────────────────────

# MSTI measure codes in the new API.
# Each entry is (MEASURE code, UNIT_MEASURE code) → friendly column name.
# We filter on BOTH to get exactly the series we want.
MSTI_MEASURES = {
    ("G",    "PT_B1GQ"):    "rd_total_pct_gdp",
    ("B",    "PT_B1GQ"):    "rd_business_pct_gdp",
    ("GV",   "PT_B1GQ"):    "rd_government_pct_gdp",
    ("H",    "PT_B1GQ"):    "rd_highered_pct_gdp",
    ("T_RS", "10P3EMP"):    "researchers_per1000_employed",
    ("G",    "USD_PPP_PS"): "rd_per_capita_usd_ppp",
    ("P_ICTPCT", "PATN"):   "ict_patents_pct",
    ("P_PCT",    "PATN"):   "total_pct_patents",
    ("P_TRIAD",  "PATN"):   "triadic_patent_families",
}

# Kept for backward compatibility with config.yaml
MSTI_INDICATORS = {k[0]: v for k, v in MSTI_MEASURES.items()}


def fetch_msti(
    indicators: Optional[list[str]] = None,
    start_year: int = 2010,
    end_year: int = 2024,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """
    Return MSTI R&D indicators as a wide country-year DataFrame.
    """
    wanted = indicators or list(MSTI_MEASURES.keys())
    logger.info("Fetching MSTI R&D indicators: %s", wanted)

    try:
        df_raw = fetch_dataset_new(
            dataflow="msti",
            key="all",
            start_period=start_year,
            end_period=end_year,
        )
    except Exception as exc:
        logger.warning("MSTI new API failed (%s); falling back to old API…", exc)
        try:
            from .oecd_api import fetch_dataset_old, normalise_old_api
            df_raw = fetch_dataset_old("MSTI_PUB", start_year=start_year, end_year=end_year)
            df = normalise_old_api(df_raw)
            save_raw(df, "oecd_msti_raw", raw_dir)
            # Old API column for indicator is "VAR"
            if "VAR" in df.columns:
                df = df[df["VAR"].isin(wanted)].copy()
                df["VAR"] = df["VAR"].map(lambda x: MSTI_MEASURES.get(x, x.lower()))
                id_cols = [c for c in ("country_code", "year") if c in df.columns]
                df = (
                    df[id_cols + ["VAR", "value"]]
                    .pivot_table(index=id_cols, columns="VAR", values="value", aggfunc="mean")
                    .reset_index()
                )
                df.columns.name = None
            logger.info("MSTI (old API fallback): %d rows", len(df))
            return df
        except Exception as exc2:
            logger.error("MSTI old API fallback also failed: %s", exc2)
            return pd.DataFrame()

    save_raw(df_raw, "oecd_msti_raw", raw_dir)
    df = normalise_new_api(df_raw)

    # The new API needs BOTH MEASURE and UNIT_MEASURE to identify each series.
    # Build a lookup: (MEASURE, UNIT_MEASURE) → friendly name
    lookup = MSTI_MEASURES  # dict of (measure, unit) → name

    if "MEASURE" not in df.columns or "UNIT_MEASURE" not in df.columns:
        logger.warning("Expected MEASURE/UNIT_MEASURE columns not found. Got: %s", df.columns.tolist())
        return df

    # Create a combined key column for matching
    df["_key"] = list(zip(df["MEASURE"], df["UNIT_MEASURE"]))
    df = df[df["_key"].isin(lookup.keys())].copy()
    df["_indicator"] = df["_key"].map(lookup)

    id_cols = [c for c in ("country_code", "year") if c in df.columns]
    df = (
        df[id_cols + ["_indicator", "value"]]
        .pivot_table(index=id_cols, columns="_indicator", values="value", aggfunc="mean")
        .reset_index()
    )
    df.columns.name = None

    logger.info("MSTI: %d country-year rows, %d indicators", len(df), len(df.columns) - 2)
    return df


# ── 2.  AI Patents (OECD official definition) ─────────────────────────────────

# Key structure for DSD_PATENTS@DF_PATENTS_OECDSPECIFIC has 11 key dimensions:
#   PATENT_AUTHORITIES . FREQ . MEASURE . UNIT_MEASURE . DATE_TYPE .
#   REF_AREA . PARTNER_AREA . AGENT_ROLE . COOPERATION_TYPE . WIPO .
#   OECD_TECHNOLOGY_PATENT
#
# Fetch all authorities/measures/countries for AI technology using dots for
# unspecified dimensions, then filter in Python.
#
# Selected series after filtering:
#   PATENT_AUTHORITIES=9P50_2, MEASURE=PF → AI triadic patent families (IP5)
#   PATENT_AUTHORITIES=9P50_1, MEASURE=AP → AI patent applications (WIPO PCT)
#
# Note: patent data typically lags 2–3 years; expect coverage to ~2021.

_AI_PATENTS_KEY = "..........AI"   # all dims open, OECD_TECHNOLOGY_PATENT=AI


def fetch_ai_patents(
    start_year: int = 2010,
    end_year: int = 2024,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """
    Return OECD AI-specific patent counts by country-year.

    Uses OECD's official AI patent definition (CPC codes + text mining,
    OECD AI Papers No. 30, 2024) via DSD_PATENTS@DF_PATENTS_OECDSPECIFIC.
    Returns two series:
      - ai_patent_families_triadic   : IP5 triadic AI patent families
      - ai_patent_applications_wipo  : WIPO PCT AI patent applications
    """
    logger.info("Fetching OECD AI patents (official OECD AI definition)…")
    try:
        df_raw = fetch_dataset_new(
            dataflow="ai_patents",
            key=_AI_PATENTS_KEY,
            start_period=start_year,
            end_period=end_year,
        )
    except Exception as exc:
        logger.error("AI patent fetch failed: %s", exc)
        return pd.DataFrame()

    df = normalise_new_api(df_raw)
    save_raw(df_raw, "oecd_ai_patents_raw", raw_dir)

    id_cols = [c for c in ("country_code", "year") if c in df.columns]
    frames = []

    # Triadic patent families: IP5 offices, patent families measure
    mask_triadic = (df["PATENT_AUTHORITIES"] == "9P50_2") & (df["MEASURE"] == "PF")
    df_tri = (
        df[mask_triadic].groupby(id_cols)["value"].sum().reset_index()
        .rename(columns={"value": "ai_patent_families_triadic"})
    )
    frames.append(df_tri)
    logger.info("  ✓ AI triadic patent families: %d country-year rows", len(df_tri))

    # WIPO PCT patent applications
    mask_wipo = (df["PATENT_AUTHORITIES"] == "9P50_1") & (df["MEASURE"] == "AP")
    df_wipo = (
        df[mask_wipo].groupby(id_cols)["value"].sum().reset_index()
        .rename(columns={"value": "ai_patent_applications_wipo"})
    )
    frames.append(df_wipo)
    logger.info("  ✓ AI WIPO patent applications: %d country-year rows", len(df_wipo))

    df_out = frames[0]
    for df_next in frames[1:]:
        df_out = df_out.merge(df_next, on=id_cols, how="outer")

    logger.info("AI Patents: %d country-year rows, %d series",
                len(df_out), len(df_out.columns) - 2)
    return df_out


# ── 3.  ICT access & usage ────────────────────────────────────────────────────

# New OECD ICT dataflow (DSD_ICT_HH_IND@DF_HH) measure codes
ICT_MEASURES = {
    "B1_HH":   "ict_hh_internet_access_pct",
    "B21_HH":  "ict_hh_broadband_access_pct",
    "B21A_HH": "ict_hh_fixed_broadband_pct",
    "B21B_HH": "ict_hh_mobile_broadband_pct",
    "A1_HH":   "ict_hh_computer_access_pct",
}


def fetch_ict(
    start_year: int = 2010,
    end_year: int = 2024,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Return ICT household/individual usage indicators as a wide country-year DataFrame."""
    logger.info("Fetching ICT access & usage…")
    try:
        df_raw = fetch_dataset_new(
            dataflow="ict",
            key="all",
            start_period=start_year,
            end_period=end_year,
        )
    except Exception as exc:
        logger.warning("ICT new API failed (%s); trying old API…", exc)
        try:
            from .oecd_api import fetch_dataset_old, normalise_old_api
            df_raw = fetch_dataset_old("ICT_HH2", start_year=start_year, end_year=end_year)
            df = normalise_old_api(df_raw)
        except Exception as exc2:
            logger.error("ICT fetch failed entirely: %s", exc2)
            return pd.DataFrame()
        save_raw(df, "oecd_ict_raw", raw_dir)
        return df

    save_raw(df_raw, "oecd_ict_raw", raw_dir)
    df = normalise_new_api(df_raw)

    measure_col = next(
        (c for c in df.columns if c == "MEASURE"), None
    )
    if measure_col is None:
        logger.warning("ICT: no MEASURE column found. Columns: %s", df.columns.tolist())
        return df

    # Filter to percentage rows only (not absolute counts)
    if "UNIT_MEASURE" in df.columns:
        df = df[df["UNIT_MEASURE"].str.contains("PT", na=False)].copy()

    df = df[df[measure_col].isin(ICT_MEASURES)].copy()
    df["_indicator"] = df[measure_col].map(ICT_MEASURES)

    id_cols = [c for c in ("country_code", "year") if c in df.columns]
    df = (
        df[id_cols + ["_indicator", "value"]]
        .pivot_table(index=id_cols, columns="_indicator", values="value", aggfunc="mean")
        .reset_index()
    )
    df.columns.name = None
    logger.info("ICT: %d country-year rows", len(df))
    return df


# ── 4.  Business R&D by industry (BERD) ──────────────────────────────────────

def fetch_berd(
    start_year: int = 2010,
    end_year: int = 2024,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Return business R&D expenditure aggregated over ICT-intensive industries."""
    logger.info("Fetching BERD by industry…")
    try:
        df_raw = fetch_dataset_new(
            dataflow="berd",
            key="all",
            start_period=start_year,
            end_period=end_year,
        )
    except Exception as exc:
        logger.warning("BERD new API failed (%s); trying old API…", exc)
        try:
            from .oecd_api import fetch_dataset_old, normalise_old_api
            df_raw = fetch_dataset_old("BERD_NACE2", start_year=start_year, end_year=end_year)
            df = normalise_old_api(df_raw)
        except Exception as exc2:
            logger.error("BERD fetch failed entirely: %s", exc2)
            return pd.DataFrame()
        save_raw(df, "oecd_berd_raw", raw_dir)
        return df

    save_raw(df_raw, "oecd_berd_raw", raw_dir)
    df = normalise_new_api(df_raw)

    id_cols = [c for c in ("country_code", "year") if c in df.columns]
    df_agg = (
        df.groupby(id_cols)["value"].sum().reset_index()
        .rename(columns={"value": "berd_ict_usd_ppp_mn"})
    )
    logger.info("BERD: %d country-year rows", len(df_agg))
    return df_agg


# ── Public entry point ────────────────────────────────────────────────────────

def fetch_all_oecd_ai(
    config: dict,
    raw_dir: str | Path = "data/raw",
) -> dict[str, pd.DataFrame]:
    """
    Fetch all OECD AI-relevant datasets per config and return a dict of DataFrames.
    Called by pipeline.py.
    """
    oecd_cfg = config.get("oecd", {})
    start = config["pipeline"]["start_year"]
    end   = config["pipeline"]["end_year"]
    results: dict[str, pd.DataFrame] = {}

    if oecd_cfg.get("msti", {}).get("enabled", True):
        inds = oecd_cfg.get("msti", {}).get("indicators") or None
        results["msti"] = fetch_msti(indicators=inds, start_year=start, end_year=end, raw_dir=raw_dir)

    if oecd_cfg.get("patents", {}).get("enabled", True):
        results["ai_patents"] = fetch_ai_patents(start_year=start, end_year=end, raw_dir=raw_dir)

    if oecd_cfg.get("ict", {}).get("enabled", True):
        results["ict"] = fetch_ict(start_year=start, end_year=end, raw_dir=raw_dir)

    if oecd_cfg.get("berd", {}).get("enabled", True):
        results["berd"] = fetch_berd(start_year=start, end_year=end, raw_dir=raw_dir)

    return results
