"""
oecd_ai.py — fetch AI-relevant indicators from OECD data sources.

All datasets now use the new OECD SDMX 2.1 API (sdmx.oecd.org) which
supports server-side year filtering, keeping responses fast and small.

Datasets covered
────────────────
  1. MSTI  — R&D expenditure & researcher counts (DSD_MSTI@DF_MSTI)
  2. Patents — AI/ICT patent applications by IPC class (DSD_PATS@DF_PATS_REGION_ST)
  3. ICT   — household internet/broadband access (DSD_ICT@DF_ICT_HH)
  4. BERD  — business R&D by industry (DSD_BERD@DF_BERD)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import requests

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


# ── 2.  AI / ICT Patents ──────────────────────────────────────────────────────

# IPC classes most relevant to AI/ML
IPC_AI_CLASSES = ["G06N", "G06F", "G06K", "G06T", "G06V", "G10L"]


def fetch_ai_patents(
    ipc_classes: Optional[list[str]] = None,
    start_year: int = 2010,
    end_year: int = 2024,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """
    Return PCT patent application counts for AI-related IPC classes, by country-year.
    """
    classes = ipc_classes or IPC_AI_CLASSES
    logger.info("Fetching AI patents for IPC classes: %s", classes)

    all_frames = []
    for ipc in classes:
        try:
            df_raw = fetch_dataset_new(
                dataflow="pats",
                key=f"all.{ipc}",
                start_period=start_year,
                end_period=end_year,
            )
            df = normalise_new_api(df_raw)
            df["ipc_class"] = ipc
            all_frames.append(df)
            logger.info("  ✓ Patents %s: %d rows", ipc, len(df))
        except Exception as exc:
            logger.warning("  ✗ Could not fetch patents for %s: %s", ipc, exc)

    if not all_frames:
        logger.warning("No patent data retrieved — trying old API fallback…")
        return _fetch_patents_old_api(classes, start_year, end_year, raw_dir)

    df_all = pd.concat(all_frames, ignore_index=True)
    save_raw(df_all, "oecd_patents_raw", raw_dir)

    id_cols = [c for c in ("country_code", "year") if c in df_all.columns]

    # Total AI patents across all classes
    df_total = (
        df_all.groupby(id_cols)["value"].sum().reset_index()
        .rename(columns={"value": "ai_patents_total"})
    )

    # Core AI/ML only (G06N)
    df_g06n = (
        df_all[df_all["ipc_class"] == "G06N"]
        .groupby(id_cols)["value"].sum().reset_index()
        .rename(columns={"value": "ai_patents_g06n_ml"})
    )

    df_out = df_total.merge(df_g06n, on=id_cols, how="left")
    logger.info("Patents: %d country-year rows", len(df_out))
    return df_out


def _fetch_patents_old_api(classes, start_year, end_year, raw_dir):
    """Fallback: fetch patents from the old OECD API."""
    from .oecd_api import fetch_dataset_old, normalise_old_api
    all_frames = []
    for ipc in classes:
        try:
            df_raw = fetch_dataset_old(
                dataset_id="PATS_IPC",
                filter_expr=f"{ipc}.PCT_APPLICATIONS",
                start_year=start_year,
                end_year=end_year,
            )
            df = normalise_old_api(df_raw)
            df["ipc_class"] = ipc
            all_frames.append(df)
        except Exception as exc:
            logger.warning("Old API patent fetch failed for %s: %s", ipc, exc)
    if not all_frames:
        return pd.DataFrame()
    df_all = pd.concat(all_frames, ignore_index=True)
    save_raw(df_all, "oecd_patents_raw", raw_dir)
    id_cols = [c for c in ("country_code", "year") if c in df_all.columns]
    return (
        df_all.groupby(id_cols)["value"].sum().reset_index()
        .rename(columns={"value": "ai_patents_total"})
    )


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


# ── 5.  Stanford HAI AI Index (best-effort) ───────────────────────────────────

STANFORD_HAI_URL = (
    "https://aiindex.stanford.edu/wp-content/uploads/"
    "2024/04/HAI_2024_AI-Index-Report.xlsx"
)


def download_stanfordhai_index(
    dest_dir: str | Path = "data/raw",
    url: str = STANFORD_HAI_URL,
) -> Optional[pd.DataFrame]:
    """Attempt to download the Stanford HAI AI Index Excel file."""
    dest = Path(dest_dir) / "stanford_hai_ai_index.xlsx"
    logger.info("Attempting Stanford HAI AI Index download…")
    try:
        resp = requests.get(url, timeout=60, stream=True,
                            headers={"User-Agent": "ai-panel-data/1.0"})
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)
        logger.info("Downloaded HAI report → %s", dest)
    except Exception as exc:
        logger.warning(
            "Stanford HAI download failed: %s\n"
            "→ Download manually from https://aiindex.stanford.edu/report/ "
            "and place at %s", exc, dest
        )
        return None

    try:
        xls = pd.ExcelFile(dest)
        sheet = next((s for s in xls.sheet_names if "invest" in s.lower() or "4.2" in s), None)
        if sheet is None:
            return None
        return xls.parse(sheet, header=1)
    except Exception as exc:
        logger.warning("Could not parse HAI Excel: %s", exc)
        return None


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
        # Note: ICT patents (P_ICTPCT), total PCT patents (P_PCT), and triadic
        # patent families (P_TRIAD) are already included in the MSTI dataset above,
        # so the separate IPC-class patent fetch is skipped.

    if oecd_cfg.get("ict", {}).get("enabled", True):
        results["ict"] = fetch_ict(start_year=start, end_year=end, raw_dir=raw_dir)

    if oecd_cfg.get("berd", {}).get("enabled", True):
        results["berd"] = fetch_berd(start_year=start, end_year=end, raw_dir=raw_dir)

    hai_df = download_stanfordhai_index(dest_dir=raw_dir)
    if hai_df is not None:
        results["stanford_hai_investment"] = hai_df

    return results
