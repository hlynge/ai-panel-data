"""
oecd_ai.py — fetch AI-relevant indicators from OECD data sources.

Data coverage
─────────────
The OECD.AI Observatory (oecd.ai/en/data) visualises data from multiple
partner organisations.  Not all of it is available via a public API.  This
module pulls what IS programmatically accessible:

  1. R&D statistics (MSTI_PUB)
       – Total / business / government / HE R&D as % of GDP
       – Researchers per 1,000 employed
  2. AI-related patents (PATS_IPC)
       – Patent applications in IPC classes G06N (ML/AI), G06F, G06K,
         G06T, G06V, G10L  (see config.yaml for the full list)
  3. ICT access & usage (ICT_HH2)
       – Internet users, broadband & mobile subscriptions, etc.
  4. Business R&D by industry (BERD_NACE2)
       – R&D spend in ICT-intensive sectors

All four datasets are fetched via the *old* OECD SDMX-JSON API which is
stable and well-documented.

For data that cannot yet be pulled automatically (e.g. private AI investment
from NetBase Quid/Stanford HAI, AI talent from LinkedIn, AI compute from
Epoch AI), the function `download_stanfordhai_index()` attempts to retrieve
the publicly released Stanford HAI AI Index spreadsheet, and clear error
messages point you to manual download pages for the rest.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .oecd_api import fetch_dataset_old, normalise_old_api
from .utils import get_logger, save_raw

logger = get_logger(__name__)

# ── 1.  R&D indicators (MSTI_PUB) ────────────────────────────────────────────

MSTI_INDICATORS = {
    "GERD_GDPB":   "rd_total_pct_gdp",
    "BERD_GDPB":   "rd_business_pct_gdp",
    "GOVERD_GDPB": "rd_government_pct_gdp",
    "HERD_GDPB":   "rd_highered_pct_gdp",
    "RESS_EMPB":   "researchers_per1000_employed",
    "GERD_PERPOP": "rd_per_capita_usd_ppp",
    "GERD_USD":    "rd_total_usd_ppp_mn",
    "BERD_USD":    "rd_business_usd_ppp_mn",
}


def fetch_msti(
    indicators: Optional[list[str]] = None,
    start_year: int = 2010,
    end_year: int = 2024,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """
    Return MSTI indicators as a tidy country-year DataFrame.

    Columns: iso2_code, year, indicator_name, value
    (iso2 is converted to iso3 in the harmonise step of the pipeline)
    """
    wanted = indicators or list(MSTI_INDICATORS.keys())
    logger.info("Fetching MSTI R&D indicators: %s", wanted)

    df_raw = fetch_dataset_old(
        dataset_id="MSTI_PUB",
        filter_expr="all",
        start_year=start_year,
        end_year=end_year,
    )
    save_raw(df_raw, "oecd_msti_raw", raw_dir)

    df = normalise_old_api(df_raw)

    # Filter to requested indicators
    if "VAR" in df.columns:
        df = df[df["VAR"].isin(wanted)].copy()
    elif "Indicator" in df.columns:
        df = df[df["Indicator"].isin(wanted)].copy()

    # Pivot to wide: one column per indicator per country-year
    id_cols = [c for c in ("country_code", "year") if c in df.columns]
    ind_col = next((c for c in ("VAR", "Indicator", "MEASURE") if c in df.columns), None)

    if ind_col:
        df = df[id_cols + [ind_col, "value"]].copy()
        df[ind_col] = df[ind_col].map(lambda x: MSTI_INDICATORS.get(x, x.lower()))
        df = df.pivot_table(index=id_cols, columns=ind_col, values="value", aggfunc="mean").reset_index()
        df.columns.name = None

    logger.info("MSTI: %d country-year rows", len(df))
    return df


# ── 2.  AI / ICT Patents (PATS_IPC) ──────────────────────────────────────────

IPC_AI_CLASSES = ["G06N", "G06F", "G06K", "G06T", "G06V", "G10L"]

# Sub-classes that most precisely capture core AI/ML (narrower cut)
IPC_CORE_AI = ["G06N"]


def fetch_ai_patents(
    ipc_classes: Optional[list[str]] = None,
    start_year: int = 2010,
    end_year: int = 2024,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """
    Return patent application counts for AI-related IPC classes, by country and year.

    The PATS_IPC dataset counts PCT international patent applications filed
    under the Patent Cooperation Treaty — a standard cross-country measure.

    Columns: country_code, year, ipc_class, patent_applications
    """
    classes = ipc_classes or IPC_AI_CLASSES
    logger.info("Fetching AI patents for IPC classes: %s", classes)

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
            logger.warning("Could not fetch patents for %s: %s", ipc, exc)

    if not all_frames:
        logger.error("No patent data retrieved.")
        return pd.DataFrame()

    df_all = pd.concat(all_frames, ignore_index=True)

    # Keep only patent application count column
    val_col = "value"
    id_cols = [c for c in ("country_code", "year", "ipc_class") if c in df_all.columns]
    df_all = df_all[id_cols + [val_col]].rename(columns={val_col: "patent_applications"})

    # Aggregate all AI classes into a single "ai_patents_total" column
    df_total = (
        df_all.groupby(["country_code", "year"])["patent_applications"]
        .sum()
        .reset_index()
        .rename(columns={"patent_applications": "ai_patents_total"})
    )

    # Also keep G06N (core AI/ML) separately
    df_g06n = (
        df_all[df_all["ipc_class"] == "G06N"]
        .rename(columns={"patent_applications": "ai_patents_g06n_ml"})
        .drop(columns="ipc_class", errors="ignore")
    )

    df_out = df_total.merge(df_g06n[["country_code", "year", "ai_patents_g06n_ml"]], on=["country_code", "year"], how="left")

    save_raw(df_all, "oecd_patents_raw", raw_dir)
    logger.info("Patents: %d country-year rows", len(df_out))
    return df_out


# ── 3.  ICT access & usage (ICT_HH2) ─────────────────────────────────────────

ICT_INDICATORS = {
    "HH_IACC":      "ict_hh_internet_access_pct",
    "HH_IRUS":      "ict_hh_internet_use_pct",
    "IND_IRUS":     "ict_ind_internet_use_pct",
    "HH_BB":        "ict_hh_broadband_access_pct",
    "HH_SBBD":      "ict_hh_singleplay_broadband_pct",
    "HH_PCMP":      "ict_hh_computer_access_pct",
    "IND_SHOP":     "ict_ind_online_shopping_pct",
    "IND_EBANK":    "ict_ind_online_banking_pct",
}


def fetch_ict(
    start_year: int = 2010,
    end_year: int = 2024,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """
    Return ICT household / individual usage indicators as a wide country-year DataFrame.
    """
    logger.info("Fetching ICT access & usage (ICT_HH2)…")
    try:
        df_raw = fetch_dataset_old(
            dataset_id="ICT_HH2",
            filter_expr="all",
            start_year=start_year,
            end_year=end_year,
        )
    except Exception as exc:
        logger.error("ICT fetch failed: %s", exc)
        return pd.DataFrame()

    save_raw(df_raw, "oecd_ict_raw", raw_dir)
    df = normalise_old_api(df_raw)

    # Identify the indicator dimension (varies by export)
    ind_col = next((c for c in ("IND", "Indicator", "MEASURE", "VAR") if c in df.columns), None)
    if ind_col is None:
        logger.warning("Cannot identify indicator column in ICT dataset; returning raw.")
        return df

    df = df[df[ind_col].isin(ICT_INDICATORS.keys())].copy()
    df[ind_col] = df[ind_col].map(lambda x: ICT_INDICATORS.get(x, x.lower()))

    id_cols = [c for c in ("country_code", "year") if c in df.columns]
    df = df.pivot_table(index=id_cols, columns=ind_col, values="value", aggfunc="mean").reset_index()
    df.columns.name = None
    logger.info("ICT: %d country-year rows", len(df))
    return df


# ── 4.  Business R&D by industry (BERD_NACE2) ────────────────────────────────

# NACE Rev.2 divisions most relevant to AI / high-tech
NACE_HI_TECH = [
    "J",    # Information and communication
    "J58",  # Publishing
    "J59",  # Motion picture, video, TV
    "J61",  # Telecommunications
    "J62",  # Computer programming, consultancy
    "J63",  # Information service activities
    "M72",  # Scientific research and development
    "C26",  # Computer, electronic and optical products
]


def fetch_berd(
    start_year: int = 2010,
    end_year: int = 2024,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """
    Return business R&D expenditure in ICT-intensive industries, by country-year.
    """
    logger.info("Fetching BERD by industry (BERD_NACE2)…")
    try:
        df_raw = fetch_dataset_old(
            dataset_id="BERD_NACE2",
            filter_expr="all",
            start_year=start_year,
            end_year=end_year,
        )
    except Exception as exc:
        logger.error("BERD fetch failed: %s", exc)
        return pd.DataFrame()

    save_raw(df_raw, "oecd_berd_raw", raw_dir)
    df = normalise_old_api(df_raw)

    # Keep total ICT sector R&D in USD PPP millions
    ind_col = next((c for c in ("IND", "Industry", "NACE", "ISIC") if c in df.columns), None)
    meas_col = next((c for c in ("MEASURE", "Measure", "MEAS") if c in df.columns), None)

    # Aggregate over all hi-tech industries
    if ind_col:
        df = df[df[ind_col].isin(NACE_HI_TECH)].copy()

    # Keep only current USD PPP if available
    if meas_col:
        usd_vals = df[meas_col].str.upper().str.contains("USD|MIO|PPP", na=False)
        if usd_vals.any():
            df = df[usd_vals].copy()

    id_cols = [c for c in ("country_code", "year") if c in df.columns]
    df_agg = (
        df.groupby(id_cols)["value"].sum().reset_index()
        .rename(columns={"value": "berd_ict_usd_ppp_mn"})
    )
    logger.info("BERD: %d country-year rows", len(df_agg))
    return df_agg


# ── 5.  Stanford HAI AI Index (best-effort download) ─────────────────────────

STANFORD_HAI_URL = (
    "https://aiindex.stanford.edu/wp-content/uploads/"
    "2024/04/HAI_2024_AI-Index-Report.xlsx"
)

INVESTMENT_SHEET = "Figure 4.2.1"   # Sheet name may vary across annual editions


def download_stanfordhai_index(
    dest_dir: str | Path = "data/raw",
    url: str = STANFORD_HAI_URL,
) -> Optional[pd.DataFrame]:
    """
    Attempt to download the Stanford HAI AI Index annual report Excel file
    and extract the country-level private AI investment table.

    This is a *best-effort* function — the URL and sheet names change each year.
    If the download fails, a clear message guides you to the manual download page.

    Returns a DataFrame or None if the download fails.
    """
    logger.info("Attempting Stanford HAI AI Index download…")
    dest = Path(dest_dir) / "stanford_hai_ai_index.xlsx"

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
            "→ Please download manually from https://aiindex.stanford.edu/report/ "
            "and place the Excel file at %s",
            exc,
            dest,
        )
        if not dest.exists():
            return None

    # Parse investment data from the Excel
    try:
        xls = pd.ExcelFile(dest)
        # Look for investment-related sheet
        investment_sheet = next(
            (s for s in xls.sheet_names if "4.2" in s or "invest" in s.lower()), None
        )
        if investment_sheet is None:
            logger.warning("Could not find investment sheet in HAI Excel. Sheets: %s", xls.sheet_names)
            return None

        df = xls.parse(investment_sheet, header=1)
        df.columns = [str(c).strip() for c in df.columns]
        logger.info("HAI investment data: %d rows from sheet '%s'", len(df), investment_sheet)
        return df

    except Exception as exc:
        logger.warning("Could not parse HAI Excel: %s", exc)
        return None


# ── Public entry point ────────────────────────────────────────────────────────

def fetch_all_oecd_ai(
    config: dict,
    raw_dir: str | Path = "data/raw",
) -> dict[str, pd.DataFrame]:
    """
    Fetch all OECD AI-relevant datasets according to *config* and return
    a dict mapping dataset name → DataFrame.

    Called by the main pipeline (pipeline.py).
    """
    oecd_cfg = config.get("oecd", {})
    start = config["pipeline"]["start_year"]
    end = config["pipeline"]["end_year"]

    results: dict[str, pd.DataFrame] = {}

    if oecd_cfg.get("msti", {}).get("enabled", True):
        inds = oecd_cfg.get("msti", {}).get("indicators") or None
        results["msti"] = fetch_msti(indicators=inds, start_year=start, end_year=end, raw_dir=raw_dir)

    if oecd_cfg.get("patents", {}).get("enabled", True):
        ipc = oecd_cfg.get("patents", {}).get("ipc_classes") or None
        results["patents"] = fetch_ai_patents(ipc_classes=ipc, start_year=start, end_year=end, raw_dir=raw_dir)

    if oecd_cfg.get("ict", {}).get("enabled", True):
        results["ict"] = fetch_ict(start_year=start, end_year=end, raw_dir=raw_dir)

    if oecd_cfg.get("berd", {}).get("enabled", True):
        results["berd"] = fetch_berd(start_year=start, end_year=end, raw_dir=raw_dir)

    # Best-effort Stanford HAI
    hai_df = download_stanfordhai_index(dest_dir=raw_dir)
    if hai_df is not None:
        results["stanford_hai_investment"] = hai_df

    return results
