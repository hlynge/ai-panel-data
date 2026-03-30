"""
pipeline.py — assemble a balanced country-year panel from all data sources.

Workflow
────────
1.  Fetch each source (OECD, World Bank, IMF, V-Dem, OpenAlex, Top500, Epoch AI).
2.  Standardise country identifiers to ISO3.
3.  Outer-merge all sources on (iso3, year).
4.  Apply the optional country filter from config.
5.  Add derived variables (patent intensity, R&D shares, …).
6.  Save as CSV / Parquet / Excel to data/processed/.

The resulting panel has one row per country-year with every available
indicator as a column.  Missing values are NaN (not dropped), so that
users can apply their own missingness strategies in downstream analysis.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pycountry

from .epochai import fetch_all_epochai
from .harmonize import apply_country_filter, standardise_iso3
from .imf import fetch_all_imf
from .oecd_ai import fetch_all_oecd_ai
from .openalex import fetch_all_openalex
from .top500 import fetch_all_top500
from .utils import ensure_dir, get_logger, load_config
from .vdem import fetch_all_vdem
from .worldbank import fetch_all_worldbank

logger = get_logger(__name__)


# ── Country validation ────────────────────────────────────────────────────────

# Build a set of valid ISO 3166-1 alpha-3 codes once at import time
_VALID_ISO3: set[str] = {c.alpha_3 for c in pycountry.countries}


def _filter_to_countries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows whose iso3 code is not a valid ISO 3166-1 alpha-3 country code.
    This strips World Bank regional/income aggregates (WLD, MIC, SSA, etc.)
    and any other non-country codes that slip through source-level filtering.
    """
    before = len(df["iso3"].unique())
    mask = df["iso3"].isin(_VALID_ISO3)
    dropped = sorted(df.loc[~mask, "iso3"].unique())
    if dropped:
        logger.info("Dropping %d non-country iso3 codes: %s", len(dropped), dropped)
    df = df[mask].copy()
    after = len(df["iso3"].unique())
    logger.info("Country filter: %d → %d unique iso3 codes", before, after)
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _std(df: pd.DataFrame, iso3_col: str = "country_code") -> pd.DataFrame:
    """Standardise the country-identifier column and rename it to 'iso3'."""
    if iso3_col not in df.columns:
        # Maybe already called 'iso3'
        if "iso3" in df.columns:
            return df
        logger.warning("No recognised country column found; trying first column.")
        iso3_col = df.columns[0]

    df = standardise_iso3(df, col=iso3_col, drop_aggregates=True)
    if iso3_col != "iso3":
        df = df.rename(columns={iso3_col: "iso3"})
    return df


def _merge_all(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-merge a list of DataFrames on [iso3, year]."""
    valid = [f for f in frames if not f.empty and "iso3" in f.columns and "year" in f.columns]
    if not valid:
        raise RuntimeError("No data frames to merge.")

    merged = valid[0]
    for df in valid[1:]:
        # Identify overlapping non-key columns and suffix them to avoid confusion
        overlap = [c for c in df.columns if c not in ("iso3", "year") and c in merged.columns]
        if overlap:
            logger.warning("Duplicate columns across sources (will suffix): %s", overlap)
        merged = merged.merge(df, on=["iso3", "year"], how="outer", suffixes=("", "_dup"))

    # Drop any accidentally duplicated columns
    dup_cols = [c for c in merged.columns if c.endswith("_dup")]
    if dup_cols:
        merged.drop(columns=dup_cols, inplace=True)

    return merged.sort_values(["iso3", "year"]).reset_index(drop=True)


# ── Derived variables ─────────────────────────────────────────────────────────

def add_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a small set of analytically useful derived columns.

    All operations are safe: if a required column is missing, the derived
    column is simply not added.
    """
    df = df.copy()

    # AI patent intensity = AI triadic patent families per unit of total R&D spending
    if "ai_patent_families_triadic" in df.columns and "rd_total_pct_gdp" in df.columns:
        df["ai_patent_intensity"] = df["ai_patent_families_triadic"] / df["rd_total_pct_gdp"].replace(0, float("nan"))

    # Log GDP per capita
    for col in ("gdp_per_capita_const2015usd", "imf_gdp_per_capita_current_usd"):
        if col in df.columns:
            import numpy as np
            df[f"log_{col}"] = np.log(df[col].replace(0, float("nan")))
            break

    # Internet access gap: 100 - % internet users
    if "internet_users_pct_pop" in df.columns:
        df["internet_access_gap_pct"] = 100 - df["internet_users_pct_pop"]

    # OECD membership flag (iso3 codes of current members)
    OECD_MEMBERS = {
        "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE", "DNK",
        "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "ISL", "IRL", "ISR",
        "ITA", "JPN", "KOR", "LVA", "LTU", "LUX", "MEX", "NLD", "NZL",
        "NOR", "POL", "PRT", "SVK", "SVN", "ESP", "SWE", "CHE", "TUR",
        "GBR", "USA",
    }
    df["oecd_member"] = df["iso3"].isin(OECD_MEMBERS).astype(int)

    return df


# ── OECD source integration ───────────────────────────────────────────────────

def _integrate_oecd(oecd_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all OECD sub-datasets into a single wide country-year DataFrame.
    """
    frames = []
    for name, df in oecd_data.items():
        if df.empty:
            continue

        # Ensure country column is standardised
        country_col = next(
            (c for c in ("country_code", "COU", "LOCATION", "iso3") if c in df.columns), None
        )
        if country_col is None:
            logger.warning("OECD sub-dataset '%s' has no recognisable country column; skipping.", name)
            continue
        df = _std(df, iso3_col=country_col)
        frames.append(df)

    return _merge_all(frames) if frames else pd.DataFrame()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(config_path: str | Path = "config.yaml") -> pd.DataFrame:
    """
    Execute the full data pipeline and return the merged panel DataFrame.

    Steps:
      1. Load config
      2. Fetch all sources
      3. Standardise country codes
      4. Merge to panel
      5. Add derived variables
      6. Apply country filter
      7. Save outputs
    """
    t0 = time.time()
    config = load_config(config_path)

    start_yr = config["pipeline"]["start_year"]
    end_yr   = config["pipeline"]["end_year"]
    out_dir  = Path(config["pipeline"].get("output_dir", "data/processed"))
    raw_dir  = Path("data/raw")
    ensure_dir(out_dir)
    ensure_dir(raw_dir)

    logger.info("=" * 60)
    logger.info("AI Panel Data Pipeline")
    logger.info("  Period : %d – %d", start_yr, end_yr)
    logger.info("  Output : %s", out_dir)
    logger.info("=" * 60)

    # ── Fetch ─────────────────────────────────────────────────────────────────
    all_frames: list[pd.DataFrame] = []

    # OECD
    if config.get("oecd", {}).get("enabled", True):
        logger.info("─── OECD ───────────────────────────────────────")
        oecd_data = fetch_all_oecd_ai(config, raw_dir=raw_dir)
        df_oecd = _integrate_oecd(oecd_data)
        if not df_oecd.empty:
            all_frames.append(df_oecd)
            logger.info("OECD combined: %d rows, %d cols", *df_oecd.shape)

    # World Bank
    if config.get("worldbank", {}).get("enabled", True):
        logger.info("─── World Bank ─────────────────────────────────")
        df_wb = fetch_all_worldbank(config, raw_dir=raw_dir)
        if not df_wb.empty:
            all_frames.append(df_wb)
            logger.info("World Bank: %d rows, %d cols", *df_wb.shape)

    # IMF
    if config.get("imf", {}).get("enabled", True):
        logger.info("─── IMF ────────────────────────────────────────")
        df_imf = fetch_all_imf(config, raw_dir=raw_dir)
        if not df_imf.empty:
            all_frames.append(df_imf)
            logger.info("IMF: %d rows, %d cols", *df_imf.shape)

    # V-Dem
    if config.get("vdem", {}).get("enabled", True):
        logger.info("─── V-Dem ──────────────────────────────────────")
        df_vdem = fetch_all_vdem(config, raw_dir=raw_dir)
        if not df_vdem.empty:
            df_vdem = _std(df_vdem, iso3_col="iso3")
            all_frames.append(df_vdem)
            logger.info("V-Dem: %d rows, %d cols", *df_vdem.shape)

    # OpenAlex — AI & ML publication counts
    if config.get("openalex", {}).get("enabled", True):
        logger.info("─── OpenAlex ───────────────────────────────────")
        df_oa = fetch_all_openalex(config, raw_dir=raw_dir)
        if not df_oa.empty:
            all_frames.append(df_oa)
            logger.info("OpenAlex: %d rows, %d cols", *df_oa.shape)

    # Top500 — supercomputer infrastructure
    if config.get("top500", {}).get("enabled", True):
        logger.info("─── Top500 ─────────────────────────────────────")
        df_t5 = fetch_all_top500(config, raw_dir=raw_dir)
        if not df_t5.empty:
            all_frames.append(df_t5)
            logger.info("Top500: %d rows, %d cols", *df_t5.shape)

    # Epoch AI — AI model counts
    if config.get("epochai", {}).get("enabled", True):
        logger.info("─── Epoch AI ───────────────────────────────────")
        df_ea = fetch_all_epochai(config, raw_dir=raw_dir)
        if not df_ea.empty:
            all_frames.append(df_ea)
            logger.info("Epoch AI: %d rows, %d cols", *df_ea.shape)

    if not all_frames:
        raise RuntimeError("All data sources returned empty DataFrames. Check your config and network.")

    # ── Merge ─────────────────────────────────────────────────────────────────
    logger.info("─── Merging sources ────────────────────────────")
    panel = _merge_all(all_frames)

    # ── Year filter (safety net) ───────────────────────────────────────────────
    panel = panel[(panel["year"] >= start_yr) & (panel["year"] <= end_yr)].copy()

    # ── Country filter ────────────────────────────────────────────────────────
    country_filter = config.get("countries", {}).get("filter") or []
    if country_filter:
        panel = apply_country_filter(panel, iso3_col="iso3", filter_codes=country_filter)

    # ── Drop rows where both iso3 and year are missing ────────────────────────
    panel = panel.dropna(subset=["iso3", "year"])

    # ── Remove non-country codes (WB aggregates etc.) ─────────────────────────
    panel = _filter_to_countries(panel)

    # ── Derived variables ─────────────────────────────────────────────────────
    panel = add_derived_variables(panel)

    # ── Sort ──────────────────────────────────────────────────────────────────
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_countries = panel["iso3"].nunique()
    n_years     = panel["year"].nunique()
    n_cols      = len(panel.columns)
    completeness = panel.drop(columns=["iso3", "year", "country_name"], errors="ignore").notna().mean().mean()

    logger.info("=" * 60)
    logger.info("Panel summary")
    logger.info("  Rows      : %d", len(panel))
    logger.info("  Countries : %d", n_countries)
    logger.info("  Years     : %d–%d (%d unique)", panel["year"].min(), panel["year"].max(), n_years)
    logger.info("  Columns   : %d", n_cols)
    logger.info("  Coverage  : %.1f%% non-missing", completeness * 100)
    logger.info("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────────────
    _save_outputs(panel, out_dir, config["pipeline"].get("output_formats", ["csv"]))

    elapsed = time.time() - t0
    logger.info("Pipeline completed in %.1fs", elapsed)
    return panel


def _save_outputs(df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    """Save the panel to the configured output formats."""
    for fmt in formats:
        fmt = fmt.lower().strip()
        if fmt == "csv":
            path = out_dir / "panel.csv"
            df.to_csv(path, index=False)
            logger.info("Saved CSV  → %s", path)
        elif fmt == "parquet":
            path = out_dir / "panel.parquet"
            df.to_parquet(path, index=False)
            logger.info("Saved Parquet → %s", path)
        elif fmt in ("excel", "xlsx"):
            path = out_dir / "panel.xlsx"
            df.to_excel(path, index=False, sheet_name="panel")
            logger.info("Saved Excel → %s", path)
        else:
            logger.warning("Unknown output format '%s'; skipping.", fmt)

    # Always save a codebook
    _save_codebook(df, out_dir)


def _save_codebook(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Generate a simple codebook CSV: column name, dtype, % non-missing,
    min/max for numerics.
    """
    rows = []
    for col in df.columns:
        series = df[col]
        pct_valid = series.notna().mean() * 100
        row = {
            "column": col,
            "dtype": str(series.dtype),
            "pct_non_missing": round(pct_valid, 1),
        }
        if pd.api.types.is_numeric_dtype(series):
            row["min"] = series.min()
            row["max"] = series.max()
            row["mean"] = round(series.mean(), 4)
        rows.append(row)

    codebook = pd.DataFrame(rows)
    path = out_dir / "codebook.csv"
    codebook.to_csv(path, index=False)
    logger.info("Saved codebook → %s", path)
