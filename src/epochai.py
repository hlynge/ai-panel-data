"""
epochai.py — AI model counts by country-year from Epoch AI's database.

Downloads the Epoch AI "All AI Models" CSV and aggregates to country-year.

Country attribution: a model is attributed to every country listed in its
"Organisation country" column (which is comma-separated for multi-org models;
roughly 32 % of rows have more than one country).

Source: https://epoch.ai/data/all_ai_models.csv
Licence: CC BY 4.0 (credit: Epoch AI Research, https://epoch.ai)

Variables produced
──────────────────
  ai_model_count    : all models with a known publication / release year
  large_model_count : models with parameter count ≥ 1 billion (a proxy for
                      compute-intensive / frontier-class models)

Coverage: ~49 countries across the panel years.  Data are heavily skewed
toward USA and China; interpret small-country counts with care.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from .utils import get_logger, save_raw, to_iso3

logger = get_logger(__name__)


_EPOCH_CSV_URL = "https://epoch.ai/data/all_ai_models.csv"
_HEADERS       = {"User-Agent": "ai-panel-data/1.0 (academic research)"}

# Parameters threshold for "large" models
_LARGE_MODEL_THRESHOLD = 1e9  # 1 billion parameters


# ── Download / cache ──────────────────────────────────────────────────────────

def _load_epoch_csv(dest: Path) -> pd.DataFrame | None:
    """
    Return the Epoch AI CSV as a DataFrame.

    Uses *dest* as a local cache: if the file already exists it is read
    directly; otherwise it is downloaded and saved.
    """
    if dest.exists():
        logger.info("Using cached Epoch AI data: %s", dest)
        try:
            return pd.read_csv(dest, low_memory=False)
        except Exception as exc:
            logger.warning("Cache read failed (%s); re-downloading…", exc)

    logger.info("Downloading Epoch AI dataset from %s…", _EPOCH_CSV_URL)
    try:
        resp = requests.get(_EPOCH_CSV_URL, headers=_HEADERS, timeout=120)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return pd.read_csv(StringIO(resp.text), low_memory=False)
    except Exception as exc:
        logger.error("Epoch AI download failed: %s", exc)
        return None


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """Return the first column whose name contains any keyword (case-insensitive)."""
    kw_lower = [k.lower() for k in keywords]
    for col in df.columns:
        if any(k in col.lower() for k in kw_lower):
            return col
    return None


def _expand_countries(df: pd.DataFrame, country_col: str) -> pd.DataFrame:
    """
    Expand rows where *country_col* holds a comma-separated list of countries.
    Each unique country gets its own row; the model is attributed to all of them.
    """
    df = df.copy()
    df[country_col] = df[country_col].fillna("").str.split(",")
    df = df.explode(country_col)
    df[country_col] = df[country_col].str.strip()
    return df[df[country_col] != ""].copy()


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_epochai(
    start_year: int = 2010,
    end_year: int = 2024,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """
    Return AI model counts by country-year from the Epoch AI database.

    Parameters
    ----------
    start_year : First year to include (inclusive).
    end_year   : Last year to include (inclusive).
    raw_dir    : Directory for raw cache files.

    Returns
    -------
    Wide DataFrame: iso3, year, ai_model_count, large_model_count
    """
    raw_dir = Path(raw_dir)
    dest    = raw_dir / "epoch_all_ai_models.csv"

    df_raw = _load_epoch_csv(dest)
    if df_raw is None:
        return pd.DataFrame()

    logger.info(
        "Epoch AI raw: %d models × %d columns", len(df_raw), len(df_raw.columns)
    )

    # ── Identify key columns (Epoch column names vary across releases) ─────────
    year_col    = _find_col(df_raw, ["publication date", "release date", "year"])
    country_col = _find_col(df_raw, ["organisation country", "organization country", "country"])
    param_col   = _find_col(df_raw, ["parameters"])

    if year_col is None or country_col is None:
        logger.error(
            "Epoch CSV: could not identify year or country column.\n"
            "Available columns: %s", df_raw.columns.tolist()
        )
        return pd.DataFrame()

    logger.info(
        "Epoch AI column mapping — year: '%s', country: '%s', params: '%s'",
        year_col, country_col, param_col or "not found",
    )

    keep_cols = [year_col, country_col] + ([param_col] if param_col else [])
    df = df_raw[keep_cols].copy()

    # ── Parse year ─────────────────────────────────────────────────────────────
    # Column may hold full dates ("2023-04-05") or just a year number ("2023").
    df["year"] = pd.to_datetime(df[year_col], errors="coerce").dt.year
    # Fall back to numeric year if datetime parse failed everywhere
    if df["year"].isna().all():
        df["year"] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # ── Year filter ────────────────────────────────────────────────────────────
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
    if df.empty:
        logger.warning("Epoch AI: no models in %d–%d after year filtering.", start_year, end_year)
        return pd.DataFrame()

    # ── Expand multi-country rows ──────────────────────────────────────────────
    df = _expand_countries(df, country_col)

    # ── Country → ISO3 ─────────────────────────────────────────────────────────
    df["iso3"] = df[country_col].apply(to_iso3)
    n_unmatched = df["iso3"].isna().sum()
    if n_unmatched:
        failed = df.loc[df["iso3"].isna(), country_col].unique().tolist()
        logger.info(
            "Epoch AI: %d rows could not be matched to ISO3 (e.g. %s); dropped.",
            n_unmatched, failed[:5],
        )
    df = df[df["iso3"].notna()].copy()

    # ── Flag large models ──────────────────────────────────────────────────────
    if param_col:
        df[param_col] = pd.to_numeric(df[param_col], errors="coerce")
        df["_is_large"] = df[param_col] >= _LARGE_MODEL_THRESHOLD
    else:
        df["_is_large"] = False

    # ── Aggregate to country-year ──────────────────────────────────────────────
    agg = (
        df.groupby(["iso3", "year"])
        .agg(
            ai_model_count  = ("iso3",     "count"),
            large_model_count = ("_is_large", "sum"),
        )
        .reset_index()
    )
    agg["large_model_count"] = agg["large_model_count"].astype(int)
    agg = agg.sort_values(["iso3", "year"]).reset_index(drop=True)

    save_raw(agg, "epochai_raw", raw_dir)
    logger.info(
        "Epoch AI: %d country-year rows, %d countries",
        len(agg), agg["iso3"].nunique(),
    )
    return agg


def fetch_all_epochai(
    config: dict,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Entry point called by the main pipeline."""
    ea_cfg = config.get("epochai", {})
    if not ea_cfg.get("enabled", True):
        logger.info("Epoch AI disabled in config.")
        return pd.DataFrame()

    start = config["pipeline"]["start_year"]
    end   = config["pipeline"]["end_year"]

    return fetch_epochai(start_year=start, end_year=end, raw_dir=raw_dir)
