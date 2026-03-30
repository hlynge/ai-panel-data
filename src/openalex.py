"""
openalex.py — AI and ML publication counts by country-year from OpenAlex.

Uses the OpenAlex REST API group-by endpoint to count scientific papers
per country per year for the Artificial Intelligence concept (C154945302)
and Machine Learning concept (C119857082).

API reference: https://docs.openalex.org/api-entities/works/group-by
No authentication required.  Uses the "polite" pool by including a mailto
in the User-Agent header (improves rate-limit headroom to ~10 req/s).

Variables produced
──────────────────
  ai_paper_count  : papers tagged with OpenAlex concept "Artificial Intelligence"
  ml_paper_count  : papers tagged with OpenAlex concept "Machine Learning"

Coverage: ~190–200 countries, 2010–2024 (complete calendar years only).
API makes 15 calls per concept (one per year); total ≤ 30 HTTP requests.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from .harmonize import iso2_to_iso3
from .utils import get_logger, save_raw

logger = get_logger(__name__)


_BASE_URL = "https://api.openalex.org/works"
_HEADERS = {
    # OpenAlex asks for a mailto address to get into the "polite" pool.
    # Using a generic placeholder; replace with your own email if desired.
    "User-Agent": "ai-panel-data/1.0 (academic research; mailto:noreply@example.com)"
}

# OpenAlex concept IDs
_AI_CONCEPT = "C154945302"   # Artificial Intelligence
_ML_CONCEPT = "C119857082"   # Machine Learning

# Polite pool supports ~10 req/s; 0.3 s gap keeps well under that
_SLEEP_BETWEEN_CALLS = 0.3


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_concept_by_country_year(
    concept_id: str,
    year: int,
    retries: int = 3,
) -> list[dict]:
    """
    Fetch paper counts grouped by country for a single concept and year.

    Returns a list of {"key": "US", "count": 12345} dicts from the
    OpenAlex group_by response (up to 200 entries per call).
    """
    params = {
        "filter":   f"concepts.id:{concept_id},publication_year:{year}",
        "group_by": "authorships.institutions.country_code",
        "per_page": 200,
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(_BASE_URL, params=params, headers=_HEADERS, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data.get("group_by", [])
        except Exception as exc:
            if attempt == retries:
                logger.error(
                    "OpenAlex: failed for concept=%s year=%d after %d attempts: %s",
                    concept_id, year, retries, exc,
                )
                return []
            wait = 2 ** attempt
            logger.warning("Attempt %d/%d failed: %s. Retrying in %ds…", attempt, retries, exc, wait)
            time.sleep(wait)
    return []


def _concept_to_df(
    concept_id: str,
    col_name: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Iterate over years and build a long country-year DataFrame.

    ISO-2 codes from OpenAlex are converted to ISO-3 using the existing
    `iso2_to_iso3` lookup table.  Countries that cannot be resolved are
    silently dropped.
    """
    rows = []
    for year in range(start_year, end_year + 1):
        groups = _fetch_concept_by_country_year(concept_id, year)
        n_matched = 0
        for entry in groups:
            # OpenAlex returns full URL keys: "https://openalex.org/countries/US"
            raw   = str(entry.get("key", "")).strip()
            iso2  = raw.rsplit("/", 1)[-1].upper()   # grab "US" from the URL tail
            count = entry.get("count", 0)
            iso3  = iso2_to_iso3(iso2)
            if iso3 and count:
                rows.append({"iso3": iso3, "year": year, col_name: count})
                n_matched += 1
        logger.info(
            "  OpenAlex %-16s %d : %d countries (%d resolved to ISO3)",
            col_name, year, len(groups), n_matched,
        )
        time.sleep(_SLEEP_BETWEEN_CALLS)

    if not rows:
        return pd.DataFrame(columns=["iso3", "year", col_name])
    return pd.DataFrame(rows)


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_openalex(
    start_year: int = 2010,
    end_year: int = 2024,
    fetch_ml: bool = True,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """
    Return AI (and optionally ML) publication counts by country-year.

    Parameters
    ----------
    start_year : First year to fetch (inclusive).
    end_year   : Last year to fetch (inclusive).
    fetch_ml   : Also fetch Machine Learning paper counts (default True).
    raw_dir    : Directory for raw cache files.

    Returns
    -------
    Wide DataFrame with columns: iso3, year, ai_paper_count[, ml_paper_count]
    """
    logger.info(
        "Fetching OpenAlex AI publication counts (%d–%d)…", start_year, end_year
    )
    df_ai = _concept_to_df(_AI_CONCEPT, "ai_paper_count", start_year, end_year)

    if fetch_ml:
        logger.info("Fetching OpenAlex ML publication counts…")
        df_ml = _concept_to_df(_ML_CONCEPT, "ml_paper_count", start_year, end_year)
        if not df_ai.empty and not df_ml.empty:
            df = df_ai.merge(df_ml, on=["iso3", "year"], how="outer")
        elif not df_ai.empty:
            df = df_ai
        else:
            df = df_ml
    else:
        df = df_ai

    if df.empty:
        logger.warning("OpenAlex returned no data.")
        return pd.DataFrame()

    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    save_raw(df, "openalex_raw", raw_dir)

    logger.info(
        "OpenAlex: %d country-year rows, %d countries",
        len(df), df["iso3"].nunique(),
    )
    return df


def fetch_all_openalex(
    config: dict,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Entry point called by the main pipeline."""
    oa_cfg = config.get("openalex", {})
    if not oa_cfg.get("enabled", True):
        logger.info("OpenAlex disabled in config.")
        return pd.DataFrame()

    start    = config["pipeline"]["start_year"]
    end      = config["pipeline"]["end_year"]
    fetch_ml = oa_cfg.get("fetch_ml", True)

    return fetch_openalex(
        start_year=start,
        end_year=end,
        fetch_ml=fetch_ml,
        raw_dir=raw_dir,
    )
