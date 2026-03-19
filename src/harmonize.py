"""
harmonize.py — standardise country identifiers across all data sources.

The problem
───────────
Each source uses a different convention:
  • OECD old API        → ISO 3166-1 alpha-2 (e.g. "US", "GB")
  • World Bank / IMF    → ISO 3166-1 alpha-3 (e.g. "USA", "GBR")
  • V-Dem               → ISO 3166-1 alpha-3  (already consistent)
  • Stanford HAI        → full country names  (e.g. "United States")
  • OECD new API        → alpha-3

Solution
────────
We convert everything to ISO 3166-1 alpha-3 (the panel's key column: `iso3`)
using the `country_converter` (coco) package, with a curated manual override
table for edge cases.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .utils import get_logger, harmonise_country_column, to_iso3

logger = get_logger(__name__)


# ── ISO2 → ISO3 lookup (for OECD old API output) ─────────────────────────────

_ISO2_TO_ISO3: dict[str, str] = {
    "AF": "AFG", "AX": "ALA", "AL": "ALB", "DZ": "DZA", "AS": "ASM",
    "AD": "AND", "AO": "AGO", "AI": "AIA", "AQ": "ATA", "AG": "ATG",
    "AR": "ARG", "AM": "ARM", "AW": "ABW", "AU": "AUS", "AT": "AUT",
    "AZ": "AZE", "BS": "BHS", "BH": "BHR", "BD": "BGD", "BB": "BRB",
    "BY": "BLR", "BE": "BEL", "BZ": "BLZ", "BJ": "BEN", "BM": "BMU",
    "BT": "BTN", "BO": "BOL", "BA": "BIH", "BW": "BWA", "BV": "BVT",
    "BR": "BRA", "IO": "IOT", "BN": "BRN", "BG": "BGR", "BF": "BFA",
    "BI": "BDI", "CV": "CPV", "KH": "KHM", "CM": "CMR", "CA": "CAN",
    "KY": "CYM", "CF": "CAF", "TD": "TCD", "CL": "CHL", "CN": "CHN",
    "CX": "CXR", "CC": "CCK", "CO": "COL", "KM": "COM", "CG": "COG",
    "CD": "COD", "CK": "COK", "CR": "CRI", "CI": "CIV", "HR": "HRV",
    "CU": "CUB", "CW": "CUW", "CY": "CYP", "CZ": "CZE", "DK": "DNK",
    "DJ": "DJI", "DM": "DMA", "DO": "DOM", "EC": "ECU", "EG": "EGY",
    "SV": "SLV", "GQ": "GNQ", "ER": "ERI", "EE": "EST", "SZ": "SWZ",
    "ET": "ETH", "FK": "FLK", "FO": "FRO", "FJ": "FJI", "FI": "FIN",
    "FR": "FRA", "GF": "GUF", "PF": "PYF", "TF": "ATF", "GA": "GAB",
    "GM": "GMB", "GE": "GEO", "DE": "DEU", "GH": "GHA", "GI": "GIB",
    "GR": "GRC", "GL": "GRL", "GD": "GRD", "GP": "GLP", "GU": "GUM",
    "GT": "GTM", "GG": "GGY", "GN": "GIN", "GW": "GNB", "GY": "GUY",
    "HT": "HTI", "HM": "HMD", "VA": "VAT", "HN": "HND", "HK": "HKG",
    "HU": "HUN", "IS": "ISL", "IN": "IND", "ID": "IDN", "IR": "IRN",
    "IQ": "IRQ", "IE": "IRL", "IM": "IMN", "IL": "ISR", "IT": "ITA",
    "JM": "JAM", "JP": "JPN", "JE": "JEY", "JO": "JOR", "KZ": "KAZ",
    "KE": "KEN", "KI": "KIR", "KP": "PRK", "KR": "KOR", "KW": "KWT",
    "KG": "KGZ", "LA": "LAO", "LV": "LVA", "LB": "LBN", "LS": "LSO",
    "LR": "LBR", "LY": "LBY", "LI": "LIE", "LT": "LTU", "LU": "LUX",
    "MO": "MAC", "MG": "MDG", "MW": "MWI", "MY": "MYS", "MV": "MDV",
    "ML": "MLI", "MT": "MLT", "MH": "MHL", "MQ": "MTQ", "MR": "MRT",
    "MU": "MUS", "YT": "MYT", "MX": "MEX", "FM": "FSM", "MD": "MDA",
    "MC": "MCO", "MN": "MNG", "ME": "MNE", "MS": "MSR", "MA": "MAR",
    "MZ": "MOZ", "MM": "MMR", "NA": "NAM", "NR": "NRU", "NP": "NPL",
    "NL": "NLD", "NC": "NCL", "NZ": "NZL", "NI": "NIC", "NE": "NER",
    "NG": "NGA", "NU": "NIU", "NF": "NFK", "MK": "MKD", "MP": "MNP",
    "NO": "NOR", "OM": "OMN", "PK": "PAK", "PW": "PLW", "PS": "PSE",
    "PA": "PAN", "PG": "PNG", "PY": "PRY", "PE": "PER", "PH": "PHL",
    "PN": "PCN", "PL": "POL", "PT": "PRT", "PR": "PRI", "QA": "QAT",
    "RE": "REU", "RO": "ROU", "RU": "RUS", "RW": "RWA", "BL": "BLM",
    "SH": "SHN", "KN": "KNA", "LC": "LCA", "MF": "MAF", "PM": "SPM",
    "VC": "VCT", "WS": "WSM", "SM": "SMR", "ST": "STP", "SA": "SAU",
    "SN": "SEN", "RS": "SRB", "SC": "SYC", "SL": "SLE", "SG": "SGP",
    "SX": "SXM", "SK": "SVK", "SI": "SVN", "SB": "SLB", "SO": "SOM",
    "ZA": "ZAF", "GS": "SGS", "SS": "SSD", "ES": "ESP", "LK": "LKA",
    "SD": "SDN", "SR": "SUR", "SJ": "SJM", "SE": "SWE", "CH": "CHE",
    "SY": "SYR", "TW": "TWN", "TJ": "TJK", "TZ": "TZA", "TH": "THA",
    "TL": "TLS", "TG": "TGO", "TK": "TKL", "TO": "TON", "TT": "TTO",
    "TN": "TUN", "TR": "TUR", "TM": "TKM", "TC": "TCA", "TV": "TUV",
    "UG": "UGA", "UA": "UKR", "AE": "ARE", "GB": "GBR", "US": "USA",
    "UM": "UMI", "UY": "URY", "UZ": "UZB", "VU": "VUT", "VE": "VEN",
    "VN": "VNM", "VG": "VGB", "VI": "VIR", "WF": "WLF", "EH": "ESH",
    "YE": "YEM", "ZM": "ZMB", "ZW": "ZWE",
    # OECD aggregate codes (keep as-is for filtering)
    "OECD": "OECD", "EU27_2020": "EU27", "G20": "G20", "WLD": "WLD",
}


def iso2_to_iso3(code: str) -> Optional[str]:
    """Convert an ISO 3166-1 alpha-2 code to alpha-3.  Returns None if unknown."""
    return _ISO2_TO_ISO3.get(str(code).strip().upper())


def standardise_iso3(
    df: pd.DataFrame,
    col: str,
    drop_aggregates: bool = True,
) -> pd.DataFrame:
    """
    Ensure *col* contains ISO3 codes.  Handles:
      • Already-ISO3 values (pass through)
      • ISO2 codes (look up in table above)
      • Country names (use country_converter)

    Parameters
    ----------
    df               : Input DataFrame.
    col              : Column containing country identifiers.
    drop_aggregates  : If True, remove rows where the converted code is
                       a regional aggregate (OECD, EU27, WLD, G20, …).

    Returns
    -------
    DataFrame with *col* replaced by ISO3 codes; unresolvable rows get NaN.
    """
    df = df.copy()

    def _convert(val):
        if pd.isna(val):
            return None
        val = str(val).strip()
        # Already ISO3?
        if len(val) == 3 and val.isupper():
            return val
        # ISO2?
        mapped = iso2_to_iso3(val)
        if mapped:
            return mapped
        # Try country_converter / pycountry via utils
        return to_iso3(val)

    df[col] = df[col].apply(_convert)

    if drop_aggregates:
        aggregates = {"OECD", "EU27", "G20", "WLD", "EU28", "EU15", "DAC"}
        mask = df[col].isin(aggregates)
        if mask.any():
            logger.info("Dropping %d aggregate rows (%s)", mask.sum(), sorted(df.loc[mask, col].unique()))
            df = df[~mask]

    return df


def apply_country_filter(
    df: pd.DataFrame,
    iso3_col: str,
    filter_codes: list[str],
) -> pd.DataFrame:
    """Keep only rows whose *iso3_col* value is in *filter_codes*."""
    if not filter_codes:
        return df
    df = df[df[iso3_col].isin(filter_codes)].copy()
    logger.info("After country filter: %d rows", len(df))
    return df
