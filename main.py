"""
main.py — entry point for the AI panel data pipeline.

Usage
─────
  # Run the full pipeline with default config:
  python main.py

  # Use a different config file:
  python main.py --config my_config.yaml

  # Restrict to a specific year range (overrides config):
  python main.py --start 2015 --end 2022

  # Restrict to specific countries:
  python main.py --countries USA GBR DEU FRA JPN KOR

  # Only run specific sources:
  python main.py --sources oecd worldbank

  # Dry-run (validate config without fetching):
  python main.py --dry-run

Output
──────
  data/processed/panel.csv        — main panel dataset
  data/processed/panel.parquet    — same, in Parquet format (faster to read)
  data/processed/codebook.csv     — column-level metadata
  data/raw/                       — cached raw downloads
"""

import argparse
import sys
from pathlib import Path

# Make sure the src/ package is importable when running from the repo root
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import get_logger, load_config
from src.pipeline import run_pipeline

logger = get_logger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a country-year panel combining OECD AI, World Bank, IMF, and V-Dem data."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument("--start", type=int, help="Override start_year from config")
    parser.add_argument("--end",   type=int, help="Override end_year from config")
    parser.add_argument(
        "--countries",
        nargs="+",
        metavar="ISO3",
        help="Override country filter; provide ISO3 codes separated by spaces",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["oecd", "worldbank", "imf", "vdem"],
        help="Run only the specified sources (default: all enabled in config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and show what would be fetched, without making any API calls",
    )
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides on top of the loaded config dict."""
    if args.start:
        config["pipeline"]["start_year"] = args.start
    if args.end:
        config["pipeline"]["end_year"] = args.end
    if args.countries:
        config["countries"]["filter"] = args.countries

    # If --sources is given, disable all others
    if args.sources:
        all_sources = ["oecd", "worldbank", "imf", "vdem"]
        for src in all_sources:
            if src in config:
                config[src]["enabled"] = src in args.sources
            else:
                config[src] = {"enabled": src in args.sources}

    return config


def dry_run(config: dict) -> None:
    """Print a summary of what the pipeline would fetch."""
    logger.info("DRY RUN — no API calls will be made.")
    logger.info("Config summary:")
    logger.info("  Period   : %d – %d", config["pipeline"]["start_year"], config["pipeline"]["end_year"])
    logger.info("  Formats  : %s", config["pipeline"].get("output_formats"))
    logger.info("  Countries: %s", config.get("countries", {}).get("filter") or "all")

    for src in ("oecd", "worldbank", "imf", "vdem"):
        enabled = config.get(src, {}).get("enabled", True)
        logger.info("  %-12s: %s", src.upper(), "ENABLED" if enabled else "disabled")

    if config.get("oecd", {}).get("enabled", True):
        msti = config["oecd"].get("msti", {}).get("indicators") or "all"
        pats = config["oecd"].get("patents", {}).get("ipc_classes") or "all"
        logger.info("    OECD MSTI indicators : %s", msti)
        logger.info("    OECD Patent IPC      : %s", pats)

    wb_inds = config.get("worldbank", {}).get("indicators", {})
    logger.info("  World Bank indicators: %d", len(wb_inds))

    imf_inds = config.get("imf", {}).get("indicators", {})
    logger.info("  IMF indicators       : %d", len(imf_inds))

    vdem_inds = config.get("vdem", {}).get("indicators", {})
    logger.info("  V-Dem indicators     : %d", len(vdem_inds))


def main() -> None:
    args = parse_args()

    # Load and (optionally) override config
    config = load_config(args.config)
    config = apply_overrides(config, args)

    if args.dry_run:
        dry_run(config)
        return

    # Run the pipeline
    try:
        panel = run_pipeline(args.config)
        logger.info("Done. Panel shape: %s", panel.shape)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
