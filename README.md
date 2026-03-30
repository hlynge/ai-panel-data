# ai-panel-data

A reproducible data pipeline that assembles a balanced country-year panel dataset for cross-national analysis of AI governance and digital development.

## Dataset

**~180 countries × 15 years (2010–2024)**
70 variables across eight thematic groups

| Source | Variables |
|--------|-----------|
| OECD MSTI | R&D expenditure, patent activity, researchers |
| OECD Patents | AI triadic patent families, WIPO PCT applications |
| OECD ICT | Household internet, broadband, and computer access |
| OECD BERD | Business R&D in ICT-intensive industries |
| World Bank WDI | Macroeconomic, digital infrastructure, education |
| World Bank WGI | Six governance indicators |
| IMF World Economic Outlook | Fiscal and macroeconomic indicators |
| V-Dem v16 | Democracy, civil liberties, rule of law |
| OpenAlex | AI and ML scientific publication counts by country-year |
| Top500 | Supercomputer infrastructure (systems count, Rmax in PFlop/s) |
| Epoch AI | AI model releases; large-model (≥1B parameter) counts |

See `data/processed/codebook.csv` for full variable descriptions, sources, units, and missingness rates.

## Structure

```
ai-panel-data/
├── main.py              # Pipeline entry point
├── config.yaml          # Data sources, indicators, and year range
├── codebook_meta.csv    # Hand-curated variable metadata (merged at save time)
├── requirements.txt     # Python dependencies
├── 01.explore.R         # R exploration script (ggplot2 plots)
├── src/
│   ├── pipeline.py      # Merge and clean all sources
│   ├── worldbank.py     # World Bank WDI + WGI fetcher
│   ├── imf.py           # IMF WEO fetcher
│   ├── vdem.py          # V-Dem loader
│   ├── oecd_ai.py       # OECD MSTI, patents, ICT, BERD fetcher
│   ├── oecd_api.py      # Low-level OECD SDMX-JSON client
│   ├── openalex.py      # OpenAlex AI/ML publication counts
│   ├── top500.py        # Top500 supercomputer list (XML download)
│   ├── epochai.py       # Epoch AI model database
│   ├── harmonize.py     # Country code standardisation
│   └── utils.py         # Logging, config, helpers
└── data/
    ├── raw/             # Cached API responses and downloaded files
    └── processed/
        ├── panel.csv        # Main panel dataset
        ├── panel.parquet    # Same, in Parquet format
        └── codebook.csv     # Variable metadata + auto-computed statistics
```

## Usage

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the full pipeline:**
```bash
python main.py
```

**Options:**
```bash
python main.py --start 2015 --end 2022          # Restrict year range
python main.py --countries USA GBR DEU FRA JPN  # Restrict to specific countries
python main.py --sources worldbank imf          # Run specific sources only
python main.py --dry-run                        # Validate config without fetching
```

**Explore in R:**
```r
source("01.explore.R")   # Requires: readr, dplyr, tidyr, ggplot2, scales
```

**Output** is written to `data/processed/`:
- `panel.csv` — main panel dataset
- `panel.parquet` — same, in Parquet format
- `codebook.csv` — variable metadata merged with auto-computed statistics (dtype, % non-missing, min, max, mean)

## Data notes

- Non-country ISO3 codes (World Bank regional and income aggregates) are filtered from the output
- R&D variables have substantial missingness — coverage is effectively limited to OECD member states
- V-Dem missingness reflects V-Dem's design: coverage limited to countries with population above 500,000
- Top500 data is heavily concentrated in the United States and China; ~30 countries represented
- Epoch AI model counts are US/China-skewed; ~49 countries represented
- OpenAlex publication counts cover all countries with at least one affiliated author
- Two IMF indicators (`GGR_NGDP`, `GGX_NGDP`) currently return no data from the API

## Requirements

- Python 3.10+
- V-Dem full dataset (v16) must be downloaded separately and the path configured in `config.yaml`
- All other data is fetched automatically via public APIs (no authentication required)

## License

Data sources are subject to their own terms of use. Code in this repository is released under the MIT License.
