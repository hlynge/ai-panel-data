# ai-panel-data

A reproducible data pipeline that assembles a balanced country-year panel dataset for cross-national analysis of AI governance and digital development.

## Dataset

**216 countries × 15 years (2010–2024)**  
54 variables across five thematic groups  
73% overall coverage (non-missing values)

| Source | Variables |
|--------|-----------|
| OECD MSTI | R&D expenditure, patent activity, researchers |
| World Bank WDI | Macroeconomic, digital infrastructure, education |
| World Bank WGI | Six governance indicators |
| IMF World Economic Outlook | Fiscal and macroeconomic indicators |
| V-Dem v16 | Democracy, civil liberties, rule of law |

See `codebook.csv` for full variable descriptions, sources, units, and missingness rates.

## Structure

```
ai-panel-data/
├── main.py              # Pipeline entry point
├── config.yaml          # Data sources, indicators, and year range
├── codebook.csv         # Variable-level metadata
├── requirements.txt     # Python dependencies
├── src/
│   ├── pipeline.py      # Merge and clean all sources
│   ├── worldbank.py     # World Bank WDI + WGI fetcher
│   ├── imf.py           # IMF WEO fetcher
│   ├── vdem.py          # V-Dem loader
│   ├── oecd_ai.py       # OECD MSTI + patents fetcher
│   ├── harmonize.py     # Country code standardisation
│   └── utils.py         # Logging, config, helpers
└── notebooks/
    └── explore.ipynb    # Exploratory analysis notebook
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

**Output** is written to `data/processed/`:
- `panel.csv` — main panel dataset
- `panel.parquet` — same, in Parquet format
- `codebook.csv` — auto-generated column-level statistics

## Data notes

- Non-country ISO3 codes (World Bank regional and income aggregates) are filtered from the output
- R&D variables have ~81% missingness — coverage is effectively limited to OECD member states
- V-Dem missingness (~19%) reflects V-Dem's design: coverage limited to countries with population above 500,000
- Two IMF indicators (`GGR_NGDP`, `GGX_NGDP`) currently return no data from the API

## Requirements

- Python 3.10+
- V-Dem full dataset (v16) must be downloaded separately and path configured in `config.yaml`
- All other data is fetched automatically via public APIs

## License

Data sources are subject to their own terms of use. Code in this repository is released under the MIT License.
