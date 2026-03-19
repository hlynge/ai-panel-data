# ai-panel-data

A Python pipeline that pulls the latest data from **OECD AI Observatory**, **World Bank WDI**, **IMF World Economic Outlook**, and **V-Dem**, then merges everything into a clean, analysis-ready **country-year panel dataset**.

---

## What you get

| Source | What it covers |
|---|---|
| **OECD MSTI** | R&D expenditure (total/business/govt/HE as % GDP), researchers per 1,000 employed |
| **OECD PATS_IPC** | AI-related patent applications (IPC classes G06N ML/AI, G06F, G06K, G06T, G06V, G10L) |
| **OECD ICT_HH2** | Internet access, broadband, mobile, online activity |
| **OECD BERD_NACE2** | Business R&D in ICT-intensive industries |
| **Stanford HAI AI Index** | Private AI investment by country (best-effort download) |
| **World Bank WDI** | GDP, trade, education, digital infra, World Governance Indicators |
| **IMF WEO** | GDP, inflation, unemployment, fiscal balances, government debt |
| **V-Dem** | Liberal democracy, civil liberties, rule of law, corruption, electoral quality |

Output: `data/processed/panel.csv` (and `.parquet`) — one row per country-year, one column per indicator — plus a `codebook.csv` with coverage statistics.

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/ai-panel-data.git
cd ai-panel-data
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Configure

Edit `config.yaml` to:
- Change the year range (`start_year` / `end_year`)
- Restrict to specific countries (`countries.filter`)
- Enable / disable sources
- Add or remove indicators

### 3. Run

```bash
python main.py
```

This fetches all sources, merges them, and writes the panel to `data/processed/`.

**Other options:**

```bash
# See what would be fetched without making API calls:
python main.py --dry-run

# Restrict to a specific period:
python main.py --start 2015 --end 2022

# Restrict to specific countries (ISO3 codes):
python main.py --countries USA GBR DEU FRA JPN KOR CHN IND BRA

# Run only selected sources:
python main.py --sources oecd worldbank
```

### 4. Explore

Open the Jupyter notebook for descriptive plots and a sample regression:

```bash
pip install jupyter matplotlib seaborn statsmodels
jupyter notebook notebooks/explore.ipynb
```

---

## V-Dem data

V-Dem does not offer a public REST API. The pipeline tries to download the dataset automatically from `v-dem.net`, but this may fail if the URL changes between annual releases.

**Manual download (recommended):**
1. Go to https://v-dem.net/data/the-v-dem-dataset/
2. Download *Country-Year: V-Dem Full+Others* (CSV format).
3. Unzip and set the path in `config.yaml`:

```yaml
vdem:
  local_file: "data/raw/V-Dem-CY-Full+Others-v15.csv"
```

**R users:** If you have the `vdemdata` R package, export from R:
```r
library(vdemdata)
write.csv(vdem, "data/raw/vdem.csv", row.names = FALSE)
```
Then point `local_file` to that CSV.

---

## Project structure

```
ai-panel-data/
├── main.py                  # Entry point — run this
├── config.yaml              # All settings: years, countries, indicators
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── utils.py             # Logging, HTTP helpers, country-code conversion
│   ├── oecd_api.py          # Low-level OECD SDMX API wrapper
│   ├── oecd_ai.py           # OECD AI Observatory datasets (MSTI, patents, ICT, BERD)
│   ├── worldbank.py         # World Bank WDI via wbgapi
│   ├── imf.py               # IMF DataMapper REST API
│   ├── vdem.py              # V-Dem dataset loader
│   ├── harmonize.py         # ISO2↔ISO3 conversion, country filtering
│   └── pipeline.py          # Merge all sources → panel; save outputs
│
├── data/
│   ├── raw/                 # Cached raw downloads (.gitignored)
│   └── processed/           # Output panel (.gitignored)
│
└── notebooks/
    └── explore.ipynb        # Descriptive analysis & sample regression
```

---

## Extending the pipeline

**Add a new World Bank indicator:** just add a line to `config.yaml`:
```yaml
worldbank:
  indicators:
    SP.DYN.LE00.IN: life_expectancy
```

**Add a new OECD dataset:** add a function to `src/oecd_ai.py` following the pattern of `fetch_msti()`, then call it from `fetch_all_oecd_ai()`.

**Add an entirely new source:** create `src/my_source.py` with a `fetch_all_my_source(config, raw_dir)` function that returns a DataFrame with columns `iso3` and `year`, then call it in `pipeline.py`.

---

## Data sources & licences

| Source | Licence / Terms |
|---|---|
| OECD SDMX API | [OECD Terms and Conditions](https://www.oecd.org/termsandconditions/) |
| World Bank WDI | [CC BY 4.0](https://datacatalog.worldbank.org/public-licenses) |
| IMF DataMapper | [IMF Terms of Use](https://www.imf.org/external/terms.htm) |
| V-Dem | [CC BY 4.0](https://v-dem.net/data_analysis/VariableGraph/) |
| Stanford HAI AI Index | [HAI Terms](https://aiindex.stanford.edu) |

---

## Tips for VS Code

- Install the **Python** and **Jupyter** extensions.
- Open the folder as a workspace: `File → Open Folder → ai-panel-data/`.
- Select your virtual environment as the Python interpreter (bottom-right status bar).
- Use the **Run and Debug** panel (`Ctrl+Shift+D`) or just press `F5` to run `main.py`.
- The **Data Wrangler** extension (Microsoft) gives you an interactive table view of CSVs — great for inspecting the panel.
