# ==============================================================================
# 01_explore.R
# Initial exploration of the OECD AI panel dataset
# Panel: 216 countries x 15 years (2010-2024)
# ==============================================================================

# -- Packages ------------------------------------------------------------------
library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)

# -- Paths ---------------------------------------------------------------------
data_dir   <- "/Users/lynge/Nextcloud/Research/Data/OECD.ai/data/processed/"
output_dir <- "outputs/"

# -- Load data -----------------------------------------------------------------
# Pipeline outputs a clean dataset -- aggregates filtered at source
df <- read_csv(file.path(data_dir, "panel.csv"))

# -- Panel structure -----------------------------------------------------------
glimpse(df)

# Countries, years, range
df %>% summarise(
  countries = n_distinct(iso3),
  years     = n_distinct(year),
  year_min  = min(year),
  year_max  = max(year)
)

# Balanced panel check (all countries should have n = 15)
df %>% count(iso3) %>% count(n, name = "countries")

# OECD vs non-OECD
df %>% distinct(iso3, oecd_member) %>% count(oecd_member)

# -- Missingness ---------------------------------------------------------------
df %>%
  summarise(across(everything(), ~ mean(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "pct_missing") %>%
  arrange(desc(pct_missing)) %>%
  print(n = 54)

# -- Variable notes ------------------------------------------------------------

# NOTE: V-Dem missingness (~19%) is by design -- V-Dem covers countries with
# population > 500k only. Not a pipeline issue.
df %>%
  filter(year == 2020) %>%
  summarise(
    below_500k  = sum(population_total < 500000, na.rm = TRUE),
    above_500k  = sum(population_total >= 500000, na.rm = TRUE),
    missing_pop = sum(is.na(population_total))
  )
