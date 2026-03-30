# ==============================================================================
# 01.explore.R
# Loads and explores the AI panel dataset produced by main.py
#
# Requires: readr, dplyr, tidyr, ggplot2, scales
#   install.packages(c("readr", "dplyr", "tidyr", "ggplot2", "scales"))
# ==============================================================================

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

# Set working directory to the folder containing this script so that relative
# paths (data/processed/...) always resolve correctly, regardless of what
# RStudio's default working directory is set to.
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

# ------------------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------------------

panel    <- read_csv("data/processed/panel.csv",  show_col_types = FALSE)
codebook <- read_csv("data/processed/codebook.csv", show_col_types = FALSE)

# Quick look
glimpse(panel)

# Codebook: show variable, label, source, unit, coverage
codebook %>%
  select(variable, label, source, unit, pct_non_missing) %>%
  print(n = Inf)

# ------------------------------------------------------------------------------
# 2. Panel structure
# ------------------------------------------------------------------------------

panel %>%
  summarise(
    countries = n_distinct(iso3),
    years     = n_distinct(year),
    year_min  = min(year),
    year_max  = max(year),
    rows      = n(),
    cols      = ncol(panel)
  )

# Balanced panel? (each country should appear once per year)
panel %>%
  count(iso3) %>%
  count(n, name = "n_countries") %>%
  rename(obs_per_country = n)

# OECD vs non-OECD split
panel %>%
  distinct(iso3, oecd_member) %>%
  count(oecd_member) %>%
  mutate(oecd_member = if_else(oecd_member == 1, "OECD member", "Non-member"))

# ------------------------------------------------------------------------------
# 3. Missingness by variable
# ------------------------------------------------------------------------------

panel %>%
  summarise(across(everything(), ~ mean(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "pct_missing") %>%
  arrange(desc(pct_missing)) %>%
  mutate(pct_missing = scales::percent(pct_missing, accuracy = 0.1)) %>%
  print(n = Inf)

# ------------------------------------------------------------------------------
# 4. Quick plots
# ------------------------------------------------------------------------------

# -- R&D spending over time (OECD members only) --------------------------------
panel %>%
  filter(oecd_member == 1, !is.na(rd_total_pct_gdp)) %>%
  ggplot(aes(x = year, y = rd_total_pct_gdp, group = iso3)) +
  geom_line(alpha = 0.4, colour = "steelblue") +
  stat_summary(aes(group = 1), fun = mean, geom = "line",
               colour = "firebrick", linewidth = 1.2) +
  labs(
    title    = "Gross domestic R&D expenditure (% GDP), OECD members",
    subtitle = "Red line = OECD mean",
    x = NULL, y = "GERD (% GDP)",
    caption  = "Source: OECD MSTI"
  ) +
  theme_minimal()

# -- Internet access vs GDP per capita (latest available year) -----------------
panel %>%
  filter(year == max(year)) %>%
  filter(!is.na(internet_users_pct_pop), !is.na(gdp_per_capita_const2015usd)) %>%
  ggplot(aes(
    x      = log(gdp_per_capita_const2015usd),
    y      = internet_users_pct_pop,
    colour = factor(oecd_member),
    label  = iso3
  )) +
  geom_point(alpha = 0.7) +
  geom_text(vjust = -0.5, size = 2.5, check_overlap = TRUE) +
  scale_colour_manual(
    values = c("0" = "grey60", "1" = "steelblue"),
    labels = c("0" = "Non-OECD", "1" = "OECD")
  ) +
  labs(
    title   = paste("Internet access vs. GDP per capita,", max(panel$year, na.rm = TRUE)),
    x       = "Log GDP per capita (const. 2015 USD)",
    y       = "Internet users (% population)",
    colour  = NULL,
    caption = "Source: World Bank WDI"
  ) +
  theme_minimal()

# -- AI publications over time (top 15 countries by 2023 count) ----------------
top_ai_countries <- panel %>%
  filter(year == 2023, !is.na(ai_paper_count)) %>%
  slice_max(ai_paper_count, n = 15) %>%
  pull(iso3)

panel %>%
  filter(iso3 %in% top_ai_countries, !is.na(ai_paper_count)) %>%
  mutate(highlight = iso3 %in% c("USA", "CHN")) %>%
  ggplot(aes(x = year, y = ai_paper_count, group = iso3,
             colour = highlight, alpha = highlight)) +
  geom_line() +
  scale_colour_manual(values = c("FALSE" = "grey60", "TRUE" = "steelblue")) +
  scale_alpha_manual(values  = c("FALSE" = 0.5,      "TRUE" = 1)) +
  scale_y_continuous(labels = scales::comma) +
  guides(colour = "none", alpha = "none") +
  labs(
    title   = "AI publications, top 15 countries",
    subtitle = "USA and China highlighted in blue",
    x = NULL, y = "AI papers (OpenAlex)",
    caption = "Source: OpenAlex concept C154945302"
  ) +
  theme_minimal()

# -- AI patents over time (OECD countries with data) ---------------------------
panel %>%
  filter(!is.na(ai_patent_families_triadic), oecd_member == 1) %>%
  ggplot(aes(x = year, y = ai_patent_families_triadic, group = iso3)) +
  geom_line(alpha = 0.4, colour = "steelblue") +
  stat_summary(aes(group = 1), fun = sum, geom = "line",
               colour = "firebrick", linewidth = 1.2) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title    = "AI triadic patent families, OECD members",
    subtitle = "Red line = OECD total",
    x = NULL, y = "AI triadic patent families",
    caption  = "Source: OECD AI Patents (official AI definition)"
  ) +
  theme_minimal()

# -- Top500 supercomputer systems by country (latest year with data) -----------
panel %>%
  filter(!is.na(top500_n_systems)) %>%
  filter(year == max(year[!is.na(top500_n_systems)])) %>%
  slice_max(top500_n_systems, n = 20) %>%
  mutate(iso3 = reorder(iso3, top500_n_systems)) %>%
  ggplot(aes(x = iso3, y = top500_n_systems)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title   = paste("Top500 supercomputer systems by country,",
                    max(panel$year[!is.na(panel$top500_n_systems)], na.rm = TRUE)),
    x = NULL, y = "Systems in Top500",
    caption = "Source: Top500.org"
  ) +
  theme_minimal()

# -- AI models released per year (top 10 countries) ----------------------------
top_model_countries <- panel %>%
  filter(!is.na(ai_model_count)) %>%
  group_by(iso3) %>%
  summarise(total = sum(ai_model_count, na.rm = TRUE)) %>%
  slice_max(total, n = 10) %>%
  pull(iso3)

panel %>%
  filter(iso3 %in% top_model_countries, !is.na(ai_model_count)) %>%
  ggplot(aes(x = year, y = ai_model_count, fill = iso3)) +
  geom_col(position = "stack") +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title   = "AI models released per year, top 10 countries",
    x = NULL, y = "Models released",
    fill    = NULL,
    caption = "Source: Epoch AI"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")
