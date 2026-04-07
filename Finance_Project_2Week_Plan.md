# 🇵🇰 Finance Project: Pakistan Inflation & Search Tracker
### *Does Economic Stress Predict Stock Performance?*
**Timeline: 2 Weeks | Language: Python | Market: Pakistan (PSX)**

---

## 📌 Project Overview

This project investigates the relationship between **Economic Stress** (inflation signals) and **Consumer Behavior** in Pakistan. By comparing search trends (Google Trends) with stock market data (PSX), we aim to prove how different sectors react during periods of high inflation and currency devaluation.

**Your Goal:** Statistically analyze whether searches for "Dollar Rate" and "Petrol Price" correlate with a dip in cyclical stocks (like Cement and Textiles) while defensive stocks (Food and Pharma) remain resilient.

---

## 🧠 The "Economic Stress" Hypothesis

> *"When the PKR devalues (Dollar Rate search spikes), search interest for discretionary goods (New Cars, Property) falls, and cyclical stocks crash, while python -m pip install --upgrade pip setuptools wheeldefensive stocks (Nestle, Abbott) maintain steady performance as essential goods are still required."*

---

## 🗂️ Data Sources

| Data Type | Entities | Source |
|---|---|---|
| **Market Index** | ^KSE100 | `yfinance` |
| **Defensive Stocks** | Nestle (NESTLE), Abbott (ABOT), National Foods (NATF) | `yfinance` |
| **Cyclical Stocks** | Lucky Cement (LUCK), HBL Bank (HBL), Gul Ahmed (GULT) | `yfinance` |
| **Inflation Stress** | "Dollar Rate", "Petrol Price", "Inflation PK" | `pytrends` |
| **Discretionary** | "New Car", "Property Pakistan", "Foreign Tour" | `pytrends` |

---

## 📅 2-Week Sprint Plan

### 📦 WEEK 1 — Data Pipeline & Alignment

- **Day 1–2: Acquisition:** Use `yfinance` and `pytrends` to download 14 years of historical data (2010–2024).
- **Day 3–4: Cleaning & Alignment:** 
    - Resample stock data from Daily → Monthly.
    - Normalize all values to a 0–100 scale using Min-Max scaling.
    - Merge into a single master dataset (`data/cleaned_merged.csv`).
- **Day 5: EDA:** Plot initial time-series charts to identify major stress events (e.g., 2018 currency devaluation, 2022-23 inflation spike).

### 🔬 WEEK 2 — Statistical Analysis & Forecasting

- **Day 6–7: Correlation Matrix:** Generate heatmaps to quantify the link between "Dollar Rate" searches and stock price movements.
- **Day 8–9: Smoothing & Rolling Averages:** Apply a 6-month rolling window to identify long-term economic shifts versus temporary noise.
- **Day 10–11: Forecasting:** Use Linear Regression to predict the next 12 months for key indicators.
- **Day 12–14: Final Reporting:** Package all visuals and findings into a summary report.

---

## 📁 Project Structure

```
Finance Project/
│
├── data/
│   ├── stocks_raw.csv
│   ├── trends_raw.csv
│   └── cleaned_merged.csv
│
├── visuals/
│   ├── chart1_defensive_vs_cyclical.png
│   ├── chart2_stress_indicators.png
│   ├── chart3_correlation_heatmap.png
│   └── chart4_rolling_averages.png
│
├── Scripts/
│   └── main.py (Unified collection & analysis script)
│
└── README.md
```

---

## ⚡ Quick Start
To run the entire analysis pipeline:
```bash
python Scripts/main.py
```
This will automatically refresh all raw data, perform cleaning, and update all charts in the `visuals/` folder.
