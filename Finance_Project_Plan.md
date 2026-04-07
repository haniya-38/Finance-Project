# 🧴 Finance Project: The "Budget vs. Bling" Tracker
### *The Lipstick Effect — A Data Science Deep Dive*
**Timeline: 2 Weeks | Language: Python**

---

## 📌 Project Overview

The **Lipstick Effect** is a famous alternative economic indicator. The theory: during economic downturns, consumers cut back on big purchases (cars, designer bags) but still treat themselves to *affordable luxuries* — like lipstick.

**Your Goal:** Statistically prove (or disprove) whether Google Search interest in affordable beauty products goes **up** when the stock market goes **down**.

> [!IMPORTANT]
> This project combines **real financial data** with **public sentiment proxies** — a genuinely impressive data science project that showcases both financial literacy and Python skills.

---

## 🧠 The Core Hypothesis

> *"When the S&P 500 drops (recession signals), Google search interest for affordable beauty products (lipstick, nail polish) rises, while searches for high-end luxury goods fall."*

You'll **test this scientifically** using correlation analysis, rolling averages, and lead-lag tests.

---

## 🗂️ Data Sources

| Data Type | What to Get | Python Tool |
|---|---|---|
| Stock Market (S&P 500) | Historical prices of `^GSPC`, `EL` (Estée Lauder), `LVMUY` (LVMH) | `yfinance` |
| Consumer Staples ETF | `XLP` — tracks everyday consumer goods | `yfinance` |
| Google Search Interest | Keywords: "Lipstick", "Nail Polish", "Louis Vuitton", "Designer Handbag" | `pytrends` |
| Inflation (Optional Bonus) | US CPI data | `pandas-datareader` or FRED API |

---

## 🛠️ Python Libraries You'll Need

```bash
pip install yfinance pytrends pandas matplotlib seaborn plotly scipy
```

| Library | Purpose |
|---|---|
| `yfinance` | Download stock/ETF historical prices |
| `pytrends` | Google Trends data via Python |
| `pandas` | Data manipulation, alignment, cleaning |
| `matplotlib` / `seaborn` | Static charts, heatmaps |
| `plotly` | Interactive dashboards |
| `scipy.stats` | Pearson correlation coefficient |

---

## 📅 2-Week Sprint Plan

---

### 📦 WEEK 1 — Data Collection & Cleaning

---

#### **Day 1–2 | Setup & Data Acquisition**

**Goal:** Get ALL raw data downloaded and saved as CSV files.

**Tasks:**
- [ OK ] Set up Python environment (virtualenv or conda)
- [ OK ] Install all libraries
- [ OK ] Write `data_collection.py` — pulls stock + Google Trends data
- [ OK ] Save raw CSVs: `sp500_raw.csv`, `trends_raw.csv`, `luxury_stocks_raw.csv`

**Code — Stock Data Collection:**
```python
import yfinance as yf
import pandas as pd

# Download S&P 500 and relevant stocks
tickers = {
    'SP500': '^GSPC',
    'Estee_Lauder': 'EL',
    'LVMH': 'LVMUY',
    'Consumer_Staples': 'XLP'
}

all_data = {}
for name, ticker in tickers.items():
    df = yf.download(ticker, start='2004-01-01', end='2024-12-31')
    all_data[name] = df['Close']

stocks_df = pd.DataFrame(all_data)
stocks_df.to_csv('data/sp500_raw.csv')
print(stocks_df.head())
```

**Code — Google Trends Collection:**
```python
from pytrends.request import TrendReq
import time

pytrends = TrendReq(hl='en-US', tz=360)

# Affordable luxury keywords
affordable = ['Lipstick', 'Nail Polish', 'Mascara']
# Big luxury keywords (control group)
expensive = ['Louis Vuitton', 'Rolex', 'Designer Handbag']

all_keywords = affordable + expensive

pytrends.build_payload(all_keywords, timeframe='2004-01-01 2024-12-31', geo='US')
trends_df = pytrends.interest_over_time()
trends_df.drop(columns=['isPartial'], inplace=True)
trends_df.to_csv('data/trends_raw.csv')
print(trends_df.head())
```

> [!TIP]
> `pytrends` can get rate-limited. Add `time.sleep(2)` between API calls if you're fetching multiple keyword groups.

---

#### **Day 3–4 | Data Cleaning & Alignment**

**Goal:** Make stock data and trend data speak the same language (same timeframe, same frequency).

**The Core Problem:** Google Trends gives *weekly* data. Stock prices are *daily*. You must align them.

**Code — Cleaning & Alignment:**
```python
import pandas as pd

# Load raw data
stocks = pd.read_csv('data/sp500_raw.csv', index_col=0, parse_dates=True)
trends = pd.read_csv('data/trends_raw.csv', index_col=0, parse_dates=True)

# Step 1: Resample stocks to WEEKLY (to match trends)
stocks_weekly = stocks.resample('W').mean()

# Step 2: Forward-fill weekends/holidays in stock data
stocks_weekly = stocks_weekly.ffill()

# Step 3: Normalize both datasets to 0-100 scale (Min-Max)
def normalize(df):
    return (df - df.min()) / (df.max() - df.min()) * 100

stocks_norm = normalize(stocks_weekly)
trends_norm = normalize(trends)

# Step 4: Merge on a common date index
merged = pd.merge(stocks_norm, trends_norm, left_index=True, right_index=True, how='inner')
merged.to_csv('data/cleaned_merged.csv')
print(f"Dataset shape: {merged.shape}")
print(merged.head())
```

**Key Concepts to Understand:**
- **`resample('W').mean()`** — converts daily → weekly by averaging
- **`.ffill()`** — forward-fills NaN gaps (weekends/holidays)
- **Min-Max Normalization** — puts everything on a 0–100 scale so you can compare apples to apples

---

#### **Day 5 | Exploratory Data Analysis (EDA)**

**Goal:** Visually explore relationships BEFORE doing any math.

**Code — EDA Plots:**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

merged = pd.read_csv('data/cleaned_merged.csv', index_col=0, parse_dates=True)

# Plot 1: Dual-axis chart — Lipstick searches vs S&P 500
fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()

ax1.plot(merged.index, merged['Lipstick'], color='pink', label='Lipstick Searches', linewidth=2)
ax2.plot(merged.index, merged['SP500'], color='navy', label='S&P 500', linewidth=2, alpha=0.7)

ax1.set_ylabel('Search Interest (Normalized)', color='pink')
ax2.set_ylabel('S&P 500 (Normalized)', color='navy')
ax1.set_title('The Lipstick Effect: Search Interest vs. Market Performance')
plt.tight_layout()
plt.savefig('visuals/lipstick_vs_sp500.png', dpi=150)
plt.show()
```

---

### 🔬 WEEK 2 — Analysis, Modeling & Visualization

---

#### **Day 6–7 | Correlation Analysis**

**Goal:** Quantify the relationship — does the lipstick index actually correlate with market downturns?

**Code — Pearson Correlation:**
```python
from scipy import stats
import pandas as pd

merged = pd.read_csv('data/cleaned_merged.csv', index_col=0, parse_dates=True)

# Calculate Pearson correlation: Lipstick searches vs S&P 500
corr, pvalue = stats.pearsonr(merged['Lipstick'].dropna(), merged['SP500'].dropna())
print(f"Pearson Correlation: {corr:.4f}")
print(f"P-Value: {pvalue:.4f}")

if pvalue < 0.05:
    print("✅ Statistically significant correlation found!")
else:
    print("❌ No statistically significant correlation.")

# Full correlation matrix
corr_matrix = merged.corr()
print(corr_matrix)
```

**Interpreting Results:**
| Correlation Value | Meaning |
|---|---|
| -0.7 to -1.0 | Strong **negative** correlation (lipstick up, market down ✅ proves theory) |
| -0.3 to -0.7 | Moderate negative correlation |
| 0 to ±0.3 | Little to no correlation |

---

#### **Day 8–9 | Rolling Averages & Lead-Lag Analysis**

**Goal:** Smooth out seasonal noise AND test if searches *predict* market moves (or vice versa).

**Code — Rolling Averages:**
```python
# 12-week rolling average to remove noise (Christmas, Black Friday spikes)
merged['Lipstick_smooth'] = merged['Lipstick'].rolling(window=12).mean()
merged['SP500_smooth'] = merged['SP500'].rolling(window=12).mean()
```

**Code — Lead-Lag Cross-Correlation:**
```python
import numpy as np

# Cross-correlation: does lipstick PREDICT the market, or follow it?
def cross_correlation(series1, series2, max_lag=26):
    results = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = series1[:lag].corr(series2[-lag:])
        elif lag > 0:
            corr = series1[lag:].corr(series2[:-lag])
        else:
            corr = series1.corr(series2)
        results[lag] = corr
    return pd.Series(results)

lag_corr = cross_correlation(merged['Lipstick_smooth'].dropna(),
                              merged['SP500_smooth'].dropna())
best_lag = lag_corr.abs().idxmax()
print(f"Best lag: {best_lag} weeks | Correlation: {lag_corr[best_lag]:.4f}")
```

> [!NOTE]
> If `best_lag` is **negative** (e.g., -3), it means lipstick searches spike **3 weeks BEFORE** the market drops — that's predictive power!

---

#### **Day 10–11 | Build the Final Dashboard**

**Goal:** Create a polished, interactive Plotly dashboard (or clean Seaborn report).

**Code — Plotly Interactive Dashboard:**
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

merged = pd.read_csv('data/cleaned_merged.csv', index_col=0, parse_dates=True)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        'Lipstick Index vs S&P 500',
        'Affordable vs Luxury Search Trends',
        'Correlation Heatmap',
        'Lead-Lag Cross-Correlation'
    ]
)

# Chart 1: Dual line
fig.add_trace(go.Scatter(x=merged.index, y=merged['Lipstick'],
              name='Lipstick', line=dict(color='hotpink')), row=1, col=1)
fig.add_trace(go.Scatter(x=merged.index, y=merged['SP500'],
              name='S&P 500', line=dict(color='navy')), row=1, col=1)

# Chart 2: Affordable vs Luxury
fig.add_trace(go.Scatter(x=merged.index, y=merged[['Lipstick','Nail Polish','Mascara']].mean(axis=1),
              name='Affordable Avg', line=dict(color='pink')), row=1, col=2)
fig.add_trace(go.Scatter(x=merged.index, y=merged[['Louis Vuitton','Rolex']].mean(axis=1),
              name='Luxury Avg', line=dict(color='gold')), row=1, col=2)

fig.update_layout(title='The Lipstick Effect Dashboard', height=700, template='plotly_dark')
fig.write_html('visuals/dashboard.html')
fig.show()
```

---

#### **Day 12–13 | Heatmap & Scatter Plot**

**Code — Correlation Heatmap:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = merged.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5)
plt.title('Keyword & Market Correlation Matrix')
plt.tight_layout()
plt.savefig('visuals/heatmap.png', dpi=150)
plt.show()
```

**Code — Scatter Plot with Regression:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.regplot(x=merged['SP500'], y=merged['Lipstick'],
            scatter_kws={'alpha': 0.3, 'color': 'hotpink'},
            line_kws={'color': 'darkred'})
plt.xlabel('S&P 500 (Normalized)')
plt.ylabel('Lipstick Search Interest (Normalized)')
plt.title('Economic Stress vs. Lipstick Index')
plt.savefig('visuals/scatter_regression.png', dpi=150)
plt.show()
```

---

#### **Day 14 | Final Writeup & Presentation Prep**

**Goal:** Package everything into a clean, presentable format.

**Deliverables Checklist:**
- [ ] `data_collection.py` — clean, documented
- [ ] `data_cleaning.py` — clean, documented
- [ ] `analysis.py` — correlation results printed and saved
- [ ] `dashboard.html` — interactive Plotly dashboard
- [ ] `visuals/` folder — all PNG charts
- [ ] `README.md` — project description, key findings, how to run
- [ ] Findings Summary (1 page) — write your conclusion: **"The data shows X about the Lipstick Effect..."**

---

## 📁 Recommended Folder Structure

```
Finance Project/
│
├── data/
│   ├── sp500_raw.csv
│   ├── trends_raw.csv
│   └── cleaned_merged.csv
│
├── visuals/
│   ├── lipstick_vs_sp500.png
│   ├── heatmap.png
│   ├── scatter_regression.png
│   └── dashboard.html
│
├── data_collection.py
├── data_cleaning.py
├── analysis.py
├── visualization.py
└── README.md
```

---

## 📚 Key Things to Explore & Learn

| Topic | Resource |
|---|---|
| Google Trends + Python | [YouTube: pytrends tutorial](https://www.youtube.com/watch?v=c8CCeWUXwMo) (from your proposal) |
| yfinance basics | `yfinance` PyPI docs — search "yfinance download" |
| Pandas time series | `pd.resample()`, `.rolling()`, `.corr()` |
| Pearson vs Spearman correlation | Use Pearson for normally distributed; Spearman if not |
| Plotly dashboards | `plotly.express` docs — very beginner friendly |
| The Lipstick Theory explained | Investopedia: "Lipstick Effect" — great background reading |

---

## ⚡ Quick Weekly Summary

```
Week 1: GET THE DATA RIGHT
  Day 1-2: Download stocks & Google Trends → Save as CSVs
  Day 3-4: Clean & align (weekly frequency, normalized)
  Day 5:   EDA — plot everything, look for patterns

Week 2: PROVE THE THEORY
  Day 6-7:  Pearson correlation — is there a relationship?
  Day 8-9:  Rolling averages + Lead-Lag test
  Day 10-11: Build Plotly interactive dashboard
  Day 12-13: Heatmap + Scatter+Regression charts
  Day 14:   Final packaging & writeup
```

> [!TIP]
> Start with **just 3 years of data** (2020–2023) to prototype fast. Once your scripts work, extend to 2004–2024 for the full picture, which will capture the **2008 Financial Crisis** and **COVID-19** — the two most dramatic recession events that will make your data compelling.
