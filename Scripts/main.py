"""
Pakistan Finance — The Lipstick Effect (2010–2024)
===================================================
Run:  streamlit run Scripts/main.py   (from project root)
  or  streamlit run main.py           (from inside Scripts/ folder)

Project: How inflation shifts consumer behavior from cyclical to defensive goods.
"""

import streamlit as st

# ── Page config (must be FIRST Streamlit call) ───────────────────────────────
st.set_page_config(page_title="Pakistan Inflation Dashboard", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

# Import modular components
from config import CSS_STYLES
st.markdown(CSS_STYLES, unsafe_allow_html=True)

from data import load_data, sector_df, ols_forecast, compute_kpis
from pages import page_overview, page_charts, page_insights

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR + ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def sidebar():
    with st.sidebar:
        st.markdown("## 📊 Pakistan Finance\n### The Lipstick Effect")
        st.markdown("---")
        st.markdown(
            "**Period:** 2010 – 2024\n\n"
            "**Sources:**\n- PSX via Yahoo Finance\n- Google Trends (PK)\n- Official CPI CSV\n\n"
            "**Model:** OLS Linear Regression  \n"
            "**Forecast:** 24 months"
        )
        st.markdown("---")
        active_page = st.radio("", ["📋 Overview", "📈 Charts & Analysis", "🔍 Key Insights"],
                               label_visibility="collapsed")
        st.markdown("---")
        st.caption("Built with Streamlit + Plotly")
    return active_page


def main():
    # Load all data (cached — instant after first run)
    with st.spinner("Loading data..."):
        data, data_norm = load_data()
        s  = sector_df(data_norm)
        fc = ols_forecast(data)
        k  = compute_kpis(data, data_norm, fc)

    active_page = sidebar()

    if   active_page == "📋 Overview":          page_overview(k)
    elif active_page == "📈 Charts & Analysis": page_charts(data_norm, s, fc)
    elif active_page == "🔍 Key Insights":      page_insights(k, fc)


if __name__ == "__main__":
    main()
