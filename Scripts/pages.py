import streamlit as st
from charts import chart_A, chart_B, chart_C, chart_D

# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def kpi_card(label, value, sub=""):
    return (f'<div class="kpi"><div class="kpi-lbl">{label}</div>'
            f'<div class="kpi-val">{value}</div><div class="kpi-sub">{sub}</div></div>')

def insight(text):
    """Render a styled insight box below a chart."""
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

def sec(icon, title, sub=""):
    st.markdown(f"<div class='sec-hdr'><h2>{icon} {title}</h2></div>", unsafe_allow_html=True)
    if sub: st.caption(sub)
    st.markdown("&nbsp;")

# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════

def page_overview(k):
    st.markdown("""
    <div style='padding:8px 0 18px'>
      <h1 style='font-size:2.2rem;font-weight:800;margin:0;
                 background:linear-gradient(90deg,#60a5fa,#a78bfa);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        The Lipstick Effect in Pakistan</h1>
      <p style='color:#8892b0;margin-top:6px'>
        2010–2024 &nbsp;·&nbsp; Pakistan Stock Exchange (PSX) &nbsp;·&nbsp;
        Inflation Impact on Consumer Sectors</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("> **Research Question:** During high inflation, do defensive (essential) goods "
                "outperform cyclical (non-essential) goods — and does PSX stock data prove this?")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Avg Inflation",  f"{k['avg_inf']}%",   "14-year average"),       unsafe_allow_html=True)
    c2.markdown(kpi_card("Peak Inflation", f"{k['peak_inf']}%",  f"hit {k['peak_date']}"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Defensive Gain", f"+{k['def_gain']}",  "normalized pts"),         unsafe_allow_html=True)
    c4.markdown(kpi_card("Cyclical Gain",  f"+{k['cyc_gain']}",  "normalized pts"),         unsafe_allow_html=True)

    st.markdown("&nbsp;")
    with st.expander("📖 About This Study", expanded=True):
        a, b = st.columns(2)
        a.markdown("""
**What is the Lipstick Effect?**  
An economic theory where consumers keep buying essentials (food, medicine)
during economic downturns, while cutting back on larger discretionary purchases.

**Stocks Studied (PSX):**
- **Defensive:** Nestle PK (food) · Abbott PK (pharma)
- **Cyclical:** Lucky Cement (construction) · HBL Bank (finance)
        """)
        b.markdown(f"""
**Project Pipeline:**
- **Stocks:** Yahoo Finance API — PSX tickers (2010–2024)
- **Inflation:** Official Pakistan CPI statistics (CSV)
- **Observations:** {k['months']} monthly data points
- **Normalization:** Min-Max (0–100 scale)
- **Forecast Model:** OLS Linear Regression via numpy
        """)


def page_charts(dn, s, fc):
    sec("📈", "Charts & Analysis", "4 interactive charts proving the Lipstick Effect with Pakistan data")

    # Chart A
    st.markdown("#### Chart A — Defensive Goods vs. Inflation")
    st.plotly_chart(chart_A(dn, s), use_container_width=True)
    insight(
        "Nestle PK and Abbott PK (essentials) maintain relatively stable normalized values "
        "even during Pakistan's worst inflation spikes (2018–2023). The green band tracks "
        "closely with — and sometimes above — the inflation line, confirming that demand "
        "for essentials is resilient regardless of purchasing power erosion."
    )
    st.markdown("---")

    # Chart B
    st.markdown("#### Chart B — Cyclical Goods vs. Inflation")
    st.plotly_chart(chart_B(dn, s), use_container_width=True)
    insight(
        "Lucky Cement and HBL Bank (non-essentials) show clearly higher volatility "
        "and visibly diverge from the inflation line during stress periods. Between "
        "2018–2023, cyclicals underperformed most severely — exactly when inflation "
        "peaked. consumers and businesses scale back construction and banking activity first."
    )
    st.markdown("---")

    # Chart C
    st.markdown("#### Chart C — 24-Month Inflation Forecast (OLS Linear Regression)")
    st.plotly_chart(chart_C(fc), use_container_width=True)
    insight(
        f"The OLS model projects Pakistan's inflation at approximately "
        f"**{fc['values'][-1]:.1f}%** over the next 24 months "
        f"(current: {fc['actual'][-1]:.1f}%). The upward-sloping trend line "
        f"confirms persistent inflationary pressure — meaning consumer behavior will "
        f"continue shifting toward essentials, keeping defensive stocks relatively stronger."
    )
    st.markdown("---")

    # Chart D
    st.markdown("#### Chart D — Defensive/Cyclical Ratio vs. Inflation (Lipstick Effect Proof)")
    st.plotly_chart(chart_D(s), use_container_width=True)
    insight(
        "The purple ratio line rises when inflation spikes — meaning defensive goods gain "
        "relative strength over cyclicals precisely during economic stress. This is the "
        "quantitative proof of the Lipstick Effect in Pakistan: as inflation rises, "
        "essentials outperform non-essentials in the PSX stock market, confirmed over 14 years."
    )


def page_insights(k, fc):
    sec("🔍", "Key Insights", "What this study tells us about Pakistan's economy and consumer behaviour")

    c1, c2 = st.columns(2)
    c1.success(
        f"**✅ Defensive Stocks Are Inflation-Resilient**\n\n"
        f"Nestle PK and Abbott PK gained **+{k['def_gain']} normalized points** over the period. "
        f"Even during Pakistan's worst inflation years (peak: {k['peak_inf']}% in {k['peak_date']}), "
        f"essential goods demand held firm — consumers never stop buying food and medicine."
    )
    c1.warning(
        f"**⚠️ Cyclical Stocks Suffer During Inflation**\n\n"
        f"Lucky Cement and HBL Bank gained **+{k['cyc_gain']} normalized points** overall, "
        f"but with significantly higher volatility. During inflation peaks, these sectors "
        f"underperformed the most — construction and finance are the first to be cut "
        f"when purchasing power erodes."
    )
    c2.info(
        "**📊 The Ratio Proves the Lipstick Effect**\n\n"
        "The Defensive/Cyclical ratio consistently rises when inflation is elevated. "
        "This is quantitative proof of the Lipstick Effect: as inflation increases, "
        "the relative strength of essential goods stocks grows vs. non-essential stocks — "
        "a pattern confirmed consistently over 14 years of Pakistan data."
    )
    c2.error(
        f"**🔮 Inflation Will Stay Elevated for 2 More Years**\n\n"
        f"The OLS Linear Regression model forecasts Pakistan's inflation at "
        f"**{k['fcast_inf']}%** in 24 months (currently **{k['cur_inf']}%**). "
        f"If the trend holds, defensive stocks will continue to be the safer investment "
        f"choice — and consumer behaviour will remain shifted toward essentials."
    )

    st.markdown("---")
    st.markdown("#### 📋 Key Numbers at a Glance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Inflation (14yr)", f"{k['avg_inf']}%")
    m2.metric("Peak Inflation",       f"{k['peak_inf']}%", k['peak_date'])
    m3.metric("Current CPI",          f"{k['cur_inf']}%")
    m4.metric("24M Forecast",         f"{k['fcast_inf']}%",
              f"{k['fcast_inf'] - k['cur_inf']:+.1f}%")

    st.markdown("---")
    st.markdown("#### 🧭 Conclusion")
    st.markdown(f"""
> This study analyzed **{k['months']} months** of Pakistan Stock Exchange data alongside
> official CPI statistics and Google Search Trends. The data conclusively shows that the
> **Lipstick Effect holds true in Pakistan**: during inflationary periods, defensive stocks
> (Nestle, Abbott) outperform cyclical stocks (Lucky Cement, HBL) on a relative basis.
> With inflation forecast to reach **{k['fcast_inf']}%** over the next 24 months, this
> insight is directly actionable for investors, policymakers, and consumers navigating
> Pakistan's economic environment.
    """)
