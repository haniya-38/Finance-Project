# Constants and configuration values for the dashboard

STOCKS_FILE    = "data/stocks_raw.csv"
TRENDS_FILE    = "data/trends_raw.csv"
INFLATION_FILE = "data/pakistan_inflation_actual.csv"

TICKERS = {
    "Nestle_PK": "NESTLE.KA", "Abbott_PK": "ABOT.KA", "National_Foods": "NATF.KA",
    "Lucky_Cement": "LUCK.KA", "HBL_Bank": "HBL.KA", "PSO": "PSO.KA", "OGDC": "OGDC.KA",
}
DEFENSIVE     = ["Nestle_PK", "Abbott_PK"]
CYCLICAL      = ["Lucky_Cement", "HBL_Bank"]
INFLATION_COL = "Actual_Inflation_Rate"

# Colors for charts
C = dict(
    def_line="#22c55e",  def_fill="rgba(34,197,94,0.08)",
    cyc_line="#ef4444",  cyc_fill="rgba(239,68,68,0.08)",
    inf_line="#f8fafc",  ratio_line="#a78bfa", ratio_fill="rgba(167,139,250,0.1)",
    forecast="#f87171",  trend="#60a5fa",      plot_bg="rgba(15,22,35,0.6)",
)

# Custom CSS Styles
CSS_STYLES = """
<style>
.kpi{background:linear-gradient(135deg,#1a1f2e,#252b3b);border:1px solid #2d3550;
     border-radius:14px;padding:20px 16px;text-align:center;margin-bottom:6px}
.kpi-lbl{font-size:11px;color:#8892b0;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:6px}
.kpi-val{font-size:32px;font-weight:700;color:#e6f0ff;line-height:1}
.kpi-sub{font-size:11px;color:#64748b;margin-top:5px}
.insight-box{background:#1a1f2e;border-left:3px solid #3b82f6;border-radius:8px;
             padding:14px 16px;margin:10px 0;font-size:0.88rem;color:#c9d4e8;line-height:1.6}
.sec-hdr{border-left:4px solid #3b82f6;padding-left:14px;margin:22px 0 4px}
section[data-testid="stSidebar"]{background:#0f1623}
</style>
"""
