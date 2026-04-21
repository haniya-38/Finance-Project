import plotly.graph_objects as go
from config import C, DEFENSIVE, CYCLICAL

_LY = dict(  # shared Plotly layout (single-axis charts)
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C["plot_bg"],
    font=dict(family="Inter,sans-serif", color="#c9d4e8"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Normalized (0–100)"),
    legend=dict(orientation="h", x=0, xanchor="left", y=1.05, yanchor="bottom", bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    margin=dict(l=50, r=20, t=40, b=45), hovermode="x unified",
)

def chart_A(dn, s):
    """Chart A: Defensive Goods vs Inflation."""
    fig = go.Figure()
    for col, clr in zip(DEFENSIVE, ["#4ade80", "#86efac"]):
        fig.add_trace(go.Scatter(x=dn.index, y=dn[col], name=col.replace("_", " "),
                                 line=dict(color=clr, width=1.5, dash="dot"), opacity=0.6))
    fig.add_trace(go.Scatter(x=s.index, y=s["Defensive"], name="Defensive Avg",
                             line=dict(color=C["def_line"], width=3),
                             fill="tozeroy", fillcolor=C["def_fill"]))
    fig.add_trace(go.Scatter(x=s.index, y=s["Inflation"], name="Inflation",
                             line=dict(color=C["inf_line"], width=2.5)))
    fig.update_layout(**_LY)
    return fig

def chart_B(dn, s):
    """Chart B: Cyclical Goods vs Inflation."""
    fig = go.Figure()
    for col, clr in zip(CYCLICAL, ["#f87171", "#fca5a5"]):
        fig.add_trace(go.Scatter(x=dn.index, y=dn[col], name=col.replace("_", " "),
                                 line=dict(color=clr, width=1.5, dash="dot"), opacity=0.6))
    fig.add_trace(go.Scatter(x=s.index, y=s["Cyclical"], name="Cyclical Avg",
                             line=dict(color=C["cyc_line"], width=3),
                             fill="tozeroy", fillcolor=C["cyc_fill"]))
    fig.add_trace(go.Scatter(x=s.index, y=s["Inflation"], name="Inflation",
                             line=dict(color=C["inf_line"], width=2.5)))
    fig.update_layout(**_LY)
    return fig

def chart_C(fc):
    """Chart C: 24-Month Inflation Forecast via OLS Linear Regression."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc["dates"], y=fc["actual"], name="Actual CPI (%)",
                             line=dict(color=C["inf_line"], width=2.5)))
    fig.add_trace(go.Scatter(x=fc["dates"], y=fc["trend"], name="OLS Trend Line",
                             line=dict(color=C["trend"], width=1.8, dash="dot"), opacity=0.8))
    fig.add_trace(go.Scatter(x=fc["fdates"], y=fc["values"], name="24-Month Forecast",
                             line=dict(color=C["forecast"], width=3, dash="dash")))
    fig.add_vrect(x0=str(fc["dates"][-1]), x1=str(fc["fdates"][-1]),
                  fillcolor="rgba(239,68,68,0.07)", layer="below", line_width=0)
    ly = {**_LY, "yaxis": dict(gridcolor="rgba(255,255,255,0.05)", title="Inflation Rate (%)")}
    fig.update_layout(**ly)
    return fig

def chart_D(s):
    """Chart D: Defensive/Cyclical Ratio vs Inflation — the Lipstick Effect proof."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s["Ratio"], name="Def/Cyc Ratio (Lipstick Proxy)",
                             line=dict(color=C["ratio_line"], width=3),
                             fill="tozeroy", fillcolor=C["ratio_fill"]))
    fig.add_trace(go.Scatter(x=s.index, y=s["Inflation"], name="Inflation Rate",
                             line=dict(color=C["inf_line"], width=2.5)))
    fig.update_layout(**_LY)
    return fig
