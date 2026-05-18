"""
dashboard.py — CFO Copilot · Enterprise Financial Intelligence Platform.

Run: streamlit run src/dashboard.py
"""

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_ENV_FILE = _ROOT / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_ENV_FILE, override=True)
except ImportError:
    if _ENV_FILE.exists():
        for _line in _ENV_FILE.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(_ROOT))

from src.data_loader import load_financial_data
from src.analysis import (
    compute_kpis, kpis_to_dataframe, summarise_kpis,
    compute_financial_health_score, compute_dupoint,
    run_scenario_analysis, NAMED_SCENARIOS, INDUSTRY_BENCHMARKS,
)
from src.forecasting import run_all_forecasts, forecast_summary
from src.llm_assistant import CFOAssistant, PROVIDER_INFO, get_provider_status

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CFO Copilot · EY Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Theme ───────────────────────────────────────────────────────────────────

C = {
    "navy":    "#1B2A4A",
    "blue":    "#2563EB",
    "teal":    "#0891B2",
    "green":   "#16A34A",
    "red":     "#DC2626",
    "amber":   "#D97706",
    "purple":  "#7C3AED",
    "gray":    "#6B7280",
    "light":   "#F1F5F9",
    "white":   "#FFFFFF",
}

SEVERITY_COLOR = {
    "critical": C["red"],
    "high":     C["amber"],
    "medium":   "#F59E0B",
    "low":      C["teal"],
    "positive": C["green"],
    "neutral":  C["blue"],
    "warning":  C["amber"],
}

st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background: #1B2A4A !important; }
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    [data-testid="stSidebar"] .stRadio label { color: #CBD5E1 !important; }
    [data-testid="stSidebar"] hr { border-color: #334155 !important; }

    /* Metric cards */
    [data-testid="stMetric"] { background:#F8FAFC; border-radius:10px; padding:12px 16px; }
    [data-testid="stMetricLabel"] { font-size:0.78rem; color:#64748B; font-weight:600; text-transform:uppercase; letter-spacing:.05em; }
    [data-testid="stMetricValue"] { font-size:1.6rem; font-weight:700; color:#1B2A4A; }

    /* Tab strip */
    .stTabs [data-baseweb="tab-list"] { gap:4px; }
    .stTabs [data-baseweb="tab"] { background:#EFF6FF; border-radius:6px 6px 0 0; padding:8px 16px; }
    .stTabs [aria-selected="true"] { background:#2563EB !important; color:white !important; }

    /* Alert boxes */
    .alert-critical { background:#FEE2E2; border-left:4px solid #DC2626; padding:10px 14px; border-radius:6px; margin:4px 0; }
    .alert-high     { background:#FEF3C7; border-left:4px solid #D97706; padding:10px 14px; border-radius:6px; margin:4px 0; }
    .alert-medium   { background:#FFF7ED; border-left:4px solid #F59E0B; padding:10px 14px; border-radius:6px; margin:4px 0; }
    .alert-low      { background:#ECFDF5; border-left:4px solid #16A34A; padding:10px 14px; border-radius:6px; margin:4px 0; }

    /* Section headers */
    .section-header { font-size:1.05rem; font-weight:700; color:#1B2A4A; margin-bottom:4px; }

    /* Hide streamlit hamburger */
    #MainMenu, footer { visibility: hidden; }

    /* Score badge */
    .grade-badge { display:inline-block; padding:4px 14px; border-radius:20px; font-weight:800; font-size:1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_all():
    datasets = load_financial_data("data")
    income   = datasets["income_statement"]
    balance  = datasets["balance_sheet"]
    cf       = datasets["cash_flow"]

    kpi_report  = compute_kpis(income, balance, cf)
    kpi_df      = kpis_to_dataframe(kpi_report)
    kpi_summary = summarise_kpis(kpi_report)
    forecasts   = run_all_forecasts(income, cf, horizon=4, strategy="auto")
    fc_df       = forecast_summary(forecasts)
    health      = compute_financial_health_score(kpi_report)
    dupoint     = compute_dupoint(income, balance)

    return datasets, kpi_report, kpi_df, kpi_summary, forecasts, fc_df, health, dupoint


# ─── Chart Helpers ────────────────────────────────────────────────────────────

def _base_layout(height=380, **kw) -> dict:
    return dict(
        height=height,
        plot_bgcolor=C["white"],
        paper_bgcolor=C["white"],
        margin=dict(l=10, r=10, t=36, b=10),
        font=dict(family="Inter, sans-serif", size=12, color="#374151"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **kw,
    )


def chart_revenue(income: pd.DataFrame, forecasts: dict) -> go.Figure:
    periods = [str(p) for p in income.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=periods, y=income["revenue"] / 1e6, name="Revenue",
                         marker_color=C["blue"], opacity=0.85))
    fig.add_trace(go.Scatter(x=periods, y=income["ebitda"] / 1e6, name="EBITDA",
                             mode="lines+markers", line=dict(color=C["green"], width=2.5)))
    if "revenue" in forecasts:
        fc = forecasts["revenue"]
        fc_x = [str(p) for p in fc.forecast.index]
        fig.add_trace(go.Scatter(x=fc_x, y=fc.forecast.values / 1e6, name="Rev Forecast",
                                 mode="lines+markers", line=dict(color=C["amber"], dash="dash", width=2)))
        fig.add_trace(go.Scatter(
            x=fc_x + fc_x[::-1],
            y=list(fc.confidence_upper / 1e6) + list(fc.confidence_lower / 1e6)[::-1],
            fill="toself", fillcolor="rgba(217,119,6,0.12)", line=dict(color="rgba(0,0,0,0)"), name="80% CI",
        ))
    fig.update_layout(title="Revenue & EBITDA + Forecast ($M)", yaxis_title="$M",
                      **_base_layout(400))
    return fig


def chart_margins(kpi_df: pd.DataFrame, benchmark: dict | None = None) -> go.Figure:
    periods = [str(p) for p in kpi_df.index]
    fig = go.Figure()
    for col, color, label in [
        ("gross_margin",       C["teal"],  "Gross Margin"),
        ("ebitda_margin",      C["green"], "EBITDA Margin"),
        ("net_profit_margin",  C["blue"],  "Net Margin"),
    ]:
        if col in kpi_df.columns:
            fig.add_trace(go.Scatter(x=periods, y=kpi_df[col] * 100, name=label,
                                     mode="lines+markers", line=dict(color=color, width=2.5)))
    if benchmark:
        for col, color, label in [
            ("ebitda_margin", C["gray"], "Benchmark EBITDA"),
            ("gross_margin",  "#94A3B8", "Benchmark Gross"),
        ]:
            if col in benchmark:
                fig.add_hline(y=benchmark[col]*100, line_dash="dot", line_color=color,
                              annotation_text=f"⌀ {label}: {benchmark[col]*100:.0f}%",
                              annotation_position="right")
    fig.update_layout(title="Margin Evolution (%)", yaxis_title="%", **_base_layout(360))
    return fig


def chart_cashflow(cash_flow: pd.DataFrame, forecasts: dict) -> go.Figure:
    periods = [str(p) for p in cash_flow.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=periods, y=cash_flow["operating_cash_flow"] / 1e6,
                         name="Operating CF", marker_color=C["teal"]))
    fig.add_trace(go.Bar(x=periods, y=cash_flow["investing_cash_flow"] / 1e6,
                         name="Investing CF", marker_color=C["red"], opacity=0.8))
    fig.add_trace(go.Bar(x=periods, y=cash_flow["financing_cash_flow"] / 1e6,
                         name="Financing CF", marker_color=C["gray"], opacity=0.8))
    if "operating_cash_flow" in forecasts:
        fc = forecasts["operating_cash_flow"]
        fc_x = [str(p) for p in fc.forecast.index]
        fig.add_trace(go.Scatter(x=fc_x, y=fc.forecast.values / 1e6, name="OCF Forecast",
                                 mode="lines+markers", line=dict(color=C["amber"], dash="dash", width=2)))
    fig.update_layout(title="Cash Flow by Category + OCF Forecast ($M)", barmode="group",
                      yaxis_title="$M", **_base_layout(370))
    return fig


def chart_ratios(kpi_df: pd.DataFrame) -> go.Figure:
    periods = [str(p) for p in kpi_df.index]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periods, y=kpi_df["current_ratio"], name="Current Ratio",
                             line=dict(color=C["teal"], width=2.5), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=periods, y=kpi_df["debt_to_equity"], name="Debt / Equity",
                             line=dict(color=C["red"], width=2.5), mode="lines+markers"))
    fig.add_hline(y=1.2, line_dash="dot", line_color=C["amber"],
                  annotation_text="Current Ratio Floor 1.2x")
    fig.add_hline(y=2.0, line_dash="dot", line_color=C["red"],
                  annotation_text="D/E Ceiling 2.0x")
    fig.update_layout(title="Liquidity & Leverage Ratios", yaxis_title="Ratio (x)",
                      **_base_layout(360))
    return fig


def chart_health_gauge(health) -> go.Figure:
    color = health.color
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health.overall,
        number={"suffix": "/100", "font": {"size": 36, "color": C["navy"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": C["gray"]},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#E2E8F0",
            "steps": [
                {"range": [0, 35],  "color": "#FEE2E2"},
                {"range": [35, 50], "color": "#FEF3C7"},
                {"range": [50, 65], "color": "#FFF7ED"},
                {"range": [65, 80], "color": "#ECFDF5"},
                {"range": [80, 100],"color": "#D1FAE5"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": health.overall,
            },
        },
        title={"text": f"Financial Health Score<br><span style='font-size:18px;color:{color}'>Grade {health.grade}</span>",
               "font": {"size": 16, "color": C["navy"]}},
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10),
                      paper_bgcolor=C["white"])
    return fig


def chart_health_breakdown(health) -> go.Figure:
    categories = ["Profitability\n(25)", "Liquidity\n(20)", "Leverage\n(20)",
                  "Growth\n(20)", "Cash Quality\n(15)"]
    scores = [health.profitability_score, health.liquidity_score, health.leverage_score,
              health.growth_score, health.cash_quality_score]
    maxes  = [25, 20, 20, 20, 15]
    colors = [C["green"] if s/m >= 0.7 else (C["amber"] if s/m >= 0.4 else C["red"])
              for s, m in zip(scores, maxes)]

    fig = go.Figure(go.Bar(
        x=categories, y=scores, marker_color=colors,
        text=[f"{s:.0f}" for s in scores], textposition="outside",
        customdata=maxes,
        hovertemplate="Score: %{y:.1f} / %{customdata}<extra></extra>",
    ))
    for i, (cat, mx) in enumerate(zip(categories, maxes)):
        fig.add_trace(go.Bar(x=[cat], y=[mx - scores[i]], marker_color="#E2E8F0",
                             showlegend=False,
                             hovertemplate=f"Remaining: {mx - scores[i]:.1f}<extra></extra>"))
    fig.update_layout(title="Score Breakdown by Dimension", barmode="stack",
                      yaxis_title="Points", showlegend=False, **_base_layout(300))
    return fig


def chart_dupoint(dupoint) -> go.Figure:
    periods = [str(p) for p in dupoint.net_profit_margin.index]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periods, y=dupoint.net_profit_margin * 100,
                             name="Net Margin (%)", mode="lines+markers",
                             line=dict(color=C["blue"], width=2.5),
                             yaxis="y"))
    fig.add_trace(go.Scatter(x=periods, y=dupoint.asset_turnover,
                             name="Asset Turnover (x)", mode="lines+markers",
                             line=dict(color=C["green"], width=2.5, dash="dot"),
                             yaxis="y2"))
    fig.add_trace(go.Scatter(x=periods, y=dupoint.equity_multiplier,
                             name="Equity Multiplier (x)", mode="lines+markers",
                             line=dict(color=C["amber"], width=2.5, dash="dash"),
                             yaxis="y2"))
    fig.update_layout(
        title="DuPont Decomposition: ROE = Net Margin × Asset Turnover × Equity Multiplier",
        yaxis=dict(title="Net Margin (%)", color=C["blue"]),
        yaxis2=dict(title="Ratio (x)", overlaying="y", side="right", color=C["green"]),
        **_base_layout(380),
    )
    return fig


def chart_scenario_waterfall(result) -> go.Figure:
    base = result.base
    scen = result.scenario

    rev_impact  = scen["revenue"] - base["revenue"]
    cogs_impact = -(scen["gross_profit"] - scen["revenue"] - (base["gross_profit"] - base["revenue"]))
    opex_impact = (base["ebitda"] - base["gross_profit"]) - (scen["ebitda"] - scen["gross_profit"])

    labels  = ["Base EBITDA", "Revenue Impact", "Gross Margin Impact", "OpEx Impact", "Scenario EBITDA"]
    values  = [base["ebitda"], rev_impact, cogs_impact, opex_impact, scen["ebitda"]]
    measure = ["absolute", "relative", "relative", "relative", "total"]
    text    = [f"${v/1e6:.2f}M" for v in values]

    fig = go.Figure(go.Waterfall(
        name="EBITDA Bridge",
        orientation="v",
        measure=measure,
        x=labels,
        y=[v / 1e6 for v in values],
        text=text,
        textposition="outside",
        connector=dict(line=dict(color=C["gray"], width=1, dash="dot")),
        increasing=dict(marker_color=C["green"]),
        decreasing=dict(marker_color=C["red"]),
        totals=dict(marker_color=C["navy"]),
    ))
    fig.update_layout(title=f"EBITDA Bridge — {result.name} Scenario ($M)",
                      yaxis_title="$M", **_base_layout(380))
    return fig


def chart_scenario_comparison(result) -> go.Figure:
    metrics = ["Revenue", "Gross Profit", "EBITDA", "Net Income"]
    base_v  = [result.base[k] / 1e6 for k in ["revenue", "gross_profit", "ebitda", "net_income"]]
    scen_v  = [result.scenario[k] / 1e6 for k in ["revenue", "gross_profit", "ebitda", "net_income"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Base", x=metrics, y=base_v, marker_color=C["blue"], opacity=0.9))
    fig.add_trace(go.Bar(name=result.name, x=metrics, y=scen_v,
                         marker_color=SEVERITY_COLOR.get(result.severity, C["amber"]), opacity=0.9))
    fig.update_layout(title="Base vs Scenario: P&L Comparison ($M)",
                      barmode="group", yaxis_title="$M", **_base_layout(360))
    return fig


def chart_benchmark_radar(latest: dict, benchmark: dict, industry: str) -> go.Figure:
    keys   = ["gross_margin", "ebitda_margin", "net_profit_margin", "operating_cash_flow_margin"]
    labels = ["Gross Margin", "EBITDA Margin", "Net Margin", "OCF Margin"]

    company_vals = [min(latest.get(k, 0) * 100, 100) for k in keys]
    bench_vals   = [benchmark.get(k, 0) * 100 for k in keys]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=company_vals + company_vals[:1], theta=labels + labels[:1],
                                  fill="toself", fillcolor="rgba(37,99,235,0.15)",
                                  line=dict(color=C["blue"], width=2.5), name="Company"))
    fig.add_trace(go.Scatterpolar(r=bench_vals + bench_vals[:1], theta=labels + labels[:1],
                                  fill="toself", fillcolor="rgba(107,114,128,0.10)",
                                  line=dict(color=C["gray"], width=2, dash="dash"), name=f"{industry} Median"))
    fig.update_layout(
        title=f"Margin Profile vs {industry} Benchmark",
        polar=dict(radialaxis=dict(visible=True, range=[0, 80])),
        **_base_layout(340),
    )
    return fig


# ─── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar(kpi_summary: dict, health):
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:10px 0 4px'>"
            "<span style='font-size:2rem'>📊</span><br>"
            "<span style='font-size:1.2rem;font-weight:800;color:#E2E8F0'>CFO Copilot</span><br>"
            "<span style='font-size:0.7rem;color:#94A3B8;letter-spacing:.1em'>ENTERPRISE FINANCIAL INTELLIGENCE</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Executive Dashboard", "KPI Analysis", "Scenario Analysis",
             "Forecasts", "Financial Health", "Anomalies & Risks", "AI CFO Assistant"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("<p style='font-size:.75rem;font-weight:700;color:#94A3B8;letter-spacing:.08em'>COMPANY PROFILE</p>",
                    unsafe_allow_html=True)
        company_name = st.text_input("Company name", value="Acme Corp", label_visibility="visible")
        industry     = st.selectbox("Industry sector", list(INDUSTRY_BENCHMARKS.keys()), index=0)

        st.markdown("---")
        # Status badges
        st.markdown("<p style='font-size:.75rem;font-weight:700;color:#94A3B8;letter-spacing:.08em'>DATA STATUS</p>",
                    unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        col_a.metric("Last Period", kpi_summary["latest_period"])
        col_b.metric("Alerts", f"{kpi_summary['anomaly_count']} ⚠️")

        grade_col = {"A": "#16A34A","B": "#65A30D","C": "#D97706","D": "#EA580C","F": "#DC2626"}
        g = health.grade
        st.markdown(
            f"<div style='text-align:center;margin:8px 0'>"
            f"<span style='font-size:.75rem;color:#94A3B8'>Health Score</span><br>"
            f"<span style='font-size:1.8rem;font-weight:800;color:{grade_col.get(g, C['amber'])}'>"
            f"{health.overall}/100 · {g}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        # AI Provider status
        st.markdown("<p style='font-size:.75rem;font-weight:700;color:#94A3B8;letter-spacing:.08em'>AI PROVIDERS</p>",
                    unsafe_allow_html=True)
        status = get_provider_status()
        free_ready = [p for p, ok in status.items() if ok and PROVIDER_INFO.get(p, {}).get("free")]
        for p in ["gemini", "groq", "mistral", "together", "openrouter"]:
            ok = status.get(p, False)
            icon = "🟢" if ok else "⚪"
            label = PROVIDER_INFO[p]["label"]
            st.markdown(
                f"<p style='font-size:.78rem;margin:2px 0'>{icon} {label}</p>",
                unsafe_allow_html=True,
            )
        if not free_ready:
            st.warning("No free provider configured. Add a key to .env")

    return page, company_name, industry


# ─── Pages ───────────────────────────────────────────────────────────────────

def page_executive(income, balance, cash_flow, kpi_df, kpi_summary, forecasts, health, industry):
    st.markdown(
        "<h2 style='color:#1B2A4A;margin-bottom:2px'>Executive Dashboard</h2>"
        "<p style='color:#64748B;margin-top:0'>Real-time financial overview & AI-powered insights</p>",
        unsafe_allow_html=True,
    )

    latest   = kpi_summary["latest"]
    prev_row = kpi_df.iloc[-2] if len(kpi_df) > 1 else kpi_df.iloc[-1]

    # ── Row 1 : Health gauge (left) + 10 KPI cards (right, 2 rows of 5) ─────
    c_gauge, c_kpis = st.columns([1.6, 3.4])
    with c_gauge:
        st.plotly_chart(chart_health_gauge(health), use_container_width=True)
        st.markdown(
            f"<p style='text-align:center;font-size:.8rem;color:{health.color};"
            f"font-weight:700;margin-top:-18px'>{health.assessment}</p>",
            unsafe_allow_html=True,
        )
    with c_kpis:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            rev = income["revenue"].iloc[-1]
            st.metric("Revenue", f"${rev/1e6:.2f}M",
                      delta=f"{latest.get('revenue_growth_qoq', 0)*100:+.1f}% QoQ")
        with k2:
            em = latest.get("ebitda_margin", 0)
            pe = prev_row.get("ebitda_margin", em)
            st.metric("EBITDA Margin", f"{em*100:.1f}%",
                      delta=f"{(em-pe)*100:+.1f}pp")
        with k3:
            st.metric("Net Margin", f"{latest.get('net_profit_margin', 0)*100:.1f}%")
        with k4:
            cr = latest.get("current_ratio", 0)
            st.metric("Current Ratio", f"{cr:.2f}x",
                      delta="✓ Healthy" if cr >= 1.5 else "⚠ Monitor")
        with k5:
            de = latest.get("debt_to_equity", 0)
            st.metric("Debt / Equity", f"{de:.2f}x",
                      delta="✓ Low" if de < 1.0 else ("⚠ High" if de > 1.5 else "Moderate"))

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        k6, k7, k8, k9, k10 = st.columns(5)
        with k6:
            ni = income["net_income"].iloc[-1]
            st.metric("Net Income", f"${ni/1e6:.2f}M")
        with k7:
            ocf = latest.get("operating_cash_flow_margin", 0)
            st.metric("OCF Margin", f"{ocf*100:.1f}%")
        with k8:
            roe = latest.get("return_on_equity", 0)
            st.metric("ROE", f"{roe*100:.1f}%")
        with k9:
            ic = latest.get("interest_coverage", 0)
            st.metric("Int. Coverage", f"{ic:.1f}x")
        with k10:
            yoy = latest.get("revenue_growth_yoy", 0)
            st.metric("Rev Growth YoY", f"{yoy*100:+.1f}%")

    st.markdown("---")

    # ── Row 2 : Revenue + Margins (equal split) ───────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_revenue(income, forecasts), use_container_width=True)
    with c2:
        benchmark = INDUSTRY_BENCHMARKS.get(industry, {})
        st.plotly_chart(chart_margins(kpi_df, benchmark), use_container_width=True)

    # ── Row 3 : Cash flow + Ratios (equal split) ──────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(chart_cashflow(cash_flow, forecasts), use_container_width=True)
    with c4:
        st.plotly_chart(chart_ratios(kpi_df), use_container_width=True)

    # ── Active alerts ─────────────────────────────────────────────────────────
    anomalies = kpi_summary.get("anomalies", [])
    if anomalies:
        st.markdown("### ⚠️ Active Alerts")
        for a in anomalies[-4:]:
            sev = a.get("severity", "medium")
            st.markdown(
                f"<div class='alert-{sev}'><b>[{a['period']}] {a['metric']}</b> — {a['description']}</div>",
                unsafe_allow_html=True,
            )

    # ── Export buttons ────────────────────────────────────────────────────────
    st.markdown("---")
    c_ex1, c_ex2, _ = st.columns([1, 1, 4])
    with c_ex1:
        kpi_csv = kpi_df.to_csv().encode()
        st.download_button("⬇ KPI Data (CSV)", kpi_csv, "kpi_data.csv", "text/csv")
    with c_ex2:
        export_json = json.dumps(kpi_summary, indent=2, default=str).encode()
        st.download_button("⬇ Summary (JSON)", export_json, "kpi_summary.json", "application/json")


def page_kpi_analysis(kpi_df, kpi_summary, industry):
    st.markdown("<h2 style='color:#1B2A4A'>KPI Analysis</h2>", unsafe_allow_html=True)

    benchmark = INDUSTRY_BENCHMARKS.get(industry, {})
    latest    = kpi_summary["latest"]

    # ── Benchmark comparison table ────────────────────────────────────────────
    if benchmark:
        st.subheader(f"vs {industry} Industry Benchmarks")
        bm_rows = []
        label_map = {
            "gross_margin": "Gross Margin",
            "ebitda_margin": "EBITDA Margin",
            "net_profit_margin": "Net Profit Margin",
            "operating_cash_flow_margin": "OCF Margin",
            "current_ratio": "Current Ratio",
            "debt_to_equity": "Debt / Equity",
            "revenue_growth_yoy": "Revenue Growth YoY",
            "return_on_equity": "Return on Equity",
        }
        for key, label in label_map.items():
            company_val = latest.get(key, None)
            bench_val   = benchmark.get(key, None)
            if company_val is None or bench_val is None:
                continue
            is_pct = key not in ("current_ratio", "debt_to_equity", "interest_coverage")
            if is_pct:
                cv_fmt = f"{company_val*100:.1f}%"
                bv_fmt = f"{bench_val*100:.1f}%"
            else:
                cv_fmt = f"{company_val:.2f}x"
                bv_fmt = f"{bench_val:.2f}x"
            # For D/E, lower is better
            if key == "debt_to_equity":
                vs = "✅ Better" if company_val < bench_val else ("⚠️ Worse" if company_val > bench_val * 1.2 else "≈ In line")
            else:
                vs = "✅ Better" if company_val >= bench_val * 0.95 else ("⚠️ Below" if company_val < bench_val * 0.8 else "≈ In line")
            bm_rows.append({"Metric": label, "Company": cv_fmt, "Benchmark": bv_fmt, "Assessment": vs})
        st.dataframe(pd.DataFrame(bm_rows), use_container_width=True, hide_index=True)
        st.plotly_chart(chart_benchmark_radar(latest, benchmark, industry), use_container_width=True)
        st.markdown("---")

    # ── Full KPI table ────────────────────────────────────────────────────────
    st.subheader("Full KPI History")
    kpi_display = kpi_df.copy()
    for col in ["revenue_growth_qoq", "revenue_growth_yoy", "gross_margin",
                "ebitda_margin", "net_profit_margin", "operating_cash_flow_margin", "return_on_equity"]:
        if col in kpi_display.columns:
            kpi_display[col] = kpi_display[col].map(
                lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
    for col in ["current_ratio", "debt_to_equity", "interest_coverage"]:
        if col in kpi_display.columns:
            kpi_display[col] = kpi_display[col].map(
                lambda x: f"{x:.2f}x" if pd.notna(x) else "—")
    kpi_display.index = kpi_display.index.astype(str)
    st.dataframe(kpi_display, use_container_width=True)

    st.markdown("---")
    selected_kpi = st.selectbox("Plot a KPI over time:", list(kpi_df.columns))
    fig = px.line(x=[str(p) for p in kpi_df.index], y=kpi_df[selected_kpi],
                  title=f"{selected_kpi} — Historical Trend",
                  labels={"x": "Period", "y": selected_kpi})
    fig.update_traces(line_color=C["blue"], line_width=2.5)
    fig.update_layout(**_base_layout(360))
    st.plotly_chart(fig, use_container_width=True)


def page_scenario(income):
    st.markdown("<h2 style='color:#1B2A4A'>Scenario Analysis & Stress Testing</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748B'>Model the impact of economic shocks on P&L. "
        "Adjust parameters or select a named scenario.</p>", unsafe_allow_html=True
    )

    # ── Named scenario presets ────────────────────────────────────────────────
    st.subheader("Quick Scenarios")
    preset_cols = st.columns(len(NAMED_SCENARIOS))
    selected_preset = None
    for i, (name, params) in enumerate(NAMED_SCENARIOS.items()):
        if preset_cols[i].button(name, use_container_width=True, key=f"preset_{name}"):
            selected_preset = (name, params)

    # Manage preset selection in session state
    if selected_preset:
        st.session_state["scenario_preset"] = selected_preset
    preset = st.session_state.get("scenario_preset", ("Base Case", NAMED_SCENARIOS["Base Case"]))
    preset_name, preset_params = preset

    st.markdown("---")

    # ── Parameter sliders ─────────────────────────────────────────────────────
    st.subheader(f"Parameters — {preset_name}")
    c1, c2 = st.columns(2)
    with c1:
        rev_delta  = st.slider("Revenue change (%)", -30, 30,
                                int(preset_params["revenue_delta"] * 100), 1,
                                help="Year-on-year change vs base period")
        cogs_delta = st.slider("COGS ratio shift (percentage points)", -5, 15,
                                int(preset_params["cogs_pp_delta"] * 100), 1,
                                help="+ve = COGS is higher share of revenue (margin erosion)")
    with c2:
        opex_delta  = st.slider("OpEx change (%)", -30, 50,
                                 int(preset_params["opex_delta"] * 100), 1,
                                 help="Change in operating expenses")
        int_delta   = st.slider("Interest expense change (%)", -50, 200,
                                 int(preset_params["interest_delta"] * 100), 5,
                                 help="Captures rate hike impact on floating debt")

    result = run_scenario_analysis(
        income,
        name=preset_name if (rev_delta, cogs_delta, opex_delta, int_delta) ==
             (int(preset_params["revenue_delta"]*100),
              int(preset_params["cogs_pp_delta"]*100),
              int(preset_params["opex_delta"]*100),
              int(preset_params["interest_delta"]*100))
        else "Custom",
        revenue_delta=rev_delta / 100,
        cogs_pp_delta=cogs_delta / 100,
        opex_delta=opex_delta / 100,
        interest_delta=int_delta / 100,
    )

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown("---")
    severity_badge = {
        "positive": ("✅ Positive Impact", C["green"]),
        "neutral":  ("ℹ️ Neutral Impact",  C["blue"]),
        "warning":  ("⚠️ Moderate Risk",   C["amber"]),
        "critical": ("🚨 Critical Risk",   C["red"]),
    }
    badge_text, badge_color = severity_badge.get(result.severity, ("—", C["gray"]))
    st.markdown(
        f"<div style='background:{badge_color}22;border-left:4px solid {badge_color};"
        f"padding:10px 16px;border-radius:6px;margin-bottom:16px'>"
        f"<b style='color:{badge_color}'>{badge_text}</b> — "
        f"EBITDA impact: <b>{result.ebitda_impact_pct:+.1f}%</b> | "
        f"Net income impact: <b>{result.net_income_impact_pct:+.1f}%</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    for col, label, base_k, scen_k in [
        (m1, "Revenue",     "revenue",     "revenue"),
        (m2, "Gross Profit","gross_profit","gross_profit"),
        (m3, "EBITDA",      "ebitda",      "ebitda"),
        (m4, "Net Income",  "net_income",  "net_income"),
    ]:
        base_v = result.base[base_k]
        scen_v = result.scenario[scen_k]
        delta  = (scen_v - base_v) / abs(base_v) * 100 if base_v else 0
        col.metric(label, f"${scen_v/1e6:.2f}M",
                   delta=f"{delta:+.1f}% vs base",
                   delta_color="normal" if delta >= 0 else "inverse")

    st.markdown("---")
    c_wf, c_comp = st.columns(2)
    with c_wf:
        st.plotly_chart(chart_scenario_waterfall(result), use_container_width=True)
    with c_comp:
        st.plotly_chart(chart_scenario_comparison(result), use_container_width=True)

    # ── Margin comparison ─────────────────────────────────────────────────────
    st.markdown("---")
    margin_data = pd.DataFrame({
        "Metric": ["Gross Margin", "EBITDA Margin", "Net Margin"],
        "Base":     [f"{result.base['gross_margin']*100:.1f}%",
                     f"{result.base['ebitda_margin']*100:.1f}%",
                     f"{result.base['net_margin']*100:.1f}%"],
        "Scenario": [f"{result.scenario['gross_margin']*100:.1f}%",
                     f"{result.scenario['ebitda_margin']*100:.1f}%",
                     f"{result.scenario['net_margin']*100:.1f}%"],
        "Δ":        [f"{(result.scenario['gross_margin'] - result.base['gross_margin'])*100:+.1f}pp",
                     f"{(result.scenario['ebitda_margin'] - result.base['ebitda_margin'])*100:+.1f}pp",
                     f"{(result.scenario['net_margin'] - result.base['net_margin'])*100:+.1f}pp"],
    })
    st.subheader("Margin Impact Summary")
    st.dataframe(margin_data, use_container_width=True, hide_index=True)

    # ── AI narrative (if provider available) ─────────────────────────────────
    if st.button("🤖 Generate AI Narrative for this Scenario", use_container_width=False):
        try:
            asst = CFOAssistant()
            with st.spinner("Generating scenario narrative…"):
                narrative = asst.generate_scenario_narrative(
                    result.name, result.base, result.scenario, result.assumptions
                )
            st.info(narrative)
        except EnvironmentError as e:
            st.warning(f"AI provider not configured: {e}")


def page_forecasts(forecasts, fc_df):
    st.markdown("<h2 style='color:#1B2A4A'>Financial Forecasts — Next 4 Quarters</h2>",
                unsafe_allow_html=True)

    # Display MAPE quality indicator
    avg_mape = fc_df["mape_pct"].mean() if "mape_pct" in fc_df.columns else None
    if avg_mape is not None:
        quality = "Excellent" if avg_mape < 5 else ("Good" if avg_mape < 15 else "Fair")
        st.info(f"Model accuracy (avg MAPE): **{avg_mape:.1f}%** — {quality}")

    st.dataframe(fc_df, use_container_width=True, hide_index=True)
    st.markdown("---")

    metric_choice = st.selectbox("Visualize forecast:", list(forecasts.keys()))
    fc = forecasts[metric_choice]
    hist_x = [str(p) for p in fc.historical.index]
    fc_x   = [str(p) for p in fc.forecast.index]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_x, y=fc.historical.values / 1e6,
                             name="Historical", line=dict(color=C["blue"], width=2.5)))
    fig.add_trace(go.Scatter(x=fc_x, y=fc.forecast.values / 1e6,
                             name="Forecast", mode="lines+markers",
                             line=dict(color=C["amber"], dash="dash", width=2.5)))
    fig.add_trace(go.Scatter(
        x=fc_x + fc_x[::-1],
        y=list(fc.confidence_upper / 1e6) + list(fc.confidence_lower / 1e6)[::-1],
        fill="toself", fillcolor="rgba(217,119,6,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="80% CI",
    ))
    # Separator line between history and forecast
    last_hist = hist_x[-1]
    fig.add_vline(x=last_hist, line_dash="dot", line_color=C["gray"],
                  annotation_text="Forecast start")
    fig.update_layout(title=f"{metric_choice.replace('_', ' ').title()} — Forecast ({fc.model_type})",
                      yaxis_title="$M", **_base_layout(420))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Model: **{fc.model_type}** | In-sample MAPE: **{fc.mape_in_sample:.1f}%** | 80% confidence interval shown")


def page_financial_health(kpi_df, health, dupoint, kpi_report):
    st.markdown("<h2 style='color:#1B2A4A'>Financial Health Assessment</h2>", unsafe_allow_html=True)

    # ── Overall score ─────────────────────────────────────────────────────────
    c_gauge, c_info = st.columns([1, 2])
    with c_gauge:
        st.plotly_chart(chart_health_gauge(health), use_container_width=True)
    with c_info:
        st.markdown(
            f"<div style='padding:16px'>"
            f"<p style='color:#64748B;font-size:.85rem;margin:0'>FINANCIAL HEALTH GRADE</p>"
            f"<p style='font-size:3.5rem;font-weight:800;color:{health.color};margin:0;line-height:1'>{health.grade}</p>"
            f"<p style='font-size:1.1rem;color:#374151;margin:4px 0'>{health.assessment}</p>"
            f"<hr style='border-color:#E2E8F0;margin:10px 0'>"
            f"<table style='width:100%;font-size:.85rem'>"
            f"<tr><td>Profitability</td><td><b>{health.profitability_score:.0f}/25</b></td>"
            f"    <td>Liquidity</td><td><b>{health.liquidity_score:.0f}/20</b></td></tr>"
            f"<tr><td>Leverage</td><td><b>{health.leverage_score:.0f}/20</b></td>"
            f"    <td>Growth</td><td><b>{health.growth_score:.0f}/20</b></td></tr>"
            f"<tr><td>Cash Quality</td><td><b>{health.cash_quality_score:.0f}/15</b></td>"
            f"    <td>Overall</td><td><b>{health.overall:.0f}/100</b></td></tr>"
            f"</table>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.plotly_chart(chart_health_breakdown(health), use_container_width=True)
    st.markdown("---")

    # ── DuPont Decomposition ──────────────────────────────────────────────────
    st.subheader("DuPont Decomposition — ROE Drivers")
    st.caption("ROE = Net Profit Margin × Asset Turnover × Equity Multiplier × 4 (annualised)")
    st.plotly_chart(chart_dupoint(dupoint), use_container_width=True)

    dupoint_df = pd.DataFrame({
        "Period": [str(p) for p in dupoint.net_profit_margin.index],
        "Net Margin": (dupoint.net_profit_margin * 100).map("{:.1f}%".format),
        "Asset Turnover": dupoint.asset_turnover.map("{:.3f}x".format),
        "Equity Multiplier": dupoint.equity_multiplier.map("{:.2f}x".format),
        "ROE (DuPont)": (dupoint.roe_dupoint * 100).map("{:.1f}%".format),
    })
    st.dataframe(dupoint_df, use_container_width=True, hide_index=True)


def page_anomalies(kpi_report):
    st.markdown("<h2 style='color:#1B2A4A'>Anomalies & Risk Signals</h2>", unsafe_allow_html=True)
    anomalies = kpi_report.anomalies

    if not anomalies:
        st.success("No anomalies detected in the current dataset. All metrics within acceptable thresholds.")
        return

    # ── Risk summary bar ─────────────────────────────────────────────────────
    from collections import Counter
    sev_counts = Counter(a.severity for a in anomalies)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Critical", sev_counts.get("critical", 0))
    c2.metric("High",     sev_counts.get("high", 0))
    c3.metric("Medium",   sev_counts.get("medium", 0))
    c4.metric("Low",      sev_counts.get("low", 0))
    st.markdown("---")

    # ── Risk heatmap (period × metric) ────────────────────────────────────────
    periods  = sorted({a.period for a in anomalies})
    metrics  = sorted({a.metric for a in anomalies})
    sev_num  = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    z_matrix = [[0] * len(periods) for _ in metrics]
    for a in anomalies:
        if a.period in periods and a.metric in metrics:
            pi = periods.index(a.period)
            mi = metrics.index(a.metric)
            z_matrix[mi][pi] = sev_num.get(a.severity, 1)

    fig_heat = go.Figure(go.Heatmap(
        z=z_matrix, x=periods, y=metrics,
        colorscale=[[0, "#FFFFFF"], [0.25, "#ECFDF5"], [0.5, "#FEF3C7"],
                    [0.75, "#FEF9C3"], [1.0, "#FEE2E2"]],
        zmin=0, zmax=4,
        hoverongaps=False,
        colorbar=dict(tickvals=[1, 2, 3, 4], ticktext=["Low", "Med", "High", "Critical"]),
    ))
    fig_heat.update_layout(title="Risk Heatmap — Period × Metric", **_base_layout(300))
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    # ── Detail list ──────────────────────────────────────────────────────────
    st.subheader(f"All Signals ({len(anomalies)} total)")
    for severity in ["critical", "high", "medium", "low"]:
        subset = [a for a in anomalies if a.severity == severity]
        if not subset:
            continue
        for a in subset:
            with st.expander(f"[{severity.upper()}] [{a.period}] {a.metric} = {a.value}"):
                st.write(a.description)
                st.caption(f"Threshold: {a.threshold}")


def page_ai_assistant(income, balance, cash_flow, kpi_summary, fc_df, health, company_name):
    st.markdown("<h2 style='color:#1B2A4A'>AI CFO Assistant</h2>", unsafe_allow_html=True)

    provider_status = get_provider_status()
    free_ready = [p for p, ok in provider_status.items() if ok and PROVIDER_INFO.get(p, {}).get("free")]
    paid_ready = [p for p, ok in provider_status.items() if ok and not PROVIDER_INFO.get(p, {}).get("free", True)]

    if not free_ready and not paid_ready:
        st.error(
            "No AI provider configured. Add one of the following to your **.env** file:\n\n"
            "```\n"
            "LLM_PROVIDER=gemini\n"
            "GEMINI_API_KEY=your_key  # Free: https://aistudio.google.com\n"
            "```\n"
            "Or use Groq (free): `GROQ_API_KEY=...` at https://console.groq.com"
        )
        return

    # ── Provider selector ─────────────────────────────────────────────────────
    all_ready = free_ready + paid_ready
    provider_labels = {p: f"{PROVIDER_INFO[p]['label']} {'(free)' if PROVIDER_INFO[p]['free'] else '(paid)'}"
                       for p in all_ready}
    default_provider = os.environ.get("LLM_PROVIDER", free_ready[0] if free_ready else all_ready[0])
    if default_provider not in all_ready and all_ready:
        default_provider = all_ready[0]

    c_prov, c_model = st.columns([2, 3])
    with c_prov:
        chosen_provider = st.selectbox(
            "AI Provider",
            options=all_ready,
            format_func=lambda p: provider_labels[p],
            index=all_ready.index(default_provider) if default_provider in all_ready else 0,
        )
    with c_model:
        default_model = PROVIDER_INFO[chosen_provider]["default_model"]
        model_choice = st.text_input("Model override (optional)", placeholder=default_model)
        if model_choice:
            os.environ["LLM_MODEL"] = model_choice
        else:
            os.environ.pop("LLM_MODEL", None)

    financial_data = {
        "company_name": company_name,
        "kpi_summary":  kpi_summary,
        "anomalies":    kpi_summary.get("anomalies", []),
        "forecast_summary": fc_df,
        "raw_income":   income,
        "raw_balance":  balance,
        "raw_cashflow": cash_flow,
        "health_score": health,
    }

    if "assistant" not in st.session_state or st.session_state.get("ai_provider") != chosen_provider:
        try:
            st.session_state.assistant    = CFOAssistant(provider=chosen_provider)
            st.session_state.chat_history = []
            st.session_state.ai_provider  = chosen_provider
        except EnvironmentError as e:
            st.error(str(e))
            return

    # ── Action buttons ────────────────────────────────────────────────────────
    btn1, btn2, btn3 = st.columns(3)
    with btn1:
        if st.button("📋 Executive Summary", use_container_width=True):
            with st.spinner(f"Generating via {PROVIDER_INFO[chosen_provider]['label']}…"):
                try:
                    summary = st.session_state.assistant.generate_executive_summary(financial_data)
                    st.session_state.chat_history.append(
                        ("assistant", f"**Executive Summary for {company_name}**\n\n{summary}")
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

    with btn2:
        if st.button("📊 Explain Health Score", use_container_width=True):
            q = (f"Explain our Financial Health Score of {health.overall}/100 (Grade {health.grade}). "
                 f"What are the top 3 improvements to move to the next grade?")
            with st.spinner("Analysing…"):
                try:
                    ans = st.session_state.assistant.answer_question(q, financial_data)
                    st.session_state.chat_history.append(("user", q))
                    st.session_state.chat_history.append(("assistant", ans))
                except Exception as e:
                    st.error(f"Error: {e}")

    with btn3:
        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.assistant.reset_conversation()
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")

    # ── Chat history ──────────────────────────────────────────────────────────
    for role, msg in st.session_state.get("chat_history", []):
        with st.chat_message(role):
            st.markdown(msg)

    if prompt := st.chat_input(f"Ask {company_name}'s CFO assistant…"):
        st.session_state.chat_history.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner(f"Thinking via {PROVIDER_INFO[chosen_provider]['label']}…"):
            try:
                answer = st.session_state.assistant.answer_question(prompt, financial_data)
            except Exception as e:
                answer = f"⚠️ Error: {e}"
        st.session_state.chat_history.append(("assistant", answer))
        with st.chat_message("assistant"):
            st.markdown(answer)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    with st.spinner("Loading financial intelligence platform…"):
        (datasets, kpi_report, kpi_df, kpi_summary,
         forecasts, fc_df, health, dupoint) = load_all()

    income    = datasets["income_statement"]
    balance   = datasets["balance_sheet"]
    cash_flow = datasets["cash_flow"]

    page, company_name, industry = render_sidebar(kpi_summary, health)

    if page == "Executive Dashboard":
        page_executive(income, balance, cash_flow, kpi_df, kpi_summary, forecasts, health, industry)
    elif page == "KPI Analysis":
        page_kpi_analysis(kpi_df, kpi_summary, industry)
    elif page == "Scenario Analysis":
        page_scenario(income)
    elif page == "Forecasts":
        page_forecasts(forecasts, fc_df)
    elif page == "Financial Health":
        page_financial_health(kpi_df, health, dupoint, kpi_report)
    elif page == "Anomalies & Risks":
        page_anomalies(kpi_report)
    elif page == "AI CFO Assistant":
        page_ai_assistant(income, balance, cash_flow, kpi_summary, fc_df, health, company_name)


if __name__ == "__main__":
    main()
