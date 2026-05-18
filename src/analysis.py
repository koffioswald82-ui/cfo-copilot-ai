"""
analysis.py — Financial KPIs, anomaly detection, health scoring,
               DuPont decomposition, and scenario stress-testing.
"""

import logging
from dataclasses import dataclass, field
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Thresholds ──────────────────────────────────────────────────────────────
REVENUE_DROP_THRESHOLD = -0.15
COST_SURGE_THRESHOLD = 0.20
CURRENT_RATIO_FLOOR = 1.2
DEBT_EQUITY_CEILING = 2.0

# ─── Industry Benchmarks ─────────────────────────────────────────────────────
INDUSTRY_BENCHMARKS: dict[str, dict[str, float]] = {
    "Technology": {
        "gross_margin": 0.65, "ebitda_margin": 0.25, "net_profit_margin": 0.15,
        "current_ratio": 1.8, "debt_to_equity": 0.5, "revenue_growth_yoy": 0.12,
        "return_on_equity": 0.20, "operating_cash_flow_margin": 0.20,
    },
    "Industrials": {
        "gross_margin": 0.35, "ebitda_margin": 0.15, "net_profit_margin": 0.08,
        "current_ratio": 1.5, "debt_to_equity": 0.8, "revenue_growth_yoy": 0.06,
        "return_on_equity": 0.12, "operating_cash_flow_margin": 0.12,
    },
    "Consumer Goods": {
        "gross_margin": 0.45, "ebitda_margin": 0.18, "net_profit_margin": 0.09,
        "current_ratio": 1.4, "debt_to_equity": 1.0, "revenue_growth_yoy": 0.05,
        "return_on_equity": 0.14, "operating_cash_flow_margin": 0.14,
    },
    "Healthcare": {
        "gross_margin": 0.55, "ebitda_margin": 0.22, "net_profit_margin": 0.12,
        "current_ratio": 2.0, "debt_to_equity": 0.6, "revenue_growth_yoy": 0.08,
        "return_on_equity": 0.16, "operating_cash_flow_margin": 0.16,
    },
    "Financial Services": {
        "gross_margin": 0.70, "ebitda_margin": 0.30, "net_profit_margin": 0.18,
        "current_ratio": 1.2, "debt_to_equity": 1.5, "revenue_growth_yoy": 0.07,
        "return_on_equity": 0.15, "operating_cash_flow_margin": 0.22,
    },
    "Retail": {
        "gross_margin": 0.40, "ebitda_margin": 0.12, "net_profit_margin": 0.05,
        "current_ratio": 1.3, "debt_to_equity": 1.2, "revenue_growth_yoy": 0.04,
        "return_on_equity": 0.10, "operating_cash_flow_margin": 0.08,
    },
    "Energy": {
        "gross_margin": 0.30, "ebitda_margin": 0.20, "net_profit_margin": 0.10,
        "current_ratio": 1.6, "debt_to_equity": 0.7, "revenue_growth_yoy": 0.05,
        "return_on_equity": 0.11, "operating_cash_flow_margin": 0.15,
    },
}


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class AnomalyReport:
    period: str
    metric: str
    value: float
    threshold: float
    description: str
    severity: str = "medium"   # low | medium | high | critical


@dataclass
class KPIReport:
    revenue_growth_qoq: pd.Series = field(default_factory=pd.Series)
    revenue_growth_yoy: pd.Series = field(default_factory=pd.Series)
    gross_margin: pd.Series = field(default_factory=pd.Series)
    ebitda_margin: pd.Series = field(default_factory=pd.Series)
    net_profit_margin: pd.Series = field(default_factory=pd.Series)
    operating_cash_flow_margin: pd.Series = field(default_factory=pd.Series)
    current_ratio: pd.Series = field(default_factory=pd.Series)
    debt_to_equity: pd.Series = field(default_factory=pd.Series)
    interest_coverage: pd.Series = field(default_factory=pd.Series)
    return_on_equity: pd.Series = field(default_factory=pd.Series)
    anomalies: list[AnomalyReport] = field(default_factory=list)


@dataclass
class FinancialHealthScore:
    overall: float                 # 0–100
    profitability_score: float     # 0–25
    liquidity_score: float         # 0–20
    leverage_score: float          # 0–20
    growth_score: float            # 0–20
    cash_quality_score: float      # 0–15
    grade: str                     # A | B | C | D | F
    assessment: str
    color: str                     # hex for UI


@dataclass
class DuPontAnalysis:
    net_profit_margin: pd.Series
    asset_turnover: pd.Series
    equity_multiplier: pd.Series
    roe_dupoint: pd.Series


@dataclass
class ScenarioResult:
    name: str
    base: dict
    scenario: dict
    assumptions: dict
    ebitda_impact_pct: float
    net_income_impact_pct: float
    severity: str   # positive | neutral | warning | critical


# ─── KPI Computation ─────────────────────────────────────────────────────────

def compute_kpis(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cash_flow: pd.DataFrame,
) -> KPIReport:
    report = KPIReport()

    rev = income["revenue"]
    report.revenue_growth_qoq = rev.pct_change(1).rename("revenue_growth_qoq")
    report.revenue_growth_yoy = rev.pct_change(4).rename("revenue_growth_yoy")

    report.gross_margin = (income["gross_profit"] / rev).rename("gross_margin")
    report.ebitda_margin = (income["ebitda"] / rev).rename("ebitda_margin")
    report.net_profit_margin = (income["net_income"] / rev).rename("net_profit_margin")
    report.operating_cash_flow_margin = (
        cash_flow["operating_cash_flow"] / rev
    ).rename("operating_cash_flow_margin")

    report.current_ratio = (
        balance["total_current_assets"] / balance["total_current_liabilities"]
    ).rename("current_ratio")

    total_debt = balance["long_term_debt"] + balance.get(
        "short_term_debt", pd.Series(0, index=balance.index)
    )
    report.debt_to_equity = (total_debt / balance["total_equity"]).rename("debt_to_equity")

    report.interest_coverage = (
        income["ebit"] / income["interest_expense"]
    ).rename("interest_coverage")

    avg_equity = balance["total_equity"].rolling(2).mean()
    report.return_on_equity = (income["net_income"] * 4 / avg_equity).rename("return_on_equity")

    report.anomalies = detect_anomalies(income, balance, cash_flow, report)
    return report


def kpis_to_dataframe(report: KPIReport) -> pd.DataFrame:
    series_fields = [
        "revenue_growth_qoq", "revenue_growth_yoy", "gross_margin",
        "ebitda_margin", "net_profit_margin", "operating_cash_flow_margin",
        "current_ratio", "debt_to_equity", "interest_coverage", "return_on_equity",
    ]
    return pd.concat([getattr(report, f) for f in series_fields], axis=1)


# ─── Anomaly Detection ────────────────────────────────────────────────────────

def detect_anomalies(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cash_flow: pd.DataFrame,
    report: KPIReport,
) -> list[AnomalyReport]:
    anomalies: list[AnomalyReport] = []

    for period, val in report.revenue_growth_qoq.dropna().items():
        if val < REVENUE_DROP_THRESHOLD:
            severity = "critical" if val < -0.25 else "high"
            anomalies.append(AnomalyReport(
                period=str(period), metric="revenue_growth_qoq",
                value=round(val * 100, 2), threshold=REVENUE_DROP_THRESHOLD * 100,
                description=(
                    f"Revenue dropped {abs(val)*100:.1f}% QoQ in {period}, "
                    f"below the {abs(REVENUE_DROP_THRESHOLD)*100:.0f}% alert threshold."
                ),
                severity=severity,
            ))

    cogs_growth = income["cost_of_goods_sold"].pct_change(1)
    rev_growth = income["revenue"].pct_change(1)
    for period in cogs_growth.dropna().index:
        cg = cogs_growth[period]
        rg = rev_growth.get(period, 0)
        if cg > COST_SURGE_THRESHOLD and cg > rg + 0.05:
            anomalies.append(AnomalyReport(
                period=str(period), metric="cogs_growth_qoq",
                value=round(cg * 100, 2), threshold=COST_SURGE_THRESHOLD * 100,
                description=(
                    f"COGS increased {cg*100:.1f}% QoQ in {period} "
                    f"while revenue grew only {rg*100:.1f}% — margin pressure detected."
                ),
                severity="high",
            ))

    for period, cr in report.current_ratio.dropna().items():
        if cr < CURRENT_RATIO_FLOOR:
            severity = "critical" if cr < 1.0 else "high"
            anomalies.append(AnomalyReport(
                period=str(period), metric="current_ratio",
                value=round(cr, 3), threshold=CURRENT_RATIO_FLOOR,
                description=(
                    f"Current ratio of {cr:.2f} in {period} is below "
                    f"the {CURRENT_RATIO_FLOOR} liquidity floor — short-term risk elevated."
                ),
                severity=severity,
            ))

    for period, de in report.debt_to_equity.dropna().items():
        if de > DEBT_EQUITY_CEILING:
            severity = "critical" if de > 3.0 else "high"
            anomalies.append(AnomalyReport(
                period=str(period), metric="debt_to_equity",
                value=round(de, 3), threshold=DEBT_EQUITY_CEILING,
                description=(
                    f"Debt-to-equity of {de:.2f}x in {period} exceeds "
                    f"the {DEBT_EQUITY_CEILING}x ceiling — leverage risk flagged."
                ),
                severity=severity,
            ))

    ocf = cash_flow["operating_cash_flow"]
    ni = income["net_income"]
    quality_flag = (ocf < ni * 0.6).rolling(2).sum()
    for period, count in quality_flag.dropna().items():
        if count >= 2:
            anomalies.append(AnomalyReport(
                period=str(period), metric="earnings_quality",
                value=round((ocf[period] / ni[period]) * 100, 1), threshold=60.0,
                description=(
                    f"Operating cash flow is only {ocf[period]/ni[period]*100:.0f}% of net income "
                    f"in {period} (2+ consecutive quarters) — earnings quality concern."
                ),
                severity="medium",
            ))

    logger.info("Anomaly detection complete. %d anomalies found.", len(anomalies))
    return anomalies


# ─── Financial Health Score ───────────────────────────────────────────────────

def compute_financial_health_score(report: KPIReport) -> FinancialHealthScore:
    """Weighted composite score 0–100 across 5 financial dimensions."""
    kdf = kpis_to_dataframe(report)
    latest = kdf.iloc[-1]

    # Profitability (0–25)
    ebitda_m = float(latest.get("ebitda_margin", 0) or 0)
    net_m    = float(latest.get("net_profit_margin", 0) or 0)
    roe      = float(latest.get("return_on_equity", 0) or 0)
    prof_score = min(25.0, (
        min(max(ebitda_m, 0) / 0.30, 1.0) * 10 +   # 30% EBITDA margin = 10 pts
        min(max(net_m, 0) / 0.15, 1.0) * 8 +         # 15% net margin   =  8 pts
        min(max(roe, 0)  / 0.20, 1.0) * 7             # 20% ROE          =  7 pts
    ))

    # Liquidity (0–20)
    cr = float(latest.get("current_ratio", 1.0) or 1.0)
    if cr >= 2.0:   liq_score = 20.0
    elif cr >= 1.5: liq_score = 15.0
    elif cr >= 1.2: liq_score = 10.0
    elif cr >= 1.0: liq_score = 5.0
    else:           liq_score = max(0.0, cr * 5)

    # Leverage (0–20)
    de = float(latest.get("debt_to_equity", 1.0) or 1.0)
    if de <= 0.5:   lev_score = 20.0
    elif de <= 1.0: lev_score = 16.0
    elif de <= 1.5: lev_score = 12.0
    elif de <= 2.0: lev_score = 8.0
    else:           lev_score = max(0.0, 8.0 - (de - 2.0) * 3)

    # Growth (0–20)
    qoq = float(latest.get("revenue_growth_qoq", 0) or 0)
    yoy = float(latest.get("revenue_growth_yoy", 0) or 0)
    growth_score = min(20.0, (
        min(max(qoq, 0) / 0.05, 1.0) * 8 +   # 5% QoQ  = 8 pts
        min(max(yoy, 0) / 0.15, 1.0) * 12     # 15% YoY = 12 pts
    ))

    # Cash Quality (0–15)
    ocf_m = float(latest.get("operating_cash_flow_margin", 0) or 0)
    cash_score = min(15.0, min(max(ocf_m, 0) / 0.20, 1.0) * 15)

    overall = prof_score + liq_score + lev_score + growth_score + cash_score

    if overall >= 80:
        grade, assessment, color = "A", "Excellent financial health", "#16A34A"
    elif overall >= 65:
        grade, assessment, color = "B", "Good health — minor areas to monitor", "#65A30D"
    elif overall >= 50:
        grade, assessment, color = "C", "Moderate — proactive management needed", "#D97706"
    elif overall >= 35:
        grade, assessment, color = "D", "Weak position — corrective action required", "#EA580C"
    else:
        grade, assessment, color = "F", "Critical stress — immediate intervention needed", "#DC2626"

    return FinancialHealthScore(
        overall=round(overall, 1),
        profitability_score=round(prof_score, 1),
        liquidity_score=round(liq_score, 1),
        leverage_score=round(lev_score, 1),
        growth_score=round(growth_score, 1),
        cash_quality_score=round(cash_score, 1),
        grade=grade,
        assessment=assessment,
        color=color,
    )


# ─── DuPont Decomposition ─────────────────────────────────────────────────────

def compute_dupoint(
    income: pd.DataFrame,
    balance: pd.DataFrame,
) -> DuPontAnalysis:
    """ROE = Net Profit Margin × Asset Turnover × Equity Multiplier."""
    net_margin       = (income["net_income"] / income["revenue"]).rename("net_profit_margin")
    asset_turnover   = (income["revenue"] / balance["total_assets"]).rename("asset_turnover")
    equity_mult      = (balance["total_assets"] / balance["total_equity"]).rename("equity_multiplier")
    roe_reconstructed = (net_margin * asset_turnover * equity_mult * 4).rename("roe_dupoint")

    return DuPontAnalysis(
        net_profit_margin=net_margin,
        asset_turnover=asset_turnover,
        equity_multiplier=equity_mult,
        roe_dupoint=roe_reconstructed,
    )


# ─── Scenario Analysis ────────────────────────────────────────────────────────

NAMED_SCENARIOS = {
    "Base Case":   {"revenue_delta": 0.00, "cogs_pp_delta": 0.00, "opex_delta": 0.00, "interest_delta": 0.00},
    "Recession":   {"revenue_delta": -0.15, "cogs_pp_delta": 0.03, "opex_delta": 0.05,  "interest_delta": 0.50},
    "Expansion":   {"revenue_delta": 0.20, "cogs_pp_delta": -0.01, "opex_delta": 0.10, "interest_delta": 0.00},
    "Cost Crisis": {"revenue_delta": 0.00, "cogs_pp_delta": 0.08, "opex_delta": 0.20,  "interest_delta": 1.00},
    "Best Case":   {"revenue_delta": 0.25, "cogs_pp_delta": -0.02, "opex_delta": -0.05, "interest_delta": -0.30},
    "Stagflation": {"revenue_delta": 0.03, "cogs_pp_delta": 0.05, "opex_delta": 0.10,  "interest_delta": 1.50},
}


def run_scenario_analysis(
    income: pd.DataFrame,
    name: str = "Custom",
    revenue_delta: float = 0.0,          # fractional change, e.g. -0.10 = -10%
    cogs_pp_delta: float = 0.0,          # percentage-point shift in COGS/Revenue ratio
    opex_delta: float = 0.0,             # fractional change in operating expenses
    interest_delta: float = 0.0,         # fractional change in interest expense
) -> ScenarioResult:
    """Apply income-statement shocks to the last period and compute the impact."""
    last = income.iloc[-1].copy()

    base_rev    = float(last["revenue"])
    base_cogs   = float(last["cost_of_goods_sold"])
    base_opex   = float(last["operating_expenses"])
    base_ebitda = float(last["ebitda"])
    base_da     = float(last["depreciation_amortization"])
    base_int    = float(last["interest_expense"])
    base_tax    = float(last["tax_expense"])
    base_ebit   = float(last["ebit"])
    base_ni     = float(last["net_income"])

    # Compute base tax rate safely
    ebt_base = base_ebit - base_int
    tax_rate = base_tax / ebt_base if ebt_base > 0 else 0.25

    # Apply shocks
    new_rev  = base_rev * (1 + revenue_delta)
    new_cogs = new_rev * (base_cogs / base_rev + cogs_pp_delta)
    new_gross_profit = new_rev - new_cogs
    new_opex = base_opex * (1 + opex_delta)
    new_ebitda = new_gross_profit - new_opex
    new_ebit   = new_ebitda - base_da
    new_int    = base_int * (1 + interest_delta)
    ebt_new    = new_ebit - new_int
    new_tax    = max(0.0, ebt_new * tax_rate)
    new_ni     = ebt_new - new_tax

    base = {
        "revenue": base_rev, "gross_profit": base_rev - base_cogs,
        "ebitda": base_ebitda, "net_income": base_ni,
        "gross_margin": (base_rev - base_cogs) / base_rev,
        "ebitda_margin": base_ebitda / base_rev,
        "net_margin": base_ni / base_rev if base_rev else 0,
    }
    scenario = {
        "revenue": new_rev, "gross_profit": new_gross_profit,
        "ebitda": new_ebitda, "net_income": new_ni,
        "gross_margin": new_gross_profit / new_rev if new_rev else 0,
        "ebitda_margin": new_ebitda / new_rev if new_rev else 0,
        "net_margin": new_ni / new_rev if new_rev else 0,
    }

    ebitda_impact = (new_ebitda - base_ebitda) / abs(base_ebitda) * 100 if base_ebitda else 0
    ni_impact     = (new_ni - base_ni) / abs(base_ni) * 100 if base_ni else 0

    if ebitda_impact >= 10:     severity = "positive"
    elif ebitda_impact >= -5:   severity = "neutral"
    elif ebitda_impact >= -20:  severity = "warning"
    else:                       severity = "critical"

    assumptions = {
        f"Revenue growth": f"{revenue_delta*100:+.1f}%",
        f"COGS ratio shift": f"{cogs_pp_delta*100:+.1f}pp",
        f"OpEx change": f"{opex_delta*100:+.1f}%",
        f"Interest expense change": f"{interest_delta*100:+.1f}%",
    }

    return ScenarioResult(
        name=name, base=base, scenario=scenario,
        assumptions=assumptions,
        ebitda_impact_pct=round(ebitda_impact, 1),
        net_income_impact_pct=round(ni_impact, 1),
        severity=severity,
    )


# ─── Summary Statistics ───────────────────────────────────────────────────────

def summarise_kpis(report: KPIReport) -> dict:
    kdf = kpis_to_dataframe(report)
    latest = kdf.iloc[-1]
    trailing = kdf.tail(4).mean()

    return {
        "latest_period": str(kdf.index[-1]),
        "latest": latest.to_dict(),
        "trailing_4q_avg": trailing.to_dict(),
        "anomaly_count": len(report.anomalies),
        "anomalies": [
            {
                "period": a.period, "metric": a.metric,
                "description": a.description, "severity": a.severity,
            }
            for a in report.anomalies
        ],
    }
