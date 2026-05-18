# CFO Copilot AI

> An AI-powered Financial Planning & Analysis assistant for CFOs and Finance Directors.

## Overview

CFO Copilot combines classical financial analysis, time-series forecasting, and large language models (Claude by Anthropic) to deliver an end-to-end FP&A intelligence platform.

```
cfo-copilot-ai/
├── data/
│   ├── income_statement.csv
│   ├── balance_sheet.csv
│   └── cash_flow.csv
├── src/
│   ├── data_loader.py      # Load & validate financial datasets
│   ├── analysis.py         # KPI computation & anomaly detection
│   ├── forecasting.py      # ARIMA + Ridge regression forecasting
│   ├── llm_assistant.py    # Claude-powered CFO assistant
│   └── dashboard.py        # Streamlit + Plotly dashboard
├── main.py                 # CLI entry point
├── requirements.txt
└── .env.example
```

## Features

| Feature | Details |
|---------|---------|
| **KPI Engine** | Revenue growth (QoQ/YoY), EBITDA margin, net profit margin, current ratio, D/E ratio, interest coverage, ROE |
| **Anomaly Detection** | Revenue drop >15%, COGS surge >20%, liquidity risk (CR < 1.2), leverage ceiling, earnings quality |
| **Forecasting** | ARIMA(1,1,1)(1,1,0)[4] with Ridge regression fallback, 80% confidence intervals |
| **AI Assistant** | Executive summaries, KPI variance explanations, multi-turn Q&A via Claude |
| **Dashboard** | Streamlit app with Plotly charts — KPIs, trends, forecasts, anomaly panel, AI chat |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run the CLI pipeline

```bash
# Without AI (no API key needed)
python main.py

# With AI executive summary
python main.py --llm

# Ask a single question
python main.py --llm --ask "What is driving the Q3 2023 revenue decline?"

# Interactive Q&A session
python main.py --llm --interactive

# Custom forecast horizon & strategy
python main.py --horizon 8 --forecast-strategy arima
```

### 4. Launch the dashboard

```bash
streamlit run src/dashboard.py
```

## KPIs Computed

### Profitability
- **Gross Margin** = Gross Profit / Revenue
- **EBITDA Margin** = EBITDA / Revenue
- **Net Profit Margin** = Net Income / Revenue

### Growth
- **Revenue Growth QoQ** — quarter-over-quarter percentage change
- **Revenue Growth YoY** — year-over-year percentage change

### Liquidity
- **Current Ratio** = Current Assets / Current Liabilities
- **Operating CF Margin** = Operating Cash Flow / Revenue

### Leverage
- **Debt-to-Equity** = Total Debt / Total Equity
- **Interest Coverage** = EBIT / Interest Expense

### Returns
- **Return on Equity** = Annualised Net Income / Avg Equity

## Anomaly Detection Rules

| Signal | Threshold |
|--------|-----------|
| Revenue drop | > 15% QoQ decline |
| COGS surge | > 20% QoQ increase (and outpacing revenue) |
| Liquidity risk | Current ratio < 1.2 |
| Leverage risk | Debt-to-equity > 2.0x |
| Earnings quality | OCF < 60% of Net Income for 2+ consecutive quarters |

## Forecasting

- **Primary**: SARIMA(1,1,1)(1,1,0)[4] — accounts for quarterly seasonality
- **Fallback**: Ridge regression with autoregressive lag features
- **Metrics forecast**: Revenue, EBITDA, Net Income, Operating Cash Flow
- **Horizon**: 4 quarters (configurable)
- **Confidence interval**: 80%

## AI Assistant (Claude)

Uses **Claude claude-sonnet-4-6** with prompt caching for efficiency. Capabilities:

- **Executive Summary** — board-ready financial narrative
- **KPI Variance Explanation** — natural-language interpretation of metric changes
- **Q&A** — multi-turn conversation grounded in your financial data

### Example questions

```
"What drove the margin compression in 2023?"
"Are we at risk of a liquidity crunch in the next two quarters?"
"Summarise our debt profile and coverage ratios."
"What's the revenue growth outlook for 2025?"
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data processing | pandas, numpy |
| Forecasting | statsmodels (ARIMA), scikit-learn (Ridge) |
| AI | Anthropic Claude API (claude-sonnet-4-6) |
| Dashboard | Streamlit, Plotly |
| Config | python-dotenv |

## License

MIT
