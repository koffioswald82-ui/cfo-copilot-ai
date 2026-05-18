"""
main.py — CFO Copilot entry point (CLI mode).

Usage:
    python main.py                          # full pipeline, no LLM
    python main.py --llm                    # include AI summaries
    python main.py --llm --ask "question"   # single Q&A
    python main.py --forecast-strategy arima
"""

import argparse
import logging
from pathlib import Path

# Load .env from the project root (works regardless of working directory)
_ENV_FILE = Path(__file__).parent / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_ENV_FILE, override=True)
except ImportError:
    # python-dotenv not installed — fall back to reading manually
    if _ENV_FILE.exists():
        import os
        for line in _ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cfo_copilot")


def print_section(title: str) -> None:
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def run_pipeline(args: argparse.Namespace) -> None:
    from src.data_loader import load_financial_data
    from src.analysis import compute_kpis, kpis_to_dataframe, summarise_kpis
    from src.forecasting import run_all_forecasts, forecast_summary

    # ── 1. Load Data ─────────────────────────────────────────────────────────
    print_section("1. Loading Financial Data")
    datasets = load_financial_data(args.data_dir)
    income = datasets["income_statement"]
    balance = datasets["balance_sheet"]
    cash_flow = datasets["cash_flow"]
    logger.info("Loaded %d periods of data.", len(income))

    # ── 2. Compute KPIs ──────────────────────────────────────────────────────
    print_section("2. Computing KPIs")
    kpi_report = compute_kpis(income, balance, cash_flow)
    kpis_to_dataframe(kpi_report)   # materialises series; used by summarise_kpis internally
    kpi_summary = summarise_kpis(kpi_report)

    print(f"\nLatest Period: {kpi_summary['latest_period']}")
    print("\n--- Latest KPIs ---")
    for k, v in kpi_summary["latest"].items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:>10.4f}")

    print("\n--- Trailing 4Q Averages ---")
    for k, v in kpi_summary["trailing_4q_avg"].items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:>10.4f}")

    # ── 3. Anomalies ─────────────────────────────────────────────────────────
    print_section(f"3. Anomaly Detection ({kpi_summary['anomaly_count']} found)")
    if kpi_report.anomalies:
        for a in kpi_report.anomalies:
            print(f"\n  [!] [{a.period}] {a.metric}")
            print(f"     {a.description}")
    else:
        print("  No anomalies detected.")

    # ── 4. Forecasting ───────────────────────────────────────────────────────
    print_section("4. Financial Forecasting")
    forecasts = run_all_forecasts(
        income, cash_flow,
        horizon=args.horizon,
        strategy=args.forecast_strategy,
    )
    fc_df = forecast_summary(forecasts)
    print(fc_df.to_string(index=False))

    # ── 5. LLM Assistant ─────────────────────────────────────────────────────
    if args.llm:
        print_section("5. AI Executive Summary (Claude)")
        try:
            from src.llm_assistant import CFOAssistant
            assistant = CFOAssistant()
            financial_data = {
                "kpi_summary": kpi_summary,
                "anomalies": kpi_summary.get("anomalies", []),
                "forecast_summary": fc_df,
                "raw_income": income,
                "raw_balance": balance,
                "raw_cashflow": cash_flow,
            }
            summary = assistant.generate_executive_summary(financial_data)
            print(summary)

            if args.ask:
                print_section("6. CFO Q&A")
                print(f"Q: {args.ask}\n")
                answer = assistant.answer_question(args.ask, financial_data)
                print(f"A: {answer}")

            if args.interactive:
                print_section("6. Interactive CFO Q&A (type 'exit' to quit)")
                while True:
                    question = input("\nYour question: ").strip()
                    if question.lower() in ("exit", "quit", "q"):
                        break
                    if not question:
                        continue
                    answer = assistant.answer_question(question, financial_data)
                    print(f"\nAssistant: {answer}")

        except EnvironmentError as e:
            logger.error("LLM unavailable: %s", e)
            print(f"\n  [SKIPPED] {e}")

    print_section("Pipeline Complete")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CFO Copilot — AI-powered financial analysis system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data", help="Path to financial CSV files")
    parser.add_argument("--horizon", type=int, default=4, help="Forecast horizon in quarters")
    parser.add_argument(
        "--forecast-strategy", choices=["auto", "arima", "ridge"], default="auto",
        help="Forecasting model strategy"
    )
    parser.add_argument("--llm", action="store_true", help="Enable AI-powered summaries")
    parser.add_argument("--ask", type=str, default=None, help="Single question for the CFO assistant")
    parser.add_argument("--interactive", action="store_true", help="Start interactive Q&A session")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
