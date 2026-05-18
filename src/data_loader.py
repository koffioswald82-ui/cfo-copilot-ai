"""
data_loader.py — Load and clean financial datasets.

Reads CSV files for income statement, balance sheet, and cash flow,
validates schema, parses period index, and returns clean DataFrames.
"""

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

# Expected columns per dataset
SCHEMA: dict[str, list[str]] = {
    "income_statement": [
        "period", "revenue", "cost_of_goods_sold", "gross_profit",
        "operating_expenses", "ebitda", "depreciation_amortization",
        "ebit", "interest_expense", "tax_expense", "net_income",
    ],
    "balance_sheet": [
        "period", "cash_and_equivalents", "accounts_receivable", "inventory",
        "other_current_assets", "total_current_assets", "property_plant_equipment",
        "intangible_assets", "other_noncurrent_assets", "total_assets",
        "accounts_payable", "short_term_debt", "other_current_liabilities",
        "total_current_liabilities", "long_term_debt", "other_noncurrent_liabilities",
        "total_liabilities", "common_stock", "retained_earnings", "total_equity",
    ],
    "cash_flow": [
        "period", "net_income", "depreciation_amortization",
        "changes_in_working_capital", "other_operating", "operating_cash_flow",
        "capital_expenditures", "acquisitions", "other_investing", "investing_cash_flow",
        "debt_issuance", "debt_repayment", "dividends_paid", "other_financing",
        "financing_cash_flow", "net_change_in_cash",
    ],
}


def _parse_period(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'YYYY-Qn' period strings to PeriodIndex (quarterly)."""
    df = df.copy()
    df["period"] = pd.PeriodIndex(df["period"].str.replace("-Q", "Q"), freq="Q")
    df = df.set_index("period").sort_index()
    return df


def _validate_schema(df: pd.DataFrame, name: str) -> None:
    required = set(SCHEMA[name]) - {"period"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")


def _clean(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Coerce numeric types, drop full-NA rows, fill isolated NAs with interpolation."""
    _validate_schema(df, name)
    numeric_cols = [c for c in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    df = df.interpolate(method="linear", limit_direction="both")
    logger.info("[%s] Loaded %d rows, %d columns.", name, len(df), len(df.columns))
    return df


def load_financial_data(data_dir: str | Path = "data") -> dict[str, pd.DataFrame]:
    """
    Load all three financial statements from CSV files.

    Parameters
    ----------
    data_dir : path to directory containing the three CSV files.

    Returns
    -------
    dict with keys 'income_statement', 'balance_sheet', 'cash_flow'.
    Each value is a clean DataFrame indexed by quarterly PeriodIndex.
    """
    data_dir = Path(data_dir)
    datasets: dict[str, pd.DataFrame] = {}

    file_map = {
        "income_statement": "income_statement.csv",
        "balance_sheet": "balance_sheet.csv",
        "cash_flow": "cash_flow.csv",
    }

    for name, filename in file_map.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        raw = pd.read_csv(path)
        parsed = _parse_period(raw)
        clean = _clean(parsed, name)
        datasets[name] = clean

    return datasets


def load_from_dataframes(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cash_flow_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Process raw DataFrames (from file upload) the same way as CSV loader."""
    return {
        "income_statement": _clean(_parse_period(income_df.copy()), "income_statement"),
        "balance_sheet":    _clean(_parse_period(balance_df.copy()), "balance_sheet"),
        "cash_flow":        _clean(_parse_period(cash_flow_df.copy()), "cash_flow"),
    }


def get_combined_view(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all three statements on the period index for a unified view.
    Suffixes are added where column names collide.
    """
    inc = datasets["income_statement"]
    bal = datasets["balance_sheet"]
    cf = datasets["cash_flow"].drop(
        columns=["net_income", "depreciation_amortization"], errors="ignore"
    )
    combined = inc.join(bal, how="inner", lsuffix="", rsuffix="_bs")
    combined = combined.join(cf, how="inner", rsuffix="_cf")
    return combined
