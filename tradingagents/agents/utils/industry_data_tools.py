"""
LangChain tool wrappers for industry analysis data functions.
"""

from langchain_core.tools import tool
from typing import Annotated

from tradingagents.dataflows.industry_analysis import (
    get_sector_industry_overview,
    get_sector_performance,
    get_industry_peers,
    get_industry_sector_news,
)


@tool
def get_sector_overview(
    ticker: Annotated[str, "Ticker symbol of the company"],
) -> str:
    """
    Retrieve sector and industry classification, business description, and company
    profile for a given ticker. Useful for understanding what industry a company
    belongs to and the sector-level context.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "NVDA")

    Returns:
        Sector/industry overview including classification, description, and key metrics.
    """
    return get_sector_industry_overview(ticker)


@tool
def get_sector_performance_vs_benchmark(
    ticker: Annotated[str, "Ticker symbol of the company"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back (default 90)"] = 90,
) -> str:
    """
    Compare the stock's recent price performance against its sector ETF benchmark.
    Identifies whether the stock is outperforming or underperforming its sector.

    Args:
        ticker: Stock ticker symbol
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of calendar days to look back (default 90)

    Returns:
        Performance comparison table and interpretation.
    """
    return get_sector_performance(ticker, curr_date, look_back_days)


@tool
def get_industry_peer_metrics(
    ticker: Annotated[str, "Ticker symbol of the company"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back (default 30)"] = 30,
) -> str:
    """
    Retrieve key valuation and financial metrics for the target company to use
    as context for industry peer comparison. Includes P/E, P/B, EV/EBITDA,
    revenue growth, profit margin, and return on equity.

    Args:
        ticker: Stock ticker symbol
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Days to look back for recent performance comparison

    Returns:
        Key metrics table for the company and its industry context.
    """
    return get_industry_peers(ticker, curr_date, look_back_days)


@tool
def get_industry_news(
    ticker: Annotated[str, "Ticker symbol of the company"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back for news (default 30)"] = 30,
) -> str:
    """
    Retrieve recent news for the sector/industry and the company itself.
    Uses the sector ETF news as a proxy for industry-wide developments and catalysts.

    Args:
        ticker: Stock ticker symbol
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back for news

    Returns:
        Industry-level and company-level news headlines.
    """
    return get_industry_sector_news(ticker, curr_date, look_back_days)
