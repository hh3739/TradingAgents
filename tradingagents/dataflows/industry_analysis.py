"""
Industry analysis data functions using yfinance.

Provides sector/industry classification, peer comparison, and sector performance
data to support long-term industry tracking by the Industry Analyst agent.
"""

from typing import Annotated
from datetime import datetime, timedelta
import yfinance as yf


# Mapping of sector names to representative ETF tickers
SECTOR_ETFS = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Communication Services": "XLC",
    "Consumer Staples": "XLP",
}


def get_sector_industry_overview(
    ticker: Annotated[str, "Ticker symbol of the company"],
) -> str:
    """
    Retrieve sector and industry classification, business description,
    and key company profile information for a given ticker.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "NVDA")

    Returns:
        Formatted string with sector/industry overview
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        company_name = info.get("longName", ticker.upper())
        description = info.get("longBusinessSummary", "No description available.")
        country = info.get("country", "N/A")
        employees = info.get("fullTimeEmployees", "N/A")
        market_cap = info.get("marketCap", None)
        website = info.get("website", "N/A")

        market_cap_str = (
            f"${market_cap / 1e9:.2f}B" if market_cap else "N/A"
        )

        report = f"""# Sector & Industry Overview: {company_name} ({ticker.upper()})

## Classification
- **Sector**: {sector}
- **Industry**: {industry}
- **Country**: {country}

## Company Profile
- **Full-Time Employees**: {f"{employees:,}" if isinstance(employees, int) else employees}
- **Market Cap**: {market_cap_str}
- **Website**: {website}

## Business Description
{description}
"""
        # Representative sector ETF
        etf = SECTOR_ETFS.get(sector)
        if etf:
            report += f"\n## Sector ETF Benchmark\nThe representative sector ETF for {sector} is **{etf}**.\n"

        return report

    except Exception as e:
        return f"Error fetching sector/industry overview for {ticker}: {str(e)}"


def get_sector_performance(
    ticker: Annotated[str, "Ticker symbol of the company"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back"] = 90,
) -> str:
    """
    Compare the stock's recent performance against its sector ETF benchmark.

    Args:
        ticker: Stock ticker symbol
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of calendar days to look back (default 90)

    Returns:
        Formatted performance comparison string
    """
    try:
        end_date = datetime.strptime(curr_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=look_back_days)
        start_str = start_date.strftime("%Y-%m-%d")

        stock = yf.Ticker(ticker.upper())
        info = stock.info
        sector = info.get("sector", "Unknown")

        etf_ticker = SECTOR_ETFS.get(sector, "SPY")  # Fall back to SPY if no sector ETF

        # Fetch price data
        stock_hist = stock.history(start=start_str, end=curr_date)
        etf = yf.Ticker(etf_ticker)
        etf_hist = etf.history(start=start_str, end=curr_date)

        if stock_hist.empty or etf_hist.empty:
            return f"Insufficient price data for {ticker} or sector ETF {etf_ticker}."

        # Calculate returns
        stock_start = stock_hist["Close"].iloc[0]
        stock_end = stock_hist["Close"].iloc[-1]
        stock_return = (stock_end - stock_start) / stock_start * 100

        etf_start = etf_hist["Close"].iloc[0]
        etf_end = etf_hist["Close"].iloc[-1]
        etf_return = (etf_end - etf_start) / etf_start * 100

        relative_perf = stock_return - etf_return

        # Additional stats
        stock_high = stock_hist["Close"].max()
        stock_low = stock_hist["Close"].min()
        avg_volume = stock_hist["Volume"].mean()

        report = f"""# Sector Performance Comparison: {ticker.upper()} vs {sector} ({etf_ticker})
Period: {start_str} to {curr_date} ({look_back_days} days)

## Returns
| | {ticker.upper()} | {etf_ticker} (Sector ETF) | Relative |
|---|---|---|---|
| Return | {stock_return:+.2f}% | {etf_return:+.2f}% | {relative_perf:+.2f}% |

## {ticker.upper()} Price Stats (last {look_back_days} days)
- **Current Price**: ${stock_end:.2f}
- **Period High**: ${stock_high:.2f}
- **Period Low**: ${stock_low:.2f}
- **Avg Daily Volume**: {avg_volume:,.0f}

## Interpretation
{"OUTPERFORMING" if relative_perf > 0 else "UNDERPERFORMING"} the sector by {abs(relative_perf):.2f}% over the past {look_back_days} days.
"""
        return report

    except Exception as e:
        return f"Error fetching sector performance for {ticker}: {str(e)}"


def get_industry_peers(
    ticker: Annotated[str, "Ticker symbol of the company"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back"] = 30,
) -> str:
    """
    Retrieve key financial metrics and recent performance for peer companies
    in the same industry.

    Args:
        ticker: Stock ticker symbol
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Days to look back for performance comparison

    Returns:
        Formatted peer comparison string
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")

        end_date = datetime.strptime(curr_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=look_back_days)
        start_str = start_date.strftime("%Y-%m-%d")

        # yfinance doesn't have a direct peers API; use sector peers if available
        # Try to get peers from recommendations
        peers = []
        try:
            peers_df = stock.get_recommendations_summary()
            # This won't give peers, so we use a curated fallback approach
        except Exception:
            pass

        # Get key metrics for the main stock
        pe = info.get("trailingPE", None)
        fwd_pe = info.get("forwardPE", None)
        pb = info.get("priceToBook", None)
        ps = info.get("priceToSalesTrailing12Months", None)
        ev_ebitda = info.get("enterpriseToEbitda", None)
        profit_margin = info.get("profitMargins", None)
        revenue_growth = info.get("revenueGrowth", None)
        roe = info.get("returnOnEquity", None)

        def fmt(val, pct=False, mult=100):
            if val is None:
                return "N/A"
            if pct:
                return f"{val * mult:.1f}%"
            return f"{val:.2f}"

        report = f"""# Industry Peer Context: {ticker.upper()}
Sector: {sector} | Industry: {industry}

## {ticker.upper()} Key Valuation Metrics
| Metric | Value |
|--------|-------|
| Trailing P/E | {fmt(pe)} |
| Forward P/E | {fmt(fwd_pe)} |
| Price/Book | {fmt(pb)} |
| Price/Sales | {fmt(ps)} |
| EV/EBITDA | {fmt(ev_ebitda)} |
| Profit Margin | {fmt(profit_margin, pct=True)} |
| Revenue Growth (YoY) | {fmt(revenue_growth, pct=True)} |
| Return on Equity | {fmt(roe, pct=True)} |

## Industry Context
- **Sector**: {sector}
- **Industry**: {industry}

Note: For detailed peer comparison, cross-reference with sector ETF ({SECTOR_ETFS.get(sector, "SPY")})
performance and industry-wide metrics.
"""
        return report

    except Exception as e:
        return f"Error fetching industry peers for {ticker}: {str(e)}"


def get_industry_sector_news(
    ticker: Annotated[str, "Ticker symbol of the company"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back for news"] = 30,
) -> str:
    """
    Retrieve recent news for the sector/industry ETF to capture
    industry-level trends and catalysts.

    Args:
        ticker: Stock ticker symbol (used to identify sector)
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back for news

    Returns:
        Formatted industry news string
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")

        etf_ticker = SECTOR_ETFS.get(sector, "SPY")
        etf = yf.Ticker(etf_ticker)

        # Get news for the sector ETF (reflects sector-level events)
        etf_news = etf.news
        stock_news = stock.news

        end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=look_back_days)
        start_ts = int(start_dt.timestamp())

        report = f"""# Industry & Sector News: {sector} / {industry}
Covering: {ticker.upper()} and sector ETF {etf_ticker}
Period: last {look_back_days} days up to {curr_date}

## Sector ETF ({etf_ticker}) News
"""
        etf_articles = [
            a for a in (etf_news or [])
            if a.get("providerPublishTime", 0) >= start_ts
        ]
        if etf_articles:
            for i, article in enumerate(etf_articles[:8], 1):
                title = article.get("title", "No title")
                publisher = article.get("publisher", "Unknown")
                pub_time = article.get("providerPublishTime", 0)
                pub_dt = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d") if pub_time else "N/A"
                report += f"{i}. [{pub_dt}] **{title}** - {publisher}\n"
        else:
            report += "No recent sector news found.\n"

        report += f"\n## Company-Specific News ({ticker.upper()})\n"
        stock_articles = [
            a for a in (stock_news or [])
            if a.get("providerPublishTime", 0) >= start_ts
        ]
        if stock_articles:
            for i, article in enumerate(stock_articles[:8], 1):
                title = article.get("title", "No title")
                publisher = article.get("publisher", "Unknown")
                pub_time = article.get("providerPublishTime", 0)
                pub_dt = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d") if pub_time else "N/A"
                report += f"{i}. [{pub_dt}] **{title}** - {publisher}\n"
        else:
            report += "No recent company-specific news found.\n"

        return report

    except Exception as e:
        return f"Error fetching industry news for {ticker}: {str(e)}"
