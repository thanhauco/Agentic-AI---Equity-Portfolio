"""
Financial Data Tools

Tools for fetching stock data, financial statements, and market information
using yfinance.
"""

import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from .cache import disk_cache


@disk_cache(expiry_seconds=3600)
def get_stock_info(ticker: str) -> Dict[str, Any]:
    """
    Fetch comprehensive stock information.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        Dictionary containing stock metadata, current price, volume, etc.
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    
    return {
        "ticker": ticker,
        "name": info.get("longName", ""),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "market_cap": info.get("marketCap", 0),
        "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
        "previous_close": info.get("previousClose", 0),
        "open": info.get("open", 0),
        "day_high": info.get("dayHigh", 0),
        "day_low": info.get("dayLow", 0),
        "volume": info.get("volume", 0),
        "avg_volume": info.get("averageVolume", 0),
        "52_week_high": info.get("fiftyTwoWeekHigh", 0),
        "52_week_low": info.get("fiftyTwoWeekLow", 0),
        "pe_ratio": info.get("trailingPE", 0),
        "forward_pe": info.get("forwardPE", 0),
        "peg_ratio": info.get("pegRatio", 0),
        "price_to_book": info.get("priceToBook", 0),
        "dividend_yield": info.get("dividendYield", 0),
        "beta": info.get("beta", 0),
        "eps": info.get("trailingEps", 0),
        "target_mean_price": info.get("targetMeanPrice", 0),
        "recommendation": info.get("recommendationKey", ""),
    }


@disk_cache(expiry_seconds=86400)  # Financials change less frequently
def get_financial_statements(ticker: str) -> Dict[str, Any]:
    """
    Retrieve income statement, balance sheet, and cash flow data.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with financial statement data
    """
    stock = yf.Ticker(ticker)
    
    # Get financial statements
    income_stmt = stock.income_stmt
    balance_sheet = stock.balance_sheet
    cash_flow = stock.cashflow
    
    def df_to_dict(df: pd.DataFrame) -> Dict[str, Any]:
        """Convert DataFrame to dictionary with string dates."""
        if df is None or df.empty:
            return {}
        result = {}
        for col in df.columns:
            col_key = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
            result[col_key] = df[col].to_dict()
        return result
    
    return {
        "ticker": ticker,
        "income_statement": df_to_dict(income_stmt),
        "balance_sheet": df_to_dict(balance_sheet),
        "cash_flow": df_to_dict(cash_flow),
    }


def get_historical_prices(
    ticker: str,
    period: str = "1y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Get historical OHLCV price data.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
    Returns:
        DataFrame with OHLCV data
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    return hist


@disk_cache(expiry_seconds=3600)
def get_key_metrics(ticker: str) -> Dict[str, Any]:
    """
    Calculate key financial metrics for analysis.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with calculated metrics
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Get financial data
    income_stmt = stock.income_stmt
    balance_sheet = stock.balance_sheet
    
    metrics = {
        "ticker": ticker,
        "profitability": {
            "gross_margin": info.get("grossMargins", 0),
            "operating_margin": info.get("operatingMargins", 0),
            "profit_margin": info.get("profitMargins", 0),
            "roe": info.get("returnOnEquity", 0),
            "roa": info.get("returnOnAssets", 0),
        },
        "valuation": {
            "pe_ratio": info.get("trailingPE", 0),
            "forward_pe": info.get("forwardPE", 0),
            "peg_ratio": info.get("pegRatio", 0),
            "price_to_book": info.get("priceToBook", 0),
            "price_to_sales": info.get("priceToSalesTrailing12Months", 0),
            "ev_to_ebitda": info.get("enterpriseToEbitda", 0),
        },
        "growth": {
            "revenue_growth": info.get("revenueGrowth", 0),
            "earnings_growth": info.get("earningsGrowth", 0),
        },
        "financial_health": {
            "current_ratio": info.get("currentRatio", 0),
            "debt_to_equity": info.get("debtToEquity", 0),
            "quick_ratio": info.get("quickRatio", 0),
        },
    }
    
    return metrics
