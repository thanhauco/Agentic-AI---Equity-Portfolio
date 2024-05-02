"""
Technical Analysis Tools

Tools for calculating technical indicators and analyzing price/volume patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import yfinance as yf

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False


def get_technical_indicators(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """
    Calculate key technical indicators for a stock.
    
    Args:
        ticker: Stock ticker symbol
        period: Historical data period
        
    Returns:
        Dictionary with technical indicators
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        return {"error": "No historical data available"}
    
    result = {
        "ticker": ticker,
        "period": period,
        "data_points": len(df),
        "latest_price": df["Close"].iloc[-1],
        "indicators": {},
    }
    
    # Calculate indicators using pandas_ta or fallback
    if HAS_PANDAS_TA:
        result["indicators"] = _calculate_with_pandas_ta(df)
    else:
        result["indicators"] = _calculate_manually(df)
    
    return result


def get_indicator_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate indicators for the entire DataFrame for plotting.
    
    Args:
        df: Historical price DataFrame
        
    Returns:
        DataFrame with indicators appended
    """
    df_with_ind = df.copy()
    
    if HAS_PANDAS_TA:
        # RSI
        df_with_ind["RSI"] = ta.rsi(df["Close"], length=14)
        
        # MACD
        macd = ta.macd(df["Close"])
        if macd is not None:
            df_with_ind = pd.concat([df_with_ind, macd], axis=1)
            
        # Bollinger Bands
        bbands = ta.bbands(df["Close"], length=20)
        if bbands is not None:
            df_with_ind = pd.concat([df_with_ind, bbands], axis=1)
            
        # Moving Averages
        df_with_ind["SMA_20"] = ta.sma(df["Close"], length=20)
        df_with_ind["SMA_50"] = ta.sma(df["Close"], length=50)
        df_with_ind["SMA_200"] = ta.sma(df["Close"], length=200)
    else:
        # Manual Fallback
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_with_ind["RSI"] = 100 - (100 / (1 + rs))
        
        df_with_ind["SMA_20"] = df["Close"].rolling(20).mean()
        df_with_ind["SMA_50"] = df["Close"].rolling(50).mean()
        df_with_ind["SMA_200"] = df["Close"].rolling(200).mean()
        
    return df_with_ind


def _calculate_with_pandas_ta(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate indicators using pandas_ta."""
    indicators = {}
    
    # RSI
    rsi = ta.rsi(df["Close"], length=14)
    if rsi is not None and len(rsi) > 0:
        indicators["rsi"] = {
            "value": rsi.iloc[-1],
            "signal": "overbought" if rsi.iloc[-1] > 70 else "oversold" if rsi.iloc[-1] < 30 else "neutral"
        }
    
    # MACD
    macd = ta.macd(df["Close"])
    if macd is not None and len(macd) > 0:
        indicators["macd"] = {
            "macd": macd.iloc[-1, 0],
            "signal": macd.iloc[-1, 1],
            "histogram": macd.iloc[-1, 2],
            "trend": "bullish" if macd.iloc[-1, 2] > 0 else "bearish"
        }
    
    # Bollinger Bands
    bbands = ta.bbands(df["Close"], length=20)
    if bbands is not None and len(bbands) > 0:
        current_price = df["Close"].iloc[-1]
        upper = bbands.iloc[-1, 0]
        lower = bbands.iloc[-1, 2]
        indicators["bollinger_bands"] = {
            "upper": upper,
            "middle": bbands.iloc[-1, 1],
            "lower": lower,
            "position": "above_upper" if current_price > upper else "below_lower" if current_price < lower else "within_bands"
        }
    
    # Moving Averages
    sma_20 = ta.sma(df["Close"], length=20)
    sma_50 = ta.sma(df["Close"], length=50)
    sma_200 = ta.sma(df["Close"], length=200)
    
    indicators["moving_averages"] = {
        "sma_20": sma_20.iloc[-1] if sma_20 is not None and len(sma_20) > 0 else None,
        "sma_50": sma_50.iloc[-1] if sma_50 is not None and len(sma_50) > 0 else None,
        "sma_200": sma_200.iloc[-1] if sma_200 is not None and len(sma_200) > 0 else None,
    }
    
    # Trend determination
    current = df["Close"].iloc[-1]
    if sma_50 is not None and sma_200 is not None:
        sma50_val = sma_50.iloc[-1]
        sma200_val = sma_200.iloc[-1]
        if current > sma50_val > sma200_val:
            indicators["trend"] = "strong_uptrend"
        elif current > sma200_val:
            indicators["trend"] = "uptrend"
        elif current < sma50_val < sma200_val:
            indicators["trend"] = "strong_downtrend"
        else:
            indicators["trend"] = "downtrend"
    
    return indicators


def _calculate_manually(df: pd.DataFrame) -> Dict[str, Any]:
    """Fallback manual calculation without pandas_ta."""
    indicators = {}
    
    # Simple RSI calculation
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    if len(rsi) > 0:
        indicators["rsi"] = {
            "value": rsi.iloc[-1],
            "signal": "overbought" if rsi.iloc[-1] > 70 else "oversold" if rsi.iloc[-1] < 30 else "neutral"
        }
    
    # Moving Averages
    indicators["moving_averages"] = {
        "sma_20": df["Close"].rolling(20).mean().iloc[-1],
        "sma_50": df["Close"].rolling(50).mean().iloc[-1],
        "sma_200": df["Close"].rolling(200).mean().iloc[-1] if len(df) >= 200 else None,
    }
    
    return indicators


def get_volume_analysis(ticker: str, period: str = "3mo") -> Dict[str, Any]:
    """
    Analyze volume patterns and momentum.
    
    Args:
        ticker: Stock ticker symbol
        period: Historical data period
        
    Returns:
        Dictionary with volume analysis
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        return {"error": "No historical data available"}
    
    avg_volume = df["Volume"].mean()
    recent_volume = df["Volume"].iloc[-5:].mean()
    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
    
    # Calculate price momentum
    returns_5d = (df["Close"].iloc[-1] / df["Close"].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
    returns_20d = (df["Close"].iloc[-1] / df["Close"].iloc[-20] - 1) * 100 if len(df) >= 20 else 0
    
    return {
        "ticker": ticker,
        "average_volume": avg_volume,
        "recent_5d_volume": recent_volume,
        "volume_ratio": volume_ratio,
        "volume_trend": "increasing" if volume_ratio > 1.2 else "decreasing" if volume_ratio < 0.8 else "stable",
        "price_momentum": {
            "5_day_return": returns_5d,
            "20_day_return": returns_20d,
        },
        "volatility": df["Close"].pct_change().std() * np.sqrt(252) * 100,  # Annualized
    }


def get_support_resistance(ticker: str, period: str = "6mo") -> Dict[str, Any]:
    """
    Identify key support and resistance levels.
    
    Args:
        ticker: Stock ticker symbol
        period: Historical data period
        
    Returns:
        Dictionary with support/resistance levels
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        return {"error": "No historical data available"}
    
    current_price = df["Close"].iloc[-1]
    high = df["High"].max()
    low = df["Low"].min()
    
    # Simple pivot points
    pivot = (high + low + current_price) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    
    return {
        "ticker": ticker,
        "current_price": current_price,
        "period_high": high,
        "period_low": low,
        "pivot": pivot,
        "resistance_1": r1,
        "resistance_2": r2,
        "support_1": s1,
        "support_2": s2,
    }
