"""
News API Tools

Tools for fetching financial news and analyst ratings.
"""

import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import yfinance as yf
from config import get_settings
from .cache import disk_cache


@disk_cache(expiry_seconds=1800)  # News expires faster
def get_stock_news(ticker: str, days: int = 7) -> List[Dict[str, Any]]:
    """
    Fetch recent news articles for a stock.
    
    Args:
        ticker: Stock ticker symbol
        days: Number of days to look back
        
    Returns:
        List of news articles with title, description, source, and date
    """
    settings = get_settings()
    
    # Try NewsAPI if configured
    if settings.news_api_key:
        return _fetch_from_newsapi(ticker, days, settings.news_api_key)
    
    # Fallback to yfinance news
    return _fetch_from_yfinance(ticker)


def _fetch_from_newsapi(ticker: str, days: int, api_key: str) -> List[Dict[str, Any]]:
    """Fetch news from NewsAPI."""
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "from": from_date,
        "sortBy": "relevancy",
        "apiKey": api_key,
        "language": "en",
        "pageSize": 20,
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        articles = []
        for article in data.get("articles", []):
            articles.append({
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "source": article.get("source", {}).get("name", ""),
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt", ""),
            })
        return articles
    except Exception as e:
        return [{"error": str(e)}]


def _fetch_from_yfinance(ticker: str) -> List[Dict[str, Any]]:
    """Fetch news from yfinance as fallback."""
    stock = yf.Ticker(ticker)
    news = stock.news
    
    articles = []
    for item in news[:20]:
        articles.append({
            "title": item.get("title", ""),
            "description": "",
            "source": item.get("publisher", ""),
            "url": item.get("link", ""),
            "published_at": datetime.fromtimestamp(
                item.get("providerPublishTime", 0)
            ).isoformat() if item.get("providerPublishTime") else "",
        })
    return articles


@disk_cache(expiry_seconds=3600)
def get_analyst_ratings(ticker: str) -> Dict[str, Any]:
    """
    Get analyst ratings and price targets.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with rating data
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Get recommendations
    try:
        recommendations = stock.recommendations
        recent_recs = recommendations.tail(10).to_dict("records") if recommendations is not None else []
    except:
        recent_recs = []
    
    return {
        "ticker": ticker,
        "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
        "target_mean_price": info.get("targetMeanPrice", 0),
        "target_high_price": info.get("targetHighPrice", 0),
        "target_low_price": info.get("targetLowPrice", 0),
        "recommendation_key": info.get("recommendationKey", ""),
        "number_of_analysts": info.get("numberOfAnalystOpinions", 0),
        "recent_recommendations": recent_recs,
        "upside_potential": (
            (info.get("targetMeanPrice", 0) - info.get("currentPrice", 1)) 
            / info.get("currentPrice", 1) * 100
            if info.get("currentPrice", 0) > 0 else 0
        ),
    }


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Simple sentiment analysis for news text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Sentiment score and classification
    """
    # Simple keyword-based sentiment (in production, use a proper NLP model)
    positive_words = [
        "growth", "profit", "gain", "surge", "beat", "strong", "positive",
        "upgrade", "bullish", "outperform", "buy", "record", "success"
    ]
    negative_words = [
        "loss", "decline", "drop", "miss", "weak", "negative", "downgrade",
        "bearish", "underperform", "sell", "concern", "risk", "warning"
    ]
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    total = positive_count + negative_count
    if total == 0:
        score = 0.5
        classification = "neutral"
    else:
        score = positive_count / total
        if score > 0.6:
            classification = "positive"
        elif score < 0.4:
            classification = "negative"
        else:
            classification = "neutral"
    
def get_sentiment_for_tickers(tickers: List[str]) -> Dict[str, float]:
    """
    Get aggregate sentiment for a list of tickers.
    
    Args:
        tickers: List of stock symbols
        
    Returns:
        Dictionary mapping tickers to sentiment score (0 to 1)
    """
    results = {}
    for ticker in tickers:
        try:
            news = get_stock_news(ticker, days=3)
            if not news or "error" in news[0]:
                results[ticker] = 0.5
                continue
                
            scores = []
            for art in news[:10]:
                sent = analyze_sentiment(art.get("title", ""))
                scores.append(sent["score"])
            
            results[ticker] = sum(scores) / len(scores) if scores else 0.5
        except:
            results[ticker] = 0.5
    return results
