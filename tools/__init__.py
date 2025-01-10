"""Tools module for AlphaAgents data fetching and analysis."""

from .financial_data import (
    get_stock_info,
    get_financial_statements,
    get_historical_prices,
    get_key_metrics,
)
from .cache import disk_cache
from .ml_engine import MLEngine
from .transformer_model import TransformerPredictor, GRUPredictor
from .sentiment_nn import FinancialSentimentClassifier, CNNTextClassifier
from .rl_agent import DQNAgent, run_rl_simulation
from .vision_patterns import ChartPatternScanner, analyze_structural_break
from .rag_engine import RAGEngine
from .signals import AlphaSignalGenerator
from .alt_data import AltDataEngine
from .news_api import (
    get_stock_news,
    get_analyst_ratings,
    analyze_sentiment,
    get_sentiment_for_tickers,
)
from .technical_analysis import (
    get_technical_indicators,
    get_volume_analysis,
    get_support_resistance,
    get_indicator_data,
)

__all__ = [
    "get_stock_info",
    "get_financial_statements",
    "get_historical_prices",
    "get_key_metrics",
    "get_stock_news",
    "get_analyst_ratings",
    "analyze_sentiment",
    "get_sentiment_for_tickers",
    "get_technical_indicators",
    "get_volume_analysis",
    "get_support_resistance",
    "get_indicator_data",
    "MLEngine",
    "TransformerPredictor",
    "GRUPredictor",
    "FinancialSentimentClassifier",
    "CNNTextClassifier",
    "DQNAgent",
    "run_rl_simulation",
    "ChartPatternScanner",
    "analyze_structural_break",
    "RAGEngine",
    "AlphaSignalGenerator",
    "AltDataEngine",
]
