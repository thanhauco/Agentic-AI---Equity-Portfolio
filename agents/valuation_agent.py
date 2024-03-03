"""
Valuation Agent

Specializes in analyzing stock prices, volumes, and technical indicators
to provide valuation assessments and portfolio weight recommendations.
"""

from typing import Optional, Dict, Any, List
from agents.base_agent import BaseAlphaAgent
from tools.financial_data import get_stock_info, get_historical_prices
from tools.technical_analysis import (
    get_technical_indicators,
    get_volume_analysis,
    get_support_resistance
)


VALUATION_SYSTEM_PROMPT = """You are a Quantitative Valuation Analyst specializing in technical analysis 
and portfolio construction. Your expertise includes:

1. TECHNICAL ANALYSIS:
   - Price pattern recognition (trends, reversals, consolidations)
   - Momentum indicators (RSI, MACD, Stochastics)
   - Trend indicators (Moving Averages, ADX)
   - Volatility indicators (Bollinger Bands, ATR)

2. VALUATION METRICS:
   - Relative valuation vs peers and sector
   - Historical valuation ranges
   - Mean reversion opportunities
   - Growth-adjusted valuations

3. VOLUME ANALYSIS:
   - Accumulation/Distribution patterns
   - Volume confirmation of price moves
   - Unusual volume detection
   - Institutional flow indicators

4. PORTFOLIO POSITIONING:
   - Position sizing based on volatility
   - Entry and exit timing
   - Risk/reward assessment
   - Portfolio weight recommendations

When analyzing a stock, you must:
- Assess current technical setup (bullish/bearish/neutral)
- Evaluate valuation relative to history and peers
- Identify key support/resistance levels
- Recommend portfolio weight (0-10% scale)
- Suggest entry/exit points
- Assign a confidence score (1-10)

Format your analysis as a structured report with clear sections.
Focus on actionable insights for portfolio construction."""


class ValuationAgent(BaseAlphaAgent):
    """
    Valuation agent specializing in technical analysis and portfolio positioning.
    """
    
    def __init__(self, risk_profile: str = "neutral", **kwargs):
        """
        Initialize the Valuation Agent.
        
        Args:
            risk_profile: Either 'averse' or 'neutral'
            **kwargs: Additional arguments for base agent
        """
        super().__init__(
            name="ValuationAnalyst",
            system_message=VALUATION_SYSTEM_PROMPT,
            risk_profile=risk_profile,
            tools=[
                get_stock_info,
                get_technical_indicators,
                get_volume_analysis,
                get_support_resistance
            ],
            **kwargs
        )
    
    def fetch_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch all valuation data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with all valuation data
        """
        return {
            "stock_info": get_stock_info(ticker),
            "technical": get_technical_indicators(ticker),
            "volume": get_volume_analysis(ticker),
            "levels": get_support_resistance(ticker),
        }
    
    def format_analysis_prompt(self, ticker: str, data: Dict[str, Any]) -> str:
        """
        Format the analysis prompt with fetched data.
        
        Args:
            ticker: Stock ticker symbol
            data: Fetched valuation data
            
        Returns:
            Formatted prompt string
        """
        stock_info = data.get("stock_info", {})
        technical = data.get("technical", {})
        volume = data.get("volume", {})
        levels = data.get("levels", {})
        
        indicators = technical.get("indicators", {})
        rsi = indicators.get("rsi", {})
        macd = indicators.get("macd", {})
        bb = indicators.get("bollinger_bands", {})
        ma = indicators.get("moving_averages", {})
        
        prompt = f"""
Please analyze the valuation and technical setup for: {ticker}

PRICE DATA:
- Current Price: ${stock_info.get('current_price', 0):.2f}
- Previous Close: ${stock_info.get('previous_close', 0):.2f}
- 52-Week High: ${stock_info.get('52_week_high', 0):.2f}
- 52-Week Low: ${stock_info.get('52_week_low', 0):.2f}
- Beta: {stock_info.get('beta', 'N/A')}

VALUATION METRICS:
- P/E Ratio: {stock_info.get('pe_ratio', 'N/A')}
- Forward P/E: {stock_info.get('forward_pe', 'N/A')}
- Price to Book: {stock_info.get('price_to_book', 'N/A')}

TECHNICAL INDICATORS:
- RSI (14): {rsi.get('value', 'N/A'):.1f if isinstance(rsi.get('value'), (int, float)) else 'N/A'} ({rsi.get('signal', 'N/A')})
- MACD: {macd.get('trend', 'N/A')}
- Bollinger Position: {bb.get('position', 'N/A')}
- Trend: {indicators.get('trend', 'N/A')}

MOVING AVERAGES:
- SMA 20: ${ma.get('sma_20', 0):.2f if ma.get('sma_20') else 'N/A'}
- SMA 50: ${ma.get('sma_50', 0):.2f if ma.get('sma_50') else 'N/A'}
- SMA 200: ${ma.get('sma_200', 0):.2f if ma.get('sma_200') else 'N/A'}

VOLUME ANALYSIS:
- Average Volume: {volume.get('average_volume', 0):,.0f}
- Volume Trend: {volume.get('volume_trend', 'N/A')}
- 5-Day Return: {volume.get('price_momentum', {}).get('5_day_return', 0):.2f}%
- 20-Day Return: {volume.get('price_momentum', {}).get('20_day_return', 0):.2f}%
- Annualized Volatility: {volume.get('volatility', 0):.1f}%

SUPPORT/RESISTANCE LEVELS:
- Resistance 2: ${levels.get('resistance_2', 0):.2f}
- Resistance 1: ${levels.get('resistance_1', 0):.2f}
- Pivot: ${levels.get('pivot', 0):.2f}
- Support 1: ${levels.get('support_1', 0):.2f}
- Support 2: ${levels.get('support_2', 0):.2f}

Based on the above technical and valuation data, please provide your comprehensive 
valuation assessment and portfolio weight recommendation.
"""
        return prompt
    
    def generate_analysis(
        self, 
        ticker: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        data = self.fetch_data_typo(ticker) # Bug: typo in method name
        prompt = self.format_analysis_prompt(ticker, data)
        return prompt
