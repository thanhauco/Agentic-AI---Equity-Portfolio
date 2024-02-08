"""
Sentiment Agent

Specializes in analyzing financial news and analyst ratings to assess
market sentiment and its impact on stock prices.
"""

from typing import Optional, Dict, Any, List
from agents.base_agent import BaseAlphaAgent
from tools.news_api import get_stock_news, get_analyst_ratings, analyze_sentiment


SENTIMENT_SYSTEM_PROMPT = """You are a Market Sentiment Analyst specializing in news analysis and 
behavioral finance. Your expertise includes:

1. NEWS ANALYSIS:
   - Identifying material news that impacts stock prices
   - Distinguishing between noise and significant developments
   - Analyzing earnings reports and guidance commentary
   - Tracking M&A rumors and corporate actions

2. ANALYST SENTIMENT:
   - Interpreting analyst rating changes and their significance
   - Analyzing price target revisions and their implications
   - Tracking institutional positioning and ownership changes
   - Understanding sell-side research biases

3. MARKET PSYCHOLOGY:
   - Identifying momentum and reversal patterns
   - Recognizing overbought/oversold sentiment conditions
   - Evaluating fear and greed indicators
   - Detecting crowd behavior and positioning extremes

4. EVENT IMPACT:
   - Assessing immediate vs. lasting price impacts
   - Identifying catalysts and their timing
   - Evaluating market expectations vs. reality
   - Anticipating sentiment shifts

When analyzing sentiment, you must:
- Summarize key news themes (bullish/bearish)
- Assess analyst consensus and recent changes
- Provide a sentiment score (-10 to +10)
- Identify potential sentiment catalysts
- Give a clear recommendation based on sentiment alone
- Assign a confidence score (1-10)

Format your analysis as a structured report with clear sections.
Be objective about the limitations of sentiment-only analysis."""


class SentimentAgent(BaseAlphaAgent):
    """
    Sentiment analysis agent specializing in news and market psychology.
    """
    
    def __init__(self, risk_profile: str = "neutral", **kwargs):
        """
        Initialize the Sentiment Agent.
        
        Args:
            risk_profile: Either 'averse' or 'neutral'
            **kwargs: Additional arguments for base agent
        """
        super().__init__(
            name="SentimentAnalyst",
            system_message=SENTIMENT_SYSTEM_PROMPT,
            risk_profile=risk_profile,
            tools=[get_stock_news, get_analyst_ratings, analyze_sentiment],
            **kwargs
        )
    
    def fetch_data(self, ticker: str, news_days: int = 14) -> Dict[str, Any]:
        """
        Fetch all sentiment data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            news_days: Number of days to look back for news
            
        Returns:
            Dictionary with all sentiment data
        """
        news = get_stock_news(ticker, days=news_days)
        ratings = get_analyst_ratings(ticker)
        
        # Analyze sentiment for each news article
        news_with_sentiment = []
        for article in news:
            if "error" not in article:
                sentiment = analyze_sentiment(
                    f"{article.get('title', '')} {article.get('description', '')}"
                )
                article["sentiment"] = sentiment
                news_with_sentiment.append(article)
        
        # Calculate overall news sentiment
        if news_with_sentiment:
            avg_score = sum(
                a["sentiment"]["score"] for a in news_with_sentiment
            ) / len(news_with_sentiment)
        else:
            avg_score = 0.5
        
        return {
            "news": news_with_sentiment,
            "analyst_ratings": ratings,
            "overall_news_sentiment": avg_score,
        }
    
    def format_analysis_prompt(self, ticker: str, data: Dict[str, Any]) -> str:
        """
        Format the analysis prompt with fetched data.
        
        Args:
            ticker: Stock ticker symbol
            data: Fetched sentiment data
            
        Returns:
            Formatted prompt string
        """
        ratings = data.get("analyst_ratings", {})
        news = data.get("news", [])
        overall_sentiment = data.get("overall_news_sentiment", 0.5)
        
        # Format top news
        news_summary = ""
        for i, article in enumerate(news[:10], 1):
            sentiment = article.get("sentiment", {})
            sentiment_label = sentiment.get("classification", "neutral")
            news_summary += f"{i}. [{sentiment_label.upper()}] {article.get('title', 'No title')}\n"
            news_summary += f"   Source: {article.get('source', 'Unknown')} | {article.get('published_at', '')}\n"
        
        prompt = f"""
Please analyze the sentiment landscape for: {ticker}

ANALYST RATINGS SUMMARY:
- Consensus: {ratings.get('recommendation_key', 'N/A')}
- Number of Analysts: {ratings.get('number_of_analysts', 0)}
- Current Price: ${ratings.get('current_price', 0):.2f}
- Target Price (Mean): ${ratings.get('target_mean_price', 0):.2f}
- Target Price (High): ${ratings.get('target_high_price', 0):.2f}
- Target Price (Low): ${ratings.get('target_low_price', 0):.2f}
- Upside Potential: {ratings.get('upside_potential', 0):.1f}%

RECENT NEWS (Last 14 Days):
{news_summary if news_summary else "No recent news available."}

OVERALL NEWS SENTIMENT SCORE: {overall_sentiment:.2f} (0=Negative, 0.5=Neutral, 1=Positive)

Based on the above sentiment data, please provide your comprehensive sentiment analysis 
and its implications for the stock's near-term performance.
"""
        return prompt
    
    def generate_analysis(
        self, 
        ticker: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a sentiment analysis for the given ticker.
        
        Args:
            ticker: Stock ticker symbol
            context: Optional additional context
            
        Returns:
            Sentiment analysis string
        """
        data = self.fetch_data(ticker)
        prompt = self.format_analysis_prompt(ticker, data)
        return prompt
