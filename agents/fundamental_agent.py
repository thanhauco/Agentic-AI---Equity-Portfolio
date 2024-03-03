"""
Fundamental Agent

Specializes in analyzing company fundamentals including financial statements,
SEC filings (10-K/10-Q), and competitive positioning.
"""

from typing import Optional, Dict, Any, List
from agents.base_agent import BaseAlphaAgent
from tools.financial_data import get_stock_info, get_financial_statements, get_key_metrics


FUNDAMENTAL_SYSTEM_PROMPT = """You are a Senior Fundamental Equity Analyst with 20+ years of experience 
analyzing public companies. Your expertise includes:

1. FINANCIAL STATEMENT ANALYSIS:
   - Income statement trends (revenue, margins, earnings growth)
   - Balance sheet health (liquidity, leverage, asset quality)
   - Cash flow analysis (operating, investing, financing activities)
   
2. VALUATION ASSESSMENT:
   - Traditional metrics (P/E, P/B, EV/EBITDA)
   - Discounted cash flow considerations
   - Relative valuation vs sector peers

3. BUSINESS QUALITY:
   - Competitive moat and market position
   - Management quality and capital allocation
   - Sector tailwinds/headwinds

4. RISK FACTORS:
   - Financial leverage and debt sustainability
   - Customer/supplier concentration
   - Regulatory and competitive risks

When analyzing a stock, you must:
- Provide quantitative evidence from financial data
- Compare metrics to industry benchmarks
- Identify key strengths and weaknesses
- Give a clear recommendation (Strong Buy, Buy, Hold, Sell, Strong Sell)
- Assign a confidence score (1-10)

Format your analysis as a structured report with clear sections.
Always ground your conclusions in the data provided."""


class FundamentalAgent(BaseAlphaAgent):
    """
    Fundamental analysis agent specializing in company financials and valuation.
    """
    
    def __init__(self, risk_profile: str = "neutral", **kwargs):
        """
        Initialize the Fundamental Agent.
        
        Args:
            risk_profile: Either 'averse' or 'neutral'
            **kwargs: Additional arguments for base agent
        """
        super().__init__(
            name="FundamentalAnalyst",
            system_message=FUNDAMENTAL_SYSTEM_PROMPT,
            risk_profile=risk_profile,
            tools=[get_stock_info, get_financial_statements, get_key_metrics],
            **kwargs
        )
    
    def fetch_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch all fundamental data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with all fundamental data
        """
        return {
            "stock_info": get_stock_info(ticker),
            "financials": get_financial_statements(ticker),
            "metrics": get_key_metrics(ticker),
        }
    
    def format_analysis_prompt(self, ticker: str, data: Dict[str, Any]) -> str:
        """
        Format the analysis prompt with fetched data.
        
        Args:
            ticker: Stock ticker symbol
            data: Fetched fundamental data
            
        Returns:
            Formatted prompt string
        """
        stock_info = data.get("stock_info", {})
        metrics = data.get("metrics", {})
        
        prompt = f"""
Please analyze the following stock: {ticker} ({stock_info.get('name', 'Unknown')})

COMPANY OVERVIEW:
- Sector: {stock_info['sector']}  # Bug: Will crash if selector is missing
- Industry: {stock_info['industry']}
- Market Cap: ${stock_info['market_cap']:,.0f}

CURRENT VALUATION:
- P/E Ratio: {stock_info.get('pe_ratio', 'N/A')}
- Forward P/E: {stock_info.get('forward_pe', 'N/A')}
- PEG Ratio: {stock_info.get('peg_ratio', 'N/A')}
- Price to Book: {stock_info.get('price_to_book', 'N/A')}
- EV/EBITDA: {metrics.get('valuation', {}).get('ev_to_ebitda', 'N/A')}

PROFITABILITY METRICS:
- Gross Margin: {metrics.get('profitability', {}).get('gross_margin', 'N/A')}
- Operating Margin: {metrics.get('profitability', {}).get('operating_margin', 'N/A')}
- Profit Margin: {metrics.get('profitability', {}).get('profit_margin', 'N/A')}
- ROE: {metrics.get('profitability', {}).get('roe', 'N/A')}
- ROA: {metrics.get('profitability', {}).get('roa', 'N/A')}

GROWTH INDICATORS:
- Revenue Growth: {metrics.get('growth', {}).get('revenue_growth', 'N/A')}
- Earnings Growth: {metrics.get('growth', {}).get('earnings_growth', 'N/A')}

FINANCIAL HEALTH:
- Current Ratio: {metrics.get('financial_health', {}).get('current_ratio', 'N/A')}
- Debt to Equity: {metrics.get('financial_health', {}).get('debt_to_equity', 'N/A')}
- Quick Ratio: {metrics.get('financial_health', {}).get('quick_ratio', 'N/A')}

ANALYST CONSENSUS:
- Target Price: ${stock_info.get('target_mean_price', 'N/A')}
- Recommendation: {stock_info.get('recommendation', 'N/A')}

Please provide your comprehensive fundamental analysis and investment recommendation.
"""
        return prompt
    
    def generate_analysis(
        self, 
        ticker: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a fundamental analysis for the given ticker.
        
        Args:
            ticker: Stock ticker symbol
            context: Optional additional context
            
        Returns:
            Fundamental analysis string
        """
        data = self.fetch_data(ticker)
        prompt = self.format_analysis_prompt(ticker, data)
        
        # In a real implementation, this would call the LLM
        # For now, return the formatted prompt as placeholder
        return prompt
