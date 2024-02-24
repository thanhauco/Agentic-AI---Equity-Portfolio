"""
Example: Portfolio Construction

Demonstrates how to use AlphaAgents to construct a portfolio from a 
universe of stocks using collaborative multi-agent reasoning and debate.
"""

import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration import AlphaGroupChat
from portfolio import PortfolioBuilder

def main():
    parser = argparse.ArgumentParser(description="Construct a portfolio using AlphaAgents.")
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT,GOOGL,AMZN,META", 
                        help="Comma-separated stock tickers")
    parser.add_argument("--risk-profile", type=str, choices=["averse", "neutral"], default="neutral", 
                        help="Investor risk profile")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]
    
    print(f"--- AlphaAgents: Portfolio Construction ---")
    print(f"Universe: {', '.join(tickers)}")
    print(f"Risk Profile: {args.risk_profile}")
    
    chat = AlphaGroupChat(risk_profile=args.risk_profile)
    builder = PortfolioBuilder(risk_profile=args.risk_profile)
    
    all_analyses = []
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        # In a real run, this would call agents. Here we mock for demo.
        # chat.analyze_stock(ticker) 
        
        # Simulated analysis results to show variety
        mock_data = {
            "ticker": ticker,
            "fundamental": {"recommendation": "buy", "confidence": 7, "weight": 5},
            "sentiment": {"recommendation": "buy", "confidence": 8, "weight": 6},
            "valuation": {"recommendation": "hold", "confidence": 5, "weight": 2}
        }
        all_analyses.append(mock_data)
    
    print("\n" + "="*50)
    print("CONSTRUCTING PORTFOLIO")
    print("="*50)
    
    portfolio = builder.build_portfolio(all_analyses)
    
    print(f"Total Invested: {portfolio.total_invested_weight:.1f}%")
    print(f"Cash Position: {portfolio.cash_weight:.1f}%")
    print(f"Selected Positions: {len(portfolio.stocks)}")
    
    print("\nAllocations:")
    for stock in portfolio.stocks:
        print(f"- {stock.ticker:6} | {stock.recommendation.value.upper():12} | Weight: {stock.weight:.1f}% | Conf: {stock.confidence:.1f}")
    
    print("="*50)

if __name__ == "__main__":
    main()
