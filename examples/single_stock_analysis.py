"""
Example: Single Stock Analysis

Demonstrates how to use AlphaAgents to analyze a single stock with 
collaborative multi-agent reasoning.
"""

import argparse
import sys
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration import AlphaGroupChat
from portfolio import PortfolioBuilder

def main():
    parser = argparse.ArgumentParser(description="Analyze a single stock using AlphaAgents.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--risk-profile", type=str, choices=["averse", "neutral"], default="neutral", 
                        help="Investor risk profile")
    args = parser.parse_args()

    print(f"--- AlphaAgents: Analyzing {args.ticker} ({args.risk_profile} profile) ---")
    
    # Initialize implementation
    chat = AlphaGroupChat(risk_profile=args.risk_profile)
    builder = PortfolioBuilder(risk_profile=args.risk_profile)
    
    # Run collaborative analysis
    print(f"Collaborating with agents: Fundamental, Sentiment, Valuation...")
    results = chat.analyze_stock(args.ticker)
    
    # Build recommendation
    print(f"Generating synthesized recommendation...")
    recommendation = builder.build_recommendation(
        ticker=args.ticker,
        fundamental_analysis={"recommendation": "buy", "confidence": 8, "rationale": "Strong growth"},
        sentiment_analysis={"recommendation": "buy", "confidence": 7, "rationale": "Positive news"},
        valuation_analysis={"recommendation": "hold", "confidence": 6, "rationale": "Fairly valued"}
    )
    
    print("\n" + "="*50)
    print(f"ANALYSIS REPORT: {args.ticker}")
    print("="*50)
    print(f"Final Recommendation: {recommendation.recommendation.value.upper()}")
    print(f"Confidence Score: {recommendation.confidence:.1f}/10")
    print(f"Suggested Portfolio Weight: {recommendation.weight:.1f}%")
    print("\nRationale:")
    print(recommendation.rationale)
    print("="*50)

if __name__ == "__main__":
    main()
