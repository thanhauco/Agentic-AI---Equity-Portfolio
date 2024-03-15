"""
Example: Backtesting AlphaAgents

Demonstrates how to run a backtest on portfolio recommendations.
"""

import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.backtest import BacktestEngine

def main():
    print("--- AlphaAgents: Backtesting Simulation ---")
    
    # Mock recommendations (in real use, these come from AlphaAgents)
    recommendations = [
        {"ticker": "AAPL", "weight": 20.0},
        {"ticker": "MSFT", "weight": 20.0},
        {"ticker": "GOOGL", "weight": 15.0},
        {"ticker": "NVDA", "weight": 25.0},
        {"ticker": "META", "weight": 10.0},
    ]
    
    engine = BacktestEngine(initial_capital=100000.0)
    
    # Run backtest for the last year
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    results = engine.run(recommendations, start_date, end_date)
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS (1 YEAR)")
    print("="*50)
    print(f"Initial Capital: $100,000.00")
    print(f"Final Value:     ${results['final_value']:,.2f}")
    print(f"Total Return:    {results['total_return_pct']:.2f}%")
    print(f"Annual Return:   {results['annualized_return_pct']:.2f}%")
    print(f"Volatility:      {results['volatility_pct']:.2f}%")
    print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:    {results['max_drawdown_pct']:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
