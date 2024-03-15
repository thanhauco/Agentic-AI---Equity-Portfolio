"""
Backtesting engine for evaluating AlphaAgents performance.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from loguru import logger
import yfinance as yf

class BacktestEngine:
    """
    Simulates portfolio performance based on AlphaAgents recommendations.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,  # 0.1% per trade
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.portfolio_history = []
        
    def run(
        self,
        recommendations: List[Dict[str, Any]],
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Run backtest against historical data.
        
        Args:
            recommendations: List of stock recommendations with weights
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        tickers = [r["ticker"] for r in recommendations]
        weights = {r["ticker"]: r["weight"] / 100.0 for r in recommendations}
        
        # Adjust weights to include cash
        total_weight = sum(weights.values())
        if total_weight > 1.0:
            logger.warning("Total weight exceeds 100%, normalizing...")
            weights = {k: v / total_weight for k, v in weights.items()}
            
        cash_weight = 1.0 - sum(weights.values())
        
        # Fetch historical data
        data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = [tickers[0]]
            
        # Calculate daily returns
        returns = data.pct_change().dropna()
        
        # Calculate portfolio returns
        port_returns = pd.Series(0, index=returns.index)
        for ticker, weight in weights.items():
            if ticker in returns.columns:
                port_returns += returns[ticker] * weight
        
        # Add risk-free return for cash (simulated at 2% annual)
        port_returns += (0.02 / 252) * cash_weight
        
        # Calculate cumulative returns
        cum_returns = (1 + port_returns).cumprod()
        portfolio_value = self.initial_capital * cum_returns
        
        # Calculate metrics
        total_return = (cum_returns.iloc[-1] - 1) * 100
        annualized_return = ((1 + total_return/100)**(252 / len(port_returns)) - 1) * 100
        volatility = port_returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (annualized_return - 2) / volatility if volatility > 0 else 0
        
        drawdown = 1 - cum_returns / cum_returns.cummax()
        max_drawdown = drawdown.max() * 100
        
        return {
            "total_return_pct": total_return,
            "annualized_return_pct": annualized_return,
            "volatility_pct": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown,
            "final_value": portfolio_value.iloc[-1],
            "history": portfolio_value.to_dict()
        }

def calculate_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate standard performance metrics."""
    total_ret = (1 + returns).prod() - 1
    ann_ret = (1 + total_ret)**(252 / len(returns)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / vol if vol > 0 else 0
    return {
        "total_return": total_ret,
        "annualized_return": ann_ret,
        "volatility": vol,
        "sharpe": sharpe
    }
