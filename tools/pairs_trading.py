"""
Cointegration and Pairs Trading Engine for AlphaAgents.

Implements statistical arbitrage strategies using cointegration tests.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from loguru import logger

def _get_statsmodels():
    try:
        from statsmodels.tsa.stattools import coint, adfuller
        return coint, adfuller
    except ImportError:
        return None, None

class PairsTrader:
    """
    Statistical Arbitrage Engine using Cointegration Analysis.
    """
    
    def __init__(self, lookback: int = 252):
        self.lookback = lookback
        self.pairs = []

    def find_cointegrated_pairs(self, price_data: pd.DataFrame, pvalue_threshold: float = 0.05) -> List[Dict[str, Any]]:
        """
        Identify cointegrated pairs from a universe of stocks.
        """
        coint_func, _ = _get_statsmodels()
        if coint_func is None:
            return [{"error": "statsmodels not available"}]
            
        n = price_data.shape[1]
        tickers = price_data.columns.tolist()
        pairs = []
        
        for i in range(n):
            for j in range(i+1, n):
                try:
                    _, pvalue, _ = coint_func(price_data.iloc[:, i], price_data.iloc[:, j])
                    if pvalue < pvalue_threshold:
                        pairs.append({
                            "pair": (tickers[i], tickers[j]),
                            "pvalue": round(pvalue, 4),
                            "is_cointegrated": True
                        })
                except Exception as e:
                    continue
                    
        self.pairs = pairs
        return sorted(pairs, key=lambda x: x["pvalue"])

    def calculate_spread(self, series_a: pd.Series, series_b: pd.Series) -> Tuple[pd.Series, float]:
        """
        Calculate the spread and hedge ratio between two cointegrated series.
        """
        # Simple linear regression for hedge ratio
        hedge_ratio = np.polyfit(series_b, series_a, 1)[0]
        spread = series_a - hedge_ratio * series_b
        return spread, hedge_ratio

    def generate_signals(self, spread: pd.Series, z_entry: float = 2.0, z_exit: float = 0.5) -> pd.Series:
        """
        Generate trading signals based on spread z-score.
        
        +1 = Long spread (Long A, Short B)
        -1 = Short spread (Short A, Long B)
        0 = Neutral / Exit
        """
        mean = spread.rolling(window=self.lookback).mean()
        std = spread.rolling(window=self.lookback).std()
        z_score = (spread - mean) / std
        
        signals = pd.Series(0, index=spread.index)
        signals[z_score > z_entry] = -1  # Short the spread
        signals[z_score < -z_entry] = 1   # Long the spread
        signals[(z_score > -z_exit) & (z_score < z_exit)] = 0  # Exit
        
        return signals

    def backtest_pair(self, series_a: pd.Series, series_b: pd.Series) -> Dict[str, Any]:
        """
        Simple backtest of a pairs trading strategy.
        """
        spread, hedge_ratio = self.calculate_spread(series_a, series_b)
        signals = self.generate_signals(spread)
        
        # Calculate returns
        spread_returns = spread.pct_change()
        strategy_returns = signals.shift(1) * spread_returns
        strategy_returns = strategy_returns.dropna()
        
        cumulative_return = (1 + strategy_returns).prod() - 1
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        return {
            "hedge_ratio": round(hedge_ratio, 4),
            "total_return_pct": round(cumulative_return * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "n_trades": int((signals.diff() != 0).sum()),
            "max_drawdown_pct": round(self._max_drawdown(strategy_returns) * 100, 2)
        }

    def _max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
