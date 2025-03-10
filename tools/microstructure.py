"""
Order Flow and Market Microstructure Engine for AlphaAgents.

Analyzes bid-ask dynamics, order imbalance, and liquidity metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from loguru import logger

class MarketMicrostructure:
    """
    Analyzes market microstructure for execution quality and alpha signals.
    """
    
    def calculate_order_imbalance(self, trades: pd.DataFrame) -> pd.Series:
        """
        Calculate order imbalance ratio.
        
        OI = (Buy Volume - Sell Volume) / Total Volume
        
        Expected columns: timestamp, price, volume, side ('B' or 'S')
        """
        trades = trades.copy()
        trades['buy_vol'] = np.where(trades['side'] == 'B', trades['volume'], 0)
        trades['sell_vol'] = np.where(trades['side'] == 'S', trades['volume'], 0)
        
        resampled = trades.resample('1min', on='timestamp').agg({
            'buy_vol': 'sum',
            'sell_vol': 'sum'
        })
        
        total = resampled['buy_vol'] + resampled['sell_vol']
        imbalance = (resampled['buy_vol'] - resampled['sell_vol']) / total
        return imbalance.fillna(0)

    def calculate_vwap(self, trades: pd.DataFrame) -> float:
        """Calculate Volume-Weighted Average Price."""
        return (trades['price'] * trades['volume']).sum() / trades['volume'].sum()

    def calculate_bid_ask_spread(self, quotes: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze bid-ask spread dynamics.
        
        Expected columns: timestamp, bid, ask, bid_size, ask_size
        """
        quotes = quotes.copy()
        quotes['spread'] = quotes['ask'] - quotes['bid']
        quotes['mid'] = (quotes['ask'] + quotes['bid']) / 2
        quotes['spread_bps'] = quotes['spread'] / quotes['mid'] * 10000
        
        return {
            "avg_spread_bps": round(quotes['spread_bps'].mean(), 2),
            "max_spread_bps": round(quotes['spread_bps'].max(), 2),
            "min_spread_bps": round(quotes['spread_bps'].min(), 2),
            "avg_bid_size": round(quotes['bid_size'].mean(), 0),
            "avg_ask_size": round(quotes['ask_size'].mean(), 0)
        }

    def kyle_lambda(self, trades: pd.DataFrame, returns: pd.Series) -> float:
        """
        Estimate Kyle's Lambda (price impact coefficient).
        
        Measures how much price moves per unit of signed order flow.
        """
        trades = trades.copy()
        trades['signed_vol'] = np.where(trades['side'] == 'B', trades['volume'], -trades['volume'])
        
        # Aggregate to match return frequency
        signed_flow = trades.resample('1min', on='timestamp')['signed_vol'].sum()
        returns_aligned = returns.reindex(signed_flow.index).dropna()
        signed_flow = signed_flow.reindex(returns_aligned.index).dropna()
        
        if len(signed_flow) < 2:
            return 0.0
            
        # Simple regression: returns = lambda * signed_flow
        coef = np.polyfit(signed_flow, returns_aligned, 1)[0]
        return round(coef * 1e6, 4)  # Scaled for readability

    def amihud_illiquidity(self, returns: pd.Series, volume: pd.Series) -> float:
        """
        Calculate Amihud Illiquidity Ratio.
        
        ILLIQ = |Return| / Dollar Volume
        Higher values = less liquid
        """
        daily_illiq = returns.abs() / volume
        avg_illiq = daily_illiq.mean()
        return round(avg_illiq * 1e6, 4)  # Scaled

def generate_mock_trades(n_trades: int = 1000, base_price: float = 100.0) -> pd.DataFrame:
    """Generate synthetic trade data for testing."""
    np.random.seed(42)
    
    timestamps = pd.date_range(start='2024-01-01 09:30', periods=n_trades, freq='1s')
    prices = base_price + np.cumsum(np.random.normal(0, 0.01, n_trades))
    volumes = np.random.randint(100, 5000, n_trades)
    sides = np.random.choice(['B', 'S'], n_trades, p=[0.52, 0.48])  # Slight buy bias
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume': volumes,
        'side': sides
    })
