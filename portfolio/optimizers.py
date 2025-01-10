"""
Advanced Portfolio Optimizers for AlphaAgents.

Implements Black-Litterman and Kelly Criterion for capital allocation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from loguru import logger

class PortfolioOptimizer:
    """
    Advanced mathematical optimization for portfolio weights.
    """
    
    @staticmethod
    def kelly_criterion(win_prob: float, win_loss_ratio: float) -> float:
        """
        Calculates the Kelly bet size.
        k = (p*b - q) / b
        where p = win prob, q = loss prob, b = win/loss ratio
        """
        if win_loss_ratio <= 0:
            return 0
        q = 1 - win_prob
        k = (win_prob * win_loss_ratio - q) / win_loss_ratio
        return max(0, k) # No leverage for now

    @staticmethod
    def black_litterman(returns: pd.DataFrame, views: Dict[str, float], view_confidences: Dict[str, float]) -> pd.Series:
        """
        Simplified Black-Litterman Model.
        Integrates market equilibrium returns with agent views.
        """
        try:
            cov = returns.cov()
            n = len(returns.columns)
            
            # 1. Prior (Market Equilibrium - Simplified as equal weights)
            w_prior = np.ones(n) / n
            tau = 0.05
            
            # 2. Extract Views
            P = np.identity(n) # Each view is for one asset
            Q = np.array([views.get(ticker, 0) for ticker in returns.columns])
            
            # 3. View uncertainty (Omega)
            omega = np.diag([1.0 / view_confidences.get(ticker, 0.1) for ticker in returns.columns])
            
            # 4. BL Formula (Condensed)
            # This is a simplified implementation for demonstration
            sigma_prior = cov * tau
            inv_p = np.linalg.inv(np.linalg.inv(sigma_prior) + P.T @ np.linalg.inv(omega) @ P)
            
            # Combined Return Vector
            mu_prior = returns.mean()
            mu_bl = inv_p @ (np.linalg.inv(sigma_prior) @ mu_prior + P.T @ np.linalg.inv(omega) @ Q)
            
            # Final weights (Mean-Variance optimization of BL returns)
            inv_cov = np.linalg.inv(cov)
            w_bl = inv_cov @ mu_bl
            w_bl /= w_bl.sum() # Normalize
            
            return pd.Series(w_bl, index=returns.columns)
            
        except Exception as e:
            logger.error(f"Black-Litterman Optimization Error: {e}")
            return pd.Series(1/len(returns.columns), index=returns.columns)

def apply_fractional_kelly(weights: pd.Series, fraction: float = 0.5) -> pd.Series:
    """Apply a safety factor to Kelly weights to avoid over-betting."""
    return weights * fraction
