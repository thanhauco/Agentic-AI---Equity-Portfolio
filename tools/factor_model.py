"""
Factor Model Engine for AlphaAgents.

Implements Fama-French factor decomposition for return attribution.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from loguru import logger

class FactorModelEngine:
    """
    Factor-based return attribution using Fama-French style decomposition.
    """
    
    # Standard factor names
    FACTORS = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]
    
    def __init__(self):
        self.factor_returns = None
        self.loadings = {}

    def get_mock_factor_returns(self, periods: int = 252) -> pd.DataFrame:
        """
        Generate mock factor returns for demonstration.
        In production, this would fetch from Ken French's data library.
        """
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=periods, freq='D')
        
        data = {
            "MKT": np.random.normal(0.0005, 0.01, periods),  # Market excess return
            "SMB": np.random.normal(0.0001, 0.005, periods),  # Small minus Big
            "HML": np.random.normal(0.0001, 0.005, periods),  # High minus Low (Value)
            "RMW": np.random.normal(0.0001, 0.004, periods),  # Robust minus Weak (Profitability)
            "CMA": np.random.normal(0.0001, 0.004, periods),  # Conservative minus Aggressive (Investment)
            "MOM": np.random.normal(0.0002, 0.008, periods),  # Momentum
        }
        
        self.factor_returns = pd.DataFrame(data, index=dates)
        return self.factor_returns

    def estimate_factor_loadings(self, stock_returns: pd.Series) -> Dict[str, float]:
        """
        Estimate factor loadings (betas) via OLS regression.
        """
        if self.factor_returns is None:
            self.get_mock_factor_returns(len(stock_returns))
            
        # Align dates
        aligned = pd.concat([stock_returns, self.factor_returns], axis=1, join='inner')
        aligned.columns = ['stock'] + self.FACTORS
        
        # Simple OLS
        X = aligned[self.FACTORS].values
        y = aligned['stock'].values
        
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        try:
            betas = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            
            result = {"alpha": betas[0]}
            for i, factor in enumerate(self.FACTORS):
                result[factor] = betas[i + 1]
                
            self.loadings = result
            return result
            
        except Exception as e:
            logger.error(f"Factor estimation error: {e}")
            return {}

    def decompose_returns(self, stock_returns: pd.Series) -> Dict[str, Any]:
        """
        Decompose stock returns into factor contributions.
        """
        loadings = self.estimate_factor_loadings(stock_returns)
        
        if not loadings or self.factor_returns is None:
            return {"error": "Could not decompose returns"}
            
        contributions = {}
        total_explained = 0
        
        for factor in self.FACTORS:
            factor_contribution = loadings.get(factor, 0) * self.factor_returns[factor].mean() * 252
            contributions[factor] = round(factor_contribution * 100, 2)
            total_explained += factor_contribution
            
        unexplained = stock_returns.mean() * 252 - total_explained - loadings.get("alpha", 0) * 252
        
        return {
            "factor_contributions_pct": contributions,
            "alpha_annualized_pct": round(loadings.get("alpha", 0) * 252 * 100, 2),
            "unexplained_pct": round(unexplained * 100, 2),
            "loadings": loadings
        }

def get_factor_exposure_report(decomposition: Dict[str, Any]) -> str:
    """Generate a human-readable factor exposure report."""
    if "error" in decomposition:
        return f"Error: {decomposition['error']}"
        
    report = "### 📊 Fama-French Factor Decomposition Report\n\n"
    report += f"**Alpha (Annualized)**: {decomposition['alpha_annualized_pct']:.2f}%\n\n"
    report += "**Factor Contributions**:\n"
    
    for factor, contribution in decomposition["factor_contributions_pct"].items():
        report += f"- **{factor}**: {contribution:.2f}%\n"
        
    report += f"\n**Unexplained**: {decomposition['unexplained_pct']:.2f}%\n"
    return report
