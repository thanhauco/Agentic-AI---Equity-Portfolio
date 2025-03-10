"""
Volatility Surface and Term Structure Engine for AlphaAgents.

Models volatility dynamics across strikes and expirations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from scipy.interpolate import griddata
from loguru import logger

class VolatilitySurface:
    """
    Builds and interpolates implied volatility surfaces.
    """
    
    def __init__(self):
        self.surface_data = None
        self.strikes = None
        self.expirations = None

    def build_surface(self, option_chain: pd.DataFrame) -> Dict[str, Any]:
        """
        Construct a volatility surface from option chain data.
        
        Expected columns: strike, expiration_days, implied_vol, option_type
        """
        calls = option_chain[option_chain['option_type'] == 'call']
        
        self.strikes = calls['strike'].unique()
        self.expirations = calls['expiration_days'].unique()
        
        # Create meshgrid
        strike_grid, exp_grid = np.meshgrid(
            np.linspace(self.strikes.min(), self.strikes.max(), 50),
            np.linspace(self.expirations.min(), self.expirations.max(), 20)
        )
        
        # Interpolate volatilities
        points = calls[['strike', 'expiration_days']].values
        values = calls['implied_vol'].values
        
        try:
            vol_grid = griddata(points, values, (strike_grid, exp_grid), method='cubic')
            self.surface_data = {
                "strike_grid": strike_grid,
                "expiration_grid": exp_grid,
                "vol_grid": vol_grid
            }
            
            return {
                "status": "success",
                "n_data_points": len(calls),
                "strike_range": [float(self.strikes.min()), float(self.strikes.max())],
                "expiration_range": [int(self.expirations.min()), int(self.expirations.max())]
            }
        except Exception as e:
            logger.error(f"Surface construction failed: {e}")
            return {"error": str(e)}

    def get_vol_at(self, strike: float, expiration_days: int) -> float:
        """Interpolate volatility at a specific point."""
        if self.surface_data is None:
            return 0.0
            
        try:
            vol = griddata(
                (self.surface_data["strike_grid"].flatten(), self.surface_data["expiration_grid"].flatten()),
                self.surface_data["vol_grid"].flatten(),
                (strike, expiration_days),
                method='linear'
            )
            return float(vol) if not np.isnan(vol) else 0.0
        except:
            return 0.0

    def get_term_structure(self, atm_strike: float) -> Dict[str, Any]:
        """
        Extract term structure (volatility vs time) at a given ATM strike.
        """
        if self.surface_data is None:
            return {"error": "No surface built"}
            
        term_structure = []
        for exp in np.linspace(7, 365, 12):  # 7 days to 1 year
            vol = self.get_vol_at(atm_strike, exp)
            term_structure.append({"days": int(exp), "vol": round(vol * 100, 2)})
            
        return {"atm_strike": atm_strike, "term_structure": term_structure}

    def get_smile(self, expiration_days: int) -> Dict[str, Any]:
        """
        Extract volatility smile at a given expiration.
        """
        if self.surface_data is None:
            return {"error": "No surface built"}
            
        smile = []
        for strike in np.linspace(self.strikes.min(), self.strikes.max(), 20):
            vol = self.get_vol_at(strike, expiration_days)
            smile.append({"strike": round(strike, 2), "vol": round(vol * 100, 2)})
            
        return {"expiration_days": expiration_days, "smile": smile}

def generate_mock_option_chain(spot_price: float, n_strikes: int = 10, n_expirations: int = 5) -> pd.DataFrame:
    """Generate synthetic option chain for testing."""
    np.random.seed(42)
    
    strikes = np.linspace(spot_price * 0.8, spot_price * 1.2, n_strikes)
    expirations = [30, 60, 90, 180, 365][:n_expirations]
    
    data = []
    for strike in strikes:
        for exp in expirations:
            moneyness = strike / spot_price
            base_vol = 0.25 + 0.1 * (moneyness - 1)**2  # Smile effect
            time_effect = 0.02 * np.sqrt(exp / 30)  # Term structure effect
            iv = base_vol + time_effect + np.random.normal(0, 0.01)
            
            data.append({
                "strike": strike,
                "expiration_days": exp,
                "implied_vol": max(0.05, iv),
                "option_type": "call"
            })
            
    return pd.DataFrame(data)
