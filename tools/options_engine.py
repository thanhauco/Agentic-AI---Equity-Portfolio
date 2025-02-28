"""
Options Greeks Calculator for AlphaAgents.

Implements Black-Scholes pricing and Greeks computation for derivatives analysis.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Any
from loguru import logger

class BlackScholesEngine:
    """
    Options pricing and Greeks calculator using Black-Scholes model.
    """
    
    @staticmethod
    def calculate_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple:
        """Calculate d1 and d2 parameters."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    @staticmethod
    def price_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Price a European call option."""
        d1, d2 = BlackScholesEngine.calculate_d1_d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def price_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Price a European put option."""
        d1, d2 = BlackScholesEngine.calculate_d1_d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> Dict[str, float]:
        """
        Calculate all Greeks for an option.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Implied volatility
            option_type: "call" or "put"
        """
        d1, d2 = BlackScholesEngine.calculate_d1_d2(S, K, T, r, sigma)
        sqrt_T = np.sqrt(T)
        
        # Delta
        if option_type == "call":
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
            
        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
        
        # Theta
        if option_type == "call":
            theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt_T) 
                     - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt_T) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            
        # Vega (same for calls and puts)
        vega = S * norm.pdf(d1) * sqrt_T / 100
        
        # Rho
        if option_type == "call":
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "theta": round(theta, 4),
            "vega": round(vega, 4),
            "rho": round(rho, 4)
        }

    @staticmethod
    def implied_volatility(market_price: float, S: float, K: float, T: float, r: float, option_type: str = "call", tol: float = 1e-5) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        """
        sigma = 0.25  # Initial guess
        for _ in range(100):
            if option_type == "call":
                price = BlackScholesEngine.price_call(S, K, T, r, sigma)
            else:
                price = BlackScholesEngine.price_put(S, K, T, r, sigma)
                
            d1, _ = BlackScholesEngine.calculate_d1_d2(S, K, T, r, sigma)
            vega = S * norm.pdf(d1) * np.sqrt(T)
            
            if vega < 1e-10:
                break
                
            diff = market_price - price
            if abs(diff) < tol:
                return sigma
                
            sigma += diff / vega
            
        return sigma
