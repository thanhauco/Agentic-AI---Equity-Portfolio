"""
Alpha Signal Generator for AlphaAgents.

Aggregates fundamental, technical, and sentiment signals into a 
unified predictive Alpha Score.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from loguru import logger

class AlphaSignalGenerator:
    """
    Combines multi-modal data into a single conviction score.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "fundamental": 0.40,
            "sentiment": 0.30,
            "technical": 0.20,
            "neural": 0.10
        }

    def generate_signal(self, 
                        fundamental_data: Dict[str, Any],
                        sentiment_score: float,
                        technical_indicators: Dict[str, Any],
                        neural_forecast: float) -> Dict[str, Any]:
        """
        Synthesize disparate data points into an Alpha Score (-1 to 1).
        """
        try:
            # 1. Fundamental Component (Mock logic based on ratios)
            f_score = 0
            pe = fundamental_data.get("pe_ratio", 20)
            roe = fundamental_data.get("roe", 0.15)
            f_score = 1.0 if pe < 15 and roe > 0.2 else 0.5 if pe < 25 else -0.5
            
            # 2. Sentiment Component (Score is usually 0 to 1, map to -1 to 1)
            s_score = (sentiment_score - 0.5) * 2
            
            # 3. Technical Component
            rsi = technical_indicators.get("RSI", 50)
            t_score = 1.0 if rsi < 30 else -1.0 if rsi > 70 else 0
            
            # 4. Neural Component (Forecast pct change)
            n_score = np.clip(neural_forecast * 10, -1, 1)
            
            # Weighted aggregation
            alpha_score = (
                f_score * self.weights["fundamental"] +
                s_score * self.weights["sentiment"] +
                t_score * self.weights["technical"] +
                n_score * self.weights["neural"]
            )
            
            conviction = "High" if abs(alpha_score) > 0.6 else "Medium" if abs(alpha_score) > 0.3 else "Low"
            
            return {
                "alpha_score": float(alpha_score),
                "conviction": conviction,
                "components": {
                    "fundamental": f_score,
                    "sentiment": s_score,
                    "technical": t_score,
                    "neural": n_score
                },
                "verdict": "BULLISH" if alpha_score > 0.2 else "BEARISH" if alpha_score < -0.2 else "NEUTRAL"
            }
        except Exception as e:
            logger.error(f"Alpha Signal Generation Error: {e}")
            return {"error": str(e)}

def get_alpha_signal_report(signal: Dict[str, Any]) -> str:
    """Generate a human-readable alpha report."""
    if "error" in signal:
        return f"Error generating alpha signal: {signal['error']}"
        
    return f"""### 🚀 Alpha Conviction Report
**Verdict**: {signal['verdict']}
**Score**: {signal['alpha_score']:.2f}
**Conviction**: {signal['conviction']}

**Component Breakdown**:
- Fundamental: {signal['components']['fundamental']:.2f}
- Sentiment: {signal['components']['sentiment']:.2f}
- Technical: {signal['components']['technical']:.2f}
- Neural: {signal['components']['neural']:.2f}
"""
