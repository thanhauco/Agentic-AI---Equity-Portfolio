"""
Macro Agent for AlphaAgents.

Analyzes broad economic trends, interest rates, and geopolitical shifts.
"""

from .base_agent import BaseAlphaAgent
from typing import Dict, Any, List

class MacroAgent(BaseAlphaAgent):
    """
    Agent specialized in global macro and interest rate cycles.
    """
    
    def __init__(self, model: str = "gpt-4o"):
        super().__init__(
            name="MacroStrategist",
            role="Global Macro & Rates Strategist",
            goal="Identify thematic tailwinds and headwinds in the global economy.",
            model=model
        )

    def get_system_prompt(self) -> str:
        return f"""You are the {self.role}. Your expertise is in connecting macro dots.
Consider:
1. Central Bank policy (Fed, ECB, BoJ)
2. Inflationary regimes (CPI, PCE)
3. Geopolitical risk factors (Supply chains, Energy)
4. Yield curve dynamics

Evaluate how these factors impact specific equity sectors.
Provide your analysis in a structured, high-conviction format."""

    def analyze_macro_context(self, context: str = "Current 2024 Market") -> str:
        """
        Produce a macro analysis report.
        """
        prompt = f"Perform a deep dive into the following macro context: {context}. Highlight impact on Technology, Energy, and Financial sectors."
        return self.chat(prompt)
