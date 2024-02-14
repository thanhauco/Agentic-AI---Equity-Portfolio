"""Risk profiles module for investment strategy configuration."""

from .risk_averse import get_risk_averse_prompts, RISK_AVERSE_FUNDAMENTAL_PROMPT
from .risk_neutral import get_risk_neutral_prompts, RISK_NEUTRAL_FUNDAMENTAL_PROMPT

__all__ = [
    "get_risk_averse_prompts",
    "get_risk_neutral_prompts",
    "RISK_AVERSE_FUNDAMENTAL_PROMPT",
    "RISK_NEUTRAL_FUNDAMENTAL_PROMPT",
]


def get_prompts_for_profile(profile: str) -> dict:
    """
    Get prompts for the specified risk profile.
    
    Args:
        profile: Either 'averse' or 'neutral'
        
    Returns:
        Dictionary of prompts for each agent type
    """
    if profile == "averse":
        return get_risk_averse_prompts()
    elif profile == "neutral":
        return get_risk_neutral_prompts()
    else:
        raise ValueError(f"Unknown risk profile: {profile}. Use 'averse' or 'neutral'.")
