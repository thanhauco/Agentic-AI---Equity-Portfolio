"""Orchestration module for multi-agent coordination."""

from .group_chat import AlphaGroupChat
from .debate_mechanism import (
    DebateManager,
    DebateResult,
    DebateRound,
    AgentPosition,
    Recommendation,
)

__all__ = [
    "AlphaGroupChat",
    "DebateManager",
    "DebateResult",
    "DebateRound",
    "AgentPosition",
    "Recommendation",
]
