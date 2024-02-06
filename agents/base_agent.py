"""
Base Agent Module

Provides the foundational agent class that all specialized agents inherit from.
Uses AutoGen's AssistantAgent as the base.
"""

from typing import Optional, Dict, Any, List, Callable
from autogen import AssistantAgent
from config import get_llm_config


class BaseAlphaAgent(AssistantAgent):
    """
    Base class for all AlphaAgents specialized agents.
    
    Extends AutoGen's AssistantAgent with:
    - Risk profile support
    - Custom tool registration
    - Enhanced logging
    """
    
    def __init__(
        self,
        name: str,
        system_message: str,
        risk_profile: str = "neutral",
        tools: Optional[List[Callable]] = None,
        **kwargs
    ):
        """
        Initialize a base alpha agent.
        
        Args:
            name: Agent identifier
            system_message: Base system prompt for the agent
            risk_profile: Either 'averse' or 'neutral'
            tools: List of callable tools the agent can use
            **kwargs: Additional arguments for AssistantAgent
        """
        self.risk_profile = risk_profile
        self._tools: Dict[str, Callable] = {}
        
        # Apply risk profile modifications to system message
        enhanced_message = self._apply_risk_profile(system_message)
        
        # Get LLM config
        llm_config = get_llm_config()
        
        # Register tools if provided
        if tools:
            for tool in tools:
                self.register_tool(tool)
        
        super().__init__(
            name=name,
            system_message=enhanced_message,
            llm_config=llm_config,
            **kwargs
        )
    
    def _apply_risk_profile(self, base_message: str) -> str:
        """
        Modify system message based on risk profile.
        
        Args:
            base_message: Original system message
            
        Returns:
            Modified system message with risk considerations
        """
        if self.risk_profile == "averse":
            risk_suffix = """

RISK PROFILE: RISK-AVERSE
- Prioritize capital preservation over growth
- Favor established companies with strong balance sheets  
- Weight negative signals more heavily than positive ones
- Recommend conservative position sizes
- Avoid highly volatile or speculative investments"""
        else:
            risk_suffix = """

RISK PROFILE: RISK-NEUTRAL  
- Balance growth potential with downside risks
- Consider both value and momentum opportunities
- Weight positive and negative signals equally
- Use standard position sizing based on conviction
- Open to calculated risks with appropriate reward potential"""
        
        return base_message + risk_suffix
    
    def register_tool(self, tool: Callable) -> None:
        """
        Register a tool function for the agent to use.
        
        Args:
            tool: Callable function with docstring describing its purpose
        """
        tool_name = tool.__name__
        self._tools[tool_name] = tool
    
    def get_tools(self) -> Dict[str, Callable]:
        """Get all registered tools."""
        return self._tools.copy()
    
    def generate_analysis(self, ticker: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate analysis for a given stock ticker.
        Must be implemented by subclasses.
        
        Args:
            ticker: Stock ticker symbol
            context: Optional additional context
            
        Returns:
            Analysis string
        """
        raise NotImplementedError("Subclasses must implement generate_analysis")
