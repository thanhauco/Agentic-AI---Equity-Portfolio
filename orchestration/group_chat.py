"""
Group Chat Manager

Orchestrates collaboration between specialized agents using AutoGen's
GroupChat functionality.
"""

from typing import List, Optional, Dict, Any, Callable
from autogen import GroupChat, GroupChatManager, UserProxyAgent
from agents import FundamentalAgent, SentimentAgent, ValuationAgent
from config import get_llm_config


class AlphaGroupChat:
    """
    Manages multi-agent collaboration for equity analysis.
    
    Coordinates between Fundamental, Sentiment, and Valuation agents
    to produce comprehensive stock analyses.
    """
    
    def __init__(
        self,
        risk_profile: str = "neutral",
        max_rounds: int = 10,
        human_input_mode: str = "NEVER"
    ):
        """
        Initialize the group chat with all specialized agents.
        
        Args:
            risk_profile: Either 'averse' or 'neutral'
            max_rounds: Maximum conversation rounds
            human_input_mode: How to handle human input
        """
        self.risk_profile = risk_profile
        self.max_rounds = max_rounds
        
        # Initialize specialized agents
        self.fundamental_agent = FundamentalAgent(risk_profile=risk_profile)
        self.sentiment_agent = SentimentAgent(risk_profile=risk_profile)
        self.valuation_agent = ValuationAgent(risk_profile=risk_profile)
        
        # Initialize user proxy for orchestration
        self.user_proxy = UserProxyAgent(
            name="Portfolio_Manager",
            human_input_mode=human_input_mode,
            system_message="""You are the Portfolio Manager coordinating the analysis.
            After all agents have provided their analysis, synthesize their findings
            into a final recommendation with portfolio weight suggestion.""",
            code_execution_config=False,
        )
        
        # Create group chat
        self.agents = [
            self.user_proxy,
            self.fundamental_agent,
            self.sentiment_agent,
            self.valuation_agent,
        ]
        
        self.group_chat = GroupChat(
            agents=self.agents,
            messages=[],
            max_round=max_rounds,
            speaker_selection_method="round_robin",
        )
        
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=get_llm_config(),
        )
    
    def analyze_stock(self, ticker: str) -> Dict[str, Any]:
        """
        Run collaborative analysis on a stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with analysis results from all agents
        """
        # Prepare initial message
        initial_message = f"""
        Please analyze the stock: {ticker}
        
        Each agent should provide their specialized analysis:
        1. FundamentalAnalyst: Analyze financials and business fundamentals
        2. SentimentAnalyst: Analyze news and market sentiment
        3. ValuationAnalyst: Analyze technicals and valuation
        
        After all analyses, reach a consensus on the recommendation.
        Risk Profile: {self.risk_profile}
        """
        
        # Collect individual analyses first
        results = {
            "ticker": ticker,
            "risk_profile": self.risk_profile,
            "fundamental_analysis": self.fundamental_agent.generate_analysis(ticker),
            "sentiment_analysis": self.sentiment_agent.generate_analysis(ticker),
            "valuation_analysis": self.valuation_agent.generate_analysis(ticker),
        }
        
        return results
    
    def run_collaborative_session(self, ticker: str) -> str:
        """
        Run a full collaborative session with agent interaction.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Full conversation transcript
        """
        initial_message = f"""
        Analyze the stock: {ticker}
        Risk Profile: {self.risk_profile}
        
        Each agent should provide their analysis, then discuss and reach consensus.
        """
        
        # Start the group chat
        self.user_proxy.initiate_chat(
            self.manager,
            message=initial_message,
        )
        
        # Return conversation history
        return self._format_conversation()
    
    def _format_conversation(self) -> str:
        """Format conversation history as string."""
        output = []
        for msg in self.group_chat.messages:
            sender = msg.get("name", "Unknown")
            content = msg.get("content", "")
            output.append(f"[{sender}]:\n{content}\n")
        return "\n".join(output)
    
    def get_agent_by_name(self, name: str):
        """Get an agent by name."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None
