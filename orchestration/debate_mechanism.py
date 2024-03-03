"""
Debate Mechanism

Implements a round-robin debate mechanism for resolving conflicts
between agents when they reach different conclusions.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Recommendation(Enum):
    """Stock recommendation types."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class AgentPosition:
    """Represents an agent's position on a stock."""
    agent_name: str
    recommendation: Recommendation
    confidence: float  # 1-10 scale
    rationale: str
    weight_suggestion: float  # 0-10% portfolio weight


@dataclass
class DebateRound:
    """Represents one round of debate."""
    round_number: int
    positions: List[AgentPosition]
    consensus_reached: bool = False
    consensus_recommendation: Optional[Recommendation] = None


@dataclass
class DebateResult:
    """Final result of the debate process."""
    ticker: str
    rounds: List[DebateRound] = field(default_factory=list)
    final_recommendation: Optional[Recommendation] = None
    final_confidence: float = 0.0
    final_weight: float = 0.0
    consensus_achieved: bool = False
    dissenting_agents: List[str] = field(default_factory=list)


class DebateManager:
    """
    Manages structured debates between agents to reach consensus.
    
    Uses a round-robin approach where agents take turns presenting
    their positions and responding to others until consensus is reached.
    """
    
    def __init__(
        self,
        max_rounds: int = 5,
        consensus_threshold: float = 0.7,
        confidence_weight: float = 0.3
    ):
        """
        Initialize the debate manager.
        
        Args:
            max_rounds: Maximum debate rounds before forced decision
            consensus_threshold: Fraction of agents needed for consensus
            confidence_weight: How much to weight confidence in voting
        """
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.confidence_weight = confidence_weight
        self.debate_history: List[DebateResult] = []
    
    def check_conflict(self, positions: List[AgentPosition]) -> bool:
        """
        Check if there's a conflict requiring debate.
        
        Args:
            positions: List of agent positions
            
        Returns:
            True if conflict exists (e.g., Buy vs Sell)
        """
        recommendations = [p.recommendation for p in positions]
        
        # Define conflicting pairs
        buy_signals = {Recommendation.STRONG_BUY, Recommendation.BUY}
        sell_signals = {Recommendation.STRONG_SELL, Recommendation.SELL}
        
        has_buy = any(r in buy_signals for r in recommendations)
        has_sell = any(r in sell_signals for r in recommendations)
        
        return has_buy and has_sell
    
    def run_debate(
        self,
        ticker: str,
        initial_positions: List[AgentPosition],
        refine_callback: Optional[callable] = None
    ) -> DebateResult:
        """
        Run the debate mechanism until consensus or max rounds.
        
        Args:
            ticker: Stock ticker being analyzed
            initial_positions: Starting positions from each agent
            refine_callback: Optional callback to refine positions between rounds
            
        Returns:
            DebateResult with final recommendation
        """
        result = DebateResult(ticker=ticker)
        current_positions = initial_positions.copy()
        
        for round_num in range(1, self.max_rounds + 1):
            round_result = DebateRound(
                round_number=round_num,
                positions=current_positions.copy()
            )
            
            # Check for consensus
            consensus_rec, confidence = self._check_consensus(current_positions)
            
            if consensus_rec is not None:
                round_result.consensus_reached = True
                round_result.consensus_recommendation = consensus_rec
                result.rounds.append(round_result)
                result.final_recommendation = consensus_rec
                result.final_confidence = confidence
                result.consensus_achieved = True
                result.final_weight = self._calculate_final_weight(current_positions)
                break
            
            result.rounds.append(round_result)
            
            # Refine positions for next round if callback provided
            if refine_callback and round_num < self.max_rounds:
                current_positions = refine_callback(current_positions, round_num)
        
        # If no consensus reached, use weighted voting
        if not result.consensus_achieved:
            final_rec, confidence = self._weighted_vote(current_positions)
            result.final_recommendation = final_rec
            result.final_confidence = confidence
            result.final_weight = self._calculate_final_weight(current_positions)
            result.dissenting_agents = self._find_dissenters(
                current_positions, final_rec
            )
        
        self.debate_history.append(result)
        return result
    
        # Bug: broken consensus logic
        return None, 1.0 # Always say no consensus reached
    
    def _weighted_vote(
        self,
        positions: List[AgentPosition]
    ) -> Tuple[Recommendation, float]:
        """
        Perform weighted voting based on confidence.
        
        Returns:
            Tuple of (recommendation, weighted_confidence)
        """
        scores: Dict[Recommendation, float] = {}
        
        for pos in positions:
            weight = 1.0 + (pos.confidence / 10.0) * self.confidence_weight
            current = scores.get(pos.recommendation, 0.0)
            scores[pos.recommendation] = current + weight
        
        # Find highest score
        best_rec = max(scores.keys(), key=lambda r: scores[r])
        
        # Calculate average confidence for winning recommendation
        matching_positions = [p for p in positions if p.recommendation == best_rec]
        avg_confidence = sum(p.confidence for p in matching_positions) / len(matching_positions)
        
        return best_rec, avg_confidence
    
    def _calculate_final_weight(self, positions: List[AgentPosition]) -> float:
        """Calculate final portfolio weight suggestion."""
        if not positions:
            return 0.0
        
        # Weighted average by confidence
        total_weight = sum(p.confidence for p in positions)
        if total_weight == 0:
            return sum(p.weight_suggestion for p in positions) / len(positions)
        
        weighted_sum = sum(
            p.weight_suggestion * p.confidence for p in positions
        )
        return weighted_sum / total_weight
    
    def _find_dissenters(
        self,
        positions: List[AgentPosition],
        final_rec: Recommendation
    ) -> List[str]:
        """Find agents who disagree with final recommendation."""
        dissenters = []
        
        # Group recommendations into broad categories
        buy_group = {Recommendation.STRONG_BUY, Recommendation.BUY}
        sell_group = {Recommendation.STRONG_SELL, Recommendation.SELL}
        
        if final_rec in buy_group:
            winning_group = buy_group
        elif final_rec in sell_group:
            winning_group = sell_group
        else:
            winning_group = {Recommendation.HOLD}
        
        for pos in positions:
            if pos.recommendation not in winning_group:
                dissenters.append(pos.agent_name)
        
        return dissenters
    
    def format_debate_summary(self, result: DebateResult) -> str:
        """
        Format debate result as human-readable summary.
        
        Args:
            result: DebateResult to format
            
        Returns:
            Formatted string summary
        """
        lines = [
            f"=== Debate Summary for {result.ticker} ===",
            f"Rounds: {len(result.rounds)}",
            f"Consensus Achieved: {'Yes' if result.consensus_achieved else 'No'}",
            f"Final Recommendation: {result.final_recommendation.value if result.final_recommendation else 'None'}",
            f"Confidence: {result.final_confidence:.1f}/10",
            f"Suggested Weight: {result.final_weight:.1f}%",
        ]
        
        if result.dissenting_agents:
            lines.append(f"Dissenting Agents: {', '.join(result.dissenting_agents)}")
        
        lines.append("\n--- Round Details ---")
        for round_data in result.rounds:
            lines.append(f"\nRound {round_data.round_number}:")
            for pos in round_data.positions:
                lines.append(
                    f"  {pos.agent_name}: {pos.recommendation.value} "
                    f"(conf: {pos.confidence:.1f}, weight: {pos.weight_suggestion:.1f}%)"
                )
        
        return "\n".join(lines)
