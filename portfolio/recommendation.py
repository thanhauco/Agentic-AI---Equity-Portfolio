"""
Portfolio Recommendation Module

Aggregates analysis from all agents and produces final portfolio recommendations.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from orchestration import DebateManager, Recommendation, AgentPosition


@dataclass
class StockRecommendation:
    """Individual stock recommendation."""
    ticker: str
    recommendation: Recommendation
    confidence: float  # 1-10
    weight: float  # 0-10%
    fundamental_summary: str
    sentiment_summary: str
    valuation_summary: str
    rationale: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass  
class PortfolioRecommendation:
    """Complete portfolio recommendation."""
    stocks: List[StockRecommendation]
    risk_profile: str
    total_invested_weight: float
    cash_weight: float
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate total invested weight."""
        self.total_invested_weight = sum(s.weight for s in self.stocks)
        self.cash_weight = 100.0 - self.total_invested_weight


class PortfolioBuilder:
    """
    Builds portfolio recommendations from individual stock analyses.
    """
    
    def __init__(
        self,
        risk_profile: str = "neutral",
        max_positions: int = 15,
        max_weight_per_stock: float = 10.0,
        min_confidence: float = 5.0
    ):
        """
        Initialize portfolio builder.
        
        Args:
            risk_profile: Either 'averse' or 'neutral'
            max_positions: Maximum number of positions in portfolio
            max_weight_per_stock: Maximum weight for any single position
            min_confidence: Minimum confidence to include in portfolio
        """
        self.risk_profile = risk_profile
        self.max_positions = max_positions
        self.max_weight_per_stock = max_weight_per_stock
        self.min_confidence = min_confidence
        self.debate_manager = DebateManager()
    
    def build_recommendation(
        self,
        ticker: str,
        fundamental_analysis: Dict[str, Any],
        sentiment_analysis: Dict[str, Any],
        valuation_analysis: Dict[str, Any]
    ) -> StockRecommendation:
        """
        Build recommendation for a single stock.
        
        Args:
            ticker: Stock ticker
            fundamental_analysis: Analysis from fundamental agent
            sentiment_analysis: Analysis from sentiment agent  
            valuation_analysis: Analysis from valuation agent
            
        Returns:
            StockRecommendation for the stock
        """
        # Create agent positions for debate
        positions = [
            AgentPosition(
                agent_name="FundamentalAnalyst",
                recommendation=self._extract_recommendation(fundamental_analysis),
                confidence=self._extract_confidence(fundamental_analysis),
                rationale=self._extract_rationale(fundamental_analysis),
                weight_suggestion=self._extract_weight(fundamental_analysis),
            ),
            AgentPosition(
                agent_name="SentimentAnalyst",
                recommendation=self._extract_recommendation(sentiment_analysis),
                confidence=self._extract_confidence(sentiment_analysis),
                rationale=self._extract_rationale(sentiment_analysis),
                weight_suggestion=self._extract_weight(sentiment_analysis),
            ),
            AgentPosition(
                agent_name="ValuationAnalyst",
                recommendation=self._extract_recommendation(valuation_analysis),
                confidence=self._extract_confidence(valuation_analysis),
                rationale=self._extract_rationale(valuation_analysis),
                weight_suggestion=self._extract_weight(valuation_analysis),
            ),
        ]
        
        # Check for conflicts and run debate if needed
        if self.debate_manager.check_conflict(positions):
            debate_result = self.debate_manager.run_debate(ticker, positions)
            final_rec = debate_result.final_recommendation
            final_conf = debate_result.final_confidence
            final_weight = debate_result.final_weight
        else:
            # No conflict - use weighted average
            final_rec = self._consensus_recommendation(positions)
            final_conf = sum(p.confidence for p in positions) / len(positions)
            final_weight = sum(p.weight_suggestion for p in positions) / len(positions)
        
        # Apply constraints
        final_weight = min(final_weight, self.max_weight_per_stock)
        
        return StockRecommendation(
            ticker=ticker,
            recommendation=final_rec,
            confidence=final_conf,
            weight=final_weight,
            fundamental_summary=str(fundamental_analysis)[:500],
            sentiment_summary=str(sentiment_analysis)[:500],
            valuation_summary=str(valuation_analysis)[:500],
            rationale=self._generate_rationale(positions, final_rec),
        )
    
    def build_portfolio(
        self,
        stock_analyses: List[Dict[str, Any]]
    ) -> PortfolioRecommendation:
        """
        Build complete portfolio from list of stock analyses.
        
        Args:
            stock_analyses: List of analysis dictionaries
            
        Returns:
            PortfolioRecommendation with all selected stocks
        """
        recommendations = []
        
        for analysis in stock_analyses:
            rec = self.build_recommendation(
                ticker=analysis["ticker"],
                fundamental_analysis=analysis.get("fundamental", {}),
                sentiment_analysis=analysis.get("sentiment", {}),
                valuation_analysis=analysis.get("valuation", {}),
            )
            
            # Filter by minimum confidence and positive recommendation
            if (rec.confidence >= self.min_confidence and 
                rec.recommendation in {Recommendation.BUY, Recommendation.STRONG_BUY}):
                recommendations.append(rec)
        
        # Sort by confidence and take top positions
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        selected = recommendations[:self.max_positions]
        
        # Normalize weights to sum to target
        total_weight = sum(r.weight for r in selected)
        if total_weight > 0:
            scale = 100 / total_weight
            for r in selected:
                r.weight *= scale
        
        return PortfolioRecommendation(
            stocks=selected,
            risk_profile=self.risk_profile,
            total_invested_weight=sum(r.weight for r in selected),
            cash_weight=100 - sum(r.weight for r in selected),
        )
    
    def _extract_recommendation(self, analysis: Dict[str, Any]) -> Recommendation:
        """Extract recommendation from analysis."""
        rec_str = analysis.get("recommendation", "hold").lower()
        mapping = {
            "strong_buy": Recommendation.STRONG_BUY,
            "buy": Recommendation.BUY,
            "hold": Recommendation.HOLD,
            "sell": Recommendation.SELL,
            "strong_sell": Recommendation.STRONG_SELL,
        }
        return mapping.get(rec_str, Recommendation.HOLD)
    
    def _extract_confidence(self, analysis: Dict[str, Any]) -> float:
        """Extract confidence score from analysis."""
        return float(analysis.get("confidence", 5.0))
    
    def _extract_rationale(self, analysis: Dict[str, Any]) -> str:
        """Extract rationale from analysis."""
        return analysis.get("rationale", "No rationale provided.")
    
    def _extract_weight(self, analysis: Dict[str, Any]) -> float:
        """Extract weight suggestion from analysis."""
        return float(analysis.get("weight", 3.0))
    
    def _consensus_recommendation(
        self,
        positions: List[AgentPosition]
    ) -> Recommendation:
        """Get consensus recommendation when no conflict."""
        # Simple majority vote
        votes = {}
        for p in positions:
            votes[p.recommendation] = votes.get(p.recommendation, 0) + 1
        return max(votes.keys(), key=lambda r: votes[r])
    
    def _generate_rationale(
        self,
        positions: List[AgentPosition],
        final_rec: Recommendation
    ) -> str:
        """Generate combined rationale."""
        lines = [f"Final Recommendation: {final_rec.value.upper()}"]
        lines.append("\nAgent Perspectives:")
        for p in positions:
            lines.append(f"- {p.agent_name}: {p.recommendation.value} (conf: {p.confidence:.1f})")
        return "\n".join(lines)
