"""
Chain-of-Thought Reasoning for AlphaAgents.

Implements structured multi-step reasoning using LLM prompting techniques
for more accurate and explainable investment decisions.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from loguru import logger

class ReasoningStep(Enum):
    GATHER_FACTS = "gather_facts"
    ANALYZE_FUNDAMENTALS = "analyze_fundamentals"
    ASSESS_SENTIMENT = "assess_sentiment"
    EVALUATE_TECHNICALS = "evaluate_technicals"
    SYNTHESIZE = "synthesize"
    SELF_VERIFY = "self_verify"
    CONCLUDE = "conclude"

@dataclass
class ThoughtChain:
    """Represents a chain of reasoning steps."""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    final_recommendation: Optional[str] = None
    confidence: float = 0.0
    
    def add_step(self, step_type: ReasoningStep, content: str, evidence: List[str] = None):
        """Add a reasoning step to the chain."""
        self.steps.append({
            "step": step_type.value,
            "thought": content,
            "evidence": evidence or []
        })
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert chain to dictionary."""
        return {
            "reasoning_chain": self.steps,
            "recommendation": self.final_recommendation,
            "confidence": self.confidence,
            "total_steps": len(self.steps)
        }

class ChainOfThoughtReasoner:
    """
    Implements Chain-of-Thought (CoT) prompting for structured reasoning.
    
    Guides LLM through multi-step analysis before reaching a conclusion.
    """
    
    COT_SYSTEM_PROMPT = """You are a senior investment analyst using structured reasoning.
    
For each analysis, you MUST follow this chain of thought:

1. GATHER FACTS: List the key data points available
2. ANALYZE FUNDAMENTALS: Assess financial health and business quality
3. ASSESS SENTIMENT: Consider market perception and news flow
4. EVALUATE TECHNICALS: Review price action and momentum indicators
5. SYNTHESIZE: Combine insights, noting conflicts and agreements
6. SELF-VERIFY: Challenge your reasoning - what could be wrong?
7. CONCLUDE: Make a final recommendation with confidence level

Output your reasoning in JSON format:
{
    "chain": [
        {"step": "gather_facts", "thought": "...", "evidence": [...]},
        ...
    ],
    "recommendation": "BUY|SELL|HOLD",
    "confidence": 0.0-1.0,
    "key_risks": ["..."]
}
"""

    FEW_SHOT_EXAMPLES = [
        {
            "input": "Analyze NVDA for investment",
            "output": {
                "chain": [
                    {"step": "gather_facts", "thought": "NVDA is a semiconductor company leading in GPU technology. Current P/E is 65, revenue growth 122% YoY.", "evidence": ["P/E: 65", "Revenue Growth: 122%"]},
                    {"step": "analyze_fundamentals", "thought": "Dominant market position in AI chips. Strong margins (>60%). High growth but premium valuation.", "evidence": ["Gross Margin: 64%", "Market Share: 80% data center GPUs"]},
                    {"step": "assess_sentiment", "thought": "Overwhelmingly positive analyst coverage. 45 Buy, 5 Hold, 0 Sell ratings.", "evidence": ["Analyst consensus: Strong Buy"]},
                    {"step": "evaluate_technicals", "thought": "RSI at 72 (overbought). Price above SMA50 and SMA200. Strong upward trend.", "evidence": ["RSI: 72", "Trend: Bullish"]},
                    {"step": "synthesize", "thought": "Exceptional fundamentals and sentiment. Technical overbought condition suggests near-term pullback risk.", "evidence": []},
                    {"step": "self_verify", "thought": "Risks: valuation multiple compression, competition from AMD/custom chips, geopolitical (Taiwan).", "evidence": []},
                    {"step": "conclude", "thought": "BUY on dips. Strong long-term thesis despite short-term overextension.", "evidence": []}
                ],
                "recommendation": "BUY",
                "confidence": 0.82,
                "key_risks": ["Valuation", "Competition", "Geopolitical"]
            }
        }
    ]
    
    def __init__(self):
        self.thought_chains: Dict[str, ThoughtChain] = {}
        
    def format_prompt(self, ticker: str, data: Dict[str, Any]) -> str:
        """Format a CoT prompt for the given stock and data."""
        prompt = f"""Analyze {ticker} using structured chain-of-thought reasoning.

Available Data:
- Fundamentals: {json.dumps(data.get('fundamentals', {}), indent=2)}
- Sentiment: {json.dumps(data.get('sentiment', {}), indent=2)}
- Technicals: {json.dumps(data.get('technicals', {}), indent=2)}

Follow the 7-step reasoning chain and output your analysis in JSON format.
"""
        return prompt
    
    def parse_response(self, response: str) -> ThoughtChain:
        """Parse LLM response into a ThoughtChain object."""
        chain = ThoughtChain()
        
        try:
            # Try to extract JSON from response
            json_match = response
            if "```json" in response:
                json_match = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_match = response.split("```")[1].split("```")[0]
                
            data = json.loads(json_match)
            
            for step in data.get("chain", []):
                step_type = ReasoningStep(step.get("step", "gather_facts"))
                chain.add_step(step_type, step.get("thought", ""), step.get("evidence", []))
                
            chain.final_recommendation = data.get("recommendation")
            chain.confidence = data.get("confidence", 0.5)
            
        except Exception as e:
            logger.warning(f"Failed to parse CoT response: {e}")
            # Create minimal chain from raw response
            chain.add_step(ReasoningStep.CONCLUDE, response[:500])
            chain.confidence = 0.3
            
        return chain
    
    def get_reasoning_summary(self, chain: ThoughtChain) -> str:
        """Generate a human-readable summary of the reasoning chain."""
        summary_lines = [f"**Reasoning Chain ({len(chain.steps)} steps):**\n"]
        
        for i, step in enumerate(chain.steps, 1):
            summary_lines.append(f"{i}. **{step['step'].replace('_', ' ').title()}**: {step['thought'][:150]}...")
            
        summary_lines.append(f"\n**Final Recommendation**: {chain.final_recommendation}")
        summary_lines.append(f"**Confidence**: {chain.confidence:.0%}")
        
        return "\n".join(summary_lines)

class SelfReflectionLoop:
    """
    Implements self-reflection for improved accuracy.
    
    The agent reviews its own reasoning and corrects potential errors.
    """
    
    REFLECTION_PROMPT = """Review your previous analysis and answer:

1. Is the reasoning logically sound?
2. Are there any biases in the analysis?
3. What evidence is missing?
4. Would a contrarian view be valid?

Based on this reflection, should you adjust your recommendation?
Output: {"adjustment_needed": true/false, "revised_confidence": 0.0-1.0, "notes": "..."}
"""
    
    def __init__(self, max_iterations: int = 2):
        self.max_iterations = max_iterations
        
    def should_reflect(self, confidence: float) -> bool:
        """Determine if self-reflection is needed based on confidence."""
        return confidence < 0.7
        
    def apply_reflection(self, original: ThoughtChain, reflection: Dict[str, Any]) -> ThoughtChain:
        """Apply reflection adjustments to the thought chain."""
        if reflection.get("adjustment_needed", False):
            original.confidence = reflection.get("revised_confidence", original.confidence)
            original.add_step(
                ReasoningStep.SELF_VERIFY,
                f"Self-reflection: {reflection.get('notes', 'No notes')}",
                []
            )
        return original
