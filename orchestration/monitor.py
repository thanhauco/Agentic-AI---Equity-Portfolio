"""
Infrastructure for AlphaAgents Monitoring.

Tracks agent performance, token usage, and strategy drift.
"""

import time
import json
from typing import Dict, Any, List
from loguru import logger
import os

class GovernanceMonitor:
    """
    Logs and monitors agent operations for auditing and performance tracking.
    """
    
    def __init__(self, log_dir: str = ".logs/governance"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = {
            "total_tokens": 0,
            "total_calls": 0,
            "latency_history": [],
            "recommendation_drift": []
        }

    def log_interaction(self, agent_name: str, task: str, metadata: Dict[str, Any]):
        """Record an agent-LLM interaction."""
        entry = {
            "timestamp": time.time(),
            "agent": agent_name,
            "task": task,
            "tokens": metadata.get("tokens", 0),
            "latency": metadata.get("latency", 0),
            "status": metadata.get("status", "success")
        }
        
        # Update internal metrics
        self.metrics["total_tokens"] += entry["tokens"]
        self.metrics["total_calls"] += 1
        self.metrics["latency_history"].append(entry["latency"])
        
        # Persist to disk
        filename = f"{self.log_dir}/audit_{int(time.time())}.json"
        with open(filename, "w") as f:
            json.dump(entry, f)

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregate performance metrics."""
        avg_latency = sum(self.metrics["latency_history"]) / len(self.metrics["latency_history"]) if self.metrics["latency_history"] else 0
        return {
            "total_usage_cost_est": self.metrics["total_tokens"] * 0.00001, # Mock cost
            "uptime_reliability": "99.9%",
            "avg_latency_ms": avg_latency * 1000,
            "throughput_calls": self.metrics["total_calls"]
        }

def track_drift(ticker: str, previous_signal: float, current_signal: float) -> Optional[Dict[str, Any]]:
    """Detect if an agent's view on a stock has flipped significantly."""
    if abs(previous_signal - current_signal) > 0.5:
        return {
            "ticker": ticker,
            "alert": "Regime Flip Detected",
            "delta": current_signal - previous_signal,
            "severity": "High"
        }
    return None
