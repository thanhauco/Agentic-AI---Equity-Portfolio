"""
Persistent Memory Engine for AlphaAgents.

Implements long-term memory for agents to remember past 
analyses and learn from previous mistakes.
"""

import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from loguru import logger

class PersistentMemory:
    """
    Agent memory layer using SQLite for cross-session persistence.
    """
    
    def __init__(self, db_path: str = ".data/memory.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT,
                    ticker TEXT,
                    recommendation TEXT,
                    rationale TEXT,
                    outcome_actual TEXT,
                    confidence FLOAT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_learned_rules (
                    agent_name TEXT PRIMARY KEY,
                    rules_json TEXT,
                    performance_history TEXT
                )
            """)

    def save_experience(self, agent_name: str, ticker: str, rec: str, rationale: str, confidence: float):
        """Record an agent's decision for future reflection."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO experiences (agent_name, ticker, recommendation, rationale, confidence) VALUES (?, ?, ?, ?, ?)",
                (agent_name, ticker, rec, rationale, confidence)
            )

    def get_past_decisions(self, ticker: str) -> List[Dict[str, Any]]:
        """Retrieve historical decisions for a specific stock."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM experiences WHERE ticker = ? ORDER BY timestamp DESC", (ticker,))
            return [dict(zip([column[0] for column in cursor.description], row)) for row in cursor.fetchall()]

    def learn_from_outcomes(self, agent_name: str):
        """
        Mock logic for prompt optimization:
        In a real app, this would use DSPy to optimize the system prompt 
        based on which rationales led to correct outcome predictions.
        """
        # Logic to analyze 'outcome_actual' vs 'recommendation'
        pass

def get_agent_context_with_memory(agent_name: str, ticker: str) -> str:
    """
    Augments the agent's prompt with memory of past analyses.
    """
    memory = PersistentMemory()
    past = memory.get_past_decisions(ticker)
    
    if not past:
        return ""
        
    context = "\n\nPAST ANALYSES MEMORY:\n"
    for p in past[:3]: # Last 3 analyses
        context += f"- {p['timestamp']}: Recommendation was {p['recommendation']} with confidence {p['confidence']}. Rationale: {p['rationale'][:100]}...\n"
    
    return context
