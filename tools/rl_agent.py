"""
Reinforcement Learning Engine for AlphaAgents.

Implements a Deep Q-Network (DQN) for autonomous portfolio rebalancing 
and trading strategy optimization.
"""

import numpy as np
import pandas as pd
import random
from collections import deque
from typing import Dict, Any, List, Tuple
from loguru import logger

def _get_rl_components():
    """Lazy load components."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        return tf, Sequential, Dense, Dropout, Adam
    except ImportError:
        return None, None, None, None, None

class DQNAgent:
    """
    Deep Q-Network Agent for portfolio management.
    """
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size  # 0: Hold/Buy, 1: Sell, 2: Neutral
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        tf, Sequential, Dense, Dropout, Adam = _get_rl_components()
        if tf is None:
            return None
            
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class PortfolioEnv:
    """
    Simplified Trading Environment for RL training.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.n_steps = len(df)
        self.current_step = 0
        self.inventory = []
        self.total_profit = 0
        self.state_size = 5 # window_size feature
        
    def reset(self):
        self.current_step = 0
        self.inventory = []
        self.total_profit = 0
        return self._get_state()
        
    def _get_state(self):
        # Window of previous 5 days price changes
        if self.current_step < 5:
            return np.zeros((1, 5))
        state = self.df['Close'].iloc[self.current_step-5:self.current_step].pct_change().fillna(0).values
        return state.reshape(1, 5)
        
    def step(self, action):
        done = False
        reward = 0
        
        current_price = self.df['Close'].iloc[self.current_step]
        
        if action == 0: # Buy
            self.inventory.append(current_price)
        elif action == 1 and len(self.inventory) > 0: # Sell
            bought_price = self.inventory.pop(0)
            reward = current_price - bought_price
            self.total_profit += reward
            
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            done = True
            
        return self._get_state(), reward, done, {}

def run_rl_simulation(df: pd.DataFrame, episodes: int = 10) -> Dict[str, Any]:
    """
    Run the RL agent through the historical data to optimize trading.
    """
    env = PortfolioEnv(df)
    agent = DQNAgent(env.state_size, 3) # 3 actions: Buy, Sell, Hold
    
    if agent.model is None:
        return {"error": "TensorFlow not available"}
        
    logger.info("Starting RL training simulation...")
    
    batch_size = 32
    history = []
    
    for e in range(episodes):
        state = env.reset()
        for time in range(env.n_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                history.append(env.total_profit)
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
    return {
        "final_profit": env.total_profit,
        "total_episodes": episodes,
        "history": history,
        "agent_type": "DQN (Deep Q-Network)",
        "strategy": "Reinforcement Learning for Portfolio Rebalancing"
    }
