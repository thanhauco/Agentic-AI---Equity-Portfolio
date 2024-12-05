# AlphaAgents: LLM-Based Multi-Agent Equity Portfolio Construction

A production-ready multi-agent investment framework that leverages GPT-4o and AutoGen for collaborative equity research, featuring specialized agents for fundamental analysis, sentiment analysis, and valuation with integrated debate mechanisms.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    AlphaAgents Framework                     │
├─────────────────────────────────────────────────────────────┤
│  Agent Layer:                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Fundamental │  │  Sentiment  │  │  Valuation  │          │
│  │    Agent    │  │    Agent    │  │    Agent    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                              │
│  Orchestration Layer:                                        │
│  ┌───────────────────────┐  ┌───────────────────────┐       │
│  │   Group Chat Manager  │  │   Debate Mechanism    │       │
│  └───────────────────────┘  └───────────────────────┘       │
│                                                              │
│  Tools Layer:                                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ yfinance │ │SEC Edgar │ │ NewsAPI  │ │pandas_ta │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
alpha_agents/
├── config/          # Configuration and settings
├── agents/          # Agent implementations
├── tools/           # Data fetching tools
├── orchestration/   # Multi-agent coordination
├── risk_profiles/   # Risk tolerance configurations
├── portfolio/       # Portfolio recommendation logic
└── examples/        # Usage examples
```

## Implementation Roadmap

### Phase 1: Core Framework (Completed)

- [x] Project structure and hierarchical dependency setup
- [x] Base agent architecture with AutoGen integration
- [x] Configuration management and API lifecycle handling

### Phase 2: Specialized Intelligence (Completed)

- [x] **Fundamental Agent**: SEC Filing & Metric Extraction
- [x] **Sentiment Agent**: Financial News & Analyst Rating Analysis
- [x] **Valuation Agent**: DCF-style reasoning & Technical screening

### Phase 3: Orchestration & Debate (Completed)

- [x] Multi-agent Group Chat Manager
- [x] **Consensus Engine**: Automated debate for conflicting signals
- [x] Chain-of-Thought (CoT) reasoning for explainable AI decisions

### Phase 4: Risk-Aware Portfolio Logic (Completed)

- [x] Risk profile profiles (Risk-Averse vs Risk-Neutral)
- [x] Automated portfolio rebalancing logic
- [x] Performance backtesting engine with CAGR/Sharpe metrics

### Phase 5: Deep Learning & Neural Forecasting (Completed)

- [x] **LSTM & GRU**: Recurrent models for time-series forecasting
- [x] **Transformer Encoder**: Attention-based price prediction
- [x] **Ensemble Learning**: Hybridizing neural outputs for robustness

### Phase 6: Quantitative Risk Lab & RL (Completed)

- [x] **Hierarchical Risk Parity (HRP)**: Clustering-based allocation
- [x] **Reinforcement Learning**: DQN agent for trading strategy optimization
- [x] **Computer Vision**: 1D-CNN for technical pattern recognition

### Phase 7: RAG & Filing Intelligence (Completed)

- [x] **FAISS Vector Store**: High-performance semantic retrieval
- [x] **SEC RAG Engine**: Contextual Q&A over financial documents
- [x] **Sentence-Transformers**: Domain-specific embedding layer

### Phase 8: Production Orchestration & Deployment (Next)

- [ ] **Dockerization**: Containerized microservices for agent mesh
- [ ] **FastAPI Bridge**: RESTful API for external terminal integration
- [ ] **Prometheus/Grafana**: Monitoring agent performance & latency
- [ ] **CI/CD Pipelines**: Automated model retraining on new data

### Phase 9: Alternative Data & Advanced Signals (Future)

- [ ] **Social Graph Sentiment**: Reddit/Twitter graph-based sentiment analysis
- [ ] **Satellite Imagery Mock**: Integration of supply chain physical monitoring
- [ ] **ESG Score Engine**: Automated sustainability & governance scoring
- [ ] **Macro-Economic LLM**: Specialized agent for global trade & rate cycle analysis

### Phase 10: Autonomous Agent Evolution

- [ ] **Self-Improving Agents**: Automated prompt optimization (DSPy)
- [ ] **Long-term Memory (Zep)**: Cross-session knowledge persistence
- [ ] **Cross-Asset Orchestration**: Crypto, Commodities, and Forex expansion

## Verification & Execution

```bash
# Run the Interactive Dashboard
streamlit run dashboard.py

# Execute Backtest Simulation
python examples/backtest_demo.py
```
