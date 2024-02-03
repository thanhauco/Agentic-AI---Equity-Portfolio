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

## Implementation Phases

### Phase 1: Core Framework

- [x] Project structure and dependencies
- [x] Base agent architecture
- [x] Configuration management

### Phase 2: Specialized Agents

- [ ] Fundamental Agent
- [ ] Sentiment Agent
- [ ] Valuation Agent

### Phase 3: Orchestration

- [ ] Group Chat Manager
- [ ] Debate Mechanism

### Phase 4: Risk Integration

- [ ] Risk profiles implementation
- [ ] Portfolio recommendations

## Verification Plan

```bash
# Run single stock analysis
python examples/single_stock_analysis.py --ticker AAPL --risk-profile neutral

# Run portfolio construction
python examples/portfolio_construction.py --tickers AAPL,GOOGL,MSFT,NVDA,META
```
