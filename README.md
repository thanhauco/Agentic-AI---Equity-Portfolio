# AlphaAgents 🤖📈

**LLM-Based Multi-Agent Framework for Equity Portfolio Construction**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![AutoGen](https://img.shields.io/badge/AutoGen-0.2+-green.svg)](https://github.com/microsoft/autogen)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

AlphaAgents is a production-ready multi-agent investment framework that leverages GPT-4o and Microsoft's AutoGen for collaborative equity research. The system employs specialized AI agents that work together through structured collaboration and debate mechanisms to generate comprehensive stock analyses and portfolio recommendations.

## Key Features

- **🧠 Multi-Agent Collaboration**: Three specialized agents (Fundamental, Sentiment, Valuation) work together
- **⚖️ Debate Mechanism**: Round-robin consensus building for conflicting recommendations
- **📊 Risk Profiles**: Support for risk-averse and risk-neutral investment strategies
- **🔧 Modular Architecture**: Easy to extend with additional agents and tools
- **📈 Real Data Integration**: yfinance, SEC Edgar, NewsAPI integrations

## Architecture

### System Overview

AlphaAgents utilizes a layered architecture to separate concerns between user interaction, agent orchestration, specialized reasoning, and data acquisition.

```mermaid
graph TD
    User([User/Terminal]) --> Dashboard[Interactive Dashboard]
    Dashboard --> Orchestrator[AlphaGroupChat Manager]
    Orchestrator --> Debate[Debate Manager]

    subgraph Agents [Specialized AI Agents]
        Fundamental[Fundamental Agent]
        Sentiment[Sentiment Agent]
        Valuation[Valuation Agent]
    end

    Orchestrator <--> Agents
    Debate <--> Agents

    subgraph Tools [Data & Analysis Tools]
        YF[yfinance API]
        NAPI[NewsAPI]
        TA[Technical Analysis Lib]
        Cache[(Disk Cache)]
    end

    Agents --> Tools
    Tools --> Cache
```

### Agent Collaboration Workflow

The system follows a structured collaborative process to reach an investment decision.

```mermaid
sequenceDiagram
    participant U as User
    participant O as Orchestrator
    participant A as Agents (F, S, V)
    participant D as Debate Manager
    participant P as Portfolio Builder

    U->>O: Request Analysis (Ticker)
    O->>A: Parallel Analysis Request
    A-->>O: Individual Reports & Recommendations
    O->>D: Check for Conflicts
    alt Conflict Detected
        D->>A: Initiate Debate Rounds
        A-->>D: Revised Positions
        D->>D: Reach Consensus / Weighted Vote
    else No Conflict
        D->>D: Standard Aggregation
    end
    D->>P: Final Integrated Sentiment
    P-->>U: Portfolio Recommendation & Rationale
```

### Debate Mechanism Logic

When agents disagree (e.g., Fundamental says "Buy" but Valuation says "Sell"), the Debate Manager intervenes.

```mermaid
flowchart LR
    Start[Agent Outputs] --> Conflict{Recommendation Conflict?}
    Conflict -- No --> Aggregate[Weighted Averaging]
    Conflict -- Yes --> Round1[Debate Round 1: Exchanging Rationales]
    Round1 --> Review{Consensus Reached?}
    Review -- Yes --> Final[Final Decision]
    Review -- No --> Round2[Debate Round 2: Rebuttal & Consensus Search]
    Round2 --> Max{Max Rounds Hit?}
    Max -- Yes --> Vote[Weighted Vote by Confidence]
    Max -- No --> Review
    Vote --> Final
    Aggregate --> Final
```

## Agents

### 1. Fundamental Agent

Analyzes company fundamentals including:

- 10-K/10-Q SEC filings
- Revenue and earnings trends
- Balance sheet health
- Sector positioning

### 2. Sentiment Agent

Processes market sentiment through:

- Financial news analysis
- Analyst ratings aggregation
- Social sentiment scoring
- Event impact assessment

### 3. Valuation Agent

Evaluates stock valuations via:

- P/E, P/B, EV/EBITDA ratios
- Technical indicators (RSI, MACD)
- Volume analysis
- Price momentum

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/alpha-agents.git
cd alpha-agents

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run single stock analysis
python examples/single_stock_analysis.py --ticker AAPL --risk-profile neutral

# Run portfolio construction
python examples/portfolio_construction.py --tickers AAPL,GOOGL,MSFT,NVDA,META
```

## Configuration

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_key
NEWS_API_KEY=your_newsapi_key  # Optional
```

## Risk Profiles

### Risk-Averse

- Prioritizes stable, dividend-paying stocks
- Avoids high-beta and volatile positions
- Focuses on margin of safety

### Risk-Neutral

- Balances growth and value opportunities
- Objective sentiment analysis
- Considers momentum alongside fundamentals

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read our contributing guidelines first.
