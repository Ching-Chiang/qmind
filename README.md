<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/tests-623%20passed-green" alt="623 tests passed">
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="MIT License">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="Alpha">
</p>

<h1 align="center">QMind</h1>
<p align="center"><b>LLM-Driven Multi-Agent Quantitative Trading System</b></p>
<p align="center">多智能体协作 · 结构化辩论 · CVRF 持续学习 · 多交易所执行</p>

---

## Overview

QMind is a **multi-agent quantitative trading system** powered by Large Language Models. Instead of relying on a single LLM for trading decisions, QMind orchestrates multiple AI agents with distinct roles—analysts, researchers, risk managers—that **analyze, debate, decide, and learn** collaboratively through a structured LangGraph pipeline.

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Dimensional Analysis** | 4 parallel analysts (Technical, Fundamental, Sentiment, Macro/News) each produce structured JSON reports |
| **Structured Bull-Bear Debate** | Disagreement-driven: low分歧 → direct consensus; high分歧 → risk降级 with confidence calibration |
| **Structured Trading Decisions** | JSON decision指令 (entry, stop-loss, take-profit, position size), not Markdown reports |
| **Triangular Risk Control** | Three independent reviewers (Aggressive / Conservative / Neutral) with one-veto否决 power + CVaR hard constraints |
| **CVRF Continuous Learning** | Every trade triggers LLM reflection → natural-language lessons → vector memory → auto-injection into future analysis prompts |
| **Multi-Exchange Execution** | Unified place/cancel/query interface for Binance, OKX, Bybit (REST + WebSocket) |
| **Built-in Strategies** | 17 quantitative strategies (ma_cross, MACD, RSI, Bollinger, KDJ, Ichimoku, etc.) with Freqtrade-style 3-layer abstraction |
| **Walk-Forward Backtesting** | Time-consistent train/val/test partitions, explicit transaction cost modeling, calibration (ECE ≤ 0.05) |
| **Ablation Framework** | Built-in single-agent vs multi-agent comparisons with token cost accounting |
| **Full Audit Trail** | Every decision logged: timestamp, model version, token usage, evidence chain, original LLM response |

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      QMind Pipeline                            │
│                                                                │
│  ┌──────┐  ┌──────────┐  ┌───────┐  ┌──────┐  ┌─────────┐    │
│  │Data  │→ │Analysts  │→ │Debate │→ │Risk  │→ │Execution │    │
│  │Collect│  │(4 roles) │  │(Bull/ │  │Control│  │(CEX/DEX) │    │
│  │      │  │ parallel │  │ Bear)  │  │(3-way)│  │ dry-run  │    │
│  └──────┘  └──────────┘  └───────┘  └──────┘  └─────────┘    │
│                                                │              │
│                                                ▼              │
│                                         ┌──────────┐          │
│                                         │ CVRF     │          │
│                                         │ Learning │          │
│                                         │ (闭环)    │          │
│                                         └──────────┘          │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# 1. Clone & install
git clone https://github.com/Ching-Chiang/qmind.git
cd qmind
pip install -e .

# 2. Set API keys in .env
echo "DEEPSEEK_API_KEY=sk-..." >> .env
# Optional: ANTHROPIC_API_KEY=sk-...  (for Claude)
# Optional: OPENAI_API_KEY=sk-...     (for GPT)

# 3. Run analysis
qmind analyze BTC/USDT
```

### Basic Usage

```bash
# Analyze a symbol (dry-run by default)
qmind analyze BTC/USDT

# Verbose mode (show analyst reasoning, debate, risk opinions)
qmind -v analyze BTC/USDT

# Choose data source
qmind --source binance analyze BTC/USDT
qmind --source yfinance analyze AAPL
qmind --source mock analyze BTC/USDT      # synthetic data for testing

# Backtest a strategy
qmind backtest --strategy ma_cross --start 2024-01 --end 2025-06

# Watch symbols (continuous monitoring)
qmind watch BTC/USDT ETH/USDT

# Learn from past trades
qmind learn --from-log trades.log
```

### CLI Options

```
Usage: qmind [OPTIONS] COMMAND [ARGS]...

Options:
  --config PATH          Config file path
  --source TEXT          Data source (binance, yfinance, tushare, mock)
  -v, --verbose          Show detailed reasoning (analyst, debate, risk)
  --help                 Show this message

Commands:
  analyze    Analyze a symbol and output trading decision
  backtest   Run strategy backtest
  watch      Monitor symbols continuously
  learn      Extract lessons from trade logs into memory
  exec       Execute a trading decision (requires --live)
```

## Project Structure

```
qmind/
├── agents/               # Multi-agent roles
│   ├── analysts/         #   Technical, Fundamental, Sentiment, News
│   ├── researchers/      #   Bull/Bear debate (Trust, Skeptic, Leader)
│   ├── risk.py           #   Triangular risk control
│   ├── protocol.py       #   Inter-agent JSON schemas
│   └── single_agent.py   #   Single-agent baseline
├── graph/                # LangGraph pipeline
│   ├── pipeline.py       #   5-stage StateGraph
│   ├── routers.py        #   Conditional routing (disagreement, risk veto)
│   └── state.py          #   AgentState TypedDict
├── llm/                  # LLM invocation layer
│   ├── client.py         #   Anthropic / OpenAI / DeepSeek unified client
│   ├── router.py         #   Dual-LLM routing (reasoning vs tool calls)
│   └── structured.py     #   Pydantic structured output with auto-retry
├── data/                 # Data sources
│   └── sources/          #   binance, yfinance, tushare, mock
├── execution/            # Exchange layer
│   ├── cex/              #   Binance, OKX, Bybit
│   ├── dex/              #   EVM, Solana (skeleton)
│   └── dry_run.py        #   Simulation mode
├── learning/             # CVRF learning system
│   ├── cvrf.py           #   LLM reflection → lessons
│   ├── memory.py         #   SQLite vector memory + similarity search
│   └── injector.py       #   Lesson injection into agent prompts
├── backtest/             # Backtesting framework
│   ├── partition.py      #   Walk-forward time partitioning
│   ├── cost_model.py     #   Transaction cost modeling (commission/slippage/spread)
│   ├── calibration.py    #   Confidence calibration (ECE + Platt scaling)
│   └── ablation.py       #   Single-agent vs multi-agent comparison
├── strategies/           # Strategy factory
│   └── builtin/          #   17 built-in strategies
├── tools/                # Tool layer (JSON Schema)
│   ├── market_data.py    #   K-line, orderbook, funding rate queries
│   └── portfolio.py      #   Position, balance, PnL queries
├── config.py             # YAML config + env var override
├── main.py               # CLI entry point (click)
└── audit_log.py          # Full decision audit trail
```

## Supported Models

| Provider | Models | Use Case |
|----------|--------|----------|
| **DeepSeek** | `deepseek-chat`, `deepseek-reasoner` | Default (best cost-performance) |
| **Anthropic** | `claude-sonnet-4-6`, `claude-opus-4-8` | Deep reasoning (analysts, risk) |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini` | Fast tool calls (trader) |

Configure via `config.yaml` or `QMIND_LLM_MODEL` env var.

## Design Principles

This project is grounded in findings from **18 peer-reviewed papers** (TradingAgents, FINCON, TiMi, FinDebate, and more) on LLM financial agents. Key design choices:

1. **Debate corrects bias, not generates alpha** — Disagreement triggers risk降级, not direction changes
2. **LLM confidence ≠ tradeable probability** — Confidence is independently calibrated (ECE ≤ 0.05)
3. **Position sizing is NOT LLM-driven** — Handled by the risk module with CVaR constraints
4. **Always report Net PnL** — Transaction costs (commission + slippage + spread + gas) are explicitly modeled
5. **Time-consistency enforced** — Point-in-time data control, no look-ahead

For full details see [CLAUDE.md](CLAUDE.md) (local only).

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
make test          # 623 tests
make lint          # ruff check
make coverage      # pytest-cov

# Format code
make format
```

## License

MIT
