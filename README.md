# QMind

**LLM-Driven Multi-Agent Quantitative Trading System**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![Tests](https://img.shields.io/badge/tests-621%20passed-green)]()
[![License](https://img.shields.io/badge/license-MIT-yellow)]()
[![Status](https://img.shields.io/badge/status-beta-orange)]()

<p align="center">
  <a href="README.md"><b>English</b></a> ·
  <a href="README.zh.md">中文</a>
</p>

---

QMind orchestrates multiple AI agents—analysts, researchers, risk managers—that **analyze, debate, decide, and learn** collaboratively through a structured LangGraph pipeline, then **continuously improve** via CVRF (Conceptual Verbal Reinforcement Learning).

---

## Features

| Feature | Description |
|---------|-------------|
| **4 Parallel Analysts** | Technical, Fundamental, Sentiment, Macro/News — each with heterogeneous LLM assignments to prevent echo chambers |
| **Disagreement-Driven Debate** | Low disagreement (δ < 0.15) → skip debate; high disagreement → confidence downgrade + position reduction, **never changes direction** |
| **Structured JSON Decisions** | Outputs executable trade instructions (entry, stop-loss, take-profit, position size), not Markdown reports |
| **Triangular Risk Control** | Aggressive / Conservative / Neutral reviewers, one-vote veto + CVaR hard constraints |
| **CVRF Learning Loop** | Trade → evaluate → LLM reflection → vector memory → auto-injection into future analysis prompts |
| **20 Built-in Strategies** | Freqtrade-style 3-layer abstraction (`indicators → entry → exit`) |
| **Walk-Forward Backtesting** | Chronological train/val/test partitions, explicit cost modeling, confidence calibration (ECE ≤ 0.05) |
| **P1-P6 Compliance** | Full Alpha Illusion protocol reports: time consistency, point-in-time data, cost tiers, ablation, bias detection |
| **Multi-Exchange** | Binance, OKX, Bybit via unified `ExchangeBase` interface |
| **Full Audit Trail** | Every decision logged: timestamp, model, token usage, evidence chain, LLM response |

## Quick Start

```bash
# Install
cd qmind
pip install -e ".[dev,sources,cex]"    # full install with all extras

# Set API key
export DEEPSEEK_API_KEY="sk-..."

# Analyze BTC/USDT (mock data, dry-run, no real trading)
qmind --source mock analyze BTC/USDT

# With real Binance data (requires proxy if behind firewall)
qmind --source binance analyze BTC/USDT
```

## CLI Usage

### Options

```
qmind [OPTIONS] COMMAND [ARGS]...

Options:
  -c, --config PATH   Config file path
  --dry-run / --live  Simulation mode (default: dry-run)
  --source TEXT       Data source: auto / binance / yfinance / tushare / mock
  -v, --verbose       Show detailed reasoning of each agent
  --help              Show this message
```

### Commands

#### `analyze` — One-shot analysis

```bash
# Default (concise summary)
qmind analyze BTC/USDT

# Verbose: see analyst reasoning, debate, risk opinions
qmind -v analyze BTC/USDT

# Choose data source
qmind --source binance analyze ETH/USDT
qmind --source yfinance analyze AAPL
qmind --source tushare analyze 000001.SZ
qmind --source mock analyze BTC/USDT    # synthetic data, no network needed

# Live trading (requires exchange API key)
qmind --live analyze BTC/USDT
```

Verbose mode output example:

```
阶段 2/5: 多维分析
  + [technical] bullish (75%)
    逻辑: 价格突破布林带上轨和20周期高位...
    信号: 均线系统 SMA20 > SMA50 > SMA200 (bullish)
    风险: RSI接近70超买区

阶段 3/5: 多空辩论
  分歧度 δ=0.247  收敛: False  降级因子: 0.55
  + 共识: 加密货币缺乏明确内在价值支撑
  - 分歧: 技术面看多 vs 基本面/消息面看空

阶段 4/5: 交易决策
  决策: LONG  置信度: 0.42
  Data-CoT: 价格突破20周期高位...
  Thesis-CoT: 建议61400入场做多...

阶段 5/5: 三角风控审核
  PASS [aggressive]   理由: 入场有支撑逻辑...
  VETO [conservative] 理由: 置信度仅0.42...
  MODIFY [neutral]    理由: 盈亏比1.6不足2.0
```

#### `watch` — Continuous monitoring

```bash
# Monitor symbols, push signal on trade decision
qmind watch BTC/USDT
qmind watch BTC/USDT ETH/USDT
qmind watch BTC/USDT --timeframe 4h --interval 300
```

Triggers notification (Feishu/email) when decision ≠ HOLD and risk approved.

#### `backtest` — Strategy backtest

```bash
# List all registered strategies
qmind list --strategies

# Backtest a strategy
qmind backtest --strategy ma_cross --start 2024-01 --end 2025-06
```

#### `learn` — CVRF learning

```bash
# View current memory store
qmind learn

# Learn from trade log
qmind learn --from-log trades.log
```

#### Other commands

```bash
qmind price BTC/USDT         # Quick price check
qmind list --strategies      # List all 20 built-in strategies
qmind list --audit           # View audit summary
qmind version                # Version info
```

## Built-in Strategies (20)

**Moving Average:** `ma_cross`, `ma_cross_triple`, `triple_ema`, `macd`, `macd_rsi`
**Reversal:** `rsi`, `cci`, `stoch`, `williams_r`, `kdj`
**Trend:** `adx_macd`, `psar`, `ichimoku`
**Volatility:** `bollinger`, `atr_stop`, `chandelier`
**Volume:** `obv`, `mfi`, `volume_breakout`
**Channel:** `donchian`

## Data Sources

| Source | Command | Requires | Notes |
|--------|---------|----------|-------|
| Binance | `--source binance` | Network → proxy for CN firewall | Real crypto data |
| Yahoo Finance | `--source yfinance` | Network | US stocks, rate-limited |
| Tushare | `--source tushare` | API token (free) | A-share / HK stocks |
| Mock | `--source mock` | None | Synthetic data for testing |

## Configuration

Create `config.yaml`:

```yaml
llm:
  default_model: deepseek-chat
  deepseek_api_key: "${DEEPSEEK_API_KEY}"   # or set env var

execution:
  dry_run: true
  default_exchange: binance

storage:
  db_path: qmind.db

notification:
  type: none          # none / feishu / email
```

Supports Anthropic (`ANTHROPIC_API_KEY`), OpenAI (`OPENAI_API_KEY`), and DeepSeek (`DEEPSEEK_API_KEY`).

## Network / Proxy

If behind a firewall (common in China), set these environment variables:

```bash
export ALL_PROXY="http://127.0.0.1:7890"    # or your proxy address
export NO_PROXY="localhost,127.0.0.1,api.deepseek.com"
```

The system auto-detects Windows system proxy from registry.

## Architecture

```
collect_data → analyze (4 agents parallel) → debate (disagreement-driven)
→ decide (financial CoT) → review_risk (triangular) → execute / reject
                                                          ↓
                                                     CVRF learn
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
│   ├── pipeline.py       #   6-stage StateGraph (incl. CVRF learn node)
│   ├── routers.py        #   Conditional routing
│   └── state.py          #   AgentState TypedDict
├── llm/                  # LLM invocation
│   ├── client.py         #   Anthropic / OpenAI / DeepSeek unified client
│   ├── router.py         #   Dual-LLM routing
│   └── structured.py     #   Pydantic structured output + auto-retry
├── data/sources/         # binance, yfinance, tushare, mock
├── execution/            # Exchange layer (Binance/OKX/Bybit + dry_run)
├── learning/             # CVRF: reflection, memory (SQLite), injector
├── backtest/             # partition, cost_model, calibration, ablation, p_report
├── strategies/builtin/   # 20 strategies
├── audit/                # Bias checker (5-class bias detection)
├── main.py               # CLI entry point (click)
└── config.py             # YAML config
```

## Tests

```bash
make test       # 621 tests, 19.75s
make lint       # ruff check
make coverage   # pytest-cov (72% coverage)
```

## Design Principles

Based on findings from 18 peer-reviewed papers on LLM financial agents (TradingAgents, FINCON, TiMi, FinDebate, Alpha Illusion, et al.):

1. **Debate corrects bias, not generates alpha** — triggers risk downgrade, never changes direction
2. **LLM confidence ≠ tradeable probability** — independently calibrated (ECE ≤ 0.05)
3. **Position sizing is NOT LLM-driven** — handled by risk module with CVaR constraints
4. **Always report Net PnL** — commission + slippage + spread + gas explicitly modeled
5. **Time-consistency enforced** — Point-in-Time data control via `TimeGuard`, no look-ahead

## License

MIT
