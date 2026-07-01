<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/tests-623%20passed-green" alt="623 测试通过">
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="MIT License">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="Alpha">
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh.md"><b>中文</b></a>
</p>

<h1 align="center">QMind</h1>
<p align="center"><b>LLM 驱动的多智能体量化交易系统</b></p>
<p align="center">多角色协作 · 结构化辩论 · CVRF 持续学习 · 多交易所执行</p>

---

## 概述

QMind 是一个 **LLM 驱动的多智能体量化交易系统**。核心思路：用多个 AI 角色协作（分析→辩论→决策→风控），替代单一 LLM 的独断式交易判断；每笔交易后通过 CVRF（概念言语强化学习）自动总结教训，让系统越交易越聪明。

### 功能特性

| 特性 | 说明 |
|------|------|
| **多维市场分析** | 4 个分析师并行（技术面/基本面/情绪面/宏观新闻），各自输出结构化 JSON 报告 |
| **结构化多空辩论** | 分歧驱动：低分歧→直接采信，高分歧→风控降级+置信度校准，不做方向判断 |
| **结构化交易决策** | 输出 JSON 决策指令（入场/止损/目标/仓位），非 Markdown 研报 |
| **三角风控审核** | 激进/保守/中立三个角度独立审核，一票否决制 + CVaR 硬约束 |
| **CVRF 持续学习** | 每笔交易结束 → LLM 总结教训 → 向量记忆存储 → 相似市况自动注入历史教训 |
| **多交易所执行** | 统一下单/撤单/查单接口，支持 Binance、OKX、Bybit（REST + WebSocket） |
| **内置策略** | 17 个量化策略（双均线、MACD、RSI、布林带、KDJ、一目均衡等），Freqtrade 三层抽象 |
| **Walk-Forward 回测** | 时间一致 train/val/test 划分，显式交易成本建模，置信度校准（ECE ≤ 0.05） |
| **消融实验框架** | 内置单 Agent vs 多 Agent 对比框架，含 Token 成本核算 |
| **全链路审计日志** | 每个决策记录：时间戳、模型版本、Token 用量、证据链、原始 LLM 响应 |

### 架构

```
┌──────────────────────────────────────────────────────────────┐
│                      QMind 流水线                              │
│                                                                │
│  ┌──────┐  ┌──────────┐  ┌───────┐  ┌──────┐  ┌─────────┐    │
│  │数据  │→ │分析师    │→ │辩论   │→ │风控  │→ │执行     │    │
│  │采集  │  │(4角色并行)│  │(多/空) │  │(三角) │  │(CEX/DEX)│    │
│  └──────┘  └──────────┘  └───────┘  └──────┘  └─────────┘    │
│                                                │              │
│                                                ▼              │
│                                         ┌──────────┐          │
│                                         │ CVRF     │          │
│                                         │ 学习循环  │          │
│                                         └──────────┘          │
└──────────────────────────────────────────────────────────────┘
```

## 快速开始

### 安装

```bash
# 1. 克隆并安装
git clone https://github.com/Ching-Chiang/qmind.git
cd qmind
pip install -e .

# 2. 在 .env 中设置 API Key
echo "DEEPSEEK_API_KEY=sk-..." >> .env
# 可选: ANTHROPIC_API_KEY=sk-...  (使用 Claude)
# 可选: OPENAI_API_KEY=sk-...     (使用 GPT)

# 3. 运行分析
qmind analyze BTC/USDT
```

### 基本用法

```bash
# 分析某个标的（默认 dry-run 模式，不实际下单）
qmind analyze BTC/USDT

# 详细模式（显示分析师推理、辩论过程、风控意见）
qmind -v analyze BTC/USDT

# 选择数据源
qmind --source binance analyze BTC/USDT
qmind --source yfinance analyze AAPL
qmind --source mock analyze BTC/USDT      # 模拟数据测试

# 回测策略
qmind backtest --strategy ma_cross --start 2024-01 --end 2025-06

# 持续监控多个标的
qmind watch BTC/USDT ETH/USDT

# 从交易日志中学习
qmind learn --from-log trades.log
```

### CLI 选项

```
用法: qmind [选项] 命令 [参数]...

选项:
  --config PATH          配置文件路径
  --source TEXT          数据源 (binance, yfinance, tushare, mock)
  -v, --verbose          显示详细推理（分析师、辩论、风控意见）
  --help                 显示帮助信息

命令:
  analyze    分析标的，输出交易决策
  backtest   运行策略回测
  watch      持续监控标的
  learn      从交易日志提取教训到经验库
  exec       执行交易决策（需要 --live）
```

## 项目结构

```
qmind/
├── agents/               # 多角色 Agent
│   ├── analysts/         #   技术面/基本面/情绪面/新闻 分析师
│   ├── researchers/      #   多空辩论（Trust / Skeptic / Leader）
│   ├── risk.py           #   三角风控（激进/保守/中立）
│   ├── protocol.py       #   Agent 间通信 JSON Schema
│   └── single_agent.py   #   单 Agent 基线对比
├── graph/                # LangGraph 流水线
│   ├── pipeline.py       #   五阶段 StateGraph
│   ├── routers.py        #   条件路由（分歧判断、风控否决）
│   └── state.py          #   AgentState TypedDict
├── llm/                  # LLM 调用层
│   ├── client.py         #   Anthropic / OpenAI / DeepSeek 统一客户端
│   ├── router.py         #   双 LLM 路由（推理用强模型，工具调用用快模型）
│   └── structured.py     #   Pydantic 结构化输出 + 自动重试
├── data/                 # 数据源
│   └── sources/          #   binance, yfinance, tushare, mock
├── execution/            # 执行层
│   ├── cex/              #   Binance, OKX, Bybit
│   ├── dex/              #   EVM, Solana（框架）
│   └── dry_run.py        #   模拟交易模式
├── learning/             # CVRF 学习系统
│   ├── cvrf.py           #   LLM 反思 → 教训
│   ├── memory.py         #   SQLite 向量记忆 + 相似度检索
│   └── injector.py       #   教训注入到 Agent prompt
├── backtest/             # 回测框架
│   ├── partition.py      #   Walk-forward 时间划分
│   ├── cost_model.py     #   交易成本建模（佣金/滑点/价差）
│   ├── calibration.py    #   置信度校准（ECE + Platt scaling）
│   └── ablation.py       #   单 Agent vs 多 Agent 对比
├── strategies/           # 策略工厂
│   └── builtin/          #   17 个内置策略
├── tools/                # 工具层（JSON Schema）
│   ├── market_data.py    #   K线、深度、资金费率查询
│   └── portfolio.py      #   持仓、余额、PnL 查询
├── config.py             # YAML 配置 + 环境变量覆写
├── main.py               # CLI 入口（click）
└── audit_log.py          # 全链路审计日志
```

## 支持的 LLM 模型

| 供应商 | 模型 | 用途 |
|--------|------|------|
| **DeepSeek** | `deepseek-chat`, `deepseek-reasoner` | 默认（性价比最高） |
| **Anthropic** | `claude-sonnet-4-6`, `claude-opus-4-8` | 深度推理（分析师、风控） |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini` | 快速工具调用（交易员） |

通过 `config.yaml` 或 `QMIND_LLM_MODEL` 环境变量配置。

## 设计原则

本项目基于 **18 篇同行评审论文**（TradingAgents、FINCON、TiMi、FinDebate 等）关于 LLM 金融 Agent 的研究结论。关键设计决策：

1. **辩论校正偏差，而非生成 alpha** — 分歧触发风控降级而非改变方向
2. **LLM 置信度 ≠ 可交易概率** — 置信度独立校准（ECE ≤ 0.05）后才用于仓位计算
3. **仓位计算不由 LLM 决定** — 由风控模块使用 CVaR 约束独立处理
4. **始终报告净 PnL** — 显式建模交易成本（佣金+滑点+价差+Gas 费）
5. **时间一致性强制** — Point-in-Time 数据控制，禁止前视偏差

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
make test          # 623 个测试
make lint          # ruff 检查
make coverage      # 测试覆盖率

# 格式化代码
make format
```

## 技术选型

| 层面 | 选型 | 原因 |
|------|------|------|
| 编排 | LangGraph StateGraph | 状态持久化、条件路由、可恢复执行 |
| LLM 调用 | Anthropic/OpenAI SDK 原生 | 少一层抽象，追 bug 更快 |
| 结构化输出 | Pydantic + `model_validate_json()` | 强类型校验，不符合自动重试 |
| 状态持久化 | SQLite (`langgraph-checkpoint-sqlite`) | 轻量、零运维 |
| 记忆系统 | SQLite 自建向量检索 | CVRF 高度定制，通用模块改起来更累 |
| 配置 | YAML + 环境变量覆写 + python-dotenv | 敏感信息不入库 |

## 协议

MIT
