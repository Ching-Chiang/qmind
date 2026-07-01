# QMind

**LLM 驱动的多智能体量化交易系统**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![Tests](https://img.shields.io/badge/tests-621%20passed-green)]()
[![License](https://img.shields.io/badge/license-MIT-yellow)]()
[![Status](https://img.shields.io/badge/status-beta-orange)]()

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh.md"><b>中文</b></a>
</p>

---

QMind 用多个 AI 智能体协作（分析师→辩论→决策→风控），替代单一 LLM 的独断式交易判断；每笔交易后通过 CVRF（概念言语强化学习）自动总结教训，让系统越交易越聪明。

---

## 功能特性

| 特性 | 说明 |
|------|------|
| **4 个分析师并行** | 技术面/基本面/情绪面/宏观新闻，异构 LLM 防回声室 |
| **分歧驱动辩论** | 低分歧 (δ<0.15) → 跳过辩论；高分歧 → 置信度降级 + 仓位缩减，**不做方向判断** |
| **结构化 JSON 决策** | 输出可执行交易指令（入场/止损/目标/仓位），非 Markdown 研报 |
| **三角风控** | 激进/保守/中立独立审核，一票否决 + CVaR 硬约束 |
| **CVRF 学习闭环** | 交易 → 评估 → LLM 反思 → 向量记忆 → 下次自动注入历史教训 |
| **20 个内置策略** | Freqtrade 三层抽象（指标 → 入场 → 出场） |
| **Walk-Forward 回测** | 时间一致划分 + 成本建模 + 置信度校准 (ECE ≤ 0.05) |
| **P1-P6 合规报告** | 时间一致性、Point-in-Time 数据、多档成本、消融、偏差检测 |
| **多交易所** | Binance/OKX/Bybit 统一接口 |
| **全链路审计** | 每条决策记录：时间戳、模型、Token、证据链、LLM 原始响应 |
| **偏差自动检查** | 5 类偏差检测：前视/幸存者/叙事/目标/成本 |

## 快速开始

```bash
# 安装
cd qmind
pip install -e ".[dev,sources,cex]"

# 设置 API Key
set DEEPSEEK_API_KEY=sk-...

# 分析 BTC/USDT（模拟数据，不实际下单）
qmind --source mock analyze BTC/USDT

# 使用币安实时数据（墙内需代理）
qmind --source binance analyze BTC/USDT
```

## CLI 用法

### 全局选项

```
qmind [选项] 命令 [参数]

选项:
  -c, --config PATH   配置文件路径
  --dry-run / --live  模拟模式（默认 dry-run）
  --source TEXT       数据源：auto / binance / yfinance / tushare / mock
  -v, --verbose       显示每个智能体的详细思考过程
  --help              显示帮助
```

### 命令详解

#### `analyze` — 一次性分析

```bash
# 简洁模式
qmind analyze BTC/USDT

# 详细模式：看分析师推理、辩论、风控意见
qmind -v analyze BTC/USDT

# 选择数据源
qmind --source binance analyze ETH/USDT
qmind --source yfinance analyze AAPL
qmind --source tushare analyze 000001.SZ
qmind --source mock analyze BTC/USDT    # 模拟数据，不依赖网络

# 实盘交易（需配置交易所 API Key）
qmind --live analyze BTC/USDT
```

详细模式输出示例：

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

#### `watch` — 持续监控

```bash
# 监控标的，有交易信号时自动推送通知
qmind watch BTC/USDT
qmind watch BTC/USDT ETH/USDT
qmind watch BTC/USDT --timeframe 4h --interval 300
```

决策不是 HOLD **且** 风控通过时触发飞书/邮件通知。

#### `backtest` — 回测

```bash
# 列出所有注册策略
qmind list --strategies

# 回测策略
qmind backtest --strategy ma_cross --start 2024-01 --end 2025-06
```

#### `learn` — CVRF 学习

```bash
# 查看当前经验库
qmind learn

# 从交易日志学习
qmind learn --from-log trades.log
```

#### 其他命令

```bash
qmind price BTC/USDT         # 快速查价
qmind list --strategies      # 列出 20 个内置策略
qmind list --audit           # 查看审计摘要
qmind version                # 版本信息
```

## 内置策略（20 个）

**均线类:** `ma_cross`, `ma_cross_triple`, `triple_ema`, `macd`, `macd_rsi`
**反转类:** `rsi`, `cci`, `stoch`, `williams_r`, `kdj`
**趋势类:** `adx_macd`, `psar`, `ichimoku`
**波动率类:** `bollinger`, `atr_stop`, `chandelier`
**成交量类:** `obv`, `mfi`, `volume_breakout`
**通道类:** `donchian`

## 数据源

| 数据源 | 命令 | 要求 | 说明 |
|--------|------|------|------|
| Binance | `--source binance` | 网络（墙内需代理） | 加密货币实时行情 |
| Yahoo Finance | `--source yfinance` | 网络 | 美股，有限流 |
| Tushare | `--source tushare` | API token（免费申请） | A 股/港股 |
| Mock | `--source mock` | 无 | 模拟数据，离线测试 |

## 网络 / 代理

如果网络受限（如中国大陆），设置代理环境变量：

```powershell
set ALL_PROXY=http://127.0.0.1:7890
set NO_PROXY=localhost,127.0.0.1,api.deepseek.com
```

系统自动检测 Windows 系统代理设置。

## 配置

创建 `config.yaml`：

```yaml
llm:
  default_model: deepseek-chat
  deepseek_api_key: "${DEEPSEEK_API_KEY}"   # 或设环境变量

execution:
  dry_run: true
  default_exchange: binance

storage:
  db_path: qmind.db

notification:
  type: none          # none / feishu / email
```

支持 DeepSeek、Anthropic（`ANTHROPIC_API_KEY`）、OpenAI（`OPENAI_API_KEY`）。

## 架构

```
数据采集 → 4分析师并行 → 分歧驱动辩论 → Financial CoT 决策 → 三角风控 → 执行/拒绝
                                                                        ↓
                                                                   CVRF 学习
```

## 项目结构

```
qmind/
├── agents/               # 多角色 Agent
│   ├── analysts/         #   技术面/基本面/情绪面/新闻
│   ├── researchers/      #   多空辩论（Trust / Skeptic / Leader）
│   ├── risk.py           #   三角风控 + CVaR
│   ├── protocol.py       #   通信 JSON Schema
│   └── single_agent.py   #   单 Agent 基线
├── graph/                # LangGraph 流水线
│   ├── pipeline.py       #   六阶段 StateGraph（含 CVRF 节点）
│   ├── routers.py        #   条件路由
│   └── state.py          #   AgentState TypedDict
├── llm/                  # LLM 调用层
│   ├── client.py         #   Anthropic/OpenAI/DeepSeek 统一客户端
│   ├── router.py         #   双 LLM 路由
│   └── structured.py     #   Pydantic 结构化输出 + 自动重试
├── data/sources/         # binance, yfinance, tushare, mock
├── execution/            # 执行层（Binance/OKX/Bybit + dry_run）
├── learning/             # CVRF 学习：反思、记忆（SQLite）、注入
├── backtest/             # 回测框架：分区、成本、校准、消融、报告
├── strategies/builtin/   # 20 个策略
├── audit/                # 偏差检查器（5 类偏差）
├── main.py               # CLI 入口
└── config.py             # 配置管理
```

## 测试

```bash
make test       # 621 个测试，19.75s
make lint       # ruff 检查
make coverage   # 测试覆盖率 72%
```

## 设计原则

基于 18 篇同行评审论文（TradingAgents、FINCON、TiMi、FinDebate、Alpha Illusion 等）的 LLM 金融 Agent 研究结论：

1. **辩论校正偏差，非生成 alpha** — 分歧触发风控降级，绝不改变方向
2. **LLM 置信度 ≠ 可交易概率** — 独立校准（ECE ≤ 0.05）后才用于仓位计算
3. **仓位不由 LLM 决定** — 风控模块用 CVaR 约束独立处理
4. **始终报告净 PnL** — 佣金+滑点+价差+Gas 费显式建模
5. **时间一致性强制** — Point-in-Time 数据控制，`TimeGuard` 禁止前视偏差

## MIT 协议
