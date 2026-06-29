# QMind — 量化交易智能体

## 项目定位

QMind 是一个 **LLM 驱动的多智能体量化交易系统**。核心思路：用多个 AI 角色协作（分析→辩论→决策→风控），替代单一 LLM 的独断式交易判断；每笔交易后通过 CVRF（概念言语强化学习）自动总结教训，让系统越交易越聪明。

---

## 一、场上有哪些成熟方案？我们借鉴谁？

### 1.1 开源量化智能体生态全景

```
                        LLM 参与度
                        高 ▲
                           │
              FinGPT       │   TradingAgents  ← ★ 骨架
              (情感分析)    │   (多Agent+辩论+风控)
              ⭐19k        │   ⭐ 50k stars
                           │
              FinRobot     │   FINCON
              (Agent平台)   │   (CVRF学习+CVaR约束)
              ⭐6.7k       │   NeurIPS 2024
                           │
    ───────────────────────────────────────────────► 交易执行能力
              Qlib         │   FinRL / FinRL-X
              (因子/回测)   │   (强化学习交易)
              ⭐40k        │
                           │
                        低 ▼
                            低 ◄────────────────────► 高
```

### 1.2 逐个点评：抄什么、不抄什么

#### TradingAgents（⭐50k+ stars，MIT / 山大维护）

> 📄 **论文**: [TradingAgents: Multi-Agents LLM Financial Trading Framework](https://arxiv.org/abs/2412.20138) · arXiv 2412.20138 · 2024.12 · UCLA/MIT
> 💾 **本地**: `references/papers/TradingAgents_2412.20138.pdf`

**目前最成熟的开源多智能体交易框架。**

| 维度     | 评价                                                                                |
| -------- | ----------------------------------------------------------------------------------- |
| 架构     | LangGraph 五阶段状态机，7+ 角色（4 分析师 + 多方/空方研究员 + 交易员 + 风控经理）   |
| 核心亮点 | **多空辩论机制**——多方和空方分别论证，辩论 2-3 轮直到收敛或达到最大轮次     |
| 风控     | **三角风控辩论**——激进/保守/中立三个角度独立审核，任一否决即取消            |
| 记忆     | ChromaDB 向量记忆，按角色分 collection（bull_memory / bear_memory / trader_memory） |
| 双 LLM   | `deep_thinking_llm`（推理）+ `quick_thinking_llm`（工具调用），不同模型各司其职 |
| 报告成本 | ~32K token / 份，约 ¥0.2-0.3（DeepSeek 定价），90-135 秒 / 完整流程                |
| 论文性能 | 年化 24%-30%（回测），超出传统策略 6-24 个百分点                                    |
| 输出格式 | **Markdown 研报**，不是执行指令——这是它最大的局限                           |
| 数据层   | yfinance + finnhub，不适合 A 股/加密货币                                            |

**→ QMind 抄它的：** 五阶段图结构 + 4 分析师 prompt + 多空辩论循环 + 三角风控 + 双 LLM 策略

**→ QMind 不抄它的：** 数据层（我们用自己的）、LangChain 封装（我们用原生 SDK）、输出格式（我们要结构化执行指令，不要 Markdown 报告）

---

#### FINCON（NeurIPS 2024，Stevens Institute / Harvard）

> 📄 **论文**: [FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making](https://arxiv.org/abs/2407.06567) · arXiv 2407.06567 · NeurIPS 2024
> 💾 **本地**: `references/papers/FINCON_2407.06567.pdf`

**目前唯一一个把"LLM 交易后学习"做成正经论文方案的系统。**

| 维度       | 评价                                                                                                                          |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 核心创新   | **CVRF（概念言语强化学习循环）**——交易结果 → LLM 用自然语言总结"学到了什么" → 更新概念记忆库 → 注入下次决策 prompt |
| 为什么重要 | 传统 RL 需要数千次迭代才能收敛，CVRF 只需**4 个 episode**                                                               |
| CVaR 约束  | Conditional Value at Risk——不是只看"亏多少"，而是看"最坏那 5% 的情况亏多少"                                                 |
| 消融实验   | 去掉 CVRF → GOOG 收益从 +25% 降到 -12%；去掉 CVaR → 组合收益从 114% 降到 15%                                                |
| 实验表现   | TSLA 单股累计收益**82.87%** vs 买入持有 6.43%；三股组合累计 **113.84%** vs Markowitz 12.64%                       |
| 代码质量   | 学术代码，工程化很差，不能直接用                                                                                              |

**→ QMind 抄它的：** CVRF 学习循环设计、CVaR 风险约束公式、情景记忆检索思路、概念梯度更新（自然语言而非数值梯度）

**→ QMind 不抄它的：** 代码（自己重写）、单股票交易逻辑（我们要多品种）

---

#### FinRobot（⭐6.7k，AI4Finance Foundation）

> 📄 **平台论文**: [FinRobot: An Open-Source AI Agent Platform for Financial Applications using LLMs](https://arxiv.org/abs/2405.14767) · arXiv 2405.14767 · 2024.05
> 📄 **CoT 论文**: [FinRobot: AI Agent for Equity Research and Valuation with LLMs](https://arxiv.org/abs/2411.08804) · arXiv 2411.08804 · 2024.11 · ICAIF 2024
> 💾 **本地**: `references/papers/FinRobot_Platform_2405.14767.pdf` / `FinRobot_CoT_2411.08804.pdf`

| 维度           | 评价                                                                      |
| -------------- | ------------------------------------------------------------------------- |
| 架构           | 四层：Agent 层 → LLM 算法层 → LLMOps/DataOps 层 → 多源 LLM 基础层      |
| 核心亮点       | **Financial CoT 三级推理链**：Data-CoT → Concept-CoT → Thesis-CoT |
| 组织模式       | Leader-Worker，一个 Manager Agent 分配任务给多个 Analyst Agent            |
| 局限           | 依赖 AutoGen（太重），99% 功能我们用不上；Stars 相对少，社区小            |
| Benchmark 评分 | AnalyScore 61 分，低于 GPT-4o 原生 66 分，高于 FinGPT 45 分               |

**→ QMind 抄它的：** Financial CoT 三级推理 chain prompt 模板

**→ QMind 不抄它的：** 整个框架（AutoGen 太重）、LLMOps 层

---

#### FinGPT（⭐19k，AI4Finance Foundation）

> 📄 **论文**: [FinGPT: Democratizing Internet-scale Data for Financial Large Language Models](https://arxiv.org/abs/2307.10485) · arXiv 2307.10485 · 2023.07 · Columbia/Rice
> 💾 **本地**: `references/papers/FinGPT_2307.10485.pdf`

| 维度     | 评价                                                                  |
| -------- | --------------------------------------------------------------------- |
| 核心能力 | 金融情感分析 + 金融指令微调 + 财报问答                                |
| 定位     | 金融大模型框架，**不是 Agent 系统**，不包含交易执行             |
| 擅长     | 新闻/社交媒体情感分析、金融 NLP                                       |
| 局限     | 单独使用无法形成交易闭环；Benchmark 45 分远低于通用 LLM；距离实盘很远 |

**→ QMind 抄它的：** LoRA 微调管线（未来做专用金融 LLM 时参考）；RAG Agent 搜索模式

**→ QMind 现在不动它：** 当前用通用 LLM（Claude/GPT/DeepSeek）足够，不需要微调

---

#### Qlib（⭐40k，Microsoft）

传统量化研究平台，不含 LLM。专注因子挖掘、回测、模型训练。QMind 的回测和因子层可以借鉴它的工程思路，但不属于"智能体"范畴。

---

### 1.3 一句话总结

| 项目                    | 给 QMind 提供什么                                       |
| ----------------------- | ------------------------------------------------------- |
| **TradingAgents** | 骨架—— LangGraph 图 + 7 角色 + 辩论风控 + 双 LLM      |
| **FINCON**        | 大脑—— CVRF 学习循环，让系统从亏损中变聪明            |
| **Freqtrade**     | 策略抽象——三层策略接口（indicators → entry → exit） |
| **FinRobot**      | prompt 补充——Financial CoT 三级推理链                 |
| **FinGPT**        | 未来备选——LoRA 微调 / RAG 搜索                        |
| **PA_Agent**      | 分析经验——两阶段分析流水线 + 经验库                   |
| **fin-agent**     | 记忆+算盘——22 策略回测 + Tushare 数据 + 持仓管理      |

---

## 二、QMind 要做什么？

### 2.1 核心功能

```
QMind = 多角色 AI 辩论 + 结构化交易决策 + 自动学习进化 + 可执行下单
```

| 功能                     | 说明                                                                              |
| ------------------------ | --------------------------------------------------------------------------------- |
| **多维度市场分析** | 4 个分析师并行，各自从基本面/技术面/情绪面/宏观面产出结构化分析报告               |
| **多空辩论**       | 多方研究员 vs 空方研究员，2-3 轮辩论直到收敛，对抗单一 LLM 的偏见和幻觉           |
| **结构化交易决策** | 交易员输出 JSON 决策指令（方向/入场价/止损/目标/仓位/置信度），不是 Markdown 研报 |
| **三角风控审核**   | 激进/保守/中立三个角度独立审核，一票否决制，硬约束不可跳过                        |
| **CVRF 持续学习**  | 每笔交易结束 → LLM 总结教训 → 存入概念记忆库 → 相似市况自动注入历史教训        |
| **多交易所执行**   | 统一的下单/撤单/查单接口，支持 CEX（12 个）+ DEX（EVM/Solana）                    |
| **策略回测**       | 从 fin-agent 迁入的 22+ 策略 + Freqtrade 三层策略接口 + 回测引擎                  |
| **后台调度**       | 定时轮询、预警触发、多品种并行监控                                                |

### 2.2 与其他方案的关键区别

| 维度           | TradingAgents | FINCON | FinRobot | **QMind** |
| -------------- | :-----------: | :-----: | :------: | :-------------: |
| 多角色辩论     |      ✅      |   ❌   |    ❌    |       ✅       |
| 输出执行指令   |    ❌ 研报    |   ✅   | ❌ 研报 |       ✅       |
| 交易后学习     |  ⚠️ 有反射  | ✅ CVRF |    ❌    |     ✅ CVRF     |
| CoT 推理链     |      ❌      |   ❌   |    ✅    |       ✅       |
| 风控一票否决   |      ✅      |   ❌   |    ❌    |       ✅       |
| 策略回测       |      ❌      |   ❌   |    ❌    |   ✅ 22+ 策略   |
| 多交易所执行   |      ❌      |   ❌   |    ❌    |       ✅       |
| 不用 LangChain |      ❌      |  ⚠️  |    ❌    |       ✅       |

---

## 三、架构设计

### 3.1 整体层级

```
┌─────────────────────────────────────────────────────────┐
│                     QMind Architecture                     │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │
│  │ 基本面   │  │ 技术面   │  │ 情绪面   │  │ 宏观/    │     │
│  │ 分析师   │  │ 分析师   │  │ 分析师   │  │ 新闻分析师│     │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘     │
│       │            │            │            │           │
│       └────────────┴────────────┴────────────┘           │
│                        │ 结构化分析报告                    │
│           ┌────────────┴────────────┐                    │
│           ▼                         ▼                    │
│    ┌─────────────┐          ┌─────────────┐              │
│    │ 多方研究员    │  ◄辩论►  │ 空方研究员    │              │
│    │ (Bull)       │  2-3轮   │ (Bear)       │              │
│    └──────┬──────┘          └──────┬──────┘              │
│           │        辩论纪要        │                      │
│           └────────────┬────────────┘                    │
│                        ▼                                 │
│                 ┌─────────────┐                          │
│                 │   交易员      │ ← Financial CoT         │
│                 │ (结构化决策)  │   Data→Concept→Thesis   │
│                 └──────┬──────┘                          │
│                        ▼                                 │
│           ┌─────────────────────┐                        │
│           │   三角风控审核        │                        │
│           │  激进 / 保守 / 中立   │ ← 任一否决即取消        │
│           │  + CVaR 硬约束       │                        │
│           └──────────┬──────────┘                        │
│                      ▼ (通过)                             │
│           ┌─────────────────────┐                        │
│           │     执行层           │                        │
│           │  12 CEX + DEX       │                        │
│           │  dryRun → 实盘确认   │                        │
│           └──────────┬──────────┘                        │
│                      ▼                                    │
│           ┌─────────────────────┐                        │
│           │   CVRF 学习循环      │  ← 交易结果反馈          │
│           │  教训 → 记忆 → 注入   │                        │
│           └─────────────────────┘                        │
│                                                           │
│  ┌──────────────────────────────────────────────────┐    │
│  │         LangGraph 状态图 + SQLite 持久化           │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 3.2 五阶段流水线详解

#### 阶段 1：数据采集（Data Collection）

```
输入：标的列表 + 时间框架
    │
    ├─ 交易所 WS 实时行情（K线/深度/资金费率/合约信息）
    ├─ PA_Agent 数据源（MT5/TV/yfinance/AkShare）
    ├─ fin-agent 数据源（Tushare A股/港股/美股）
    └─ 可选：外部新闻/情绪数据 API
    │
    ▼
输出：统一格式的结构化市场数据包
    {
      "symbol": "BTC/USDT",
      "klines": {"1m": [...], "5m": [...], "1h": [...], "1d": [...]},
      "orderbook": {"bids": [...], "asks": [...]},
      "funding_rate": 0.0001,
      "open_interest": 1234567890,
      "news": [...],
      "timestamp": "2026-06-29T12:00:00Z"
    }
```

#### 阶段 2：多维分析（Analysis）— 4 个分析师并行

每个分析师接收数据包，按自己的专业视角输出**结构化报告**（JSON，不是闲聊）：

| 分析师          | 关注维度                     | 核心能力                                   | 参考来源                       |
| --------------- | ---------------------------- | ------------------------------------------ | ------------------------------ |
| 基本面分析师    | 财务数据、估值、行业对比     | P/E、ROE、DCF、现金流分析                  | PA_Agent 分析引擎              |
| 技术面分析师    | K线形态、指标信号、量价关系  | MA/MACD/RSI/KDJ/布林带/形态识别            | fin-agent technical_indicators |
| 情绪分析师      | 市场情绪、资金流向、持仓变化 | 多空比、资金费率、持仓量变化、Social Media | FinGPT 情感分析思路            |
| 宏观/新闻分析师 | 政策、宏观经济、重大事件     | 利率、CPI/GDP、监管政策、突发新闻          | Tushare 宏观数据               |

```
并行执行（asyncio.gather，单分析师超时不影响整体）
    │
    ▼
每个分析师输出结构化报告 JSON：
    {
      "analyst": "technical",
      "stance": "看多",           // 看多 / 中性 / 看空
      "confidence": 0.72,         // 0.0 - 1.0
      "core_reason": "...",       // 核心逻辑（一句话）
      "key_signals": [...],       // 关键信号列表
      "risk_factors": [...],      // 风险因素
      "support_price": 86200,     // 支撑位
      "resistance_price": 89500   // 压力位
    }
```

**抗同质化策略**：不同分析师用不同 LLM 或不同 temperature——比如基本面分析师用 Claude（谨慎），技术面用 GPT-4o（快速），情绪面用 DeepSeek（便宜），避免四个报告变成同一种声音。

#### 阶段 3：多空辩论（Debate）— 核心差异化

这是 QMind 最重要的环节。不是让一个 LLM 自己和自己辩论（会变废话），而是：

```
多方研究员 (Bull Researcher)
    │
    │  接收：4 个分析师报告
    │  任务：构建看多论证，提出入场理由和证据
    │  输出：多方辩论稿 (JSON)
    │
    ├─── Round 1 ───┤
    │  多方：发表看多论点
    │  空方：反驳多方论点，提出看空证据
    │  → 判断是否收敛（使用 LLM-as-Judge 评分）
    │
    ├─── Round 2（如未收敛）──┤
    │  多方：回应空方质疑，补充新论据
    │  空方：再次反驳，提出未覆盖的风险
    │  → 再次判断收敛
    │
    ├─── Round 3（如仍未收敛，最多到此）──┤
    │  多方 + 空方各自最终陈述
    │  → 强制收敛，输出争议点标注
    │
    ▼
输出：辩论纪要
    {
      "rounds": 2,
      "converged": true,
      "final_stance": "看多",
      "bull_core_argument": "...",
      "bear_core_counter": "...",
      "agreement_points": [...],     // 双方共识
      "disagreement_points": [...],  // 双方分歧（重要！）
      "consensus_confidence": 0.68,  // 收敛置信度
      "debate_transcript": [...]     // 完整辩论记录（用于学习）
    }
```

**关键设计**：

- 多方和空方**必须用不同 LLM 或不同 temperature**——同模型同温度辩论会变成相互附和
- 收敛判断不是简单比较 stance，而是用 LLM-as-Judge 评估双方是否"在不同事实判断上达成一致"
- 辩论纪要是后续 CVRF 学习的重要输入——事后可以回看"当时谁对了、谁错了"

#### 阶段 4：交易决策（Decision）

交易员接收辩论纪要 + 4 份分析师报告，执行 **Financial CoT 三级推理**：

```
Data-CoT：     原始数据 → "K线在支撑位附近企稳，MACD金叉，量能放大"
Concept-CoT：  数据结论 → "处于下跌趋势末端反转初期，波动率收缩后即将突破"
Thesis-CoT：   概念框架 → "突破震荡区间上沿的概率 > 继续下跌，建议做多"
```

交易员输出**结构化决策指令**（这是跟 TradingAgents 最大的不同——它输出 Markdown 研报，我们输出可执行的 JSON）：

```json
{
  "decision": "LONG",
  "symbol": "BTC/USDT",
  "entry": {
    "type": "limit",
    "price": 87200.00,
    "quantity": 0.15,
    "order_type": "GTC"
  },
  "stop_loss": {
    "price": 86200.00,
    "type": "stop_market",
    "reason": "跌破阶段2支撑位"
  },
  "take_profit": [
    {"price": 89500.00, "ratio": 0.5, "reason": "第一压力位"},
    {"price": 91000.00, "ratio": 0.5, "reason": "前高"}
  ],
  "position_size_pct": 12.5,
  "confidence": 0.72,
  "time_horizon": "4h",
  "reasoning_chain": {
    "data_cot": "...",
    "concept_cot": "...",
    "thesis_cot": "..."
  },
  "risk_reward_ratio": 2.3,
  "max_acceptable_loss_pct": 1.15
}
```

#### 阶段 5：风控审核（Risk）— 一票否决

三个风控角色独立审核交易员的决策，**各自写审核意见**：

| 风控角色 | 倾向     | 关注点                                 | 权重 |
| -------- | -------- | -------------------------------------- | ---- |
| 激进风控 | 偏向通过 | "这笔交易的机会有多大？错过会后悔吗？" | 1/3  |
| 保守风控 | 偏向否决 | "最坏情况是什么？能不能承受？"         | 1/3  |
| 中立风控 | 客观评估 | "风险收益比是否合理？有没有盲区？"     | 1/3  |

```
任一风控否决 → 取消交易
全部通过 → 加上 CVaR 硬约束校验 → 进入执行
```

**CVaR 硬约束**（来自 FINCON）：

```
CVaR(95%) = 在历史最差 5% 交易日的平均亏损
如果 当前仓位 × 预期最大波动 > CVaR 阈值 → 强制缩小仓位或拒绝
```

风控审核输出：

```json
{
  "approved": true,
  "veto_count": 0,
  "adjustments": {
    "position_size_pct": 8.0   // 风控可能缩小仓位
  },
  "aggressive_opinion": "...",
  "conservative_opinion": "...",
  "neutral_opinion": "...",
  "cvar_check": {
    "passed": true,
    "current_cvar_exposure": 3200,
    "threshold": 5000,
    "margin": 1800
  }
}
```

### 3.3 CVRF 学习循环（闭环）

这是 QMind 与所有其他系统的最大区别——**每一笔交易都是一次学习**。

```
交易完结（止盈/止损/手动平仓）
    │
    ▼
┌──────────────────────────────────────┐
│  交易结果评估                          │
│  ├─ PnL (盈亏金额 + 百分比)            │
│  ├─ 持仓时长                          │
│  ├─ 最大浮亏 (MAE)                    │
│  ├─ 最大浮盈 (MFE)                    │
│  ├─ 滑点 vs 预期                       │
│  └─ 执行质量（是否按计划执行）           │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│  CVRF 反思（Conceptual Verbal RL）    │
│                                      │
│  LLM 提示词：                          │
│  "这是一笔 {盈利/亏损} 的交易。         │
│   当初做决策时我们认为：{原分析报告}     │
│   辩论中多方说：{多方论点}              │
│   辩论中空方说：{空方论点}              │
│   实际发生了什么：{市场走势}            │
│                                      │
│   请你回答：                           │
│   1. 当时哪个判断是对的？哪个是错的？    │
│   2. 空方提出的风险哪些应验了？          │
│   3. 从这笔交易中学到的 3 条教训是什么？  │
│   4. 下次遇到类似市况应该注意什么？"     │
│                                      │
│  输出：结构化教训 JSON                  │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│  概念记忆库更新（SQLite）               │
│                                      │
│  lessons 表结构：                      │
│  ├─ id                               │
│  ├─ timestamp                        │
│  ├─ market_condition_vector (JSON)   │  ← 市况特征向量
│  │   ├─ trend: "downtrend_reversal"  │
│  │   ├─ volatility: "low"            │
│  │   ├─ market_cycle: "accumulation" │
│  │   └─ ...                          │
│  ├─ lessons (JSON[])                 │  ← 自然语言教训列表
│  │   └─ [{lesson, confidence, source}]│
│  ├─ trade_outcome (JSON)             │  ← 交易结果
│  ├─ was_bull_correct: bool           │  ← 多方对了吗？
│  ├─ was_bear_correct: bool           │  ← 空方对了吗？
│  └─ embedding (BLOB)                 │  ← 用于相似市况检索
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│  下次交易时自动注入                    │
│                                      │
│  当前市况 → 向量相似度检索 → TOP-5 相关教训 │
│  → 注入到 4 个分析师的 prompt 中：       │
│  "注意：历史上类似市况曾出现以下教训：     │
│   1. 回调时过早入场导致被止损 (3次相似)   │
│   2. 突破确认后再入场盈利更高 (2次相似)"  │
└──────────────────────────────────────┘
```

### 3.4 策略层设计

```
Freqtrade 三层抽象 + fin-agent 22 策略
        │
        ▼
┌─────────────────────────────────────────────┐
│  BaseStrategy                               │
│  ├─ populate_indicators(df) → df            │
│  ├─ populate_entry_signal(df) → df          │
│  └─ populate_exit_signal(df) → df           │
│                                              │
│  策略注册表 (registry)                        │
│  @register_strategy("ma_cross")             │
│  class MACrossStrategy(BaseStrategy): ...   │
│                                              │
│  内置策略（从 fin-agent 迁入）：               │
│  双均线 / MACD / RSI / KDJ / 布林带 /        │
│  Donchian / ADX+MACD / 三均线 / 量价突破 /    │
│  ATR止损 / CCI / Williams%R / ...共22+       │
└─────────────────────────────────────────────┘
```

策略和 AI Agent 的关系：

- **回测模式**：纯策略引擎运行，不涉及 LLM，验证策略本身的统计有效性
- **Agent 模式**：LLM Agent 分析 + 辩论 + 决策，策略信号作为**参考输入**之一（注入技术面分析师）
- **混合模式**：策略信号 + AI 分析投票加权

---

## 四、论文审阅后的架构修正

> 📄 完整精读报告：`references/papers/_SYNTHESIS.md`（531 行，18 篇论文逐篇分析）

### 三条"绝不"（来自 18 篇论文审阅的最强信号）

1. **绝不让 LLM 置信度冒充可交易概率** — 必须独立校准（ECE ≤ 0.05 才可用于仓位计算）
2. **绝不让 LLM 输出直接决定仓位大小** — 仓位计算必须独立于 LLM，由风控模块处理
3. **绝不报告 Gross PnL 而不报告 Net PnL** — Gross - Net 的差值就是 alpha 幻觉的量化度量

### 辩论机制的重新定位（最重要发现）

4 篇独立论文收敛到同一结论：**辩论的真正作用是偏差校正，不是 alpha 生成。**

| 证据 | 来源 |
|------|------|
| Debate Agent Sharpe ≈ avg(鹰派, 鸽派) + 0.001，未击败最佳单 Agent (p=0.769) | MacroAgent @ arxiv 2606 |
| 36 组实验中多 Agent 辩论胜率 < 20%，增轮无益 | Zhang et al. 2025 |
| 同模型 Agent 输出高度收敛（回声室效应） | LLM ≠ Human @ arxiv 2502 |
| 辩论退化促成了 FinDebate 的完整防退化协议 | FinDebate @ EMNLP 2025 |

**修正方案**：QMind 辩论阶段**不做方向判断**，只做风险降级。

```
低分歧 (δ<0.15) → 直接采信最强 Agent，不做辩论（避免稀释）
高分歧 (δ≥0.15) → 启动风控审核模式：
  ├─ 方向锁定（绝不改变 Long/Short/Neutral）
  ├─ 单轮辩论（多轮导致主题漂移）
  └─ 只输出：置信度降级因子 + 仓位缩减比例
```

### 执行层解耦（借鉴 TiMi @ ICLR 2026）

TiMi 的 200+ 交易对实盘验证证明：**LLM 负责策略生成（离线），代码负责执行（实时）**。

```
═══ LLM 离线做的事 ═══          ═══ LLM 绝不实时做的事 ═══
信息抽取 → 结构化 JSON           直接决定仓位大小
策略逻辑 → 可执行 Python 代码     自报交易概率
概率校准 → Platt scaling        实时方向辩论
                                (降级为离线风控审核)
```

### P0 必须立刻修改（否则方法论不可信）

| # | 修改项 | 工作量 | 关键来源 |
|---|--------|:---:|------|
| 1 | 回测增加时间一致性划分（walk-forward, 显式日期边界） | 2-3天 | P1, MR-2 |
| 2 | Point-in-Time 数据控制（as_of 时间戳，经验库时间过滤） | 3-5天 | P1, 五罪 Look-Ahead |
| 3 | 显式建模交易成本 + 强制报告 Net PnL | 2-3天 | P5, MR-4 |
| 4 | 禁止 LLM 置信度直接用于仓位控制 + 独立校准模块 | 3-5天 | P4, 五罪 Objective |

### P1 重要修正

| # | 修改项 | 工作量 | 关键来源 |
|---|--------|:---:|------|
| 5 | 执行时机语义：信号 t 收盘 → 执行 t+1 收盘 (next-close) | 1天 | MR-3 |
| 6 | 回测窗口 ≥ 3-5年跨体制 + 分段报告 | 2-3天 | Failure 5 |
| 7 | 辩论机制重新设计（方向锁定 + 单轮 + 风控模式） | 3-5天 | MacroAgent, FinDebate |
| 8 | 建立单 Agent 强基线 + 消融实验框架 | 3-5天 | CPH |
| 9 | 异构 LLM 混用避免回声室（不同模型 per Agent） | 2-3天 | LLM≠Human |
| 10 | 动态股票池 U_t（退市/ST 纳入而非剔除） | 2-3天 | 五罪 Survivorship |

---

## 五、技术选型

### 核心决策

| 层面       | 用什么                                                               | 为什么                                         |
| ---------- | -------------------------------------------------------------------- | ---------------------------------------------- |
| 编排       | LangGraph StateGraph + conditional edges                             | 状态持久化（checkpoint）、条件路由、可恢复执行 |
| LLM 调用   | Anthropic SDK / OpenAI SDK 原生                                      | 少一层抽象，追 bug 更快，API 文档即真相        |
| 工具定义   | JSON Schema 裸写                                                     | 跟 Claude Code 同款格式，不依赖框架            |
| 结构化输出 | Pydantic + `parse()`                                                | 强类型校验，输出不符合 schema 自动重试         |
| 状态持久化 | SQLite (`langgraph-checkpoint-sqlite`)                               | 轻量、零运维、方便调试                         |
| 记忆系统   | SQLite 自建（lessons + embeddings 表）                               | CVRF 是高度定制记忆，通用模块改起来更累        |
| LLM 选型   | Claude（深度推理）+ GPT-4o（快速工具调用）+ DeepSeek（高性价比批量） | 不同任务用最合适的模型                         |
| Python     | 3.11+                                                                | match/case、更好的类型提示                     |

### 关于 LangChain 的取舍说明

QMind **用 LangChain 的一部分，但不是全盘接受**。具体：

#### ✅ 用（LangChain 最成熟的核心）

| 模块 | 理由 |
|------|------|
| `langgraph` + `langgraph-checkpoint-sqlite` | 图编排是 LangChain 做得最好的部分，StateGraph 的 checkpoint/条件路由没有替代品。TradingAgents 也用这个 |

#### ❌ 不用（抽象层弊大于利）

| 模块 | 不用理由 |
|------|----------|
| `langchain` LLM 封装（ChatOpenAI、ChatAnthropic 等） | 多一层封装多一层心智负担。直接调 SDK 代码更短、出错时看官方文档更快。LangChain 版本升级经常 breaking change，SDK 向后兼容 |
| `langchain` BaseTool / `@tool` | JSON Schema 裸写跟 Claude Code 同款格式，不依赖框架。BaseTool 的 pydantic_v1/v2 兼容问题已经坑了无数项目 |
| `langchain` output parser | Pydantic 的 `model_validate_json()` + `beta.parse()` 官方原生支持，不需要中间层 |
| `langchain` 记忆模块 | CVRF 学习循环的 memory 是高度定制的（市况向量 + 教训权重 + 时间衰减 + 体制变化检测），LangChain 通用对话记忆改起来比自建还累 |
| `langchain` callback/追踪 | Cost tracking 自己写不到 50 行 |

**所以"不用 LangChain"这个说法不精确。更准确的说法是：用 LangGraph（LangChain 的图编排），跳过其他可有可无的抽象层。** Beyond Agent Architecture（2606.08285）审计了 30 个 LLM 交易系统，用 LangChain 完整封装的普遍比原生 SDK 的更难复现——多一层就多一个可能出配置问题的点。

### 关于 ChromaDB/向量数据库

也不用。CVRF 的记忆检索场景很轻（<= 1000 条教训，cosine similarity 算一次不到 1ms），SQLite 存浮点向量 + Python 算相似度就够了。加一个 ChromaDB 意味着多一个进程、多一个备份问题、多一个调试维度。**只在教训条目超过 10 万条时才需要考虑专用向量库。**

---

## 六、参考项目存放

```
quant agent/
├── references/                    # 参考资源（只读！不修改）
│   ├── papers/                    # 📄 论文 PDF
│   │   ├── TradingAgents_2412.20138.pdf    (1.9 MB)
│   │   ├── FINCON_2407.06567.pdf           (14 MB — NeurIPS 长文)
│   │   ├── FinRobot_Platform_2405.14767.pdf (5.2 MB)
│   │   ├── FinRobot_CoT_2411.08804.pdf     (2.3 MB)
│   │   └── FinGPT_2307.10485.pdf           (719 KB)
│   │
│   ├── TradingAgents/             # git clone — 骨架参考（⭐50k）
│   ├── FINCON/                    # git clone — CVRF 学习系统参考
│   ├── FinRobot/                  # git clone — CoT prompt 参考（⭐6.7k）
│   └── FinGPT/                    # git clone — 备选，情感分析微调参考（⭐19k）
│
├── CLAUDE.md                      # 本文档
├── main.py                        # 入口
├── config.py                      # 统一配置
│
├── graph/                         # LangGraph 状态图
│   ├── state.py                   # AgentState TypedDict
│   ├── pipeline.py                # 主图：五阶段流水线
│   └── routers.py                 # 条件路由
│
├── agents/                        # 多角色 Agent
│   ├── analysts/                  # 4 个分析师
│   ├── researchers/               # 多方/空方 + 辩论
│   ├── trader.py                  # 交易员
│   └── risk.py                    # 风控三角辩论
│
├── llm/                           # LLM 调用层
│   ├── client.py                  # 多供应商统一入口
│   └── structured.py             # Pydantic 结构化输出
│
├── tools/                         # 工具层（JSON Schema）
│   ├── market_data.py
│   ├── order.py
│   ├── backtest.py
│   └── portfolio.py
│
├── learning/                      # CVRF 学习系统
│   ├── cvrf.py
│   ├── memory.py
│   └── risk_constraints.py
│
├── data/                          # 数据源
│   ├── sources/                   # MT5/TV/yfinance/AkShare/Tushare
│   └── ws/                        # 交易所 WebSocket
│
├── strategies/                    # 策略工厂
│   ├── base.py
│   ├── registry.py
│   └── builtin/                   # 22+ 策略
│
├── backtest/                      # 回测引擎
│   └── engine.py
│
└── execution/                     # 执行层
    ├── base.py
    ├── cex/                       # 12 个交易所
    └── dex/                       # EVM + Solana
```

---

## 七、开发计划

### 总览

```
Phase 0: 基础设施     ─  2 周  ─  建项目结构 + LLM 层 + 基础工具
Phase 1: 单 Agent 基线  ─  3 周  ─  先做最强的单 Agent（证明基线，不堆多 Agent）
Phase 2: 多 Agent 协作  ─  3 周  ─  分析师并行 + 辩论风控（修正版设计）
Phase 3: 执行 + 回测    ─  3 周  ─  交易所适配 + 回测框架 + TiMi 解耦
Phase 4: CVRF 学习      ─  3 周  ─  教训总结 → 记忆库 → 注入
Phase 5: 部署 + 监控    ─  2 周  ─  dryRun 验证 + 实盘配置 + 审计日志
```
> ⚠️ **Phase 0 开始前必须先完成 P0 修正**（见第四章），否则后面的回测结果在方法论上不可信。

---

### Phase 0 — 基础设施（2 周）

**目标**：建好项目架子，让所有代码能跑、能测试、能追踪成本。

#### 0.1 项目结构 + Python 工程化

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| pyproject.toml + poetry/pip 依赖锁定 | `pyproject.toml`, `uv.lock` | `pip install -e .` 一键安装 | P0 |
| ruff + mypy 配置 | `pyproject.toml` [tool.ruff] | `make lint` 0 错误 | P1 |
| pytest 目录 + conftest | `tests/conftest.py` | `make test` 跑通空测试套件 | P1 |
| pre-commit hooks | `.pre-commit-config.yaml` | commit 前自动 lint | P2 |

#### 0.2 LLM 统一调用层

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| Anthropic + OpenAI + DeepSeek 统一客户端 | `llm/client.py` | 一行代码切换 provider，带 Token 计数和成本追踪 | P0 |
| 结构化输出封装 (Pydantic parse) | `llm/structured.py` | 输出不符 schema 自动重试（max_retries=3） | P0 |
| Token 用量 + API 成本落地 | `llm/cost_tracker.py` | 每次调用记录 prompt_tokens / completion_tokens / $cost | P0 |
| 双 LLM 路由（推理用强模型，工具调用用快模型） | `llm/router.py` | 自动把数据分析丢给便宜模型，分析推理丢给强模型 | P1 |

#### 0.3 基础数据模型

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| AgentState TypedDict | `graph/state.py` | 包含 symbol, klines, orderbook, analyses, debate, decision, risk, execution 完整字段 | P0 |
| MarketData Pydantic 模型 | `data/models.py` | OHLCV + OrderBook + FundingRate + OpenInterest，带 `as_of` 时间戳 | P0 |
| 结构化通信协议 Schema（Agent 间 JSON 格式） | `agents/protocol.py` | 分析师报告、研究员辩论稿、交易员指令、风控审核 各一套 JSON Schema | P0 |
| 时间完整性校验器 | `data/time_guard.py` | 禁止任何 timestamp ≥ 决策时刻的数据进入 prompt | P0 |

---

### Phase 1 — 单 Agent 强基线（3 周）

**目标**：建立一个高质量的单 Agent 交易系统，**证明在没有多 Agent 辩论的情况下能跑出什么水平**。这是 Phase 2 多 Agent 的对比基线。如果单 Agent 已经够好，多 Agent 的增量贡献必须通过消融实验证明。

#### 1.1 单 Agent 核心

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| LLM-based 决策 Agent（单角色，综合分析+决策） | `agents/single_agent.py` | 输入 K 线 + 指标 → 输出结构化决策 JSON | P0 |
| Financial CoT 三级推理 prompt | `agents/single_agent.py` (prompt) | Data-CoT → Concept-CoT → Thesis-CoT 三段式 | P1 |
| 工具层：行情查询 | `tools/market_data.py` | `get_klines()`, `get_orderbook()`, `get_funding_rate()` 返回标准 Pydantic | P0 |
| 工具层：回测（调用 fin-agent 引擎） | `tools/backtest.py` | 输入策略参数 → 输出回测报告 | P1 |
| 工具层：持仓/余额 | `tools/portfolio.py` | 查询当前持仓、可用余额、已实现 PnL | P1 |

#### 1.2 数据源适配

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| yfinance 适配器 | `data/sources/yfinance_source.py` | 美股数据获取 + as_of 标记 | P1 |
| AkShare 适配器 | `data/sources/akshare_source.py` | A 股数据获取 + as_of 标记 | P1 |
| Tushare 适配器 | `data/sources/tushare_source.py` | A 股/港股/宏观数据 + as_of 标记 | P0 |

#### 1.3 回测框架（必须含 P0 修正）

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| walk-forward 时间划分引擎 | `backtest/partition.py` | 显式 train/val/test 日历边界，禁止随机抽样 | P0 |
| 交易成本显式建模 | `backtest/cost_model.py` | 佣金 + bid-ask spread + 滑点 + Gas 费（可配置每层开关） | P0 |
| 多档位成本敏感性报告 | `backtest/report.py` | 自动输出 0/10/25 bps 三档 Gross/Net 曲线 | P0 |
| 跨体制分段报告 | `backtest/report.py` | 按牛市/熊市/震荡市分段 Sharpe/Return/MDD | P1 |
| 校准模块（ECE + Platt scaling） | `backtest/calibration.py` | LLM 置信度 → 校准后概率，ECE ≤ 0.05 才允许用于仓位计算 | P0 |

---

### Phase 2 — 多 Agent 协作（3 周）

**目标**：基于修正后的辩论设计（方向锁定 + 风控模式），实现多 Agent 系统。**每一步都做消融实验，对比 Phase 1 的单 Agent 基线**。

#### 2.1 4 个分析师

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| 技术面分析师 prompt | `agents/analysts/technical.py` | 输出结构化技术面报告（趋势/支撑阻力/指标信号/波动率） | P0 |
| 基本面分析师 prompt | `agents/analysts/fundamental.py` | 输出结构化基本面报告（估值/盈利/行业对比） | P1 |
| 市场情绪分析师 prompt | `agents/analysts/sentiment.py` | 输出结构化情绪报告（多空比/资金流向/社交媒体） | P2 |
| 宏观/新闻分析师 prompt | `agents/analysts/news.py` | 输出结构化宏观报告（政策/经济数据/突发事件） | P1 |
| 分析师并行调度 + 超时降级 | `agents/analysts/runner.py` | 4 个分析师同时运行，单分析师超时不影响其他 | P0 |

#### 2.2 LangGraph 图编排

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| 五阶段 StateGraph 定义 | `graph/pipeline.py` | 数据采集 → 分析 → 辩论 → 决策 → 风控 → 执行 | P0 |
| 条件路由：分歧判断 | `graph/routers.py` | δ<0.15 跳过辩论直接采信；δ≥0.15 启动风控审核 | P0 |
| 条件路由：风控否决分支 | `graph/routers.py` | 任一否决 → 取消执行，记录原因 | P0 |
| SQLite checkpoint | `graph/pipeline.py` (checkpointer) | 进程重启可从中断点恢复 | P1 |

#### 2.3 修正版辩论系统

> 根据论文审阅结论：辩论不做方向判断，只做风险降级。

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| 分歧检测器（δ 阈值） | `agents/researchers/disagreement.py` | 计算多 Agent 信号分歧度，触发辩论或跳过 | P0 |
| Trust Agent（证据锚定 + 风险识别） | `agents/researchers/trust.py` | 验证决策是否可以追溯到具体数据 | P1 |
| Skeptic Agent（漏洞识别，方向不改变） | `agents/researchers/skeptic.py` | prompt 明确：不改变方向，只输出风险点 | P1 |
| 辩论 Leader（置信度降级因子） | `agents/researchers/leader.py` | 输出仓位缩减比例，不做方向判断 | P0 |

#### 2.4 三角风控（仍保留，论文证实有效）

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| 激进风控 prompt | `agents/risk.py` (Aggressive) | 偏向机会评估 | P1 |
| 保守风控 prompt | `agents/risk.py` (Conservative) | 偏向风险评估，有一票否决权 | P0 |
| 中立风控 prompt | `agents/risk.py` (Neutral) | 客观风险收益比评估 | P0 |
| CVaR 硬约束（FINCON 公式 + 在线 conformal 增强） | `learning/risk_constraints.py` | 历史最差 5% 平均亏损 ≤ 当前仓位风险敞口 | P1 |

#### 2.5 消融实验

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| 单 Agent 基线 vs 多 Agent 对比框架 | `backtest/ablation.py` | 同一回测窗口，报告单 Agent / 多Agent无辩论 / 多Agent有辩论 三套指标 | P0 |
| 辩论轮次 Token 成本计入净 PnL | `backtest/ablation.py` | 每轮辩论的 LLM 调用数 × Token 单价 = 成本 | P0 |
| 角色相似度/分歧率测量 | `backtest/ablation.py` | 协方差矩阵 + 平均成对分歧度 | P1 |

---

### Phase 3 — 执行层 + 策略工厂（3 周）

**目标**：实现 TiMi 风格解耦——LLM 离线生成策略，纯代码在线执行。

#### 3.1 执行层

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| 交易所基类 + 工厂 | `execution/base.py`, `execution/factory.py` | `create_exchange("binance", config)` → Exchange 实例 | P0 |
| Binance REST + WS | `execution/cex/binance.py` | getPrice / getBalance / placeOrder / cancelOrder / WS stream | P0 |
| OKX REST + WS | `execution/cex/okx.py` | 同上接口 | P1 |
| Bybit REST + WS | `execution/cex/bybit.py` | 同上接口 | P1 |
| 下单校验器（价格校验 + 数量校验 + 风控上限） | `execution/validator.py` | 下单前检查：价格是否合理、数量是否超风控上限、是否 dryRun | P0 |

#### 3.2 策略工厂（Freqtrade 三层抽象）

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| 策略基类 + 注册表 | `strategies/base.py`, `strategies/registry.py` | `@register_strategy("ma_cross") class ...` | P0 |
| 双均线策略 | `strategies/builtin/ma_cross.py` | 从 fin-agent 迁入 | P1 |
| MACD 金叉死叉 | `strategies/builtin/macd.py` | 从 fin-agent 迁入 | P1 |
| RSI 超买超卖 | `strategies/builtin/rsi.py` | 从 fin-agent 迁入 | P1 |
| 布林带突破 | `strategies/builtin/boll.py` | 从 fin-agent 迁入 | P1 |
| 唐奇安通道 | `strategies/builtin/donchian.py` | 从 fin-agent 迁入 | P2 |
| 其余 17+ 策略 | `strategies/builtin/*.py` | 从 fin-agent 逐批迁入 | P2 |

#### 3.3 TiMi 解耦：LLM 策略生成 → 代码执行

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| LLM 策略生成器（从分析/辩论生成可执行策略代码） | `strategies/llm_generator.py` | 输入分析报告 → 输出 Python 策略类代码 + 参数配置 | P1 |
| 策略代码运行时编译器 | `strategies/compiler.py` | 安全 compile + exec，沙箱隔离 | P1 |
| 策略三层结构（Strategy/Function/Parameter Layer） | `strategies/layers.py` | 单向依赖：Strategy → Function → Parameter | P2 |

---

### Phase 4 — CVRF 学习系统（3 周）

**目标**：每笔交易结束后自动总结教训，存入经验库，下次相似市况注入决策 prompt。

#### 4.1 核心循环

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| 交易结果评估器（PnL/MAE/MFE/持仓时长/滑点） | `learning/evaluator.py` | 输入交易记录 → 输出结构化评估 JSON | P0 |
| CVRF 反思 prompt (LLM 总结教训) | `learning/cvrf.py` | "这笔交易学到了什么" 结构化输出（入场/仓位/止损/市况 四维度） | P0 |
| 概念记忆库 SQLite 模型 | `learning/memory.py` | lessons 表（市况向量 + 教训 + 来源 + 权重） | P0 |
| 情景相似度检索（cosine sim on 市况向量） | `learning/memory.py` | 当前市况 → TOP-5 历史教训 | P0 |
| 教训注入模块（插入分析师 prompt 顶部） | `learning/injector.py` | "历史上类似市况曾有 N 次出现了 X 问题" | P0 |

#### 4.2 增强组件

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| CVRF 输出结构化校验器 | `learning/validator.py` | 验证 memory ID 存在、动作类型合法、meta-prompt 反向测试 | P1 |
| 记忆衰减（数据驱动，非固定时间） | `learning/memory.py` | 按市场体制变化触发记忆版本修订 | P2 |
| 体制变化检测器 | `learning/regime_detector.py` | 波动率/相关性结构变化 → 挂起旧经验检索 | P2 |

#### 4.3 回测教训闭环

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| 回测 → 教训总结 → 记忆注入 完整闭环 | `learning/cvrf_pipeline.py` | 回测窗口运行完 → 逐笔总结教训 → 存入记忆 → 下一窗口自动读取 | P1 |

---

### Phase 5 — 部署 + 监控（2 周）

**目标**：让系统在 dryRun 模式下稳定运行，配置化参数，完整审计日志。

#### 5.1 配置 + 入口

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| 统一配置（API Key 加密存储） | `config.py` | TOML/YAML 配置，cryptography 加密密钥 | P0 |
| 主入口 CLI | `main.py` | `qmind --mode backtest --config config.toml` | P0 |
| dryRun 模式开/关 + 模拟成本计算 | `execution/dry_run.py` | dryRun=true 时不实际下单，但计算并报告 Net PnL | P0 |

#### 5.2 审计 + 可复现

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| 全链路审计日志（P0-P6 兼容） | `audit_log.py` | 每个决策记录：时间戳、模型版本、Token 用量、证据链、原始 LLM 响应 | P0 |
| P1-P6 报告生成器 | `backtest/p_report.py` | 自动生成符合 Alpha Illusion P1-P6 标准的回测报告 | P1 |
| 5 类偏差自动检查器 | `audit/bias_checker.py` | 回测完成后自动扫描 look-ahead / survivorship / narrative / objective / cost 偏差 | P1 |

#### 5.3 调度 + 通知

| 任务 | 文件/产出 | 验收标准 | 优先级 |
|------|-----------|----------|:---:|
| 定时轮询调度器 | `scheduler.py` | 从 fin-agent 迁入，支持分钟级/小时级/日级轮询 | P1 |
| 邮件/飞书通知 | `notification.py` | 从 fin-agent 迁入，支持交易信号推送 + 异常告警 | P2 |
| 多品种并行监控 | `scheduler.py` (multi-symbol) | 同时监控 N 个交易对，各自独立决策 | P2 |

---

### 依赖图

```
Phase 0 ──────────────────────────────────────────────────┐
  ├─ 0.1 pyproject.toml                                    │
  ├─ 0.2 LLM client  ← (Phase 1,2,4 都依赖) ──────┐       │
  ├─ 0.3 data models ← (所有 Phase 依赖) ─────────┐│       │
  │                                                 ▼▼      │
Phase 1 ── 单 Agent 基线 ──────────────────────────────┐   │
  ├─ 1.1 单 Agent 决策 ← 依赖 0.2, 0.3                  │   │
  ├─ 1.2 数据源 ← 依赖 0.3                              │   │
  ├─ 1.3 回测框架（P0 修正实现）─────────────────┐      │   │
  │         （walk-forward + cost + calibration）  │      │   │
  │                                                 │      │   │
  ▼─────────────────────────────────────────────────▼      ▼   │
Phase 2 ── 多 Agent ── 依赖 1.3 回测框架做消融             │   │
  ├─ 2.1 4 分析师 ← 依赖 0.2 LLM client                    │   │
  ├─ 2.2 LangGraph 图 ← 依赖 0.3 state                      │   │
  ├─ 2.3 修正辩论系统                                       │   │
  ├─ 2.4 三角风控                                           │   │
  ├─ 2.5 消融实验 ← 依赖 1.3 回测框架                       │   │
  │                                                         │   │
Phase 3 ── 执行 + 策略 ← 可并行 Phase 1/2                  │   │
  ├─ 3.1 交易所适配 ← 依赖 0.3 data models                  │   │
  ├─ 3.2 策略工厂 ← 依赖 fin-agent 迁入                     │   │
  ├─ 3.3 TiMi 解耦                                          │   │
  │                                                         │   │
Phase 4 ── CVRF 学习 ← 依赖 Phase 1/2 跑出交易记录         │   │
  ├─ 4.1 核心循环 ← 依赖 0.2 LLM client                     ▼   │
  └─ 4.3 闭环验证 ← 依赖 1.3 回测框架 ─────────────────────┘   │
                                                                │
Phase 5 ── 部署 ← 依赖所有 Phase                             ──┘
```

---

### Todo 表总览

```
Phase 0: 基础设施 (2 周)
  [ ] 0.1  Python 工程化：pyproject.toml / ruff / pytest / pre-commit
  [ ] 0.2  LLM 统一调用层：client / structured / cost_tracker / router
  [ ] 0.3  基础数据模型：state / MarketData / protocol schemas / time_guard

Phase 1: 单 Agent 强基线 (3 周)
  [ ] 1.1  LLM 决策 Agent + Financial CoT prompt
  [ ] 1.2  数据源适配：yfinance / AkShare / Tushare（含 as_of 标记）
  [ ] 1.3  回测框架：walk-forward partition / cost_model / calibration
  [ ] 1.4  工具层：market_data / backtest / portfolio

Phase 2: 多 Agent 协作 (3 周)
  [ ] 2.1  4 分析师 prompt + 并行调度
  [ ] 2.2  LangGraph 五阶段图 + 条件路由
  [ ] 2.3  修正版辩论系统：分歧检测 / Trust / Skeptic / Leader
  [ ] 2.4  三角风控 + CVaR 硬约束
  [ ] 2.5  消融实验：单 Agent vs 多 Agent vs 辩论 / 成本追踪 / 相似度

Phase 3: 执行层 + 策略 (3 周)
  [ ] 3.1  交易所适配：Binance / OKX / Bybit + 下单校验器
  [ ] 3.2  策略工厂：基类 + 注册表 + 内置 22 策略迁入
  [ ] 3.3  TiMi 解耦：LLM 策略生成器 → 代码编译器 → 三层结构

Phase 4: CVRF 学习 (3 周)
  [ ] 4.1  交易评估 + CVRF 反思 prompt
  [ ] 4.2  概念记忆库 SQLite + 相似度检索 + 教训注入
  [ ] 4.3  回测 → 教训 → 记忆 → 下一窗口 闭环

Phase 5: 部署 + 监控 (2 周)
  [ ] 5.1  配置管理 + CLI 入口 + dryRun 模式
  [ ] 5.2  全链路审计日志 + P1-P6 报告 + 偏差检查器
  [ ] 5.3  定时调度 + 邮件/飞书通知 + 多品种并行
```

### 交付物检查清单

- [ ] **Phase 0 结束时**：`pip install -e . && make test && make lint` 一键跑通，LLM 调用成本可追踪
- [ ] **Phase 1 结束时**：单 Agent 在回测窗口输出可复现的 Net PnL，ECE ≤ 0.05
- [ ] **Phase 2 结束时**：消融实验报告证明多 Agent 的净增量贡献（或证明无增量但有风控价值）
- [ ] **Phase 3 结束时**：dryRun 模式下完整跑通 分析 → 辩论 → 决策 → 风控 → 模拟执行
- [ ] **Phase 4 结束时**：经验库有 > 100 条教训条目，回测窗口 2 比窗口 1 有可检测的绩效提升
- [ ] **Phase 5 结束时**：系统能以 `qmind --mode live --dry-run` 连续运行 7 天不中断

---

## 八、附录：论文清单

所有论文已下载到 `references/papers/`，共 **18 篇**，总计 56 MB。

### A. 核心借鉴论文（5 篇）

| # | 论文 | arXiv | 发表 | 大小 | 核心贡献 | 优先级 |
|---|------|-------|------|------|----------|:---:|
| 1 | **TradingAgents** | [2412.20138](https://arxiv.org/abs/2412.20138) | 2024.12 · UCLA/MIT | 1.9 MB | LangGraph 多智能体交易框架 + 7 角色 + 辩论机制 | ⭐⭐⭐⭐⭐ |
| 2 | **FINCON** | [2407.06567](https://arxiv.org/abs/2407.06567) | NeurIPS 2024 | 14 MB | CVRF 概念言语强化学习 + CVaR 风险约束 | ⭐⭐⭐⭐⭐ |
| 3 | **FinRobot (Platform)** | [2405.14767](https://arxiv.org/abs/2405.14767) | 2024.05 · AI4Finance | 5.2 MB | 四层 Agent 平台架构白皮书 | ⭐⭐⭐⭐ |
| 4 | **FinRobot (CoT)** | [2411.08804](https://arxiv.org/abs/2411.08804) | ICAIF 2024 | 2.3 MB | Financial CoT 三级推理链 (Data→Concept→Thesis) | ⭐⭐⭐⭐ |
| 5 | **FinGPT** | [2307.10485](https://arxiv.org/abs/2307.10485) | 2023.07 · Columbia | 719 KB | 金融 LLM + LoRA 微调 + RLSP | ⭐⭐⭐ |

### B. 综述/评估论文（7 篇）

| # | 论文 | arXiv | 发表 | 大小 | 核心贡献 | 优先级 |
|---|------|-------|------|------|----------|:---:|
| 6 | **LLM Agent in Financial Trading: A Survey** | [2408.06361](https://arxiv.org/abs/2408.06361) | 2024.07 → 2026.03 v2 | 676 KB | LLM Agent 交易最全面综述，含架构/数据/回测对比 | ⭐⭐⭐⭐⭐ |
| 7 | **Agentic Trading: When LLM Agents Meet Financial Markets** | [2605.19337](https://arxiv.org/abs/2605.19337) | 2026.05 | 11 MB | 77 项研究可复现性审计，仅 2/19 报告时间一致性 | ⭐⭐⭐⭐⭐ |
| 8 | **The New Quant: LLMs in Financial Prediction and Trading** | [2510.05533](https://arxiv.org/abs/2510.05533) | 2025.10 | 524 KB | 50+ 研究任务分类法，含 agentic systems 专项 | ⭐⭐⭐⭐ |
| 9 | **The Alpha Illusion** | [2605.16895](https://arxiv.org/abs/2605.16895) | 2026.05 | 1.5 MB | ⚠️ LLM Agent 回测 alpha ≠ 可部署证据，提出 P1-P6 报告协议 | ⭐⭐⭐⭐⭐ |
| 10 | **Toward Reliable Evaluation of LLM Financial MAS** | [2603.27539](https://arxiv.org/abs/2603.27539) | 2026.03 | 473 KB | 12 系统评估，协调优先假说 (CPH)，5 类评估失败 | ⭐⭐⭐⭐⭐ |
| 11 | **Beyond Agent Architecture** | [2606.08285](https://arxiv.org/abs/2606.08285) | 2026.06 | 383 KB | 30 项交易研究的执行假设与可复现性审计 | ⭐⭐⭐⭐ |
| 12 | **LLM Agents Do Not Replicate Human Market Traders** | [2502.15800](https://arxiv.org/abs/2502.15800) | 2025.02 | 5.0 MB | 实验金融证据：LLM Agent ≠ 人类交易员 | ⭐⭐⭐ |

### C. 多智能体架构相关（4 篇）

| # | 论文 | arXiv | 发表 | 大小 | 核心贡献 | 优先级 |
|---|------|-------|------|------|----------|:---:|
| 13 | **TiMi (Trade in Minutes)** | [2510.04787](https://arxiv.org/abs/2510.04787) | ICLR 2026 | 3.8 MB | ⭐ 策略开发与分钟级部署解耦，200+ 交易对验证 | ⭐⭐⭐⭐⭐ |
| 14 | **FinDebate** | [2509.17395](https://arxiv.org/abs/2509.17395) | EMNLP 2025 | 894 KB | 5 Agent 辩论 + 安全辩论协议 + RAG | ⭐⭐⭐⭐ |
| 15 | **Macro Economists in the Machine** | [2606.08283](https://arxiv.org/abs/2606.08283) | 2026.06 | 1.9 MB | 鹰/鸽/辩论 Agent → 辩论 = 偏差校正而非 alpha | ⭐⭐⭐⭐ |
| 16 | **Evaluating LLMs in Finance Requires Explicit Bias Consideration** | [2602.14233](https://arxiv.org/abs/2602.14233) | ICML 2026 | — | ⚠️ 164 篇论文审查，5 类偏差清单 + 结构效度框架 | ⭐⭐⭐⭐⭐ |

### D. 组合优化/风控/方法（4 篇）

| # | 论文 | arXiv | 发表 | 大小 | 核心贡献 | 优先级 |
|---|------|-------|------|------|----------|:---:|
| 17 | **Signature-Informed Transformer for Asset Allocation** | [2510.03129](https://arxiv.org/abs/2510.03129) | ICML 2026 | 933 KB | Path Signature + Attention + CVaR 端到端组合优化 | ⭐⭐⭐⭐ |
| 18 | **Adversarially Robust Control of CVaR via Kelly Conformal Inference** | [2606.00320](https://arxiv.org/abs/2606.00320) | ICML 2026 | 1.0 MB | 在线 distribution-free CVaR 控制 + adversarial regret 保证 | ⭐⭐⭐⭐ |

### E. ICML 2026 待收录论文（暂未找到 arXiv 版）

| # | 论文 | ICML 链接 | 与 QMind 的关系 |
|---|------|-----------|-----------------|
| E1 | JEPA Latent Market States in U.S. Equities | [poster/65643](https://icml.cc/virtual/2026/poster/65643) | 无监督市场状态表征学习 → 可用于 CVRF 情景识别 |
| E2 | Decision-focused Sparse Tangent Portfolio Optimization | [poster/64722](https://icml.cc/virtual/2026/poster/64722) | 稀疏组合优化 → 可改进执行层的资产选择 |
| E3 | Global Merger-Arbitrage Forecasting with LLMs | [poster/60833](https://icml.cc/virtual/2026/poster/60833) | 并购套利预测 → 可用于新闻/宏观分析师 |
| E4 | Error Propagation in Dynamic Programming | [poster/65273](https://icml.cc/virtual/2026/poster/65273) | DP 误差传播 → CVRF 学习循环的理论基础 |
| E5 | Learning The ESG Geometry with Domain Aware LMs | [poster/65502](https://icml.cc/virtual/2026/poster/65502) | ESG 表征学习 → 未来多模态市场分析 |
| E6 | XAI Methods Cannot Satisfy Financial AI Explainability | [poster/67240](https://icml.cc/virtual/2026/poster/67240) | 监管合规 → 风控模块的可解释性要求 |
| E7 | Online Conformal Prediction Via Universal Portfolio Algorithms | [poster/65365](https://icml.cc/virtual/2026/poster/65365) | 在线 conformal prediction → 实时风控 |

> 待 arXiv 版发布后补充下载。

### 阅读建议（按 QMind 构建顺序）

1. **先读综述**（#6、#7、#9、#10）— 建立全局视野，避开已知坑
2. **再读骨架**（#1）— 理解 TradingAgents 五阶段图结构
3. **再读大脑**（#2）— 理解 CVRF 学习循环
4. **再读竞品**（#13、#14、#15）— 知道别人怎么做多 Agent 交易
5. **再读坑**（#16、#9、#10）— 5 类偏差、评估陷阱、可复现性问题
6. **最后读方法**（#17、#18 + E 系列）— 按需查阅具体技术

### BibTeX（新增）

```bibtex
@article{song2025timi,
  title   = {Trade in Minutes! Rationality-Driven Agentic System
             for Quantitative Financial Trading},
  author  = {Song, Zifan and Song, Kaitao and Hu, Guosheng and others},
  journal = {arXiv preprint arXiv:2510.04787},
  year    = {2025},
  note    = {Accepted at ICLR 2026}
}

@article{ding2024llmsurvey,
  title   = {Large Language Model Agent in Financial Trading: A Survey},
  author  = {Ding, Han and Li, Yinheng and Wang, Junhao and others},
  journal = {arXiv preprint arXiv:2408.06361},
  year    = {2024}
}

@article{xia2026agentic,
  title   = {Agentic Trading: When LLM Agents Meet Financial Markets},
  author  = {Xia, Yihan and others},
  journal = {arXiv preprint arXiv:2605.19337},
  year    = {2026}
}

@article{ye2026alpha,
  title   = {The Alpha Illusion: Reported Alpha from LLM Trading Agents
             Should Not Be Treated as Deployment Evidence},
  author  = {Ye, Yuxuan and Xu, Zenglin and others},
  journal = {arXiv preprint arXiv:2605.16895},
  year    = {2026}
}

@article{kong2026bias,
  title   = {Position: Evaluating LLMs in Finance Requires
             Explicit Bias Consideration},
  author  = {Kong, Yaxuan and Lee, Hoyoung and Hwang, Yoontae and others},
  journal = {arXiv preprint arXiv:2602.14233},
  year    = {2026},
  note    = {Accepted at ICML 2026}
}

@article{hwang2025signature,
  title   = {Signature-Informed Transformer for Asset Allocation},
  author  = {Hwang, Yoontae and Zohren, Stefan},
  journal = {arXiv preprint arXiv:2510.03129},
  year    = {2025},
  note    = {Accepted at ICML 2026}
}

@article{chen2026cvar,
  title   = {Adversarially Robust Control of Conditional Value-at-Risk
             via Kelly Conformal Inference},
  author  = {Chen, Catherine and Shen, Jingyan and Yang, Xinyu and Lei, Lihua},
  journal = {arXiv preprint arXiv:2606.00320},
  year    = {2026},
  note    = {Accepted at ICML 2026}
}

@article{nguyen2026reliable,
  title   = {Toward Reliable Evaluation of LLM-Based Financial
             Multi-Agent Systems},
  author  = {Nguyen, Phat and Pham, Thang},
  journal = {arXiv preprint arXiv:2603.27539},
  year    = {2026}
}

@article{cai2025findebate,
  title   = {FinDebate: Multi-Agent Collaborative Intelligence for
             Financial Analysis},
  author  = {Cai, Tianshi and others},
  journal = {arXiv preprint arXiv:2509.17395},
  year    = {2025},
  note    = {Accepted at FinNLP@EMNLP 2025}
}
```