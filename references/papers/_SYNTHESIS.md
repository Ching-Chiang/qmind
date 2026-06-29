# QMind 架构必须修改的清单

基于 18 篇论文精读的综合分析。

---

## 一、致命问题（必须立刻修改，否则系统不可用）

### 致命问题 1：无时间完整性控制，存在严重前视偏差风险

- **问题**: QMind 当前没有任何 Point-in-Time 控制机制。LLM 的知识截止日期未与回测窗口对齐，RAG/经验库检索未做时间戳过滤，增量分析 (keep_analysis) 未校验上下文是否包含未来信息。

- **来源论文**: Beyond Agent Architecture (2606.08285), Five Sins Survey, Agentic Trading Survey (2605.19337), Reliable Evaluation MAS (2603.27539)

- **QMind 当前设计的问题**:
  - PA_Agent 的增量分析 (`keep_analysis`) 自动将上次分析结论作为新分析的上下文，但没有检查上次结论中引用的数据时间戳是否越界
  - 经验库按周期位置检索历史案例，但没有强制 `timestamp < 当前决策时刻` 的过滤
  - fin-agent 的 Tushare 数据拉取不记录 `as_of` 时间戳
  - 所有 LLM 调用的 prompt 中未见知识截止日期约束

- **修改方案**:
  1. 在数据管道层增加 `as_of` 时间戳字段，所有数据源统一标记可用时间
  2. 在 Prompt 构建器中增加 PiT 校验钩子：任何输入数据的时间戳必须严格早于当前决策时刻
  3. 经验库检索增加时间过滤器：`WHERE timestamp < decision_time`
  4. 强制声明所用 LLM 的模型版本和知识截止日期，回测窗口必须包含截止后区间
  5. 增量分析 (keep_analysis) 的上下文复用增加时间边界检查

---

### 致命问题 2：回测框架缺乏时间一致性划分，等同于用未来信息调参

- **问题**: 当前项目的回测（tsla_spread_backtest.py、freqtrade 回测）未显式声明 train/val/test 的日历时间边界，未使用 walk-forward 或 purged k-fold 划分，无法排除超参数泄露。Agentic Trading Survey 发现 90% 的 LLM 交易研究缺失此项。

- **来源论文**: Agentic Trading Survey (2605.19337, MR-2), Beyond Agent Architecture (2606.08285, P1), Reliable Evaluation MAS

- **QMind 当前设计的问题**:
  - `tsla_spread_backtest.py` 未声明时间划分协议
  - freqtrade 回测使用默认超参搜索，无密封测试集
  - fin-agent 的 22+ 策略回测无 walk-forward 验证
  - 没有 "单次使用的密封测试集" (single-use holdout set) 概念

- **修改方案**:
  1. 所有回测强制使用 walk-forward 或 purged k-fold 时间划分，随机抽样在时序金融数据中禁用
  2. 显式声明 train/val/test 的日历日期边界，写入配置文件
  3. 超参数搜索和模型选择必须使用独立的 validation set，保留一个从未被查看的 test set 仅在最末评估一次
  4. 报告超参数搜索预算（跑了几组实验、选优标准）

---

### 致命问题 3：交易成本完全未建模，所有绩效指标均为不可部署的毛指标

- **问题**: 当前项目没有任何组件显式建模交易成本和 LLM 推理成本。套利策略的 spread 计算仅含价差，不含手续费、滑点、市场冲击、Gas 费。`tsla_spread_backtest.py` 未扣除任何摩擦。PA_Agent 不连接券商，但如果未来要评估策略绩效，必须建模成本。

- **来源论文**: Beyond Agent Architecture (2606.08285, P5 — 30 项研究中仅 14 项有成本处理), Reliable Evaluation MAS, Five Sins Survey (Cost Bias), SIT (外生 0/5/10 bps 敏感性)

- **QMind 当前设计的问题**:
  - Room 217 套利引擎的 spread 计算不含手续费（Binance 0.1%、Hyperliquid 0.024% 等）
  - 无滑点建模（大单的市场冲击）
  - 加密货币特有的 Gas 费（DEX 交互）未计入
  - 无 LLM Token 成本追踪（PA_Agent 每次分析的实际费用）
  - `dryRun=true` 模式下连模拟成本都不计算

- **修改方案**:
  1. 在套利引擎中增加显式成本层：
     - 手续费（各交易所不同费率配置）
     - 买卖价差（bid-ask spread）
     - 线性滑点（按成交量比例）
     - Gas 费（ETH/Solana 链上交互估算）
  2. 在 PA_Agent 中增加 Token 用量成本追踪，存入分析落盘
  3. **任何绩效数字必须同时报告 Gross 和 Net 两条曲线**——两者差距即为 alpha 幻觉的量化度量
  4. 至少报告 3 档成本假设下的敏感性（0 bps / 10 bps / 25 bps 或加密货币场景 5 bps / 15 bps / 30 bps）

---

### 致命问题 4：LLM 输出的 "置信度" 直接用于决策，但完全未校准

- **问题**: PA_Agent 两阶段分析输出的"置信度"字段来自 LLM 自报，没有任何独立校准。五罪调查论文明确指出：LLM 的"语言置信度"与实际胜率之间无可靠映射关系。这是 P4 (Epistemic Calibration) 的核心失败。

- **来源论文**: Beyond Agent Architecture (2606.08285, P4), Five Sins Survey (Objective Bias), FinDebate (硬编码 70-80%), MacroAgent (置信度仅用于诊断)

- **QMind 当前设计的问题**:
  - PA_Agent 分析结果中的"置信度"直接来自 LLM 自然语言输出
  - 没有历史准确率追踪来校准
  - 如果未来将此置信度用于仓位控制，将产生系统性错误

- **修改方案**:
  1. **绝对禁止用 LLM 自报置信度直接做仓位大小输入**
  2. 建立独立的校准模块：
     - 记录每次预测的"语言置信度"和"实际结果"
     - 计算 ECE (Expected Calibration Error)
     - 用 Platt scaling 或贝叶斯更新将 LLM 置信度映射为校准后概率
  3. 仓位控制逻辑必须独立于 LLM，基于校准后概率和风险预算计算

---

### 致命问题 5：无生存者偏差控制

- **问题**: 164 篇论文中仅 2 篇提及生存者偏差（1.2%），是五罪中最被忽视的一项。QMind 同样没有处理——回测股票池如果是用事后成分股列表，隐含严重的 hindsight bias。

- **来源论文**: Five Sins Survey (Survivorship Bias), Reliable Evaluation MAS (Failure 2), Beyond Agent Architecture

- **QMind 当前设计的问题**:
  - fin-agent 的选股器从 Tushare 拉取当前股票列表做回测，未使用历史时点成分股
  - 退市/ST/停牌股票被无声排除
  - 回测收益被系统性高估（据 Elton et al. 估计仅共同基金就有 0.9% 年化幸存者偏差）

- **修改方案**:
  1. 为每个历史时间点 t 构建动态可交易宇宙 U_t（包含当时上市但后续退市的实体）
  2. 回测必须从 U_t 采样，而非从期末静止股票池采样
  3. 分别报告幸存实体和非幸存实体的绩效指标
  4. Tushare 数据拉取记录成分股变更历史

---

## 二、重要修正（影响性能/可靠性，应该修改）

### 重要修正 1：执行时机语义必须精确化

- **问题**: 当前系统的信号产生时间与假设执行时间之间的关系不明确。回测中同一信号用 "同收盘" 还是 "次开盘" 成交，结果差异巨大。

- **来源论文**: Beyond Agent Architecture (2606.08285), Agentic Trading Survey (MR-3, MR-4)

- **QMind 当前设计的问题**:
  - Room 217 的套利信号是实时事件驱动，但回测 (`tsla_spread_backtest.py`) 未声明执行时机假设
  - 无执行延迟建模

- **修改方案**:
  1. 显式采用 next-close（最保守假设）：信号形成于 t 收盘后，以 t+1 收盘价成交
  2. 或至少采用 next-open：信号形成于 close，以次日开盘价成交
  3. 禁止 same-close 假设（隐含乐观偏差）
  4. 记录决策时间戳与执行时间戳

---

### 重要修正 2：策略同质化风险管理（多 Agent 共享同一 LLM 的系统性风险）

- **问题**: LLM Agent 的行为实验证明，同模型 Agent 策略高度收敛（最终组合价值区间 668-690 极窄），且模型越强越不像人。如果 QMind 所有 Agent 使用同一 LLM（如全部用 DeepSeek），会产生系统性相关风险——所有人同时做同样的事。

- **来源论文**: LLM Agents Don't Replicate Humans (2502.15800), Reliable Evaluation MAS (Coordination Primacy Hypothesis)

- **QMind 当前设计的问题**:
  - PA_Agent 目前使用单一 LLM 配置（API Base URL / Model / Key 全局统一）
  - fin-agent 支持多模型但实际运行中同一会话仅用单一模型

- **修改方案**:
  1. 在多 Agent 场景下混合多个异构模型（DeepSeek + Kimi + Qwen）
  2. 对同一模型的不同 Agent 注入差异化参数（不同 temperature、不同 system prompt 变体、不同信息集）
  3. 增加策略方差监测指标——如果多个 Agent 输出高度一致（如 >90% 相同方向），发出 "回声室警告"

---

### 重要修正 3：辩论/多 Agent 协调的实际收益必须通过消融实验验证

- **问题**: 多篇论文指出多 Agent 辩论的增量贡献可疑。MacroAgent 发现 Debate Shark = avg(Hawkish, Dovish) + 0.001，即辩论无独立信息增量。Zhang et al. 报告 36 组实验中辩论胜率不到 20%。在建立单 Agent 基线之前堆叠 Agent 数量，可能只是在更好地过拟合。

- **来源论文**: MacroAgent (Debate δSharpe=+0.001), Beyond Agent Architecture (P6 — 多 Agent 解聚), Reliable Evaluation MAS (CPH), Five Sins Survey

- **QMind 当前设计的问题**:
  - 如果 QMind 设计多 Agent 辩论（Bull vs Bear + Facilitator），没有单 Agent 基线对比
  - 没有消融实验设计来区分各组件的边际贡献

- **修改方案**:
  1. **在引入任何多 Agent 架构之前，先建立强单 Agent 基线**
  2. 设计消融实验矩阵：单 Agent vs 多 Agent 无辩论 vs 多 Agent 有辩论
  3. 报告辩论轮次的 Token 成本，计入净 PnL
  4. 测量角色相似度和分歧率——如果两个 "分析师" 90% 情况说同样的话，它们不是独立专家

---

### 重要修正 4：回测窗口必须跨越多市场体制

- **问题**: TradingAgents 的 Sharpe 5.60-8.21 是仅 3 个月牛市的结果。如果 QMind 的回测只覆盖单一市场体制（如 2024 年 AI 牛市），得出的 Sharpe 同样不可信。

- **来源论文**: Reliable Evaluation MAS (Failure 5 — Regime-Shift Blindness), TradingAgents 精读, Beyond Agent Architecture

- **QMind 当前设计的问题**:
  - `tsla_spread_backtest.py` 未声明回测窗口的体制覆盖
  - fin-agent 回测未分段报告不同市场状态的绩效

- **修改方案**:
  1. 回测窗口至少覆盖 3-5 年，必须包含牛市、熊市、震荡市三种体制
  2. 按体制分段报告绩效（如 "2022 熊市 Sharpe X, 2023 震荡 Sharpe Y, 2024 牛市 Sharpe Z"）
  3. 报告多窗口滚动评估的均值和方差，而非单窗口点估计

---

### 重要修正 5：Prompt/协议必须结构化并可审计

- **问题**: TradingAgents 最大的透明度问题是 prompt 模板未公开。Five Sins Survey 的结构效度框架要求每个决策的理由可追溯到特定数据来源。

- **来源论文**: TradingAgents 精读, Five Sins Survey (Rationale Robustness), Reliable Evaluation MAS

- **QMind 当前设计的问题**:
  - PA_Agent 已有部分落盘（Prompt/响应/JSON/Token 用量/追问记录），但缺乏 "证据锚定" 层——每个结论绑定到特定数据点
  - fin-agent 的 LLM 调用未完整落盘

- **修改方案**:
  1. 在 PA_Agent 落盘中增加 "证据锚定" 字段：每条分析结论必须引用具体的数据行/时间戳/来源
  2. 公开所有 prompt 模板和通信协议 schema（JSON Schema 定义）
  3. 增加事实一致性审计器：自动检测 LLM 输出中引用的不存在事件、错误数值

---

### 重要修正 6：CVRF 的风险组件 M_r 本身是 LLM，缺乏校验

- **问题**: FINCON 的 CVRF 循环每一步（概念化总结、meta-prompt 生成、prompt 更新）都依赖 LLM 调用。这意味成本线性增长 + 可能链式放大幻觉。

- **来源论文**: FINCON 精读

- **QMind 当前设计的问题**:
  - 如果照搬 FINCON 的 CVRF 循环，M_r 的输出没有结构化校验
  - 概念化总结可能产生幻觉（不存在的 memory ID、错误的动作分类）

- **修改方案**:
  1. 对 M_r 的输出增加结构化校验层——验证生成的 memory ID 是否存在、动作类型是否在合法集合内
  2. 对 meta-prompt 做 "反向测试"——应用新 prompt 后的 Agent 行为是否与 meta-prompt 描述的方向一致
  3. 考虑用规则引擎辅助关键风险决策，而非完全依赖 LLM

---

## 三、辩论机制重新设计

### 辩论的真正作用：偏差校正，而非 alpha 生成

这是本批论文中最重要的发现，来自多个独立证据源的收敛：

1. **MacroAgent 的直接证据**：Debate Agent 的 Sharpe 几乎精确等于 Hawkish 和 Dovish 两个单 Agent Sharpe 的算术平均值（δSharpe=+0.001），且 Debate 没有击败更强的单 Agent（p=0.769）。

2. **Zhang et al. (2025) 的系统性证据**：36 组实验（4 模型 x 9 基准）中，多 Agent 辩论胜率不到 20%，且增轮或增人无益甚至有害。

3. **Beyond Agent Architecture (P6)**：多 Agent 共识不等于独立专家汇聚——同源模型产生关联错误，回声室效应是真实的。

4. **FinDebate 的安全协议设计**（反向验证）：正是因为辩论容易退化，FinDebate 才设计了方向锁定 + 单轮辩论 + 回退机制的全套防退化协议。

**结论**：辩论阶段的收益完全来自 "平均掉错误 prior 的偏差"，而非辩论本身产生新的独立信息。辩论是偏差保险，不是 alpha 引擎。

---

### QMind 辩论阶段重新设计

**当前假设**（据 CLAUDE.md 中 "多视角辩论: 多方/空方辩论机制可以显著减少 confirmation bias"）：辩论 = 提升决策质量。

**修正后设计**：

```
┌─────────────────────────────────────────────────────────────┐
│                 QMind 修正后辩论架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                │
│  │ Agent A   │  │ Agent B   │  │ Agent C   │  多 Agent 独立  │
│  │ (强模型)  │  │ (强模型)  │  │ (异构模型)│  生成信号        │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘                │
│        │               │               │                      │
│        ▼               ▼               ▼                      │
│  ┌─────────────────────────────────────────┐                  │
│  │         分歧检测器 (δ 阈值)              │                  │
│  │  δ = mean(|s_A - s_B|, |s_A - s_C|, ...)│                  │
│  └────────────────┬────────────────────────┘                  │
│                   │                                           │
│         ┌─────────┴──────────┐                                │
│         ▼                    ▼                                │
│    δ < 0.15               δ ≥ 0.15                            │
│    (低分歧)               (高分歧)                              │
│         │                    │                                 │
│         ▼                    ▼                                │
│  ┌─────────────┐    ┌──────────────────┐                      │
│  │ 直接采信    │    │ 启动风控审核流程  │                      │
│  │ 最强Agent   │    │ (非方向辩论!)     │                      │
│  │ 的信号      │    │                  │                      │
│  │             │    │ Trust Agent:     │                      │
│  │ 不做辩论    │    │ 强化证据锚定     │                      │
│  │ (避免稀释)  │    │                  │                      │
│  │             │    │ Skeptic Agent:   │                      │
│  │             │    │ 识别风险漏洞     │                      │
│  │             │    │ (不改变方向!)    │                      │
│  │             │    │                  │                      │
│  │             │    │ Leader:          │                      │
│  │             │    │ 输出置信度降级    │                      │
│  │             │    │ 因子 + 仓位缩减   │                      │
│  └─────────────┘    └──────────────────┘                      │
│                                                               │
│  核心原则：                                                    │
│  1. 辩论不改变方向判断，只做风险降级                            │
│  2. 无分歧时不辩论（避免稀释最优信号）                          │
│  3. 有分歧时辩论只输出 "置信度降级因子" → 仓位缩减              │
│  4. 方向判断由最强单 Agent 或动态加权给出                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 是否保留多方/空方辩论？

**保留，但降级为 "风控审核器" 而非 "方向决策器"**。

改进清单：

1. **方向锁定**（借鉴 FinDebate）：辩论 Agent 的 prompt 中反复强调 "CRITICAL: DO NOT change the directional signal (Long/Short/Neutral)。Your role is to refine risk assessment and evidence anchoring, not to override the trading direction."

2. **单轮辩论**（借鉴 FinDebate 的 ablation 结论）：多轮辩论导致主题漂移 (thematic drift)，max_round=1。

3. **回退机制**：如果辩论后核心结论被破坏（方向完全反转），直接返回原始方案。

4. **辩论成本计入净 PnL**（借鉴 P6）：每一轮辩论的 Token 成本必须追踪和报告。

5. **竞争评估优于共识机制**（借鉴 Reliable Evaluation MAS）：不采用简单多数投票，奖励 "逆向准确度"——当 Agent 的独立分析与共识不一致但最终被证明正确时给予更高权重。

6. **动态先验权重**（借鉴 MacroAgent 的未来方向）：用一个元模型 (regime detector) 判断当前市场状态的先验偏向，给更适配当前体制的 Agent 更高权重。这比等权平均更有理论基础。

---

## 四、评估/回测框架要求

### QMind 自己的回测应遵守的标准 (P1-P6)

基于 Beyond Agent Architecture 和 Agentic Trading Survey 的 MR-1 至 MR-7：

| 协议 | 名称 | QMind 必须实现 |
|------|------|---------------|
| **P1/MR-2** | 时间完整性 | 模型版本+知识截止日期披露；回测窗口必须包含截止后区间；RAG 语料带时间戳；walk-forward 划分 |
| **P2/MR-1** | 动态宇宙 | 时变可交易股票池；退市/停牌处理；流动性过滤；成分股变更记录 |
| **P3** | 反事实稳健性 | 输入强反向证据（如看跌新闻），测量方向翻转率、置信度变化、仓位变化 |
| **P4** | 校准 | ECE 期望校准误差；可靠性曲线；LLM 置信度绝不直接用于仓位控制 |
| **P5/MR-4** | 现实执行 | 分层毛→净净化（佣金+价差+滑点+市场冲击+Token 成本+Gas 费）；多档位敏感性 |
| **P6/MR-7** | 多 Agent 解聚 | 单 Agent 基线；角色相似度；分歧率；辩论轮次成本；多 Agent 净回报增量 |
| **MR-3** | 动作语义 | 决策时间戳；价格形成假设（next-close）；仓位约束 |
| **MR-5** | 泄漏审计 | 显式检查的泄漏向量；新闻发布时间 vs 决策时间对齐；多重检验校正 |
| **MR-6** | 可复现工件 | 代码仓库；固定环境文件；数据获取指引；评估脚本 |

### dryRun 模式应报告的指标

当前 `dryRun=true` 仅阻止实际下单，但不计算模拟绩效。应强制报告：

```
每笔模拟交易：
  - 信号产生时间戳
  - 假设执行时间戳和执行价格假设
  - 毛收益 (gross PnL)
  - 手续费 (commission)
  - 买卖价差 (spread cost)
  - 滑点估算 (slippage)
  - Gas 费 (如涉及 DEX)
  - Token API 费用
  - 净收益 (net PnL)

汇总统计：
  - 累计净收益率 (Net CR)
  - 年化净收益率
  - 净 Sharpe Ratio (同时报告 Gross 和 Net)
  - 最大回撤 MDD
  - 胜率 (Win Rate)
  - 盈亏比 (Profit Factor)
  - 换手率 (Turnover)
  - 总 Token 成本和 API 费用
  - 按市场体制分段的子报告
  - ECE 校准误差
```

---

## 五、CVRF 学习循环的改进

基于新论文发现，FINCON 的 CVRF 循环应从以下几个方面增强：

### 1. 概念化维度的定制化（来源：FINCON 精读）

FINCON 的概念维度（momentum, news, filing, ECC audio）按其 7 类 Analyst 定制。QMind 需要重定义自己的概念维度：

| QMind 数据源 | 对应概念维度 |
|-------------|-------------|
| K 线 OHLCV | 趋势动量和均值回归、波动率体制 |
| 技术指标 (MACD/RSI/布林带) | 超买超卖信号、背离度 |
| 财务数据 (Tushare) | 估值因子、盈利质量 |
| 新闻/社交媒体 | 情绪极性和叙事主题 |
| 资金流向 | 主力/散户行为分歧 |
| 链上数据 (加密货币) | 大额转账、交易所余额 |

### 2. CVaR 的在线自适应增强（来源：RUCC Conformal CVaR 论文）

FINCON 使用固定 1% 分位数 CVaR。RUCC 提供了更强的替代方案：

- 用 Rockafellar-Uryasev 变分表示将 CVaR 控制归约为两层在线优化
- **不依赖任何数据分布假设**（无 stationarity、no i.i.d.），适合市场体制切换
- 参数通过 AdaGrad-FTRL 自适应调整，无需手动调学习率

**增强方案**：
```
FINCON 现有: 固定 α=1% CVaR → 触发 risk-averse 模式
RUCC 增强:   在线 CVaR 控制 → λ_t 动态调整风险敞口
             - 外层 CDT: 保证 CVaR ≤ 目标水平
             - 内层 AdaGrad-FTRL: 避免过度保守
```

### 3. CVRF 学习率 τ 的改进（来源：FINCON 精读 + SIT）

FINCON 用相邻 episode 交易动作的重叠百分比作为学习率。如果 QMind 动作为连续值（仓位百分比），建议用动作分布的 KL 散度或 SIT 风格的签名距离。

### 4. 记忆衰减率的数据驱动标定（来源：FINCON 精读 + Reliable Evaluation MAS）

FINCON 的记忆衰减率按数据源类型手工设定。QMind 应：
- 标定不同市场的实际数据时效性差异（A 股 T+1, 美股 T+0, 期货 T+0）
- 用事件驱动的体制变化检测来触发记忆修订（而非固定时间衰减）
- 在回撤或波动率升高时自动挂起检索（circuit breaker），避免结构断裂时用旧体制经验做决策

### 5. CVRF 输出的结构化校验（来源：FINCON 精读）

M_r 组件（风险反思 LLM）的每一步——概念化总结、meta-prompt 生成、prompt 更新——全部依赖 LLM。必须增加：
- 验证生成的 memory ID 是否存在
- 验证动作类型是否在合法集合内
- 对 meta-prompt 做反向测试——应用后 Agent 的方向是否与预期一致

---

## 六、执行层的重新设计

### TiMi 的解耦模式是否适合 QMind？

**适合，且应该是 QMind 执行层的核心架构参考**。

TiMi 的核心洞察——"策略的深度推理" 与 "执行的机械效率" 可以且应该解耦——直接适用于 QMind：

| 阶段 | QMind 对应 | 用什么 | 何时运行 |
|------|-----------|--------|---------|
| **Policy** (策略制定) | PA_Agent 两阶段分析 + fin-agent 选股 | LLM 深度推理 | 离线，按需 |
| **Optimization** (参数优化) | 回测引擎校准 | CVaR 端到端优化 (SIT 风格) | 离线，仿真 |
| **Deployment** (实盘执行) | Room 217 套利引擎 | Python/Node.js 纯代码，无 LLM | 实时，分钟/秒级 |

### Agent 决策 vs 代码执行的边界

**明确的边界**：

```
═══════════════════════════════════════════════════════════
                    LLM 应该做的事 (离线)
═══════════════════════════════════════════════════════════
  Stage 1: 信息抽取 (FinRobot Data-CoT 风格)
    - 从财报/新闻/K线中提取结构化 JSON
    - 计算并验证关键指标
    - 输出: 结构化数据摘要

  Stage 2: 策略生成 (TiMi Policy Stage 风格)
    - 识别宏观市场模式
    - 生成策略逻辑和参数范围
    - 输出: 可执行 Python 代码 + 参数配置

  Stage 3: 概率校准 (独立校准模块)
    - 不依赖 LLM 自报置信度
    - 基于历史准确率做 Platt scaling
    - 输出: 校准后概率

═══════════════════════════════════════════════════════════
                  LLM 绝不能做的事 (运行时)
═══════════════════════════════════════════════════════════
  1. 直接决定仓位大小 → 必须由独立风控模块处理
  2. 自报交易概率 → 必须经校准模块转换
  3. 实时决策推理 → 必须预编译为代码
  4. 方向辩论 → 降级为风控审核（不改变方向）
═══════════════════════════════════════════════════════════
```

**TiMi 分层编程设计对 QMind 的直接借鉴**：

```
Strategy Layer (策略层):  信号生成规则、仓位管理逻辑、入场/出场条件
                          ↓ 单向依赖
Function Layer (功能层):  技术指标计算、数据预处理、订单执行封装
                          ↓ 单向依赖
Parameter Layer (参数层): 止盈/止损阈值、仓位上限、筛选阈值
                          所有参数外化、集中管理
```

配合递进式优化：优先调参数 → 不够再换算法组件 → 最后才改策略逻辑。

**核心指标参考（TiMi 基准）**：
- 端到端延迟：137ms（内部逻辑 5ms）
- 支持交易对：213（LLM Agent 方法仅 28-81）
- 实盘 Sharpe：0.74-0.86（扣除成本后）

---

## 七、优先级排序

### P0 — 必须立刻修改（不修改则系统在方法论上不可信）

| 优先级 | 修改项 | 工作量估算 | 来源 |
|--------|--------|-----------|------|
| **1** | 所有回测增加时间一致性划分（walk-forward，显式日期边界） | 2-3 天 | P1, MR-2, 五罪 |
| **2** | 增加 Point-in-Time 数据控制（as_of 时间戳，经验库时间过滤） | 3-5 天 | P1, 五罪 Look-Ahead |
| **3** | 显式建模交易成本并强制报告 Net PnL | 2-3 天 | P5, MR-4, 五罪 Cost |
| **4** | 禁止 LLM 置信度直接用于仓位控制，建立独立校准模块 | 3-5 天 | P4, 五罪 Objective |

### P1 — 重要修正（显著影响性能和可靠性）

| 优先级 | 修改项 | 工作量估算 | 来源 |
|--------|--------|-----------|------|
| **5** | 执行时机语义精确化为 next-close | 1 天 | MR-3, MR-4 |
| **6** | 回测窗口扩展到 3-5 年跨体制 + 分段报告 | 2-3 天 | Failure 5 |
| **7** | 辩论机制重新设计（方向锁定 + 单轮 + 风控审核） | 3-5 天 | MacroAgent, FinDebate, P6 |
| **8** | 建立单 Agent 基线 + 消融实验框架 | 3-5 天 | P6, CPH |
| **9** | 策略同质化风险缓解（异构模型 + 差异化参数） | 2-3 天 | LLM-Human 行为实验 |
| **10** | 动态生存者偏差控制（时变股票池 U_t） | 2-3 天 | 五罪 Survivorship |

### P2 — 增强改进（提升上限）

| 优先级 | 修改项 | 工作量估算 | 来源 |
|--------|--------|-----------|------|
| **11** | CVRF 循环增强（RUCC 在线 CVaR + 结构化校验） | 5-7 天 | RUCC, FINCON |
| **12** | 执行层 TiMi 解耦重构（策略生成→代码执行） | 7-10 天 | TiMi |
| **13** | Financial CoT 三层推理嵌入 PA_Agent（Data→Concept→Thesis） | 5-7 天 | FinRobot CoT |
| **14** | Prompt 结构化 + 证据锚定 + 事实一致性审计 | 5-7 天 | 五罪 Rationale, FinDebate |
| **15** | 多档位成本敏感性分析自动化 | 1-2 天 | P5 |
| **16** | 完整审计日志（PIT 全链路可追溯） | 3-5 天 | MR-5, MR-6 |

### P3 — 前沿探索（长期方向）

| 优先级 | 修改项 | 工作量估算 | 来源 |
|--------|--------|-----------|------|
| **17** | Path Signature 特征替代传统技术指标 | 10-15 天 | SIT |
| **18** | CVaR 端到端决策聚焦学习（DFL）替代 predict-then-optimize | 10-15 天 | SIT |
| **19** | CVRF 宏观信念跨品种广播 | 5-7 天 | FINCON |
| **20** | 社交媒体/链上数据融合入 Agent 决策管道 | 7-10 天 | LLM Survey 2408 |

---

**核心原则总结**（三条 "绝不"）：

1. **绝不让语言置信度冒充可交易概率** — calibrate everything
2. **绝不让 LLM 输出直接决定仓位大小** — sizing must live outside the LLM
3. **绝不报告 Gross PnL 而不报告 Net PnL** — the gap is the alpha illusion itself

**以及第四条**（来自本批论文最强的信号）：
4. **绝不在建立严格单 Agent 基线之前，声称多 Agent 辩论带来 alpha** — 36 组实验胜率不到 20%，δSharpe=+0.001