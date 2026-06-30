"""
分歧检测器 (δ) — 计算多 Agent 信号分歧度。

基于论文审阅结论:
- δ < 0.15 → 低分歧，直接采信最强 Agent，不做辩论
- δ >= 0.15 → 启动风控审核模式（方向锁定 + 单轮 + 仓位缩减）
"""

from __future__ import annotations

import statistics
from typing import Any

from qmind.graph.state import AnalystReport


def compute_disagreement(analyses: list[AnalystReport]) -> dict[str, Any]:
    """计算分析师之间的分歧度"""
    if not analyses:
        return {"delta": 0.0, "level": "none", "stances": [], "confidences": [],
                "strongest_analyst": None, "strongest_stance": None,
                "strongest_confidence": 0.0, "needs_debate": False}

    stances = [a.stance for a in analyses]
    confidences = [a.confidence for a in analyses]

    # 立场映射为数值
    stance_map = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
    stance_values = [stance_map.get(s, 0.0) for s in stances]

    # 分歧度 δ = 加权立场值的标准差
    # 用置信度作为权重
    if sum(confidences) > 0:
        weights = [c / sum(confidences) for c in confidences]
        mean = sum(v * w for v, w in zip(stance_values, weights, strict=True))
        variance = sum(w * (v - mean) ** 2 for v, w in zip(stance_values, weights, strict=True))
        delta = (variance ** 0.5) * 0.5  # 归一化到 ~0-1 范围
    else:
        delta = statistics.stdev(stance_values) * 0.5 if len(stance_values) > 1 else 0.0

    delta = min(delta, 1.0)

    # 分级
    if delta < 0.15:
        level = "low"
    elif delta < 0.35:
        level = "medium"
    else:
        level = "high"

    # 找出置信度最高的分析师（最强 Agent）
    max_conf_idx = max(range(len(analyses)), key=lambda i: analyses[i].confidence)
    strongest = analyses[max_conf_idx] if analyses else None

    return {
        "delta": round(delta, 4),
        "level": level,
        "stances": stances,
        "confidences": confidences,
        "strongest_analyst": strongest.analyst if strongest else None,
        "strongest_stance": strongest.stance if strongest else None,
        "strongest_confidence": strongest.confidence if strongest else 0.0,
        "needs_debate": delta >= 0.15,
    }
