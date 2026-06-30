"""
教训注入模块 — 将历史教训插入分析师 prompt 顶部。

"注意：历史上类似市况曾出现以下教训：
 1. 回调时过早入场导致被止损 (3次相似)
 2. 突破确认后再入场盈利更高 (2次相似)"
"""

from __future__ import annotations

from qmind.graph.state import MarketConditionVector
from qmind.learning.memory import MemoryStore


class LessonInjector:
    """教训注入器 — 将历史教训插入分析 prompt"""

    def __init__(self, memory: MemoryStore):
        self.memory = memory

    def build_injection(
        self,
        condition: MarketConditionVector,
        symbol: str = "",
        top_k: int = 5,
    ) -> str:
        """构建要注入的教训文本"""
        results = self.memory.search_similar(
            condition=condition,
            top_k=top_k,
            min_similarity=0.3,
            symbol=symbol if symbol else None,
        )

        if not results:
            return ""

        lines = ["⚠️ 注意：历史上类似市况曾出现以下教训："]
        for entry, sim in results:
            for lesson in entry.lessons:
                source_marker = f" ({entry.symbol})" if entry.symbol else ""
                lines.append(
                    f"  📌 {lesson.lesson} "
                    f"(置信度: {lesson.confidence:.0%}, "
                    f"相似度: {sim:.0%}, "
                    f"来源{source_marker})"
                )

        lines.append("─── 请参考以上历史教训进行分析 ───")
        return "\n".join(lines)

    def inject_into_prompt(self, prompt: str, condition: MarketConditionVector, symbol: str = "") -> str:
        """将教训注入到 prompt 顶部"""
        injection = self.build_injection(condition, symbol)
        if not injection:
            return prompt
        return f"{injection}\n\n{prompt}"
