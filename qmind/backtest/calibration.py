"""
Confidence Calibration Module — ECE 计算 + Platt Scaling。

P0 关键模块：禁止任何 LLM 置信度直接用作交易概率，必须经过校准。

核心功能:
    1. ECE (Expected Calibration Error) 计算
    2. Platt Scaling 参数拟合
    3. 可靠性图数据生成
    4. 校准后验证 (ECE <= 0.05)

用法:
    calibrator = ConfidenceCalibrator()
    confs = [0.9, 0.7, 0.3, ...]
    outcomes = [True, False, True, ...]

    # 评估校准误差
    result = calibrator.calculate_ece(confs, outcomes)
    print(f"ECE={result.ece:.4f}, passed={result.passed}")

    # 训练 Platt scaling
    model = calibrator.platt_scale(confs, outcomes)
    calibrated = model.transform(confs)

    # 验证校准后概率
    ok = calibrator.validate(calibrated, outcomes)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


# ============================================================================
# Data Types
# ============================================================================


class BinData(BaseModel):
    """单个分箱的校准数据。"""

    bin_index: int = Field(..., ge=0, description="分箱序号")
    bin_confidence: float = Field(..., ge=0.0, le=1.0, description="箱内平均置信度")
    bin_accuracy: float = Field(..., ge=0.0, le=1.0, description="箱内平均准确率")
    bin_count: int = Field(..., ge=0, description="箱内样本数")
    bin_gap: float = Field(..., ge=0.0, le=1.0, description="|confidence - accuracy|")


class ECEResult(BaseModel):
    """ECE 计算结果。"""

    ece: float = Field(..., ge=0.0, le=1.0, description="Expected Calibration Error")
    max_calibration_error: float = Field(
        ..., ge=0.0, le=1.0, description="Maximum Calibration Error (MCE)"
    )
    bin_data: list[BinData] = Field(default_factory=list, description="各分箱详细数据")
    passed: bool = Field(..., description="是否 ECE <= 0.05")
    n_samples: int = Field(..., ge=0, description="样本总数")


class CalibratedModel(BaseModel):
    """Platt Scaling 校准模型。"""

    alpha: float = Field(..., description="Platt scaling 斜率参数")
    beta: float = Field(..., description="Platt scaling 截距参数")
    ece_before: float = Field(
        ..., ge=0.0, le=1.0, description="校准前 ECE"
    )
    ece_after: float = Field(
        ..., ge=0.0, le=1.0, description="校准后 ECE"
    )
    n_samples: int = Field(..., ge=0, description="训练样本数")
    passed: bool = Field(..., description="校准后 ECE <= 0.05")

    def transform(self, confidences: list[float]) -> np.ndarray:
        """应用 Platt scaling 将原始置信度转换为校准后概率。"""
        confs = np.asarray(confidences, dtype=np.float64)
        # 将 [0, 1] 映射到 logit 空间，通过 epsilon 裁剪避免极端值
        eps = 1e-12
        confs = np.clip(confs, eps, 1.0 - eps)
        logits = np.log(confs / (1.0 - confs))
        calibrated = 1.0 / (1.0 + np.exp(self.alpha * logits + self.beta))
        return calibrated


# ============================================================================
# Confidence Calibrator
# ============================================================================


class ConfidenceCalibrator:
    """置信度校准器 — ECE 计算 + Platt Scaling + 可靠性图。

    校准目标:
        一个完美校准的模型应满足:
            P(y=1 | confidence = p) = p,  for all p in [0, 1]

    ECE 定义 (Guo et al. 2017):
        ECE = sum_{m=1}^M (|B_m| / n) * |acc(B_m) - conf(B_m)|

    Platt Scaling (Platt 1999):
        P(y=1|x) = 1 / (1 + exp(alpha * f(x) + beta))
        其中 f(x) = log(p / (1-p)) 是 log-odds 变换。

    References:
        - Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
        - Platt, "Probabilistic Outputs for SVMs", Advances in Large Margin
          Classifiers, 1999
        - Alpha Illusion (arXiv 2605.16895): ECE <= 0.05 方可部署
    """

    def __init__(self, n_bins_default: int = 10):
        """初始化校准器。

        Args:
            n_bins_default: ECE 计算的默认分箱数，默认 10。
        """
        if n_bins_default < 1:
            raise ValueError("n_bins_default must be >= 1")
        self.n_bins_default = n_bins_default

    # ------------------------------------------------------------------
    # ECE Calculation
    # ------------------------------------------------------------------

    def calculate_ece(
        self,
        confidences: list[float],
        outcomes: list[bool],
        n_bins: int | None = None,
    ) -> ECEResult:
        """计算 Expected Calibration Error (ECE)。

        使用等宽分箱法，将 [0, 1] 区间均匀划分为 M 个箱，
        对每个箱计算 |accuracy - confidence| 的加权平均。

        Args:
            confidences: 模型输出的置信度列表，范围 [0, 1]。
            outcomes: 对应的真实结果列表 (True = 正确/盈利)。
            n_bins: 分箱数，默认使用 self.n_bins_default。

        Returns:
            ECEResult 包含 ECE、MCE、分箱详细数据和是否通过校验。

        Raises:
            ValueError: 输入为空、长度不一致、置信度越界。
        """
        confs, outc = self._validate_inputs(confidences, outcomes)
        n = len(confs)
        m = n_bins or self.n_bins_default

        if m > n:
            warnings.warn(
                f"n_bins={m} > n_samples={n}, reducing to n_bins={n}",
                UserWarning,
                stacklevel=2,
            )
            m = n

        # ── 等宽分箱 ──────────────────────────────────────────────────
        bin_edges = np.linspace(0.0, 1.0, m + 1, dtype=np.float64)
        bin_indices = np.clip(
            np.searchsorted(bin_edges[1:-1], confs, side="right"),
            0,
            m - 1,
        )

        # ── 逐箱统计 ──────────────────────────────────────────────────
        bin_data: list[BinData] = []
        ece = 0.0
        mce = 0.0

        for i in range(m):
            mask = bin_indices == i
            count = int(mask.sum())
            if count == 0:
                continue

            bin_confs = confs[mask]
            bin_outc = outc[mask]

            bin_confidence = float(np.mean(bin_confs))
            bin_accuracy = float(np.mean(bin_outc))
            gap = abs(bin_confidence - bin_accuracy)

            weight = count / n
            ece += weight * gap
            mce = max(mce, gap)

            bin_data.append(
                BinData(
                    bin_index=i,
                    bin_confidence=round(bin_confidence, 6),
                    bin_accuracy=round(bin_accuracy, 6),
                    bin_count=count,
                    bin_gap=round(gap, 6),
                )
            )

        passed = ece <= 0.05

        return ECEResult(
            ece=round(ece, 6),
            max_calibration_error=round(mce, 6),
            bin_data=bin_data,
            passed=passed,
            n_samples=n,
        )

    # ------------------------------------------------------------------
    # Platt Scaling
    # ------------------------------------------------------------------

    def platt_scale(
        self,
        confidences: list[float],
        outcomes: list[bool],
        alpha_init: float = 1.0,
        beta_init: float = 0.0,
    ) -> CalibratedModel:
        """用 Platt scaling 拟合校准模型。

        Platt scaling 通过优化 binary cross-entropy loss 学习参数
        alpha, beta，将原始置信度映射为校准后概率。

        优化目标:
            min_{alpha, beta} -sum [ y * log(p) + (1-y) * log(1-p) ]
            其中 p = 1 / (1 + exp(alpha * logit + beta))
                logit = log(conf / (1 - conf))

        Args:
            confidences: 训练置信度列表，范围 [0, 1]。
            outcomes: 训练结果列表 (True = 正确/盈利)。
            alpha_init: alpha 初始值，默认 1.0。
            beta_init: beta 初始值，默认 0.0。

        Returns:
            CalibratedModel 包含学习到的 alpha/beta、
            校准前后 ECE 和是否通过校验。

        Raises:
            RuntimeError: scipy 未安装或优化失败。
            ValueError: 输入无效。
        """
        confs, outc = self._validate_inputs(confidences, outcomes)
        n = len(confs)

        # ── 计算校准前 ECE ────────────────────────────────────────────
        ece_before = self.calculate_ece(confidences, outcomes).ece

        # ── 准备 logit 变换 ────────────────────────────────────────────
        eps = 1e-12
        confs_clipped = np.clip(confs, eps, 1.0 - eps)
        logits = np.log(confs_clipped / (1.0 - confs_clipped))
        targets = outc.astype(np.float64)

        # ── 优化 binary cross-entropy ─────────────────────────────────
        try:
            from scipy.optimize import minimize
        except ImportError as exc:
            raise RuntimeError(
                "scipy is required for Platt scaling. "
                "Install it with: pip install scipy>=1.12.0"
            ) from exc

        def neg_log_likelihood(params: np.ndarray) -> float:
            """Binary cross-entropy 负对数似然。"""
            a, b = params
            p = 1.0 / (1.0 + np.exp(a * logits + b))
            p = np.clip(p, eps, 1.0 - eps)
            return -float(np.sum(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p)))

        result = minimize(
            neg_log_likelihood,
            x0=np.array([alpha_init, beta_init], dtype=np.float64),
            method="L-BFGS-B",
            bounds=[(1e-6, 1e6), (-1e6, 1e6)],
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            raise RuntimeError(
                f"Platt scaling optimization failed: {result.message}"
            )

        alpha, beta = float(result.x[0]), float(result.x[1])

        # ── 计算校准后 ECE ────────────────────────────────────────────
        calibrated = 1.0 / (1.0 + np.exp(alpha * logits + beta))
        ece_after = self.calculate_ece(
            calibrated.tolist(), outcomes
        ).ece

        passed = ece_after <= 0.05

        return CalibratedModel(
            alpha=round(alpha, 6),
            beta=round(beta, 6),
            ece_before=round(ece_before, 6),
            ece_after=round(ece_after, 6),
            n_samples=n,
            passed=passed,
        )

    # ------------------------------------------------------------------
    # Reliability Diagram
    # ------------------------------------------------------------------

    def reliability_diagram(
        self,
        confidences: list[float],
        outcomes: list[bool],
        n_bins: int | None = None,
    ) -> dict[str, Any]:
        """生成可靠性图数据。

        可靠性图是校准诊断的标准可视化工具:
            - X 轴: 平均置信度 (每个 bin)
            - Y 轴: 平均准确率 (每个 bin)
            - 对角线: 完美校准
            - 偏差: gap = |accuracy - confidence|

        Returns:
            dict 包含:
            - "bins": list[BinData] — 每个分箱的数据
            - "ece": float — ECE 值
            - "mce": float — MCE 值
            - "perfect_line": list[dict] — 对角线数据点 (绘图用)
            - "n_samples": int
            - "passed": bool
        """
        result = self.calculate_ece(confidences, outcomes, n_bins=n_bins)
        perfect_line = [
            {"confidence": round(v, 3), "accuracy": round(v, 3)}
            for v in np.linspace(0.0, 1.0, 11)
        ]

        return {
            "bins": [b.model_dump() for b in result.bin_data],
            "ece": result.ece,
            "mce": result.max_calibration_error,
            "perfect_line": perfect_line,
            "n_samples": result.n_samples,
            "passed": result.passed,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        calibrated_probs: list[float],
        outcomes: list[bool],
    ) -> bool:
        """验证校准后概率是否满足部署条件 (ECE <= 0.05)。

        这是 Alpha Illusion (arXiv 2605.16895) 要求的硬性条件:
            ECE <= 0.05 方可用于仓位计算。

        Args:
            calibrated_probs: 校准后概率列表，范围 [0, 1]。
            outcomes: 对应的真实结果列表。

        Returns:
            True 如果 ECE <= 0.05，否则 False。
        """
        result = self.calculate_ece(calibrated_probs, outcomes)
        return result.passed

    # ------------------------------------------------------------------
    # Input Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        confidences: list[float],
        outcomes: list[bool],
    ) -> tuple[np.ndarray, np.ndarray]:
        """校验输入并转为 numpy 数组。

        Raises:
            ValueError: 输入为空、长度不一致、置信度越界。
        """
        if not confidences or not outcomes:
            raise ValueError("confidences and outcomes must not be empty")

        if len(confidences) != len(outcomes):
            raise ValueError(
                f"confidences ({len(confidences)}) and outcomes "
                f"({len(outcomes)}) must have the same length"
            )

        confs = np.asarray(confidences, dtype=np.float64)
        outc = np.asarray(outcomes, dtype=np.float64)

        if np.any((confs < 0.0) | (confs > 1.0)):
            raise ValueError(
                f"confidences must be in [0, 1], got range "
                f"[{float(confs.min())}, {float(confs.max())}]"
            )

        if not np.all(np.isin(outc, [0.0, 1.0])):
            raise ValueError("outcomes must be boolean or {0, 1}")

        return confs, outc
