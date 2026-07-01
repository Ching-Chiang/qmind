"""qmind/backtest/calibration.py 单元测试。

覆盖范围:
  1. 完美校准 → ECE ≈ 0.0
  2. 随机置信度 → ECE > 0
  3. 5 bins vs 10 bins 一致性
  4. Platt scaling 改善 ECE
  5. MCE 计算
  6. 分箱数据完整性 (bin count 之和 = n_samples)
  7. 空输入 → ValueError
  8. 边界: 所有置信度相同
  9. 边界: 所有结果相同
 10. validate() 通过/不通过
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qmind.backtest.calibration import (
    BinData,
    CalibratedModel,
    ConfidenceCalibrator,
    ECEResult,
)


@pytest.fixture
def calibrator() -> ConfidenceCalibrator:
    return ConfidenceCalibrator(n_bins_default=10)


# ============================================================================
# 1. Perfect calibration → ECE ≈ 0.0
# ============================================================================


class TestPerfectCalibration:
    """当每个箱内置信度 = 准确率时，ECE 应为 0 (或接近 0)。"""

    def test_perfect_ece_is_zero(self, calibrator: ConfidenceCalibrator):
        """每个箱内置信度恰好等于准确率 → ECE = 0.0。"""
        rng = np.random.default_rng(42)
        n = 500
        # 生成置信度，让每个样本的 outcome 概率恰好等于置信度
        confs = rng.uniform(0.05, 0.95, n).tolist()
        outcomes = [rng.random() < c for c in confs]
        result = calibrator.calculate_ece(confs, outcomes)
        # 大样本下 ECE 应接近 0 (Guo et al. 2017 收敛性)
        assert result.ece < 0.03, f"ECE={result.ece} should be near 0 for perfectly calibrated"
        assert result.passed is True

    def test_perfect_bin_alignment(self, calibrator: ConfidenceCalibrator):
        """手动构造完美对齐的分箱: 箱内置信度 == 准确率。"""
        confs = [0.1] * 100 + [0.3] * 100 + [0.5] * 100 + [0.7] * 100 + [0.9] * 100
        outcomes = (
            [True] * 10 + [False] * 90  # 10% accurate
            + [True] * 30 + [False] * 70  # 30%
            + [True] * 50 + [False] * 50  # 50%
            + [True] * 70 + [False] * 30  # 70%
            + [True] * 90 + [False] * 10  # 90%
        )
        result = calibrator.calculate_ece(confs, outcomes, n_bins=5)
        assert result.ece < 0.01, f"ECE={result.ece} should be ~0 for bin-aligned data"
        assert result.passed is True


# ============================================================================
# 2. Random confidences → ECE > 0
# ============================================================================


class TestRandomConfidences:
    """无关联的随机置信度与结果 → ECE 通常 > 0。"""

    def test_random_ece_positive(self, calibrator: ConfidenceCalibrator):
        """随机置信度与独立随机结果 → ECE 不接近 0。"""
        rng = np.random.default_rng(12345)
        confs = rng.uniform(0.0, 1.0, 300).tolist()
        outcomes = rng.choice([True, False], 300).tolist()
        result = calibrator.calculate_ece(confs, outcomes)
        # 随机配对几乎不可能 ECE < 0.01
        assert result.ece > 0.01, (
            f"ECE={result.ece} should be > 0 for random confidences vs random outcomes"
        )
        assert result.n_samples == 300

    def test_random_with_5_bins(self, calibrator: ConfidenceCalibrator):
        """验证不同随机种子下 ECE 均为正。"""
        for seed in [99, 101, 2024]:
            rng = np.random.default_rng(seed)
            confs = rng.uniform(0.0, 1.0, 200).tolist()
            outcomes = rng.choice([True, False], 200).tolist()
            result = calibrator.calculate_ece(confs, outcomes, n_bins=5)
            assert result.ece > 0.005, f"seed={seed}: ECE={result.ece} too low for random data"


# ============================================================================
# 3. ECE with 5 bins vs 10 bins (consistency)
# ============================================================================


class TestBinCountConsistency:
    """不同分箱数应产生一致的校准评估趋势。"""

    def test_ece_trend_same_direction(self, calibrator: ConfidenceCalibrator):
        """过拟合数据: 校准差 → 两种分箱都给出 ECE > 0.05。"""
        rng = np.random.default_rng(42)
        # 构造严重过拟合 (高置信度但准确率低)
        confs = rng.uniform(0.7, 1.0, 400).tolist()
        outcomes = rng.choice([True, False], 400, p=[0.4, 0.6]).tolist()
        ece_5 = calibrator.calculate_ece(confs, outcomes, n_bins=5).ece
        ece_10 = calibrator.calculate_ece(confs, outcomes, n_bins=10).ece
        # 两者都检测到校准问题
        assert ece_5 > 0.05, f"ECE(5)={ece_5} should detect miscalibration"
        assert ece_10 > 0.05, f"ECE(10)={ece_10} should detect miscalibration"

    def test_bin_count_sums_match(self, calibrator: ConfidenceCalibrator):
        """5 箱与 10 箱各自的 bin counts 之和都等于 n_samples。"""
        rng = np.random.default_rng(7)
        confs = rng.uniform(0.1, 0.9, 150).tolist()
        outcomes = rng.choice([True, False], 150).tolist()
        for n_bins in [5, 10]:
            result = calibrator.calculate_ece(confs, outcomes, n_bins=n_bins)
            total = sum(b.bin_count for b in result.bin_data)
            assert total == result.n_samples, (
                f"n_bins={n_bins}: sum(bin_counts)={total} != n_samples={result.n_samples}"
            )

    def test_default_n_bins_10(self, calibrator: ConfidenceCalibrator):
        """未指定 n_bins 时默认使用 10 箱。"""
        rng = np.random.default_rng(1)
        confs = rng.uniform(0.0, 1.0, 200).tolist()
        outcomes = rng.choice([True, False], 200).tolist()
        result = calibrator.calculate_ece(confs, outcomes)
        assert len(result.bin_data) <= 10


# ============================================================================
# 4. Platt scaling improves ECE
# ============================================================================


class TestPlattScaling:
    """Platt scaling 应降低 ECE (校准改善)。"""

    def test_reduces_ece_for_overconfidence(self, calibrator: ConfidenceCalibrator):
        """过置信数据: 校准后 ECE < 校准前 ECE。"""
        rng = np.random.default_rng(42)
        n = 300
        # 过置信: 模型预测高概率但实际只有 40% 正确
        confs = rng.beta(8, 2, n).tolist()   # 集中在 ~0.8
        outcomes = [rng.random() < 0.4 for _ in range(n)]
        model = calibrator.platt_scale(confs, outcomes)
        assert model.ece_after < model.ece_before, (
            f"Platt scaling did not improve ECE: {model.ece_before} -> {model.ece_after}"
        )

    def test_improves_underconfidence(self, calibrator: ConfidenceCalibrator):
        """欠置信数据: 校准后 ECE 改善。"""
        rng = np.random.default_rng(123)
        n = 300
        # 欠置信: 模型预测低概率但实际 80% 正确
        confs = rng.beta(2, 8, n).tolist()   # 集中在 ~0.2
        outcomes = [rng.random() < 0.8 for _ in range(n)]
        model = calibrator.platt_scale(confs, outcomes)
        assert model.ece_after < model.ece_before, (
            f"Platt scaling did not improve ECE: {model.ece_before} -> {model.ece_after}"
        )

    def test_passed_flag_with_good_data(self, calibrator: ConfidenceCalibrator):
        """校准后 ECE <= 0.05 → passed=True。"""
        rng = np.random.default_rng(42)
        n = 500
        # 近乎完美校准的数据
        confs = rng.uniform(0.1, 0.9, n).tolist()
        outcomes = [rng.random() < c for c in confs]
        model = calibrator.platt_scale(confs, outcomes)
        # 已经接近校准，Platt 不应破坏校准
        assert model.passed is True, (
            f"Platt scaling should pass for well-calibrated data, "
            f"ece_after={model.ece_after}"
        )

    def test_transform_produces_valid_probs(self, calibrator: ConfidenceCalibrator):
        """Platt 变换后的概率在 [0, 1] 范围内。"""
        rng = np.random.default_rng(42)
        confs = rng.uniform(0.0, 1.0, 100).tolist()
        outcomes = rng.choice([True, False], 100).tolist()
        model = calibrator.platt_scale(confs, outcomes)
        cal = model.transform(confs)
        assert np.all(cal >= 0.0), "Calibrated probs should be >= 0"
        assert np.all(cal <= 1.0), "Calibrated probs should be <= 1"

    def test_transform_monotonic(self, calibrator: ConfidenceCalibrator):
        """Platt scaling 是保序变换: 更高置信度 → 更高校准概率。"""
        rng = np.random.default_rng(42)
        confs = sorted(rng.uniform(0.0, 1.0, 50).tolist())
        outcomes = rng.choice([True, False], 50).tolist()
        model = calibrator.platt_scale(confs, outcomes)
        cal = model.transform(confs)
        # 检查单调性 (允许少量浮点误差)
        diffs = np.diff(cal)
        assert np.all(diffs >= -1e-6), "Platt scaling should be monotonic (tiny FP noise allowed)"

    def test_model_fields(self, calibrator: ConfidenceCalibrator):
        """CalibratedModel 包含所有必要字段。"""
        confs = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        outcomes = [True, True, False, False, True, False]
        model = calibrator.platt_scale(confs, outcomes)
        assert model.alpha > 0, f"alpha={model.alpha} should be positive"
        assert isinstance(model.beta, float)
        assert model.n_samples == 6
        assert model.ece_before >= 0.0
        assert model.ece_after >= 0.0


# ============================================================================
# 5. MCE calculation
# ============================================================================


class TestMCE:
    """Maximum Calibration Error 计算验证。"""

    def test_mce_is_max_bin_gap(self, calibrator: ConfidenceCalibrator):
        """MCE 应等于所有箱中最大的 gap。"""
        rng = np.random.default_rng(42)
        confs = rng.uniform(0.0, 1.0, 200).tolist()
        outcomes = rng.choice([True, False], 200).tolist()
        result = calibrator.calculate_ece(confs, outcomes, n_bins=5)
        expected_mce = max(b.bin_gap for b in result.bin_data)
        assert abs(result.max_calibration_error - expected_mce) < 1e-10, (
            f"MCE={result.max_calibration_error} != max gap={expected_mce}"
        )

    def test_mce_gt_ece(self, calibrator: ConfidenceCalibrator):
        """MCE 通常大于或等于 ECE (MCE 是最大值，ECE 是加权平均)。"""
        rng = np.random.default_rng(42)
        confs = rng.uniform(0.0, 1.0, 300).tolist()
        outcomes = rng.binomial(1, 0.5, 300).tolist()
        result = calibrator.calculate_ece(confs, outcomes, n_bins=5)
        assert result.max_calibration_error >= result.ece, (
            f"MCE={result.max_calibration_error} should be >= ECE={result.ece}"
        )

    def test_mce_zero_with_perfect_bins(self, calibrator: ConfidenceCalibrator):
        """完美对齐时 MCE 也应为 0。"""
        confs = [0.2] * 50 + [0.8] * 50
        outcomes = [True] * 10 + [False] * 40 + [True] * 40 + [False] * 10
        result = calibrator.calculate_ece(confs, outcomes, n_bins=2)
        assert result.max_calibration_error < 0.01


# ============================================================================
# 6. Bin data integrity (sum of bin counts = n_samples)
# ============================================================================


class TestBinDataIntegrity:
    """分箱统计数据的完整性检查。"""

    def test_bin_counts_sum(self, calibrator: ConfidenceCalibrator):
        """所有箱的样本数应等于总样本数。"""
        rng = np.random.default_rng(42)
        for n in [10, 50, 100, 500]:
            confs = rng.uniform(0.0, 1.0, n).tolist()
            outcomes = rng.choice([True, False], n).tolist()
            result = calibrator.calculate_ece(confs, outcomes, n_bins=10)
            total = sum(b.bin_count for b in result.bin_data)
            assert total == n, f"n={n}: sum(bin_counts)={total} != {n}"

    def test_no_empty_bins_included(self, calibrator: ConfidenceCalibrator):
        """空箱不应出现在 bin_data 中。"""
        rng = np.random.default_rng(42)
        # 将置信度限制在一个小区间，使某些箱为空
        confs = rng.uniform(0.4, 0.6, 50).tolist()
        outcomes = rng.choice([True, False], 50).tolist()
        result = calibrator.calculate_ece(confs, outcomes, n_bins=10)
        for b in result.bin_data:
            assert b.bin_count > 0, f"Bin {b.bin_index} is empty but still in bin_data"

    def test_bin_gap_non_negative(self, calibrator: ConfidenceCalibrator):
        """每个箱的 gap 应为非负数。"""
        rng = np.random.default_rng(42)
        confs = rng.uniform(0.0, 1.0, 200).tolist()
        outcomes = rng.choice([True, False], 200).tolist()
        result = calibrator.calculate_ece(confs, outcomes, n_bins=5)
        for b in result.bin_data:
            assert b.bin_gap >= 0.0, f"Bin {b.bin_index}: gap={b.bin_gap} < 0"
            assert b.bin_confidence >= 0.0
            assert b.bin_accuracy >= 0.0
            assert b.bin_confidence <= 1.0
            assert b.bin_accuracy <= 1.0


# ============================================================================
# 7. Empty input → ValueError
# ============================================================================


class TestEmptyInput:
    """空输入应引发 ValueError。"""

    def test_empty_confidences(self, calibrator: ConfidenceCalibrator):
        with pytest.raises(ValueError, match="must not be empty"):
            calibrator.calculate_ece([], [True, False])

    def test_empty_outcomes(self, calibrator: ConfidenceCalibrator):
        with pytest.raises(ValueError, match="must not be empty"):
            calibrator.calculate_ece([0.3], [])

    def test_both_empty(self, calibrator: ConfidenceCalibrator):
        with pytest.raises(ValueError, match="must not be empty"):
            calibrator.calculate_ece([], [])

    def test_empty_for_platt(self, calibrator: ConfidenceCalibrator):
        with pytest.raises(ValueError, match="must not be empty"):
            calibrator.platt_scale([], [])

    def test_empty_for_validate(self, calibrator: ConfidenceCalibrator):
        with pytest.raises(ValueError, match="must not be empty"):
            calibrator.validate([], [])


# ============================================================================
# 8. Edge: all same confidence
# ============================================================================


class TestAllSameConfidence:
    """所有置信度相同时的 ECE 行为。"""

    def test_all_confidence_0_5(self, calibrator: ConfidenceCalibrator):
        """所有置信度 = 0.5。"""
        confs = [0.5] * 200
        outcomes = [True] * 100 + [False] * 100
        result = calibrator.calculate_ece(confs, outcomes, n_bins=5)
        # 所有值落在一个箱中 (0.5 在 0.4-0.6 箱)
        # 箱内准确率 = 0.5, 置信度 = 0.5 → gap ≈ 0
        assert result.ece < 0.01, f"ECE={result.ece} should be ~0 when conf=acc=0.5"
        # 仅一个非空箱
        assert len(result.bin_data) == 1

    def test_all_confidence_1_0(self, calibrator: ConfidenceCalibrator):
        """所有置信度 = 1.0, 但准确率 < 1.0 → 高 ECE。"""
        confs = [1.0] * 100
        outcomes = [True] * 60 + [False] * 40
        result = calibrator.calculate_ece(confs, outcomes, n_bins=10)
        assert result.ece > 0.3, f"ECE={result.ece} should be large for overconfidence"
        assert result.passed is False

    def test_all_confidence_0_0(self, calibrator: ConfidenceCalibrator):
        """所有置信度 = 0.0, 但准确率 > 0.0 → 高 ECE。"""
        confs = [0.0] * 100
        outcomes = [True] * 30 + [False] * 70
        result = calibrator.calculate_ece(confs, outcomes, n_bins=10)
        assert result.ece > 0.2, f"ECE={result.ece} should be large for underconfidence"

    def test_single_value_platt(self, calibrator: ConfidenceCalibrator):
        """单值置信度在 Platt 中不会崩溃。"""
        confs = [0.7] * 50
        outcomes = [True] * 35 + [False] * 15
        model = calibrator.platt_scale(confs, outcomes)
        assert model.ece_after >= 0.0
        cal = model.transform(confs)
        assert len(cal) == 50
        assert np.all(cal >= 0.0) and np.all(cal <= 1.0)


# ============================================================================
# 9. Edge: all same outcome
# ============================================================================


class TestAllSameOutcome:
    """所有结果相同时的 ECE 行为。"""

    def test_all_true(self, calibrator: ConfidenceCalibrator):
        """所有结果 = True。"""
        rng = np.random.default_rng(42)
        confs = rng.uniform(0.0, 1.0, 200).tolist()
        outcomes = [True] * 200
        result = calibrator.calculate_ece(confs, outcomes, n_bins=10)
        # 高置信度箱内准确率 ≈ 1.0 → gap 可测
        assert result.ece > 0.0
        assert result.n_samples == 200

    def test_all_false(self, calibrator: ConfidenceCalibrator):
        """所有结果 = False。"""
        rng = np.random.default_rng(42)
        confs = rng.uniform(0.0, 1.0, 200).tolist()
        outcomes = [False] * 200
        result = calibrator.calculate_ece(confs, outcomes, n_bins=10)
        assert result.ece > 0.0
        assert result.n_samples == 200

    def test_all_true_with_constant_conf(self, calibrator: ConfidenceCalibrator):
        """所有结果 = True, 所有置信度 = 0.8 → gap 可预测。"""
        confs = [0.8] * 100
        outcomes = [True] * 100
        result = calibrator.calculate_ece(confs, outcomes, n_bins=5)
        # gap = |0.8 - 1.0| = 0.2, 单箱, weight = 1.0
        assert abs(result.ece - 0.2) < 0.01, f"ECE={result.ece} should be 0.2"
        assert result.passed is False


# ============================================================================
# 10. validate() method with passing and failing cases
# ============================================================================


class TestValidateMethod:
    """validate() 是 calculate_ece().passed 的便捷封装。"""

    def test_validate_passes(self, calibrator: ConfidenceCalibrator):
        """完美校准数据 → validate() == True。"""
        rng = np.random.default_rng(42)
        n = 500
        confs = rng.uniform(0.1, 0.9, n).tolist()
        outcomes = [rng.random() < c for c in confs]
        assert calibrator.validate(confs, outcomes) is True

    def test_validate_fails(self, calibrator: ConfidenceCalibrator):
        """严重过置信数据 → validate() == False。"""
        confs = [0.95] * 100
        outcomes = [True] * 30 + [False] * 70
        assert calibrator.validate(confs, outcomes) is False

    def test_validate_passed_flag_alignment(self, calibrator: ConfidenceCalibrator):
        """validate() 与 ECEResult.passed 一致。"""
        rng = np.random.default_rng(42)
        confs = rng.uniform(0.0, 1.0, 300).tolist()
        outcomes = rng.choice([True, False], 300).tolist()
        result = calibrator.calculate_ece(confs, outcomes)
        assert calibrator.validate(confs, outcomes) == result.passed

    def test_validate_edge_threshold(self, calibrator: ConfidenceCalibrator):
        """ECE 刚好在 0.05 边界附近应正确反映 passed。"""
        # 构造 ECE 略高于 0.05 的数据
        rng = np.random.default_rng(999)
        # 小样本 + 轻微不匹配 → ECE 可能略 > 0.05
        confs = rng.uniform(0.5, 0.9, 50).tolist()
        outcomes = rng.choice([True, False], 50, p=[0.6, 0.4]).tolist()
        result = calibrator.calculate_ece(confs, outcomes, n_bins=5)
        # 这只是记录验证结果，不假设通过与否
        assert calibrator.validate(confs, outcomes) == result.passed


# ============================================================================
# 附加: 输入校验
# ============================================================================


class TestInputValidation:
    """_validate_inputs 的边界情况。"""

    def test_length_mismatch(self, calibrator: ConfidenceCalibrator):
        with pytest.raises(ValueError, match="must have the same length"):
            calibrator.calculate_ece([0.1, 0.2], [True])

    def test_confidence_out_of_range(self, calibrator: ConfidenceCalibrator):
        with pytest.raises(ValueError, match="confidences must be in"):
            calibrator.calculate_ece([-0.1, 0.5], [True, False])

    def test_confidence_above_one(self, calibrator: ConfidenceCalibrator):
        with pytest.raises(ValueError, match="confidences must be in"):
            calibrator.calculate_ece([1.5, 0.5], [True, False])

    def test_invalid_outcomes_type(self, calibrator: ConfidenceCalibrator):
        with pytest.raises(ValueError, match="outcomes must be boolean"):
            calibrator.calculate_ece([0.5, 0.5], [1, 2])

    def test_n_bins_gt_n_samples_warns(self, calibrator: ConfidenceCalibrator):
        """n_bins > n_samples 时发出警告并降低 bin 数。"""
        confs = [0.3, 0.7]
        outcomes = [True, False]
        with pytest.warns(UserWarning, match="reducing to n_bins=2"):
            result = calibrator.calculate_ece(confs, outcomes, n_bins=10)
        assert result.ece >= 0.0
        assert len(result.bin_data) <= 2

    def test_invalid_n_bins_default(self):
        """n_bins_default < 1 应引发 ValueError。"""
        with pytest.raises(ValueError, match="n_bins_default must be >= 1"):
            ConfidenceCalibrator(n_bins_default=0)


# ============================================================================
# 附加: 可靠性图
# ============================================================================


class TestReliabilityDiagram:
    """reliability_diagram() 输出完整性。"""

    def test_diagram_keys(self, calibrator: ConfidenceCalibrator):
        rng = np.random.default_rng(42)
        confs = rng.uniform(0.1, 0.9, 200).tolist()
        outcomes = rng.choice([True, False], 200).tolist()
        diag = calibrator.reliability_diagram(confs, outcomes)
        assert "bins" in diag
        assert "ece" in diag
        assert "mce" in diag
        assert "perfect_line" in diag
        assert "n_samples" in diag
        assert "passed" in diag
        assert diag["n_samples"] == 200
        assert len(diag["perfect_line"]) == 11

    def test_diagram_empty_input(self, calibrator: ConfidenceCalibrator):
        with pytest.raises(ValueError):
            calibrator.reliability_diagram([], [])


# ============================================================================
# 附加: n_bins_default 构造
# ============================================================================


class TestConstructor:
    def test_default_n_bins(self):
        c = ConfidenceCalibrator()
        assert c.n_bins_default == 10

    def test_custom_n_bins(self):
        c = ConfidenceCalibrator(n_bins_default=20)
        assert c.n_bins_default == 20
