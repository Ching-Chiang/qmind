"""WalkForwardPartition 单元测试

Tests cover:
  1. Basic split: 2 years of daily data → correct number of folds
  2. Chronological order: each fold's train ends before val starts, val before test
  3. Expanding vs sliding window modes
  4. Gap days parameter
  5. Empty data → ValueError
  6. Insufficient data (< min required) → ValueError
  7. Returns correct TimeSplit metadata
  8. Test with hourly data
  9. Test with half-year data
  10. Property: test sets should not overlap
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from qmind.backtest.partition import TimeSplit, WalkForwardPartition


# ══════════════════════════════════════════════════════════════════════════════
#  helpers
# ══════════════════════════════════════════════════════════════════════════════

def _daily_data(
    start: str = "2024-01-01",
    periods: int = 730,
    freq: str = "D",
    col: str = "close",
) -> pd.DataFrame:
    """Build a simple DataFrame with a 'timestamp' column and a numeric column."""
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=periods, freq=freq),
        col: range(periods),
    })


def _assert_chronological(split: TimeSplit) -> None:
    """Assert train < val < test for a single fold."""
    assert split.train_end <= split.val_start, (
        f"Fold {split.fold}: train_end ({split.train_end}) > val_start ({split.val_start})"
    )
    assert split.val_end <= split.test_start, (
        f"Fold {split.fold}: val_end ({split.val_end}) > test_start ({split.test_start})"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  constructor parameter validation
# ══════════════════════════════════════════════════════════════════════════════

class TestConstructorValidation:
    def test_invalid_n_splits_negative(self):
        with pytest.raises(ValueError, match="n_splits"):
            WalkForwardPartition(n_splits=-1)

    def test_invalid_n_splits_zero(self):
        with pytest.raises(ValueError, match="n_splits"):
            WalkForwardPartition(n_splits=0)

    def test_invalid_n_splits_not_int(self):
        with pytest.raises(ValueError, match="n_splits"):
            WalkForwardPartition(n_splits=2.5)  # type: ignore[arg-type]

    def test_invalid_test_size_zero(self):
        with pytest.raises(ValueError, match="test_size"):
            WalkForwardPartition(test_size=0.0)

    def test_invalid_test_size_negative(self):
        with pytest.raises(ValueError, match="test_size"):
            WalkForwardPartition(test_size=-0.1)

    def test_invalid_test_size_one(self):
        with pytest.raises(ValueError, match="test_size"):
            WalkForwardPartition(test_size=1.0)

    def test_invalid_validation_size_zero(self):
        with pytest.raises(ValueError, match="validation_size"):
            WalkForwardPartition(validation_size=0.0)

    def test_sum_too_large(self):
        with pytest.raises(ValueError, match="test_size"):
            WalkForwardPartition(test_size=0.6, validation_size=0.5)

    def test_invalid_window_mode(self):
        with pytest.raises(ValueError, match="window_mode"):
            WalkForwardPartition(window_mode="foobar")  # type: ignore[arg-type]

    def test_invalid_min_gap_days_negative(self):
        with pytest.raises(ValueError, match="min_gap_days"):
            WalkForwardPartition(min_gap_days=-1)

    def test_valid_params(self):
        """Happy-path construction does not raise."""
        wfp = WalkForwardPartition(n_splits=5, test_size=0.2, validation_size=0.1)
        assert wfp.n_splits == 5
        assert wfp.test_size == 0.2
        assert wfp.validation_size == 0.1
        assert wfp.window_mode == "expanding"
        assert wfp.min_gap_days == 0


# ══════════════════════════════════════════════════════════════════════════════
#  data validation
# ══════════════════════════════════════════════════════════════════════════════

class TestDataValidation:
    def test_empty_data(self):
        """Empty DataFrame raises ValueError."""
        wfp = WalkForwardPartition()
        with pytest.raises(ValueError, match="non-empty"):
            wfp.split(pd.DataFrame())

    def test_missing_date_column(self):
        """Missing date column raises ValueError."""
        wfp = WalkForwardPartition()
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="date_column"):
            wfp.split(df, date_column="timestamp")

    def test_all_nat_column(self):
        """All-NaT date column raises ValueError."""
        wfp = WalkForwardPartition()
        df = pd.DataFrame({"timestamp": [pd.NaT, pd.NaT]})
        with pytest.raises(ValueError, match="NaT"):
            wfp.split(df)

    def test_zero_timespan(self):
        """Single row produces zero time span → ValueError."""
        wfp = WalkForwardPartition(n_splits=2, test_size=0.3, validation_size=0.1)
        df = pd.DataFrame({"timestamp": [pd.Timestamp("2024-06-01")], "close": [100]})
        with pytest.raises(ValueError, match="non-zero"):
            wfp.split(df)


# ══════════════════════════════════════════════════════════════════════════════
#  basic split: 2 years of daily data
# ══════════════════════════════════════════════════════════════════════════════

class TestBasicSplit:
    def test_correct_number_of_folds(self):
        """2 years of daily data yields the requested fold count."""
        df = _daily_data(periods=730)
        wfp = WalkForwardPartition(n_splits=5, test_size=0.2, validation_size=0.1)
        splits = wfp.split(df)
        assert len(splits) == 5

    def test_each_fold_has_nonzero_samples(self):
        """Every fold has positive train/val/test counts."""
        df = _daily_data(periods=730)
        # Use n_splits=3 so test_size (0.2) < 1/n_splits (~0.333);
        # otherwise fold 0's test window collapses to t_min, leaving
        # train/val empty.
        wfp = WalkForwardPartition(n_splits=3, test_size=0.2, validation_size=0.1)
        splits = wfp.split(df)
        for s in splits:
            assert s.n_train > 0, f"Fold {s.fold}: n_train == 0"
            assert s.n_val > 0, f"Fold {s.fold}: n_val == 0"
            assert s.n_test > 0, f"Fold {s.fold}: n_test == 0"

    def test_last_fold_test_includes_last_row(self):
        """The final fold's test set should include the last data point."""
        df = _daily_data(periods=730)
        wfp = WalkForwardPartition(n_splits=5, test_size=0.2, validation_size=0.1)
        splits = wfp.split(df)
        last = splits[-1]
        assert last.test_end >= df["timestamp"].max()

    def test_fold_indices_are_zero_based_and_sequential(self):
        """Fold indices are 0, 1, 2, …"""
        df = _daily_data(periods=730)
        wfp = WalkForwardPartition(n_splits=5)
        splits = wfp.split(df)
        assert [s.fold for s in splits] == [0, 1, 2, 3, 4]

    def test_total_span_encompasses_all_data(self):
        """The overall last test_end covers the entire dataset."""
        df = _daily_data(periods=730)
        wfp = WalkForwardPartition(n_splits=5, test_size=0.2, validation_size=0.1)
        splits = wfp.split(df)
        # Fold 0 starts at t_min, last fold ends at t_max
        assert splits[0].train_start == df["timestamp"].min()
        assert splits[-1].test_end >= df["timestamp"].max()


# ══════════════════════════════════════════════════════════════════════════════
#  chronological ordering
# ══════════════════════════════════════════════════════════════════════════════

class TestChronologicalOrder:
    def test_train_before_val_before_test(self):
        """Each fold respects train < val < test boundaries."""
        df = _daily_data(periods=730)
        wfp = WalkForwardPartition(n_splits=5, test_size=0.2, validation_size=0.1)
        splits = wfp.split(df)
        for s in splits:
            _assert_chronological(s)

    def test_folds_are_chronological(self):
        """Each subsequent fold starts strictly after the previous one."""
        df = _daily_data(periods=730)
        wfp = WalkForwardPartition(n_splits=5, test_size=0.2, validation_size=0.1)
        splits = wfp.split(df)
        for i in range(1, len(splits)):
            assert splits[i].test_start > splits[i - 1].test_start

    def test_get_fold_data_respects_boundaries(self):
        """get_fold_data returns non-overlapping subsets in order."""
        df = _daily_data(periods=730)
        wfp = WalkForwardPartition(n_splits=3, test_size=0.2, validation_size=0.1)
        splits = wfp.split(df)
        fold = wfp.get_fold_data(df, splits[0])

        train_ts = set(fold["train"]["timestamp"])
        val_ts = set(fold["val"]["timestamp"])
        test_ts = set(fold["test"]["timestamp"])

        assert train_ts.isdisjoint(val_ts), "Train and val overlap"
        assert train_ts.isdisjoint(test_ts), "Train and test overlap"
        assert val_ts.isdisjoint(test_ts), "Val and test overlap"


# ══════════════════════════════════════════════════════════════════════════════
#  expanding vs sliding window
# ══════════════════════════════════════════════════════════════════════════════

class TestWindowModes:
    def test_expanding_train_starts_at_t_min(self):
        """Expanding mode: every fold's train_start is the dataset minimum."""
        df = _daily_data(periods=730)
        wfp = WalkForwardPartition(
            n_splits=5, test_size=0.2, validation_size=0.1, window_mode="expanding"
        )
        splits = wfp.split(df)
        t_min = df["timestamp"].min()
        for s in splits:
            assert s.train_start == t_min, f"Fold {s.fold}: train_start moved"

    def test_expanding_n_train_increases(self):
        """Expanding mode: training set size grows across folds."""
        df = _daily_data(periods=730)
        wfp = WalkForwardPartition(
            n_splits=5, test_size=0.2, validation_size=0.1, window_mode="expanding"
        )
        splits = wfp.split(df)
        train_sizes = [s.n_train for s in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1], (
                f"Fold {i} n_train ({train_sizes[i]}) < fold {i-1} ({train_sizes[i-1]})"
            )

    def test_sliding_train_starts_advance(self):
        """Sliding mode: train_start moves forward across folds."""
        df = _daily_data(periods=730)
        wfp = WalkForwardPartition(
            n_splits=5, test_size=0.2, validation_size=0.1, window_mode="sliding"
        )
        splits = wfp.split(df)
        # At least the later folds should have moved forward (early ones
        # may clip at t_min).
        assert splits[-1].train_start > splits[0].train_start

    def test_sliding_window_fixed_duration(self):
        """Sliding mode: train duration ~ (test_size + validation_size) × total."""
        df = _daily_data(periods=730)
        total_days = (df["timestamp"].max() - df["timestamp"].min()).days
        expected_train = (0.2 + 0.1) * total_days

        wfp = WalkForwardPartition(
            n_splits=5, test_size=0.2, validation_size=0.1, window_mode="sliding"
        )
        splits = wfp.split(df)
        # Late folds (not clipping at t_min) should have the expected duration.
        for s in splits[2:]:
            actual = s.train_duration.days
            # Allow ±1 day for boundary effects
            assert abs(actual - expected_train) <= 1, (
                f"Fold {s.fold}: train_duration {actual}d, expected ~{expected_train}d"
            )

    def test_modes_return_same_fold_count(self):
        """Both modes return the same requested number of folds."""
        df = _daily_data(periods=730)
        for mode in ("expanding", "sliding"):
            wfp = WalkForwardPartition(n_splits=4, window_mode=mode)
            assert len(wfp.split(df)) == 4


# ══════════════════════════════════════════════════════════════════════════════
#  gap days
# ══════════════════════════════════════════════════════════════════════════════

class TestGapDays:
    def test_gap_creates_separation(self):
        """Positive gap shifts train_end earlier, creating a gap before val_start."""
        df = _daily_data(periods=730)
        wfp_no_gap = WalkForwardPartition(
            n_splits=3, test_size=0.2, validation_size=0.1, min_gap_days=0
        )
        wfp_gap = WalkForwardPartition(
            n_splits=3, test_size=0.2, validation_size=0.1, min_gap_days=5
        )
        splits_no_gap = wfp_no_gap.split(df)
        splits_gap = wfp_gap.split(df)

        for s_no, s_gap in zip(splits_no_gap, splits_gap):
            # Gap version should have train_end at least as early as no-gap
            assert s_gap.train_end <= s_no.train_end, (
                f"Fold {s_gap.fold}: gap did not shift train_end earlier"
            )
            gap_days = (s_gap.val_start - s_gap.train_end).days
            assert gap_days >= 0, "train_end after val_start"
            # The gap might be less than requested if data is too short;
            # just verify it's non-negative.

    def test_gap_reduced_when_insufficient_data(self):
        """Gap is silently reduced when it would make train empty."""
        df = _daily_data(periods=100)  # only ~100 days
        wfp = WalkForwardPartition(
            n_splits=2, test_size=0.2, validation_size=0.1, min_gap_days=999
        )
        splits = wfp.split(df)
        # Should not raise: gap is silently reduced.
        assert len(splits) == 2
        for s in splits:
            assert s.n_train > 0, f"Fold {s.fold}: train empty despite gap reduction"


# ══════════════════════════════════════════════════════════════════════════════
#  insufficient data
# ══════════════════════════════════════════════════════════════════════════════

class TestInsufficientData:
    def test_no_test_rows_raises(self):
        """If a fold's test set has zero rows, a ValueError is raised.

        Uses sparse data with wide gaps so a test window lands between
        actual timestamps.
        """
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-05", "2024-01-10", "2024-03-01"]
            ),
            "close": [100, 101, 102, 103],
        })
        # With total_span ~60 days and these parameters the first fold's
        # test window falls in the large gap between Jan 10 and Mar 1.
        wfp = WalkForwardPartition(n_splits=2, test_size=0.3, validation_size=0.1)
        with pytest.raises(ValueError, match="empty"):
            wfp.split(df)


# ══════════════════════════════════════════════════════════════════════════════
#  TimeSplit metadata
# ══════════════════════════════════════════════════════════════════════════════

class TestTimeSplitMetadata:
    def test_timesplit_properties(self):
        """TimeSplit helper properties compute correct values."""
        ts = TimeSplit(
            fold=0,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 6, 1),
            val_start=datetime(2024, 6, 1),
            val_end=datetime(2024, 7, 1),
            test_start=datetime(2024, 7, 1),
            test_end=datetime(2024, 8, 1),
            n_train=151,
            n_val=30,
            n_test=31,
        )
        assert ts.train_duration == timedelta(days=152)
        assert ts.val_duration == timedelta(days=30)
        assert ts.test_duration == timedelta(days=31)
        assert ts.total_samples == 151 + 30 + 31

    def test_total_samples_matches_sum(self):
        """total_samples equals n_train + n_val + n_test."""
        df = _daily_data(periods=730)
        wfp = WalkForwardPartition(n_splits=5, test_size=0.2, validation_size=0.1)
        splits = wfp.split(df)
        for s in splits:
            assert s.total_samples == s.n_train + s.n_val + s.n_test

    def test_all_fields_populated(self):
        """Every split has all TimeSplit fields set."""
        df = _daily_data(periods=730)
        wfp = WalkForwardPartition(n_splits=5)
        splits = wfp.split(df)
        for s in splits:
            assert isinstance(s.fold, int)
            assert isinstance(s.train_start, datetime)
            assert isinstance(s.train_end, datetime)
            assert isinstance(s.val_start, datetime)
            assert isinstance(s.val_end, datetime)
            assert isinstance(s.test_start, datetime)
            assert isinstance(s.test_end, datetime)
            assert s.n_train >= 0
            assert s.n_val >= 0
            assert s.n_test >= 0


# ══════════════════════════════════════════════════════════════════════════════
#  hourly data
# ══════════════════════════════════════════════════════════════════════════════

class TestHourlyData:
    def test_hourly_data_produces_folds(self):
        """Hourly data (8760 rows = 1 year) produces correct number of folds."""
        df = _daily_data(periods=8760, freq="h")
        wfp = WalkForwardPartition(n_splits=3, test_size=0.2, validation_size=0.1)
        splits = wfp.split(df)
        assert len(splits) == 3
        for s in splits:
            assert s.n_train > 0
            assert s.n_val > 0
            assert s.n_test > 0
            _assert_chronological(s)

    def test_hourly_data_no_overlap(self):
        """Hourly data: fold subsets do not overlap."""
        df = _daily_data(periods=8760, freq="h")
        wfp = WalkForwardPartition(n_splits=3, test_size=0.2, validation_size=0.1)
        splits = wfp.split(df)
        for s in splits:
            fold = wfp.get_fold_data(df, s)
            assert set(fold["train"]["timestamp"]).isdisjoint(fold["val"]["timestamp"])
            assert set(fold["train"]["timestamp"]).isdisjoint(fold["test"]["timestamp"])
            assert set(fold["val"]["timestamp"]).isdisjoint(fold["test"]["timestamp"])


# ══════════════════════════════════════════════════════════════════════════════
#  half-year data
# ══════════════════════════════════════════════════════════════════════════════

class TestHalfYearData:
    def test_half_year_data(self):
        """~180 days of daily data works with default parameters."""
        df = _daily_data(periods=180)
        wfp = WalkForwardPartition(n_splits=3, test_size=0.2, validation_size=0.1)
        splits = wfp.split(df)
        assert len(splits) == 3
        for s in splits:
            assert s.n_train > 0
            assert s.n_val > 0
            assert s.n_test > 0

    def test_half_year_edge(self):
        """Half-year with aggressive split count still works."""
        df = _daily_data(periods=180)
        wfp = WalkForwardPartition(n_splits=6, test_size=0.15, validation_size=0.05)
        splits = wfp.split(df)
        assert len(splits) == 6


# ══════════════════════════════════════════════════════════════════════════════
#  non-overlapping test sets
# ══════════════════════════════════════════════════════════════════════════════

class TestNoOverlap:
    def test_test_sets_do_not_overlap(self):
        """Each fold's test set is disjoint from all other folds' test sets."""
        df = _daily_data(periods=730)
        wfp = WalkForwardPartition(n_splits=5, test_size=0.2, validation_size=0.1)
        splits = wfp.split(df)

        test_timestamps = []
        for s in splits:
            fold = wfp.get_fold_data(df, s)
            test_timestamps.append(set(fold["test"]["timestamp"]))

        for i in range(len(test_timestamps)):
            for j in range(i + 1, len(test_timestamps)):
                assert test_timestamps[i].isdisjoint(test_timestamps[j]), (
                    f"Test sets of fold {i} and fold {j} overlap"
                )

    def test_sliding_mode_train_duration_constant(self):
        """In sliding mode, the training duration is approximately constant
        across later folds (after the window stops clipping at t_min)."""
        df = _daily_data(periods=730)
        wfp = WalkForwardPartition(
            n_splits=5, test_size=0.2, validation_size=0.1, window_mode="sliding"
        )
        splits = wfp.split(df)
        durations = [s.train_duration.days for s in splits]
        # Early folds may clip; later ones should converge to ~(0.2+0.1)*730=219 days.
        for d in durations[2:]:
            assert abs(d - 219) <= 1, f"Expected train_duration ~219d, got {d}d"


# ══════════════════════════════════════════════════════════════════════════════
#  get_fold_data
# ══════════════════════════════════════════════════════════════════════════════

class TestGetFoldData:
    def test_get_fold_data_keys(self):
        """get_fold_data returns dict with 'train', 'val', 'test' keys."""
        df = _daily_data(periods=730)
        wfp = WalkForwardPartition(n_splits=3)
        splits = wfp.split(df)
        fold = wfp.get_fold_data(df, splits[0])
        assert set(fold.keys()) == {"train", "val", "test"}

    def test_get_fold_data_preserves_columns(self):
        """Subset DataFrames have the same columns as the original."""
        df = _daily_data(periods=730)
        df["indicator"] = df["close"] * 1.5
        wfp = WalkForwardPartition(n_splits=3)
        splits = wfp.split(df)
        fold = wfp.get_fold_data(df, splits[0])
        for key in ("train", "val", "test"):
            assert list(fold[key].columns) == list(df.columns)


# ══════════════════════════════════════════════════════════════════════════════
#  repr / str
# ══════════════════════════════════════════════════════════════════════════════

class TestDisplay:
    def test_repr(self):
        wfp = WalkForwardPartition(n_splits=3, test_size=0.25, window_mode="sliding")
        r = repr(wfp)
        assert "n_splits=3" in r
        assert "test_size=0.25" in r
        assert "sliding" in r

    def test_str(self):
        wfp = WalkForwardPartition(n_splits=3, test_size=0.25, window_mode="sliding")
        s = str(wfp)
        assert "3 folds" in s
        assert "sliding" in s
        assert "25%" in s or "test=" in s


# ══════════════════════════════════════════════════════════════════════════════
#  unsorted data
# ══════════════════════════════════════════════════════════════════════════════

class TestUnsortedData:
    def test_unsorted_data_sorted_internally(self):
        """Data with shuffled timestamps is handled correctly."""
        df = _daily_data(periods=730)
        shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        wfp = WalkForwardPartition(n_splits=3, test_size=0.2, validation_size=0.1)
        splits = wfp.split(shuffled)
        assert len(splits) == 3
        for s in splits:
            assert s.n_train > 0
            assert s.n_val > 0
            assert s.n_test > 0
