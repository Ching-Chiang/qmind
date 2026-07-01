"""Walk-Forward Time Partitioning for time-series cross-validation.

Strictly chronological walk-forward splits with explicit date boundaries,
supporting expanding/sliding windows, train/validation/test separation,
and a configurable gap between training and test periods to prevent
information leakage.

Usage:
    from qmind.backtest.partition import WalkForwardPartition

    wfp = WalkForwardPartition(n_splits=5, test_size=0.2, validation_size=0.1)
    splits = wfp.split(data)        # -> list[TimeSplit]
    fold_data = wfp.get_fold_data(data, splits[0])  # -> {"train": ..., "val": ..., "test": ...}
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field


class TimeSplit(BaseModel):
    """Metadata for a single walk-forward fold.

    Contains the time boundaries and sample counts for the training,
    validation, and test periods within one fold.

    Boundaries follow the convention:
        train : [train_start, train_end)   — half-open
        val   : [val_start,   val_end)     — half-open
        test  : [test_start,  test_end]    — closed (includes both ends)

    This ensures adjacent sets do not overlap and the last test point
    is unambiguously included.
    """

    fold: int = Field(description="Zero-based fold index")
    train_start: datetime = Field(description="Start of training period (inclusive)")
    train_end: datetime = Field(description="End of training period (exclusive)")
    val_start: datetime = Field(description="Start of validation period (inclusive)")
    val_end: datetime = Field(description="End of validation period (exclusive)")
    test_start: datetime = Field(description="Start of test period (inclusive)")
    test_end: datetime = Field(description="End of test period (inclusive)")
    n_train: int = Field(default=0, description="Number of training samples in this fold")
    n_val: int = Field(default=0, description="Number of validation samples in this fold")
    n_test: int = Field(default=0, description="Number of test samples in this fold")

    @property
    def train_duration(self) -> timedelta:
        """Total duration of the training period."""
        return self.train_end - self.train_start

    @property
    def val_duration(self) -> timedelta:
        """Total duration of the validation period."""
        return self.val_end - self.val_start

    @property
    def test_duration(self) -> timedelta:
        """Total duration of the test period."""
        return self.test_end - self.test_start

    @property
    def total_samples(self) -> int:
        """Total samples across train, val, and test."""
        return self.n_train + self.n_val + self.n_test


class WalkForwardPartition:
    """Walk-Forward time-series cross-validation partitioner.

    Produces strictly chronological splits for quantitative backtesting.
    Each fold preserves temporal order — training precedes validation,
    which precedes testing.  No random shuffling is ever applied.

    Two window modes are supported:

    **expanding** (default)
        The training set grows as folds progress, always anchored to the
        earliest available timestamp.  Later folds have strictly more
        training data.

    **sliding**
        The training set has a fixed time duration equal to the combined
        duration of the validation and test windows (i.e. ``(test_size +
        validation_size) * total_span``).  The training window slides
        forward with each fold so that it always ends at ``val_start``,
        making the left edge advance across folds.  Early folds may clip
        at ``t_min`` if the window extends before the available data.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds to produce.  Must be >= 1.
    test_size : float, default 0.2
        Fraction of the total time span allocated to the test set in each
        fold.  Must be in ``(0, 1)``.
    validation_size : float, default 0.1
        Fraction of the total time span allocated to the validation set in
        each fold.  Must be in ``(0, 1)``.  ``test_size + validation_size``
        must be strictly less than 1.
    window_mode : Literal["expanding", "sliding"], default "expanding"
        Whether the training window grows or stays fixed.
    min_gap_days : int, default 0
        Minimum number of calendar days between the end of the training set
        and the start of the validation set.  A positive gap helps prevent
        look-ahead leakage.  If the gap cannot be honoured without
        producing an empty training set the gap is silently reduced.

    Examples
    --------
    .. code-block:: python

        from qmind.backtest.partition import WalkForwardPartition
        import pandas as pd

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=500, freq="D"),
            "close": range(500),
        })

        wfp = WalkForwardPartition(n_splits=3, test_size=0.2, validation_size=0.1)
        splits = wfp.split(df)
        assert len(splits) == 3

        # Inspect the first fold
        s0 = splits[0]
        print(s0.train_start, "->", s0.train_end, f"({s0.n_train} rows)")
        print(s0.val_start,   "->", s0.val_end,   f"({s0.n_val} rows)")
        print(s0.test_start,  "->", s0.test_end,  f"({s0.n_test} rows)")

        # Retrieve the actual DataFrames for fold 0
        fold = wfp.get_fold_data(df, splits[0])
        print(fold["train"].shape, fold["val"].shape, fold["test"].shape)
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        window_mode: Literal["expanding", "sliding"] = "expanding",
        min_gap_days: int = 0,
    ):
        self._validate_params(n_splits, test_size, validation_size, window_mode, min_gap_days)
        self.n_splits = n_splits
        self.test_size = test_size
        self.validation_size = validation_size
        self.window_mode = window_mode
        self.min_gap_days = min_gap_days

    # ── Validation ──────────────────────────────────────────────────────

    @staticmethod
    def _validate_params(
        n_splits: int,
        test_size: float,
        validation_size: float,
        window_mode: str,
        min_gap_days: int,
    ) -> None:
        """Validate constructor parameters and raise on invalid combinations."""
        if not isinstance(n_splits, int) or n_splits < 1:
            raise ValueError(f"n_splits must be an int >= 1, got {n_splits!r}")
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be in (0, 1), got {test_size}")
        if not 0 < validation_size < 1:
            raise ValueError(f"validation_size must be in (0, 1), got {validation_size}")
        if test_size + validation_size >= 1.0:
            raise ValueError(
                f"test_size ({test_size}) + validation_size ({validation_size})"
                f" = {test_size + validation_size} must be < 1.0"
            )
        if window_mode not in ("expanding", "sliding"):
            raise ValueError(
                f"window_mode must be 'expanding' or 'sliding', got {window_mode!r}"
            )
        if not isinstance(min_gap_days, int) or min_gap_days < 0:
            raise ValueError(f"min_gap_days must be an int >= 0, got {min_gap_days!r}")

    def _validate_data(self, data: pd.DataFrame, date_column: str) -> None:
        """Validate the input DataFrame."""
        if data is None or data.empty:
            raise ValueError("Input data must be a non-empty DataFrame")
        if date_column not in data.columns:
            raise ValueError(
                f"date_column {date_column!r} not found in DataFrame columns: "
                f"{list(data.columns)}"
            )
        if data[date_column].isna().all():
            raise ValueError(f"Column {date_column!r} contains all NaT values")

    # ── Core logic ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_timestamps(
        df: pd.DataFrame,
        date_column: str,
    ) -> tuple[pd.Series, pd.Timestamp, pd.Timestamp, float]:
        """Extract and sort timestamps, returning the time range metadata.

        Returns
        -------
        timestamps : pd.Series
            Sorted datetime series.
        t_min : pd.Timestamp
            Minimum timestamp.
        t_max : pd.Timestamp
            Maximum timestamp.
        total_seconds : float
            Total span in seconds (``t_max - t_min``).
        """
        timestamps = pd.to_datetime(df[date_column])
        t_min = timestamps.min()
        t_max = timestamps.max()
        total_seconds = (t_max - t_min).total_seconds()
        if total_seconds <= 0:
            raise ValueError(
                f"Data must span a non-zero time range. "
                f"t_min={t_min}, t_max={t_max}"
            )
        return timestamps, t_min, t_max, total_seconds

    def _compute_fold_boundaries(
        self,
        fold: int,
        t_min: pd.Timestamp,
        total_seconds: float,
    ) -> dict[str, pd.Timestamp]:
        """Compute the theoretical time boundaries for a single fold.

        Parameters
        ----------
        fold : int
            Zero-based fold index.
        t_min : pd.Timestamp
            Earliest timestamp in the full dataset.
        total_seconds : float
            Total time span of the full dataset in seconds.

        Returns
        -------
        dict with keys ``train_start``, ``train_end``, ``val_start``,
        ``val_end``, ``test_start``, ``test_end``.
        """
        # The fold's test window ends at a fixed fraction of total time.
        end_offset = (fold + 1) / self.n_splits * total_seconds
        test_end = t_min + timedelta(seconds=end_offset)

        # Work backwards: test -> val -> train
        test_start_raw = test_end - timedelta(seconds=self.test_size * total_seconds)
        test_start = max(t_min, test_start_raw)

        val_duration = self.validation_size * total_seconds
        val_start_raw = test_start - timedelta(seconds=val_duration)
        val_start = max(t_min, val_start_raw)
        val_end = test_start

        if self.window_mode == "expanding":
            train_start = t_min
        else:  # sliding
            # Training window has the same duration as val + test combined.
            train_duration = (self.test_size + self.validation_size) * total_seconds
            train_start = max(t_min, val_start - timedelta(seconds=train_duration))

        train_end = val_start

        # Apply min_gap: shift train_end earlier to leave a gap.
        if self.min_gap_days > 0:
            gap = timedelta(days=self.min_gap_days)
            candidate_end = val_start - gap
            if candidate_end > train_start:
                train_end = candidate_end
            # If gap cannot be honoured (would make train empty), silently
            # keep train_end at val_start so training still gets data.

        return {
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
        }

    # ── Public API ──────────────────────────────────────────────────────

    def split(
        self,
        data: pd.DataFrame,
        date_column: str = "timestamp",
    ) -> list[TimeSplit]:
        """Produce walk-forward train / validation / test splits.

        Parameters
        ----------
        data : pd.DataFrame
            Time-series data.  Must contain a column named by
            ``date_column`` with datetime-like values.  Rows need not be
            pre-sorted — they are sorted internally.
        date_column : str, default ``"timestamp"``
            Name of the column containing timestamps.

        Returns
        -------
        list[TimeSplit]
            One entry per fold, ordered chronologically.  Each entry
            describes the time boundaries and row counts for the train,
            validation, and test periods.

        Raises
        ------
        ValueError
            If the data is empty, the date column is missing, the time
            span is zero, or any fold's test set contains zero rows.
        """
        self._validate_data(data, date_column)
        df = data.sort_values(date_column).reset_index(drop=True)
        timestamps, t_min, t_max, total_seconds = self._resolve_timestamps(df, date_column)

        results: list[TimeSplit] = []
        for fold in range(self.n_splits):
            bounds = self._compute_fold_boundaries(fold, t_min, total_seconds)

            # Filter rows (train/val half-open, test closed).
            train_mask = (timestamps >= bounds["train_start"]) & (
                timestamps < bounds["train_end"]
            )
            val_mask = (timestamps >= bounds["val_start"]) & (
                timestamps < bounds["val_end"]
            )
            test_mask = (timestamps >= bounds["test_start"]) & (
                timestamps <= bounds["test_end"]
            )

            n_train = int(train_mask.sum())
            n_val = int(val_mask.sum())
            n_test = int(test_mask.sum())

            if n_test == 0:
                raise ValueError(
                    f"Fold {fold}: Test set is empty. "
                    f"Window [{bounds['test_start']}, {bounds['test_end']}] "
                    f"contains no data points.  Consider increasing "
                    f"``test_size``, decreasing ``n_splits``, or reducing "
                    f"``min_gap_days``."
                )

            results.append(
                TimeSplit(
                    fold=fold,
                    train_start=bounds["train_start"],
                    train_end=bounds["train_end"],
                    val_start=bounds["val_start"],
                    val_end=bounds["val_end"],
                    test_start=bounds["test_start"],
                    test_end=bounds["test_end"],
                    n_train=n_train,
                    n_val=n_val,
                    n_test=n_test,
                )
            )

        return results

    def get_fold_data(
        self,
        data: pd.DataFrame,
        split: TimeSplit,
        date_column: str = "timestamp",
    ) -> dict[str, pd.DataFrame]:
        """Extract the train / validation / test DataFrames for one fold.

        Parameters
        ----------
        data : pd.DataFrame
            Original time-series data (the same passed to ``split()``).
        split : TimeSplit
            A fold metadata object returned by ``split()``.
        date_column : str, default ``"timestamp"``
            Name of the column containing timestamps.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys ``"train"``, ``"val"``, ``"test"`` mapping to the
            corresponding subsets of the input data.
        """
        timestamps = pd.to_datetime(data[date_column])
        train = data[(timestamps >= split.train_start) & (timestamps < split.train_end)]
        val = data[(timestamps >= split.val_start) & (timestamps < split.val_end)]
        test = data[(timestamps >= split.test_start) & (timestamps <= split.test_end)]
        return {"train": train, "val": val, "test": test}

    # ── Display ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_splits={self.n_splits}, "
            f"test_size={self.test_size}, "
            f"validation_size={self.validation_size}, "
            f"window_mode='{self.window_mode}', "
            f"min_gap_days={self.min_gap_days})"
        )

    def __str__(self) -> str:
        mode_label = "expanding" if self.window_mode == "expanding" else "sliding"
        return (
            f"WalkForward({self.n_splits} folds, {mode_label}, "
            f"test={self.test_size:.0%}, val={self.validation_size:.0%}, "
            f"gap={self.min_gap_days}d)"
        )
