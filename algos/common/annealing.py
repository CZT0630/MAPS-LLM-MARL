"""Distillation weight schedules driven by environment-step progress."""

from __future__ import annotations

import math


def _finite_non_negative(value: float, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {parsed}")
    return parsed


def _valid_progress(progress: float) -> float:
    parsed = float(progress)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"progress must be finite and in [0, 1], got {parsed}")
    return parsed


class AnnealingSchedule:
    """Piecewise-constant high/low/zero distillation schedule."""

    def __init__(
        self,
        lambda_high: float = 0.8,
        lambda_low: float = 0.15,
        stage1_end: float = 0.3,
        stage2_end: float = 0.7,
    ):
        self.lambda_high = _finite_non_negative(lambda_high, "lambda_high")
        self.lambda_low = _finite_non_negative(lambda_low, "lambda_low")
        if self.lambda_high < self.lambda_low:
            raise ValueError(
                "lambda_high must be greater than or equal to lambda_low"
            )
        self.stage1_end = float(stage1_end)
        self.stage2_end = float(stage2_end)

        if (
            not math.isfinite(self.stage1_end)
            or not math.isfinite(self.stage2_end)
            or not 0.0 <= self.stage1_end <= self.stage2_end <= 1.0
        ):
            raise ValueError(
                "stage thresholds must be finite and satisfy "
                "0 <= stage1_end <= stage2_end <= 1, "
                f"got stage1_end={self.stage1_end}, "
                f"stage2_end={self.stage2_end}"
            )

    def get_lambda(self, progress: float) -> float:
        """Return the weight for environment-step progress in [0, 1]."""
        parsed = _valid_progress(progress)
        if parsed < self.stage1_end:
            return self.lambda_high
        if parsed < self.stage2_end:
            return self.lambda_low
        return 0.0

    def to_dict(self) -> dict[str, float | str]:
        return {
            "type": "annealing",
            "lambda_high": self.lambda_high,
            "lambda_low": self.lambda_low,
            "stage1_end": self.stage1_end,
            "stage2_end": self.stage2_end,
            "progress_unit": "environment_steps",
        }


class FixedSchedule:
    """Constant distillation weight without annealing."""

    def __init__(self, lambda_value: float = 0.15):
        self.lambda_value = _finite_non_negative(lambda_value, "lambda_value")

    def get_lambda(self, progress: float) -> float:
        """Return the fixed weight after validating training progress."""
        _valid_progress(progress)
        return self.lambda_value

    def to_dict(self) -> dict[str, float | str]:
        return {
            "type": "fixed",
            "lambda_value": self.lambda_value,
            "progress_unit": "environment_steps",
        }
