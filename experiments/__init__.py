"""Unified experiment runners."""

from pathlib import Path
import sys

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from .runner import run_baseline, run_phase1_audit

__all__ = ["run_baseline", "run_phase1_audit"]
