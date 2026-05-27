#!/usr/bin/env python3
"""Entry wrapper for Canary-1B evaluation."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "eval_canary.py"), run_name="__main__")
