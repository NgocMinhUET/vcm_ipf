"""Lightweight timing context manager for profiling pipeline stages."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from phase1.utils.log import get_logger

logger = get_logger("timer")


@contextmanager
def timer(label: str, log: bool = True) -> Generator[dict, None, None]:
    """Context manager that records elapsed time in milliseconds.

    Usage:
        with timer("detect") as t:
            run_detector(frame)
        print(t["elapsed_ms"])
    """
    result: dict = {"label": label, "elapsed_ms": 0.0}
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
        if log:
            logger.debug("%s: %.1f ms", label, result["elapsed_ms"])
