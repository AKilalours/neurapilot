"""NeuraPilot telemetry — node-level latency tracking and session stats.

Design: zero-dependency context manager pattern; timing data flows into
the SQLite interaction log for offline analysis.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class NodeTimer:
    """Records timing for a single pipeline node."""
    name: str
    start_ns: int = field(default_factory=time.monotonic_ns)
    end_ns: int = 0

    def stop(self) -> None:
        self.end_ns = time.monotonic_ns()

    @property
    def elapsed_ms(self) -> float:
        if self.end_ns == 0:
            return 0.0
        return (self.end_ns - self.start_ns) / 1_000_000


@dataclass
class PipelineTrace:
    """Aggregated timing for an entire agent pipeline run."""
    nodes: list[NodeTimer] = field(default_factory=list)
    _start_ns: int = field(default_factory=time.monotonic_ns)

    def start_node(self, name: str) -> NodeTimer:
        t = NodeTimer(name=name)
        self.nodes.append(t)
        return t

    @property
    def total_ms(self) -> int:
        end = time.monotonic_ns()
        return int((end - self._start_ns) / 1_000_000)

    def summary(self) -> dict[str, float]:
        return {n.name: round(n.elapsed_ms, 1) for n in self.nodes if n.end_ns > 0}


@contextmanager
def timed_node(trace: PipelineTrace, name: str) -> Generator[NodeTimer, None, None]:
    """Context manager that starts and stops a node timer."""
    t = trace.start_node(name)
    try:
        yield t
    finally:
        t.stop()
