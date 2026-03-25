"""Performance metrics and hashrate tracking.

Tracks:
- Classical hash rate (H/s)
- Quantum iterations per second
- Quantum advantage ratio (classical_time / quantum_time)
- Solutions found, accepted, rejected
- Challenge fetch latency
- VDF computation time
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class MiningStats:
    """Accumulated mining statistics."""
    # Counters
    challenges_fetched: int = 0
    solutions_found: int = 0
    solutions_accepted: int = 0
    solutions_rejected: int = 0
    grover_runs: int = 0
    classical_nonces_tried: int = 0

    # Timing (seconds)
    total_mining_time: float = 0.0
    total_quantum_time: float = 0.0
    total_classical_time: float = 0.0
    total_vdf_time: float = 0.0

    # Rewards
    total_reward_qnk: float = 0.0

    # Rates (computed)
    _last_rate_update: float = field(default=0.0, repr=False)
    _rate_window_nonces: int = field(default=0, repr=False)
    _rate_window_start: float = field(default=0.0, repr=False)
    current_hashrate: float = 0.0  # H/s

    @property
    def acceptance_rate(self) -> float:
        if self.solutions_found == 0:
            return 0.0
        return self.solutions_accepted / self.solutions_found

    @property
    def quantum_advantage(self) -> float:
        """Ratio of classical time saved by quantum search."""
        if self.total_quantum_time == 0:
            return 0.0
        return self.total_classical_time / self.total_quantum_time

    @property
    def avg_vdf_time_ms(self) -> float:
        if self.solutions_found == 0:
            return 0.0
        return (self.total_vdf_time / self.solutions_found) * 1000

    @property
    def avg_quantum_time_ms(self) -> float:
        if self.grover_runs == 0:
            return 0.0
        return (self.total_quantum_time / self.grover_runs) * 1000

    def update_hashrate(self, nonces_this_batch: int) -> None:
        """Update rolling hashrate estimate."""
        now = time.monotonic()
        if self._rate_window_start == 0:
            self._rate_window_start = now
            self._rate_window_nonces = 0

        self._rate_window_nonces += nonces_this_batch
        elapsed = now - self._rate_window_start

        # Update rate every 5 seconds
        if elapsed >= 5.0:
            self.current_hashrate = self._rate_window_nonces / elapsed
            self._rate_window_start = now
            self._rate_window_nonces = 0


class MetricsTracker:
    """Tracks and reports mining performance metrics."""

    def __init__(self, log_interval: float = 30.0, track_advantage: bool = True):
        self.stats = MiningStats()
        self._log_interval = log_interval
        self._track_advantage = track_advantage
        self._start_time = time.monotonic()
        self._last_log = 0.0

    def record_challenge(self) -> None:
        self.stats.challenges_fetched += 1

    def record_grover_run(self, duration_seconds: float) -> None:
        self.stats.grover_runs += 1
        self.stats.total_quantum_time += duration_seconds

    def record_classical_search(self, nonces_tried: int, duration_seconds: float) -> None:
        self.stats.classical_nonces_tried += nonces_tried
        self.stats.total_classical_time += duration_seconds
        self.stats.update_hashrate(nonces_tried)

    def record_vdf(self, duration_seconds: float) -> None:
        self.stats.total_vdf_time += duration_seconds

    def record_solution_found(self) -> None:
        self.stats.solutions_found += 1

    def record_solution_accepted(self, reward_qnk: float) -> None:
        self.stats.solutions_accepted += 1
        self.stats.total_reward_qnk += reward_qnk

    def record_solution_rejected(self) -> None:
        self.stats.solutions_rejected += 1

    def maybe_log(self) -> None:
        """Log metrics if enough time has passed since last log."""
        now = time.monotonic()
        if now - self._last_log < self._log_interval:
            return
        self._last_log = now
        self._log_stats()

    def _log_stats(self) -> None:
        """Output current mining statistics."""
        uptime = time.monotonic() - self._start_time
        stats = self.stats

        logger.info(
            "mining_stats",
            uptime_min=f"{uptime / 60:.1f}",
            hashrate=f"{stats.current_hashrate:.1f} H/s",
            solutions=f"{stats.solutions_accepted}/{stats.solutions_found}",
            reward=f"{stats.total_reward_qnk:.6f} QUG",
            grover_runs=stats.grover_runs,
            avg_quantum_ms=f"{stats.avg_quantum_time_ms:.1f}",
            avg_vdf_ms=f"{stats.avg_vdf_time_ms:.1f}",
            quantum_advantage=f"{stats.quantum_advantage:.2f}x" if self._track_advantage else "n/a",
        )

    def get_summary(self) -> dict:
        """Return metrics as a dictionary for reporting."""
        uptime = time.monotonic() - self._start_time
        s = self.stats
        return {
            "uptime_seconds": uptime,
            "hashrate_hs": s.current_hashrate,
            "challenges_fetched": s.challenges_fetched,
            "solutions_found": s.solutions_found,
            "solutions_accepted": s.solutions_accepted,
            "solutions_rejected": s.solutions_rejected,
            "acceptance_rate": s.acceptance_rate,
            "total_reward_qnk": s.total_reward_qnk,
            "grover_runs": s.grover_runs,
            "classical_nonces_tried": s.classical_nonces_tried,
            "avg_quantum_time_ms": s.avg_quantum_time_ms,
            "avg_vdf_time_ms": s.avg_vdf_time_ms,
            "quantum_advantage": s.quantum_advantage,
        }
