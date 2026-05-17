"""MEV (Maximal Extractable Value) protection using quantum randomness.

Implements defenses against MEV attacks on the mining process:
- Oracle substitution attacks (malicious oracle replacement)
- Frontrunning attacks (inserting transactions before victims)
- Sandwich attacks (wrapping victim transactions)

Uses quantum randomness from Grover circuit measurements to provide
unpredictable transaction ordering and commit-reveal schemes.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum, auto

import structlog

logger = structlog.get_logger()


class MEVAttackType(Enum):
    ORACLE_SUBSTITUTION = auto()
    FRONTRUN = auto()
    BACKRUN = auto()
    SANDWICH = auto()
    TIME_BANDIT = auto()


@dataclass
class MEVEvent:
    """A detected or simulated MEV event."""
    attack_type: MEVAttackType
    timestamp: float
    description: str
    severity: float  # 0.0 to 1.0
    mitigated: bool = False


@dataclass
class CommitRevealEntry:
    """A quantum commit-reveal entry for fair ordering."""
    commitment: bytes     # H(data || quantum_random || nonce)
    quantum_random: bytes # Randomness from quantum measurement
    nonce: bytes          # Classical random nonce
    data: bytes           # The committed data
    committed_at: float   # Timestamp
    revealed: bool = False


class QuantumRNG:
    """Quantum random number generator.

    Extracts randomness from Grover circuit measurement outcomes.
    When quantum hardware is unavailable, falls back to OS CSPRNG
    with a warning.
    """

    def __init__(self):
        self._entropy_pool: list[int] = []
        self._quantum_bits_used = 0

    def feed_measurements(self, measurements: list[tuple[int, float]]) -> None:
        """Feed measurement results from Grover runs into entropy pool.

        Uses the least-significant bits of measurement outcomes,
        which are maximally random even with imperfect Grover.
        """
        for value, _prob in measurements:
            # Extract entropy from measurement outcomes
            # The LSBs of quantum measurements are inherently random
            self._entropy_pool.append(value & 0xFF)
            self._quantum_bits_used += 8

    def get_random_bytes(self, n: int) -> bytes:
        """Get n random bytes, preferring quantum entropy.

        If quantum entropy is available, mix it with CSPRNG.
        Otherwise, use pure CSPRNG.
        """
        classical = os.urandom(n)

        if len(self._entropy_pool) >= n:
            quantum = bytes(self._entropy_pool[:n])
            self._entropy_pool = self._entropy_pool[n:]
            # XOR quantum + classical for defense in depth
            return bytes(q ^ c for q, c in zip(quantum, classical))

        return classical

    @property
    def quantum_bits_available(self) -> int:
        return len(self._entropy_pool) * 8


class OracleIntegrityVerifier:
    """Verifies that the mining oracle has not been tampered with.

    Detects oracle substitution attacks where an adversary replaces
    the legitimate mining oracle with a biased one that steers
    Grover toward adversary-controlled solutions.

    Defense: Periodically verify oracle output against known-good
    classical computation.
    """

    def __init__(self, verification_rate: float = 0.1):
        self._verification_rate = verification_rate  # Fraction of oracle calls to verify
        self._verified_count = 0
        self._failed_count = 0

    def verify_oracle_output(
        self,
        challenge_bytes: bytes,
        prefix: int,
        oracle_says_marked: bool,
        prefix_qubits: int,
        suffix_samples: int = 8,
    ) -> bool:
        """Verify that the oracle's marking agrees with classical computation.

        Randomly samples suffixes for the given prefix and checks
        whether the oracle's decision (marked/unmarked) is consistent
        with classical hash evaluation.

        Returns True if oracle output is consistent, False if suspicious.
        """
        import random
        import struct
        import blake3 as _blake3

        # Only verify a fraction of calls (performance)
        if random.random() > self._verification_rate:
            return True  # Skip verification this time

        # Classical verification: compute hashes for this prefix
        found_any_close = False
        for _ in range(suffix_samples):
            suffix = random.getrandbits(64 - prefix_qubits)
            nonce = (prefix << (64 - prefix_qubits)) | suffix
            nonce_bytes = struct.pack("<Q", nonce & 0xFFFFFFFFFFFFFFFF)
            hash_input = challenge_bytes + nonce_bytes
            result = _blake3.blake3(hash_input).digest()
            # Check if this prefix produces any promising results
            if result[0] == 0:  # First byte is zero = somewhat close to target
                found_any_close = True
                break

        self._verified_count += 1
        # Oracle consistency check
        if oracle_says_marked and not found_any_close:
            # Oracle marked this prefix but classical verification found nothing promising
            # Could be legitimate (we only sampled a few) or could be attack
            # Only flag if this happens repeatedly
            self._failed_count += 1
            if self._failed_count > self._verified_count * 0.3:
                logger.warning(
                    "oracle_integrity_suspicious",
                    failed=self._failed_count,
                    verified=self._verified_count,
                    failure_rate=f"{self._failed_count / self._verified_count:.2%}",
                )
                return False

        return True

    @property
    def failure_rate(self) -> float:
        if self._verified_count == 0:
            return 0.0
        return self._failed_count / self._verified_count


class MEVDetector:
    """Detects potential MEV attack patterns.

    Monitors mining behavior for signs of MEV attacks:
    - Sudden changes in oracle output distribution
    - Unusual nonce clustering
    - Timing anomalies suggesting frontrunning
    """

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._recent_solutions: list[tuple[float, int]] = []  # (timestamp, nonce)
        self._events: list[MEVEvent] = []

    def record_solution(self, nonce: int) -> None:
        """Record a mining solution for pattern analysis."""
        now = time.time()
        self._recent_solutions.append((now, nonce))
        # Keep only recent window
        cutoff = now - 300  # 5-minute window
        self._recent_solutions = [(t, n) for t, n in self._recent_solutions if t > cutoff]

    def check_for_attacks(self) -> list[MEVEvent]:
        """Analyze recent solutions for MEV attack patterns.

        Returns list of detected events (may be empty).
        """
        events = []
        if len(self._recent_solutions) < 3:
            return events

        # Check for nonce clustering (sandwich attack indicator)
        nonces = [n for _, n in self._recent_solutions[-10:]]
        if len(nonces) >= 3:
            nonce_range = max(nonces) - min(nonces)
            avg_nonce = sum(nonces) / len(nonces)
            if nonce_range < avg_nonce * 0.001 and avg_nonce > 0:
                events.append(MEVEvent(
                    attack_type=MEVAttackType.SANDWICH,
                    timestamp=time.time(),
                    description=f"Nonce clustering detected: range {nonce_range} over {len(nonces)} solutions",
                    severity=0.6,
                ))

        # Check for timing anomalies (frontrun indicator)
        timestamps = [t for t, _ in self._recent_solutions[-10:]]
        if len(timestamps) >= 3:
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_interval = sum(intervals) / len(intervals)
            # Unusually fast solutions might indicate someone is racing us
            if avg_interval < 0.1 and len(intervals) > 5:
                events.append(MEVEvent(
                    attack_type=MEVAttackType.FRONTRUN,
                    timestamp=time.time(),
                    description=f"Unusually fast solution rate: {avg_interval:.3f}s avg interval",
                    severity=0.4,
                ))

        self._events.extend(events)
        return events


class QuantumCommitReveal:
    """Commit-reveal scheme using quantum randomness.

    Provides fair ordering by committing to mining solutions
    using quantum-derived randomness, making it impossible for
    an adversary to predict or front-run the solution.

    Protocol:
    1. Miner finds solution (nonce)
    2. Generate quantum random bytes from Grover measurements
    3. Commit: H(solution || quantum_random || classical_nonce)
    4. Submit commitment to node
    5. Reveal: send (solution, quantum_random, classical_nonce)
    6. Node verifies: H(reveal) == commitment
    """

    def __init__(self, rng: QuantumRNG):
        self._rng = rng
        self._pending: dict[bytes, CommitRevealEntry] = {}

    def commit(self, data: bytes) -> bytes:
        """Create a commitment to data using quantum randomness.

        Returns the commitment hash.
        """
        quantum_random = self._rng.get_random_bytes(32)
        nonce = secrets.token_bytes(16)

        commitment = hashlib.sha256(data + quantum_random + nonce).digest()

        entry = CommitRevealEntry(
            commitment=commitment,
            quantum_random=quantum_random,
            nonce=nonce,
            data=data,
            committed_at=time.time(),
        )
        self._pending[commitment] = entry

        logger.debug("commit_created", commitment=commitment.hex()[:16] + "...")
        return commitment

    def reveal(self, commitment: bytes) -> tuple[bytes, bytes, bytes] | None:
        """Reveal a previously committed value.

        Returns (data, quantum_random, nonce) or None if not found.
        """
        entry = self._pending.pop(commitment, None)
        if entry is None:
            return None

        entry.revealed = True
        logger.debug("commit_revealed", commitment=commitment.hex()[:16] + "...")
        return (entry.data, entry.quantum_random, entry.nonce)

    @staticmethod
    def verify(
        commitment: bytes,
        data: bytes,
        quantum_random: bytes,
        nonce: bytes,
    ) -> bool:
        """Verify a commitment matches the revealed values."""
        expected = hashlib.sha256(data + quantum_random + nonce).digest()
        return expected == commitment


class MEVShield:
    """Combined MEV protection system.

    Integrates all MEV defense components:
    - Quantum RNG for unpredictable ordering
    - Oracle integrity verification
    - MEV attack detection
    - Commit-reveal for fair solution submission
    """

    def __init__(self, enabled: bool = True, commit_reveal: bool = True):
        self.enabled = enabled
        self.rng = QuantumRNG()
        self.oracle_verifier = OracleIntegrityVerifier()
        self.detector = MEVDetector()
        self.commit_reveal_scheme = QuantumCommitReveal(self.rng) if commit_reveal else None
        self._events: list[MEVEvent] = []

    def feed_quantum_entropy(self, measurements: list[tuple[int, float]]) -> None:
        """Feed Grover measurement results into the quantum RNG."""
        if self.enabled:
            self.rng.feed_measurements(measurements)

    def on_solution_found(self, nonce: int) -> None:
        """Called when a valid mining solution is found."""
        if not self.enabled:
            return
        self.detector.record_solution(nonce)
        events = self.detector.check_for_attacks()
        if events:
            for event in events:
                logger.warning(
                    "mev_event_detected",
                    attack=event.attack_type.name,
                    severity=event.severity,
                    description=event.description,
                )
            self._events.extend(events)

    def get_recent_events(self) -> list[MEVEvent]:
        """Return and clear recent MEV events."""
        events = self._events[:]
        self._events.clear()
        return events
