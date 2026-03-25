"""Quantum oracle construction for mining.

The oracle marks nonce prefixes that are likely to yield valid solutions
when combined with classical suffix search. This is a hybrid approach:

Phase 1 (current): Statistical oracle
  - Pre-sample random suffixes for each prefix
  - Mark prefixes where sampled hashes trend closer to target
  - Grover amplifies these "promising" prefixes

Phase 2 (future): Partial BLAKE3 quantum circuit
  - Encode first 4-8 rounds of BLAKE3 as reversible quantum gates
  - Directly evaluate hash prefix in superposition

Phase 3 (research): Full BLAKE3 quantum oracle
  - Complete BLAKE3 as quantum circuit (~10K+ qubits needed)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import blake3 as _blake3
import structlog

logger = structlog.get_logger()

# Try to import QPanda; fall back to stub for testing without quantum deps
try:
    import pyqpanda as pq
    HAS_QPANDA = True
except ImportError:
    pq = None  # type: ignore[assignment]
    HAS_QPANDA = False


@dataclass
class OracleStats:
    """Statistics from oracle construction."""
    prefix_qubits: int
    total_prefixes: int
    promising_prefixes: int
    samples_per_prefix: int
    best_score: float = 0.0
    avg_score: float = 0.0


@dataclass
class MiningOracle:
    """Hybrid classical-quantum mining oracle.

    For each nonce prefix, samples random suffixes and scores how
    close the resulting hashes are to the difficulty target. Prefixes
    with lower (closer to target) hash outputs are marked as "promising"
    and amplified by Grover's algorithm.
    """

    challenge_bytes: bytes
    target_bytes: bytes
    prefix_qubits: int = 20
    suffix_samples: int = 16
    # Bit patterns that the oracle marks as "good"
    _marked_prefixes: set[int] = field(default_factory=set, repr=False)
    _stats: OracleStats | None = field(default=None, repr=False)

    def build(self) -> OracleStats:
        """Construct the oracle by sampling the nonce space.

        For each possible prefix (2^prefix_qubits values), sample
        `suffix_samples` random suffixes and compute the VDF hash.
        Score each prefix by how close its best hash is to the target.
        Mark the top ~sqrt(N) prefixes as promising.
        """
        import math
        import random

        num_prefixes = 2 ** self.prefix_qubits
        # We want to mark ~sqrt(N) prefixes for optimal Grover
        # But also cap at a reasonable fraction
        mark_count = max(1, int(math.sqrt(num_prefixes)))

        target_int = int.from_bytes(self.target_bytes, "big")

        # Score each prefix: lower score = closer to target = better
        prefix_scores: list[tuple[int, float]] = []

        for prefix in range(num_prefixes):
            best_score = float("inf")
            prefix_bytes = prefix.to_bytes((self.prefix_qubits + 7) // 8, "big")

            for _ in range(self.suffix_samples):
                # Generate random suffix bits
                suffix = random.getrandbits(64 - self.prefix_qubits)
                # Combine prefix + suffix into full nonce
                nonce = (prefix << (64 - self.prefix_qubits)) | suffix
                nonce_bytes = struct.pack("<Q", nonce & 0xFFFFFFFFFFFFFFFF)

                # Compute BLAKE3(challenge || nonce) — just the initial hash
                # (not full VDF — too expensive for oracle construction)
                hash_input = self.challenge_bytes + nonce_bytes
                result = _blake3.blake3(hash_input).digest()
                result_int = int.from_bytes(result, "big")

                # Score: distance to target (lower = better)
                score = result_int / max(1, target_int)
                best_score = min(best_score, score)

            prefix_scores.append((prefix, best_score))

        # Sort by score (best first) and mark top candidates
        prefix_scores.sort(key=lambda x: x[1])
        self._marked_prefixes = {p for p, _ in prefix_scores[:mark_count]}

        scores = [s for _, s in prefix_scores]
        self._stats = OracleStats(
            prefix_qubits=self.prefix_qubits,
            total_prefixes=num_prefixes,
            promising_prefixes=mark_count,
            samples_per_prefix=self.suffix_samples,
            best_score=prefix_scores[0][1] if prefix_scores else 0.0,
            avg_score=sum(scores) / len(scores) if scores else 0.0,
        )

        logger.info(
            "oracle_built",
            prefixes=num_prefixes,
            marked=mark_count,
            best_score=f"{self._stats.best_score:.4f}",
        )
        return self._stats

    @property
    def stats(self) -> OracleStats | None:
        return self._stats

    @property
    def marked_prefixes(self) -> set[int]:
        return self._marked_prefixes

    def is_marked(self, prefix: int) -> bool:
        """Check if a prefix is marked as promising by the oracle."""
        return prefix in self._marked_prefixes

    def get_marked_list(self) -> list[int]:
        """Return sorted list of marked prefix values."""
        return sorted(self._marked_prefixes)

    def build_qpanda_oracle(self, qubits: list, ancilla) -> object | None:
        """Build a QPanda quantum circuit implementing this oracle.

        The oracle flips the ancilla qubit for nonce prefixes that
        are in the marked set.

        For Phase 1, this is a "lookup-table" oracle: we explicitly
        encode which prefixes to mark using multi-controlled X gates.

        Args:
            qubits: List of QPanda qubit objects (prefix register).
            ancilla: Single QPanda qubit for phase kickback.

        Returns:
            QPanda QCircuit implementing the oracle, or None if QPanda unavailable.
        """
        if not HAS_QPANDA:
            logger.warning("qpanda_not_available", msg="Cannot build quantum oracle without pyqpanda")
            return None

        circuit = pq.QCircuit()

        for prefix in self._marked_prefixes:
            # Encode prefix as a multi-controlled X gate
            # For prefix bits: apply X to qubits where bit is 0 (to match |prefix>)
            # Then multi-controlled X on ancilla
            # Then undo the X gates

            flip_indices = []
            for i in range(self.prefix_qubits):
                bit = (prefix >> (self.prefix_qubits - 1 - i)) & 1
                if bit == 0:
                    circuit << pq.X(qubits[i])
                    flip_indices.append(i)

            # Multi-controlled X: flip ancilla when all qubits are |1>
            if len(qubits) > 1:
                controls = [qubits[i] for i in range(self.prefix_qubits)]
                # Use Toffoli decomposition for multi-controlled gate
                circuit << _multi_controlled_x(controls, ancilla)
            else:
                circuit << pq.CNOT(qubits[0], ancilla)

            # Undo the X flips
            for i in flip_indices:
                circuit << pq.X(qubits[i])

        return circuit


def _multi_controlled_x(controls: list, target) -> object:
    """Build a multi-controlled X gate using QPanda.

    For n controls, decomposes into O(n) Toffoli gates using
    ancilla qubits. QPanda's built-in decomposition handles this.
    """
    if len(controls) == 1:
        return pq.CNOT(controls[0], target)
    elif len(controls) == 2:
        return pq.Toffoli(controls[0], controls[1], target)
    else:
        # QPanda supports multi-controlled gates natively
        circuit = pq.QCircuit()
        # Use recursive decomposition with work qubits
        # For now, use QPanda's built-in control mechanism
        gate = pq.X(target)
        for ctrl in controls:
            gate = gate.control(ctrl)
        circuit << gate
        return circuit
