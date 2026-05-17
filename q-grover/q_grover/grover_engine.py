"""QPanda Grover's algorithm implementation for quantum mining.

Implements the full Grover search circuit:
1. Initialize qubits in uniform superposition (Hadamard)
2. Apply oracle (marks valid nonce prefixes)
3. Apply diffusion operator (amplifies marked states)
4. Repeat optimal number of times
5. Measure to get candidate nonce prefix

Supports multiple backends:
- QPanda simulator (CPU/GPU) for development
- Origin Pilot quantum cloud
- Direct QPU connection
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import structlog

from .difficulty import optimal_grover_iterations
from .oracle import MiningOracle

logger = structlog.get_logger()

# Try to import QPanda
try:
    import pyqpanda as pq
    HAS_QPANDA = True
except ImportError:
    pq = None  # type: ignore[assignment]
    HAS_QPANDA = False


@dataclass
class GroverResult:
    """Result from a Grover search execution."""
    measured_prefix: int           # The nonce prefix found by Grover
    probability: float             # Measurement probability (from shots)
    iterations_used: int           # Number of Grover iterations applied
    shots: int                     # Number of measurement shots
    circuit_depth: int             # Total circuit depth
    execution_time_ms: float       # Wall-clock execution time
    backend: str                   # Backend used
    all_candidates: list[tuple[int, float]] = field(default_factory=list)
    # Top candidate prefixes sorted by measurement frequency


@dataclass
class GroverConfig:
    """Configuration for a Grover search run."""
    num_qubits: int = 20
    shots: int = 1024
    iterations: int = 0  # 0 = auto-calculate optimal
    noise_adaptive: bool = True
    optimization_level: int = 1
    backend: str = "simulator"


class GroverEngine:
    """Grover's algorithm engine using QPanda.

    Builds and executes quantum circuits for mining nonce search.
    """

    def __init__(self, config: GroverConfig | None = None):
        self.config = config or GroverConfig()
        self._qvm = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the quantum virtual machine.

        Returns True if QPanda is available and QVM initialized successfully.
        """
        if not HAS_QPANDA:
            logger.warning(
                "qpanda_unavailable",
                msg="pyqpanda not installed; using classical fallback",
            )
            return False

        try:
            self._qvm = pq.CPUQVM()
            self._qvm.init_qvm()
            self._initialized = True
            logger.info("qvm_initialized", backend=self.config.backend)
            return True
        except Exception as e:
            logger.error("qvm_init_failed", error=str(e))
            return False

    def shutdown(self) -> None:
        """Release quantum resources."""
        if self._qvm is not None:
            try:
                self._qvm.finalize()
            except Exception:
                pass
            self._qvm = None
            self._initialized = False

    def search(
        self,
        oracle: MiningOracle,
        *,
        cloud_backend: object | None = None,
    ) -> GroverResult:
        """Execute Grover search using the provided oracle.

        If cloud_backend is provided and in cloud mode, transpiles the
        circuit to native gates and submits to Origin Pilot.
        If QPanda is available locally, runs a real quantum circuit.
        Otherwise, falls back to classical simulation of Grover.

        Args:
            oracle: Mining oracle with marked prefixes.
            cloud_backend: OriginPilotBackend instance (if cloud execution desired).
        """
        # Check if cloud execution is requested
        if cloud_backend is not None:
            from .hardware import OriginPilotBackend
            if isinstance(cloud_backend, OriginPilotBackend) and cloud_backend.is_cloud:
                return self._search_cloud(oracle, cloud_backend)

        if self._initialized and HAS_QPANDA:
            return self._search_qpanda(oracle)
        return self._search_classical_fallback(oracle)

    def _search_cloud(self, oracle: MiningOracle, backend) -> GroverResult:
        """Run Grover search via Origin Pilot cloud transpilation."""
        from .transpiler import cap_prefix_qubits, transpile_full_grover

        start = time.monotonic()
        n = cap_prefix_qubits(self.config.num_qubits, is_real_qpu=True)
        num_marked = len(oracle.marked_prefixes)

        if self.config.iterations > 0:
            iterations = self.config.iterations
        else:
            iterations = optimal_grover_iterations(n, num_marked)

        # Reduce iterations for noisy QPU
        if self.config.noise_adaptive:
            iterations = max(1, iterations // 2)

        # Cap marked prefixes to those fitting in the qubit count
        max_prefix = (1 << n) - 1
        valid_marked = {p for p in oracle.marked_prefixes if p <= max_prefix}
        if not valid_marked:
            # No valid prefixes at reduced qubit count; use top ones
            valid_marked = {p & max_prefix for p in oracle.marked_prefixes}
            valid_marked.discard(0)
            if not valid_marked:
                valid_marked = {0}

        # Transpile to native gate set
        tc = transpile_full_grover(
            marked_prefixes=valid_marked,
            prefix_qubits=n,
            iterations=iterations,
        )

        logger.info(
            "grover_cloud_transpiled",
            qubits=n,
            iterations=iterations,
            native_gates=tc.gate_count,
            depth=tc.transpiled_depth,
        )

        # Submit to cloud
        circuit_ir = tc.to_ir()
        counts = backend.execute(circuit_ir, None, self.config.shots)

        # Parse cloud results
        candidates = self._parse_results(counts, n)
        elapsed_ms = (time.monotonic() - start) * 1000

        best_prefix, best_prob = candidates[0] if candidates else (0, 0.0)

        logger.info(
            "grover_cloud_complete",
            iterations=iterations,
            best_prefix=best_prefix,
            best_prob=f"{best_prob:.4f}",
            candidates=len(candidates),
            time_ms=f"{elapsed_ms:.1f}",
        )

        return GroverResult(
            measured_prefix=best_prefix,
            probability=best_prob,
            iterations_used=iterations,
            shots=self.config.shots,
            circuit_depth=tc.transpiled_depth,
            execution_time_ms=elapsed_ms,
            backend="origin_pilot",
            all_candidates=candidates[:20],
        )

    def _search_qpanda(self, oracle: MiningOracle) -> GroverResult:
        """Run Grover's algorithm using QPanda quantum circuit."""
        start = time.monotonic()
        n = self.config.num_qubits

        # Calculate optimal iterations
        num_marked = len(oracle.marked_prefixes)
        if self.config.iterations > 0:
            iterations = self.config.iterations
        else:
            iterations = optimal_grover_iterations(n, num_marked)

        # Noise-adaptive: reduce iterations on noisy hardware
        if self.config.noise_adaptive and self.config.backend != "simulator":
            # Noisy hardware benefits from fewer iterations
            iterations = max(1, iterations // 2)

        # Allocate qubits: n for search register + 1 ancilla
        qubits = self._qvm.qAlloc_many(n + 1)
        cbits = self._qvm.cAlloc_many(n)

        search_qubits = qubits[:n]
        ancilla = qubits[n]

        # Build the circuit
        prog = pq.QProg()

        # Initialize ancilla in |-> state for phase kickback
        prog << pq.X(ancilla) << pq.H(ancilla)

        # Put search register in uniform superposition
        for q in search_qubits:
            prog << pq.H(q)

        # Build oracle circuit
        oracle_circuit = oracle.build_qpanda_oracle(search_qubits, ancilla)
        if oracle_circuit is None:
            logger.error("oracle_build_failed")
            return self._search_classical_fallback(oracle)

        # Build diffusion operator
        diffusion = self._build_diffusion(search_qubits)

        # Apply Grover iterations
        for _ in range(iterations):
            prog << oracle_circuit  # Oracle: mark valid states
            prog << diffusion       # Diffusion: amplify marked states

        # Measure search register
        for i, q in enumerate(search_qubits):
            prog << pq.Measure(q, cbits[i])

        # Run with multiple shots
        result = self._qvm.run_with_configuration(prog, cbits, self.config.shots)

        # Parse measurement results
        candidates = self._parse_results(result, n)
        elapsed_ms = (time.monotonic() - start) * 1000

        best_prefix, best_prob = candidates[0] if candidates else (0, 0.0)

        logger.info(
            "grover_complete",
            iterations=iterations,
            best_prefix=best_prefix,
            best_prob=f"{best_prob:.4f}",
            candidates=len(candidates),
            time_ms=f"{elapsed_ms:.1f}",
        )

        return GroverResult(
            measured_prefix=best_prefix,
            probability=best_prob,
            iterations_used=iterations,
            shots=self.config.shots,
            circuit_depth=iterations * 2 + n + 2,  # Approximate
            execution_time_ms=elapsed_ms,
            backend="qpanda",
            all_candidates=candidates[:20],  # Top 20
        )

    def _build_diffusion(self, qubits: list) -> object:
        """Build the Grover diffusion operator.

        D = 2|s><s| - I where |s> = H^n|0>

        Implementation:
        1. Apply H to all qubits
        2. Apply X to all qubits
        3. Multi-controlled Z (flip phase of |111...1>)
        4. Apply X to all qubits
        5. Apply H to all qubits
        """
        circuit = pq.QCircuit()
        n = len(qubits)

        # H on all qubits
        for q in qubits:
            circuit << pq.H(q)

        # X on all qubits
        for q in qubits:
            circuit << pq.X(q)

        # Multi-controlled Z: apply Z to last qubit controlled by all others
        # Equivalent to: phase flip on |111...1> state
        if n == 1:
            circuit << pq.Z(qubits[0])
        elif n == 2:
            circuit << pq.CZ(qubits[0], qubits[1])
        else:
            # H-CNOT-H on last qubit = CZ; extend with controls
            circuit << pq.H(qubits[-1])
            # Multi-controlled X on last qubit
            gate = pq.X(qubits[-1])
            for ctrl in qubits[:-1]:
                gate = gate.control(ctrl)
            circuit << gate
            circuit << pq.H(qubits[-1])

        # X on all qubits
        for q in qubits:
            circuit << pq.X(q)

        # H on all qubits
        for q in qubits:
            circuit << pq.H(q)

        return circuit

    def _parse_results(
        self, result: dict, num_qubits: int
    ) -> list[tuple[int, float]]:
        """Parse QPanda measurement results into (prefix, probability) pairs.

        Args:
            result: QPanda measurement result dict {bitstring: count}.
            num_qubits: Number of search qubits.

        Returns:
            List of (prefix_int, probability) sorted by probability descending.
        """
        total_shots = sum(result.values())
        candidates = []
        for bitstring, count in result.items():
            # QPanda returns bitstrings like "00101..."
            prefix_int = int(bitstring, 2) if isinstance(bitstring, str) else int(bitstring)
            prob = count / total_shots if total_shots > 0 else 0.0
            candidates.append((prefix_int, prob))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _search_classical_fallback(self, oracle: MiningOracle) -> GroverResult:
        """Classical simulation of Grover's algorithm.

        When QPanda is not available, simulate the quantum search
        by directly using the oracle's marked prefixes. This gives
        the same result as a perfect quantum computer would.

        Used for development/testing without quantum hardware.
        """
        start = time.monotonic()
        n = self.config.num_qubits
        num_marked = len(oracle.marked_prefixes)

        iterations = optimal_grover_iterations(n, num_marked)

        # In perfect Grover, the marked prefixes are amplified.
        # Simulate by computing the final probability distribution.
        total_states = 2 ** n
        if num_marked == 0:
            elapsed_ms = (time.monotonic() - start) * 1000
            return GroverResult(
                measured_prefix=0,
                probability=0.0,
                iterations_used=iterations,
                shots=self.config.shots,
                circuit_depth=0,
                execution_time_ms=elapsed_ms,
                backend="classical_fallback",
            )

        # After k iterations of Grover, probability of measuring a marked state:
        # P(marked) = sin^2((2k+1) * theta)
        # where sin(theta) = sqrt(M/N)
        theta = math.asin(math.sqrt(num_marked / total_states))
        prob_marked = math.sin((2 * iterations + 1) * theta) ** 2

        # Each marked prefix gets equal probability
        prob_per_prefix = prob_marked / num_marked

        # Build candidate list
        candidates = [(p, prob_per_prefix) for p in sorted(oracle.marked_prefixes)]

        # "Measure": pick the highest-probability prefix (in practice, random)
        import random
        if candidates:
            measured = random.choices(
                [c[0] for c in candidates],
                weights=[c[1] for c in candidates],
                k=1,
            )[0]
        else:
            measured = 0

        elapsed_ms = (time.monotonic() - start) * 1000

        logger.info(
            "grover_classical_fallback",
            iterations=iterations,
            marked_count=num_marked,
            prob_marked=f"{prob_marked:.4f}",
            measured=measured,
            time_ms=f"{elapsed_ms:.1f}",
        )

        return GroverResult(
            measured_prefix=measured,
            probability=prob_per_prefix,
            iterations_used=iterations,
            shots=self.config.shots,
            circuit_depth=iterations * 2 + n + 2,
            execution_time_ms=elapsed_ms,
            backend="classical_fallback",
            all_candidates=candidates[:20],
        )
