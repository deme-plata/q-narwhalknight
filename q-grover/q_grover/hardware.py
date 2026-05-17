"""Hardware backend abstraction for quantum execution.

Supports multiple backends:
- SimulatorBackend: QPanda CPU/GPU simulator (development/testing)
- OriginPilotBackend: Origin Pilot quantum cloud API
- LocalQPUBackend: Direct QPanda hardware connection

All backends implement the same interface, allowing the miner
to switch between simulation and real quantum hardware transparently.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()

try:
    import pyqpanda as pq
    HAS_QPANDA = True
except ImportError:
    pq = None  # type: ignore[assignment]
    HAS_QPANDA = False


@dataclass
class HardwareInfo:
    """Information about the quantum hardware backend."""
    name: str
    backend_type: str  # "simulator", "cloud", "local_qpu"
    max_qubits: int
    native_gates: list[str]
    connectivity: str  # "all-to-all", "linear", "grid", etc.
    avg_fidelity: float  # 0.0 to 1.0, estimated gate fidelity
    is_available: bool


class HardwareBackend(abc.ABC):
    """Abstract base class for quantum hardware backends."""

    @abc.abstractmethod
    def info(self) -> HardwareInfo:
        """Return hardware capabilities and status."""

    @abc.abstractmethod
    def initialize(self) -> bool:
        """Initialize the backend. Returns True on success."""

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release hardware resources."""

    @abc.abstractmethod
    def allocate_qubits(self, count: int) -> list:
        """Allocate qubits for circuit execution."""

    @abc.abstractmethod
    def allocate_cbits(self, count: int) -> list:
        """Allocate classical bits for measurement."""

    @abc.abstractmethod
    def execute(self, program, cbits, shots: int) -> dict:
        """Execute a quantum program and return measurement results."""

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """Check if backend is initialized and ready."""


class SimulatorBackend(HardwareBackend):
    """QPanda CPU simulator backend.

    Uses QPanda's CPUQVM for full state-vector simulation.
    Supports up to ~30 qubits (limited by memory: 2^30 complex amplitudes).
    Perfect fidelity (no noise) — good for development and testing.
    """

    def __init__(self, max_qubits: int = 28):
        self._max_qubits = max_qubits
        self._qvm = None
        self._ready = False

    def info(self) -> HardwareInfo:
        return HardwareInfo(
            name="QPanda CPU Simulator",
            backend_type="simulator",
            max_qubits=self._max_qubits,
            native_gates=["H", "X", "Y", "Z", "CNOT", "CZ", "Toffoli", "RX", "RY", "RZ"],
            connectivity="all-to-all",
            avg_fidelity=1.0,
            is_available=HAS_QPANDA,
        )

    def initialize(self) -> bool:
        if not HAS_QPANDA:
            logger.warning("simulator_init_failed", reason="pyqpanda not installed")
            return False
        try:
            self._qvm = pq.CPUQVM()
            self._qvm.init_qvm()
            self._ready = True
            logger.info("simulator_initialized", max_qubits=self._max_qubits)
            return True
        except Exception as e:
            logger.error("simulator_init_error", error=str(e))
            return False

    def shutdown(self) -> None:
        if self._qvm:
            try:
                self._qvm.finalize()
            except Exception:
                pass
            self._qvm = None
        self._ready = False

    def allocate_qubits(self, count: int) -> list:
        if not self._qvm:
            raise RuntimeError("Simulator not initialized")
        return self._qvm.qAlloc_many(count)

    def allocate_cbits(self, count: int) -> list:
        if not self._qvm:
            raise RuntimeError("Simulator not initialized")
        return self._qvm.cAlloc_many(count)

    def execute(self, program, cbits, shots: int) -> dict:
        if not self._qvm:
            raise RuntimeError("Simulator not initialized")
        return self._qvm.run_with_configuration(program, cbits, shots)

    def is_ready(self) -> bool:
        return self._ready


class OriginPilotBackend(HardwareBackend):
    """Origin Pilot quantum cloud backend.

    Connects to Origin Pilot's quantum cloud service for execution
    on real superconducting quantum processors. Requires API credentials.

    This backend handles:
    - Circuit transpilation to native gate set {RZ, SX, CZ}
    - Qubit mapping to hardware topology
    - Asynchronous job submission and result polling
    - Fallback to local simulator if credentials missing or cloud unavailable
    """

    def __init__(self, api_url: str = "", api_token: str = "", qpu_name: str = ""):
        self._api_url = api_url
        self._api_token = api_token
        self._qpu_name = qpu_name
        self._ready = False
        self._cloud_mode = False
        self._api_client = None  # OriginPilotAPIClient when cloud mode
        self._qvm = None  # QPanda fallback when cloud unavailable

    def info(self) -> HardwareInfo:
        return HardwareInfo(
            name=f"Origin Pilot ({self._qpu_name or 'default'})",
            backend_type="cloud" if self._cloud_mode else "simulator",
            max_qubits=72 if self._cloud_mode else 28,
            native_gates=["RZ", "SX", "CZ"],
            connectivity="grid" if self._cloud_mode else "all-to-all",
            avg_fidelity=0.995 if self._cloud_mode else 1.0,
            is_available=self._ready,
        )

    def initialize(self) -> bool:
        if not self._api_url or not self._api_token:
            logger.warning(
                "origin_pilot_no_credentials",
                msg="API URL and token required; falling back to simulator",
            )
            return self._init_fallback()

        # Try cloud connection
        try:
            from .origin_api import OriginPilotAPIClient

            self._api_client = OriginPilotAPIClient(
                api_url=self._api_url,
                api_token=self._api_token,
            )
            self._api_client.authenticate()
            self._cloud_mode = True
            self._ready = True

            # Verify target QPU is available
            if self._qpu_name:
                qpus = self._api_client.list_qpus()
                qpu_names = [q.name for q in qpus]
                if self._qpu_name not in qpu_names:
                    available = ", ".join(qpu_names) or "none"
                    logger.warning(
                        "origin_pilot_qpu_not_found",
                        requested=self._qpu_name,
                        available=available,
                    )

            logger.info(
                "origin_pilot_cloud_initialized",
                qpu=self._qpu_name or "auto",
                cloud=True,
            )
            return True

        except Exception as e:
            logger.warning(
                "origin_pilot_cloud_failed",
                error=str(e),
                msg="Falling back to local simulator",
            )
            if self._api_client:
                self._api_client.close()
                self._api_client = None
            return self._init_fallback()

    def _init_fallback(self) -> bool:
        """Fall back to QPanda local simulator."""
        self._cloud_mode = False
        if not HAS_QPANDA:
            logger.warning("origin_pilot_fallback_failed", reason="pyqpanda not installed")
            return False
        self._qvm = pq.CPUQVM()
        self._qvm.init_qvm()
        self._ready = True
        logger.info("origin_pilot_simulator_fallback")
        return True

    def shutdown(self) -> None:
        if self._api_client:
            self._api_client.close()
            self._api_client = None
        if self._qvm:
            try:
                self._qvm.finalize()
            except Exception:
                pass
            self._qvm = None
        self._cloud_mode = False
        self._ready = False

    @property
    def is_cloud(self) -> bool:
        """True if using real Origin Pilot cloud, False if simulator fallback."""
        return self._cloud_mode

    def allocate_qubits(self, count: int) -> list:
        if self._cloud_mode:
            # Cloud mode: return placeholder indices (circuit built as IR)
            return list(range(count))
        if self._qvm:
            return self._qvm.qAlloc_many(count)
        raise RuntimeError("Origin Pilot backend not initialized")

    def allocate_cbits(self, count: int) -> list:
        if self._cloud_mode:
            return list(range(count))
        if self._qvm:
            return self._qvm.cAlloc_many(count)
        raise RuntimeError("Origin Pilot backend not initialized")

    def execute(self, program, cbits, shots: int) -> dict:
        """Execute a circuit.

        In cloud mode, `program` should be a list[dict] (transpiled IR).
        In fallback mode, uses QPanda QVM directly.
        """
        if self._cloud_mode:
            return self._execute_cloud(program, shots)
        if self._qvm:
            return self._qvm.run_with_configuration(program, cbits, shots)
        raise RuntimeError("Origin Pilot backend not initialized")

    def _execute_cloud(self, circuit_ir: list[dict], shots: int) -> dict:
        """Submit circuit IR to Origin Pilot and wait for results."""
        if not self._api_client:
            raise RuntimeError("Cloud API client not available")

        from .origin_api import JobTimeoutError

        # Determine QPU name
        qpu = self._qpu_name
        if not qpu:
            qpus = self._api_client.list_qpus()
            online = [q for q in qpus if q.status == "online"]
            if not online:
                raise RuntimeError("No QPU available on Origin Pilot")
            # Pick QPU with shortest queue
            qpu = min(online, key=lambda q: q.queue_depth).name

        # Count qubits used in circuit
        all_qubits = set()
        for gate in circuit_ir:
            all_qubits.update(gate.get("qubits", []))
        num_qubits = max(all_qubits) + 1 if all_qubits else 1

        # Submit job
        job_id = self._api_client.submit_job(
            circuit_ir=circuit_ir,
            shots=shots,
            qpu_name=qpu,
            num_qubits=num_qubits,
            metadata={"source": "q-grover", "type": "mining"},
        )

        # Poll for result (quantum jobs: 10s-120s typical)
        try:
            result = self._api_client.wait_for_result(
                job_id, timeout=300.0, initial_delay=2.0, max_delay=15.0
            )
        except JobTimeoutError:
            self._api_client.cancel_job(job_id)
            raise

        logger.info(
            "origin_pilot_execution_complete",
            job_id=job_id,
            qpu=result.qpu_name,
            shots=result.shots_completed,
            time_ms=f"{result.execution_time_ms:.1f}",
        )

        return result.counts

    def is_ready(self) -> bool:
        return self._ready


class LocalQPUBackend(HardwareBackend):
    """Direct local QPU backend via QPanda.

    For machines with a directly connected quantum processor
    (e.g., Origin Pilot OS running on a quantum computer).
    """

    def __init__(self):
        self._ready = False
        self._qvm = None

    def info(self) -> HardwareInfo:
        return HardwareInfo(
            name="Local QPU (QPanda)",
            backend_type="local_qpu",
            max_qubits=50,  # Depends on hardware
            native_gates=["RZ", "SX", "CZ"],
            connectivity="grid",
            avg_fidelity=0.99,
            is_available=HAS_QPANDA,
        )

    def initialize(self) -> bool:
        if not HAS_QPANDA:
            return False
        # TODO: Use QPanda's real hardware QVM when available
        # For now, fall back to CPUQVM
        try:
            self._qvm = pq.CPUQVM()
            self._qvm.init_qvm()
            self._ready = True
            logger.info("local_qpu_initialized", msg="Using CPUQVM as QPU placeholder")
            return True
        except Exception as e:
            logger.error("local_qpu_init_failed", error=str(e))
            return False

    def shutdown(self) -> None:
        if self._qvm:
            try:
                self._qvm.finalize()
            except Exception:
                pass
            self._qvm = None
        self._ready = False

    def allocate_qubits(self, count: int) -> list:
        if not self._qvm:
            raise RuntimeError("Local QPU not initialized")
        return self._qvm.qAlloc_many(count)

    def allocate_cbits(self, count: int) -> list:
        if not self._qvm:
            raise RuntimeError("Local QPU not initialized")
        return self._qvm.cAlloc_many(count)

    def execute(self, program, cbits, shots: int) -> dict:
        if not self._qvm:
            raise RuntimeError("Local QPU not initialized")
        return self._qvm.run_with_configuration(program, cbits, shots)

    def is_ready(self) -> bool:
        return self._ready


def create_backend(backend_type: str, **kwargs) -> HardwareBackend:
    """Factory function to create the appropriate backend.

    Args:
        backend_type: "simulator", "origin_pilot", or "local_qpu".
        **kwargs: Backend-specific configuration.

    Returns:
        Configured HardwareBackend instance.
    """
    if backend_type == "simulator":
        return SimulatorBackend(max_qubits=kwargs.get("max_qubits", 28))
    elif backend_type == "origin_pilot":
        return OriginPilotBackend(
            api_url=kwargs.get("api_url", ""),
            api_token=kwargs.get("api_token", ""),
            qpu_name=kwargs.get("qpu_name", ""),
        )
    elif backend_type == "local_qpu":
        return LocalQPUBackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
