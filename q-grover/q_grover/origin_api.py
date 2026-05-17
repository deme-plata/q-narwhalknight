"""Origin Pilot quantum cloud API client.

Handles authentication, job submission, polling, and result retrieval
for Origin Pilot's quantum computing service. Jobs are submitted as
serialized circuit IR and executed on superconducting QPUs.

Quantum jobs are asynchronous: submit returns a job_id, then poll
until COMPLETED/FAILED/CANCELLED. Typical execution: 10s-120s.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import httpx
import structlog

logger = structlog.get_logger()


class JobState(str, Enum):
    """Origin Pilot job lifecycle states."""
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class QPUInfo:
    """Information about an available quantum processor."""
    name: str
    num_qubits: int
    status: str  # "online", "offline", "maintenance"
    native_gates: list[str]
    connectivity: str  # "grid", "heavy-hex", etc.
    avg_1q_fidelity: float
    avg_2q_fidelity: float
    queue_depth: int  # Jobs waiting


@dataclass(frozen=True)
class JobStatus:
    """Status of a submitted quantum job."""
    job_id: str
    state: JobState
    progress: float  # 0.0 to 1.0
    message: str = ""
    queue_position: int | None = None
    estimated_seconds: float | None = None


@dataclass(frozen=True)
class JobResult:
    """Result from a completed quantum job."""
    job_id: str
    counts: dict[str, int]  # Measurement results {bitstring: count}
    shots_completed: int
    execution_time_ms: float
    qpu_name: str


class OriginPilotError(Exception):
    """Base exception for Origin Pilot API errors."""


class AuthenticationError(OriginPilotError):
    """Failed to authenticate with Origin Pilot."""


class QPUOfflineError(OriginPilotError):
    """Target QPU is offline or in maintenance."""


class QueueFullError(OriginPilotError):
    """Job queue is full; try again later."""


class CircuitTooDeepError(OriginPilotError):
    """Circuit depth exceeds QPU decoherence limit."""


class JobTimeoutError(OriginPilotError):
    """Job did not complete within the polling timeout."""


class OriginPilotAPIClient:
    """Client for Origin Pilot quantum cloud API.

    Handles the full lifecycle: authenticate -> submit -> poll -> result.

    Usage:
        client = OriginPilotAPIClient(api_url="https://...", api_token="...")
        client.authenticate()
        qpus = client.list_qpus()
        job_id = client.submit_job(circuit_ir, shots=1024, qpu_name="wuyuan-72")
        result = client.wait_for_result(job_id, timeout=120)
    """

    # Rate limit: max requests per second
    _RATE_LIMIT_RPS = 5
    _MIN_REQUEST_INTERVAL = 1.0 / _RATE_LIMIT_RPS

    def __init__(
        self,
        api_url: str,
        api_token: str,
        timeout: float = 30.0,
    ):
        if not api_url:
            raise ValueError("api_url is required")
        if not api_token:
            raise ValueError("api_token is required")

        self._api_url = api_url.rstrip("/")
        self._api_token = api_token
        self._authenticated = False
        self._last_request_time = 0.0

        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
                "User-Agent": "q-grover/0.1.0",
            },
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._MIN_REQUEST_INTERVAL:
            time.sleep(self._MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make a rate-limited API request."""
        self._rate_limit()
        url = f"{self._api_url}{path}"

        try:
            resp = self._client.request(method, url, **kwargs)
        except httpx.ConnectError as e:
            raise OriginPilotError(f"Connection failed: {e}") from e
        except httpx.TimeoutException as e:
            raise OriginPilotError(f"Request timeout: {e}") from e

        if resp.status_code == 401:
            raise AuthenticationError("Invalid or expired API token")
        if resp.status_code == 429:
            raise QueueFullError("Rate limit exceeded; retry after backoff")
        if resp.status_code == 503:
            raise QPUOfflineError("Service unavailable")

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = resp.text[:500]
            raise OriginPilotError(f"HTTP {resp.status_code}: {body}") from e

        return resp.json()

    def authenticate(self) -> bool:
        """Verify API credentials and establish session.

        Returns:
            True if authentication succeeded.

        Raises:
            AuthenticationError: If credentials are invalid.
        """
        try:
            data = self._request("GET", "/api/v1/auth/verify")
            self._authenticated = True
            logger.info(
                "origin_pilot_authenticated",
                user=data.get("username", "unknown"),
            )
            return True
        except AuthenticationError:
            self._authenticated = False
            raise
        except OriginPilotError as e:
            logger.warning("origin_pilot_auth_failed", error=str(e))
            self._authenticated = False
            raise AuthenticationError(f"Authentication failed: {e}") from e

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    def list_qpus(self) -> list[QPUInfo]:
        """List available quantum processors.

        Returns:
            List of QPUInfo with status and capabilities.
        """
        data = self._request("GET", "/api/v1/qpus")
        qpus = []
        for item in data.get("qpus", []):
            qpus.append(QPUInfo(
                name=item["name"],
                num_qubits=item["num_qubits"],
                status=item.get("status", "unknown"),
                native_gates=item.get("native_gates", ["RZ", "SX", "CZ"]),
                connectivity=item.get("connectivity", "grid"),
                avg_1q_fidelity=item.get("avg_1q_fidelity", 0.995),
                avg_2q_fidelity=item.get("avg_2q_fidelity", 0.99),
                queue_depth=item.get("queue_depth", 0),
            ))

        logger.debug("origin_pilot_qpus", count=len(qpus))
        return qpus

    def submit_job(
        self,
        circuit_ir: list[dict],
        shots: int,
        qpu_name: str,
        *,
        num_qubits: int | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Submit a quantum circuit for execution.

        Args:
            circuit_ir: Circuit as list of gate instructions.
                Each gate: {"gate": "RZ", "qubits": [0], "params": [1.5707]}
            shots: Number of measurement shots.
            qpu_name: Target QPU name.
            num_qubits: Total qubits needed (auto-detected if None).
            metadata: Optional job metadata (tags, description).

        Returns:
            Job ID string for tracking.

        Raises:
            CircuitTooDeepError: Circuit too deep for QPU coherence time.
            QPUOfflineError: Target QPU is not available.
            QueueFullError: Job queue is at capacity.
        """
        payload = {
            "circuit": circuit_ir,
            "shots": shots,
            "qpu_name": qpu_name,
        }
        if num_qubits is not None:
            payload["num_qubits"] = num_qubits
        if metadata:
            payload["metadata"] = metadata

        try:
            data = self._request("POST", "/api/v1/jobs", json=payload)
        except OriginPilotError as e:
            err_str = str(e).lower()
            if "circuit depth" in err_str or "too deep" in err_str:
                raise CircuitTooDeepError(str(e)) from e
            if "offline" in err_str or "maintenance" in err_str:
                raise QPUOfflineError(str(e)) from e
            if "queue" in err_str and "full" in err_str:
                raise QueueFullError(str(e)) from e
            raise

        job_id = data["job_id"]
        logger.info(
            "origin_pilot_job_submitted",
            job_id=job_id,
            qpu=qpu_name,
            shots=shots,
            gates=len(circuit_ir),
        )
        return job_id

    def get_job_status(self, job_id: str) -> JobStatus:
        """Check the status of a submitted job.

        Args:
            job_id: Job ID from submit_job().

        Returns:
            Current job status.
        """
        data = self._request("GET", f"/api/v1/jobs/{job_id}/status")
        return JobStatus(
            job_id=job_id,
            state=JobState(data["state"]),
            progress=data.get("progress", 0.0),
            message=data.get("message", ""),
            queue_position=data.get("queue_position"),
            estimated_seconds=data.get("estimated_seconds"),
        )

    def get_job_result(self, job_id: str) -> JobResult:
        """Retrieve results of a completed job.

        Args:
            job_id: Job ID from submit_job().

        Returns:
            Measurement results.

        Raises:
            OriginPilotError: If job is not in COMPLETED state.
        """
        data = self._request("GET", f"/api/v1/jobs/{job_id}/result")

        if data.get("state") != "COMPLETED":
            raise OriginPilotError(
                f"Job {job_id} is not completed (state={data.get('state')})"
            )

        return JobResult(
            job_id=job_id,
            counts=data["counts"],
            shots_completed=data.get("shots_completed", sum(data["counts"].values())),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            qpu_name=data.get("qpu_name", ""),
        )

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job.

        Args:
            job_id: Job ID to cancel.

        Returns:
            True if cancellation was accepted.
        """
        try:
            self._request("POST", f"/api/v1/jobs/{job_id}/cancel")
            logger.info("origin_pilot_job_cancelled", job_id=job_id)
            return True
        except OriginPilotError as e:
            logger.warning("origin_pilot_cancel_failed", job_id=job_id, error=str(e))
            return False

    def wait_for_result(
        self,
        job_id: str,
        timeout: float = 300.0,
        initial_delay: float = 2.0,
        max_delay: float = 15.0,
    ) -> JobResult:
        """Poll until job completes, then return result.

        Uses exponential backoff: 2s, 4s, 8s, 15s, 15s, ...

        Args:
            job_id: Job ID to wait for.
            timeout: Maximum wait time in seconds.
            initial_delay: First poll delay in seconds.
            max_delay: Maximum poll delay in seconds.

        Returns:
            JobResult with measurement counts.

        Raises:
            JobTimeoutError: If timeout exceeded.
            OriginPilotError: If job failed or was cancelled.
        """
        start = time.monotonic()
        delay = initial_delay

        while True:
            status = self.get_job_status(job_id)

            if status.state == JobState.COMPLETED:
                return self.get_job_result(job_id)

            if status.state == JobState.FAILED:
                raise OriginPilotError(
                    f"Job {job_id} failed: {status.message}"
                )

            if status.state == JobState.CANCELLED:
                raise OriginPilotError(f"Job {job_id} was cancelled")

            elapsed = time.monotonic() - start
            if elapsed + delay > timeout:
                raise JobTimeoutError(
                    f"Job {job_id} did not complete within {timeout}s "
                    f"(state={status.state.value}, progress={status.progress:.0%})"
                )

            logger.debug(
                "origin_pilot_polling",
                job_id=job_id,
                state=status.state.value,
                progress=f"{status.progress:.0%}",
                queue_pos=status.queue_position,
                delay=f"{delay:.1f}s",
            )

            time.sleep(delay)
            delay = min(delay * 2, max_delay)
