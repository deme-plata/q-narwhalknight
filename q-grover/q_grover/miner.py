"""Main quantum mining loop.

Orchestrates the full mining pipeline:
1. Fetch challenge from QNK node
2. Build quantum oracle from challenge + difficulty
3. Run Grover search to find promising nonce prefixes
4. Classical suffix search on Grover-suggested prefixes
5. VDF computation on candidate nonces
6. Submit valid solutions to QNK node
7. Repeat

Supports both quantum (QPanda/Grover) and classical (brute-force) modes.
"""

from __future__ import annotations

import signal
import struct
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import structlog

from .api_client import MiningChallenge, QNKClient
from .config import Config
from .difficulty import optimal_prefix_qubits, optimal_grover_iterations
from .grover_engine import GroverConfig, GroverEngine
from .mev_shield import MEVShield
from .metrics import MetricsTracker
from .oracle import MiningOracle
from .vdf import VDF_ITERATIONS, check_difficulty, compute_vdf

logger = structlog.get_logger()


class QuantumMiner:
    """Main quantum mining engine.

    Connects Grover search with classical VDF verification
    and QNK network communication.
    """

    def __init__(self, config: Config):
        self.config = config
        self._running = False

        # Initialize components
        self._client = QNKClient(
            base_url=config.node.url,
            fallback_urls=config.node.fallback_urls,
            timeout=config.node.timeout,
        )

        grover_cfg = GroverConfig(
            num_qubits=config.quantum.prefix_qubits,
            shots=config.quantum.shots,
            iterations=config.quantum.grover_iterations,
            noise_adaptive=config.quantum.noise_adaptive,
            optimization_level=config.quantum.optimization_level,
            backend=config.quantum.backend,
        )
        self._grover = GroverEngine(grover_cfg)
        self._mev = MEVShield(
            enabled=config.mev.enabled,
            commit_reveal=config.mev.commit_reveal,
        )
        self._metrics = MetricsTracker(
            log_interval=config.metrics.log_interval,
            track_advantage=config.metrics.track_advantage,
        )

        # Initialize hardware backend for cloud execution
        self._hw_backend = None
        if config.quantum.backend == "origin_pilot":
            from .hardware import OriginPilotBackend
            self._hw_backend = OriginPilotBackend(
                api_url=config.quantum.origin_pilot.api_url,
                api_token=config.quantum.origin_pilot.api_token,
                qpu_name=config.quantum.origin_pilot.qpu_name
                or config.quantum.origin_pilot.qpu_preference,
            )

        self._current_challenge: MiningChallenge | None = None

    def start(self) -> None:
        """Start the mining loop."""
        self._running = True

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Initialize hardware backend (cloud or local)
        if self._hw_backend:
            hw_ok = self._hw_backend.initialize()
            if hw_ok and hasattr(self._hw_backend, 'is_cloud') and self._hw_backend.is_cloud:
                mode = "quantum (Origin Pilot cloud)"
            else:
                mode = "quantum (Origin Pilot simulator fallback)"
        else:
            hw_ok = False

        # Initialize Grover QVM (for local circuit execution)
        qvm_ok = self._grover.initialize()
        if not hw_ok:
            mode = "quantum (QPanda)" if qvm_ok else "classical fallback"
        logger.info(
            "miner_started",
            address=self.config.miner.address[:12] + "...",
            node=self.config.node.url,
            mode=mode,
            prefix_qubits=self.config.quantum.prefix_qubits,
        )

        try:
            self._mining_loop()
        finally:
            self._grover.shutdown()
            if self._hw_backend:
                self._hw_backend.shutdown()
            self._client.close()
            self._log_final_stats()

    def stop(self) -> None:
        """Signal the mining loop to stop."""
        self._running = False

    def _handle_signal(self, signum, frame):
        logger.info("shutdown_signal", signal=signum)
        self.stop()

    def _mining_loop(self) -> None:
        """Core mining loop: challenge → search → verify → submit."""
        while self._running:
            try:
                # Step 1: Fetch fresh challenge
                challenge = self._fetch_challenge()
                if challenge is None:
                    time.sleep(self.config.node.poll_interval)
                    continue

                # Step 2: Build oracle for this challenge
                oracle = self._build_oracle(challenge)

                # Step 3: Run Grover search for promising prefixes
                grover_start = time.monotonic()
                result = self._grover.search(oracle, cloud_backend=self._hw_backend)
                grover_elapsed = time.monotonic() - grover_start
                self._metrics.record_grover_run(grover_elapsed)

                # Feed measurement entropy to MEV shield
                if result.all_candidates:
                    self._mev.feed_quantum_entropy(result.all_candidates)

                # Step 4: Classical suffix search on Grover-suggested prefixes
                candidates = result.all_candidates or [(result.measured_prefix, result.probability)]
                solution = self._classical_suffix_search(challenge, candidates)

                if solution is not None:
                    nonce, hash_bytes = solution
                    self._metrics.record_solution_found()
                    self._mev.on_solution_found(nonce)

                    # Step 5: Submit solution
                    self._submit_solution(challenge, nonce, hash_bytes)

                # Log periodic metrics
                self._metrics.maybe_log()

                # Brief pause before next challenge if no solution found
                if solution is None and not challenge.is_expired():
                    time.sleep(0.1)

            except ConnectionError as e:
                logger.error("connection_error", error=str(e))
                time.sleep(5.0)
            except Exception as e:
                logger.error("mining_error", error=str(e), exc_info=True)
                time.sleep(2.0)

    def _fetch_challenge(self) -> MiningChallenge | None:
        """Fetch a new challenge, reusing current if still valid."""
        if self._current_challenge and not self._current_challenge.is_expired():
            return self._current_challenge

        try:
            challenge = self._client.get_challenge()
            self._metrics.record_challenge()
            self._current_challenge = challenge
            return challenge
        except ConnectionError:
            return None

    def _build_oracle(self, challenge: MiningChallenge) -> MiningOracle:
        """Construct a mining oracle for the given challenge."""
        prefix_qubits = optimal_prefix_qubits(
            challenge.target_bytes,
            max_qubits=self.config.quantum.prefix_qubits,
        )

        oracle = MiningOracle(
            challenge_bytes=challenge.challenge_bytes,
            target_bytes=challenge.target_bytes,
            prefix_qubits=prefix_qubits,
            suffix_samples=16,
        )
        oracle.build()
        return oracle

    def _classical_suffix_search(
        self,
        challenge: MiningChallenge,
        prefix_candidates: list[tuple[int, float]],
    ) -> tuple[int, bytes] | None:
        """Search suffix space for each Grover-suggested prefix.

        For each promising prefix from Grover, try random suffixes
        and compute the full VDF to check if the hash meets difficulty.

        Args:
            challenge: Current mining challenge.
            prefix_candidates: List of (prefix, probability) from Grover.

        Returns:
            (nonce, hash_bytes) if valid solution found, else None.
        """
        prefix_qubits = self.config.quantum.prefix_qubits
        suffix_bits = 64 - prefix_qubits
        max_suffix = (1 << suffix_bits) - 1
        nonces_tried = 0
        search_start = time.monotonic()

        # Try each Grover-suggested prefix
        for prefix, prob in prefix_candidates:
            if not self._running:
                break
            if challenge.is_expired():
                break

            # Sweep suffix space for this prefix
            # Start with random offset to avoid all miners trying same suffixes
            import random
            suffix_start = random.randint(0, max_suffix)

            suffixes_per_prefix = min(
                max_suffix + 1,
                max(1000, 2 ** min(suffix_bits, 16)),  # At least 1000, up to 64K
            )

            for i in range(suffixes_per_prefix):
                if not self._running:
                    break

                suffix = (suffix_start + i) & max_suffix
                nonce = ((prefix << suffix_bits) | suffix) & 0xFFFFFFFFFFFFFFFF

                # Compute full VDF
                vdf_start = time.monotonic()
                hash_bytes = compute_vdf(challenge.challenge_bytes, nonce)
                vdf_elapsed = time.monotonic() - vdf_start
                self._metrics.record_vdf(vdf_elapsed)
                nonces_tried += 1

                # Check difficulty
                if check_difficulty(hash_bytes, challenge.target_bytes):
                    search_elapsed = time.monotonic() - search_start
                    self._metrics.record_classical_search(nonces_tried, search_elapsed)
                    logger.info(
                        "solution_found",
                        nonce=nonce,
                        prefix=prefix,
                        nonces_tried=nonces_tried,
                        time_s=f"{search_elapsed:.2f}",
                    )
                    return (nonce, hash_bytes)

                # Update hashrate periodically
                if nonces_tried % 100 == 0:
                    self._metrics.stats.update_hashrate(100)

        search_elapsed = time.monotonic() - search_start
        self._metrics.record_classical_search(nonces_tried, search_elapsed)
        return None

    def _submit_solution(
        self,
        challenge: MiningChallenge,
        nonce: int,
        hash_bytes: bytes,
    ) -> None:
        """Submit a valid solution to the QNK node."""
        try:
            result = self._client.submit_solution(
                miner_address=self.config.miner.address,
                nonce=nonce,
                hash_hex=hash_bytes.hex(),
                difficulty_target=challenge.difficulty_target,
                challenge_hash=challenge.challenge_hash,
                hash_rate=self._metrics.stats.current_hashrate / 1000.0,  # Convert to KH/s
                worker_name=self.config.miner.worker_name,
            )

            if result.accepted:
                self._metrics.record_solution_accepted(result.reward_qnk)
                logger.info(
                    "block_mined",
                    reward=result.reward_qnk,
                    height=result.block_height,
                    total_reward=self._metrics.stats.total_reward_qnk,
                )
            else:
                self._metrics.record_solution_rejected()

            # Invalidate challenge after submission
            self._current_challenge = None

        except ConnectionError as e:
            logger.error("submit_connection_error", error=str(e))

    def _log_final_stats(self) -> None:
        """Log final statistics on shutdown."""
        summary = self._metrics.get_summary()
        logger.info(
            "miner_stopped",
            uptime_min=f"{summary['uptime_seconds'] / 60:.1f}",
            solutions=f"{summary['solutions_accepted']}/{summary['solutions_found']}",
            total_reward=f"{summary['total_reward_qnk']:.6f} QUG",
            hashrate=f"{summary['hashrate_hs']:.1f} H/s",
            grover_runs=summary['grover_runs'],
            quantum_advantage=f"{summary['quantum_advantage']:.2f}x",
        )
