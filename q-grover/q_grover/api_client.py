"""HTTP client for Q-NarwhalKnight mining API."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
import structlog

logger = structlog.get_logger()


@dataclass(frozen=True)
class MiningChallenge:
    """A mining challenge from the QNK node."""
    challenge_hash: str        # Hex-encoded 32-byte challenge
    difficulty_target: str     # Hex-encoded 32-byte target
    block_height: int          # Height to mine
    vdf_iterations: int        # VDF iteration count (informational; actual is always 100)
    block_reward: float        # QUG reward
    expires_at: str            # ISO timestamp
    server_version: str        # Server version
    network_hashrate_hs: float | None = None
    connected_miners: int | None = None
    min_miner_version: str | None = None
    fetched_at: float = 0.0    # Local monotonic timestamp when fetched

    @property
    def challenge_bytes(self) -> bytes:
        return bytes.fromhex(self.challenge_hash)

    @property
    def target_bytes(self) -> bytes:
        return bytes.fromhex(self.difficulty_target)

    def is_expired(self) -> bool:
        # Challenges have 300s server-side validity; use 280s client-side for safety
        return (time.monotonic() - self.fetched_at) > 280.0


@dataclass(frozen=True)
class SubmitResult:
    """Result of submitting a mining solution."""
    accepted: bool
    reward_qnk: float
    block_height: int
    message: str
    server_version: str
    update_available: bool = False


class QNKClient:
    """HTTP client for Q-NarwhalKnight mining endpoints."""

    def __init__(
        self,
        base_url: str = "https://quillon.xyz",
        fallback_urls: list[str] | None = None,
        timeout: float = 30.0,
        miner_version: str = "0.1.0",
    ):
        self._base_url = base_url.rstrip("/")
        self._fallback_urls = [u.rstrip("/") for u in (fallback_urls or [])]
        self._miner_version = miner_version
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={
                "User-Agent": f"q-grover/{miner_version}",
                "Accept": "application/json",
            },
            follow_redirects=True,
        )
        self._active_url = self._base_url

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _get_urls(self) -> list[str]:
        """Return active URL first, then fallbacks."""
        urls = [self._active_url]
        for u in [self._base_url] + self._fallback_urls:
            if u not in urls:
                urls.append(u)
        return urls

    def get_challenge(self) -> MiningChallenge:
        """Fetch current mining challenge from node.

        Tries primary URL first, then fallbacks on failure.
        """
        last_error = None
        for url in self._get_urls():
            try:
                resp = self._client.get(f"{url}/api/v1/mining/challenge")
                resp.raise_for_status()
                body = resp.json()

                # Handle both wrapped {"success":true,"data":{...}} and flat responses
                data = body.get("data", body) if isinstance(body, dict) else body

                challenge = MiningChallenge(
                    challenge_hash=data["challenge_hash"],
                    difficulty_target=data["difficulty_target"],
                    block_height=data["block_height"],
                    vdf_iterations=data.get("vdf_iterations", 100),
                    block_reward=data.get("block_reward", 0.0),
                    expires_at=data.get("expires_at", ""),
                    server_version=data.get("server_version", ""),
                    network_hashrate_hs=data.get("network_hashrate_hs"),
                    connected_miners=data.get("connected_miners"),
                    min_miner_version=data.get("min_miner_version"),
                    fetched_at=time.monotonic(),
                )
                self._active_url = url
                logger.debug(
                    "challenge_fetched",
                    height=challenge.block_height,
                    target=challenge.difficulty_target[:16] + "...",
                    url=url,
                )
                return challenge

            except (httpx.HTTPError, KeyError, ValueError) as e:
                last_error = e
                logger.warning("challenge_fetch_failed", url=url, error=str(e))
                continue

        raise ConnectionError(f"Failed to fetch challenge from all nodes: {last_error}")

    def submit_solution(
        self,
        *,
        miner_address: str,
        nonce: int,
        hash_hex: str,
        difficulty_target: str,
        challenge_hash: str,
        hash_rate: float | None = None,
        worker_name: str = "",
    ) -> SubmitResult:
        """Submit a mining solution to the node."""
        payload = {
            "miner_address": miner_address,
            "nonce": nonce,
            "hash": hash_hex,
            "difficulty_target": difficulty_target,
            "challenge_hash": challenge_hash,
            "miner_id": "q-grover",
            "miner_version": self._miner_version,
        }
        if hash_rate is not None:
            payload["hash_rate"] = hash_rate
        if worker_name:
            payload["worker_name"] = worker_name

        last_error = None
        for url in self._get_urls():
            try:
                resp = self._client.post(
                    f"{url}/api/v1/mining/submit",
                    json=payload,
                )
                resp.raise_for_status()
                body = resp.json()
                data = body.get("data", body) if isinstance(body, dict) else body

                result = SubmitResult(
                    accepted=data.get("accepted", False),
                    reward_qnk=data.get("reward_qnk", 0.0),
                    block_height=data.get("block_height", 0),
                    message=data.get("message", ""),
                    server_version=data.get("server_version", ""),
                    update_available=data.get("update_available", False),
                )
                self._active_url = url

                if result.accepted:
                    logger.info(
                        "solution_accepted",
                        reward=result.reward_qnk,
                        height=result.block_height,
                        nonce=nonce,
                    )
                else:
                    logger.warning(
                        "solution_rejected",
                        message=result.message,
                        nonce=nonce,
                    )
                return result

            except (httpx.HTTPError, KeyError, ValueError) as e:
                last_error = e
                logger.warning("submit_failed", url=url, error=str(e))
                continue

        raise ConnectionError(f"Failed to submit solution to all nodes: {last_error}")
