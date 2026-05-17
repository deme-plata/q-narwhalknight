"""BLAKE3 VDF (Verifiable Delay Function) implementation.

Must exactly match Q-NarwhalKnight's verification algorithm:
1. hash_input = challenge_bytes(32) + nonce_le_bytes(8) = 40 bytes
2. initial = BLAKE3(hash_input)
3. Chain 100 iterations: current = BLAKE3(current)
4. Result must be < difficulty_target (lexicographic byte comparison)
"""

from __future__ import annotations

import struct

import blake3 as _blake3

# QNK always uses exactly 100 VDF iterations regardless of the vdf_iterations
# field in the challenge response. That field is informational/future use.
VDF_ITERATIONS = 100


def compute_vdf(challenge_bytes: bytes, nonce: int) -> bytes:
    """Compute the BLAKE3 VDF hash for a challenge+nonce pair.

    Args:
        challenge_bytes: 32-byte challenge hash from the node.
        nonce: 64-bit unsigned nonce value.

    Returns:
        32-byte final hash after VDF iterations.
    """
    # Step 1: Build 40-byte input (challenge || nonce_le)
    nonce_bytes = struct.pack("<Q", nonce)  # uint64 little-endian
    hash_input = challenge_bytes + nonce_bytes

    # Step 2: Initial BLAKE3 hash
    current = _blake3.blake3(hash_input).digest()

    # Step 3: Chain 100 BLAKE3 iterations
    for _ in range(VDF_ITERATIONS):
        current = _blake3.blake3(current).digest()

    return current


def check_difficulty(hash_bytes: bytes, target_bytes: bytes) -> bool:
    """Check if hash meets difficulty target.

    Args:
        hash_bytes: 32-byte hash result from VDF.
        target_bytes: 32-byte difficulty target.

    Returns:
        True if hash < target (lexicographic byte comparison).
    """
    return hash_bytes < target_bytes


def solve_nonce(
    challenge_bytes: bytes,
    target_bytes: bytes,
    start_nonce: int = 0,
    max_attempts: int = 0,
) -> tuple[int, bytes] | None:
    """Classical brute-force nonce search (baseline).

    Args:
        challenge_bytes: 32-byte challenge hash.
        target_bytes: 32-byte difficulty target.
        start_nonce: Starting nonce value.
        max_attempts: Max nonces to try (0 = unlimited).

    Returns:
        (nonce, hash) tuple if found, None if exhausted.
    """
    nonce = start_nonce
    attempts = 0
    while max_attempts == 0 or attempts < max_attempts:
        result = compute_vdf(challenge_bytes, nonce)
        if check_difficulty(result, target_bytes):
            return (nonce, result)
        nonce += 1
        attempts += 1
    return None


def compute_hashrate(
    challenge_bytes: bytes, duration_seconds: float = 1.0
) -> float:
    """Measure classical hash rate (hashes/second).

    Runs VDF computations for the given duration and returns the rate.
    """
    import time
    count = 0
    start = time.monotonic()
    nonce = 0
    while (time.monotonic() - start) < duration_seconds:
        compute_vdf(challenge_bytes, nonce)
        nonce += 1
        count += 1
    elapsed = time.monotonic() - start
    return count / elapsed if elapsed > 0 else 0.0
