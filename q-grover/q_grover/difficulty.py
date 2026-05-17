"""Difficulty target analysis and qubit allocation.

Maps QNK difficulty targets to optimal quantum circuit parameters:
- How many qubits to allocate for nonce prefix
- How many Grover iterations to run
- Expected quantum advantage ratio
"""

from __future__ import annotations

import math


def leading_zero_bits(target_bytes: bytes) -> int:
    """Count leading zero bits in difficulty target.

    A target like 0x0000ffff... has 16 leading zero bits.
    More leading zeros = harder difficulty.
    """
    bits = 0
    for byte in target_bytes:
        if byte == 0:
            bits += 8
        else:
            # Count leading zeros in this byte
            bits += 8 - byte.bit_length()
            break
    return bits


def difficulty_from_target(target_bytes: bytes) -> float:
    """Convert difficulty target to a scalar difficulty value.

    Difficulty = 2^256 / target_value.
    Higher difficulty = harder to find valid nonce.
    """
    target_int = int.from_bytes(target_bytes, "big")
    if target_int == 0:
        return float("inf")
    max_target = (1 << 256) - 1
    return max_target / target_int


def search_space_size(target_bytes: bytes) -> int:
    """Estimate the nonce search space size for a given difficulty.

    Returns the expected number of classical hash attempts needed
    to find a valid nonce (on average).
    """
    diff = difficulty_from_target(target_bytes)
    return max(1, int(diff))


def optimal_prefix_qubits(
    target_bytes: bytes,
    max_qubits: int = 24,
    min_qubits: int = 8,
) -> int:
    """Calculate optimal number of qubits for nonce prefix search.

    Strategy: Use Grover to search a prefix subspace. The prefix
    determines which region of nonce space to explore classically.

    We want the prefix space to be large enough that at least one
    prefix has a high probability of containing a valid nonce when
    combined with classical suffix search.

    Args:
        target_bytes: 32-byte difficulty target.
        max_qubits: Maximum qubits available.
        min_qubits: Minimum qubits to use.

    Returns:
        Optimal number of qubits for prefix search.
    """
    zero_bits = leading_zero_bits(target_bytes)

    # For current QNK difficulty (16 leading zero bits), the expected
    # number of attempts is ~65536. With 20 prefix qubits (1M prefixes),
    # we get ~15 valid prefixes per suffix — good Grover amplification.
    #
    # Scale qubits with difficulty:
    # - Easy (8-12 zero bits): 8-12 qubits
    # - Medium (12-20 zero bits): 12-20 qubits
    # - Hard (20+ zero bits): 20-24 qubits
    qubits = min(max_qubits, max(min_qubits, zero_bits + 4))
    return qubits


def optimal_grover_iterations(num_qubits: int, num_solutions: int = 1) -> int:
    """Calculate optimal number of Grover iterations.

    For a search space of size N = 2^num_qubits with M solutions,
    the optimal number of iterations is approximately:
        (pi/4) * sqrt(N/M)

    More iterations than optimal DECREASES success probability.

    Args:
        num_qubits: Number of search qubits.
        num_solutions: Estimated number of valid solutions in search space.

    Returns:
        Optimal iteration count.
    """
    n = 2 ** num_qubits
    if num_solutions >= n:
        return 1
    ratio = n / max(1, num_solutions)
    iterations = int(math.pi / 4.0 * math.sqrt(ratio))
    return max(1, iterations)


def quantum_advantage_ratio(num_qubits: int) -> float:
    """Calculate the quantum speedup ratio.

    Grover provides quadratic speedup: O(sqrt(N)) vs O(N).
    For num_qubits = 20, N = 2^20 = 1,048,576:
    - Classical: ~1,048,576 attempts
    - Quantum: ~1,024 iterations (sqrt)
    - Speedup: ~1024x

    Returns:
        Theoretical speedup factor.
    """
    n = 2 ** num_qubits
    return math.sqrt(n)


def estimate_mining_time(
    target_bytes: bytes,
    classical_hashrate: float,
    num_qubits: int,
    quantum_overhead_factor: float = 10.0,
) -> dict[str, float]:
    """Estimate mining time for classical vs quantum approaches.

    Args:
        target_bytes: Difficulty target.
        classical_hashrate: Classical hashes per second.
        num_qubits: Qubits allocated for Grover search.
        quantum_overhead_factor: Overhead per Grover iteration vs classical hash.
            (QPanda simulation is ~10x slower than native BLAKE3)

    Returns:
        Dict with estimated times and speedup.
    """
    search_size = search_space_size(target_bytes)
    classical_time = search_size / max(1.0, classical_hashrate)

    grover_iters = optimal_grover_iterations(num_qubits)
    quantum_time = grover_iters * quantum_overhead_factor / max(1.0, classical_hashrate)

    return {
        "search_space": search_size,
        "classical_seconds": classical_time,
        "quantum_seconds": quantum_time,
        "grover_iterations": grover_iters,
        "speedup_ratio": classical_time / quantum_time if quantum_time > 0 else float("inf"),
        "prefix_qubits": num_qubits,
    }
