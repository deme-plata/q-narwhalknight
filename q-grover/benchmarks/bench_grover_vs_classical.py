#!/usr/bin/env python3
"""Benchmark: Grover quantum search vs classical brute-force mining.

Compares:
1. Classical nonce throughput (BLAKE3 VDF hashes/second)
2. Grover search time (oracle build + quantum iterations)
3. Effective quantum advantage ratio

Run: python -m benchmarks.bench_grover_vs_classical
"""

from __future__ import annotations

import math
import time

from q_grover.difficulty import (
    optimal_grover_iterations,
    optimal_prefix_qubits,
    quantum_advantage_ratio,
)
from q_grover.grover_engine import GroverConfig, GroverEngine
from q_grover.oracle import MiningOracle
from q_grover.vdf import compute_hashrate, compute_vdf


def bench_classical_hashrate(duration: float = 5.0) -> float:
    """Measure classical BLAKE3 VDF hash rate."""
    print(f"\n{'='*60}")
    print("Classical BLAKE3 VDF Benchmark")
    print(f"{'='*60}")

    challenge = bytes(range(32))
    rate = compute_hashrate(challenge, duration)
    print(f"  Duration: {duration:.1f}s")
    print(f"  Hash rate: {rate:.1f} H/s ({rate/1000:.3f} KH/s)")
    return rate


def bench_oracle_construction(prefix_qubits: int = 12) -> float:
    """Measure oracle construction time."""
    print(f"\n{'='*60}")
    print(f"Oracle Construction Benchmark ({prefix_qubits} qubits)")
    print(f"{'='*60}")

    challenge = bytes(range(32))
    target = bytes([0x00, 0x00] + [0xFF] * 30)

    start = time.monotonic()
    oracle = MiningOracle(
        challenge_bytes=challenge,
        target_bytes=target,
        prefix_qubits=prefix_qubits,
        suffix_samples=16,
    )
    stats = oracle.build()
    elapsed = time.monotonic() - start

    print(f"  Prefixes: {stats.total_prefixes:,}")
    print(f"  Marked: {stats.promising_prefixes}")
    print(f"  Samples/prefix: {stats.samples_per_prefix}")
    print(f"  Build time: {elapsed:.3f}s")
    print(f"  Best score: {stats.best_score:.4f}")
    return elapsed


def bench_grover_search(prefix_qubits: int = 12) -> float:
    """Measure Grover search time (classical fallback)."""
    print(f"\n{'='*60}")
    print(f"Grover Search Benchmark ({prefix_qubits} qubits, classical fallback)")
    print(f"{'='*60}")

    challenge = bytes(range(32))
    target = bytes([0x00, 0x00] + [0xFF] * 30)

    oracle = MiningOracle(
        challenge_bytes=challenge,
        target_bytes=target,
        prefix_qubits=prefix_qubits,
        suffix_samples=16,
    )
    oracle.build()

    cfg = GroverConfig(num_qubits=prefix_qubits, shots=1024)
    engine = GroverEngine(cfg)

    start = time.monotonic()
    result = engine.search(oracle)
    elapsed = time.monotonic() - start

    print(f"  Iterations: {result.iterations_used}")
    print(f"  Best prefix: {result.measured_prefix}")
    print(f"  Probability: {result.probability:.4f}")
    print(f"  Search time: {elapsed * 1000:.1f}ms")
    print(f"  Backend: {result.backend}")
    return elapsed


def compare_advantage():
    """Compare quantum advantage across different qubit counts."""
    print(f"\n{'='*60}")
    print("Quantum Advantage Comparison")
    print(f"{'='*60}")
    print(f"  {'Qubits':>8} {'Search Space':>14} {'Grover Iters':>14} {'Advantage':>12}")
    print(f"  {'-'*8} {'-'*14} {'-'*14} {'-'*12}")

    for qubits in [8, 12, 16, 20, 24]:
        n = 2 ** qubits
        iters = optimal_grover_iterations(qubits, 1)
        advantage = quantum_advantage_ratio(qubits)
        print(f"  {qubits:>8} {n:>14,} {iters:>14,} {advantage:>11.0f}x")


def full_pipeline_benchmark():
    """Benchmark the full mining pipeline (oracle + grover + classical verify)."""
    print(f"\n{'='*60}")
    print("Full Mining Pipeline Benchmark")
    print(f"{'='*60}")

    challenge = bytes(range(32))
    target = bytes([0x00] + [0xFF] * 31)  # Easy difficulty for testing

    for qubits in [8, 10, 12]:
        print(f"\n  --- {qubits} prefix qubits ---")

        # Oracle construction
        t0 = time.monotonic()
        oracle = MiningOracle(
            challenge_bytes=challenge,
            target_bytes=target,
            prefix_qubits=qubits,
            suffix_samples=8,
        )
        oracle.build()
        t_oracle = time.monotonic() - t0

        # Grover search
        cfg = GroverConfig(num_qubits=qubits, shots=512)
        engine = GroverEngine(cfg)
        t1 = time.monotonic()
        result = engine.search(oracle)
        t_grover = time.monotonic() - t1

        # Classical suffix search (limited)
        t2 = time.monotonic()
        prefix = result.measured_prefix
        suffix_bits = 64 - qubits
        found = False
        nonces_tried = 0
        for suffix in range(min(10000, 2**suffix_bits)):
            nonce = ((prefix << suffix_bits) | suffix) & 0xFFFFFFFFFFFFFFFF
            h = compute_vdf(challenge, nonce)
            nonces_tried += 1
            if h < target:
                found = True
                break
        t_classical = time.monotonic() - t2

        total = t_oracle + t_grover + t_classical
        print(f"  Oracle build:    {t_oracle*1000:>8.1f}ms")
        print(f"  Grover search:   {t_grover*1000:>8.1f}ms")
        print(f"  Classical verify: {t_classical*1000:>8.1f}ms ({nonces_tried} nonces)")
        print(f"  Total pipeline:  {total*1000:>8.1f}ms")
        print(f"  Solution found:  {found}")


if __name__ == "__main__":
    classical_rate = bench_classical_hashrate(duration=3.0)
    bench_oracle_construction(prefix_qubits=10)
    bench_grover_search(prefix_qubits=10)
    compare_advantage()
    full_pipeline_benchmark()

    print(f"\n{'='*60}")
    print("Benchmark Complete")
    print(f"{'='*60}")
