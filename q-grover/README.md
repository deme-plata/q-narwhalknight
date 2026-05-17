# q-grover

**Quantum Grover miner for Q-NarwhalKnight** — the first quantum mining application targeting Origin Pilot OS / QPanda.

Uses Grover's algorithm to achieve quadratic speedup in mining nonce search, connecting to the live QNK blockchain network.

## How It Works

```
1. Fetch challenge  ──► GET /api/v1/mining/challenge
2. Build oracle     ──► Score nonce prefixes via sampling
3. Grover search    ──► QPanda circuit amplifies promising prefixes
4. Classical verify  ──► BLAKE3 VDF on Grover-suggested nonces
5. Submit solution  ──► POST /api/v1/mining/submit
```

**Quantum advantage:** Grover's algorithm searches the nonce prefix space in O(sqrt(N)) iterations instead of O(N), providing up to 1024x speedup with 20 qubits.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Optional: install QPanda for quantum execution
pip install pyqpanda

# Check node status
q-grover status --node https://quillon.xyz

# Benchmark your hardware
q-grover benchmark

# Start mining (classical fallback if no QPanda)
q-grover mine --address qnkYOUR_WALLET_ADDRESS_HERE --node https://quillon.xyz

# Mine with specific qubit count
q-grover mine --address qnk... --qubits 16

# Force classical-only mode
q-grover mine --address qnk... --classical
```

## Configuration

Copy `q-grover.toml` to your working directory or `~/.config/q-grover/config.toml`:

```toml
[node]
url = "https://quillon.xyz"

[miner]
address = "qnk..."  # Your wallet address
worker_name = "my-quantum-miner"

[quantum]
backend = "simulator"    # simulator, origin_pilot, local_qpu
prefix_qubits = 20       # Qubits for Grover search
shots = 1024              # Measurement shots per run
```

Environment variable overrides:
- `Q_GROVER_ADDRESS` — wallet address
- `Q_GROVER_NODE_URL` — node URL
- `Q_GROVER_BACKEND` — quantum backend
- `Q_GROVER_QUBITS` — prefix qubits

## Architecture

### Hybrid Quantum-Classical Mining

Since full BLAKE3 as a quantum circuit requires thousands of qubits (beyond current hardware), we use a **hybrid approach**:

1. **Nonce splitting:** 64-bit nonce = prefix (20 bits) + suffix (44 bits)
2. **Quantum search:** Grover searches the prefix space (2^20 = 1M candidates)
3. **Classical verification:** For each Grover-suggested prefix, classically try suffixes
4. **VDF computation:** 100 iterations of BLAKE3 chaining (always classical)

This provides sqrt(2^20) = 1024x fewer quantum iterations compared to brute-force prefix search.

### Mining Protocol (must match QNK server exactly)

```python
# 1. Hash input: challenge(32 bytes) + nonce(8 bytes little-endian) = 40 bytes
hash_input = challenge_bytes + nonce.to_bytes(8, 'little')

# 2. Initial BLAKE3 hash
current = blake3(hash_input)

# 3. VDF: 100 iterations of BLAKE3 chaining
for _ in range(100):
    current = blake3(current)

# 4. Check: current < difficulty_target (lexicographic bytes)
```

### MEV Protection

Built-in defenses against Maximal Extractable Value attacks:
- **Quantum RNG** from Grover measurement entropy
- **Oracle integrity verification** — detect tampered oracles
- **Commit-reveal scheme** — quantum-random commitments
- **Attack pattern detection** — nonce clustering, timing anomalies

## Project Structure

```
q-grover/
  q_grover/
    cli.py              # Command-line interface
    miner.py            # Main mining loop
    grover_engine.py    # QPanda Grover circuit builder
    oracle.py           # Mining oracle construction
    vdf.py              # BLAKE3 VDF (must match QNK server)
    difficulty.py       # Difficulty analysis + qubit allocation
    api_client.py       # HTTP client for QNK mining API
    mev_shield.py       # MEV protection
    hardware.py         # QPanda backend abstraction
    metrics.py          # Performance tracking
    config.py           # TOML configuration
  tests/                # pytest test suite
  benchmarks/           # Performance benchmarks
```

## Requirements

- Python >= 3.10
- `blake3` — BLAKE3 hashing (must match QNK server)
- `httpx` — HTTP client
- `pyqpanda` — QPanda quantum framework (optional; falls back to classical)

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Benchmarks

```bash
python -m benchmarks.bench_grover_vs_classical
```

## Origin Pilot OS

q-grover is designed to run natively on Origin Pilot quantum OS:

1. Install Origin Pilot OS (based on QPanda framework)
2. `pip install q-grover[qpanda]`
3. Configure `quantum.backend = "origin_pilot"` in config
4. Set Origin Pilot API credentials
5. Mine with real quantum hardware acceleration

## License

MIT
