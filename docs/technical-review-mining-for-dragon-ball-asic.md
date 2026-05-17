# Technical Review: Q-NarwhalKnight Mining Architecture for Dragon Ball ASIC/FPGA

**Date:** 2026-04-13  
**Network:** Q-NarwhalKnight mainnet-genesis ($1B market cap, ~14.8M blocks)  
**Purpose:** Reference document for Dragon Ball Miner ASIC/FPGA design  
**Status:** BLAKE3 lane live; VDF lane in development; LWMA difficulty wired and pending activation

---

## 1. Current Mining Architecture

### 1.1 Mining Algorithm: BLAKE3 x 100

The primary mining algorithm is **100 sequential BLAKE3 hash operations** per nonce:

```
input = challenge_hash || nonce || miner_address
h₀ = BLAKE3(input)
h₁ = BLAKE3(h₀)
h₂ = BLAKE3(h₁)
...
h₉₉ = BLAKE3(h₉₈)
result = h₉₉
```

**Why BLAKE3 x 100 (not x 1)?**
- Prevents trivially cheap single-hash ASICs from flooding the network
- Adds ~100x memory bandwidth pressure per nonce (each hash = 64 bytes state)
- Sequential dependency: each round depends on the previous output
- Matches the Xcrypto `blake3.chain` instruction in the QUG-V1 RTL

### 1.2 Difficulty Target

A solution is valid if `result ≤ difficulty_target` where the target is a 256-bit hash with N leading zero bits.

**Current:** Fixed 16-bit difficulty (hardcoded)  
**After LWMA activation:** Dynamic, adjusting every block to target 1.0 blocks/second

Difficulty is expressed as **leading zero bits** in the target hash:
- 16 bits = `0x0000FFFF...` (current floor)
- 24 bits = `0x000000FF...` (harder)
- 32 bits = `0x00000000FF...` (much harder)

For ASIC/FPGA: the comparator is a simple byte-by-byte check against the target hash.

### 1.3 Mining Protocol (Server-Side)

Miners interact via HTTP REST API:

```
1. GET /api/v1/mining/challenge
   → { challenge_hash, difficulty_target, block_height, vdf_iterations, block_reward, expires_at }

2. Compute: for each nonce, hash BLAKE3 x 100, check against difficulty_target

3. POST /api/v1/mining/submit
   → { challenge_hash, nonce, hash, miner_address, ... }
```

**Challenge lifetime:** ~50 seconds (configurable via K-parameter gauge)  
**Challenge is deterministic:** Same height → same challenge on all nodes (consensus-bound)

### 1.4 Block Reward Distribution

**Phase A (deployed):** Difficulty-weighted rewards using leading zero bits.

Each miner's share is proportional to `2^(leading_zeros)`:
- Miner A finds hash with 16 leading zeros → weight = 2^16 = 65,536
- Miner B finds hash with 20 leading zeros → weight = 2^20 = 1,048,576
- Miner B gets 16x more reward than Miner A

**Implication for ASIC:** Finding harder solutions (more leading zeros than minimum) earns proportionally more reward. An ASIC that can evaluate more nonces per second will find harder solutions more often.

---

## 2. LWMA Difficulty Adjustment (Phase B.2 — Wired, Pending Activation)

### 2.1 Algorithm

LWMA (Linearly Weighted Moving Average) replaces the fixed 16-bit difficulty.

```
Window:         120 blocks
Weights:        Linear (1,2,3...120) — recent blocks weighted more
Solvetime:      Clamped to [1ms, 6×target] per block
Adjustment:     Clamped to [0.5×, 2.0×] per step
Floor:          16 leading zero bits minimum
Target:         1.0 blocks per second (1000ms block time)
```

**Pure function of chain state** — same inputs produce same output on every node. No background timer, no mutable state. Mirrors the emission controller pattern.

### 2.2 What This Means for ASIC Miners

| Parameter | Current (Fixed) | After LWMA |
|-----------|----------------|------------|
| Difficulty | Always 16 bits | 16+ bits, adjusts per block |
| Block rate | ~3.46 bps (unregulated) | Converges to ~1.0 bps |
| Revenue per hash | Higher (easy target) | Lower (harder target) but stable |
| Network hashrate response | None | Difficulty tracks hashrate within ~120 blocks |

**Key insight:** LWMA will increase difficulty from 16 to ~18-20 bits to bring block rate from 3.46 to 1.0 bps. A Dragon Ball ASIC entering the network will cause difficulty to rise proportionally to its added hashrate. The network self-regulates.

### 2.3 ASIC Design Consideration

The ASIC should be designed to handle **variable difficulty targets**. The target hash changes per challenge (every ~50 seconds). The ASIC must:

1. Accept a new 256-bit target from the host
2. Compare each candidate hash against the variable target
3. Report the solution with the lowest hash found (maximizes difficulty-weighted reward)

Do NOT hardcode the difficulty comparator.

---

## 3. Dual-Lane Mining (Phase C — In Development)

### 3.1 Architecture

After Phase C activation, blocks accept solutions from two independent lanes:

| Lane | Algorithm | Hardware | Reward Share |
|------|-----------|----------|-------------|
| **BLAKE3 (GPU/ASIC)** | BLAKE3 x 100 | GPU, FPGA, ASIC | 50% |
| **Genus-2 VDF (CPU)** | Jacobian doubling x T | CPU (sequential) | 50% |

### 3.2 BLAKE3 Lane (ASIC Target)

This is the lane Dragon Ball should target. The algorithm is unchanged:

```c
// Mining inner loop (per nonce)
void mine_nonce(uint8_t *challenge, uint64_t nonce, uint8_t *result) {
    uint8_t input[72]; // 32 (challenge) + 8 (nonce) + 32 (address)
    memcpy(input, challenge, 32);
    memcpy(input + 32, &nonce, 8);
    memcpy(input + 40, miner_address, 32);
    
    uint8_t h[32];
    blake3(input, 72, h);        // First hash
    for (int i = 1; i < 100; i++) {
        blake3(h, 32, h);        // Chain: each hash depends on previous
    }
    memcpy(result, h, 32);
}
```

**ASIC optimization opportunities:**
- Pipeline the 100 sequential BLAKE3 operations (14 stages × 100 = 1400 pipeline stages for full throughput)
- Or: single BLAKE3 core doing 100 iterations per nonce (latency-optimized)
- Nonce space is embarrassingly parallel: each nonce is independent
- Target comparison is trivial: leading-zero-byte check

### 3.3 BLAKE3 Core Specification

BLAKE3 compresses a 64-byte block using 7 rounds of column+diagonal operations on a 4x4 state matrix (16 × 32-bit words).

**Per round:**
- 8 quarter-round functions (G functions)
- Each G: 4 additions, 4 XORs, 4 rotations (right-rotate by 16, 12, 8, 7)
- Total per round: 32 additions + 32 XORs + 32 rotations

**Per compression (7 rounds):**
- 224 additions + 224 XORs + 224 rotations

**Per mining candidate (100 compressions):**
- 22,400 additions + 22,400 XORs + 22,400 rotations
- Plus: message schedule permutation (16-word lookup per round)

### 3.4 QUG-V1 RTL Reference (What's Already Built)

The `qug-v1-rtl/` directory contains a complete BLAKE3 accelerator:

| Component | File | Description |
|-----------|------|-------------|
| BLAKE3 round | `rtl/xcrypto/blake3_round.sv` | 2-stage pipelined round (G functions) |
| Xcrypto unit | `rtl/xcrypto/xcrypto_unit.sv` | FSM: init → rounds × 7 → finalize |
| RISC-V core | `rtl/core/qug_pipeline.sv` | 7-stage pipeline with Xcrypto dispatch |
| Memory | `rtl/memory/bram_sp.sv`, `bram_dp.sv` | Single/dual-port BRAM |

**Key instruction: `blake3.chain rd, rs1, rs2`**
- `rs1` = message address in memory
- `rs2[6:0]` = chain length (number of sequential hashes)
- For mining: `rs2 = 100` (BLAKE3 x 100 in hardware)
- Runs entirely in the Xcrypto pipeline — no CPU intervention per chain step

**FPGA estimates (single tile, Kintex-7 XC7K325T @ 100 MHz):**
- LUTs: ~20,700 (10.2%)
- DSPs: 12 (1.4%)
- BRAMs: 34 (7.6%)
- Headroom for 4-6 tiles per FPGA

### 3.5 VDF Lane (Not for ASIC — CPU Only)

The VDF lane uses Genus-2 hyperelliptic Jacobian doubling. Each step `D → 2D` is **inherently sequential** — a GPU or ASIC with 10,000 cores evaluates one step at the same speed as one CPU core. This lane is specifically designed to be ASIC-resistant.

**Dragon Ball should NOT implement the VDF lane in hardware.** The ASIC ceiling is ~2-5x (clock speed advantage only, no parallelism advantage). The ROI doesn't justify the silicon area.

---

## 4. What the Dragon Ball ASIC Needs to Implement

### 4.1 Core Mining Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dragon Ball Mining ASIC                       │
│                                                                  │
│  ┌──────────┐   ┌──────────────┐   ┌────────────┐              │
│  │  Nonce   │──→│  BLAKE3 x100 │──→│  Target    │──→ Solution  │
│  │Generator │   │  Chain Core   │   │ Comparator │              │
│  └──────────┘   └──────────────┘   └────────────┘              │
│                                                                  │
│  Inputs from host:                                              │
│  - challenge_hash [256 bits]                                    │
│  - miner_address [256 bits]                                     │
│  - difficulty_target [256 bits]  ← VARIABLE, changes per block  │
│  - nonce_range_start [64 bits]                                  │
│  - nonce_range_end [64 bits]                                    │
│                                                                  │
│  Outputs to host:                                               │
│  - best_nonce [64 bits]                                         │
│  - best_hash [256 bits]                                         │
│  - solutions_found [32 bits]                                    │
│  - hashes_per_second [32 bits]                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 BLAKE3 x 100 Chain Core

Two design approaches:

**Option A: Latency-optimized (single core, 100 iterations)**
```
Time per nonce = 100 × (7 rounds × 2 cycles) = 1,400 cycles
At 500 MHz: 1,400 / 500M = 2.8 μs per nonce
Throughput: ~357K hashes/sec per core
Area: ~21K LUTs (from QUG-V1 estimates)
```

**Option B: Throughput-optimized (pipelined, 100 stages)**
```
Pipeline: 100 BLAKE3 compression stages
Each stage: 7 rounds × 2 cycles = 14 cycles
Pipeline depth: 1,400 cycles
Throughput: 1 nonce per 14 cycles (once pipeline is full)
At 500 MHz: ~35.7M hashes/sec per pipeline
Area: ~2.1M LUTs (100x single core) — likely needs tiling
```

**Recommended: Option A with multiple cores.** More practical for ASIC/FPGA. Each core takes ~21K LUTs. A Kintex-7 XC7K325T (203K LUTs) fits ~8 cores. A dedicated ASIC at 7nm could fit hundreds.

### 4.3 Target Comparator

Simple 256-bit lexicographic comparison:

```verilog
// hash[0..31] <= target[0..31] → solution valid
wire solution_valid;
assign solution_valid = (hash < target);  // Big-endian byte comparison
```

**Important:** Also track the **best** (lowest) hash found. Submitting harder solutions earns more reward due to difficulty-weighted rewards.

### 4.4 Host Interface

The ASIC needs to communicate with a host CPU running the miner software:

| Interface | Purpose | Bandwidth |
|-----------|---------|-----------|
| PCIe Gen3 x4 | Challenge/solution exchange | Minimal (< 1 KB/s) |
| SPI/UART | Low-cost host interface | Sufficient for most setups |
| GPIO | Status LEDs, reset | Trivial |

Mining is bandwidth-light. The bottleneck is compute, not I/O.

---

## 5. Network Economics for ASIC Miners

### 5.1 Current State (Pre-LWMA)

| Metric | Value |
|--------|-------|
| Block rate | ~3.46 bps |
| Block reward | ~0.082 QUG (~Era 0 emission rate) |
| Difficulty | 16 bits (fixed) |
| Network hashrate | Estimated from block rate |
| Annual emission | 2,625,000 QUG |
| Market cap | ~$1B |

### 5.2 After LWMA Activation

| Metric | Value |
|--------|-------|
| Block rate | ~1.0 bps (regulated) |
| Block reward | ~0.082 QUG (emission rate unchanged) |
| Difficulty | Dynamic, 16+ bits |
| Revenue per hash | Decreases as more hashrate joins |
| Difficulty response time | ~120 blocks (~2 minutes) |

### 5.3 After Dual-Lane Activation (Phase C)

| Metric | BLAKE3 Lane | VDF Lane |
|--------|-------------|----------|
| Reward share | 50% | 50% |
| Hardware | GPU / FPGA / ASIC | CPU only |
| Difficulty adjustment | LWMA, 120-block window | LWMA, 240-block window |
| Dragon Ball target | **YES** | No |

### 5.4 Revenue Model

```
ASIC_revenue = (ASIC_hashrate / network_hashrate) × 0.50 × block_reward × blocks_per_day

Example:
- ASIC: 1 GH/s (BLAKE3 x100)
- Network: 10 GH/s total
- Block reward: 0.082 QUG
- Blocks/day: 86,400 (at 1 bps)
- Daily revenue: (1/10) × 0.50 × 0.082 × 86,400 = 354 QUG/day

Note: Difficulty-weighted rewards mean finding 20-bit solutions (vs 16-bit minimum)
earns 16x more per solution. Faster hardware finds more high-difficulty solutions.
```

---

## 6. Design Constraints for Dragon Ball

### 6.1 Must Support

| Requirement | Reason |
|-------------|--------|
| Variable 256-bit difficulty target | LWMA changes target per block |
| BLAKE3 x 100 sequential chain | The mining algorithm |
| 64-bit nonce space | 2^64 nonces per challenge |
| Best-hash tracking | Maximizes difficulty-weighted reward |
| Challenge refresh every ~50s | New challenge per block |

### 6.2 Should Support

| Feature | Reason |
|---------|--------|
| Multiple BLAKE3 cores | Parallel nonce evaluation |
| Low-power mode | For home miners (FPGA/small ASIC) |
| Firmware update | Algorithm parameters may change with upgrades |
| Hash rate reporting | Miner software needs real-time stats |

### 6.3 Does NOT Need

| Feature | Reason |
|---------|--------|
| Genus-2 VDF | CPU-only lane, ASIC has no advantage |
| Network stack | Host CPU handles HTTP |
| Wallet/signing | Host CPU handles authentication |
| Block validation | That's the full node's job |

---

## 7. Timeline Alignment

| Milestone | Q-NarwhalKnight | Dragon Ball |
|-----------|----------------|-------------|
| **Now** | Phase A deployed, LWMA wired | Evaluate QUG-V1 RTL |
| **Week 2-3** | LWMA activated on mainnet | Run Vivado synthesis on Kintex-7 |
| **Week 4-6** | Phase C (VDF lane) on canary | BLAKE3 x100 benchmark on FPGA |
| **Month 2-3** | Dual-lane mining activated | ASIC tape-out planning |
| **Month 6+** | Stable dual-lane operation | ASIC production |

---

## 8. References

- QUG-V1 RTL: `qug-v1-rtl/` (38 files, ~9,200 lines SystemVerilog)
- BLAKE3 specification: https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.pdf
- LWMA difficulty algorithm: `crates/q-mining/src/difficulty.rs`
- Mining fairness tests: `crates/q-mining/tests/mining_fairness_tests.rs` (34 tests)
- Emission controller: `crates/q-storage/src/emission_controller.rs`
- Zawy12 LWMA analysis: https://github.com/zawy12/difficulty-algorithms/issues/3
- Phase B-D technical review: `docs/technical-review-mining-phases-B-C-D-v2.md`
- FPGA Collaboration Proposal v3: `papers/dragon-ball-fpga-collaboration-v3.pdf`
