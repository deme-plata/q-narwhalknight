# Dual-Lane Mining: CPU-Fair Sequential Work Alongside GPU Proof-of-Work

**Q-NarwhalKnight Consensus Enhancement Proposal**

*Demetri et al.*  
*April 2026*  
*Version 1.0 — Community Draft*

---

## Abstract

We propose a dual-lane mining architecture for the Q-NarwhalKnight blockchain that allocates block rewards equally between two independent computational lanes: a traditional BLAKE3 proof-of-work lane optimized for parallel hardware (GPUs, FPGAs, ASICs), and a novel Genus-2 Verifiable Delay Function (VDF) lane that is inherently sequential and resistant to parallelization. The VDF lane ensures that CPU miners — including consumer laptops and desktops — can earn meaningful mining rewards regardless of the GPU hashrate on the network. We describe the mathematical foundations, the economic model, the difficulty adjustment mechanism for each lane, and the security properties of the system. The design has been peer-reviewed by independent AI systems (DeepSeek, ChatGPT) and incorporates their feedback.

---

## 1. Introduction

### 1.1 The Problem: GPU Dominance in Proof-of-Work

Proof-of-work blockchains face an inherent centralization pressure: miners who invest in specialized hardware (GPUs, FPGAs, ASICs) can evaluate hash functions orders of magnitude faster than consumer CPUs. This creates an economic moat where only well-capitalized miners can participate profitably, contradicting the original vision of "one CPU, one vote" [1].

Q-NarwhalKnight currently uses BLAKE3 x 100 (100 sequential BLAKE3 hash operations per nonce) as its proof-of-work function. While the 100-round chain adds sequential depth, the nonce space remains embarrassingly parallel — a GPU with 10,000 CUDA cores evaluates 10,000 nonces simultaneously, giving it a ~10,000x throughput advantage over a single CPU core.

### 1.2 The Solution: A Reserved Lane for Sequential Work

Rather than replacing GPU mining (which provides network security through hashrate), we propose **adding** a second mining lane that is fundamentally incompatible with parallel hardware acceleration. This lane uses a Verifiable Delay Function (VDF) based on Genus-2 hyperelliptic curve Jacobian arithmetic — a computation where each step depends on the output of the previous step, making parallelism useless.

The dual-lane architecture:
- Preserves existing GPU mining (BLAKE3 lane) with 50% of block reward
- Adds CPU-fair mining (VDF lane) with 50% of block reward
- Requires no changes to existing mining hardware or software
- Activates via height-gated consensus upgrade (backward compatible)

### 1.3 Design Principles

1. **Additive, not replacement:** GPU miners keep their lane. CPU miners get a new one.
2. **Economically self-balancing:** If a lane is undermined, its rewards accumulate, attracting participants.
3. **Cryptographically sound:** VDF sequential work is provable and efficiently verifiable.
4. **Minimal trust assumptions:** Both lanes produce independently verifiable proofs.
5. **Height-gated activation:** Every change has an announced activation height and can be rolled back.

---

## 2. Background

### 2.1 Verifiable Delay Functions

A Verifiable Delay Function (VDF) is a function f: X → Y that requires a specified number of sequential computational steps to evaluate, but whose output can be efficiently verified [2]. Formally, a VDF satisfies:

- **Sequential:** Computing f(x) requires at least T sequential steps, even with unbounded parallelism.
- **Efficiently verifiable:** Given (x, y, proof), verification that y = f(x) takes time O(polylog(T)).
- **Unique output:** For each input x, there is exactly one valid output y.

VDFs were introduced by Boneh et al. [2] and have found applications in randomness beacons, leader election, and proofs of elapsed time.

### 2.2 Genus-2 Hyperelliptic Curves

A genus-2 hyperelliptic curve over a finite field GF(p) is defined by:

```
C: y^2 = f(x) = x^5 + a_4*x^4 + a_3*x^3 + a_2*x^2 + a_1*x + a_0
```

The Jacobian J(C) of such a curve is an abelian group of dimension 2. Elements of the Jacobian are represented in Mumford form as pairs of polynomials (u(x), v(x)) where:

```
u(x) = x^2 + u_1*x + u_0
v(x) = v_1*x + v_0
```

with deg(v) < deg(u) and u | (v^2 - f).

### 2.3 The Doubling Operation

The VDF's core operation is **Jacobian doubling**: given an element D in J(C), compute 2D. This is performed using Cantor's algorithm [3], which for genus-2 curves involves:

1. Compute the resultant of two degree-2 polynomials over GF(p)
2. Perform polynomial GCD and reduction
3. Apply Cantor's reduction to ensure the result is in Mumford form

Each doubling requires approximately 6 field multiplications, 4 field additions, and 2 modular reductions over a 256-bit (or larger) prime field. Crucially, **step T depends on the output of step T-1** — there is no way to compute step T without first computing all preceding steps.

### 2.4 Why Genus-2 Over RSA Groups?

Most VDF constructions use RSA groups (Z/NZ)* where N = pq is a product of unknown primes [2, 4]. We chose Genus-2 Jacobians for several reasons:

| Property | RSA Group VDF | Genus-2 Jacobian VDF |
|----------|--------------|---------------------|
| Trusted setup | Required (someone must know p, q) | **Not required** (curve is public) |
| Group order | Unknown (feature, but limits proof techniques) | Computable (Schoof-like algorithms) |
| Verification | Wesolowski proof [4] | Wesolowski proof (adapted) |
| Post-quantum | Vulnerable to Shor's algorithm | **Partially resistant** (Regev's attack [5] less efficient on genus-2) |
| Performance | Fast squaring in Z/NZ | Slower doubling (more complex group law) |

The absence of a trusted setup is critical for a decentralized mining system — no party should possess trapdoor information that could allow them to compute VDFs faster than the public.

---

## 3. Dual-Lane Mining Architecture

### 3.1 Block Structure

Each block in Q-NarwhalKnight can contain solutions from either or both lanes:

```
Block {
    header: BlockHeader,
    blake3_solutions: Vec<Blake3Solution>,    // GPU lane
    vdf_solutions: Vec<VDFSolution>,          // CPU lane
    transactions: Vec<Transaction>,
    ...
}
```

A block is valid if it contains at least one valid solution from **either** lane. Blocks never stall waiting for a specific lane — if only GPU miners are active, blocks are still produced normally.

### 3.2 BLAKE3 Lane (GPU/ASIC)

The existing mining algorithm is unchanged:

```
input = challenge_hash || nonce || miner_address
h_0 = BLAKE3(input)
h_i = BLAKE3(h_{i-1})  for i = 1..99
result = h_99

Valid if: result <= difficulty_target_blake3
```

- **Difficulty:** Adjusted by LWMA (120-block window, targeting 1.0 bps) [6]
- **Reward share:** 50% of block reward
- **Hardware advantage:** GPUs, FPGAs, ASICs (parallel nonce evaluation)

### 3.3 VDF Lane (CPU)

The new sequential mining algorithm:

```
// 1. Hash challenge to a Jacobian element (deterministic)
D_0 = HashToJacobian(challenge_hash || miner_address, curve_params)

// 2. Perform T sequential doublings (THE sequential work)
D_i = 2 * D_{i-1}  for i = 1..T

// 3. Generate Wesolowski proof (efficient verification)
proof = WesolowskiProve(D_0, D_T, T)

// 4. Submit
VDFSolution {
    vdf_output: D_T,
    vdf_proof: proof,
    vdf_iterations: T,
    miner_address: addr,
}
```

- **Difficulty:** VDF iteration count T, adjusted by LWMA (240-block window)
- **Reward share:** 50% of block reward
- **Hardware advantage:** Single-core clock speed only (~2-5x ASIC ceiling)

### 3.4 Why Parallelism Doesn't Help

Consider a GPU with 10,000 cores attempting the VDF lane:

```
Core 0: D_0 → D_1 → D_2 → ... → D_T  (sequential, takes T steps)
Core 1: (idle — nothing to compute until D_T is known)
Core 2: (idle)
...
Core 9999: (idle)
```

A single CPU core at 3.5 GHz performs the same computation at the same speed as one GPU core at 1.5 GHz — but **faster**, because desktop CPUs have higher single-thread clock speeds than GPU shader cores.

**Estimated performance:**

| Hardware | Clock Speed | VDF Doublings/sec | Time for T=1M | 
|----------|------------|-------------------|--------------|
| Desktop CPU (i7-12700K) | 5.0 GHz | ~2,000,000 | ~0.5s |
| Laptop CPU (i5-1240P) | 4.4 GHz | ~1,500,000 | ~0.67s |
| Server CPU (Xeon) | 3.0 GHz | ~1,200,000 | ~0.83s |
| GPU core (RTX 4090) | 2.5 GHz | ~800,000 | ~1.25s |
| ASIC (theoretical) | 5-10 GHz | ~4,000,000 | ~0.25s |

A consumer desktop CPU is **faster** than a GPU core for VDF evaluation. The ASIC advantage ceiling is ~2-5x (clock speed scaling only), compared to ~10,000x for hash-based PoW.

---

## 4. Economic Model

### 4.1 Reward Split

The total block reward is divided equally between the two lanes:

```
total_reward = emission_controller.calculate_reward()

blake3_reward = total_reward * BLAKE3_SHARE_BPS / 10000  // 5000 bps = 50%
vdf_reward    = total_reward * VDF_SHARE_BPS / 10000     // 5000 bps = 50%
```

The split is a **consensus parameter** encoded in the height-gated upgrade, not an environment variable. Changing it requires a new consensus upgrade with announced activation height.

### 4.2 Difficulty-Weighted Rewards Within Each Lane

Within each lane, rewards are distributed proportionally to solution difficulty using the Phase A weighting formula:

```
weight_i = 2^(leading_zeros_i)

reward_i = (total_lane_reward * weight_i) / sum(all_weights)
```

This incentivizes miners to find the hardest solutions they can, not just the minimum difficulty. A miner who finds a 20-bit solution earns 16x more than a miner who finds a 16-bit solution.

### 4.3 Unclaimed Lane Rewards

If no VDF solutions are submitted for a block:

```
Blocks 1-100 after activation:   VDF rewards redistributed to BLAKE3 lane (grace period)
Blocks 101+:                     Unclaimed VDF rewards BURNED (permanent supply reduction)
```

This creates a powerful economic incentive:
- During the grace period: GPU miners receive 100% of rewards (no penalty for low VDF participation)
- After the grace period: unclaimed VDF rewards create **deflationary pressure**
- If 50% of blocks have no VDF solutions, the effective emission rate drops by 25%
- This makes VDF mining increasingly attractive over time

### 4.4 The "Jackpot Problem" Mitigation

A strategic miner could wait until VDF rewards have accumulated for 99 blocks, then submit one proof to claim the entire jackpot. To prevent this:

```
max_accumulated_vdf_reward = 5 * per_block_vdf_allocation
```

Accumulated rewards are capped at 5x the single-block allocation. Any excess is burned immediately. This limits the jackpot incentive while still allowing accumulation to attract miners.

### 4.5 Simultaneous Mining

A single mining machine can — and should — mine both lanes simultaneously:

```
Thread 0-N:   GPU mining BLAKE3 (parallel nonce evaluation)
Thread N+1:   CPU mining VDF (sequential doubling on one core)
```

This is the natural deployment model. A miner with a GPU and a CPU earns from both lanes without conflict. The two computations use different hardware resources and do not interfere.

---

## 5. Difficulty Adjustment

### 5.1 Independent Per-Lane LWMA

Each lane has its own LWMA (Linearly Weighted Moving Average) difficulty adjuster [6]:

| Parameter | BLAKE3 Lane | VDF Lane |
|-----------|-------------|----------|
| Window | 120 blocks | 240 blocks |
| Max adjustment per step | 2.0x | 1.5x |
| Target rate | Determined by participation | Determined by participation |
| Difficulty unit | Leading zero bits | VDF iteration count T |
| Floor | 16 bits | T_min (calibrated from benchmarks) |

The VDF lane uses a wider window (240 vs 120) and gentler maximum adjustment (1.5x vs 2.0x) because changes to T are more visible to miners (each T change affects the wall-clock time of VDF evaluation).

### 5.2 Pure Function of Chain State

Both difficulty adjusters follow the **emission controller pattern** — a pure function of chain-visible data:

```rust
fn calculate_difficulty_for_next_block(
    previous_difficulty: u32,
    recent_timestamps: &[u64],
    activation_height: u64,
    next_height: u64,
    target_block_time_secs: u64,
) -> u32
```

Same inputs produce same output on every node. No background timer, no mutable state, no network communication. Called at challenge endpoint, block template creation, and block validation. If any node disagrees, they compute the same function with the same chain data and get the same answer.

### 5.3 Lane Target Rates

Rather than prescribing fixed rates per lane, we let the difficulty adjusters **find the natural equilibrium** based on actual participation:

- If 90% of mining power is GPUs → BLAKE3 lane produces 90% of solutions at its equilibrium difficulty
- VDF lane finds its own equilibrium based on CPU participation
- The 50% reward allocation is what creates the CPU incentive — it's a "reserved seat" regardless of participation levels

---

## 6. Security Analysis

### 6.1 Sequential Work Guarantee

**Claim:** Computing T Jacobian doublings requires at least T sequential steps, even with unbounded parallelism.

**Argument:** Each doubling D_{i+1} = 2 * D_i requires knowing D_i as input. The group law for Genus-2 Jacobians involves polynomial operations (resultant, GCD, reduction) that depend on all coordinates of D_i. There is no known shortcut to compute D_T from D_0 without computing all intermediate values D_1, ..., D_{T-1}.

This is the same sequential work argument used by Wesolowski [4] and Pietrzak [7] VDFs, applied to a different group.

### 6.2 Verification Efficiency

The Wesolowski proof allows verification in O(log T) group operations:

```
Verifier receives: (D_0, D_T, proof, T)
Verifier computes: Check that proof is consistent with T doublings from D_0 to D_T
Time: O(log T) doublings (not T doublings)
```

For T = 1,000,000: evaluation takes ~0.5-2 seconds, verification takes ~20 doublings ≈ 10 microseconds. The asymmetry is critical — the server can verify VDF proofs from hundreds of miners per second.

### 6.3 ASIC Resistance

The VDF lane is not "ASIC-proof" — it is "ASIC-capped." An ASIC can increase clock speed beyond what consumer CPUs achieve:

| Hardware | Estimated advantage over consumer CPU |
|----------|--------------------------------------|
| Consumer CPU | 1x (baseline) |
| High-end desktop CPU | 1.5x |
| Custom ASIC | 2-5x (clock speed limited by physics) |

Compare this to BLAKE3 PoW where ASICs achieve 10,000-1,000,000x advantage. The VDF lane's ASIC ceiling is fundamentally bounded by the speed of sequential logic — you cannot parallelize what is inherently sequential.

### 6.4 Quantum Considerations

The VDF's sequential work property is independent of the discrete log problem. Even if a quantum computer could solve the genus-2 HECC DLP (via Regev's attack [5] or Shor-like algorithms), it would still need to perform T sequential doublings to compute the VDF output. The quantum advantage, if any, would be in the per-step computation time — not in reducing the number of steps.

**Honest framing:** The quantum resistance of the VDF lane is **conjectured, not proven**. Recent work (IACR 2024/2004 [5]) shows improved quantum attacks on genus-2 HECC. We present the VDF lane as "CPU-fair" rather than "quantum-resistant" to avoid overclaiming.

### 6.5 Attack Vectors

| Attack | Severity | Mitigation |
|--------|----------|------------|
| GPU mining VDF lane | Low — GPUs are slower per core than CPUs | Natural economic deterrent (lower clock speed = less reward) |
| ASIC for VDF lane | Low — 2-5x advantage ceiling | Not cost-effective for a 50% reward share with 2-5x ceiling |
| Withholding VDF solutions | Low — rewards accumulate then burn | Jackpot cap at 5x prevents strategic withholding |
| Time-warp attack on VDF difficulty | Medium — manipulate timestamps to lower T | LWMA solvetime clamping: [1ms, 6x target] |
| VDF output grinding | Not possible — output is deterministic from input + T | Verified by Wesolowski proof |
| 51% attack on one lane | Medium — attacker controls one lane's rewards | Other lane continues independently; block validity requires either lane |

---

## 7. Implementation

### 7.1 Existing Code

The VDF implementation is complete in `crates/q-vdf/src/genus2_vdf.rs` (622 lines):

- Genus-2 curve parameter sets for pq128, pq192, pq256 security levels
- Jacobian element representation in Mumford form
- Cantor's doubling algorithm
- Hash-to-Jacobian-element mapping (deterministic)
- Wesolowski proof generation and verification
- Adaptive difficulty wrapper with LWMA integration
- Comprehensive benchmarks in `crates/q-vdf/benches/`

The server already handles dual-path verification, and the MiningSolution struct has VDF fields (all Optional, currently None).

### 7.2 Deployment Plan

| Phase | What | Timeline |
|-------|------|----------|
| **B (done)** | LWMA difficulty adjustment for BLAKE3 lane | Activated at height 14,900,000 |
| **C.1** | Benchmark VDF on production hardware | 1 week |
| **C.2** | Implement VDF computation in miner binary | 2 weeks |
| **C.3** | Add VDF parameters to challenge endpoint | 1 week |
| **C.4** | Test on Delta Docker with dual-lane mining | 1 week |
| **C.5** | Announce activation height to community | 1 week before activation |
| **C.6** | Deploy and activate | Height-gated, instant at activation |

### 7.3 Standalone VDF Prover

For miners who want to run VDF computation without the full mining binary:

```bash
q-vdf-prover --challenge <hex> --iterations 1000 --security pq128
# Output: vdf_output (hex), vdf_proof (hex)
```

This allows lightweight integration — a miner can call the standalone prover as a subprocess and submit the output through the standard mining API.

### 7.4 Miner Calibration

```bash
q-miner calibrate --vdf
# Output:
#   CPU: Intel i7-12700K @ 5.0 GHz
#   VDF doublings/sec: 2,100,000 (pq128)
#   Estimated time for T=1,000,000: 0.48s
#   Estimated time for T=2,000,000: 0.95s
#   Recommended: Mine both BLAKE3 + VDF lanes
```

---

## 8. Comparison With Other Approaches

| Approach | CPU Fair? | ASIC Ceiling | Complexity | Used By |
|----------|-----------|-------------|-----------|---------|
| Memory-hard PoW (Ethash) | Partial — GPUs still dominate | ~10x | High | Ethereum (pre-PoS) |
| RandomX | Yes — CPU-optimized | ~3x | Very high | Monero |
| Proof-of-Stake | N/A — no mining | N/A | Medium | Ethereum 2.0 |
| **Dual-lane VDF (this paper)** | **Yes — reserved 50% for CPU** | **2-5x** | **Medium** | **Q-NarwhalKnight** |

Key difference: dual-lane mining does **not** penalize GPU miners. It adds a CPU lane alongside the existing GPU lane. GPU miners keep their full 50% share. This is a Pareto improvement — no one is worse off, and CPU miners gain access.

---

## 9. Future Work

### 9.1 Dynamic Reward Split

The initial 50/50 split may not be optimal long-term. A governance mechanism could adjust the split based on:
- Participation ratio between lanes
- Network security requirements
- Community vote

### 9.2 FPGA Collaboration

In collaboration with Dragon Ball Miner, we are developing FPGA and ASIC implementations targeting the BLAKE3 lane. The QUG-V1 SoC (16-core RISC-V with Xcrypto BLAKE3 extension) is in Phase 1A RTL delivery for Xilinx Kintex-7 FPGA [8].

### 9.3 VDF Lane Upgrades

If the Genus-2 VDF is found to have unexpected weaknesses, the VDF lane can be upgraded to a different sequential function via a height-gated consensus upgrade. Candidates include:
- Isogeny-based VDFs (SIDH/SIKE family)
- Lattice-based sequential functions
- AES-based iterated encryption

### 9.4 Multi-Lane Extension

The dual-lane architecture generalizes naturally to N lanes, each targeting different hardware profiles. For example:
- Lane 1: BLAKE3 (GPU/ASIC)
- Lane 2: Genus-2 VDF (CPU)
- Lane 3: Memory-hard function (high-RAM hardware)
- Lane 4: Bandwidth-bound function (network nodes)

---

## 10. Conclusion

Dual-lane mining provides a principled solution to GPU dominance in proof-of-work blockchains. By reserving 50% of block rewards for a sequential computation that CPUs can perform competitively, we create meaningful economic participation for the broadest possible set of miners — from gaming PCs to cloud VMs to Raspberry Pis.

The design is conservative:
- GPU miners are not harmed (their 50% share is unchanged)
- The VDF lane is an addition, not a replacement
- All changes are height-gated and reversible
- The VDF implementation is based on well-studied mathematical objects (genus-2 Jacobians)
- The economic model is self-balancing (unclaimed rewards accumulate then burn)

We believe this represents the most promising path toward truly democratic mining, where the barrier to entry is a consumer CPU rather than a GPU farm.

---

## References

[1] S. Nakamoto, "Bitcoin: A Peer-to-Peer Electronic Cash System," 2008.

[2] D. Boneh, J. Bonneau, B. Bunz, and B. Fisch, "Verifiable Delay Functions," in CRYPTO 2018, LNCS 10991, pp. 757-788.

[3] D.G. Cantor, "Computing in the Jacobian of a hyperelliptic curve," Mathematics of Computation, vol. 48, no. 177, pp. 95-101, 1987.

[4] B. Wesolowski, "Efficient Verifiable Delay Functions," in EUROCRYPT 2019, LNCS 11478, pp. 379-407. Also: IACR ePrint 2018/623.

[5] O. Regev et al., "Improved Quantum Attacks on Hyperelliptic Curve Cryptosystems," IACR ePrint 2024/2004.

[6] Zawy12, "LWMA Difficulty Algorithms," https://github.com/zawy12/difficulty-algorithms/issues/3

[7] K. Pietrzak, "Simple Verifiable Delay Functions," in ITCS 2019.

[8] Q-NarwhalKnight, "QUG-V1 Mining SoC RTL — Phase 1A Technical Review," April 2026.

---

*This document is a community draft. We welcome feedback, criticism, and contributions. Contact: quillon.xyz*

*Q-NarwhalKnight is an open-source project. All code is available at code.quillon.xyz.*
