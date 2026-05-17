# Track 1: BLAKE3 + Genus-2 Jacobian VDF Mining — Peer Review Document

**Date:** 2026-04-07
**Chain Status:** DAG-Knight producing blocks at 3/sec, 250 solutions/block, mining operational
**Constraint:** MUST NOT break existing mining. Any change needs height-gated activation.
**Prepared for:** DeepSeek + ChatGPT peer review

---

## 1. THE PROBLEM

CPU miners have left the network. The current mining algorithm (BLAKE3 x 100 iterations) is **trivially GPU-dominated** because:

- Each nonce candidate is independent (embarrassingly parallel)
- 100 BLAKE3 hashes per nonce is fast even sequentially (~300ns on modern CPU)
- GPUs try millions of nonces/sec vs thousands for CPU
- No memory-hardness, no sequential dependency *between* nonces
- The "VDF" is not actually a VDF — it's just iterated hashing with a tiny iteration count

This hurts the HIBT exchange listing crowdfund because community size and miner count are listing metrics.

---

## 2. CURRENT MINING ARCHITECTURE (What Actually Runs Today)

### 2.1 The Algorithm

```
For each nonce candidate:
  1. input = challenge_hash[32 bytes] || nonce[8 bytes LE] = 40 bytes
  2. h = BLAKE3(input)
  3. for i in 0..99:
       h = BLAKE3(h)
  4. if h < difficulty_target: SOLUTION FOUND
```

**Total work per nonce:** 100 BLAKE3 hashes (1 initial + 99 iterations)
**Hardcoded:** `VDF_ITERATIONS = 100` in both CPU miner and GPU kernel
**Server verification:** Identical — recomputes 100 BLAKE3 hashes, compares result

### 2.2 Key Code Locations

| Component | File | Lines | What It Does |
|-----------|------|-------|--------------|
| CPU miner | `gui/slint-wallet/src/miner.rs` | 74-90 | `mine_hash()` — 100 BLAKE3 iterations |
| GPU kernel | `crates/q-mining/src/gpu.rs` | 60-281 | OpenCL BLAKE3 kernel, 99 VDF iterations |
| GPU (wallet) | `gui/slint-wallet/src/gpu_miner.rs` | 185-218 | Same OpenCL kernel in wallet |
| Challenge gen | `crates/q-api-server/src/handlers.rs` | 9133-9472 | `get_mining_challenge()` |
| Solution submit | `crates/q-api-server/src/handlers.rs` | 8671-9130 | `submit_mining_solution()` |
| VDF verify | `crates/q-api-server/src/main.rs` | 15050-15235 | Dual-path: Genus-2 OR BLAKE3 |
| Difficulty check | `crates/q-api-server/src/handlers.rs` | 9550-9552 | `hash < target` byte-wise comparison |

### 2.3 Mining Pipeline Architecture

```
[Miner] --GET--> /api/v1/mining/challenge
                   |
                   v
         CachedChallenge {
           challenge_hash: BLAKE3(version || height || difficulty || vdf_iters),
           difficulty_target: [0x00, 0x00, 0xFF, ...],
           block_height: u64,
           vdf_iterations: u32 (dynamic formula, but miners ignore it),
           block_reward: f64,
           expires_at: timestamp
         }
                   |
[Miner computes]   |
  for each nonce:  |
    h = BLAKE3(challenge || nonce)
    repeat 99x: h = BLAKE3(h)
    if h < target: submit
                   |
[Miner] --POST--> /api/v1/mining/submit
         {miner_address, nonce, hash, difficulty_target, challenge_hash}
                   |
                   v
         HTTP handler: validate, queue to 1 of 8 sharded mpsc channels
         Return 200 immediately (zero-lock fast path)
                   |
                   v
         Background shard consumer (8 parallel tasks):
           Batch 2000 submissions OR 50ms timeout
           rayon par_iter: verify each submission
             - PATH A: Genus-2 Jacobian (if vdf_output present) -- NOT USED TODAY
             - PATH B: BLAKE3 x100 (all current miners)
             - Difficulty check: hash < target
           Pass verified solutions to block producer
                   |
                   v
         Block producer: collect up to 250 solutions, create block
         Broadcast via P2P gossipsub
```

### 2.4 What's NOT Implemented

| Feature | Status | Location |
|---------|--------|----------|
| **Difficulty adjustment** | STUB — returns current difficulty unchanged | `crates/q-mining/src/difficulty.rs:28-33` |
| **Genus-2 VDF in miners** | Fields exist but all set to `None` | `gui/slint-wallet/src/miner.rs:659-663` |
| **Block weight by VDF quality** | Not implemented — all valid blocks equal | `crates/q-dag-knight/src/ordering_rules.rs` |
| **Hybrid mining reward split** | Commented as 50/50 concept | `crates/q-mining/src/hybrid_mining.rs:3-4` |

---

## 3. GENUS-2 JACOBIAN VDF (Already Implemented, Not Activated)

### 3.1 What Exists

A full Genus-2 hyperelliptic Jacobian VDF implementation sits in `crates/q-vdf/src/genus2_vdf.rs` (622 lines):

**Mathematical basis:**
- Genus-2 curve: `y^2 = x^5 + a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0`
- Jacobian J(C) is a 2-dimensional abelian variety
- Elements represented in Mumford form: `D = (u(x), v(x))` where `u = x^2 + u1*x + u0`, `v = v1*x + v0`
- Doubling via Cantor's algorithm: `D -> 2D`
- **No known quantum speedup** for discrete log on genus-2 Jacobians (vs Shor's algorithm for RSA)

**Three security levels:**
- `pq128()`: 256-bit prime field
- `pq192()`: 384-bit prime field  
- `pq256()`: 512-bit prime field

**VDF evaluation:**
```rust
// crates/q-vdf/src/genus2_vdf.rs:290-331
pub async fn evaluate(&self, input: &[u8], iterations: u64) -> Result<VDFOutput> {
    let g = JacobianElement::from_hash(input, &self.curve);
    for i in 0..iterations {
        g = self.double_jacobian(&g)?;  // D -> 2D (Cantor's algorithm)
    }
    // Output: serialized Mumford coordinates
}
```

**Critical property: SEQUENTIAL.** Each doubling depends on the previous result. This **cannot be parallelized** — exactly the property that makes a real VDF.

### 3.2 Server Already Handles It

The dual-path verification in `main.rs:15095-15235` already supports both:

```
IF submission has genus2_vdf_output AND genus2_vdf_proof:
    PATH A: Genus-2 Jacobian verification
      1. seed = BLAKE3(challenge || nonce)
      2. Verify Wesolowski proof: SHA3("genus2-wesolowski-challenge" || seed || output || iterations)
      3. Verify SHA3-256(vdf_output) == submitted_hash
      
ELSE:
    PATH B: Legacy BLAKE3 x100 verification
      1. Recompute BLAKE3 x100(challenge || nonce)
      2. Compare to submitted hash

THEN: hash < difficulty_target (both paths)
```

### 3.3 What's Missing for Activation

1. **Miner-side Genus-2 computation** — wallet currently sends `vdf_output: None`
2. **Activation height** — no height gate defined in `q-consensus-guard`
3. **Difficulty calibration** — how many Genus-2 iterations = what time target?
4. **GPU resistance analysis** — is Genus-2 doubling actually GPU-hard?

---

## 4. THE CORE DESIGN QUESTION

### Option A: Replace BLAKE3 with Genus-2 (Pure Sequential VDF)

```
Mining becomes:
  1. seed = BLAKE3(challenge || nonce)
  2. g = JacobianElement::from_hash(seed)
  3. for i in 0..N: g = double_jacobian(g)   // SEQUENTIAL, ~2-4 sec
  4. hash = SHA3-256(g.to_bytes())
  5. if hash < target: SOLUTION
```

**Pros:**
- Truly sequential — GPU gets zero advantage over CPU
- Post-quantum secure (no known quantum speedup)
- Already implemented server-side
- Wesolowski proof enables O(log N) verification (fast for server)

**Cons:**
- Each attempt takes 2-4 seconds (vs nanoseconds for BLAKE3)
- Miners can only try ~1 nonce every 2-4 seconds
- GPU miners lose ALL investment overnight — community backlash?
- Verification is more complex (Wesolowski proofs)
- Single-threaded only — multi-core CPUs waste cores
- **Risk:** If Genus-2 doubling turns out to be GPU-accelerable (bignum math), plan fails

### Option B: Dual-Proof System (BLAKE3 + Genus-2)

```
Two valid proof types, miner chooses one:
  
  TYPE 1 (GPU-friendly): BLAKE3 x100 with HARDER difficulty target
  TYPE 2 (CPU-friendly): Genus-2 VDF with EASIER difficulty target

Block contains proof_type field. Server verifies accordingly.
Both earn same block reward.
```

**Pros:**
- Doesn't break existing GPU miners
- CPU miners have a viable path
- Market finds natural GPU/CPU equilibrium
- Gradual transition possible (adjust relative difficulties over time)

**Cons:**
- Two difficulty targets to manage
- Potential "easier proof" gaming — miners always pick the lower-difficulty path
- Complexity in consensus (do both proof types have equal block weight?)
- Attack vector: adversary optimizes for whichever proof is currently easier

### Option C: Hybrid Mining (BLAKE3 for PoW + Genus-2 for Bonus)

```
Required: BLAKE3 x100 proof (as today) — this earns base reward
Optional: Genus-2 VDF proof — this earns bonus reward (e.g., +50%)

Block must have BLAKE3 proof. Genus-2 proof is additional.
CPU miners who can't compete on BLAKE3 still earn via Genus-2 bonus.
```

**Pros:**
- Zero risk to existing mining (BLAKE3 still works exactly as today)
- CPU miners get meaningful income via Genus-2 bonus
- GPU miners keep working, not alienated
- Simplest to implement — existing verification handles both paths

**Cons:**
- CPU miners still can't compete on base reward
- GPU miners can also compute Genus-2 (sequentially) for the bonus
- Doesn't fundamentally change GPU dominance, just adds a small CPU incentive
- More complex reward calculation

### Option D: Memory-Hard PoW (RandomX/ProgPoW style) — NOT RECOMMENDED

Mentioned for completeness. Would require a completely new mining algorithm, breaks everything, and is outside the existing codebase architecture. Not viable without a hard fork.

---

## 5. CONSTRAINTS AND RISKS

### 5.1 Non-Negotiable Constraints

1. **Mining works today. Keep it that way.** Any change MUST be height-gated.
2. **DAG-Knight consensus must not be affected.** Block ordering is topological, not PoW-weighted.
3. **Server verification throughput.** Currently handles 320K submissions/sec via 8-shard rayon pipeline. Genus-2 verification uses Wesolowski proofs (O(log N)), which is fast, but needs benchmarking.
4. **Backward compatibility.** Old miners must keep working until activation height.
5. **No hard fork.** Everything via height-gated soft activation.

### 5.2 Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Genus-2 doubling is GPU-accelerable via bignum libraries | HIGH | Benchmark on actual GPUs before committing |
| Dual difficulty creates arbitrage/gaming | MEDIUM | Tie difficulties to observed hashrate per proof type |
| CPU miners still can't compete (bonus too small) | MEDIUM | Make Genus-2 bonus significant (50%+) |
| Verification bottleneck (Genus-2 proofs slow to verify) | LOW | Wesolowski proofs are O(log N) — fast |
| Breaking existing miners during transition | HIGH | Height-gated activation with 2-week notice |
| Difficulty adjustment doesn't exist yet | HIGH | Must implement before any mining change |

### 5.3 Missing Prerequisite: Difficulty Adjustment

**CRITICAL:** There is NO difficulty adjustment algorithm implemented. `difficulty.rs` is a stub that returns the current difficulty unchanged. Any mining system change MUST include implementing difficulty adjustment first, or the network will be unstable.

Current difficulty target is static: `[0x00, 0x00, 0xFF, 0xFF, ...]` (2 leading zero bytes).

---

## 6. QUESTIONS FOR PEER REVIEWERS

### For Cryptographers (DeepSeek):

1. **Is Genus-2 Jacobian doubling truly sequential?** Can a GPU with thousands of cores and fast bignum libraries (e.g., CGBN) accelerate Cantor's algorithm? The modular arithmetic involves 256-512 bit primes. GPUs are decent at this.

2. **Wesolowski proof security in dual-proof context.** If we accept both BLAKE3 and Genus-2 proofs, does the Wesolowski proof's Fiat-Shamir challenge remain secure when the challenge domain includes both proof types?

3. **Is the current Genus-2 implementation correct?** The `double_jacobian()` function (genus2_vdf.rs:334-383) implements Cantor's algorithm. Has this been verified against a reference implementation?

4. **Parameter selection.** The `pq128()` curve uses `y^2 = x^5 + x^2 - 1` over a 256-bit prime. Is this a well-studied curve? What's the actual security level against classical and quantum attacks?

5. **Grinding attacks.** With dual proofs, can a miner grind on the BLAKE3 nonce to find inputs that make the Genus-2 VDF output unusually easy? (Probably not, since VDF output depends on the full sequential computation, but worth confirming.)

### For Systems Engineers (ChatGPT):

1. **Difficulty adjustment algorithm.** What algorithm should we implement? Bitcoin-style (2016-block window)? LWMA (Linear Weighted Moving Average)? DAA (Difficulty Adjustment Algorithm from BCH)? We need one that works with DAG-Knight (multiple blocks per round, not a single chain).

2. **Dual-difficulty balancing.** If we go with Option B (dual-proof), how do we prevent miners from always choosing the easier proof? Should difficulties be coupled (when one gets easier, the other gets harder)?

3. **Verification throughput.** Current pipeline processes 320K BLAKE3 verifications/sec. Genus-2 Wesolowski verification involves SHA3-256 and bignum comparison. What's the expected throughput? Is the 8-shard rayon architecture sufficient?

4. **Migration path.** How do we go from "100% BLAKE3 miners" to "mixed BLAKE3 + Genus-2" without a flag day? Suggested: height-gated activation, 2-week announcement, old miners keep working on BLAKE3 path, new miners can choose either.

5. **Economic equilibrium.** If GPU miners earn X via BLAKE3 and CPU miners earn Y via Genus-2, what's the expected ratio? Does it converge to a stable equilibrium or oscillate?

---

## 7. RECOMMENDED APPROACH

**We recommend Option B (Dual-Proof) with the following implementation plan:**

### Phase 0: Difficulty Adjustment (Prerequisite)
- Implement LWMA difficulty adjustment in `crates/q-mining/src/difficulty.rs`
- Separate difficulty targets for BLAKE3 and Genus-2 proof types
- Target: 1 BLAKE3 block per 15s, 1 Genus-2 block per 15s (30s total average)

### Phase 1: Genus-2 Miner Activation (Height-Gated)
- Wire `compute_genus2_vdf()` into slint-wallet CPU miner
- Set activation height ~20,000 blocks in the future
- CPU miner switches to Genus-2 path after activation
- GPU miners continue on BLAKE3 path (no change for them)
- Server already handles both paths (dual verification exists)

### Phase 2: Reward Calibration
- Monitor GPU vs CPU solution submission rates
- Adjust relative difficulties to target 50/50 block ratio
- Or: weight Genus-2 blocks higher (1.5x reward) to attract CPU miners

### Phase 3: GPU Genus-2 Prevention (If Needed)
- If GPUs start computing Genus-2 VDF, increase iteration count
- Genus-2 iterations should target 2-4 seconds per attempt on CPU
- GPU bignum throughput for 256-bit modular arithmetic is ~10-100x slower than CPU per-thread
- But GPU has thousands of threads — need to ensure VDF is truly sequential

---

## 8. APPENDIX: GENUS-2 VDF BENCHMARK REQUEST

Before committing to any approach, we need benchmarks:

```
Benchmark 1: Genus-2 doubling speed
  - CPU (single thread, x86-64): How many doublings/sec on pq128 curve?
  - GPU (single thread equivalent): How fast is 256-bit modular arithmetic on GPU?
  - Target: 500-2000 doublings for 2-4 second VDF on CPU

Benchmark 2: Verification speed
  - Wesolowski proof verification for Genus-2 VDF
  - How many verifications/sec on server (rayon, 8 cores)?
  - Must not bottleneck the 320K submissions/sec pipeline

Benchmark 3: Memory usage
  - Genus-2 Mumford representation size
  - Wesolowski proof size
  - Impact on P2P block propagation (blocks already contain up to 250 solutions)
```

---

*This document is for peer review. No code changes should be made until reviewers respond.*
*"Twelve leagues deep, and every league a lesson."*
