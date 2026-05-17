# Technical Plan: Instant-Bootstrap Nodes via Recursive Lattice zk-SNARK

**Date:** 2026-05-13
**Author:** Server Beta
**Target:** Q1 2027 testnet, Q3 2027 mainnet activation
**Risk class:** CRITICAL (changes the consensus trust model)
**Related work:** `crates/q-ivc/` (gadget research), BAL-001 state-root header field, Phase 2 backfill

---

## The Goal in One Sentence

A new node should download `(state_tip, π_tip)`, verify `π_tip` in **≤ 10 ms**, and **immediately** start mining, validating new blocks, and serving balances — while old blocks back-fill in the background and become explorer-available the moment they land.

---

## Why This Matters

Today a new node has to:
1. Bootstrap by trusting the BAL-001 checkpoint snapshot (signed by genesis nodes), or
2. Sync the full chain (~11.4 M blocks, hours-to-days).

Neither is good. The checkpoint is a trust dependency on the genesis-node operators. The full sync is a UX disaster.

A recursive zk-SNARK collapses both into a single cryptographic check. The node trusts **math, not operators.** No checkpoint signers, no sync wait.

The "lattice" part of the spec matters because QNK is post-quantum throughout (Dilithium5 signatures, Kyber1024 KEM). A recursive SNARK over an elliptic curve (Groth16, Nova, HyperNova) would re-introduce a quantum-broken dependency at the most critical security primitive in the system. Lattice-based recursive arguments — LaBRADOR (2023), LatticeFold (2024), Greyhound (2024) — let us stay post-quantum end-to-end.

---

## What the Proof Attests

The recursive proof `π_n` is a cryptographic witness that:

> *"There exists a sequence of `n` valid blocks `B_1, B_2, ..., B_n` such that:*
>   *(1) `B_1` is the genesis block,*
>   *(2) for every `i`, `state_{i} = δ(state_{i-1}, B_i)` according to the QNK state-transition function,*
>   *(3) every signature inside every block verified,*
>   *(4) every coinbase emission respected the schedule,*
>   *(5) the resulting `state_n` is the one I claim."*

Verifying `π_n` is equivalent to having replayed every block from genesis — but in 10ms instead of hours.

---

## The Folding Loop (IVC)

This works by *incrementally* extending the proof. The prover (genesis nodes, validators) maintains an ongoing folded proof:

```
π_0  = trivial proof (state_0 = genesis state)
π_1  = Prove( δ(state_0, B_1) = state_1  AND  verify(π_0) )
π_2  = Prove( δ(state_1, B_2) = state_2  AND  verify(π_1) )
...
π_n  = Prove( δ(state_{n-1}, B_n) = state_n  AND  verify(π_{n-1}) )
```

Each new block's prover step folds `B_{n+1}`'s validity *and* the verification of `π_n` into a new succinct proof `π_{n+1}`. The verifier (a new node) just checks `π_n` once. Constant verification cost, regardless of how many blocks the chain has accumulated.

This is **Incrementally Verifiable Computation (IVC)** — first formalized by Valiant (2008), made efficient by Nova (2021), and now reachable from lattice assumptions by LatticeFold (2024).

---

## Cryptographic Stack — Three Candidates

| Scheme | Verification cost | Proof size | Maturity | Post-quantum |
|--------|------------------|-----------|----------|--------------|
| **Nova / HyperNova** (elliptic curve) | ~5 ms | ~5 KB | Production-ready (deployed in Lurk, Aleo) | ❌ Not PQ |
| **LatticeFold** (Boneh-Chen-Tairi 2024) | ~50 ms (target → 10 ms with FFI/SIMD) | ~150 KB | Research-grade, no production implementations | ✅ Module-SIS / RLWE |
| **LaBRADOR-IVC** (Beullens-Seiler 2023 + recursion wrapper) | ~30 ms (target → 10 ms) | ~50 KB | Reference impl exists, no IVC mode yet | ✅ Module-SIS |

**Recommended path:**
- **Phase 1 (Q4 2026 → Q1 2027):** Implement Nova over BN254 in `crates/q-ivc/`. Validates the architecture end-to-end. Not deployable as the final answer because BN254 is quantum-broken, but it lets us debug the circuit, the state-transition function, and the integration without inventing lattice cryptography simultaneously. **Already 80% of what we need is the circuit, not the proof system.**
- **Phase 2 (Q2-Q3 2027):** Swap the proof system from Nova to LatticeFold. The circuit stays the same. The prover/verifier swap is contained to ~2K lines of code.
- **Phase 3 (Q4 2027 → 2028):** Production hardening, prover hardware acceleration (FFI to lattice-NTT C library, AVX-512 / GPU prover).

This is the standard *de-risk the system, then de-risk the crypto* pattern.

---

## The Circuit: δ(state, block) → state'

Verification of one block must be expressible as an arithmetic circuit. QNK's state transition is unusually circuit-friendly because it's already mostly hash-based:

### Inputs (public)
- `state_root_prev` — BLAKE3 root over sorted (wallet → balance) map at block N-1
- `state_root_next` — claimed root at block N
- `block_hash` — BLAKE3 of block N's header

### Witness (private)
- The block body (transactions, coinbase, signatures)
- Merkle paths into `state_root_prev` for every wallet touched by the block
- Updated Merkle paths into `state_root_next`
- Dilithium5 signatures + verification witnesses

### Constraints
1. `block_hash = BLAKE3(header)` ← gadget `q_ivc::gadgets::blake3::verify_hash`
2. Every transaction's Dilithium5 signature verifies ← gadget `q_ivc::gadgets::dilithium::verify` (already being built)
3. For each transaction `(from, to, amount, fee)`:
   - Merkle path proves `(from, balance_old)` ∈ `state_root_prev`
   - Merkle path proves `(to, balance_old_to)` ∈ `state_root_prev`
   - `balance_old ≥ amount + fee` (range check via signed-norm gadget)
   - New balances `(from, balance_old - amount - fee)` and `(to, balance_old_to + amount)` consistent with `state_root_next`
4. Coinbase: emission amount matches scheduled rate at height N ← lookup table from emission controller
5. NTT/anchor election verifications for the DAG-Knight specific consensus parts ← we already have NTT butterfly gadget

We have **most of the gadgets already**. The remaining work is:
- Merkle path circuit over the balance map (medium effort, ~1 month)
- Glue circuit that composes everything (medium effort, ~1 month)
- Nova integration (the recursion wrapper, ~6 weeks)

---

## What the Bootstrap Wire Protocol Looks Like

### Today (v10.9.x)
```
NewNode → Peer: GET /api/v1/checkpoint/snapshot
Peer    → NewNode: { state at height C, signed by genesis nodes }
NewNode: trusts signatures, accepts snapshot, begins forward sync
```

### After this work (v11.x)
```
NewNode → Peer: GET /api/v1/proof/tip
Peer    → NewNode: {
  current_tip_height: N,
  state_root_at_N: 0x...,
  recursive_proof_π_N: 0x... (~50 KB),
  block_N_header: { ... }
}
NewNode: verify_proof(π_N, state_root_at_N, block_N_header) → boolean
   if true: state_root accepted, node ready to mine + transact
   if false: peer is lying — try another peer
```

10 ms verify, no checkpoint signers required.

---

## Instant-Mode Operation

The moment `π_tip` verifies, the new node:

| Capability | Available immediately? | How? |
|------------|----------------------|------|
| Mine new blocks | ✅ | Has `state_root` and `tip_height` → can build candidate block at N+1 |
| Receive new blocks via gossipsub | ✅ | Validate against `state_root_N` |
| Send transactions | ✅ | Knows account state from `state_root_N` (via Merkle proofs fetched on-demand from peers) |
| Query its own balance | ✅ | Ask a peer for Merkle path into `state_root_N`, verify locally |
| Query own transactions in mempool | ✅ | Local mempool |
| Validate gossipsub votes | ✅ | Same as any other validator |
| Look up historical block by hash | ❌ until backfilled | See "Progressive Archival" below |
| Look up historical transaction by hash | ❌ until backfilled | Same |
| Re-validate the chain from genesis | ❌ until backfilled | Doesn't need to — `π_tip` already did this |

The node is a **full participating peer** the instant `π_tip` verifies. The only thing it can't do is answer historical look-ups, because it doesn't have the data yet.

---

## Progressive Archival: Explorer During Backfill

This is the part of your spec that needs the most care, because two queries — "what's my current balance" and "show me block #5,000,000" — must behave very differently.

### Two query classes

| Query class | Source of truth | Available |
|-------------|----------------|-----------|
| **Current-state** (balance, contract storage, pool reserves, mempool) | `state_root_N` + Merkle paths | Immediately |
| **Historical-block** (block-by-hash, block-by-height, tx receipt, event log at height H) | The block bytes at height H | Only after H is backfilled |

### API surface during backfill

The new node exposes an additional header on historical queries:

```
GET /api/v1/blocks/5000000
HTTP/1.1 202 Accepted
X-QNK-Archive-Status: backfilling
X-QNK-Archive-Progress: 4321000/11400000 (37.9%)
X-QNK-Archive-ETA: 2026-05-14T03:00:00Z
X-QNK-Archive-Lowest-Indexed: 4321001
X-QNK-Archive-Tip: 11400000
Content-Type: application/json

{ "available": false, "reason": "block not yet backfilled",
  "lowest_indexed_height": 4321001,
  "tip_height": 11400000,
  "this_height": 5000000 }
```

If the block *is* available:
```
GET /api/v1/blocks/11399000
HTTP/1.1 200 OK
X-QNK-Archive-Status: backfilling
X-QNK-Archive-Progress: 4321000/11400000 (37.9%)
Content-Type: application/json

{ block data ... }
```

Once backfill completes, the header drops:
```
HTTP/1.1 200 OK
X-QNK-Archive-Status: complete
```

### Frontend wallet/explorer behavior

The wallet UI already pulls `/api/v1/status` for tip height, peers, etc. We extend that to include archive state, then the UI:

- Shows the **current** balance / pool prices / mempool live (always works)
- For historical look-ups, if the response is 202 with `backfilling`, the UI renders:
  > "Block #5,000,000 not yet indexed on this node. Currently at 38% archive — ETA tomorrow morning. [Query Epsilon] or [Wait and retry]"
- Provides an explicit "Query a peer node" button that proxies the look-up to Epsilon (which has full archive). Optional, opt-in, marked as "less trustless".

### Reasoning

This is honest about node capability without being a trust regression. A node operator who runs their own node and asks for block #5,000,000 from an instant-bootstrap node deserves a true answer — *"my node doesn't have it yet, here's the percentage, here's the ETA"* — not silent fallback to a different peer.

---

## Phased Rollout (Q4 2026 → Q4 2027)

### Phase 0 — Foundations (already underway, `crates/q-ivc/`)
- Finish q-ivc gadgets: NTT butterfly ✓, BLAKE3 verify_hash ✓, Dilithium high_bits/use_hint ⏳, signed-norm range-check ⏳, Merkle path ✗.
- All gadgets pass arkworks R1CS tests with positive + adversarial coverage.
- Deliverable: standalone gadget library, no integration.

### Phase 1 — Single-block proof of correctness (Nova on BN254, Q1 2027)
- Implement the δ-circuit composing all gadgets.
- Prove + verify a *single* block transition.
- Verifier runs as an integration test, not in production.
- Deliverable: `crates/q-ivc-prover/`, `crates/q-ivc-verifier/`.

### Phase 2 — IVC fold (Nova IVC, Q2 2027)
- Wrap the δ-circuit in Nova's `StepCircuit` trait.
- Generate `π_n` for `n ∈ {10, 100, 1_000, 10_000}` blocks.
- Measure: verifier latency (target ≤ 10 ms), proof size, prover throughput.
- Deliverable: working IVC tree on testnet.

### Phase 3 — Optional production integration (Q3 2027)
- Genesis nodes (Epsilon, Beta, Gamma, Delta) maintain `π_tip` alongside the chain.
- New nodes can *optionally* download `π_tip` via `GET /api/v1/proof/tip` and verify.
- Block validation still happens block-by-block on every node (proof is advisory).
- Deliverable: instant-bootstrap optional code path; old checkpoint flow still works.

### Phase 4 — Lattice migration (Q4 2027 → Q1 2028)
- Replace Nova/BN254 with LatticeFold or LaBRADOR-IVC.
- Same circuit, different proof system.
- Deliverable: full PQ stack — no elliptic curve anywhere in consensus.

### Phase 5 — Mandatory verification at activation height (2028, mainnet upgrade)
- Activation height ~12 months out.
- After height H, blocks must carry a valid recursive proof folding the prior `π_{n-1}` with their own validity.
- Old checkpoint flow deprecated.
- Deliverable: trustless chain in the strict cryptographic sense.

---

## Performance Budget — What "10 ms verify" Buys

| Hardware tier | Verify latency target | Notes |
|---------------|-----------------------|-------|
| Modern desktop / M2 / 16-core x86 | **≤ 5 ms** | Headroom |
| Laptop / 8-core mobile x86 | **≤ 10 ms** | Spec target |
| Raspberry Pi 5 / mobile ARM | **≤ 100 ms** | Acceptable for first-launch one-time cost |
| Browser WASM (no AVX) | **≤ 250 ms** | Acceptable — wallet first-load |

10 ms verify means the bootstrap process is *imperceptible* — between clicking "join network" and the node accepting transactions, the user sees nothing but the boot banner.

The hard part is **prover** cost on the genesis nodes. Folding one block must keep up with block production. With 1 BPS target:
- Prover budget per block: < 1 second on dedicated hardware
- Nova on BN254 with R1CS folding: ~5-10 seconds for a non-trivial step circuit on commodity hardware → need to optimize or accept that proof generation runs at half-speed on a dedicated prover machine
- LatticeFold prover: research-grade, no production benchmarks yet

The prover cost is the gating constraint. The plan above (Nova first, lattice second) addresses this by letting us measure prover cost on real circuits before committing to the lattice flavor.

---

## Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Lattice IVC proof system not production-ready by Q3 2027 | High | Delayed lattice migration | Ship with Nova/BN254 first, migrate later. Don't block instant-bootstrap on the lattice flavor. |
| Prover throughput < 1 BPS | Medium | Genesis prover falls behind tip | Use dedicated prover machines (Epsilon-class hardware). Distribute proving across multiple validators. |
| Verifier latency > 10 ms in WASM | Medium | Slow first-launch in browser | Use vacant CPU between page-loads, cache π_tip in IndexedDB. |
| Bug in δ-circuit accepts an invalid state transition | **Critical** | Chain-wide consensus failure if mandatory | Phase 3 keeps proof advisory for ≥ 6 months. Real block validation runs in parallel. Adversarial testing — generate millions of malformed blocks and ensure circuit rejects them. |
| Trusted setup for Nova (BN254) | Low | One-time ceremony required, not perpetually | Acceptable for Phase 1-3 (advisory mode). Phase 4 lattice migration eliminates the setup entirely. |
| Backfill takes longer than expected | Low | Long "explorer pending" UI state | Already mitigated — Phase 2 sync we just shipped is bounded. Communicate ETA honestly in HTTP headers + UI. |

---

## What This Plan Is *Not*

- **Not** a replacement for the BAL-001 state-root header. The state-root is the public input to the proof. It still goes into block headers and is gossiped normally.
- **Not** a "skip block validation" feature. Existing nodes continue to validate blocks block-by-block. The recursive proof is supplementary trust, not a replacement.
- **Not** a way to compress block storage. Old blocks still need to be backfilled if a node wants to serve historical queries. The proof attests state, not bytes.
- **Not** something that lands before Q1 2027 in any usable form. The next 6 months are gadget completion. Realistic instant-bootstrap mainnet date: late 2027 / early 2028.

---

## Immediate Next Steps (Next 30 Days)

This plan is a roadmap, not a sprint. The next actionable work is:

1. **Finish q-ivc Dilithium gadget** (in-progress, task #22): high_bits / use_hint already drafted, signed-norm range check still uses positive-only fallback. ETA: 1 week.
2. **Add Merkle-path-in-balance-map gadget** to `crates/q-ivc/src/gadgets/merkle.rs`. This is the missing piece that lets the δ-circuit prove balance updates. ETA: 3 weeks.
3. **Add a `/api/v1/status/archive` endpoint** that reports `lowest_indexed_height`, `tip_height`, `archive_eta`. This is the UX foundation for progressive archival — usable even before any SNARK work lands, because Phase 2 backfill already produces this state. ETA: 2 days.
4. **Prototype the δ-circuit on a single transfer** (no recursion yet). Goal: prove that one signed Dilithium5 transfer from A → B is valid against a known balance map. ETA: 4 weeks.

If you give the green light, I can land item 3 today as part of the v10.9.16 cycle — it's pure additive instrumentation, no consensus risk, and it lets the wallet UI start preparing for the progressive-archive UX immediately. Items 1-2-4 are the q-ivc track that's already in-flight.

— Server Beta, 2026-05-13
