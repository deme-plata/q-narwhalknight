# Consensus Layer v2 Design — Q-NarwhalKnight
**Date:** 2026-05-10  
**Status:** Design proposal — adapts the May 2026 audit recommendations to the existing codebase  
**Basis:** Technical review 2026-05-10 + external audit draft

---

## 1. Starting Point — What We Already Have

The draft proposes building a BFT consensus layer from scratch. We do not need to. The Q-NarwhalKnight codebase already contains working, initialized implementations of every hard component:

| Component | Draft proposes | We already have | Initialized | Load-bearing |
|-----------|---------------|-----------------|-------------|--------------|
| P2P transport | gossipsub | libp2p-rust + gossipsub + Kademlia DHT | ✅ | ✅ Production |
| Narwhal mempool | — | `crates/q-narwhal-core/` — `ProductionMempool`, `NarwhalCore`, `ReliableBroadcast` | ✅ `main.rs:7075` | ❌ not in commit path |
| Narwhal BFT | Bracha RB | `NarwhalCore::new_with_byzantine_threshold(node_id, Phase1, f=1)` | ✅ `main.rs:7186` — f=1, 2f+1=3 quorum, SQIsign | ❌ not in commit path |
| DAG ordering | "add later" | `crates/q-dag-knight/` — `DAGKnightConsensus`, `anchor_election`, `commit_logic`, `vertex_creator` | ✅ `main.rs:7107` — f=3 (10 validators) | ❌ not in commit path |
| Bullshark | — | `NarwhalBullsharkVm` in `q-vm` — tested in `narwhal_bullshark_test.rs` | ⚠️ test only | ❌ |
| Balance commitment | Sparse Merkle Trie | BLAKE3 `compute_balance_root_for_block()` | ✅ | ⚠️ shadow mode today |
| Storage | any KV store | RocksDB with column families | ✅ | ✅ Production |
| Block format | new format | `QBlock` with `state_root` field | ✅ | ✅ |
| Upgrade gating | unspecified | `q-consensus-guard` upgrade gate | ✅ | ✅ Production |
| Signatures | Ed25519 | Ed25519 + Dilithium5 + SQIsign (isogeny PQ) | ✅ | ✅ |

**The gap is not missing components. Every component is written and initialized. The gap is a single missing wire: block commit decisions still go through `lockfree_producer.rs` (linear N+1) rather than through DAG-Knight anchor election.**

`NarwhalCore` is live in `ServerState` with f=1 and 2f+1=3 quorum. `DAGKnightConsensus` is live with f=3 (designed for 10 validators). `ProductionMempool` is receiving transactions. None of these are consulted when deciding which block to commit.

---

## 2. How DAG-Knight Fits — Multi-Proposer Ordering

The draft proposes **round-robin single proposer** (one validator proposes per height). That is the right model for plain HotStuff/Tendermint. DAG-Knight is strictly better for this network and we already have it.

### Round-robin (draft proposal)
```
Height 100: Epsilon proposes
Height 101: Beta proposes
Height 102: Gamma proposes
Height 103: Delta proposes
```
Only one validator produces per height. Throughput = single node throughput. If proposer is offline, round times out and rotates.

### DAG-Knight multi-proposer (our system)
```
Round r:
  Epsilon  → produces vertex V_ε  (txs + ref to prior vertices)
  Beta     → produces vertex V_β  (txs + ref to prior vertices)
  Gamma    → produces vertex V_γ  (txs + ref to prior vertices)
  Delta    → produces vertex V_δ  (txs + ref to prior vertices)

DAG-Knight anchor election → selects V_ε as anchor for round r
Commit: all vertices causally before V_ε are committed in deterministic order
```

Every validator produces every round. DAG-Knight's anchor election determines **which vertex becomes the commit anchor** — not who is allowed to produce. Throughput scales with validator count. A crashed validator's vertices are simply absent from the DAG; the remaining n-1 anchor election still works.

**For n=4 (f=1): we need 3 vertices in a round to have enough causal references for the anchor rule to work. If one validator is offline, 3 vertices are available — f=1 is maintained.**

### What this means for block structure

The current system has one block per height. In the DAG-Knight model, a "block" becomes a **committed batch** — the set of transactions from all vertices in one DAG round, ordered by the anchor. The `QBlock` format does not need to change: the batch is still a `QBlock` with height, prev_hash, txs, and `state_root`. Only the source of the transactions changes (from one producer to multi-validator DAG merge).

---

## 3. Full Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     VALIDATOR SET  n=4  (f=1, need 3f+1=4)              │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │   EPSILON    │  │     BETA     │  │    GAMMA     │  │    DELTA    │ │
│  │  (genesis)   │  │ (checkpoint) │  │ (checkpoint) │  │  (bootstrap)│ │
│  │  89.149.x.x  │  │ 185.182.x.x  │  │ 109.205.x.x  │  │  5.79.x.x   │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘ │
│         │                 │                  │                 │        │
│         └────────────────┬┴──────────────────┴─────────────────┘        │
│                          │  libp2p gossipsub (already working)           │
│                          ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                  DAG-KNIGHT LAYER                                 │  │
│  │  Each validator produces vertices every round (txs + DAG refs)   │  │
│  │  Anchor election → selects commit vertex for this round          │  │
│  │  Bracha RB (f=1) ensures all honest nodes receive each vertex    │  │
│  │  Committed anchor → deterministic tx ordering across all nodes   │  │
│  └───────────────────────────┬───────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                  EXECUTION LAYER                                  │  │
│  │  Apply committed txs in anchor order (ALL txs — no silent skip)  │  │
│  │  Invalid tx → mark FAILED in execution receipt, still included   │  │
│  │  compute_balance_root_for_block() → new balance_root             │  │
│  │  State written to RocksDB                                        │  │
│  └───────────────────────────┬───────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                  COMMIT & BROADCAST                               │  │
│  │  QBlock { height, txs, prev_hash, balance_root, signatures }     │  │
│  │  ≥ 2f+1 validators sign the committed block (threshold sig)      │  │
│  │  Gossipsub broadcasts to full network                            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component-by-Component Design

### 4.1 DAG-Knight — Vertex Production and Anchor Election

Each validator runs the existing DAG-Knight engine. The change: **DAG-Knight's output is the block ordering, not an advisory metric**.

**Vertex structure:**
```rust
pub struct DagVertex {
    pub producer_id: PeerId,
    pub round: u64,
    pub txs: Vec<Transaction>,
    pub strong_refs: Vec<VertexHash>,  // references to prior round vertices
    pub weak_refs: Vec<VertexHash>,    // optional causal refs
    pub signature: Signature,
}
```

**Anchor election rule** (from DAG-Knight paper — already in codebase):  
A vertex V is an anchor if it is referenced (directly or indirectly) by ≥ f+1 vertices in the next two rounds. When an anchor is elected, all vertices in its causal history are committed in a deterministic topological order.

**What this gives us:**  
- No single proposer — Epsilon offline means 3/4 validators still elect anchors
- Throughput = sum of all validator mempools, not one mempool
- Finality latency ≈ 2 rounds (≈ 2× network RTT between validators)

### 4.2 Bracha Reliable Broadcast — Upgrade to f=1

Current: `BalanceFinalityEngine::new(0, ...)` — f=0, single message delivers immediately.  
Target: `BalanceFinalityEngine::new(1, ...)` — f=1, requires 2f+1=3 ECHO messages before READY, 2f+1=3 READY messages before DELIVER.

This is a **one-line change** in `main.rs:3636` plus updating the quorum thresholds in `balance_finality.rs`.

**Effect:** A Byzantine validator cannot cause an honest validator to deliver a vertex that other honest validators did not receive. This is the safety property that binds DAG-Knight's anchor election to correct execution.

**Quorum requirements for n=4, f=1:**
```
ECHO quorum:   2f+1 = 3 of 4 validators
READY trigger: f+1  = 2 of 4 validators
READY quorum:  2f+1 = 3 of 4 validators
DELIVER:       after 2f+1 READY
```

### 4.3 BalanceRootV1 — From Shadow Mode to Enforcement

**Current state (as of today):** Shadow mode — compute and log, no rejection.  
**Target state:** Part of the PREVOTE-equivalent in DAG-Knight.

When a validator receives a committed anchor from DAG-Knight, it must:
1. Re-execute all transactions in the committed batch (same order — deterministic)
2. Compute its own `balance_root_candidate` using `compute_balance_root_for_block()`
3. If `candidate != anchor.balance_root` → do NOT add this anchor to its local DAG, log CRITICAL alert
4. If they match → add anchor, write state, gossip committed block

This replaces the draft's "PREVOTE(nil)" with "do not reference this vertex in future rounds." In DAG-Knight, an unreferenced vertex cannot become an anchor — this is the liveness-safe equivalent of a PREVOTE rejection.

**The balance root is the consensus mechanism.** A Byzantine node that produces blocks with wrong balance roots gets no references from honest validators and cannot elect anchors.

### 4.4 Transaction Execution — No Silent Skip

The draft correctly identifies this as P0. The current `continue;` at `balance_consensus.rs:857` is a consensus break.

**Target rule:**
```rust
match apply_transfer(&tx, &mut state) {
    Ok(_) => receipt.push(TxResult::Success(tx.hash)),
    Err(InsufficientFunds) => receipt.push(TxResult::Failed(tx.hash, "insufficient_funds")),
    Err(e) => return Err(e), // abort block — should not happen with valid mempool
}
```

Every validator applies the same rule deterministically. A transaction with insufficient funds produces `TxResult::Failed` — it is still included in the block and the tx_root covers it. Two validators executing the same block **must** produce the same post-state and the same balance_root.

### 4.5 Quorum-Signed Checkpoints

Current: `CHECKPOINT_DATA` — hardcoded static array signed by no one, from Epsilon alone.  
Target: A checkpoint is valid only if it carries signatures from ≥ 2f+1 validators over `(height, balance_root, state_hash)`.

```rust
pub struct Checkpoint {
    pub height: u64,
    pub balance_root: [u8; 32],      // from compute_balance_root_for_block()
    pub validator_signatures: Vec<(PeerId, Signature)>, // ≥ 2f+1 required
}
```

New nodes joining:
1. Download the most recent quorum-signed checkpoint
2. Verify ≥ 2f+1 signatures from the known validator set
3. Apply checkpoint state to RocksDB
4. Set `checkpoint_applied = true`
5. Sync remaining blocks from checkpoint height to tip

This eliminates the "single trust source" problem — Epsilon's checkpoint was unilateral. A future checkpoint requires agreement from 3 of 4 validators.

### 4.6 Divergence Detection and Repair

**Detection** (already partially implemented):
- Every 60s: compute local balance_root, request signed roots from all validators
- If local root ≠ majority root → DIVERGED state (log + alert + halt block production)

**Repair** — two levels:

*Level 1: Full resync from quorum checkpoint*  
If divergence is large (many blocks off), download the latest quorum-signed checkpoint and replay from there. Safe because the checkpoint is signed by 2f+1 validators.

*Level 2: Targeted wallet repair (requires SMT — future work)*  
The current BLAKE3 balance root is a flat hash over all balances. It proves the entire state is committed but cannot prove a single wallet's balance without sending all balances. For targeted repair, we need a **Sparse Merkle Trie** (SMT) where each leaf is a wallet and internal nodes are hashes of children. An SMT supports Merkle proofs: "wallet X has balance Y as of root Z" — verifiable with O(log n) hashes.

**This is the main gap between our current BalanceRootV1 and the draft's proposal.**  
The path: BalanceRootV1 (BLAKE3, detection only) → BalanceRootV2 (SMT, targeted repair).  
We do not need to block on this — BLAKE3 detection is the critical first step.

---

## 5. Mapping to Existing Code

| Design component | File | Current state | Change needed |
|------------------|------|---------------|---------------|
| DAG-Knight consensus | `crates/q-dag-knight/src/` — `anchor_election.rs`, `commit_logic.rs` | Initialized in `main.rs:7107` with f=3, stored in `ServerState.dag_knight` | Wire `commit_logic` output as block commit trigger instead of `lockfree_producer.rs` N+1 |
| Narwhal mempool | `crates/q-narwhal-core/src/production_mempool.rs` | Initialized in `main.rs:7075`, `ServerState.production_mempool` | Make block producer pull batches from mempool instead of local tx queue |
| NarwhalCore BFT | `crates/q-narwhal-core/src/lib.rs` | Initialized in `main.rs:7186` with f=1, 2f+1=3 quorum, SQIsign | Wire `certificate` production into gossip block handler — received blocks must carry certificate before commit |
| BalanceFinalityEngine | `crates/q-api-server/src/main.rs:3636` | `BalanceFinalityEngine::new(0, ...)` — f=0 shadow | Change to `new(1, ...)` — already have the f=1 logic in `NarwhalCore` |
| BalanceRootV1 shadow | `crates/q-api-server/src/main.rs:11364` | ✅ Shadow mode (today) | Flip `warn!` → `return` after soak confirms root agreement |
| BalanceRootV1 producer | `crates/q-api-server/src/block_producer.rs:967` | ✅ Computes root | Already done |
| Storage wiring | `crates/q-api-server/src/lockfree_producer.rs:592` | ✅ Wired | Already done |
| Transfer silent skip | `crates/q-storage/src/balance_consensus.rs:857` | `continue;` — consensus break | Replace with `TxResult::Failed` — deterministic execution required before enforcement |
| Multi-producer | `crates/q-api-server/src/lockfree_producer.rs` | Epsilon only produces | Wire Beta/Gamma/Delta to run `vertex_creator.rs` and submit to Narwhal mempool |
| Quorum checkpoints | `crates/q-storage/src/balance_checkpoint.rs` | Single-node `CHECKPOINT_DATA` | Add `Vec<(ValidatorId, Signature)>` field; require 2f+1 sigs to accept |
| Signed balance API | `crates/q-api-server/src/integrity_api.rs` | Unsigned compute | Add node Ed25519/SQIsign signature to `/api/v1/integrity/balance-root` response |

---

## 6. Fault Tolerance Matrix (Target State)

| Scenario | n=4 f=1 behaviour |
|----------|-------------------|
| One validator offline | 3 remain; DAG-Knight anchor election still works (need 2f+1=3 references). Chain continues. |
| One Byzantine proposer | Produces vertex with wrong balance_root → honest validators refuse to reference it → cannot become anchor → no commit. Chain continues with other vertices. |
| One Byzantine voter | Signs conflicting anchors → equivocation proof → slashing (future). With f=1, still needs 2 more honest validators to commit anything. |
| Network partition (2/2 split) | Neither partition has 2f+1 = 3 validators → neither side commits. Chain halts (safely). Heals when partition resolves. |
| Balance corruption on one node | Node's balance_root diverges → it cannot verify incoming anchors → logs DIVERGED → operator resyncs. Other 3 validators continue producing. |
| Replay bug like May 9 | Balance_root changes on affected node → next block from that node rejected by 3 honest validators → cannot commit → bug is detected and contained immediately. |

---

## 7. Implementation Roadmap

Phased to preserve network continuity — each phase is independently deployable.

```
Phase 1 — Foundation (NOW, in progress)
────────────────────────────────────────
✅ BalanceRootV1 shadow mode (committed today)
✅ Gate bumped to height 20,000,000
🔄 Build on Epsilon q2 (running)
⏳ Fix transfer `continue` → TxResult::Failed  [P0 — do before Phase 2]
⏳ Deploy + soak (watch for SHADOW MISMATCH in logs)

Phase 2 — Multi-producer (2 weeks)
────────────────────────────────────────
⏳ Wire Beta and Gamma as co-vertex-producers
⏳ DAG-Knight anchor election becomes the commit decision (not block_producer.rs N+1)
⏳ All 4 validators run `produce_vertex()` every round
⏳ Committed anchor → QBlock written to chain

Phase 3 — State enforcement (2 weeks after Phase 2 soak)
────────────────────────────────────────
⏳ Bracha f=0 → f=1 (one-line change + quorum math)
⏳ BalanceRootV1: shadow → enforce (flip `warn!` → `return`)
⏳ Validators reject vertices with mismatched balance_root
⏳ New enforcement gate height set after soak confirms root agreement

Phase 4 — Checkpoint hardening (3 weeks)
────────────────────────────────────────
⏳ Quorum-signed checkpoints (2f+1 signatures over height + balance_root)
⏳ New node bootstrap verifies checkpoint signatures before applying
⏳ Remove single-node CHECKPOINT_DATA trust dependency

Phase 5 — Targeted repair / SMT (4 weeks, parallel)
────────────────────────────────────────
⏳ BalanceRootV2: Sparse Merkle Trie replacing flat BLAKE3
⏳ Merkle proof API: prove wallet balance against committed root
⏳ Auto-repair: on divergence, request proofs from 2f+1 peers, apply corrections
```

**Estimated total: 10-11 weeks to full BFT consensus with automatic divergence repair.**  
Each phase is independently valuable and independently deployable.

---

## 8. What the Draft Proposed vs. What We Should Build

| Draft | Our adapted design |
|-------|-------------------|
| Round-robin single proposer | **DAG-Knight multi-proposer** — every validator produces every round |
| HotStuff/Tendermint voting rounds | **DAG-Knight anchor election** — implicit voting via DAG references |
| Sparse Merkle Trie from day 1 | **BLAKE3 now (Phase 1-3), SMT later (Phase 5)** — don't block on SMT |
| "Replace Bracha with standard BFT" | **Keep Bracha, fix f=0 → f=1** — Bracha RB IS the reliable broadcast primitive |
| Build gossipsub | **Already have it** — libp2p-rust in production |
| n ≥ 4 validators | **n=4: Epsilon + Beta + Gamma + Delta** — use what we have |

---

## 9. Why DAG-Knight + Bracha is Better Than HotStuff for This Network

The draft suggests HotStuff because it is simpler. For a **4-node, mining-reward, continuous-block** network, DAG-Knight wins on three points:

1. **Throughput under load**: All 4 validators produce simultaneously. A mining surge that hits one node propagates to the DAG and gets committed regardless of who is the "round proposer." HotStuff forces all transactions through one proposer per round.

2. **Graceful degradation**: If Epsilon (the 10Gbit supernode) is the HotStuff proposer and goes down, the round times out before rotating. In DAG-Knight, Epsilon's absence means 3 vertices instead of 4 — the round still completes.

3. **We already have it**: The DAG-Knight code exists, is partially tested, and understands the network's block format. Replacing it with HotStuff would require implementing a new consensus engine from scratch. Wiring the existing one is 1/10th the work.

The tradeoff: DAG-Knight's finality proof is more complex than HotStuff's explicit PRECOMMIT certificates. For audit and verification purposes, explicit signatures (as in the checkpoint design above) supplement this.

---

## 10. Summary

The external draft is correct about the destination. It is wrong about the path — it proposes replacing what we have with standard primitives when what we have, properly wired, is better than those primitives.

The May 9 incident happened because DAG-Knight was not in the commit path, Bracha was set to f=0, and BalanceRootV1 was not enforced. The fix is to wire the existing components, not to rebuild them.

After Phase 3 (enforced balance root + f=1 Bracha + DAG-Knight anchoring):  
A Byzantine Epsilon producing a corrupted block → other validators refuse to reference its vertex → it cannot be elected anchor → it cannot commit. The May 9 scenario becomes a consensus failure with a loud error, not a silent balance overwrite.

**That is the target state.**
