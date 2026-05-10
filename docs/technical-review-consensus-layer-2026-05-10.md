# Technical Review: Consensus Layer State — May 10, 2026

**Prepared by:** Multi-agent codebase audit  
**Scope:** Block consensus, balance state agreement, fault tolerance, recovery mechanisms  
**Motivation:** The May 9 balance replay incident destroyed correct wallet balances on Epsilon and revealed that no other node held correct state to recover from. This document explains why, and what needs to be built.

---

## 1. Executive Summary

After weeks of work on sync reliability, block propagation, and replay logic, the consensus layer has a clear and honest status: **block ordering is solved; balance state agreement is not.**

The network reaches consensus on *which blocks exist and in what order* via DAG-Knight and gossipsub. That part works. But there is **no Byzantine-fault-tolerant mechanism for agreeing on wallet balances** — the state that results from applying those blocks. Each node computes balances independently. When nodes diverge, the system detects it (via balance hash comparison) but does nothing to correct it.

The practical consequence: Epsilon has been the de facto ground truth for wallet state since genesis. When its balances were corrupted, there was no peer to recover from. The three other nodes (Beta, Gamma, Alpha) either had incomplete state (checkpoint-bootstrapped, missing early history) or no mechanism to assert their values as authoritative.

This is not a bug that was introduced recently. It is a gap that was always present in the design, masked by the fact that Epsilon had never diverged before.

---

## 2. What Works Today

**Block production and propagation** function correctly:
- Epsilon produces blocks sequentially by height
- Blocks are gossiped to all peers via gossipsub (`/qnk/mainnet-genesis/blocks`)
- Peers validate block signatures and store blocks in RocksDB
- Turbo sync fills gaps in block history when nodes are behind

**P2P connectivity** is healthy:
- 24 peers connected to Epsilon at time of audit
- Kademlia DHT handles peer discovery
- Block-pack (turbo sync) protocol delivers historical blocks reliably

**Balance tracking** works in the happy path:
- Mining rewards embedded in blocks are applied deterministically on every node
- SSE stream broadcasts mining events in real time
- In-memory wallet balances stay consistent with RocksDB under normal operation

---

## 3. The Consensus Gap — What Is Not Implemented

### 3.1 Block Production Is Centralized

`default_total_validators = 1` in `crates/q-api-server/src/config.rs`. The four "producers" in the block producer pool are four parallel workers within a single Epsilon process — not four independent validators. Epsilon is the only node that produces blocks. If it goes offline, block production halts entirely.

**Impact:** No blocks = no mining rewards, no confirmed transactions, no chain progress. The other three nodes (Beta, Gamma, Gamma) are passive consumers of Epsilon's block stream, not co-producers.

### 3.2 Block Validation Does Not Verify Balance Correctness

When a node receives a block via gossip (`crates/q-api-server/src/main.rs` gossipsub handler), it validates:
- Block signature (Ed25519)
- Network ID match
- Height monotonicity

It does **not** validate whether the transactions in the block are affordable. A block containing a transfer from an address with zero balance will pass validation. The invalid transfer is silently skipped during processing (`continue` in `balance_consensus.rs:857`), meaning the producing node and the receiving node end up with different balance states — a consensus divergence from a single block.

### 3.3 DAG-Knight Is Not Driving Block Ordering

The DAG-Knight code exists in `crates/q-dag-knight/` and is wired into the AppState. But block production in `block_producer.rs` does not call DAG-Knight's ordering engine to determine the next canonical block. Blocks are produced at `current_height + 1` deterministically. DAG-Knight runs in parallel and logs metrics, but the main chain is a simple linear chain, not a DAG with BFT-ordered finality.

The `produce_blocks()` loop has a 60-second timeout that's designed to work around DAG-Knight hanging (comment at main.rs line 16970). In practice, block production proceeds without waiting for DAG-Knight confirmation.

**Impact:** The theoretical guarantees of DAG-Knight (BFT finality, O(1) amortized communication) are not in effect. The chain is linear with a single producer — equivalent to a simple PoW-style chain without the mining competition.

### 3.4 Bracha Reliable Broadcast Is Neutered

The Bracha RB engine in `balance_finality_engine.rs` is architecturally sound. It implements the correct three-phase protocol (SEND → ECHO → READY) with proper signature handling. But it is initialized with:

```rust
BalanceFinalityEngine::new(
    0,  // f = 0 (shadow mode)
    ...
)
```

With `f = 0`:
- Echo quorum = `2f+1 = 1`
- Ready amplify = `f+1 = 1`

A single message from a single node immediately delivers and writes to RocksDB. This is not Byzantine fault-tolerant — it is a simple gossip relay. The Bracha protocol provides ordering guarantees here, not safety guarantees.

Additionally, Bracha only handles *non-block* balance updates (DEX credits, out-of-band rewards). Mining rewards come from blocks and bypass the Bracha path entirely.

**Status:** Bracha is deployed but provides zero Byzantine resilience. Raising `f` to 1 would require a hard fork (nodes with `f=0` and `f=1` have incompatible quorum expectations).

### 3.5 Balance Divergence Detection Does Not Trigger Recovery

Every 5 minutes, each node calls `do_combined_state_sync()`, which includes a divergence check:

```rust
// state_sync_api.rs ~line 983
if &our_hash != peer_hash {
    error!("🚨 [DIVERGENCE CHECK] CRITICAL: Balance hash MISMATCH with peer!");
    error!("   Our hash:  {}", &our_hash[..24]);
    error!("   Peer hash: {}", &peer_hash[..24]);
    error!("   Run convergence migration to fix: ...");
    // ← nothing else happens here
}
```

The check computes a BLAKE3 hash of all wallet balances, compares it with a peer's hash, and logs a CRITICAL error if they differ. **It does not correct the divergence.** No peer querying, no majority vote, no automatic repair. The error sits in the log while both nodes continue with their inconsistent views of wallet state.

### 3.6 CHECKPOINT_DATA Is a Static, Unverified Snapshot

The balance checkpoint embedded in the binary (`crates/q-storage/src/balance_checkpoint.rs`) contains 1,326 wallet balances at height 16,538,868. It was generated offline from Epsilon's RocksDB and has a SHA256 hash for integrity checking — but the hash proves only that the data hasn't changed since it was embedded, not that it was correct when generated.

No validator signatures from Beta or Gamma were collected when the checkpoint was created. There is no quorum certificate. This is a unilateral snapshot from a single node, distributed as a binary constant.

When nodes bootstrap from this checkpoint, they are trusting Epsilon's historical state implicitly. If Epsilon had a bug at checkpoint height, that bug is now permanent across all checkpoint-bootstrapped nodes.

### 3.7 FullStateSnapshot Sync Is Add-Only

Every 5 minutes, nodes pull `GET /api/v1/sync/full-state` from peers. The merge logic is:

```rust
// For each wallet in snapshot:
// - If we already have a balance for this wallet: KEEP OURS (never overwrite)
// - If we don't have this wallet: ADD IT
```

This prevents a fast-syncing or malicious peer from overwriting correct balances with lower values — intentional after the replay incident. But it also means **a node with corrupted state can never be corrected by peers**. If Epsilon had wallet X at 1484 QUG (wrong) and Gamma had wallet X at 3200 QUG (correct), Epsilon would keep 1484 forever under this merge policy, because it "already has a value."

The only escape from this is `Q_BALANCE_AUTHORITY_PEER`, which does a full overwrite — but that is a manual one-time operation, not consensus.

### 3.8 Transfer Transactions Are Not Gossiped Until In a Block

When a user submits a transfer to a node, the node adds it to its local mempool (`tx_pool`). The transaction is included in the next block Epsilon produces. Until it appears in a block, **other nodes have no knowledge of it**.

If Epsilon crashes between receiving the transaction and producing the next block, the transaction is permanently lost. The user's wallet shows the debit on Epsilon's side (possibly) but the network never confirms it.

---

## 4. Why the Epsilon Incident Was Inevitable

The replay bug that corrupted Epsilon's wallet balances is fully documented in `docs/incident-report-balance-replay-2026-05-09.md`. But the deeper reason the incident caused lasting damage is architectural:

1. **Epsilon is the sole block producer.** Its view of world state propagates to all nodes through the blocks it produces. There is no other node to disagree with it at the block level.

2. **Block validation doesn't check balance correctness.** Even if Epsilon produced a block with wrong coinbase amounts, other nodes would accept it.

3. **Divergence detection only logs.** Beta and Gamma's balance hash check disagreed with Epsilon after the corruption, but they had no mechanism to vote Epsilon out or select a correct peer.

4. **The "correct" state existed on no single node.** Gamma had the right checkpoint-era balances. Epsilon had the right genesis-era balances before corruption. Neither had the complete correct picture without the other.

5. **Recovery required manual guesswork.** There was no quorum certificate, no Merkle proof, no `2f+1` agreement on the pre-corruption state.

This is the definition of a system with `f = 0` fault tolerance: any single node's failure breaks the whole thing.

---

## 5. Current Fault Tolerance

| Metric | Target | Actual |
|--------|--------|--------|
| Block producers | ≥ 3 (f=1) | 1 (Epsilon only) |
| Balance BFT threshold (f) | 1 (tolerate 1 faulty node) | 0 (any node failure is fatal) |
| Nodes needed for balance recovery | 2 (majority of 3) | N/A — no recovery mechanism |
| Transaction loss on node crash | Should be 0 | 100% of mempool at crash time |
| Auto-correction of divergence | Should be automatic | Never happens |
| Checkpoint verification | Should require 2f+1 sigs | None — generated by 1 node |

---

## 6. What Needs to Be Built — Prioritized

### Priority 1: Multi-Producer Block Production (f=1 Block Layer)

**What:** Enable at least Beta and Gamma to produce blocks, not just Epsilon.

**How:**
- In `config.rs`, change `total_validators` to 3 (Beta, Gamma, Epsilon)
- Assign each node a `validator_index` (0, 1, 2)
- Implement round-robin or VRF-based leader election so one validator is block producer per round
- Each node only produces when it is the elected leader for that round

**Complexity:** Medium. The block producer infrastructure exists; wiring leader election is the new work.

**Impact:** Network survives Epsilon going offline. Block production continues with Beta and Gamma.

### Priority 2: Balance Root in Block Headers (Cryptographic State Commitment)

**What:** The block header's `state_root` field exists but is not consistently populated by the producer or enforced by receivers. Make it mandatory.

**How:**
- After every block's transactions are applied, compute `compute_balance_root_for_block()` (already exists in `lib.rs`)
- Include this as `state_root` in the block header
- On block receipt, receiving nodes compute their own state_root after applying the block's transactions
- Reject blocks where `computed_state_root != block.header.state_root`
- This makes balance divergence a block rejection, not a silent mismatch

**Complexity:** Low-Medium. The computation already exists. The validation enforcement is the new code (currently the mismatch logs an error but does not reject).

**Impact:** Blocks now cryptographically commit to the resulting balance state. A node with divergent state will refuse new blocks until it reconciles — making divergence visible and self-correcting.

**Warning:** This will cause block rejections at first until all nodes are consistent. Deploy after reconciling all nodes' balances.

### Priority 3: Bracha f=1 with Consensus Upgrade

**What:** Raise `f` in `BalanceFinalityEngine` from 0 to 1. This requires 3 nodes to echo a balance update before it is finalized.

**How:**
- Change `BalanceFinalityEngine::new(0, ...)` to `BalanceFinalityEngine::new(1, ...)`
- Echo quorum becomes `2*1+1 = 3`
- Ready amplify becomes `1+1 = 2`
- This requires 3 active validators (Beta, Gamma, Epsilon) to process any balance update

**Complexity:** Low in code — one integer change. High in deployment — requires all nodes to upgrade simultaneously (hard fork for the consensus topic).

**Impact:** Any single node's corruption or equivocation fails to finalize a false balance update. Requires coordination with Priority 1 (need 3 producers to have 3 Bracha participants).

### Priority 4: Peer Majority Balance Reconciliation

**What:** When a node detects a balance hash mismatch, instead of just logging it, query all known validators for their balance of each divergent wallet and accept the majority value.

**How:**
- Add a new P2P endpoint: `GET /api/v1/consensus/wallet-balance?address={hex}` that returns the node's RocksDB value with its validator signature
- When divergence is detected, call this endpoint on all known validators for all divergent wallets
- If 2 of 3 nodes agree on a value, write that value locally (with max-wins guard — only accept if ≥ current)
- This is the missing "corrective" step in the divergence check

**Complexity:** Medium. New endpoint, new reconciliation loop in `state_sync_api.rs`.

**Impact:** Nodes can self-heal after divergence without manual intervention. The Epsilon incident recovery would have been: detect mismatch → query Beta/Gamma → 2-of-3 agree on 3200 QUG → write 3200 to Epsilon's RocksDB.

### Priority 5: Multi-Node Checkpoint Generation

**What:** Future checkpoints should require 2-of-3 validator signatures.

**How:**
- Add a checkpoint generation endpoint that returns the balance snapshot with the validator's Ed25519 signature
- Require signatures from at least 2 validators before embedding in the binary
- Embed the validator signatures alongside the data in `balance_checkpoint.rs`
- Verify signatures on load (not just SHA256 of data)

**Complexity:** Medium. New tooling for checkpoint generation.

**Impact:** No single node controls the trusted starting state for new joiners.

---

## 7. What Can Be Done Without a Hard Fork

Priorities 1, 2, and 4 can be deployed without a protocol-level hard fork:

- **Multi-producer** requires configuration changes and new leader election code, but is backward-compatible for receiving nodes
- **State root enforcement** can be deployed as a soft warning first (log but don't reject), then hardened to rejection after all nodes converge
- **Peer majority reconciliation** is additive — a new optional endpoint plus new logic in the divergence handler

Priority 3 (Bracha f=1) is the hard fork because it changes quorum thresholds. All nodes must upgrade simultaneously.

---

## 8. Immediate Stabilization (Before Restructuring)

While the full consensus layer is being built, these operational measures reduce the risk of another incident:

1. **Daily balance snapshot to all nodes:** Write a cron job that runs `GET /api/v1/sync/full-state` against Epsilon and writes the output to Gamma and Beta's local files. Not a consensus mechanism, but creates a daily backup of Epsilon's authoritative state.

2. **Alert on balance hash mismatch:** Make the divergence check send an email/webhook immediately, not just a log. This would have flagged the Epsilon corruption within 5 minutes instead of 12 hours.

3. **Mandatory balance integrity check before replay:** Add `if !is_checkpoint_applied() { return Ok(0); }` at the very top of `replay_post_checkpoint_balances()` as a hard guard (in addition to the existing check). Never run replay on Epsilon under any circumstances.

4. **Restore user wallet:** The 1,716 QUG gap in the user's wallet requires a targeted repair: query Gamma's RocksDB for the correct value, then write it to Epsilon using a max-wins targeted write tool. Provide wallet address to proceed.

---

## 9. Honest Assessment

The system has a solid foundation: RocksDB storage, gossipsub P2P, turbo sync, a working block format, SSE streaming, and the Bracha infrastructure. None of that needs to be thrown away.

What's missing is the layer that ties it together into a fault-tolerant network: leader election for multi-producer blocks, state root enforcement that makes balance divergence cause block rejection rather than a silent log entry, and a reconciliation mechanism that lets nodes correct themselves when they fall out of sync.

This is approximately 4-8 weeks of focused engineering:
- 1-2 weeks: multi-producer leader election + Priority 2 state root enforcement
- 1-2 weeks: peer reconciliation endpoint + divergence auto-correction  
- 2-3 weeks: Bracha f=1 coordinated upgrade + testing across all nodes
- 1 week: multi-node checkpoint tooling

Until then, the operational risk is: Epsilon is a single point of failure for both block production and balance state truth. The improvements from this session (max-wins guard, is_checkpoint_applied guard) prevent the specific replay bug from recurring, but do not address the underlying architecture.
