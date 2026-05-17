# Quillon Consensus System — How It Works
**Date:** 2026-05-10

---

## Diagram 1 — The Network (Target State)

Every node is equal. No node is trusted more than any other.
A user can connect to any of them and get a verifiable answer.

```
                        USER / MINER
                            │
                    ┌───────┼───────┐
                    │               │
              quillon.xyz      direct API
              (any node)       (any node)
                    │               │
    ────────────────┴───────────────┴────────────────
    │                                               │
    ▼                                               ▼

┌─────────────────┐   gossip   ┌─────────────────┐
│    EPSILON      │◄──────────►│      BETA        │
│  genesis node   │            │  checkpoint node │
│  10 Gbit        │            │  100 Mbit        │
│                 │            │                  │
│ produces vertex │            │ produces vertex  │
│ verifies blocks │            │ verifies blocks  │
│ signs commits   │            │ signs commits    │
└────────┬────────┘            └────────┬─────────┘
         │                              │
         │          gossip              │
         │    ┌─────────────────┐       │
         └───►│     GAMMA       │◄──────┘
         ┌───►│  checkpoint node│◄──────┐
         │    │  1 Gbit         │       │
         │    │                 │       │
         │    │ produces vertex │       │
         │    │ verifies blocks │       │
         │    │ signs commits   │       │
         │    └────────┬────────┘       │
         │             │                │
         │    ┌────────┴────────┐       │
         └────┤     DELTA       ├───────┘
              │  bootstrap node │
              │  1 Gbit         │
              │                 │
              │ produces vertex │
              │ verifies blocks │
              │ signs commits   │
              └─────────────────┘

  Rule: a block is only FINAL when 3 of 4 nodes independently
  computed the same balance_root and signed it.
  If any node is corrupted — the other 3 reject its blocks.
```

---

## Diagram 2 — One DAG-Knight Round

Every validator produces a vertex every round.
No single proposer. No waiting for one node.

```
ROUND r

  EPSILON ──────► Vertex Vε { txs: [T1, T5, T9],  refs: [V*r-1] }
  BETA    ──────► Vertex Vβ { txs: [T2, T6, T10], refs: [V*r-1] }
  GAMMA   ──────► Vertex Vγ { txs: [T3, T7, T11], refs: [V*r-1] }
  DELTA   ──────► Vertex Vδ { txs: [T4, T8, T12], refs: [V*r-1] }

Each vertex is broadcast via Bracha RB.
Delivery requires 3-of-4 nodes to ECHO the vertex.
A Byzantine node cannot inject a vertex only some nodes see.

                  Bracha RB (f=1)
                  ┌─────────────────────────────────┐
                  │  sender broadcasts Vε            │
                  │  3 nodes send ECHO(Vε)           │
                  │  2 nodes send READY(Vε)  ──────► │ DELIVER Vε
                  │  3 nodes send READY(Vε)          │ (all honest nodes)
                  └─────────────────────────────────┘

ROUND r+1  — anchor election

  Each validator looks at round r vertices.
  An anchor is a vertex referenced by ≥ 3 round-r+1 vertices.

  Vε is referenced by Vβ', Vγ', Vδ'  →  Vε is elected ANCHOR

  Commit rule:
  All transactions causally before Vε are committed
  in a deterministic topological order.

  Same rule, same DAG, same order on ALL nodes. ✓
```

---

## Diagram 3 — Balance Root Verification

This is what makes the system trustless.
Every validator independently checks the math.

```
COMMITTED BATCH (from DAG-Knight anchor)
ordered txs: [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12]

                    ┌─────────────────────────────────────────────┐
                    │  Each validator runs the SAME execution:    │
                    │                                             │
                    │  start_state = balance_root[prev block]     │
                    │                                             │
                    │  for each tx in ordered_txs:                │
                    │    if tx.from.balance >= tx.amount:         │
                    │      debit tx.from                          │
                    │      credit tx.to                           │
                    │      result = Success                        │
                    │    else:                                     │
                    │      result = Failed  (no balance change)   │
                    │                                             │
                    │  balance_root = BLAKE3(all wallet balances) │
                    └─────────────────────────────────────────────┘

EPSILON computes:  balance_root = 0xABCD...
BETA    computes:  balance_root = 0xABCD...   ✓ match
GAMMA   computes:  balance_root = 0xABCD...   ✓ match
DELTA   computes:  balance_root = 0xABCD...   ✓ match

All 4 agree → 3-of-4 sign the block → FINAL


What if one node is wrong?

EPSILON computes:  balance_root = 0xABCD...
BETA    computes:  balance_root = 0x1234...   ✗ MISMATCH
GAMMA   computes:  balance_root = 0xABCD...
DELTA   computes:  balance_root = 0xABCD...

Beta refuses to sign → only 3 signatures → block still commits
Beta logs DIVERGED → operator notified → targeted repair

What if Epsilon is corrupted?

EPSILON computes:  balance_root = 0xDEAD...   ✗ (corrupt)
BETA    computes:  balance_root = 0xABCD...
GAMMA   computes:  balance_root = 0xABCD...
DELTA   computes:  balance_root = 0xABCD...

Epsilon's block is REJECTED by Beta, Gamma, Delta
Epsilon cannot elect an anchor (vertices not referenced)
Chain continues on the 3 honest nodes
Operator is alerted — May 9 scenario is impossible
```

---

## Diagram 4 — From Transaction to Final Block

```
USER submits transaction T
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│  MEMPOOL (Narwhal ProductionMempool)                           │
│  Transaction gossiped to all 4 validators via libp2p           │
│  Each validator holds T in its local mempool                   │
└────────────────────────────────────────────────────────────────┘
         │
         ▼  (every DAG-Knight round, ~1 second)
┌────────────────────────────────────────────────────────────────┐
│  VERTEX PRODUCTION                                             │
│  Each validator picks txs from its mempool                     │
│  Assembles vertex with txs + references to prior vertices      │
│  Signs vertex with its key                                     │
│  Broadcasts via Bracha RB (requires 3-of-4 ECHO)              │
└────────────────────────────────────────────────────────────────┘
         │
         ▼  (every 2 rounds, ~2 seconds)
┌────────────────────────────────────────────────────────────────┐
│  ANCHOR ELECTION (DAG-Knight)                                  │
│  One vertex from round r elected as anchor                     │
│  All txs in anchor's causal history are committed              │
│  Order is deterministic — same on every node                   │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│  EXECUTION                                                     │
│  Apply committed txs in anchor order                           │
│  Invalid tx → TxResult::Failed (no balance change, recorded)  │
│  Valid tx   → TxResult::Success (balances updated)            │
│  Compute balance_root = BLAKE3(all balances sorted by address) │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│  QUORUM SIGNING                                                │
│  Validator broadcasts:                                         │
│    COMMIT(height, balance_root, prev_hash, signature)          │
│                                                                │
│  Wait for 3-of-4 matching COMMIT messages                      │
│  Assemble QBlock with 3 validator signatures                   │
│  Write to RocksDB as FINAL (cannot be reversed)               │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│  BROADCAST                                                     │
│  Final block gossiped to all peers (light clients, miners)     │
│  Anyone can verify: check 3 signatures against known keys      │
│  Anyone can verify: re-execute txs, check balance_root         │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
  USER sees confirmed balance
  Verifiable against block signature — no trust required
```

---

## Diagram 5 — New Node Bootstrap (No Trust Required)

```
NEW NODE joins the network
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  FETCH QUORUM CHECKPOINT                                │
│                                                         │
│  Request latest checkpoint from peers                   │
│                                                         │
│  Checkpoint {                                           │
│    height:    16,538,868                                │
│    balance_root: 0xABCD...                              │
│    signatures: [                                        │
│      Epsilon: sig_ε,   ← verifiable against known key  │
│      Beta:    sig_β,   ← verifiable against known key  │
│      Gamma:   sig_γ,   ← verifiable against known key  │
│    ]  ← 3-of-4: quorum proven                          │
│  }                                                      │
│                                                         │
│  Verify: all 3 signatures valid ✓                      │
│  Verify: balance_root matches checkpoint state ✓        │
│  Apply checkpoint state to local RocksDB                │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  SYNC BLOCKS (checkpoint height → tip)                  │
│                                                         │
│  For each block:                                        │
│    1. Verify ≥ 3 validator signatures                   │
│    2. Re-execute all txs                                │
│    3. Compute local balance_root                        │
│    4. Check: local_root == block.balance_root           │
│    5. If mismatch → reject block, try different peer    │
│    6. If match → write to chain                         │
│                                                         │
│  Cannot be fooled by a single corrupt peer.             │
│  Cannot be fooled by 1 corrupt validator.               │
│  The math is the trust.                                 │
└─────────────────────────────────────────────────────────┘
         │
         ▼
  Node is at tip
  Every balance independently verified
  No single node trusted
```

---

## Diagram 6 — Current State vs Target State

```
TODAY                              TARGET

  User                               User
    │                                  │
    ▼                                  ▼
 Epsilon  ← single truth           Any of 4 nodes
    │      no verification             │
    │      trust required           3-of-4 signed
    │                                  │
  Beta  ─┐                         consensus
  Gamma ─┼─ accept whatever        verified state
  Delta ─┘   Epsilon says              │
                                   balance_root
                                   in every block

  May 9 incident:                  May 9 scenario:
  Replay overwrites balances        Epsilon's blocks have
  Beta/Gamma accept it              wrong balance_root
  Users see wrong balance           Beta/Gamma/Delta
  No detection until                REJECT those blocks
  user reports it                   Operator alerted
                                    Chain continues
                                    User balance protected

  Trust model: trust Epsilon        Trust model: trust the math
```
