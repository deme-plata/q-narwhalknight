# Technical Review: Consensus Layer State — May 10, 2026

**Prepared by:** Multi-agent codebase audit (3 parallel agents)
**Scope:** Block consensus, balance state agreement, BalanceRootV1 plan, fault tolerance, recovery mechanisms
**Motivation:** The May 9 balance replay incident destroyed correct wallet balances on Epsilon. No other node held correct state to recover from. This document explains the architectural reason why, documents what is built and what is planned, and maps a concrete path forward.

---

## 1. Current Architecture (Where We Are)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    BLOCK PRODUCTION — CENTRALIZED                        │
│                                                                          │
│   Epsilon (89.149.241.126)          Beta (185.182.185.227)               │
│   ┌───────────────────────┐         ┌───────────────────┐                │
│   │  Block Producer x4    │         │  Block Consumer   │                │
│   │  (4 parallel workers) │         │  (receives only)  │                │
│   │  total_validators=1   │         │  total_validators │                │
│   │                       │         │  not set          │                │
│   │  ┌─────────────────┐  │         └───────────────────┘                │
│   │  │ Produce block   │  │                                              │
│   │  │ at height N+1   │──┼──────► gossipsub /qnk/mainnet-genesis/blocks │
│   │  │ (no DAG-Knight  │  │                ▼              ▼              │
│   │  │  ordering)      │  │         Gamma (109.205.176.60)               │
│   │  └─────────────────┘  │         ┌───────────────────┐                │
│   └───────────────────────┘         │  Block Consumer   │                │
│                                     │  (receives only)  │                │
│           ▲                         └───────────────────┘                │
│           │ only Epsilon can                                             │
│           │ produce blocks                                               │
│           │                                                              │
│   If Epsilon goes offline → NO new blocks → network halts               │
└──────────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  BALANCE STATE — NOT UNDER CONSENSUS                     │
│                                                                          │
│   Miner submits solution to Epsilon                                      │
│          │                                                               │
│          ▼                                                               │
│   Epsilon produces block (height N)                                      │
│   • Embeds coinbase tx (reward for miner)                                │
│   • Applies to local RocksDB: wallet[miner] += reward                   │
│   • Broadcasts block via gossipsub                                       │
│          │                                                               │
│          ▼                              ▼                                │
│   Beta receives block               Gamma receives block                 │
│   • Applies same coinbase tx         • Applies same coinbase tx          │
│   • wallet[miner] += reward          • wallet[miner] += reward           │
│   • ✅ All 3 nodes agree on reward                                       │
│                                                                          │
│   But for TRANSFER transactions:                                         │
│                                                                          │
│   User submits transfer to Epsilon                                       │
│          │                                                               │
│          ▼                                                               │
│   Epsilon adds to mempool (local only)                                   │
│   Beta/Gamma DO NOT know about this tx yet                               │
│          │                                                               │
│          ▼                                                               │
│   Next block includes the transfer                                       │
│   All nodes apply it ── BUT ──                                           │
│                                                                          │
│   If the sender's balance is WRONG on Epsilon (e.g., post-replay bug):  │
│   • Epsilon skips the tx (insufficient funds error, continues)           │
│   • Beta/Gamma also had wrong balance (synced from Epsilon)              │
│   • No node detects the skip — silent consensus failure                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. What Works Today

**Block propagation** — gossipsub delivers blocks to all peers within seconds.

**Turbo sync** — nodes that fall behind receive historical blocks in batch via the block-pack protocol.

**Mining rewards** — coinbase transactions in blocks are processed deterministically on every node that receives the block. Reward consensus works because rewards are embedded in blocks.

**P2P connectivity** — Epsilon had 24 peers at time of audit; DHT routing and peer discovery are healthy.

**Balance divergence detection** — every 5 minutes, nodes compare a BLAKE3 hash of their wallet state with a peer. Mismatches are logged.

**SSE streaming** — real-time mining reward events work end-to-end from block receipt through frontend.

---

## 3. The Consensus Gap — What Is Not Implemented

### 3.1 Block Production Is Centralized

```
config.rs default: total_validators = 1

  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │   Epsilon   │     │    Beta     │     │    Gamma    │
  │  VALIDATOR  │     │  observer   │     │  observer   │
  │  produces   │     │  receives   │     │  receives   │
  │  ALL blocks │     │  blocks     │     │  blocks     │
  └──────┬──────┘     └─────────────┘     └─────────────┘
         │
         │  if offline: ┌─────────────────────────────────┐
         └─────────────►│ NETWORK HALT — no new blocks    │
                         │ miners earn nothing              │
                         │ transfers cannot confirm         │
                         └─────────────────────────────────┘
```

The four "producers" visible in the block producer pool are four parallel goroutines inside Epsilon's single process — not four independent validators. Beta and Gamma are not configured as block producers.

### 3.2 Block Validation Does Not Verify Balance Correctness

```
Block arrives at Beta via gossipsub
         │
         ▼
  ┌──────────────────────────────────────────────┐
  │  Current validation checks:                  │
  │  ✅ Network ID matches                        │
  │  ✅ Block signature (Ed25519) valid           │
  │  ✅ Height is monotonically increasing        │
  │  ✅ Parent hash is known                      │
  │                                              │
  │  NOT checked:                                │
  │  ❌ Are the transfers in this block affordable?│
  │  ❌ Does the state_root match our balance state?│
  │  ❌ Is the coinbase amount within emission bounds?│
  └──────────────────────────────────────────────┘
         │
         ▼
  Block accepted. Transactions applied.
  If a transfer was from a zero-balance wallet:
  balance_consensus.rs line 857: `continue;`  ← silently skipped
         │
         ▼
  Epsilon had balance X for wallet A  ──► Skips transfer from A
  Beta had balance Y for wallet A    ──► Applies transfer from A
  
  Result: silent consensus divergence from a single block
```

### 3.3 DAG-Knight Is Not Driving Block Ordering

```
What the code says:              What actually happens:
                                 
 block_producer.rs               block_producer.rs
 ┌──────────────────────┐        ┌──────────────────────┐
 │ dag_knight.committed │        │ produce at height N+1│
 │ _round → select next │  ──►  │ (ignores DAG output)  │
 │ block to produce     │        │ 60s timeout bypasses │
 └──────────────────────┘        │ any DAG-Knight hang  │
                                 └──────────────────────┘

DAG-Knight crates exist and run in parallel.
They log metrics. They do not control block ordering.
Chain is linear: height N → N+1 → N+2 (single producer).
```

### 3.4 Bracha Reliable Broadcast Is Neutered (f=0)

```
Standard Bracha Reliable Broadcast (f=1, 4 nodes):
                                                    
  Proposer                Node A    Node B    Node C  
     │                      │         │         │    
     │──── SEND(v) ─────────►──────────►──────────►   
     │                      │         │         │    
     │    ◄── ECHO(v) ───────◄─────────◄─────────◄   
     │                      │         │         │    
  Wait for 2f+1=3 ECHOs     │         │         │    
     │──── READY(v) ────────►──────────►──────────►   
     │                      │         │         │    
  Wait for 2f+1=3 READYs    │         │         │    
     │                      │         │         │    
  ✅  DELIVER(v)             ✅        ✅        ✅   
                                                    
─────────────────────────────────────────────────────
                                                    
Current implementation (f=0, shadow mode):          
                                                    
  Proposer                                          
     │                                              
     │──── SEND(v) ─────────►  ANY one node        
     │                                              
  echo_quorum = 2*0+1 = 1    ◄── ECHO from 1 node  
     │                                              
  ✅  DELIVER immediately                           
                                                    
  One message = delivered.                          
  Byzantine fault tolerance = ZERO.                 
```

Code: `BalanceFinalityEngine::new(0, ...)` in `main.rs:3636`
Changing `0` to `1` requires quorum = 3 across all active validators — a coordinated hard fork.

### 3.5 Divergence Detection Does Not Trigger Recovery

```
Every 5 minutes on each node:

  state_sync_api.rs: do_combined_state_sync()
         │
         ▼
  Compute our balance hash:
  BLAKE3( sorted(wallet_addr || balance_amount) )
         │
         ▼
  Fetch peer's balance_state_hash
         │
         ├── hashes match ──► ✅ log INFO, continue
         │
         └── hashes differ ──► 🚨 log CRITICAL error
                                    │
                                    ▼
                                "CRITICAL: Balance hash MISMATCH"
                                    │
                                    ▼
                                ← nothing else happens →
                                
  Nodes continue with divergent state forever.
  No peer query. No majority vote. No self-correction.
```

### 3.6 State Sync Is Add-Only (Cannot Correct Corruption)

```
FullStateSnapshot sync (every 5 min, pull-based):

  Receiving node merge policy:
  
  for wallet in peer_snapshot:
    if wallet in our_local_db:
      KEEP OURS (never overwrite existing value)   ← intentional after replay bug
    else:
      ADD wallet from peer
      
  Result:
  
  Epsilon has wallet X = 1484 QUG (wrong, post-corruption)
  Gamma  has wallet X = 3200 QUG (correct)
  
  Gamma sends snapshot to Epsilon ──► Epsilon KEEPS 1484 (has existing value)
  Epsilon sends snapshot to Gamma ──► Gamma KEEPS 3200 (has existing value)
  
  Both nodes "win" — neither corrects the other.
  Manual intervention (Q_BALANCE_AUTHORITY_PEER) is the only escape.
```

---

## 4. BalanceRootV1 — The Bridge Between Block Consensus and Balance Agreement

This is the most important planned piece. It was designed and documented on May 6, 2026 in `docs/technical-review-balance-root-v1-implementation-2026-05-06.md`. Here is where it stands and why it matters for the consensus gap.

### 4.1 What BalanceRootV1 Does

```
Current block header (mainnet today):
┌─────────────────────────────────────┐
│ BlockHeader                         │
│   height:         17,671,000        │
│   prev_hash:      0xABCD...         │
│   timestamp:      1778392616        │
│   tx_root:        0x1234...         │ ← Merkle root of transactions
│   state_root:     [0, 0, 0, ...]    │ ← always zero (gate at u64::MAX)
│   vdf_proof:      ...               │
└─────────────────────────────────────┘

After BalanceRootV1 activates (height 18,600,000):
┌─────────────────────────────────────┐
│ BlockHeader                         │
│   height:         18,600,000+       │
│   prev_hash:      0xABCD...         │
│   timestamp:      ...               │
│   tx_root:        0x1234...         │ ← transactions
│   balance_root:   0x9F2A...         │ ← NEW: BLAKE3 of all wallet balances
│   state_root:     [0, 0, ...]       │ ← still zero (StateRootV1 at u64::MAX)
└─────────────────────────────────────┘
```

### 4.2 How the Root Is Computed

```
compute_balance_root_for_block() — lib.rs line 4429
                                                        
  Load all wallet_balance_* keys from RocksDB           
          │                                             
          ▼                                             
  Filter: remove wallets with balance = 0               
          │                                             
          ▼                                             
  Sort: lexicographic ascending on address bytes        
          │                                             
          ▼                                             
  For each (address, balance):                          
    leaf = BLAKE3(address_bytes || balance_u128_BE)     
          │                                             
          ▼                                             
  root = BLAKE3("balance_root_v1" || leaf_0 || leaf_1 || ... || leaf_N)
  ▲                                                     
  └── domain separator "balance_root_v1" prevents       
      collision with other BLAKE3 uses in the codebase  
                                                        
  Balance encoding: big-endian u128 (canonical spec)    
  Legacy compute_balance_state_hash(): little-endian,   
    no domain sep — still used for diagnostics only     
```

### 4.3 What BalanceRootV1 Enables (When Wired)

```
WITHOUT BalanceRootV1 (today):

  Epsilon produces block ──► All nodes accept it ──► Each node computes
                                                       its own balances
                                                       independently ──►
                                                       
  Epsilon: wallet A = 1484  ┐
  Beta:    wallet A = 3200  ├── diverge silently
  Gamma:   wallet A = 3200  ┘   forever

────────────────────────────────────────────────────────────────────────

WITH BalanceRootV1 (after height 18,600,000):

  Epsilon produces block
  • Applies transactions to local RocksDB
  • compute_balance_root_for_block() → root = 0x9F2A...
  • Sets block.header.balance_root = 0x9F2A...
  • Broadcasts block
         │
         ▼
  Beta receives block
  • Applies same transactions to local RocksDB
  • compute_balance_root_for_block() → root = 0x3B1C...
  • Compares: 0x9F2A ≠ 0x3B1C
  • ❌ REJECT BLOCK — balance root mismatch
  
  Beta cannot accept new blocks until its balance state
  agrees with the producing node.
  
  This makes divergence VISIBLE and SELF-CORRECTING:
  diverged node cannot advance ──► must resync ──► reconciles.
```

### 4.4 Current Implementation Status

```
Component                                     Status
─────────────────────────────────────────────────────────────────
Upgrade gate (height 18,600,000, mandatory)   ✅ Done
  crates/q-consensus-guard/src/upgrade_gate.rs:139

compute_balance_root_for_block() function      ✅ Done
  crates/q-storage/src/lib.rs:4429

Balance root called from block producer        ❌ NOT DONE
  block_producer.rs ~line 954
  Currently: state_root = [0;32] always

Block validation checks balance_root           ❌ NOT DONE
  main.rs ~line 11380
  Currently: no balance root check in gossip block handler

Shadow-mode logging (warn, don't reject)       ❌ NOT DONE
  The phase before enforcement — essential for soak testing

balance_determinism_tests.rs test suite        ❌ NOT DONE
  10 tests proving cross-node hash agreement

Health endpoint exposes balance_root_v1        ❌ NOT DONE
  Needed to compare Epsilon vs Delta during soak

14-day soak on Delta test container            ❌ NOT STARTED
  Required before any enforcement deploy
─────────────────────────────────────────────────────────────────

Activation height 18,600,000 is ~928,000 blocks from current tip
(17,672,000 as of May 10). At 1 bps that is ~10.7 days away.

⚠️  WARNING: The gate will fire in ~11 days regardless of
    whether the code is wired. At that height, if balance_root
    is [0;32] in block headers (current state), nodes running
    any enforcement logic will reject those blocks.

    If enforcement is NOT wired (current state), nothing bad
    happens — the gate is irrelevant if nobody checks it.
    But the opportunity to use 18,600,000 as the enforcement
    height will pass. A new height must then be chosen.
```

### 4.5 BalanceRootV1 vs the Incident — Would It Have Helped?

```
Replay bug corrupts Epsilon's balances (May 9):

  Without BalanceRootV1 (what happened):
  
  Epsilon (wallet A = 1484) produces block N+1 ──►
  balance_root not checked ──►
  Beta/Gamma accept block N+1 ──►
  Divergence grows silently for hours ──►
  User sees wrong balance, no automated detection

──────────────────────────────────────────────────────────────
  
  With BalanceRootV1 (hypothetical):
  
  Epsilon (wallet A = 1484) produces block N+1
  • compute_balance_root_for_block() → root_epsilon = 0xAAA...
  • Broadcasts block with balance_root = 0xAAA...
  
  Beta (wallet A = 3200) receives block N+1
  • Applies transactions
  • compute_balance_root_for_block() → root_beta = 0xBBB...
  • 0xAAA ≠ 0xBBB → REJECT BLOCK
  
  Epsilon can no longer extend the chain with its
  corrupted state. Block production halts.
  
  Alert fires within seconds (block rejected).
  Operator investigates → finds corruption → manual repair.
  Network paused, not silently wrong.
  
  BalanceRootV1 turns "silent corruption" into "loud halt".
```

---

## 5. CHECKPOINT_DATA — Static Unverified Snapshot

```
balance_checkpoint.rs:

  pub static CHECKPOINT_DATA: &[(&str, &str)] = &[
    ("0a3f...b1c2", "482000000000000000000000000"),
    ("1b4e...c2d3", "1000000000000000000000000000"),
    // ... 1,324 more wallets ...
  ];

  CHECKPOINT_HEIGHT    = 16,538,868
  CHECKPOINT_SHA256    = "eabbeadf85d0..."
  CHECKPOINT_WALLET_COUNT = 1,326

How it was generated:
  scripts/gen_balance_checkpoint.py  ──►  reads Epsilon's RocksDB  ──►
  outputs Rust constant array

Verification on load:
  SHA256( sort(wallet_hex || amount_str) ) == CHECKPOINT_SHA256 ?
  
  This proves: data wasn't corrupted in transit.
  This does NOT prove: data was correct when generated.
  
  ┌───────────────────────────────────────────────────┐
  │  Signatures from Beta or Gamma: NONE              │
  │  Quorum certificate: NONE                         │
  │  Multiple validator agreement: NONE               │
  │  Epsilon is trusted unilaterally.                 │
  └───────────────────────────────────────────────────┘

If Epsilon's balances were wrong at height 16,538,868
→ the checkpoint is wrong
→ all checkpoint-bootstrapped nodes start from wrong state
→ BalanceRootV1 would immediately detect mismatch (correct behavior)
→ but recovery still requires manual intervention
```

---

## 6. Fault Tolerance Analysis

```
Network: Alpha, Beta, Gamma, Epsilon (4 nodes)
Target: f=1 (tolerate 1 Byzantine faulty node)
Requirement: n ≥ 2f+1 → n ≥ 3 nodes needed for f=1

              Block Production   Balance Consensus   Bracha BFT
              ─────────────────  ──────────────────  ──────────
Target (f=1)  3+ producers       2f+1=3 agree        echo_q=3
              leader election    before finalizing    amplify=2

Current state 1 producer         0 nodes needed      echo_q=1
              (Epsilon only)     (no consensus,       amplify=1
                                  local compute)      (f=0)

Fault          CRITICAL           CRITICAL            0 faults
tolerance:     f=0                f=0                 tolerated

Practical f:   0                  0                   0

Even with 4 nodes, actual Byzantine fault tolerance is ZERO.
Any single node failure disrupts the system:
  • Epsilon offline → no new blocks
  • Epsilon corrupted → balances diverge silently
  • Beta offline → no effect on consensus (it's not producing)
  • Gamma offline → no effect on consensus (it's not producing)
```

---

## 7. The Incident Through an Architectural Lens

```
Timeline of the May 9 incident:

May 9, ~18:00 UTC
  │
  │  v10.7.6 replay ran on Epsilon (genesis node)
  │  ┌──────────────────────────────────────────────────────┐
  │  │ replay_post_checkpoint_balances():                   │
  │  │  1. Load CHECKPOINT_DATA (balances at h=16,538,868) │
  │  │  2. Replay local blocks 16,538,869 → current        │
  │  │  3. Call save_wallet_balances(&replay_map)           │
  │  │     ← NO max-wins guard (since fixed in v10.7.8)    │
  │  │  4. wallet[user] = 1484 (checkpoint value)           │
  │  │     overwrites RocksDB where wallet[user] = 3200     │
  │  └──────────────────────────────────────────────────────┘
  │
  ▼
Epsilon: wallet[user] = 1484 QUG (wrong)
Beta:    wallet[user] = ??? (synced from Epsilon over time)
Gamma:   wallet[user] = 3200 QUG (correct, unaffected by replay)
  │
  ▼
Divergence check fires 5 minutes later:
  "🚨 CRITICAL: Balance hash MISMATCH with peer"
  → logged → nothing else happens
  │
  ▼
Epsilon continues producing blocks with 1484 in state
No block rejection (BalanceRootV1 not wired)
Frontend shows 1484 QUG to user
User has lost 1,716 QUG they actually earned
  │
  ▼
Recovery requires:
  • Find a node with correct value (Gamma has 3200)
  • Manually write correct value to Epsilon
  • No automated mechanism exists
  
ROOT CAUSE SUMMARY:
  1. Replay ran without is_checkpoint_applied() guard     ← fixed v10.7.8
  2. save_wallet_balances had no max-wins semantics       ← fixed v10.7.8
  3. Divergence detection doesn't trigger correction     ← NOT FIXED
  4. BalanceRootV1 not wired → corruption not detectable ← NOT FIXED
  5. No multi-producer → only Epsilon enforces state     ← NOT FIXED
```

---

## 8. Fix Priority Map

```
                        NOW          +2 weeks      +4 weeks      +8 weeks
                         │               │              │              │
PRIORITY 1               │               │              │              │
Multi-producer           │◄──── 1-2 weeks work ────────►│              │
leader election          │               │              │              │
  • Beta + Gamma can     │               │              │              │
    produce blocks       │               │              │              │
  • Round-robin or VRF   │               │              │              │
  • f=1 block layer      │               │              │              │
                         │               │              │              │
PRIORITY 2               │               │              │              │
BalanceRootV1 wiring     │◄── 1 week ───►│              │              │
  • Wire into producer   │               │              │              │
  • Shadow-mode logging  │ ⚠️ height      │              │              │
  • 14-day Delta soak    │ 18,600,000    │              │              │
  • Enforce at gate      │ fires ~May 21 │              │              │
                         │               │              │              │
PRIORITY 3               │               │              │              │
Peer majority            │               │◄─── 1-2 weeks ───►│         │
reconciliation           │               │              │    │         │
  • /consensus/wallet    │               │              │    │         │
    -balance endpoint    │               │              │    │         │
  • Auto-correct on      │               │              │    │         │
    divergence detect    │               │              │    │         │
                         │               │              │    │         │
PRIORITY 4               │               │              │    │         │
Bracha f=1 hard fork     │               │              │    ◄─ 2-3w ─►│
  • Coordinated upgrade  │               │              │              │
  • quorum: 3 nodes      │               │              │              │
  • All nodes same day   │               │              │              │
```

---

## 9. Proposed: Peer Majority Balance Reconciliation

This is the missing corrective step that must accompany BalanceRootV1. When a block is rejected due to balance root mismatch, the diverged node needs a way to repair itself.

```
New endpoint: GET /api/v1/consensus/wallet-balance?address={hex}
Response: { "address": "...", "balance": 3200000000..., "height": 17671391,
            "validator_sig": "ed25519:...", "node_id": "..." }

Reconciliation flow (triggers on block rejection OR divergence detection):

  Diverged node (Epsilon, has 1484)
         │
         ▼
  Query ALL known validators:
  GET Beta:  /consensus/wallet-balance?address=<wallet>  → 3200  sig_B
  GET Gamma: /consensus/wallet-balance?address=<wallet>  → 3200  sig_G
         │
         ▼
  Count responses: 2-of-2 agree on 3200
  (need 2-of-3 for Byzantine safety with f=1)
         │
         ▼
  Max-wins guard: 3200 > 1484 (current local value)
  ✅ Write 3200 to RocksDB
         │
         ▼
  wallet[user] = 3200  ✅ Corrected automatically

Design notes:
  • Each response must carry a validator Ed25519 signature
  • Max-wins guard prevents a Byzantine peer from writing a LOWER value
  • Reconciliation only writes higher values (add-only semantics for amounts)
  • Works for recovery from replay bugs AND from block replay gaps
```

---

## 10. Recommended Immediate Actions

### 10.1 Wire BalanceRootV1 into Shadow Mode (This Week)

```
Files to change:

1. block_producer.rs (~line 954)
   Add: balance_root = compute_balance_root_for_block()
         when is_upgrade_active(BalanceRootV1, next_height)

2. main.rs (~line 11380, gossip block handler)
   Add: compare block.header.balance_root with computed root
         WARN on mismatch (shadow mode — do not reject yet)

3. handlers.rs (health endpoint)
   Add: balance_root_v1: hex::encode(computed_root)
        balance_root_active: is_upgrade_active(BalanceRootV1, height)

4. balance_determinism_tests.rs (new file)
   10 tests per the May 6 design doc

Risk: LOW — shadow mode logs but never rejects. Zero production impact.
Deadline: Before height 18,600,000 (~May 21, 2026 at 1 bps)
```

### 10.2 Alert on Balance Hash Mismatch (This Week)

```
state_sync_api.rs divergence check — change from:
  error!("🚨 CRITICAL: Balance hash MISMATCH")
  // nothing else

To:
  error!("🚨 CRITICAL: Balance hash MISMATCH");
  // send webhook / email notification
  trigger_balance_mismatch_alert(&our_hash, peer_hash, our_height).await;

Implementation: post to a webhook URL from Q_ALERT_WEBHOOK env var.
This gives ops team <5 min awareness instead of discovering during
user complaints.
```

### 10.3 Restore User Wallet (Immediate)

```
User wallet address: [not yet provided]

Recovery plan:
  1. Get wallet address from user
  2. Query Gamma (unaffected): GET /api/v1/consensus/wallet-balance?address=...
     (or direct RocksDB read on Gamma: rocksdb_cli get wallet_balance_{hex})
  3. Verify value is 3200 QUG (matches user's recollection)
  4. Write targeted correction to Epsilon's RocksDB with max-wins guard:
     if gamma_value > epsilon_value: write gamma_value to Epsilon
  5. Verify: fetch balance from Epsilon frontend after write
```

---

## 11. Honest State Summary

```
┌────────────────────────────────────────────────────────────────┐
│              CONSENSUS LAYER STATUS — MAY 10, 2026             │
├──────────────────────────────┬─────────────────────────────────┤
│ Block ordering (which blocks │ ✅ WORKS                         │
│ are canonical)               │    gossipsub + linear chain      │
│                              │    (not DAG-Knight ordered)      │
├──────────────────────────────┼─────────────────────────────────┤
│ DAG-Knight BFT finality      │ ⚠️  INFRASTRUCTURE ONLY          │
│                              │    runs in parallel, not used   │
│                              │    for block ordering            │
├──────────────────────────────┼─────────────────────────────────┤
│ Mining reward consensus      │ ✅ WORKS (via blocks)            │
│                              │    deterministic from coinbase  │
├──────────────────────────────┼─────────────────────────────────┤
│ Transfer tx consensus        │ ⚠️  PARTIAL                      │
│                              │    applied when in block        │
│                              │    lost if node crashes before  │
│                              │    including in block           │
├──────────────────────────────┼─────────────────────────────────┤
│ Balance state commitment     │ ⚠️  PLANNED, NOT WIRED           │
│ (BalanceRootV1)              │    function exists, gate set    │
│                              │    producer + validator not done│
│                              │    fires height ~18,600,000     │
├──────────────────────────────┼─────────────────────────────────┤
│ Balance divergence detection │ ✅ DETECTS                       │
│                              │ ❌ DOES NOT CORRECT              │
├──────────────────────────────┼─────────────────────────────────┤
│ Bracha BFT (f)               │ ⚠️  DEPLOYED, f=0 (shadow)       │
│                              │    quorum=1, no Byzantine safety │
│                              │    f=1 upgrade = hard fork      │
├──────────────────────────────┼─────────────────────────────────┤
│ Block producer count         │ ❌ 1 (Epsilon only)              │
│                              │    Beta/Gamma are consumers     │
├──────────────────────────────┼─────────────────────────────────┤
│ Network fault tolerance (f)  │ ❌ f=0                           │
│                              │    any node failure = problem   │
└──────────────────────────────┴─────────────────────────────────┘

The infrastructure for f=1 consensus exists: Bracha engine,
upgrade gate system, balance root computation, divergence detection.
None of it is fully wired or enforced. The gap is integration,
not invention.
```

---

## 12. References

| Topic | Location |
|---|---|
| BalanceRootV1 full plan | `docs/technical-review-balance-root-v1-implementation-2026-05-06.md` |
| Bracha RB design | `docs/technical-review-balance-finality-bracha-rb-2026-05-01.md` |
| May 9 incident report | `docs/incident-report-balance-replay-2026-05-09.md` |
| `compute_balance_root_for_block()` | `crates/q-storage/src/lib.rs:4429` |
| BalanceRootV1 upgrade gate | `crates/q-consensus-guard/src/upgrade_gate.rs:139` |
| Bracha engine initialization (f=0) | `crates/q-api-server/src/main.rs:3636` |
| Balance divergence check | `crates/q-api-server/src/state_sync_api.rs:983` |
| Block producer (single validator) | `crates/q-api-server/src/config.rs:64` |
| CHECKPOINT_DATA | `crates/q-storage/src/balance_checkpoint.rs` |
| Transfer tx silent skip | `crates/q-storage/src/balance_consensus.rs:857` |
| Turbo sync coinbase-only mode | `crates/q-api-server/src/main.rs:6370` |

---

*v2.0 — 2026-05-10 — updated with BalanceRootV1 status and diagrams*
