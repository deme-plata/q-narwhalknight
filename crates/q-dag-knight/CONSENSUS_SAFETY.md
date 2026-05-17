# Q-NarwhalKnight BFT Consensus Safety Properties

## v1.0.69-beta - Tail Fork Protection Implementation

This document describes the BFT consensus safety properties and the protections
against tail forking vulnerabilities in pipelined BFT systems.

---

## 1. Tail Forking Vulnerability

### What is Tail Forking?
Tail forking occurs in pipelined BFT consensus when a malicious leader proposes
conflicting blocks during the finality window (before commits are final). This
can cause:
- Chain reorganizations
- Double-spending attacks
- Consensus failures
- Network partitions

### Attack Vector
1. Leader L proposes block B at height H
2. Before B is finalized, L proposes conflicting block B' at height H
3. Different validators vote on B vs B'
4. Chain forks, violating safety

---

## 2. Protection Mechanisms Implemented

### 2.1 Delta-Delayed Commit Rule (δ = 4 rounds)
**File:** `commit_logic.rs`

```rust
const DELTA_COMMIT_DELAY: u64 = 4;
```

Blocks are only committed after δ=4 rounds of subsequent vertices are built
on top. This ensures:
- Sufficient propagation time
- Enough votes for finality
- Time for conflicting proposals to be detected

### 2.2 Tail Fork Detection
**File:** `commit_logic.rs`

The `TailForkDetection` struct monitors for:
- Multiple proposals from same leader in same round
- Conflicting vertices with same parent
- Stake-based threshold violations

```rust
pub struct TailForkDetection {
    proposals_by_leader: HashMap<(ValidatorId, Round), Vec<VertexId>>,
    recent_proposals: VecDeque<(SystemTime, ValidatorId, VertexId)>,
    conflicting_proposals: HashSet<VertexId>,
}
```

Detection functions:
- `detect_tail_fork()` - Check for multiple proposals
- `detect_conflicting_vertices()` - Check parent conflicts
- `is_byzantine_leader()` - Track repeated violations
- `get_fork_evidence()` - Collect proof of misbehavior

### 2.3 Ancestor Finality Check
**File:** `block_producer.rs`

Before proposing a new block, we verify that the proposed height doesn't
exceed the committed round by more than δ+1:

```rust
if proposed_height > committed_round + delta + 1 {
    warn!("TAIL FORK PROTECTION: Cannot propose block too far ahead");
    return None;
}
```

This ensures:
- Parents are sufficiently finalized
- No gaps in commit chain
- Prevents speculative proposal attacks

### 2.4 View Change Protocol
**File:** `voting_coordinator.rs`

Leader timeout and rotation prevents stuck/malicious leaders:

```rust
pub struct VotingState {
    current_view: u64,
    current_leader: Option<ValidatorId>,
    last_leader_proposal_time: Option<SystemTime>,
    consecutive_leader_timeouts: u32,
    view_change_votes: HashMap<u64, HashSet<ValidatorId>>,
}
```

View change triggers:
- Leader timeout (30 seconds default)
- Missing proposals
- Detected Byzantine behavior
- 2f+1 view change votes

### 2.5 Byzantine Voting Threshold
**File:** `commit_logic.rs`, `voting_coordinator.rs`

All critical operations require 2f+1 stake (>66%):

```rust
let quorum_stake = (total_stake * 2) / 3 + 1;
if collected_stake >= quorum_stake {
    // Safe to commit
}
```

---

## 3. Safety Guarantees

### 3.1 Agreement
If honest validator commits block B at height H, no honest validator
commits different block B' at height H.

**Enforced by:**
- Delta-delayed commits
- 2f+1 voting threshold
- Tail fork detection

### 3.2 Validity
Only valid blocks proposed by authorized leaders can be committed.

**Enforced by:**
- VRF-based anchor election
- Cryptographic signature verification
- Leader rotation on timeout

### 3.3 Liveness
If f < n/3 validators are Byzantine, the system eventually makes progress.

**Enforced by:**
- View change protocol
- Leader timeout detection
- Automatic leader rotation

### 3.4 Finality
Once committed, blocks are never reverted.

**Enforced by:**
- Delta-delayed commit rule
- Ancestor finality check
- No speculative execution

---

## 4. Attack Resistance Summary

| Attack Type | Protection Mechanism |
|-------------|---------------------|
| Double voting | 2f+1 quorum requirement |
| Leader equivocation | Tail fork detection |
| Proposal withholding | Leader timeout + view change |
| Chain reorganization | Delta-delayed commits |
| Fork attacks | Ancestor finality check |
| Network partitioning | Gossipsub + redundant paths |

---

## 5. Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DELTA_COMMIT_DELAY` | 4 rounds | Rounds before commit |
| `leader_timeout` | 30 seconds | Time before view change |
| `max_consecutive_timeouts` | 3 | Forced rotation threshold |
| `VOTING_THRESHOLD` | 2f+1 | Byzantine fault tolerance |

---

## 6. Comparison to Pipelined BFT Systems

Q-NarwhalKnight avoids the tail forking vulnerabilities present in:
- HotStuff (mitigated by chaining but not eliminated)
- Tendermint (2-round latency helps but complex)
- PBFT (vulnerable during view changes)

Our improvements:
1. **Explicit tail fork detection** - Not just implicit through voting
2. **Ancestor finality enforcement** - Block production gated on commits
3. **Automatic view change** - No manual intervention needed
4. **VRF-enhanced leader election** - Unpredictable leader schedule

---

## 7. Future Enhancements

- [ ] Slashing for detected Byzantine behavior
- [ ] Cryptographic evidence accumulation
- [ ] Cross-epoch finality proofs
- [ ] Light client verification

---

*Last updated: v1.0.69-beta*
*Author: Q-NarwhalKnight Consensus Team*
