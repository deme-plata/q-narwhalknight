# Issue #029: TLA+ Formal Specification of DAG-Knight Consensus

**State**: `open`
**Priority**: HIGH
**Labels**: `starship-endgame`, `consensus`, `formal-verification`, `academic`
**Assigned**: Gamma
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

Leslie Lamport would ask: "Where's your TLA+ spec?" Our DAG-Knight consensus engine has zero-message BFT ordering, homological fork detection (Betti numbers), and quantum anchor election — all critical safety properties that need formal verification.

A TLA+ specification would:
1. Prove liveness guarantees under f < n/3 Byzantine faults
2. Verify safety (no two honest nodes commit conflicting blocks)
3. Model the turbo-sync protocol and prove sync-down is impossible
4. Prove the emission controller converges to exactly 21M QUG

## Current State

- No formal specification exists
- Consensus correctness relies on test suites (125+ tests) and manual review
- DAG-Knight paper (DISC 2021) has theoretical proofs but our implementation diverges (quantum VDF, adaptive emission)
- Sync-down protection exists at 2 layers but not formally verified

## Acceptance Criteria

- [ ] `specs/dagknight_consensus.tla` — Core DAG-Knight ordering specification
- [ ] `specs/turbo_sync.tla` — Sync protocol with sync-down impossibility proof
- [ ] `specs/emission_controller.tla` — Emission convergence to 21M proof
- [ ] `specs/fork_detection.tla` — Homological fork detection correctness
- [ ] TLC model checker passes all invariants (no counterexamples)
- [ ] Safety property: `NoConflictingCommits == \A b1, b2 \in committed: Compatible(b1, b2)`
- [ ] Liveness property: `Eventually(\A tx \in submitted: tx \in committed)`
- [ ] Document any assumptions/simplifications vs real implementation

## Implementation Hints

```tla
---- MODULE DAGKnightConsensus ----
EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    Validators,      \* Set of validator IDs
    MaxHeight,       \* Maximum block height to model-check
    F                \* Maximum Byzantine faults (F < |Validators|/3)

VARIABLES
    dag,             \* DAG of blocks: [height -> Set(Block)]
    committed,       \* Set of committed blocks
    heights,         \* heights[v] = current height of validator v
    messages         \* In-flight P2P messages

TypeInvariant ==
    /\ dag \in [0..MaxHeight -> SUBSET Block]
    /\ committed \subseteq UNION {dag[h] : h \in 0..MaxHeight}
    /\ \A v \in Validators: heights[v] \in 0..MaxHeight

\* SAFETY: No conflicting commits
Safety == \A b1, b2 \in committed:
    b1.height = b2.height => b1 = b2

\* LIVENESS: All submitted transactions eventually committed
Liveness == \A tx \in submitted: <>(tx \in committed)

\* SYNC-DOWN IMPOSSIBILITY: Height never decreases
NoSyncDown == \A v \in Validators:
    heights'[v] >= heights[v]
====
```

## References

- Lamport, "Specifying Systems" (TLA+ textbook)
- DAG-Knight: DISC 2021 (Gelashvili et al.)
- Amazon's use of TLA+ for S3/DynamoDB: https://lamport.azurewebsites.net/tla/amazon.html

## Depends On

None — can be done independently of code changes.

## Files

- `specs/dagknight_consensus.tla` — Main consensus spec
- `specs/turbo_sync.tla` — Sync protocol spec
- `specs/emission_controller.tla` — Emission model spec
- `specs/fork_detection.tla` — Fork detection spec
- `specs/README.md` — How to run TLC model checker
