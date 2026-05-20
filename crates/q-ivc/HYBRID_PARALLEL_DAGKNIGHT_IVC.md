# Hybrid DAG-Knight + q-IVC Scaling Plan

## TL;DR
Use **one canonical DAG-Knight consensus lane** for ordering/finality and scale throughput with a **hybrid execution/proving fabric**:

1. **Parallel block builders** (deterministic winner selection, single ordered block output).
2. **Intra-block transaction partitioning** into conflict-free execution lanes.
3. **q-IVC recursive aggregation** of per-lane proofs into a single tip proof.
4. Optional **domain shards as execution shards only** (not consensus shards), bridged by async intents and delayed atomic settlement.

This avoids full consensus sharding's cross-shard 2PC tax while still enabling high parallelism.

---

## Why not "more DAG-Knights" for throughput?

Running multiple DAG-Knight instances only helps throughput if each instance owns a disjoint state slice. That is effectively sharding and introduces:

- cross-shard state dependency handling,
- atomicity/rollback complexity,
- shard balancing pressure,
- and fork-choice coupling between shards.

If multiple DAG-Knights all vote on the same tx-set, you get redundancy/fault tolerance, not linear throughput.

---

## Smarter hybrid: keep consensus singular, parallelize everything else

## Layer A — Canonical ordering (single DAG-Knight lane)
- Keep one canonical DAG-Knight consensus timeline.
- Tune cadence and anchor policy for predictable proof windows.
- Emit deterministic block descriptors (tx list + state roots + lane metadata).

## Layer B — Parallel execution scheduler
At block construction time:
- Build a read/write conflict graph from candidate transactions.
- Partition into **conflict-free execution lanes** (graph coloring / optimistic SCC splitting).
- Execute lanes in parallel with deterministic tie-break rules.
- Produce lane receipts:
  - pre-state root,
  - post-state root,
  - touched key commitments,
  - lane-local witness commitments.

This gives parallelism without exposing consensus to lane-level non-determinism.

## Layer C — q-IVC proof fan-out and fold
Map each lane receipt to a lane circuit instance, then fold:

- **C1 (lane proofs):** each lane proves valid state transition.
- **C2 (block merge proof):** proves all lane transitions compose to block post-state.
- **C3 (chain fold):** fold block proof into recursive chain tip proof.

Result: one succinct on-chain/verifier artifact while proving work scales horizontally.

---

## Optional "soft sharding" mode (execution-only shards)

If workload is heavily domain-separated (e.g., AMM pairs, NFT mints, transfers), introduce execution shards:

- Consensus still global/single (no consensus sharding).
- Shards only define execution queues and prover pools.
- Cross-shard actions become **intent messages** with delayed settlement windows.
- For strict atomic UX, use protocol-level batch intents plus timeout+refund semantics (instead of global 2PC).

This is a practical middle ground before true multi-consensus sharding.

---

## Concrete next implementation steps (repo-aligned)

1. Add a `ProverPool` abstraction in `crates/q-ivc/src/host/`:
   - work-stealing queue,
   - lane/block proof job descriptors,
   - deterministic merge ordering.
2. Refactor `DeltaBlockCircuit` to support:
   - per-lane witness segments,
   - composition constraints for lane merge.
3. Add a merge circuit in `crates/q-ivc/src/recursion/`:
   - verifies all lane proofs,
   - enforces canonical block state root.
4. Extend DAG-Knight mempool integration to export conflict-graph metadata for deterministic lane partitioning.
5. Add metrics:
   - proving lag vs consensus tip,
   - lane conflict ratio,
   - merge proof latency,
   - invalid-intent timeout rates (if soft sharding enabled).

---

## Decision rubric

Use this matrix to choose where to invest next:

- Need higher trustless verification scalability / faster bootstrap? -> prioritize q-IVC fan-out/fold.
- Need higher raw TPS with similar state locality? -> prioritize conflict-aware parallel execution lanes.
- Need massive state growth across distinct domains? -> add execution-only soft sharding first.
- Need strict synchronous composability across all state? -> avoid hard sharding as long as possible.

---

## Bottom line

A **hybrid DAG-Knight + q-IVC** architecture should treat parallelism as:

- **consensus-serial**,
- **execution-parallel**,
- **proof-massively-parallel**.

That captures most upside of "multiple DAG-Knights" without immediately paying full consensus-sharding complexity costs.
