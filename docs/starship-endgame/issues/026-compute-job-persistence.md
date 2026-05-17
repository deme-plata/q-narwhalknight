# Issue #026: Compute Job Queue Persistence — Survive Restarts

**State**: `in_progress`
**Priority**: MEDIUM
**Labels**: `starship-endgame`, `compute`, `reliability`
**Assigned**: Beta
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

All compute job queues are in-memory (`VecDeque` / `tokio::sync::mpsc`). When a node restarts — for auto-update, crash recovery, or OS reboot — all queued and in-flight jobs are lost. Clients get no notification and must resubmit. For long-running jobs (model fine-tuning, large ZK proofs, render batches), this means hours of wasted compute.

## Current State

- `InferenceWorkerPool` uses in-memory `mpsc` channel for task queue
- `TunnelManager::submit_verified_task()` is fire-and-forget
- `Orchestrator` layer assignments are rebuilt from scratch on startup
- No WAL (write-ahead log) for compute jobs
- `PaaSBillingManagerV2` has `BalanceReservation` but no job-to-reservation linkage

## Persistence Strategy

```
Job submitted → Write to RocksDB WAL { job_id, status: Queued, payload, submitted_at }
  → Orchestrator picks up → Update status: InProgress { started_at, worker_id }
  → Job completes → Update status: Completed { result_hash, duration_ms }
  → Settlement → Update status: Settled { tx_hash }

Node restarts:
  → Scan WAL for status: Queued | InProgress
  → Re-queue Queued jobs (in submission order)
  → InProgress jobs: if elapsed < timeout → re-queue; if elapsed > timeout → mark Failed
  → Notify clients of Failed jobs via SSE
```

## Acceptance Criteria

- [ ] `ComputeJobWAL` backed by RocksDB column family
- [ ] Jobs written to WAL before entering in-memory queue
- [ ] On restart: recover Queued jobs, timeout InProgress jobs
- [ ] Job status transitions: Queued → InProgress → Completed → Settled | Failed
- [ ] `GET /api/v1/compute/jobs/:id` — Job status + result
- [ ] `GET /api/v1/compute/jobs?status=queued` — List jobs by status
- [ ] SSE `compute-job-update` event for status transitions
- [ ] WAL compaction: remove Settled/Failed jobs older than 24 hours

## Depends On

- #001 (Orchestrator job dispatch)
- #021 (Billing — job-to-reservation linkage)

## Progress

**Current**: job_wal.rs (1153 lines) — Append-only WAL backed by RocksDB column family with CompactedRange tracking. Job status transitions: Queued → InProgress → Completed → Settled/Failed. Recovery on restart with timeout handling for long-running jobs. Compaction removes stale entries > 24 hours.

## Files

- `crates/q-compute/src/job_wal.rs` — ComputeJobWAL, recovery, compaction, timeout handling
- `crates/q-compute/src/orchestrator.rs` — WAL integration for job dispatch
- `crates/q-compute/src/inference_pool.rs` — Persist inference tasks
- `crates/q-api-server/src/compute_api.rs` — Job status endpoints
- `crates/q-api-server/src/streaming.rs` — `compute-job-update` SSE event
