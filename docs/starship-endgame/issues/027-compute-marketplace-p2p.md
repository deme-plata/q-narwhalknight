# Issue #027: Compute Marketplace P2P Protocol — Bid/Ask for Work

**State**: `open`
**Priority**: MEDIUM
**Labels**: `starship-endgame`, `p2p`, `economics`
**Assigned**: Epsilon
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

Issue #017 describes the Proof-of-Useful-Work marketplace concept, but the P2P protocol for it doesn't exist. Nodes need a way to:

1. **Announce** available compute capacity (I have 6 idle cores + 1 GPU)
2. **Request** compute work (I need 100 ZK proofs generated)
3. **Bid** on work (I'll do it for 50 micro-QUG per proof)
4. **Accept** bids (job assigned to lowest bidder with reputation > threshold)
5. **Verify** results (spot-check or full verification)
6. **Settle** payment (on-chain after verification)

## Current State

- Gossipsub topic `/qnk/mainnet2026.1/compute-capacity` exists for capacity announcements
- `PeerRegistry` tracks peer capacity but not pricing/availability
- No bid/ask protocol — work assignment is coordinator-decided, not market-driven
- `TunnelPayload` has task types but no marketplace message types

## P2P Message Types

```rust
enum MarketplaceMessage {
    // Seller (compute provider) messages
    CapacityAnnouncement { cores: u16, gpu: Option<GpuDevice>, price_per_cpu_sec: u64 },
    BidSubmission { job_id: [u8; 32], price: u64, estimated_duration_ms: u64 },
    ResultSubmission { job_id: [u8; 32], result_hash: [u8; 32], proof: Vec<u8> },

    // Buyer (job submitter) messages
    JobPosting { job_id: [u8; 32], task_type: ComputeLayer, requirements: JobRequirements },
    BidAcceptance { job_id: [u8; 32], winner_peer: PeerId },
    VerificationResult { job_id: [u8; 32], accepted: bool },
}
```

## Acceptance Criteria

- [ ] Gossipsub topic `/qnk/{network}/compute-marketplace` for marketplace messages
- [ ] `MarketplaceMessage` enum with serialization (serde + bincode)
- [ ] `MarketplaceManager` struct with order book (bids sorted by price)
- [ ] Job posting → bid collection (5s window) → winner selection
- [ ] Winner selection: lowest price among peers with reputation > 0.5
- [ ] Result verification via `ResultVerifier` (existing in tunnel.rs)
- [ ] Settlement via `PaaSBillingManagerV2` after verification passes
- [ ] `GET /api/v1/compute/marketplace/orderbook` — Current bids/asks

## Depends On

- #002 (P2P tunnels — encrypted channels for job data transfer)
- #017 (Proof-of-Useful-Work — marketplace concept)
- #021 (Billing — settlement after job completion)
- #022 (Node reputation — winner selection criteria)

## Files

- `crates/q-compute/src/marketplace.rs` — NEW: Order book + bid/ask protocol
- `crates/q-compute/src/lib.rs` — `MarketplaceMessage` enum
- `crates/q-compute/src/tunnel.rs` — Wire marketplace into gossipsub
- `crates/q-api-server/src/compute_api.rs` — Marketplace REST endpoints
