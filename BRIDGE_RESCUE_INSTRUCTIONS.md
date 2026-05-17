# Bridge Rescue Project — Multi-Server Instructions

> **Status**: Phase 1 — Safety-first implementation
> **Branch**: `feature/bridge-rescue`
> **Created**: 2026-03-08
> **Coordinator**: Server Beta (185.182.185.227)

## Problem Statement

The Q-NarwhalKnight codebase has 4 cross-chain bridge APIs (Bitcoin, Ethereum, Zcash, IronFish) that are fully wired into the API server with routes, storage, frontend UIs, and a 7-of-11 multi-sig committee. However, **none of them can complete a real swap** because:

1. **No deposit verification** — wrapped tokens mint without proving the deposit happened
2. **Placeholder keys** — e.g., hardcoded `[0x02; 33]` bank pubkey in bitcoin_bridge_api.rs:220
3. **HTLC contract not deployed** — Ethereum `htlc_address` is `None`
4. **q-bitcoin-bridge crate deactivated** — commented out in Cargo.toml line 18
5. **No timeout/refund safety** — stalled swaps = lost funds with no recovery path
6. **No admin kill-switch** — no way to freeze bridge operations in an emergency

If someone clicks "Swap BTC → QNK" today, the API accepts the request and mints wrapped tokens, but **nobody verifies the Bitcoin deposit actually happened**. This is a money-loss scenario.

---

## Server Assignments

### Server Beta (185.182.185.227) — COORDINATOR
- **Role**: Bridge committee coordinator, API server, code changes
- **Tasks**: Fix bridge safety logic, add deposit verification, wire q-bitcoin-bridge crate
- **Working dir**: `/opt/orobit/shared/q-narwhalknight`
- **Claude Code**: Already running (primary dev server)

### Server Delta (5.79.79.158) — ETHEREUM BRIDGE NODE + BITCOIN KNOTS
- **Role**: Reth full node already running on port 8545, Bitcoin Knots Docker
- **Tasks**:
  - Verify Reth is synced: `curl localhost:8545 -X POST -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'`
  - Deploy HTLC smart contract on Ethereum
  - Run Bitcoin Knots Docker: `docker pull bitcoinknots/bitcoin:29.3`
- **Bitcoin Knots setup**:
  ```bash
  mkdir -p /opt/bitcoin-knots/data
  docker run -d --name bitcoin-knots \
    -v /opt/bitcoin-knots/data:/home/bitcoin/.bitcoin \
    -p 8332:8332 -p 8333:8333 \
    bitcoinknots/bitcoin:29.3 \
    -printtoconsole \
    -rpcallowip=185.182.185.227/32 \
    -rpcbind=0.0.0.0 \
    -rpcuser=qnk \
    -rpcpassword=CHANGE_ME_SECURE_PASSWORD \
    -prune=10000 \
    -txindex=0
  ```
- **Test**: `docker exec bitcoin-knots bitcoin-cli -rpcuser=qnk -rpcpassword=... getblockchaininfo`

### Server Gamma (109.205.176.60) — BACKUP BRIDGE NODE
- **Role**: Run bridge committee member, backup validator
- **Tasks**: Deploy bridge-enabled binary, participate in attestation gossipsub
- **RAM**: 7.8GB + 4GB swap (sufficient)

### Server Epsilon (89.149.241.126) — BRIDGE TESTING + ZCASH NODE
- **Role**: 10Gbit supernode, run Zcash lightwalletd for bridge testing
- **Tasks**: Install zcashd/lightwalletd, test shielded swap flow
- **Storage**: `/home/orobit/bridge-nodes/` (1.4TB NVMe free)
- **NEVER use `/tmp`** (40GB root partition)

### Server Alpha (161.35.219.10) — DOCKER INTEGRATION TESTS
- **Role**: Run bridge integration tests in Docker containers
- **Tasks**: Docker compose with regtest BTC + Ganache ETH for CI testing

---

## What Each Claude Code Instance Should Do On First Boot

1. Read this file
2. Read `CLAUDE.md` for server-specific paths and rules
3. Check assigned tasks above
4. Run: `git pull origin feature/bridge-rescue`
5. Follow the safety checklist below before touching ANY bridge code

---

## Safety Rules (MANDATORY)

1. **NEVER mint wrapped tokens without cryptographic proof of deposit**
2. **NEVER skip HTLC timelock verification**
3. **ALL swaps must have a refund path** (timeout-based)
4. **Bridge committee attestations required** for amounts > 0.1 BTC equivalent
5. **Test on testnet/regtest FIRST** — no mainnet bridge operations until 72-hour soak test
6. **Every bridge endpoint must validate**: deposit_tx exists, confirmations >= threshold, amount matches
7. **Max amount caps enforced**: BTC 0.1, ETH 1.0, ZEC 10, IRON 100 (initial limits)
8. **Admin kill-switch**: `POST /api/v1/bridge/admin/freeze` must work before going live

---

## Confirmation Thresholds

| Chain | Confirmations Required | Rationale |
|-------|----------------------|-----------|
| BTC   | >= 3                 | ~30 minutes, standard for exchanges |
| ETH   | >= 12                | ~3 minutes, post-merge finality |
| ZEC   | >= 10                | ~25 minutes, transparent + shielded |
| IRON  | >= 10                | ~10 minutes, similar to ZEC |

---

## Architecture After Rescue

```
┌─────────────────────────────────────────────────────┐
│                    BRIDGE NETWORK                     │
├──────────┬──────────┬──────────┬──────────┬─────────┤
│  Alpha   │   Beta   │  Gamma   │  Delta   │ Epsilon │
│ (Docker) │ (Coord)  │ (Backup) │ (ETH+BTC)│ (ZEC)   │
│ CI tests │ API+Cmte │ Cmte 2/3 │ Reth     │ zcashd  │
│          │ Cmte 1/3 │          │ BTC Knots│ Bridge  │
└──────────┴──────────┴──────────┴──────────┴─────────┘
         ↕ gossipsub: /qnk/.../bridge-attestations ↕
```

---

## Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `crates/q-api-server/src/bitcoin_bridge_api.rs` | BTC swap endpoints | 725 |
| `crates/q-api-server/src/ethereum_bridge_api.rs` | ETH swap endpoints | ~650 |
| `crates/q-api-server/src/zcash_bridge_api.rs` | ZEC swap endpoints | ~900 |
| `crates/q-api-server/src/ironfish_bridge_api.rs` | IRON swap endpoints | ~875 |
| `crates/q-api-server/src/bridge_committee.rs` | 7-of-11 multi-sig committee | 1200 |
| `crates/q-api-server/src/bridge_tokens.rs` | Wrapped token mint/burn | ~200 |
| `crates/q-api-server/src/bridge_safety.rs` | **NEW** — Deposit verification + safety | — |
| `crates/q-bitcoin-bridge/` | Bitcoin RPC client (deactivated) | ~300K |
| `Cargo.toml:18` | Workspace members (q-bitcoin-bridge commented) | — |

---

## Git Collaboration

**Repository**: `code.quillon.xyz` (self-hosted, served by Beta's nginx → git-http-backend)
**Branch**: `feature/bridge-rescue`

### Pull from any server:
```bash
git clone https://code.quillon.xyz/repo.git q-narwhalknight-src
# OR if already cloned:
cd /path/to/q-narwhalknight-src && git pull origin feature/bridge-rescue
```

### Before pulling on remote servers:
Run on Beta first: `cd /opt/orobit/shared/q-narwhalknight && git update-server-info`

### Task Tracking

| # | Title | Assignee | Priority | Status |
|---|-------|----------|----------|--------|
| A1 | Write BRIDGE_RESCUE_INSTRUCTIONS.md | Beta | Immediate | DONE |
| A2 | Add deposit verification to all 4 bridge APIs | Beta | CRITICAL | IN PROGRESS |
| A3 | Add swap timeouts + auto-refund background task | Beta | CRITICAL | PENDING |
| A4 | Replace placeholder keys with real derivation | Beta | HIGH | PENDING |
| A5 | Add admin kill-switch + max amount limits | Beta | HIGH | PENDING |
| A6 | Activate q-bitcoin-bridge crate | Beta | MEDIUM | PENDING |
| B1 | Set up Bitcoin Knots Docker on Delta | Delta | HIGH | PENDING |
| C1 | Verify Reth synced + deploy HTLC contract | Delta | HIGH | PENDING |
| D1 | Install zcashd on Epsilon | Epsilon | HIGH | PENDING |
| D2 | Install ironfish on Epsilon | Epsilon | HIGH | PENDING |
| E1 | Bridge committee multi-node testing | All | MEDIUM | PENDING |
| E2 | Frontend safety warnings + confirmation counters | Beta | MEDIUM | PENDING |
