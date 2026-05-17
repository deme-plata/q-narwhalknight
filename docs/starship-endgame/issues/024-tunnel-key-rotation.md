# Issue #024: Tunnel Encryption Key Rotation — Forward Secrecy

**State**: `in_progress`
**Priority**: MEDIUM
**Labels**: `starship-endgame`, `security`, `p2p`
**Assigned**: Beta+Epsilon
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

`CryptoHandshake` in `tunnel.rs` establishes NOISE XX encrypted tunnels with static Curve25519 keys. Once a tunnel is established, the same session keys are used for the tunnel's entire lifetime. If a session key is compromised, all past and future traffic on that tunnel is exposed.

We need periodic key rotation (rekeying) to provide forward secrecy — compromising one epoch's keys reveals nothing about other epochs.

## Current State

- `CryptoHandshake` uses `Noise_XX_25519_ChaChaPoly_SHA256` pattern
- Keys are generated once during handshake and never rotated
- `TunnelStream` wraps the NOISE transport state but has no rekey mechanism
- Snow crate supports `rekey()` on `TransportState` but we don't call it
- No epoch-based rotation, no key derivation for forward secrecy

## Key Rotation Protocol

```
Epoch 0: Initial NOISE XX handshake → session keys K0
  ... traffic encrypted with K0 ...

Epoch 1 (after 1 hour or 10GB transferred, whichever first):
  → Initiator sends REKEY message
  → Both sides call transport_state.rekey()
  → New session keys K1 derived from K0 + fresh entropy
  → K0 securely zeroed (zeroize crate)
  ... traffic encrypted with K1 ...

Epoch 2: Same pattern → K2 from K1
```

## Acceptance Criteria

- [ ] `TunnelStream::rekey()` calls `snow::TransportState::rekey()`
- [ ] Auto-rekey every 1 hour or 10GB transferred (configurable)
- [ ] Old keys zeroed with `zeroize` crate after rotation
- [ ] REKEY coordination message between peers (both sides must rekey simultaneously)
- [ ] Graceful handling of rekey failure (close tunnel, re-establish)
- [ ] Rekey counter in tunnel stats (`GET /api/v1/compute/tunnels`)
- [ ] Test: verify old ciphertext undecryptable after rekey

## Depends On

- #002 (P2P tunnel NOISE XX handshake — must be complete first)

## Progress

**Current**: tunnel_rekey.rs (477 lines) — RekeyManager with epoch-based rotation (1 hour or 10GB threshold), zeroize integration for old key cleanup. Auto-rekey coordination via TunnelPayload::Rekey message. Rekey stats tracking in tunnel telemetry.

## Files

- `crates/q-compute/src/tunnel_rekey.rs` — RekeyManager, epoch tracking, zeroize
- `crates/q-compute/src/tunnel.rs` — TunnelStream::rekey() integration, TunnelPayload::Rekey variant
- `crates/q-compute/Cargo.toml` — zeroize dependency
