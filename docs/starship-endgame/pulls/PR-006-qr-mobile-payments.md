# PR #006: QR Code Mobile Payments for Brick-and-Mortar

**State**: `merged` (commit 32503bb8)
**Head**: `feature/safe-batched-sync-v1.0.2`
**Base**: `main`
**Author**: Server Beta
**Created**: 2026-03-10
**Labels**: `payments`, `mobile`, `pos`
**Closes**: #019, #020

---

## Summary

Implements QR-code-based mobile payments for physical shops using the QNK (QUG) cryptocurrency. Mirrors the NFC payment experience but using QR codes — the approach used by Bitcoin Lightning, Solana Pay, and most crypto payment systems.

### Technology Choice: QR Codes (not NFC)

NFC payments require Apple Pay/Google Pay certification and are locked to card networks. QR codes work on ANY phone with a camera, require no third-party certification, and provide instant confirmation via SSE (2-3 second DAG-Knight finality).

### What's included

- **Payment Request API** (`payment_request_api.rs`)
  - `POST /api/v1/payment-requests` — Create payment request with TTL
  - `GET /api/v1/payment-requests/:id` — Check status
  - SSE `payment-confirmed` event for real-time notification
  - URI format: `quillon:ADDRESS?amount=X&memo=Y&request_id=Z`

- **Merchant POS Mode** (`POSMode.tsx`)
  - Touch-friendly amount entry for tablets/phones
  - Full-screen QR code display for customer scanning
  - Real-time payment confirmation with visual feedback
  - Receipt display with tx hash and block height
  - Works offline for QR display, needs connectivity for confirmation

## Payment Flow

```
Merchant (tablet/phone)          Customer (phone)
        │                              │
        ├─ Enter amount: 10.50 QUG ───►│
        │                              │
        ├─ Generate QR code ──────────►│
        │  (quillon:0xABC?             │
        │   amount=10.5&               │
        │   memo=Coffee&               │
        │   request_id=pay_xyz)        │
        │                              │
        │              Scan QR ◄───────┤
        │              Wallet fills ◄──┤
        │              Tap "Pay" ◄─────┤
        │                              │
        │  ✓ PAYMENT RECEIVED ◄────────┤
        │  (SSE: payment-confirmed)    │
        │                              │
        ├─ Show receipt ──────────────►│
        │  [New Payment]               │
```

## Files Changed

| File | Change |
|------|--------|
| `crates/q-api-server/src/payment_request_api.rs` | NEW — Payment request endpoints |
| `crates/q-api-server/src/lib.rs` | MODIFIED — Add payment request routes |
| `crates/q-api-server/src/streaming.rs` | MODIFIED — Add payment-confirmed SSE event |
| `gui/quantum-wallet/src/components/POSMode.tsx` | NEW — Merchant POS component |
| `gui/quantum-wallet/src/App.tsx` | MODIFIED — Add /pos route |

## Test Plan

- [ ] Create payment request → returns valid QR URI
- [ ] Payment request expires after TTL
- [ ] SSE fires `payment-confirmed` when matching tx received
- [ ] POS mode renders on mobile (< 400px width)
- [ ] POS mode renders on tablet (landscape)
- [ ] QR code scannable by phone cameras
- [ ] Amount auto-fills in wallet after QR scan
- [ ] "New Payment" resets state correctly

## Risk Assessment

- **Consensus impact**: ZERO — uses existing transaction system
- **Security**: Payment requests are ephemeral (in-memory, auto-expire)
- **Privacy**: No customer data stored, only wallet addresses
- **Rollback**: Safe — removing routes returns to previous behavior
