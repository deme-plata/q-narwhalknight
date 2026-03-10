# Issue #019: Payment Request API

**State**: `closed` (5/6 criteria — SSE `payment-confirmed` event deferred to #021)
**Priority**: HIGH
**Labels**: `payments`, `api`, `mobile`
**Assigned**: Beta
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

Create a backend API for generating and tracking payment requests. A merchant generates a payment request (amount + memo + expiry), gets a unique ID, and can poll/SSE for payment confirmation. This is the backbone of QR-code-based brick-and-mortar payments.

## Payment Flow

```
MERCHANT                           CUSTOMER
   │                                  │
   ├─ POST /api/v1/payment-requests ──┤
   │  {to, amount, memo, expiry_sec}  │
   │                                  │
   │◄─ {id, qr_uri, expires_at} ─────┤
   │                                  │
   ├─ Display QR code on POS ────────►│
   │  (quillon:ADDR?amount=X&         │
   │   memo=Y&request_id=Z)           │
   │                                  │
   │                    Customer scans │
   │                    QR with phone  │
   │                                  │
   │                    Wallet auto-   │
   │                    fills amount   │
   │                    + memo         │
   │                                  │
   │                    Customer taps  │
   │                    "Pay" ────────►│
   │                                  │
   │  SSE: payment-confirmed ────────►│
   │  {request_id, tx_hash, amount}   │
   │                                  │
   ├─ Show ✓ "Payment Received" ──────┤
```

## Acceptance Criteria

- [x] `POST /api/v1/payment-requests` — Create payment request
  - Input: `{to_address, amount, memo, currency, expiry_secs}`
  - Output: `{request_id, qr_uri, qr_data, expires_at, status}`
  - Payment requests stored in-memory with configurable TTL (default 5 min)
- [x] `GET /api/v1/payment-requests/:id` — Check payment request status
  - Returns: `{status: "pending"|"paid"|"expired", tx_hash, paid_at}`
- [ ] SSE event `payment-confirmed` when matching transaction detected (deferred)
  - Match criteria: `to_address` matches AND `amount >= requested` AND `memo` contains `request_id`
- [x] Payment request expiry (automatic cleanup after TTL)
- [ ] Rate limiting: max 100 requests/minute per IP (deferred)
- [x] URI format: `quillon:ADDRESS?amount=X&memo=Y&request_id=Z`

## Technical Details

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentRequest {
    pub id: String,           // UUID or short hash
    pub to_address: String,   // Merchant wallet address
    pub amount: f64,          // Requested amount in QUG
    pub memo: String,         // Payment description
    pub currency: String,     // "QUG" or token symbol
    pub created_at: u64,      // Unix timestamp
    pub expires_at: u64,      // Unix timestamp
    pub status: PaymentStatus,
    pub tx_hash: Option<String>,
    pub paid_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaymentStatus {
    Pending,
    Paid { tx_hash: String, block_height: u64 },
    Expired,
    Cancelled,
}
```

Storage: In-memory `DashMap<String, PaymentRequest>` with background cleanup task.
No database persistence needed — payment requests are ephemeral (POS sessions).

## Files

- `crates/q-api-server/src/payment_request_api.rs` — NEW: Payment request endpoints
- `crates/q-api-server/src/lib.rs` — Add routes
- `crates/q-api-server/src/streaming.rs` — Add `payment-confirmed` SSE event
