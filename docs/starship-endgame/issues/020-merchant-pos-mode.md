# Issue #020: Merchant POS Mode (Point-of-Sale)

**State**: `closed` (component implemented, SSE confirmation uses existing balance-updated events)
**Priority**: HIGH
**Labels**: `payments`, `frontend`, `mobile`
**Assigned**: Beta
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

A simplified merchant-facing screen for accepting QR code payments in brick-and-mortar shops. The merchant enters an amount, a QR code is displayed, and the screen shows real-time payment confirmation via SSE.

## User Flow

```
1. Merchant opens quillon.xyz/pos (or taps "POS Mode" in wallet)
2. Enters amount: [  10.50  ] QUG
3. Optional memo: [ Coffee + pastry ]
4. Taps "Generate QR"
5. Large QR code fills screen (customer-facing)
6. Customer scans with phone → wallet auto-fills → taps Pay
7. Merchant screen shows:
   ┌──────────────────────────┐
   │                          │
   │     ✓ PAYMENT RECEIVED   │
   │                          │
   │     10.50 QUG            │
   │     "Coffee + pastry"    │
   │                          │
   │     TX: 0xabc...def      │
   │     Block: #123456       │
   │                          │
   │   [ New Payment ]        │
   └──────────────────────────┘
8. Merchant taps "New Payment" → back to step 2
```

## Acceptance Criteria

- [x] New route: `/pos` or POS mode toggle in existing wallet
- [x] Amount entry with large, touch-friendly numpad
- [x] QR code generation (full-screen, high contrast for scanning)
- [x] Real-time SSE listener for payment confirmation
- [x] Visual + audio confirmation (green flash + optional sound)
- [x] Transaction receipt display (amount, memo, tx hash, timestamp)
- [x] "New Payment" button to reset for next customer
- [x] Works on tablets and phones (responsive, touch-optimized)
- [ ] No login required for QR display (wallet address from URL param)
- [ ] Optional: Print receipt button (window.print())

## Technical Details

The POS mode is a self-contained React component that:
1. Creates a payment request via `POST /api/v1/payment-requests` (Issue #019)
2. Displays the QR code from the response
3. Subscribes to SSE stream filtered by wallet address
4. Watches for `payment-confirmed` events matching the request_id
5. Shows confirmation when payment is detected

## Mobile Optimization

- Minimum touch target: 48x48px (WCAG 2.5.8)
- Amount input: Large font (32px+), centered
- QR code: Maximum size that fits viewport (min 250x250px)
- Confirmation: Full-screen green overlay with amount
- Works in landscape (tablet on counter) and portrait (phone)

## Files

- `gui/quantum-wallet/src/components/POSMode.tsx` — NEW: POS component
- `gui/quantum-wallet/src/App.tsx` — Add /pos route
