# AI API Payment System - Design Document

**Date**: October 29, 2025
**Status**: Design Phase
**Target**: Implement pay-per-token AI API with competitive pricing

---

## Overview

Implement a payment system for the AI chat API that:
1. Requires wallet balance to use AI features
2. Deducts tokens dynamically based on AI usage
3. Uses oracle price feeds for dynamic USD pricing
4. Supports both native token (QNK) and QUGUSD stablecoin
5. Offers competitive pricing lower than market rates

---

## Competitive Pricing Analysis

### Market Rates (as of 2025):
- **OpenAI GPT-4**: $0.03 per 1K input tokens, $0.06 per 1K output tokens
- **Anthropic Claude**: $0.015 per 1K input tokens, $0.075 per 1K output tokens
- **Google Gemini**: $0.00025 per 1K tokens (Pro), $0.0001 per 1K tokens (Flash)

### Q-NarwhalKnight Target Pricing (50% cheaper than competition):
- **Base Rate**: $0.0001 per 1K tokens (~10x cheaper than GPT-4)
- **Input Tokens**: $0.00005 per 1K tokens
- **Output Tokens**: $0.0001 per 1K tokens

**Rationale**:
- Distributed AI reduces infrastructure costs
- Lower margins to gain market share
- Dynamic pricing allows adjustments based on network load

---

## Architecture

```
┌─────────────────┐
│   User Wallet   │
│  (QNK Balance)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Payment Middleware Layer      │
│ - Check balance                 │
│ - Calculate cost (oracle price) │
│ - Reserve tokens                │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   AI Inference Endpoint         │
│ - Generate tokens               │
│ - Track usage                   │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Payment Settlement            │
│ - Deduct exact cost             │
│ - Refund unused reservation     │
│ - Log transaction               │
└─────────────────────────────────┘
```

---

## Database Schema

### New Tables

#### `ai_credits` table:
```sql
wallet_address TEXT PRIMARY KEY,
balance_qnk INTEGER NOT NULL DEFAULT 0,
balance_qugusd INTEGER NOT NULL DEFAULT 0,
total_spent_qnk INTEGER NOT NULL DEFAULT 0,
total_spent_qugusd INTEGER NOT NULL DEFAULT 0,
total_tokens_generated INTEGER NOT NULL DEFAULT 0,
created_at INTEGER NOT NULL,
updated_at INTEGER NOT NULL
```

#### `ai_transactions` table:
```sql
tx_id TEXT PRIMARY KEY,
wallet_address TEXT NOT NULL,
chat_id TEXT NOT NULL,
input_tokens INTEGER NOT NULL,
output_tokens INTEGER NOT NULL,
cost_usd_cents INTEGER NOT NULL,
cost_qnk INTEGER NOT NULL,
payment_token TEXT NOT NULL (QNK or QUGUSD),
oracle_price_usd_cents INTEGER NOT NULL,
timestamp INTEGER NOT NULL,
status TEXT NOT NULL (pending, completed, refunded)
```

---

## Oracle Integration

### Price Feed Sources:
1. **Internal Q-Oracle** (crates/q-oracle) - consensus-based price aggregation
2. **Fallback**: CoinGecko API for QNK/USD price
3. **Update Frequency**: Every 60 seconds

### Price Calculation:
```rust
// Example: 1000 output tokens
let tokens = 1000;
let cost_usd_cents = (tokens as f64 * 0.01) as u64; // $0.0001 per 1K tokens = 0.01 cents per token
let qnk_price_usd_cents = oracle.get_qnk_price_usd_cents().await?; // e.g., 500 ($5.00 per QNK)
let cost_qnk = (cost_usd_cents * 100_000_000) / qnk_price_usd_cents; // Convert to atomic units
```

---

## Payment Flow

### 1. Pre-Generation Check:
```
POST /api/chat/{id}/stream?content=...&max_tokens=512

→ Check wallet balance
→ Calculate max cost (512 tokens * price)
→ Reserve funds
→ If insufficient balance: Return 402 Payment Required
→ If sufficient: Proceed to generation
```

### 2. During Generation:
```
→ Track actual tokens generated
→ Update cost in real-time
```

### 3. Post-Generation Settlement:
```
→ Calculate final cost (actual tokens * price)
→ Deduct from wallet
→ Refund unused reservation
→ Log transaction
→ Return generation stats + cost
```

---

## API Changes

### New Endpoints:

#### GET /api/wallet/{address}/credits
```json
{
  "success": true,
  "data": {
    "wallet_address": "0x...",
    "balance_qnk": 1000000000,
    "balance_qugusd": 500000000,
    "total_spent_qnk": 50000000,
    "total_tokens_generated": 1000000,
    "current_price_usd_per_qnk": 5.00
  }
}
```

#### POST /api/wallet/{address}/deposit
```json
{
  "token": "QNK",
  "amount": 1000000000,
  "tx_hash": "0x..."
}
```

### Modified Endpoints:

#### POST /api/chat/{id}/stream (Now requires payment)
**Request**:
```
POST /api/chat/{chat_id}/stream?content=Hello&max_tokens=100
Headers:
  X-Wallet-Address: 0x...
  X-Wallet-Signature: 0x... (sign: "ai-inference:{chat_id}:{timestamp}")
```

**Response** (SSE Stream):
```
event: token
data: {"token": "Hello", "cumulative": "Hello", "cost_usd_cents": 0.001}

event: complete
data: {
  "tokens_generated": 50,
  "latency_ms": 1000,
  "tokens_per_second": 50,
  "cost_usd_cents": 0.005,
  "cost_qnk": 1000,
  "payment_token": "QNK",
  "balance_remaining_qnk": 999999000
}
```

---

## Implementation Steps

### Phase 1: Database & Storage (1-2 days)
- [ ] Add `ai_credits` table to storage schema
- [ ] Add `ai_transactions` table
- [ ] Implement credit management functions
- [ ] Add transaction logging

### Phase 2: Oracle Integration (1-2 days)
- [ ] Integrate q-oracle crate
- [ ] Implement price fetching (QNK/USD)
- [ ] Add price caching (60s TTL)
- [ ] Implement fallback price feeds

### Phase 3: Payment Middleware (2-3 days)
- [ ] Create payment middleware module
- [ ] Implement wallet authentication
- [ ] Add balance checking
- [ ] Implement token reservation
- [ ] Add payment settlement logic

### Phase 4: API Integration (1-2 days)
- [ ] Modify chat endpoints to require payment
- [ ] Add credit management endpoints
- [ ] Implement real-time cost tracking in SSE stream
- [ ] Add payment error handling

### Phase 5: Frontend Integration (1-2 days)
- [ ] Add wallet balance display
- [ ] Show cost per message
- [ ] Add deposit flow UI
- [ ] Display transaction history

### Phase 6: Testing & Deployment (1-2 days)
- [ ] Unit tests for payment logic
- [ ] Integration tests
- [ ] Load testing with payment
- [ ] Deploy to production

**Total Estimated Time**: 7-12 days

---

## Security Considerations

1. **Signature Verification**: All API calls must include wallet signature
2. **Rate Limiting**: Prevent abuse even with valid payments
3. **Double-Spend Prevention**: Atomic balance updates with database transactions
4. **Price Oracle Security**: Use multiple price sources, detect manipulation
5. **Refund Protection**: Automatic refunds if generation fails

---

## Configuration

### Environment Variables:
```bash
# Pricing
AI_PRICING_MODEL=pay_per_token
AI_INPUT_TOKEN_PRICE_USD_CENTS=0.00005  # $0.00005 per 1K tokens
AI_OUTPUT_TOKEN_PRICE_USD_CENTS=0.0001  # $0.0001 per 1K tokens

# Oracle
ORACLE_UPDATE_INTERVAL_SECS=60
ORACLE_FALLBACK_API=https://api.coingecko.com/api/v3/simple/price

# Payment
MIN_BALANCE_QNK=10000000  # 0.1 QNK minimum
PAYMENT_RESERVE_BUFFER=1.1  # Reserve 110% of estimated cost
```

---

## Future Enhancements

1. **Subscription Plans**: Monthly flat-rate for power users
2. **Volume Discounts**: Cheaper rates for high-volume users
3. **Free Tier**: 1000 tokens/day for new users
4. **Token Bundles**: Pre-purchase tokens at discount
5. **Referral Credits**: Earn credits by referring users
6. **Staking Rewards**: Stake QNK for discounted AI access

---

## Success Metrics

- **Cost per 1K tokens**: <$0.0001 (10x cheaper than GPT-4)
- **Payment latency**: <50ms overhead
- **Oracle uptime**: >99.9%
- **Transaction accuracy**: 100% (no double-charges)
- **User adoption**: Target 1000 paid users in first month

---

*This is a comprehensive system that requires careful implementation. Let's start with Phase 1 and build incrementally.*
