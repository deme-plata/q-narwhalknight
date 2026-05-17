# Real-Time DEX Implementation Plan

## Overview
Implement comprehensive SSE real-time updates for all DEX data, remove all mock data, and implement real swap functionality.

## Current State Analysis

### Mock Data Found
1. **Token Price Charts** (TokenDetailsModal.tsx lines 76-120)
   - Generates fake price data with 100ms intervals
   - Random walk with artificial trends
   - Not connected to real oracle data

2. **Token Transaction History** (TokenDetailsModal.tsx lines 122-166)
   - Generates 50 fake transactions
   - Random wallet addresses
   - No connection to blockchain

3. **Swap Functionality** (DexScreen.tsx line 1089)
   - "Swap Tokens" button does nothing
   - Just a placeholder

### Working SSE Already in Place
- **Nitro Boosts** (DexScreen.tsx lines 61-176)
  - ✅ Already implemented
  - Listens to `nitro_boost` and `nitro_boosts_update` events
  - Updates `boostedTokens` Map in real-time

## Implementation Tasks

### 1. Backend: Add SSE Broadcasting for Token Data

#### 1.1 Add Token Price/Change SSE Events
**File**: `crates/q-api-server/src/handlers.rs`

Add handler to broadcast token price updates:
```rust
/// Broadcast token price update via SSE
pub async fn broadcast_token_price_update(
    state: &Arc<AppState>,
    token_id: String,
    price: f64,
    change_24h: f64,
    volume_24h: f64,
) {
    let sse_event = crate::StreamEvent::Custom {
        event_type: "token_price_update".to_string(),
        data: serde_json::json!({
            "token_id": token_id,
            "price": price,
            "change_24h": change_24h,
            "volume_24h": volume_24h,
            "timestamp": chrono::Utc::now().timestamp()
        }),
        timestamp: chrono::Utc::now(),
    };

    if let Err(e) = state.event_broadcaster.broadcast(sse_event) {
        warn!("Failed to broadcast token price update: {}", e);
    }
}
```

#### 1.2 Modify Oracle Handler to Broadcast Updates
**File**: `crates/q-api-server/src/handlers.rs`

Find the oracle price endpoint and add SSE broadcasting after price updates.

#### 1.3 Add Transaction SSE Events
```rust
/// Broadcast new transaction via SSE
pub async fn broadcast_transaction(
    state: &Arc<AppState>,
    token_id: String,
    tx_type: String, // "buy", "sell", "transfer"
    amount: f64,
    price: f64,
    from: String,
    to: String,
    tx_hash: String,
) {
    let sse_event = crate::StreamEvent::Custom {
        event_type: "token_transaction".to_string(),
        data: serde_json::json!({
            "token_id": token_id,
            "type": tx_type,
            "amount": amount,
            "price": price,
            "value": amount * price,
            "from": from,
            "to": to,
            "tx_hash": tx_hash,
            "timestamp": chrono::Utc::now().timestamp_millis()
        }),
        timestamp: chrono::Utc::now(),
    };

    if let Err(e) = state.event_broadcaster.broadcast(sse_event) {
        warn!("Failed to broadcast transaction: {}", e);
    }
}
```

#### 1.4 Add Real-Time Price Data Points SSE
```rust
/// Broadcast price data point (for live charts)
pub async fn broadcast_price_datapoint(
    state: &Arc<AppState>,
    token_id: String,
    price: f64,
    volume: f64,
) {
    let sse_event = crate::StreamEvent::Custom {
        event_type: "price_datapoint".to_string(),
        data: serde_json::json!({
            "token_id": token_id,
            "price": price,
            "volume": volume,
            "timestamp": chrono::Utc::now().timestamp_millis()
        }),
        timestamp: chrono::Utc::now(),
    };

    if let Err(e) = state.event_broadcaster.broadcast(sse_event) {
        warn!("Failed to broadcast price datapoint: {}", e);
    }
}
```

### 2. Backend: Implement Real Swap Functionality

#### 2.1 Add Swap Handler
**File**: `crates/q-api-server/src/handlers.rs`

```rust
#[derive(Debug, Deserialize)]
pub struct SwapRequest {
    pub from_token: String,  // Token ID or "QUG" for native
    pub to_token: String,
    pub amount_in: u64,
    pub min_amount_out: u64, // Slippage protection
    pub wallet_address: String,
}

pub async fn execute_swap(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SwapRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    info!("Executing swap: {} {} for {}", request.amount_in, request.from_token, request.to_token);

    // 1. Validate tokens exist
    // 2. Check user balance
    // 3. Calculate exchange rate from liquidity pools
    // 4. Apply slippage check
    // 5. Execute swap (update balances)
    // 6. Broadcast transaction SSE event
    // 7. Broadcast price update SSE event
    // 8. Return swap result

    // TODO: Implement full swap logic with liquidity pool integration

    Ok(Json(ApiResponse::success(serde_json::json!({
        "from_token": request.from_token,
        "to_token": request.to_token,
        "amount_in": request.amount_in,
        "amount_out": calculated_amount_out,
        "exchange_rate": rate,
        "transaction_id": tx_id
    }))))
}
```

#### 2.2 Wire Up Swap Route
**File**: `crates/q-api-server/src/main.rs`

Add route:
```rust
.route("/api/v1/dex/swap", post(handlers::execute_swap))
```

### 3. Frontend: Add SSE for Token Price Updates

#### 3.1 Update DexScreen.tsx
Add SSE listener for token price updates (add to existing SSE setup around line 100):

```typescript
// Listen for token price updates
eventSource.addEventListener('token_price_update', (event) => {
  if (!mounted) return;
  try {
    const data = JSON.parse(event.data);
    console.log('📈 Received token price update:', data);

    // Update token in list
    setTokens(prev => prev.map(token =>
      token.id === data.token_id
        ? { ...token, price: data.price, change24h: data.change_24h, volume24h: data.volume_24h }
        : token
    ));
  } catch (err) {
    console.error('Failed to parse token price update:', err);
  }
});
```

### 4. Frontend: Remove Mock Price Data

#### 4.1 Update TokenDetailsModal.tsx
**Replace lines 76-120 with API integration:**

```typescript
// Fetch real price history from backend
useEffect(() => {
  if (!token) return;

  const fetchPriceHistory = async () => {
    try {
      const response = await qnkAPI.getTokenPriceHistory(token.id, timeframe);
      if (response.success && response.data) {
        setPriceData(response.data);
      }
    } catch (error) {
      console.error('Failed to fetch price history:', error);
      // Fallback to current price if no history
      setPriceData([{
        timestamp: Date.now(),
        price: token.price,
        volume: token.volume24h
      }]);
    }
  };

  fetchPriceHistory();

  // Set up SSE for real-time price updates
  const sseUrl = import.meta.env.VITE_API_URL ?
    `${import.meta.env.VITE_API_URL}/v1/events` :
    '/api/v1/events';

  const eventSource = new EventSource(sseUrl);

  eventSource.addEventListener('price_datapoint', (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.token_id === token.id) {
        // Append new data point to chart
        setPriceData(prev => [...prev, {
          timestamp: data.timestamp,
          price: data.price,
          volume: data.volume
        }]);
      }
    } catch (err) {
      console.error('Failed to parse price datapoint:', err);
    }
  });

  return () => {
    eventSource.close();
  };
}, [token, timeframe]);
```

### 5. Frontend: Remove Mock Transaction Data

#### 5.1 Update TokenDetailsModal.tsx
**Replace lines 122-166 with API integration:**

```typescript
// Fetch real transaction history from backend
useEffect(() => {
  if (!token) return;

  const fetchTransactions = async () => {
    try {
      const response = await qnkAPI.getTokenTransactions(token.id);
      if (response.success && response.data) {
        setTransactions(response.data);
      }
    } catch (error) {
      console.error('Failed to fetch transactions:', error);
      setTransactions([]);
    }
  };

  fetchTransactions();

  // Set up SSE for real-time transaction updates
  const sseUrl = import.meta.env.VITE_API_URL ?
    `${import.meta.env.VITE_API_URL}/v1/events` :
    '/api/v1/events';

  const eventSource = new EventSource(sseUrl);

  eventSource.addEventListener('token_transaction', (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.token_id === token.id) {
        // Prepend new transaction to list
        setTransactions(prev => [data, ...prev].slice(0, 50)); // Keep last 50
      }
    } catch (err) {
      console.error('Failed to parse transaction:', err);
    }
  });

  return () => {
    eventSource.close();
  };
}, [token]);
```

### 6. Frontend: Implement Real Swap

#### 6.1 Update DexScreen.tsx Swap Button
**Replace line 1089 with real swap handler:**

```typescript
const handleSwap = async () => {
  if (!swapAmount || parseFloat(swapAmount) <= 0) {
    alert('Please enter a valid swap amount');
    return;
  }

  const walletAddress = localStorage.getItem('walletAddress');
  if (!walletAddress) {
    alert('Please connect your wallet first');
    return;
  }

  // Find token IDs
  const fromToken = tokens.find(t => t.symbol === swapFrom);
  const toToken = tokens.find(t => t.symbol === swapTo);

  if (!fromToken || !toToken) {
    alert('Invalid token selection');
    return;
  }

  // Check balance
  if (fromToken.balance < parseFloat(swapAmount)) {
    alert(`Insufficient ${swapFrom} balance. You have ${fromToken.balance.toFixed(4)}`);
    return;
  }

  try {
    // Calculate minimum output with slippage (0.5%)
    const expectedOutput = parseFloat(swapAmount) * 0.95; // Mock rate for now
    const minOutput = expectedOutput * 0.995; // 0.5% slippage tolerance

    const response = await qnkAPI.executeSwap({
      from_token: fromToken.id === 'native-qug' ? 'QUG' : fromToken.id,
      to_token: toToken.id === 'qugusd-stable' ? 'QUGUSD' : toToken.id,
      amount_in: Math.floor(parseFloat(swapAmount) * 1_000_000_000), // Convert to base units
      min_amount_out: Math.floor(minOutput * 1_000_000_000),
      wallet_address: walletAddress
    });

    if (response.success && response.data) {
      alert(`✅ Swap successful!\n\nSwapped: ${swapAmount} ${swapFrom}\nReceived: ${(response.data.amount_out / 1_000_000_000).toFixed(4)} ${swapTo}\n\nTransaction: ${response.data.transaction_id}`);

      // Refresh tokens to update balances
      window.location.reload();
    } else {
      alert(`❌ Swap failed: ${response.error || 'Unknown error'}`);
    }
  } catch (error) {
    console.error('Swap failed:', error);
    alert('❌ Swap failed. Please try again.');
  }
};

// Update button
<button
  onClick={handleSwap}
  className="w-full py-4 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-xl font-bold text-white hover:shadow-lg hover:shadow-quantum-cyan/50 transition-all"
>
  Swap Tokens
</button>
```

### 7. Frontend: Add API Methods

#### 7.1 Update api.ts
**File**: `gui/quantum-wallet/src/services/api.ts`

Add these methods:
```typescript
// Get token price history
async getTokenPriceHistory(tokenId: string, timeframe: string): Promise<ApiResponse<PriceDataPoint[]>> {
  return this.get(`/api/v1/oracle/price-history/${tokenId}?timeframe=${timeframe}`);
},

// Get token transactions
async getTokenTransactions(tokenId: string): Promise<ApiResponse<TokenTransaction[]>> {
  return this.get(`/api/v1/transactions/token/${tokenId}`);
},

// Execute swap
async executeSwap(request: {
  from_token: string;
  to_token: string;
  amount_in: number;
  min_amount_out: number;
  wallet_address: string;
}): Promise<ApiResponse<any>> {
  return this.post('/api/v1/dex/swap', request);
}
```

## Implementation Order

1. ✅ **Phase 1**: Nitro Boosts SSE (COMPLETE)
2. **Phase 2**: Backend swap functionality
3. **Phase 3**: Backend SSE for token prices/transactions
4. **Phase 4**: Frontend swap implementation
5. **Phase 5**: Frontend SSE for price updates in token list
6. **Phase 6**: Remove mock price charts, add real data
7. **Phase 7**: Remove mock transactions, add real data
8. **Phase 8**: Test everything end-to-end

## Testing Plan

### Manual Tests
1. Open two wallets side-by-side
2. Execute swap in wallet A → Verify balance updates in both
3. Watch token price update in real-time without refresh
4. Open token details modal → Verify live price chart
5. Execute transaction → Verify it appears in transaction list instantly
6. Boost a token → Verify it moves to top in both wallets

### Expected Behavior
- No page refreshes needed
- All data updates in real-time via SSE
- Swap executes with proper balance checks
- Transaction history is real (from blockchain)
- Price charts show real oracle data

## Files to Modify

### Backend
- `crates/q-api-server/src/handlers.rs` - Add SSE broadcasting + swap handler
- `crates/q-api-server/src/main.rs` - Wire up swap route

### Frontend
- `gui/quantum-wallet/src/components/DexScreen.tsx` - Add SSE for prices, implement swap
- `gui/quantum-wallet/src/components/TokenDetailsModal.tsx` - Remove mocks, add SSE
- `gui/quantum-wallet/src/services/api.ts` - Add API methods

## Current Status
- [x] Nitro Boosts SSE working
- [ ] Token price SSE
- [ ] Transaction SSE
- [ ] Real swap functionality
- [ ] Remove mock price data
- [ ] Remove mock transaction data

---

**Next Step**: Start with implementing backend swap functionality, then add SSE broadcasting for all data points.
