# Nitro Points Backend Integration Guide

## Summary

Currently, Nitro Points boosts are stored only in browser localStorage, which means:
- ❌ Each wallet only sees their own boosts
- ❌ Boosts are lost on page refresh or different browsers
- ❌ No real-time synchronization across users

This guide explains how to add backend storage and synchronization for Nitro Points.

## What Was Already Added

### 1. Backend Handler Functions (`crates/q-api-server/src/handlers.rs`)

Two new endpoint handlers were added at the end of the file (lines 3412-3531):

```rust
/// Get all Nitro boosts for all tokens (aggregated by token_id)
pub async fn get_nitro_boosts(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<HashMap<String, u64>>>, StatusCode>

/// Add a Nitro boost to a token (costs user Nitro Points)
pub async fn add_nitro_boost(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AddNitroBoostRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode>
```

**What they do:**
- `get_nitro_boosts`: Returns aggregated boost points for all tokens `{token_id: total_points}`
- `add_nitro_boost`: Records a new boost (validates 50-500 point range) and broadcasts SSE event

### 2. Data Structures

```rust
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct NitroBoost {
    pub token_id: String,
    pub points: u64,
    pub wallet_address: String,
    pub timestamp: u64,
}

#[derive(Debug, Deserialize)]
pub struct AddNitroBoostRequest {
    pub token_id: String,
    pub points: u64,
    pub wallet_address: String,
}
```

## What Still Needs To Be Done

### Step 1: Add RocksDB Column Family

**File**: `crates/q-api-server/src/lib.rs`

Find where database is initialized (search for `DB::open` or column family creation) and add:

```rust
// Add to column families list
let cf_names = vec![
    "default",
    "wallets",
    "transactions",
    "blocks",
    "nitro_boosts",  // <-- ADD THIS
];
```

### Step 2: Wire Up Routes in main.rs

**File**: `crates/q-api-server/src/main.rs`

Add these routes after the DEX routes (around line 1365):

```rust
// Nitro Points / Token Boosting System
.route("/api/v1/nitro/boosts", get(handlers::get_nitro_boosts))
.route("/api/v1/nitro/boost", post(handlers::add_nitro_boost))
```

### Step 3: Update Frontend API Service

**File**: `gui/quantum-wallet/src/services/api.ts`

Add these methods to the `qnkAPI` object:

```typescript
// Get all Nitro boosts (aggregated by token)
async getNitroBoosts(): Promise<ApiResponse<Record<string, number>>> {
  return this.get('/api/v1/nitro/boosts');
},

// Add Nitro boost to a token
async addNitroBoost(tokenId: string, points: number, walletAddress: string): Promise<ApiResponse<any>> {
  return this.post('/api/v1/nitro/boost', {
    token_id: tokenId,
    points,
    wallet_address: walletAddress
  });
}
```

### Step 4: Update DexScreen.tsx Frontend Logic

**File**: `gui/quantum-wallet/src/components/DexScreen.tsx`

Replace localStorage usage with API calls:

#### A. Load boosts from API on mount (replace lines 62-77):

```typescript
// Load Nitro boosts from backend instead of localStorage
useEffect(() => {
  const fetchNitroBoosts = async () => {
    try {
      const response = await qnkAPI.getNitroBoosts();
      if (response.success && response.data) {
        // Convert Record<string, number> to Map
        const boostMap = new Map(Object.entries(response.data));
        setBoostedTokens(boostMap);
        console.log('✅ Loaded Nitro boosts from backend:', response.data);
      }
    } catch (error) {
      console.error('Failed to fetch Nitro boosts:', error);
    }
  };

  fetchNitroBoosts();
  // Refresh every 10 seconds for near-real-time updates
  const interval = setInterval(fetchNitroBoosts, 10000);
  return () => clearInterval(interval);
}, []);
```

#### B. Post boosts to API when confirmed (replace lines 491-534):

```typescript
const confirmNitroBoost = async () => {
  if (!nitroBoostToken) return;

  // Check if user has enough points
  if (nitroPoints < boostCost) {
    alert(`❌ Insufficient Nitro Points!\n\nYou need ${boostCost} points but only have ${nitroPoints} points.\n\nClick on the Nitro Points display in the topbar to purchase more points.`);
    return;
  }

  // Get wallet address
  const walletAddress = localStorage.getItem('walletAddress');
  if (!walletAddress) {
    alert('❌ Please connect your wallet first');
    return;
  }

  try {
    // Post boost to backend
    const response = await qnkAPI.addNitroBoost(
      nitroBoostToken.id,
      boostCost,
      walletAddress
    );

    if (!response.success) {
      alert(`❌ Failed to add Nitro boost: ${response.error}`);
      return;
    }

    // Deduct points from local state
    const newPoints = nitroPoints - boostCost;
    setNitroPoints(newPoints);
    localStorage.setItem('nitroPoints', newPoints.toString());

    // Update local boosted tokens map
    const newBoosted = new Map(boostedTokens);
    const existingPoints = newBoosted.get(nitroBoostToken.id) || 0;
    newBoosted.set(nitroBoostToken.id, existingPoints + boostCost);
    setBoostedTokens(newBoosted);

    // Show visual effect
    setNitroBoostTokens(prev => {
      const newSet = new Set(prev);
      newSet.add(nitroBoostToken.id);
      return newSet;
    });

    setTimeout(() => {
      setNitroBoostTokens(prev => {
        const newSet = new Set(prev);
        newSet.delete(nitroBoostToken.id);
        return newSet;
      });
    }, 2000);

    setNitroBoostToken(null);
    setBoostCost(100);
    alert(`🚀 Nitro Boost activated!\n\nSpent ${boostCost} points on ${nitroBoostToken.symbol}\n\nRemaining Points: ${newPoints}\nTotal Boost on ${nitroBoostToken.symbol}: ${existingPoints + boostCost} points`);

  } catch (error) {
    console.error('Failed to add Nitro boost:', error);
    alert('❌ Failed to add Nitro boost. Please try again.');
  }
};
```

### Step 5: Build and Test

```bash
# Rebuild backend
cd /opt/orobit/shared/q-narwhalknight
timeout 36000 cargo build --release --package q-api-server

# Rebuild frontend
cd gui/quantum-wallet
npm run build

# Restart server
killall q-api-server
Q_DB_PATH=./data ./target/release/q-api-server --port 8080
```

### Step 6: Test Cross-Wallet Synchronization

1. Open wallet A in one browser
2. Add Nitro boost to a token (spend points)
3. Open wallet B in another browser/incognito
4. Verify wallet B sees the boost on the token list
5. Add more boosts from wallet B
6. Verify both wallets see the combined boosts

## How It Works

### Data Flow:

```
User clicks "Nitro Boost" in wallet A
    ↓
Frontend calls POST /api/v1/nitro/boost
    ↓
Backend stores in RocksDB nitro_boosts column family
    ↓
Backend broadcasts SSE event (optional, for instant updates)
    ↓
All connected wallets poll GET /api/v1/nitro/boosts every 10s
    ↓
Wallets update their boost display in real-time
```

### Storage Format:

**RocksDB Key**: `wallet_address:token_id:timestamp`
Example: `0x123abc:native-qug:1729000000`

**RocksDB Value**: JSON-serialized NitroBoost object
```json
{
  "token_id": "native-qug",
  "points": 100,
  "wallet_address": "0x123abc...",
  "timestamp": 1729000000
}
```

### Aggregation:

When `GET /api/v1/nitro/boosts` is called, the backend:
1. Iterates through all nitro_boosts entries
2. Sums points by token_id
3. Returns: `{"native-qug": 450, "qugusd-stable": 200}`

## Benefits of This Approach

✅ **Shared Visibility**: All wallets see all boosts
✅ **Persistence**: Boosts survive browser restarts
✅ **Real-time**: 10-second polling for near-instant updates
✅ **Blockchain Storage**: Boosts stored in RocksDB alongside transactions
✅ **Scalable**: Can handle thousands of boosts efficiently

## Optional Enhancement: SSE Real-Time Updates

For instant updates without polling, implement SSE listener in DexScreen.tsx:

```typescript
useEffect(() => {
  // Connect to SSE stream
  const eventSource = new EventSource('http://localhost:8080/api/v1/events');

  eventSource.addEventListener('message', (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'nitro_boost') {
        // Update boosted tokens instantly
        const newBoosted = new Map(boostedTokens);
        const existing = newBoosted.get(data.token_id) || 0;
        newBoosted.set(data.token_id, existing + data.points);
        setBoostedTokens(newBoosted);

        console.log(`⚡ Real-time Nitro boost: ${data.points} points on ${data.token_id}`);
      }
    } catch (error) {
      console.error('Failed to parse SSE event:', error);
    }
  });

  return () => eventSource.close();
}, []);
```

This would give instant updates (< 100ms) instead of 10-second polling.

## Files Modified

- ✅ `crates/q-api-server/src/handlers.rs` - Added handler functions
- 🔨 `crates/q-api-server/src/lib.rs` - Need to add nitro_boosts column family
- 🔨 `crates/q-api-server/src/main.rs` - Need to add routes
- 🔨 `gui/quantum-wallet/src/services/api.ts` - Need to add API methods
- 🔨 `gui/quantum-wallet/src/components/DexScreen.tsx` - Need to replace localStorage with API calls

## Completion Checklist

- [x] Create backend handler functions
- [ ] Add nitro_boosts column family to RocksDB
- [ ] Wire up routes in main.rs
- [ ] Add frontend API methods
- [ ] Update DexScreen.tsx to use API instead of localStorage
- [ ] Build backend and frontend
- [ ] Test cross-wallet synchronization
- [ ] (Optional) Add SSE real-time updates

---

**Implementation Status**: Backend handlers added, database and routing changes pending.
