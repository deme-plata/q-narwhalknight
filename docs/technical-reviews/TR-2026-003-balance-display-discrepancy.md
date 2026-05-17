# Technical Review TR-2026-003: Balance Display Discrepancy

**Date**: 2026-03-10
**Severity**: HIGH (User-facing balance inconsistency)
**Affected Components**: Dashboard.tsx, TransactionScreenV2.tsx, App.tsx
**Reported By**: User observation + miner bug report (Liss)
**Status**: FIXED and deployed (v9.6.1 frontend, 2026-03-10)
**Peer Review**: Approved with enhancements (incorporated below)

---

## 1. Problem Statement

The Dashboard page displays **151 QUG** for a wallet while the Transaction page displays **139 QUG** for the same wallet on the same server (Beta). The backend balance is confirmed correct at approximately **139.35 QUG** (Beta) and **139.47 QUG** (Epsilon).

This discrepancy:
- Undermines user trust in the system
- Was independently reported by miner "Liss" as "balance bouncing" behavior
- Persists across page refreshes due to localStorage contamination

---

## 2. Architecture Overview

### 2.1 Balance Data Flow

```
                    ┌──────────────────┐
                    │  Backend Server   │
                    │                   │
                    │  HashMap (live)   │──── API Response ──────┐
                    │  RocksDB (durable)│──── (15s sync lag) ──┐ │
                    │  SSE Stream       │──── balance-updated ─┤ │
                    └──────────────────┘                       │ │
                                                               │ │
                    ┌──────────────────┐                       │ │
                    │     App.tsx       │◄──────────────────────┘ │
                    │                   │                         │
                    │  nodeData.balance │◄── SSE balance-updated  │
                    │  localStorage     │◄── safeCacheBalance()   │
                    │                   │                         │
                    │  Passes as prop:  │                         │
                    │  ├─ liveBalance ──┼──► Dashboard.tsx        │
                    │  └─ currentBalance┼──► TransactionScreenV2  │
                    └──────────────────┘                         │
                                                                 │
     ┌─────────────────────────────┐    ┌────────────────────────┘
     │       Dashboard.tsx         │    │  TransactionScreenV2.tsx
     │                             │    │
     │  walletBalances[] ◄─── API ─┼────┘
     │  walletBalances[] ◄─── SSE  │    │  walletBalances[] ◄── API
     │  walletBalances[] ◄─── prop │    │  stableBalance ◄── Math.max()
     │                             │    │
     │  ⚠️ MERGE LOGIC (line 963) │    │  ⚠️ HIGH-WATER MARK (line 659)
     │  Keeps higher of SSE vs API │    │  Uses max(prev, cached, new)
     │                             │    │
     │  DISPLAYS: walletBalances   │    │  DISPLAYS: stableBalance
     │  = 151 QUG (STALE)         │    │  = 139 QUG (CORRECT)
     └─────────────────────────────┘    └─────────────────────────
```

### 2.2 Balance Sources (per component)

| Source | Description | Update Frequency |
|--------|-------------|------------------|
| **API /wallet/balance** | Reads from in-memory HashMap | On page load, every 30s |
| **SSE balance-updated** | Pushed on every block reward | Real-time (~1/sec during mining) |
| **localStorage cachedBalance** | Persisted by `safeCacheBalance()` | On every balance update |
| **liveBalance prop** | App.tsx → Dashboard via React prop | On every SSE event |
| **currentBalance prop** | App.tsx → TransactionScreenV2 via prop | On every SSE event |

---

## 3. Root Cause Analysis

### 3.1 PRIMARY: Dashboard Merge Logic Prevents Downward Correction

**File**: `gui/quantum-wallet/src/components/Dashboard.tsx` lines 963-976

```typescript
// v8.6.2: Merge instead of replace — preserve higher QUG balance from SSE
// to prevent flicker when API response races with SSE updates.
setWalletBalances(prevWallets => {
  return balances.map(newWallet => {
    if (newWallet.symbol === 'QUG') {
      const prevQug = prevWallets.find(w => w.symbol === 'QUG');
      if (prevQug && prevQug.balance > newWallet.balance) {
        // SSE already gave us a higher balance — keep it, just update history
        return { ...newWallet, balance: prevQug.balance, history: ... };
      }
    }
    return newWallet;
  });
});
```

**Problem**: When the API returns the correct balance (139 QUG), this code compares it against `prevQug.balance` (151 QUG from a prior SSE event). Since 151 > 139, it **discards the correct API value** and **keeps the stale 151**.

**This logic was added in v8.6.2** to prevent "flicker" during SSE/API races, but it creates a one-way ratchet: balances can only go UP in the Dashboard, never down.

### 3.2 SECONDARY: TransactionScreenV2 High-Water Mark

**File**: `gui/quantum-wallet/src/components/TransactionScreenV2.tsx` lines 652-659

```typescript
const candidates = [
  isValidBalance(previousHighest, wallet.symbol) ? previousHighest : 0,
  cachedValue,
  isValidBalance(wallet.balance, wallet.symbol) ? wallet.balance : 0
].filter(v => v >= 0);
const validBalance = candidates.length > 0 ? Math.max(...candidates) : 0;
```

**Same pattern**: Uses `Math.max()` across all known values. However, TransactionScreenV2 currently shows 139 because:
1. Its `previousHighest` ref resets on component remount
2. `cachedBalance` in localStorage was updated to 139 by App.tsx
3. The `stableBalance` state initialized from `currentBalance` prop (139)

**Risk**: If the user navigated to Transaction page while the stale 151 was still in localStorage, it would show 151 too.

### 3.3 TERTIARY: liveBalance Prop Sync (Dashboard line 1973-1991)

```typescript
// v8.6.5: Sync liveBalance prop into walletBalances
useEffect(() => {
  if (liveBalance === undefined || liveBalance === null) return;
  setWalletBalances(wallets => {
    const qugWallet = wallets.find(w => w.symbol === 'QUG');
    if (!qugWallet) return wallets;
    if (Math.abs(qugWallet.balance - liveBalance) < 1e-12) return wallets;
    return wallets.map(wallet =>
      wallet.symbol === 'QUG' ? { ...wallet, balance: liveBalance } : wallet
    );
  });
}, [liveBalance]);
```

**Problem**: This effect does NOT have the same one-way ratchet — it will update to whatever `liveBalance` is. However, `liveBalance` comes from App.tsx's `nodeData.balance`, which applies SSE balance-updated events that can be HIGHER than actual (due to pending rewards not yet confirmed). Once a high SSE value is written to `walletBalances`, the merge logic at line 963 locks it in.

**Post-fix behavior** *(per peer review suggestion #2)*: With the ratchet removed, if SSE emits a briefly inflated value, the Dashboard will show it temporarily. This spike is expected to be **short-lived** — within the 15-second HashMap→RocksDB sync window, the next API fetch (every 30s) will correct the display to the authoritative backend value. This transient spike is acceptable and far less harmful than a permanently stuck stale balance.

### 3.4 TERTIARY: localStorage Cache Contamination

**File**: `App.tsx` line 238

```typescript
safeCacheBalance(pendingBalanceUpdate);
```

When an SSE event delivers a temporarily inflated balance, `safeCacheBalance` writes it to localStorage. On next page load, all components initialize from this stale cache value.

**Why SSE can emit a temporarily inflated balance** *(per peer review suggestion #1)*: Mining rewards are written to RocksDB immediately during block processing (`balance_consensus.rs` line 322-373), and SSE events are emitted at the same time. However, the in-memory HashMap used by the API endpoint is only synced from RocksDB every 15 seconds (`main.rs` ~line 18972). So within that 15-second window, SSE may report a new balance that the API cannot yet confirm. If a block is later orphaned or reorged, the SSE value becomes stale while the API eventually returns the corrected balance. This is the core reason the API must be trusted as the eventual source of truth.

### 3.5 Backend: HashMap vs RocksDB 15-Second Lag

**File**: `crates/q-api-server/src/main.rs` (~line 18972-19168)

The in-memory HashMap and RocksDB are synced every 15 seconds. During this window:
- API endpoint reads from HashMap (may have stale data)
- SSE events emit from block processing (most current)
- RocksDB has durable data (most authoritative)

This creates a race condition where different API calls can return different values within the 15-second window. The key insight is that `balance_consensus.rs` writes mining rewards **only to RocksDB** (not the in-memory HashMap), so the HashMap lags behind the durable store. SSE events emit from the block processing path (which has the newest data), while the API reads from the HashMap (which is up to 15 seconds stale). This asymmetry is the fundamental reason SSE can report values the API hasn't caught up to yet — and why removing the one-way ratchet is safe: the API will eventually converge to the correct value.

---

## 4. Reproduction Steps

1. Start mining on any wallet
2. Accumulate balance over time (e.g., reach 139 QUG)
3. Observe Dashboard — may show 151 QUG if an SSE event once pushed a higher value
4. Navigate to Transaction page — shows 139 QUG (correct)
5. Navigate back to Dashboard — still shows 151 QUG
6. Hard refresh the page — Dashboard may STILL show 151 if localStorage cached the stale value

---

## 5. Impact Assessment

| Aspect | Severity | Details |
|--------|----------|---------|
| **User Trust** | HIGH | Users see different balances on different pages |
| **Financial Accuracy** | MEDIUM | No actual fund loss; display-only bug |
| **Mining Confidence** | HIGH | Miners question if rewards are being credited correctly |
| **Consensus Safety** | NONE | Backend balances are correct; this is purely frontend |
| **DEX Operations** | LOW | DEX cooldown guards prevent stale balance during swaps |

---

## 6. Proposed Fix

### 6.1 Remove One-Way Ratchet from Dashboard (CRITICAL)

**File**: `Dashboard.tsx` line 963-976

Replace the merge logic with a direct assignment that trusts the API response:

```typescript
// FIX: Trust API response as authoritative. If SSE gave a higher balance,
// it will arrive again via liveBalance prop or next SSE event.
// The API is the source of truth after the 15-second sync window.
if (!dexSwapCooldownRef.current && !globalCooldownActive) {
  setWalletBalances(balances);
}
```

**Rationale**: The original "flicker prevention" is better solved by the existing `liveBalance` prop sync (line 1973-1991) which updates in real-time from SSE. The API fetch every 30s should be allowed to correct stale values.

### 6.2 Replace Math.max() High-Water Mark in TransactionScreenV2

**File**: `TransactionScreenV2.tsx` lines 652-659

```typescript
// FIX: Use API balance directly, only fall back to cache if API returns 0
const validBalance = isValidBalance(wallet.balance, wallet.symbol) && wallet.balance > 0
  ? wallet.balance
  : (cachedValue > 0 ? cachedValue : previousHighest);
```

**Rationale**: The `Math.max()` pattern assumes balance can only go up, which is incorrect — users spend coins, DEX swaps reduce balances, and backend corrections are valid.

### 6.3 Add localStorage TTL

**File**: `App.tsx` `safeCacheBalance()` function

```typescript
function safeCacheBalance(balance: number) {
  if (isValidBalance(balance)) {
    localStorage.setItem('cachedBalance', balance.toString());
    localStorage.setItem('cachedBalanceTimestamp', Date.now().toString());
  }
}

// On read:
function getCachedBalance(): number {
  const cached = localStorage.getItem('cachedBalance');
  const timestamp = localStorage.getItem('cachedBalanceTimestamp');
  if (!cached || !timestamp) return 0;
  // Expire cache after 5 minutes
  if (Date.now() - parseInt(timestamp) > 300_000) {
    localStorage.removeItem('cachedBalance');
    localStorage.removeItem('cachedBalanceTimestamp');
    return 0;
  }
  return parseFloat(cached);
}
```

### 6.4 Remove Dead SSE Code from Dashboard

**File**: `Dashboard.tsx` lines 1248-1620

This code is unreachable because line 1232 contains an early `return` statement with the comment "using App.tsx SSE connection instead". The SSE setup logic that follows (creating EventSource, subscribing to balance/block events) can never execute. This dead code adds ~370 lines of confusion — developers reading Dashboard.tsx may incorrectly believe it has its own SSE connection when in fact all SSE is centralized in App.tsx. Removing it reduces confusion and code size.

---

## 7. Testing Plan

| Test | Method | Expected Result |
|------|--------|-----------------|
| Balance decreases display correctly | Send QUG from wallet, verify Dashboard updates | Dashboard shows new lower balance |
| Mining rewards still display instantly | Mine blocks, verify Dashboard updates in real-time | Balance increments appear via SSE within 1s |
| DEX swap cooldown still works | Execute DEX swap, verify no stale overwrites | Balance holds during cooldown period |
| Page refresh shows correct balance | Refresh after balance decrease | localStorage has correct value |
| Cross-page consistency | Compare Dashboard vs Transaction page | Both show same balance |
| localStorage expiry | Wait 5+ minutes without SSE, refresh | Falls back to API fetch, not stale cache |

---

## 8. Risk Assessment for Fix

| Risk | Mitigation |
|------|------------|
| Balance flicker during SSE/API race | liveBalance prop sync (line 1973) handles real-time; 30s API fetch is slow enough to not race |
| Brief zero-balance flash on page load | Keep localStorage cache initialization, just add TTL |
| DEX balance regression | DEX cooldown guards (lines 958-961) remain unchanged |
| Transient SSE spike displayed | Spike is short-lived (< 15s); next API fetch corrects it. Far less harmful than permanent stale value |
| API failure retains stale balance | If API is unreachable, Dashboard retains last known state from React state (not localStorage). On recovery, next successful fetch corrects it. If the stale value was inflated, it persists only until API recovers — an acceptable trade-off vs the previous permanent ratchet |

---

## 9. Post-Deployment Verification

*(Added per peer review suggestion #3)*

After deployment, verify the fix is working:

1. **Immediate**: Hard refresh (Ctrl+F5) on Beta subdomain and Epsilon top domain. Both Dashboard and Transaction page should show the same balance (within ~0.15 QUG cross-server variance due to block propagation timing).
2. **24-hour soak**: Monitor for any user reports of balance bouncing or stale display. Check browser console for `[fetchWalletBalances]` logs — the `SKIPPING` message should only appear during active DEX swaps, not during normal mining.
3. **Cross-page consistency**: Navigate Dashboard → Transaction → Dashboard multiple times. Balance should remain consistent across transitions.
4. **Mining reward flow**: While mining, verify that new rewards appear on Dashboard within ~1 second (via SSE liveBalance prop), and that the 30-second API fetch does not revert the balance to a lower value.

---

## 10. Communication Plan

*(Added per peer review suggestion #4)*

- **Miner "Liss"**: Direct Discord message explaining the fix (balance bouncing resolved, hard refresh required). Already prepared.
- **Release note**: v9.6.1 changelog should include: "Fixed: Dashboard could display inflated balance that didn't match Transaction page. Both pages now show authoritative backend balance."
- **General miners**: If additional reports come in, instruct users to hard refresh (Ctrl+F5) to clear cached frontend assets.

---

## 11. Timeline

- **Severity**: HIGH — directly affects user trust
- **Complexity**: LOW — localized to 3 frontend files, no backend changes
- **Consensus impact**: ZERO — purely frontend display logic
- **Fix deployed**: 2026-03-10 (v9.6.1 frontend build, pushed to Epsilon)

---

## 12. Appendix: Code References

> **Note**: Line numbers were verified against the codebase on 2026-03-10 (branch `feature/safe-batched-sync-v1.0.2`). If the codebase has evolved since this review, verify line numbers against your current branch before applying changes.

| File | Line(s) | Description |
|------|---------|-------------|
| `Dashboard.tsx` | 963-976 | **PRIMARY BUG**: Merge logic keeps stale higher balance |
| `Dashboard.tsx` | 584-603 | Anti-drop logic with 1% threshold |
| `Dashboard.tsx` | 1973-1991 | liveBalance prop sync (correct behavior) |
| `Dashboard.tsx` | 1232 | Early return disabling local SSE |
| `Dashboard.tsx` | 1248-1620 | Dead/unreachable SSE code |
| `TransactionScreenV2.tsx` | 162-176 | stableBalance init from localStorage |
| `TransactionScreenV2.tsx` | 199-200 | Math.max() high-water mark |
| `TransactionScreenV2.tsx` | 652-659 | Math.max() in balance validation |
| `App.tsx` | 149-173 | nodeData.balance init from localStorage |
| `App.tsx` | 215-270 | pendingBalanceUpdate effect (instant up, 50ms debounce down) |
| `App.tsx` | 527-554 | SSE balance-updated handler |
| `handlers.rs` | 693-707 | Backend: Dashboard reads from in-memory HashMap |
| `handlers.rs` | 6492-6511 | Backend: Wallet balance — HashMap then RocksDB fallback |
| `balance_consensus.rs` | 322-373 | Backend: Mining reward writes to RocksDB only (NOT HashMap — this is why API lags behind SSE) |
| `main.rs` | ~18972-19168 | Backend: 15-second HashMap ↔ RocksDB sync |

---

*Review prepared for AI peer review. Peer review received and incorporated 2026-03-10. All line numbers verified against branch `feature/safe-batched-sync-v1.0.2` as of 2026-03-10. Fix deployed same day.*
