# Multi-Wallet + CDP Activity Implementation

## Overview

This document tracks the implementation of:
1. CDP mint transactions appearing in Recent Activity
2. Multi-wallet support on dashboard (QUG, Bitcoin, Ethereum, etc.)
3. Atomic swap preparation infrastructure

## Part 1: CDP Transactions in Recent Activity ✅ COMPLETE

### Frontend Changes

**File**: `gui/quantum-wallet/src/components/MintQUGUSDModal.tsx`

**Status**: ✅ Complete - Built and deployed

Added event dispatch on successful mint (line 75-86):
```typescript
// Dispatch CDP mint event for Dashboard to catch
const currentWalletAddress = localStorage.getItem('walletAddress');
window.dispatchEvent(new CustomEvent('cdp-mint', {
  detail: {
    transaction_id: response.data.transaction_id,
    collateral_amount: parseFloat(collateralAmount),
    minted_amount: parseFloat(qugusdAmount),
    collateral_ratio: response.data.collateral_ratio,
    wallet_address: currentWalletAddress,
    timestamp: new Date().toISOString(),
  }
}));
```

**File**: `gui/quantum-wallet/src/components/Dashboard.tsx`

**Status**: ✅ Complete - Built and deployed

Added CDP event listener in the useEffect (lines 529-563):

```typescript
// Listen for CDP mint events
const handleCDPMint = (event: Event) => {
  const customEvent = event as CustomEvent;
  const data = customEvent.detail;

  console.log('💵 CDP Mint Event:', data);

  // Create transaction for Recent Activity
  const cdpTransaction: Transaction = {
    id: `cdp-mint-${data.transaction_id}`,
    type: 'send', // Sending QUG to CDP vault
    amount: data.collateral_amount,
    from: data.wallet_address,
    to: 'CDP Vault (QUGUSD Minting)',
    timestamp: data.timestamp,
    txHash: data.transaction_id,
  };

  // Add to recent transactions
  setRecentTransactions(prev => [cdpTransaction, ...prev]);

  console.log('💵 CDP transaction added to recent activity');
};

window.addEventListener('cdp-mint', handleCDPMint);

// Cleanup
return () => {
  mounted = false;
  if (eventSource) {
    eventSource.close();
  }
  window.removeEventListener('cdp-mint', handleCDPMint);
};
```

## Part 2: Multi-Wallet Dashboard

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Wallet Dashboard                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   QUG Wallet │  │ BTC Wallet   │  │  ETH Wallet  │      │
│  │              │  │              │  │              │      │
│  │ Balance:     │  │ Balance:     │  │ Balance:     │      │
│  │ 1,234.56 QUG │  │ 0.0523 BTC   │  │ 2.456 ETH    │      │
│  │              │  │              │  │              │      │
│  │ [Send] [Rec] │  │ [Send] [Rec] │  │ [Send] [Rec] │      │
│  │ [Swap] [DEX] │  │ [Swap]       │  │ [Swap]       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  USDT Wallet │  │ QUGUSD CDP   │  │  + Add Wallet│      │
│  │              │  │              │  │              │      │
│  │ Balance:     │  │ Minted:      │  │  [Bitcoin]   │      │
│  │ 5,000 USDT   │  │ 265.63 USD   │  │  [Ethereum]  │      │
│  │              │  │ Collateral:  │  │  [Litecoin]  │      │
│  │ [Send] [Rec] │  │ 10 QUG       │  │  [Custom]    │      │
│  │ [Swap]       │  │ [Mint] [Burn]│  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Wallet Types Interface

**File**: `gui/quantum-wallet/src/types/wallets.ts` (NEW)

```typescript
export enum WalletType {
  QUG = 'QUG',
  BITCOIN = 'BTC',
  ETHEREUM = 'ETH',
  LITECOIN = 'LTC',
  USDT = 'USDT',
  QUGUSD_CDP = 'QUGUSD_CDP',
  CUSTOM = 'CUSTOM',
}

export interface WalletConfig {
  type: WalletType;
  name: string;
  symbol: string;
  address: string;
  balance: number;
  usdValue: number;
  icon: string; // Icon component name or URL
  color: string; // Gradient color for card
  actions: WalletAction[];
  network?: string; // e.g., "mainnet", "testnet"
  derivationPath?: string; // HD wallet path
}

export interface WalletAction {
  type: 'send' | 'receive' | 'swap' | 'mint' | 'burn' | 'dex' | 'stake';
  label: string;
  enabled: boolean;
  handler: () => void;
}

export interface AtomicSwapPair {
  fromWallet: WalletType;
  toWallet: WalletType;
  enabled: boolean;
  fee: number; // Percentage
  minAmount: number;
  maxAmount: number;
}
```

### Wallet Card Component

**File**: `gui/quantum-wallet/src/components/WalletCard.tsx` (NEW)

```typescript
import { motion } from 'framer-motion';
import { Send, Download, ArrowLeftRight, TrendingUp } from 'lucide-react';
import { WalletConfig } from '../types/wallets';

interface WalletCardProps {
  wallet: WalletConfig;
  onAction: (action: string) => void;
}

export function WalletCard({ wallet, onAction }: WalletCardProps) {
  return (
    <motion.div
      className="backdrop-blur-xl rounded-2xl p-6 relative overflow-hidden"
      style={{
        background: `linear-gradient(135deg, ${wallet.color}15, ${wallet.color}05)`,
        border: `2px solid ${wallet.color}30`,
        boxShadow: `0 0 20px ${wallet.color}20`,
      }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4, boxShadow: `0 8px 30px ${wallet.color}30` }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className={`w-12 h-12 rounded-xl flex items-center justify-center`}
            style={{ background: `${wallet.color}20` }}
          >
            <span className="text-2xl">{wallet.icon}</span>
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">{wallet.name}</h3>
            <p className="text-xs text-gray-400">{wallet.symbol}</p>
          </div>
        </div>
        <div className="text-right">
          <div className="text-xs text-gray-400">Balance</div>
          <div className="text-sm font-bold text-white">
            {wallet.balance.toFixed(4)} {wallet.symbol}
          </div>
        </div>
      </div>

      {/* Address */}
      <div className="mb-4 p-3 rounded-lg bg-black/30">
        <div className="text-xs text-gray-400 mb-1">Address</div>
        <div className="font-mono text-xs text-white truncate">
          {wallet.address}
        </div>
      </div>

      {/* USD Value */}
      <div className="mb-4">
        <div className="text-2xl font-bold text-white">
          ${wallet.usdValue.toLocaleString('en-US', { minimumFractionDigits: 2 })}
        </div>
        <div className="text-xs text-gray-400">USD Value</div>
      </div>

      {/* Actions */}
      <div className="grid grid-cols-2 gap-2">
        {wallet.actions.map((action) => (
          <motion.button
            key={action.type}
            onClick={() => action.enabled && onAction(action.type)}
            disabled={!action.enabled}
            className="p-3 rounded-lg font-medium text-sm flex items-center justify-center gap-2 disabled:opacity-30 disabled:cursor-not-allowed"
            style={{
              background: action.enabled ? `${wallet.color}20` : 'rgba(100,100,100,0.1)',
              border: `1px solid ${wallet.color}30`,
              color: 'white',
            }}
            whileHover={action.enabled ? { scale: 1.05 } : {}}
            whileTap={action.enabled ? { scale: 0.95 } : {}}
          >
            {getActionIcon(action.type)}
            {action.label}
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
}

function getActionIcon(type: string) {
  switch (type) {
    case 'send': return <Send className="w-4 h-4" />;
    case 'receive': return <Download className="w-4 h-4" />;
    case 'swap': return <ArrowLeftRight className="w-4 h-4" />;
    case 'dex': return <TrendingUp className="w-4 h-4" />;
    default: return null;
  }
}
```

### Updated Dashboard Integration

**File**: `gui/quantum-wallet/src/components/Dashboard.tsx`

Add wallet cards section before Recent Activity (around line 835):

```typescript
{/* Multi-Wallet Section */}
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
  {wallets.map((wallet) => (
    <WalletCard
      key={wallet.type}
      wallet={wallet}
      onAction={(action) => handleWalletAction(wallet.type, action)}
    />
  ))}

  {/* Add Wallet Card */}
  <motion.div
    className="backdrop-blur-xl rounded-2xl p-6 border-2 border-dashed border-gray-600 flex flex-col items-center justify-center cursor-pointer hover:border-amber-500/50 transition-colors"
    onClick={() => setShowAddWalletModal(true)}
    whileHover={{ scale: 1.02 }}
    whileTap={{ scale: 0.98 }}
  >
    <div className="w-16 h-16 rounded-full bg-gray-800 flex items-center justify-center mb-4">
      <span className="text-4xl text-gray-500">+</span>
    </div>
    <h3 className="text-lg font-bold text-gray-400">Add Wallet</h3>
    <p className="text-xs text-gray-500 mt-1">Bitcoin, Ethereum, etc.</p>
  </motion.div>
</div>
```

## Part 3: Atomic Swap Infrastructure

### Atomic Swap Contract Interface

**File**: `gui/quantum-wallet/src/types/atomicSwap.ts` (NEW)

```typescript
export interface AtomicSwapContract {
  id: string;
  fromChain: string;
  toChain: string;
  fromAmount: number;
  toAmount: number;
  fromAddress: string;
  toAddress: string;
  hashlock: string; // SHA256 hash
  timelock: number; // Unix timestamp
  status: 'initiated' | 'locked' | 'claimed' | 'refunded' | 'expired';
  createdAt: string;
  expiresAt: string;
}

export interface SwapQuote {
  fromSymbol: string;
  toSymbol: string;
  fromAmount: number;
  toAmount: number;
  exchangeRate: number;
  fee: number;
  estimatedTime: number; // seconds
  expiresIn: number; // seconds until quote expires
}
```

### Backend API Endpoints Needed

**File**: `crates/q-api-server/src/atomic_swap_api.rs` (NEW - TODO)

```rust
/// Atomic swap endpoints for cross-chain trading
pub fn create_atomic_swap_router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/initiate", post(initiate_swap))
        .route("/lock", post(lock_funds))
        .route("/claim", post(claim_swap))
        .route("/refund", post(refund_swap))
        .route("/status/:swap_id", get(get_swap_status))
        .route("/quote", post(get_swap_quote))
}

#[derive(Deserialize)]
struct InitiateSwapRequest {
    from_chain: String,      // "QUG", "BTC", "ETH"
    to_chain: String,
    from_amount: f64,
    to_amount: f64,
    recipient_address: String,
    hashlock: String,        // SHA256 hash of secret
    timelock: u64,          // Unix timestamp
}

#[derive(Serialize)]
struct InitiateSwapResponse {
    swap_id: String,
    contract_address: String,
    hashlock: String,
    timelock: u64,
    status: String,
}
```

## Implementation Tasks

### Phase 1: CDP Activity ✅ COMPLETE
- [x] Update MintQUGUSDModal to dispatch event
- [x] Add CDP event listener to Dashboard
- [x] Remove page reload on mint success
- [x] Rebuild frontend with fixes
- [ ] User testing - verify CDP transactions appear in activity
- [ ] Backend integration - implement real balance deduction

### Phase 2: Multi-Wallet Dashboard
- [ ] Create wallet types and interfaces (`types/wallets.ts`)
- [ ] Create WalletCard component
- [ ] Update Dashboard to show wallet grid
- [ ] Add Bitcoin wallet integration
- [ ] Add Ethereum wallet integration
- [ ] Create "Add Wallet" modal

### Phase 3: Atomic Swap Preparation
- [ ] Create atomic swap types (`types/atomicSwap.ts`)
- [ ] Backend: Implement atomic swap API
- [ ] Frontend: Create swap quote modal
- [ ] Frontend: Create swap execution flow
- [ ] Add hashlock/timelock cryptography
- [ ] Integrate with Bitcoin/Ethereum nodes

## Testing Checklist

### CDP Activity
- [ ] Mint QUGUSD successfully
- [ ] Verify transaction appears in Recent Activity
- [ ] Check transaction shows correct details (amount, type, etc.)
- [ ] Verify event cleanup on unmount

### Multi-Wallet
- [ ] Display multiple wallet cards
- [ ] Show correct balances for each wallet
- [ ] Actions buttons work correctly
- [ ] Add new wallet flow works
- [ ] Wallets persist in localStorage

### Atomic Swap
- [ ] Get swap quote successfully
- [ ] Initiate swap creates contract
- [ ] Hashlock verification works
- [ ] Timelock expiration triggers refund
- [ ] Claim swap with correct secret

## Security Considerations

1. **Private Keys**: All private keys must be encrypted at rest
2. **Hashlock Secret**: Must be generated client-side with crypto.subtle
3. **Timelock**: Must allow sufficient time for both parties (typically 24-48 hours)
4. **Refund Path**: Always ensure refund is possible after timelock expires
5. **Rate Limiting**: Implement rate limits on swap initiation
6. **Cross-Chain Verification**: Verify transactions on both chains before claiming

## Next Steps

1. ✅ Complete CDP event dispatch
2. 🔄 Add CDP listener to Dashboard (in progress)
3. Create wallet types and WalletCard component
4. Design "Add Wallet" modal UI
5. Research Bitcoin/Ethereum wallet integration libraries
6. Design atomic swap contract specification

---

**Status**: Part 1 (CDP Activity) - 50% complete
**Next**: Complete Dashboard CDP listener integration
