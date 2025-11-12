# Dashboard → Transaction Navigation Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DASHBOARD SCREEN                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │  QUG Card    │  │ QUGUSD Card  │  │  USD Card    │             │
│  │              │  │              │  │              │             │
│  │ Balance:     │  │ Balance:     │  │ Balance:     │             │
│  │ 1000.00 QUG  │  │ 500.00 QUGUSD│  │ 100.00 USD   │             │
│  │              │  │              │  │              │             │
│  │  [Mini Graph]│  │  [Mini Graph]│  │  [Mini Graph]│             │
│  │              │  │              │  │              │             │
│  │  [ Send ]    │  │  [ Send ]    │  │ [Add][Send]  │             │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
│         │                 │                 │                      │
│         │ Click           │ Click           │ Click Send           │
│         ▼                 ▼                 ▼                      │
└─────────┼─────────────────┼─────────────────┼────────────────────┘
          │                 │                 │
          │                 │                 │
    ┌─────▼─────────────────▼─────────────────▼─────┐
    │          App.tsx handleCoinSendClick          │
    │  localStorage.setItem('selectedCoinForSend',  │
    │                    coinSymbol)                 │
    │  setCurrentScreen('transactions')             │
    └───────────────────┬───────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    TRANSACTION SCREEN V2                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │           SELECTED WALLET CARD (Pre-selected)                  │ │
│  │  ┌──┐                                                          │ │
│  │  │🪙│  Quillon Graph               Balance: 1000.00 QUG       │ │
│  │  └──┘  Sending from QUG wallet                                │ │
│  │                                                                │ │
│  │  Select Coin to Send: [QUG ▼] [QUGUSD] [USD]                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Recipient Address: [_____________________________] 📷        │ │
│  │                                                                │ │
│  │  Amount (QUG): [________]                                     │ │
│  │  Available: 1000.00 QUG        Fee: 0.00001 QUG              │ │
│  │                                                                │ │
│  │  Memo (Optional): [_____________________________]             │ │
│  │                                                                │ │
│  │  [ ] Quantum Privacy Mixer                                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │              [ Sign & Broadcast Transaction ]                  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Interaction Flow

```
┌──────────────┐
│  Dashboard   │
│  Component   │
└──────┬───────┘
       │
       │ 1. User clicks coin card/button
       │    onNavigateToSend(coinSymbol)
       │
       ▼
┌──────────────┐
│   App.tsx    │
│              │
│  - Stores    │
│    coin in   │
│    localStorage
│              │
│  - Navigates │
│    to txn    │
│    screen    │
└──────┬───────┘
       │
       │ 2. Screen changes
       │
       ▼
┌──────────────┐
│ TransactionV2│
│              │
│  - Reads     │◄─── localStorage.getItem('selectedCoinForSend')
│    selected  │
│    coin      │
│              │
│  - Fetches   │◄─── API: /api/v1/wallets/{address}/balance
│    balances  │◄─── API: /api/v1/multi-token/balance
│              │◄─── API: /api/v1/payment/balance
│  - Displays  │
│    wallet    │
│    card      │
│              │
│  - User can  │
│    switch    │
│    coins via │
│    dropdown  │
└──────────────┘
```

## State Management

```
┌─────────────────────────────────────────────────────────────────┐
│                     STATE FLOW                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Dashboard Click                                                │
│       │                                                          │
│       ▼                                                          │
│  localStorage.setItem('selectedCoinForSend', 'QUG')            │
│       │                                                          │
│       ▼                                                          │
│  App.tsx: setCurrentScreen('transactions')                      │
│       │                                                          │
│       ▼                                                          │
│  TransactionV2 Mounts                                           │
│       │                                                          │
│       ▼                                                          │
│  useState(() => {                                               │
│    const stored = localStorage.getItem('selectedCoinForSend')  │
│    localStorage.removeItem('selectedCoinForSend') ◄── Clear!   │
│    return stored || 'QUG'                                      │
│  })                                                             │
│       │                                                          │
│       ▼                                                          │
│  useEffect(() => {                                              │
│    fetchBalances() ◄── Fetch QUG, QUGUSD, USD                 │
│  }, [])                                                         │
│       │                                                          │
│       ▼                                                          │
│  setWalletBalances([...])                                       │
│       │                                                          │
│       ▼                                                          │
│  Render wallet card for selectedCoin                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow for Balance Validation

```
┌────────────────────────────────────────────────────────────────┐
│              BALANCE VALIDATION FLOW                            │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User enters amount                                            │
│       │                                                         │
│       ▼                                                         │
│  validateTransaction()                                          │
│       │                                                         │
│       ├─► Get selectedWallet from walletBalances array         │
│       │                                                         │
│       ├─► Check toAddress is valid                             │
│       │                                                         │
│       ├─► Check amount is valid number > 0                     │
│       │                                                         │
│       ├─► Calculate totalRequired = amount + fee               │
│       │                                                         │
│       └─► Compare selectedWallet.balance >= totalRequired      │
│                │                                                │
│                ├─► ✅ Valid: Enable Send button                │
│                │                                                │
│                └─► ❌ Invalid: Show error with coin symbol     │
│                     "Insufficient balance. Required: X.XX QUG" │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## User Experience Timeline

```
Time    Dashboard                  LocalStorage           TransactionV2
────────────────────────────────────────────────────────────────────────
  0s    [User viewing cards]       [Empty]               [Not rendered]
        
  1s    [User clicks QUG card]     
                    ↓
  1s    onNavigateToSend('QUG')   
                    ↓
  1s                               setItem('selected     
                                   CoinForSend', 'QUG')
                    ↓
  1s    Navigate to 'transactions'
                                                          
  2s                                                      [Component mounts]
                                                          
  2s                                                      getItem('selected
                                                          CoinForSend')
                                   ← returns 'QUG'
                                   
  2s                                                      removeItem('selected
                                                          CoinForSend')
                                   [Empty again]
                                   
  2s                                                      [Fetching balances...]
                                   
  3s                                                      [Balances loaded]
                                                          [Wallet card shows QUG]
                                                          [Form ready]
                                                          
  4s    [User can return]                               [User can send]
```

## Error Handling Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    ERROR SCENARIOS                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Scenario 1: No wallet address in localStorage                   │
│      → fetchBalances() early returns                             │
│      → walletBalances = []                                       │
│      → selectedWallet = undefined                                │
│      → Form hidden (conditional render on selectedWallet)        │
│                                                                   │
│  Scenario 2: API fetch fails                                     │
│      → catch block logs warning                                  │
│      → Balance shows 0 for that coin                             │
│      → User can still see form but validation will fail          │
│                                                                   │
│  Scenario 3: No coin pre-selected (direct navigation)            │
│      → selectedCoin defaults to 'QUG'                            │
│      → Normal flow continues                                     │
│                                                                   │
│  Scenario 4: Invalid coin symbol in localStorage                 │
│      → selectedCoin set to invalid value                         │
│      → selectedWallet = undefined (find returns nothing)         │
│      → User sees dropdown, can select valid coin                 │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

**Diagram Version**: 1.0  
**Last Updated**: 2025-11-05  
**Status**: Implementation Complete
