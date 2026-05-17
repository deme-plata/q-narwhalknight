# Quillon Mobile Wallet -- Technical Review Document

**Version**: 1.0.0
**Date**: 2026-03-24
**Status**: Draft for Peer Review
**Authors**: Server Beta Engineering
**Classification**: Internal -- Pre-Development Architecture Review

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Technology Stack & Justification](#2-technology-stack--justification)
3. [App Architecture](#3-app-architecture)
4. [Screen-by-Screen Specification](#4-screen-by-screen-specification)
5. [Security Architecture](#5-security-architecture)
6. [Networking & Real-Time Data](#6-networking--real-time-data)
7. [Performance Considerations](#7-performance-considerations)
8. [DEX Mobile UX](#8-dex-mobile-ux)
9. [Mining on Mobile](#9-mining-on-mobile)
10. [QR Code & Payment Flows](#10-qr-code--payment-flows)
11. [Offline & Background Behavior](#11-offline--background-behavior)
12. [Testing Strategy](#12-testing-strategy)
13. [Build & Distribution](#13-build--distribution)
14. [Migration & Compatibility](#14-migration--compatibility)
15. [Risk Assessment & Mitigations](#15-risk-assessment--mitigations)
16. [Development Timeline](#16-development-timeline-phased)
17. [Open Questions for Review](#17-open-questions-for-review)

---

## 1. Executive Summary

### What We Are Building

The Quillon Mobile Wallet is a native Android application for the Q-NarwhalKnight blockchain (QUG). It provides self-custodial wallet management, token transfers, DEX trading, mining pool monitoring, and point-of-sale payment acceptance from a mobile device. The application targets Expo SDK 55 with a managed workflow, producing a distributable APK/AAB for Google Play and a sideloadable APK from `quillon.xyz/downloads/`.

### Why Mobile

The existing Quillon wallet ecosystem consists of two clients:

1. **Web wallet** (`gui/quantum-wallet/`) -- React/TypeScript SPA served from `quillon.xyz`. Feature-rich (99+ components), but requires a desktop browser and persistent tab for SSE events.
2. **Desktop wallet** (`gui/slint-wallet/`) -- Rust/Slint native application with OAuth2 PKCE, Ed25519 signing, auto-updater, and PoS mode. Requires a desktop OS and manual binary downloads.

Neither client addresses the primary use case for retail cryptocurrency: mobile payments. Merchants accepting QUG at a register need a phone-native PoS flow. Users checking balances, approving swaps, or monitoring mining rewards need an app that survives Android lifecycle events, receives push notifications, and authenticates via biometrics in under one second.

### Target Audience

- **Retail users**: Check balance, send/receive QUG, swap tokens on the DEX.
- **Merchants**: Accept QUG payments via full-screen QR PoS mode with countdown timer.
- **Miners**: Monitor pool performance, hashrate, earnings, and remotely manage desktop miners.
- **Power users**: Access the full DEX (AMM swaps, liquidity provision, slippage control).

### Key Differentiators from Web/Desktop

| Capability | Web Wallet | Desktop (Slint) | Mobile (Proposed) |
|---|---|---|---|
| Biometric auth | No | No | Yes (fingerprint/face) |
| Push notifications | No (SSE only) | No | Yes (FCM) |
| QR scanning | No | No | Yes (camera) |
| Offline tx signing | No | Yes | Yes |
| Background sync | Tab must be open | Process must run | WorkManager task |
| PoS mode | No | Yes (basic) | Yes (full-screen, NFC-ready) |
| Auto-update | No (web = instant) | Yes (polling) | Yes (EAS Update OTA) |

---

## 2. Technology Stack & Justification

### Core Framework: Expo SDK 55 (Managed Workflow)

**Choice**: Expo with managed workflow over bare React Native or Flutter.

**Rationale**:
- **Code reuse from web wallet**: The web wallet already uses `@noble/ed25519`, `@noble/hashes`, and `@scure/bip39` for cryptographic operations. These pure-JS libraries run identically in React Native's Hermes engine. A Flutter port would require rewriting all crypto logic in Dart or bridging to C.
- **Managed workflow reduces native overhead**: Camera, secure storage, biometrics, and push notifications are all available as Expo modules without ejecting.
- **EAS Build/Update**: Over-the-air updates for JS bundle changes allow shipping fixes without Play Store review cycles. Only native module changes require a full rebuild.
- **Trade-off acknowledged**: Managed workflow limits access to raw Android APIs (e.g., Android HCE for NFC). NFC payment support is deferred to a future bare-workflow migration if demand materializes.

### Navigation: expo-router (File-Based)

File-based routing mirrors Next.js conventions. The `(tabs)` group provides the primary tab navigator; the `(auth)` group gates unauthenticated screens. Stack navigation within each tab uses standard expo-router stacks.

### UI Framework: react-native-paper (Material Design 3)

**Rationale**: MD3 provides a production-grade design system with dark/light theme support, accessible components, and consistent visual language. The mobile wallet adopts MD3 to feel native on Android rather than replicating the web aesthetic.

Custom theme tokens map Quillon brand colors into MD3 color scheme slots:

```typescript
// src/theme/index.ts
export const quilTheme: MD3Theme = {
  ...MD3DarkTheme,
  colors: {
    ...MD3DarkTheme.colors,
    primary: '#00BCD4',        // Quillon cyan
    secondary: '#7C4DFF',      // Quantum purple
    tertiary: '#FF6E40',       // Orange accent
    background: '#121218',     // Deep space background
    surface: '#121218',
    surfaceVariant: '#1E1E2A', // Card background
  },
};
```

### State Management: Zustand

**Choice**: Zustand over Redux, MobX, or Jotai.

**Rationale**:
1. **Zero boilerplate**: A store is a single function returning an object.
2. **Selective subscriptions**: Components subscribe to individual slices without re-rendering on unrelated state changes.
3. **Persist middleware**: Integrates with `expo-secure-store` for encrypted state persistence.

```typescript
// stores/walletStore.ts
import { create } from 'zustand';

interface WalletState {
  address: string | null;
  balance: number;
  isUnlocked: boolean;
  setBalance: (bal: number) => void;
}

export const useWalletStore = create<WalletState>()((set) => ({
  address: null,
  balance: 0,
  isUnlocked: false,
  setBalance: (bal) => {
    if (bal >= 0 && bal <= 21_000_000) {
      set({ balance: bal });
    }
  },
}));
```

### Cryptographic Libraries

| Library | Purpose | Notes |
|---|---|---|
| `@noble/ed25519` | Transaction signing | Same as web wallet. Pure JS, audited. |
| `@noble/hashes` | SHA3-256 key derivation | `SHA3-256(mnemonic_string) -> 32-byte seed` -- must match server. |
| `@scure/bip39` | Mnemonic generation/validation | 24-word (256-bit entropy) BIP39 English wordlist. |

**Critical compatibility constraint**: The key derivation path MUST produce identical addresses to the server (`crates/q-wallet/`) and Slint wallet (`gui/slint-wallet/src/wallet.rs`). The derivation is: `SHA3-256(mnemonic_text_bytes) -> Ed25519 signing key -> verifying key bytes -> "qnk" + hex(pubkey)`. This is NOT standard BIP32/BIP44 HD derivation; it is a single SHA3 hash of the mnemonic string.

```typescript
// services/wallet.ts — MUST match walletAuth.ts:keypairFromMnemonic()
export async function deriveAddress(mnemonic: string): Promise<string> {
  const mnemonicBytes = new TextEncoder().encode(mnemonic);
  const privateKey = sha3_256(mnemonicBytes);
  const publicKey = await ed.getPublicKeyAsync(privateKey);
  privateKey.fill(0); // Zero after use
  return 'qnk' + bytesToHex(publicKey);
}
```

### Additional Dependencies

| Package | Purpose |
|---|---|
| `expo-secure-store` | Encrypted key storage (Android Keystore backing) |
| `expo-local-authentication` | Biometric (fingerprint/face) authentication |
| `expo-camera` | QR code scanning via barcode detection |
| `react-native-qrcode-svg` | QR code rendering for receive addresses and PoS |
| `react-native-reanimated` 3 | Layout animations, shared element transitions |
| `expo-haptics` | Tactile feedback on swap execution, payment receipt |
| `expo-task-manager` | Background balance polling via Android WorkManager |
| `expo-notifications` | Push notifications (FCM transport) |
| `expo-network` | Network state detection (online/offline) |
| `@gorhom/bottom-sheet` | Token selector, slippage settings |
| `victory-native` | Price charts on DEX screen |

---

## 3. App Architecture

### File Structure

```
quillon-mobile/
  app/
    (auth)/
      _layout.tsx              -- Stack navigator for auth flow
      login.tsx                -- Create wallet / Import mnemonic
      backup.tsx               -- Mnemonic backup verification (write-down flow)
    (tabs)/
      _layout.tsx              -- Bottom tab navigator (5 tabs)
      index.tsx                -- Dashboard (balance, recent txs, token bar)
      send.tsx                 -- Send QUG/tokens
      receive.tsx              -- Receive address + QR + PoS mode
      dex.tsx                  -- DEX swap interface
      history.tsx              -- Full transaction history
    mining.tsx                 -- Mining stats (stack screen from dashboard)
    settings.tsx               -- Settings (stack screen from dashboard)
    tx/[id].tsx                -- Transaction detail (dynamic route)
    _layout.tsx                -- Root layout: auth guard, theme provider, SSE init
  src/
    services/
      api.ts                   -- REST client with failover (ported from web wallet)
      sse.ts                   -- SSE EventSource manager with reconnection
      wallet.ts                -- BIP39 mnemonic + Ed25519 signing + address derivation
      auth.ts                  -- Session management + biometric unlock
      secureStorage.ts         -- Encrypted key read/write via expo-secure-store
    stores/
      walletStore.ts           -- Balance, address, tokens, lock state
      dexStore.ts              -- Swap quotes, selected tokens, slippage
      networkStore.ts          -- Chain height, peers, sync %, server status
      settingsStore.ts         -- Theme, auto-lock timeout, currency preference
    components/
      BalanceCard.tsx           -- Hero balance display with gradient background
      TokenList.tsx             -- Scrollable token portfolio with prices
      TransactionItem.tsx       -- Single tx row (hash, amount, status, timestamp)
      QRScanner.tsx             -- Camera-based QR code reader
      QRDisplay.tsx             -- QR code renderer (address, payment request)
      TokenSelector.tsx         -- Bottom sheet token picker with search
      SwapInterface.tsx         -- Token A -> Token B swap form
      MiningStats.tsx           -- Pool hashrate, shares, earnings display
      BiometricGate.tsx         -- Biometric/PIN unlock overlay
      PosFullScreen.tsx         -- Full-screen PoS QR with countdown timer
      SkeletonLoader.tsx        -- Animated placeholder during API fetches
    hooks/
      useBalance.ts            -- Subscribe to balance via SSE + polling fallback
      useSSE.ts                -- SSE connection lifecycle hook
      useAuth.ts               -- Biometric check + auto-lock timer
      useDexQuote.ts           -- Debounced swap quote fetching
    theme/
      index.ts                 -- MD3 theme tokens (dark + light)
    utils/
      formatBalance.ts         -- Number formatting (subscript zeros, locale)
      validateAddress.ts       -- qnk-prefixed hex address validation
      sanitizeBalance.ts       -- MAX_SANE_BALANCE guard (21M QUG cap)
  assets/
    icon.png                   -- App icon (1024x1024)
    splash.png                 -- Splash screen
  app.json                     -- Expo config (bundle ID, permissions, scheme)
  eas.json                     -- EAS Build profiles (development, preview, production)
```

### Data Flow Architecture

```
                     ┌──────────────────────────────┐
                     │        Expo App Shell         │
                     │   (_layout.tsx: auth guard)   │
                     └──────────┬───────────────────┘
                                │
               ┌────────────────┼────────────────┐
               ▼                ▼                ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │  Zustand     │ │  SSE Manager │ │  REST Client │
      │  Stores      │ │  (sse.ts)    │ │  (api.ts)    │
      │              │ │              │ │              │
      │ walletStore  │◄┤  balance     │ │  /transfer   │
      │ dexStore     │ │  newBlock    │ │  /dex/swap   │
      │ networkStore │ │  tokenPrice  │ │  /wallet/*   │
      └──────────────┘ └──────┬───────┘ └──────┬───────┘
               ▲               │                │
               │               ▼                ▼
               │       ┌──────────────────────────┐
               │       │  https://quillon.xyz/api  │
               └───────│  SSE: /api/v1/events     │
                       │  REST: /api/v1/*         │
                       └──────────────────────────┘
```

### Auth Guard Logic

The root `_layout.tsx` wraps the entire app in an auth check:

```typescript
// app/_layout.tsx (simplified)
export default function RootLayout() {
  const isUnlocked = useWalletStore(s => s.isUnlocked);
  const hasWallet = useWalletStore(s => s.address !== null);

  if (!hasWallet) return <Redirect href="/(auth)/login" />;
  if (!isUnlocked) return <BiometricGate />;
  return <Stack />;
}
```

---

## 4. Screen-by-Screen Specification

### 4.1 Login / Onboarding

```
┌──────────────────────────────┐
│         QUILLON              │
│     ◆ Quantum Wallet         │
│                              │
│  ┌──────────────────────┐    │
│  │  Create New Wallet   │    │
│  └──────────────────────┘    │
│                              │
│  ┌──────────────────────┐    │
│  │  Import Mnemonic     │    │
│  └──────────────────────┘    │
│                              │
│  ┌──────────────────────┐    │
│  │  Server Vault (OAuth)│    │
│  └──────────────────────┘    │
│                              │
│  v10.1.3       quillon.xyz   │
└──────────────────────────────┘
```

**Data sources**: None (local-only operations except OAuth).
**User interactions**: (a) Create: generate 24-word mnemonic, display backup screen, derive address. (b) Import: paste or type mnemonic, validate BIP39, derive address. (c) Server Vault: launch OAuth2 PKCE flow in system browser, receive callback with bearer token.
**State**: `walletStore.setAddress(derived)`, `secureStorage.setMnemonic(encrypted)`.
**Error states**: Invalid mnemonic checksum (red warning), OAuth timeout, network unreachable.

### 4.2 Dashboard

```
┌──────────────────────────────┐
│  ◇ Quillon          ⚙ ▪▪▪   │
├──────────────────────────────┤
│  ┌────────────────────────┐  │
│  │  ██████████████████████│  │
│  │  12,450.3827 QUG       │  │
│  │  ≈ $1,245.03 USD       │  │
│  │  qnk7a3f...8e2d  📋   │  │
│  └────────────────────────┘  │
│                              │
│  Token Bar (horizontal scroll)│
│  [QUG ▲2.3%] [WETH] [QUGUSD]│
│                              │
│  Recent Transactions         │
│  ┌────────────────────────┐  │
│  │ ↑ Sent  -50.00 QUG    │  │
│  │   to qnk8b...  3m ago │  │
│  ├────────────────────────┤  │
│  │ ↓ Recv +125.00 QUG    │  │
│  │   from qnk3a... 1h ago│  │
│  ├────────────────────────┤  │
│  │ ⛏ Mining +5.0000 QUG  │  │
│  │   Block #9,412,008     │  │
│  └────────────────────────┘  │
│                              │
├──────────────────────────────┤
│  🏠   ↑   ↓   ⇄   📜       │
│ Home Send Recv DEX History   │
└──────────────────────────────┘
```

**Data sources**: `GET /api/v1/wallet/{address}/balance` (polling fallback), SSE `BalanceUpdated` events, `GET /api/v1/wallet/{address}/history?limit=10`, `GET /api/v1/dex/tokens` (token bar prices).
**User interactions**: Tap balance card to toggle QUG/USD. Tap address to copy (with haptic + auto-clear 60s). Tap transaction to navigate to `tx/[id]`. Pull-to-refresh. Tap gear icon for settings.
**State**: `walletStore.balance`, `walletStore.tokens`, transaction list in local component state.
**Error states**: Network offline banner (yellow), stale data indicator, zero balance empty state.
**Animations**: Balance counter animation (`withSpring`). Token bar horizontal scroll. Transaction list `FadeIn`.

### 4.3 Send

```
┌──────────────────────────────┐
│  ← Send QUG                  │
├──────────────────────────────┤
│  To Address                  │
│  ┌────────────────────┐ [📷] │
│  │ qnk...             │     │
│  └────────────────────┘     │
│                              │
│  Amount                      │
│  ┌────────────────────┐     │
│  │ 0.00          QUG ▼│     │
│  └────────────────────┘     │
│  Available: 12,450.38 QUG    │
│  ≈ $0.00 USD                │
│                              │
│  Memo (optional)             │
│  ┌────────────────────┐     │
│  │                    │     │
│  └────────────────────┘     │
│                              │
│  Network Fee: ~0.001 QUG     │
│                              │
│  ┌────────────────────────┐  │
│  │    Review Transaction  │  │
│  └────────────────────────┘  │
└──────────────────────────────┘
```

**Data sources**: `walletStore.balance` for available amount, `walletStore.tokens` for token selector.
**User interactions**: Paste address or scan QR. Enter amount. Tap "Review" for confirmation modal with biometric auth. After auth, local Ed25519 signing, then `POST /api/v1/transfer`.
**Error states**: Invalid address format, insufficient balance, network error (queue for offline).

### 4.4 Receive (+ PoS Mode)

```
┌──────────────────────────────┐
│  ← Receive                   │
├──────────────────────────────┤
│       ┌──────────────┐       │
│       │  ▓▓▓▓▓▓▓▓▓▓  │       │
│       │  ▓▓ QR Code▓▓ │       │
│       │  ▓▓▓▓▓▓▓▓▓▓  │       │
│       └──────────────┘       │
│                              │
│  qnk7a3f8b...2e8d    📋     │
│                              │
│  ┌────────────────────────┐  │
│  │   Share Address        │  │
│  └────────────────────────┘  │
│  ┌────────────────────────┐  │
│  │   💳 Point of Sale     │  │
│  └────────────────────────┘  │
└──────────────────────────────┘
```

**PoS Waiting Screen** (full-screen after entering amount):

```
┌──────────────────────────────┐
│       ┌──────────────┐       │
│       │  ▓▓ QR CODE ▓▓│       │
│       │  qnk:ADDR?   │       │
│       │  amount=25.00 │       │
│       └──────────────┘       │
│                              │
│    Waiting for payment...    │
│    ⏱ 4:32 remaining         │
│                              │
│  ┌────────────────────────┐  │
│  │       Cancel           │  │
│  └────────────────────────┘  │
└──────────────────────────────┘
```

**Payment detection**: SSE `BalanceUpdated` where `new_balance >= old + expected`. Fallback: REST polling every 5s. On receipt: confetti + haptic burst + "Payment Received" banner.

### 4.5 DEX

```
┌──────────────────────────────┐
│  ← DEX                 ⚙    │
├──────────────────────────────┤
│  You Pay                     │
│  ┌────────────────────┐     │
│  │ [QUG ▼]    100.00  │     │
│  └────────────────────┘     │
│          ↕ (swap direction)  │
│  You Receive                 │
│  ┌────────────────────┐     │
│  │ [WETH ▼]    0.0312 │     │
│  └────────────────────┘     │
│                              │
│  Rate: 1 QUG = 0.000312 WETH│
│  Fee: 0.3%  Slippage: 0.5%  │
│  Price Impact: 0.02%         │
│                              │
│  ┌────────────────────────┐  │
│  │       Swap             │  │
│  └────────────────────────┘  │
│                              │
│  ┌────────────────────────┐  │
│  │  Price Chart  (24h ▼)  │  │
│  │  ╱╲  ╱╲╱╲              │  │
│  │ ╱  ╲╱    ╲╱╲           │  │
│  └────────────────────────┘  │
└──────────────────────────────┘
```

**Data sources**: `GET /api/v1/dex/tokens`, `GET /api/v1/dex/quote` (debounced 500ms), `GET /api/v1/oracle/price-history/{token}`, `POST /api/v1/dex/swap`.
**User interactions**: Token selectors open bottom sheet. Auto-calculate output. Swap arrows reverse direction. Gear for slippage (0.1%, 0.5%, 1.0%, custom). "Swap" opens biometric confirmation.
**Error states**: Insufficient balance, excessive price impact (>5% warning, >10% block), quote expired, insufficient liquidity.

### 4.6 History

```
┌──────────────────────────────┐
│  ← History            🔍     │
├──────────────────────────────┤
│  Filter: [All ▼] [Token ▼]  │
├──────────────────────────────┤
│  Today                       │
│  ┌────────────────────────┐  │
│  │ ↑ Sent  -50.00 QUG    │  │
│  │   to qnk8b...  14:32  │  │
│  │   ✓ Confirmed (6 conf) │  │
│  ├────────────────────────┤  │
│  │ ⇄ Swap  QUG → WETH    │  │
│  │   100 QUG → 0.031 WETH│  │
│  ├────────────────────────┤  │
│  │ ⛏ Mining +5.0000 QUG  │  │
│  │   Block #9,412,008     │  │
│  └────────────────────────┘  │
│  ... (infinite scroll)       │
└──────────────────────────────┘
```

**Data sources**: `GET /api/v1/wallet/{address}/history?offset=0&limit=50`. Filters client-side for cached window.
**FlatList config**: `initialNumToRender={15}`, `maxToRenderPerBatch={10}`, `windowSize={5}`, fixed row height (72dp).

### 4.7 Mining (Monitoring Only)

```
┌──────────────────────────────┐
│  ← Mining Dashboard          │
├──────────────────────────────┤
│  Pool Status: ● Connected    │
│  ┌──────────┐ ┌──────────┐  │
│  │ Hashrate │ │ Shares   │  │
│  │ 245 H/s  │ │ 1,247    │  │
│  └──────────┘ └──────────┘  │
│  ┌──────────┐ ┌──────────┐  │
│  │ Earnings │ │ Blocks   │  │
│  │ 52.4 QUG │ │ 3 found  │  │
│  └──────────┘ └──────────┘  │
│                              │
│  Pool Info                   │
│  Workers: 12  |  Fee: 1.5%   │
│  Network Diff: 1,284,921     │
└──────────────────────────────┘
```

**No local mining** -- mobile CPUs are not competitive and drain battery. This is monitoring only.

### 4.8 Settings

Security, network, appearance, and about sections. Biometric toggle, auto-lock timeout, theme, chain height, delete wallet with confirmation.

---

## 5. Security Architecture

### Threat Model

Self-custodial wallet holding Ed25519 private keys derived from BIP39 mnemonics. Primary threat: key extraction leading to fund theft.

### Key Storage

```
┌────────────────────────────────────────────┐
│           Android Keystore (TEE/SE)        │
│  ┌──────────────────────────────────────┐  │
│  │  AES-256-GCM encryption key          │  │
│  │  (hardware-backed, non-exportable)   │  │
│  └──────────────────────────┬───────────┘  │
│                             │ encrypts      │
│  ┌──────────────────────────▼───────────┐  │
│  │  expo-secure-store entry:            │  │
│  │  "wallet_mnemonic" = AES(mnemonic)   │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
```

Mnemonic encrypted at rest via `expo-secure-store`, backed by Android Keystore on API 23+. Hardware-backed on devices with TEE/SE. Warning displayed on software-only devices.

### Authentication Flow

```
App Launch
    │
    ├─ Has wallet? ──No──► Login screen
    │
    Yes
    │
    ├─ Biometric enabled? ──No──► PIN entry screen
    │
    Yes
    │
    ├─ expo-local-authentication.authenticateAsync()
    │    │
    │    ├─ Success ──► Decrypt mnemonic ──► Derive keypair ──► Unlock
    │    │
    │    └─ Failure ──► PIN fallback (3 attempts, then lockout 60s)
    │
    └─ Auto-lock timer starts (configurable)
```

### Signing Flow

Transaction signing occurs entirely on-device. Private key derived in-memory, used to sign, then zeroed:

```typescript
export async function signTransaction(
  mnemonic: string,
  payload: Record<string, unknown>
): Promise<{ signature: string; publicKey: string }> {
  const mnemonicBytes = new TextEncoder().encode(mnemonic);
  const privateKey = sha3_256(mnemonicBytes);
  try {
    const message = new TextEncoder().encode(JSON.stringify(payload));
    const signature = await ed.signAsync(message, privateKey);
    const publicKey = await ed.getPublicKeyAsync(privateKey);
    return {
      signature: bytesToHex(signature),
      publicKey: bytesToHex(publicKey),
    };
  } finally {
    privateKey.fill(0); // Zero key material
  }
}
```

### Additional Protections

| Protection | Implementation | Notes |
|---|---|---|
| Clipboard auto-clear | `setTimeout(() => Clipboard.setString(''), 60000)` | Prevents clipboard scraping |
| Anti-screenshot | `expo-screen-capture: preventScreenCapture()` on backup screen | Protects mnemonic display |
| Root detection | `expo-device.isRootedExperimentalAsync()` | Warning only (do not block) |
| Certificate pinning | Custom fetch wrapper with SHA-256 pin for quillon.xyz | Prevents MITM |
| Auto-lock | `AppState.addEventListener('change')` starts timer on background | Configurable timeout |
| PIN brute-force | Exponential backoff: 3 failures = 60s, 6 = 5m, 9 = 30m | Slows offline attacks |

---

## 6. Networking & Real-Time Data

### REST Client with Failover

Ported from web wallet's `api.ts` failover logic:

- **Primary server**: `https://quillon.xyz` (Epsilon, 10Gbit).
- **Failover cooldown**: 30 seconds.
- **Return-to-primary**: Poll every 60 seconds when on backup.
- **Health check**: `GET /api/v1/health` with 3-second timeout.
- **Authentication**: X-Wallet-Auth header (Ed25519 signature of `SHA3-256(address_bytes || timestamp_le_8 || path_utf8)`).

### SSE / Real-Time Events

Custom fetch-based SSE parser (React Native lacks native EventSource):

```typescript
export class SSEManager {
  private controller: AbortController | null = null;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 30000;

  async connect(url: string, handlers: SSEHandlers) {
    this.controller = new AbortController();
    try {
      const response = await fetch(`${url}/api/v1/events`, {
        headers: { Accept: 'text/event-stream' },
        signal: this.controller.signal,
      });
      const reader = response.body?.getReader();
      // Parse SSE frames, route to handlers
      this.reconnectDelay = 1000; // Reset on success
    } catch (err) {
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
      setTimeout(() => this.connect(url, handlers), this.reconnectDelay);
    }
  }
}
```

**Events consumed**: `BalanceUpdated`, `NewBlock`, `TokenPriceUpdate`, `MiningReward`.

### Background Sync

`expo-task-manager` with WorkManager:

```typescript
TaskManager.defineTask('BALANCE_SYNC', async () => {
  const address = await SecureStore.getItemAsync('wallet_address');
  if (!address) return BackgroundFetch.BackgroundFetchResult.NoData;
  const res = await fetch(`https://quillon.xyz/api/v1/wallet/${address}/balance`);
  return res.ok ? BackgroundFetch.BackgroundFetchResult.NewData
                : BackgroundFetch.BackgroundFetchResult.Failed;
});

BackgroundFetch.registerTaskAsync('BALANCE_SYNC', {
  minimumInterval: 15 * 60, // Android minimum
  stopOnTerminate: false,
  startOnBoot: true,
});
```

### Battery Optimization

| App State | SSE | REST Polling | Background Task |
|---|---|---|---|
| Foreground | Active | On-demand | Not needed |
| Background (<5m) | Active (keep-alive) | Disabled | Not yet |
| Background (>5m) | Disconnected | Disabled | Every 15 min |
| Killed | N/A | N/A | Every 15 min |

---

## 7. Performance Considerations

### Bundle Size Budget: < 15 MB APK

| Component | Estimated Size |
|---|---|
| Hermes JS bundle | ~3 MB |
| react-native-paper + MD3 | ~1.5 MB |
| Noble crypto libraries | ~200 KB |
| Expo modules (camera, secure-store, etc.) | ~5 MB |
| Assets (icons, splash, fonts) | ~2 MB |
| React Native runtime | ~3 MB |
| **Total** | **~14.7 MB** |

### Startup Time Target: < 2 seconds to interactive

Hermes bytecode load (~300ms) -> Zustand hydrate from secure store (~200ms) -> Auth check (~100ms) -> Render dashboard skeleton (~100ms) -> First API response fills data.

### Memory Budget: < 150 MB RSS

Dashboard renders skeleton loader immediately. Cached data from previous session shown while fresh data loads.

### FlatList Virtualization

- `initialNumToRender={15}`
- `maxToRenderPerBatch={10}`
- `windowSize={5}`
- `getItemLayout` with fixed 72dp row height
- `keyExtractor={item => item.tx_hash}`

---

## 8. DEX Mobile UX

### Token Selector Bottom Sheet

```
┌──────────────────────────────┐
│  ── (drag handle) ──         │
│  Select Token                │
│  ┌────────────────────┐     │
│  │ 🔍 Search tokens   │     │
│  └────────────────────┘     │
│  Your Tokens                 │
│  [◆ QUG]  12,450.38         │
│  [◆ WETH]  0.0312           │
│  All Tokens                  │
│  [◆ BTC-WRAPPED] 0.00       │
│  ...                         │
└──────────────────────────────┘
```

Non-zero balances first. Search by symbol and name.

### Real-Time Quote Updates

```typescript
function useDexQuote(tokenIn: string, tokenOut: string, amountIn: string) {
  const [quote, setQuote] = useState<DexQuote | null>(null);
  useEffect(() => {
    if (!amountIn || parseFloat(amountIn) <= 0) { setQuote(null); return; }
    const timer = setTimeout(async () => {
      const q = await api.getDexQuote(tokenIn, tokenOut, amountIn);
      setQuote(q);
    }, 500);
    return () => clearTimeout(timer);
  }, [tokenIn, tokenOut, amountIn]);
  return { quote };
}
```

### Swap Execution

1. User taps "Swap" -> Confirmation bottom sheet (amounts, rate, fee, slippage, minimum received)
2. Biometric/PIN prompt
3. `POST /api/v1/dex/swap` with signed payload
4. Success: haptic burst + confetti animation

---

## 9. Mining on Mobile

### No Local Mining

Mobile CPUs are not competitive. Sustained CPU usage drains battery in minutes. This is monitoring-only.

### Pool Share Monitoring

Stats from server API: hashrate, shares, blocks found, pending balance, pool-wide stats, payout history.

### Push Notifications

| Event | Notification |
|---|---|
| Block found | "Block Found! #9,412,128 (+5.0 QUG)" |
| Payout received | "Payout: 10.00 QUG sent to wallet" |
| Worker offline >10 min | "Worker Offline: worker-01 disconnected" |

FCM token registered via `POST /api/v1/notifications/register`.

---

## 10. QR Code & Payment Flows

### QR Code Format

```
qnk:qnk7a3f8b...2e8d?amount=25.00&memo=Coffee%20order%20%2342
```

- **Scheme**: `qnk:`
- **Address**: 67-character (`qnk` + 64 hex)
- **amount** (optional): Decimal QUG
- **memo** (optional): URL-encoded string
- **token** (optional): Non-QUG token symbol

### Deep Link Handling

```json
// app.json
{
  "expo": {
    "scheme": "qnk",
    "android": {
      "intentFilters": [{
        "action": "VIEW",
        "data": [{ "scheme": "qnk" }],
        "category": ["DEFAULT", "BROWSABLE"]
      }]
    }
  }
}
```

### PoS Payment Detection

Dual-channel monitoring:
1. **SSE** (primary): `BalanceUpdated` where `new_balance >= initial + expected`
2. **REST** (fallback): Polling every 5s

On detection: vibration (long-short-long), success animation, amount display.

---

## 11. Offline & Background Behavior

### Cached Wallet Data

Zustand persist middleware hydrates from `expo-secure-store` on cold start. Cached: balance, last 50 transactions, token list, network status, settings.

### Transaction Queue

```typescript
interface PendingTransaction {
  id: string;
  signedPayload: string;
  createdAt: number;
  retryCount: number;
  lastError?: string;
}

// Offline: sign locally, push to queue (secure store)
// On network recovery: flush FIFO via POST /api/v1/transfer
// Max 3 retries per transaction
```

### Push Notifications

Register: `POST /api/v1/notifications/register { address, fcm_token, platform: "android" }`.
Events: incoming transfer, mining reward, swap completion.

---

## 12. Testing Strategy

### Unit Tests (Jest + @testing-library/react-native)

| Module | Target | Focus |
|---|---|---|
| `services/wallet.ts` | 100% | Key derivation matches server |
| `utils/formatBalance.ts` | 100% | Edge cases: 0, MAX, NaN, negative |
| `utils/validateAddress.ts` | 100% | Valid qnk + 64 hex, reject malformed |
| `stores/walletStore.ts` | 95% | Balance sanitization, persist/hydrate |
| `services/api.ts` | 90% | Failover logic, auth headers |
| Components | 80% | Render states, interactions |

### Critical Cross-Wallet Compatibility Test

```typescript
test('address derivation matches server and desktop', async () => {
  const mnemonic = 'abandon abandon abandon abandon abandon abandon ' +
                   'abandon abandon abandon abandon abandon about';
  const mnemonicBytes = new TextEncoder().encode(mnemonic);
  const seed = sha3_256(mnemonicBytes);
  const publicKey = await ed25519.getPublicKeyAsync(seed);
  const address = 'qnk' + bytesToHex(publicKey);
  expect(address).toMatch(/^qnk[0-9a-f]{64}$/);
  expect(address.length).toBe(67);
  // Must match: cargo test --package q-wallet -- test_known_mnemonic
});
```

### Security Audit Checklist

- [ ] Mnemonic never logged (grep for console.log containing seed/mnemonic/private)
- [ ] Private key zeroed after signing
- [ ] Secure store entries encrypted (verify Android Keystore backing)
- [ ] No plaintext secrets in APK (apktool d + grep)
- [ ] Certificate pinning active (attempt MITM with mitmproxy)
- [ ] Root detection warning displayed
- [ ] Auto-lock triggers correctly on background
- [ ] PIN brute-force lockout enforced
- [ ] Clipboard cleared after address copy

---

## 13. Build & Distribution

### EAS Build Configuration

```json
{
  "build": {
    "development": {
      "distribution": "internal",
      "android": { "buildType": "apk", "gradleCommand": ":app:assembleDebug" }
    },
    "preview": {
      "distribution": "internal",
      "android": { "buildType": "apk" }
    },
    "production": {
      "android": { "buildType": "app-bundle" }
    }
  }
}
```

### Distribution Channels

1. **Google Play Store**: AAB via EAS Submit.
2. **Direct APK**: `quillon.xyz/downloads/quillon-wallet-v{VERSION}.apk` (sideload).
3. **EAS Update (OTA)**: JS bundle updates without Play Store review.

### Version Management

`/api/v1/version` returns `min_mobile_version`. App compares on launch:

```typescript
const serverVersion = await api.getVersion();
if (serverVersion.min_mobile_version > APP_VERSION) {
  // Show mandatory update screen
}
```

---

## 14. Migration & Compatibility

### Mnemonic Import

All three wallet clients use identical key derivation:

```
Input:  BIP39 mnemonic string (12 or 24 words)
Step 1: SHA3-256(mnemonic_string_as_utf8_bytes)  ->  32-byte seed
Step 2: Ed25519 signing key from seed
Step 3: Ed25519 verifying key (public key)
Step 4: Address = "qnk" + hex(public_key_bytes)
```

This is NOT BIP32/BIP44. One mnemonic = one address. A mnemonic from Slint desktop imports directly into mobile and produces the same address.

### API Compatibility

Mobile targets `/api/v1/*` exclusively. Server maintains v1 backward compatibility.

### Transaction Format

All clients produce identical signed transaction payloads:
- Fields: `from`, `to`, `amount` (24-decimal fixed-point), `nonce`, `memo`, `timestamp`
- Signature: Ed25519 over SHA3-256 of canonical transaction bytes
- Submission: `POST /api/v1/transfer`

---

## 15. Risk Assessment & Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Private key extraction (rooted device) | Medium | Critical | Warn on root; hardware Keystore; PIN brute-force protection |
| R2 | MITM on API connection | Low | Critical | TLS 1.3 + certificate pinning |
| R3 | Supply chain attack (npm) | Medium | Critical | Lock versions; audit @noble/@scure; no postinstall |
| R4 | Battery drain from SSE/polling | Medium | Low | Disconnect SSE >5m background; WorkManager 15m |
| R5 | Key derivation mismatch | Low | Critical | Cross-wallet test vectors; release-blocking test |
| R6 | Stale cached balance | Medium | Medium | Timestamps; visual staleness indicator; SSE |
| R7 | Transaction double-submit | Low | High | Nonce-based replay protection; queue dedup |
| R8 | Offline tx never broadcast | Low | Medium | Queue persistence; retry on recovery; notification |
| R9 | Google Play rejection | Medium | Medium | Comply with financial services policy; disclosures |
| R10 | expo-secure-store software fallback | Medium | High | Detect hardware backing; warn; stronger PIN |

---

## 16. Development Timeline (Phased)

### Phase 1: MVP (4 weeks)

| Week | Tasks |
|---|---|
| 1 | Scaffolding, wallet service, secure storage, login screen |
| 2 | Dashboard, REST client, SSE manager, settings |
| 3 | Send (QR scan, signing, confirmation), Receive (QR display, PoS) |
| 4 | History (FlatList, filters), offline mode, testing, APK build |

### Phase 2: DEX (2 weeks)

| Week | Tasks |
|---|---|
| 5 | DEX screen, token selector, swap form, quote fetching |
| 6 | Swap execution + biometric, liquidity view, testing |

### Phase 3: Polish (2 weeks)

| Week | Tasks |
|---|---|
| 7 | Mining stats, PoS mode, push notifications (FCM) |
| 8 | Biometric refinement, auto-lock, background sync, animations |

### Phase 4: Hardening (2 weeks)

| Week | Tasks |
|---|---|
| 9 | Security audit, cert pinning, root detection, pen testing |
| 10 | Performance profiling, E2E tests, Play Store submission |

**Total**: 10 weeks to production.

---

## 17. Open Questions for Review

1. **Single-address vs BIP44 HD**: Current derivation = one address per mnemonic. No rotation. Should mobile introduce BIP44 as a protocol upgrade?

2. **Expo managed vs bare workflow**: Managed blocks NFC (Android HCE). Start bare to avoid future migration?

3. **SSE vs WebSocket**: SSE is unidirectional, reconnection manual. Add WebSocket endpoint for mobile?

4. **Post-quantum on mobile**: Server supports Dilithium5/Kyber1024. @noble has no lattice crypto. Include WASM Dilithium5, or defer?

5. **Offline nonce management**: Without network, can't query nonce. (a) Cache + local increment? (b) Timestamp-based? (c) Require network?

6. **Token price oracle trust**: Centralized price feed. Cross-reference independent oracle?

7. **Push notification privacy**: FCM token links address to device. Opt-in with disclosure, or privacy-preserving system?

8. **Minimum Android version**: API 23 (6.0) vs API 28 (9.0) for guaranteed hardware-backed Keystore?

9. **APK sideloading security**: Publish SHA-256 checksums on separate channel? HTTPS + cert pinning sufficient?

10. **Multi-wallet support**: One wallet per app instance. Power users want multiple. V1 or defer?

---

## Appendix A: API Endpoint Reference

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/api/v1/health` | No | Server status, height, peers |
| GET | `/api/v1/version` | No | Server version, min mobile version |
| GET | `/api/v1/wallet/{addr}/balance` | Yes | QUG + token balances |
| GET | `/api/v1/wallet/{addr}/history` | Yes | Transaction history (paginated) |
| POST | `/api/v1/transfer` | Yes | Submit signed transaction |
| GET | `/api/v1/dex/tokens` | No | List DEX tokens with prices |
| GET | `/api/v1/dex/quote` | No | Swap quote |
| POST | `/api/v1/dex/swap` | Yes | Execute swap |
| GET | `/api/v1/oracle/price-history/{token}` | No | Price chart data |
| GET | `/api/v1/mining/stats/{wallet}` | Yes | Personal mining stats |
| GET | `/api/v1/mining/pool/stats` | No | Pool-wide stats |
| GET | `/api/v1/events` | Yes | SSE real-time event stream |
| POST | `/api/v1/notifications/register` | Yes | Register FCM push token |

## Appendix B: Key Derivation Test Vectors

```
Mnemonic: (standard 12-word BIP39 test vector)
Step 1:   SHA3-256(mnemonic_utf8) -> 32 bytes
Step 2:   Ed25519 private key = step1_output
Step 3:   Ed25519 public key = derive(private_key)
Step 4:   Address = "qnk" + hex(public_key)
```

**Cross-validation**:
```bash
# Server (Rust):
cargo test --package q-wallet -- test_known_mnemonic_derivation --nocapture
# Desktop (Slint):
cargo test --package slint-wallet -- test_address_derivation --nocapture
# Mobile (TypeScript):
npx jest wallet-compat.test.ts
```

All three must produce the same address. Release-blocking test.

---

## 18. Peer Review Response (2026-03-24)

This section documents decisions made in response to the peer review.

### Accepted Changes (Implemented)

| # | Finding | Action Taken |
|---|---|---|
| PR-1 | Key derivation used `mnemonicToSeedSync` (BIP39 PBKDF2) instead of `SHA3-256(mnemonic_utf8)` | **Fixed**: `wallet.ts` now uses `sha3_256(new TextEncoder().encode(mnemonic))` directly, matching `walletAuth.ts:keypairFromMnemonic()` |
| PR-2 | SSE buffer may accumulate without bound | **Fixed**: Added `MAX_BUFFER_SIZE = 64KB` guard; buffer flushed if exceeded |
| PR-3 | SSE max retries should fall back to polling | **Fixed**: After 20 retries, emits `polling_fallback` event for `useBalance` hook |
| PR-4 | PIN brute-force lockout stored in memory only | **Fixed**: Lockout state persisted to `expo-secure-store` via `PinLockoutState` interface; survives app restart |
| PR-5 | Offline transaction queue unsafe without nonce mgmt | **Fixed**: MVP requires network for send; `expo-network` check added to `send.tsx`; offline queue deferred to Phase 2 |
| PR-6 | No crash reporting or error monitoring | **Added**: `src/services/monitoring.ts` with Sentry integration (auto-redacts wallet addresses and mnemonics) |
| PR-7 | No internationalization framework | **Added**: `src/i18n/` with i18next + react-i18next; English strings extracted to `en.json` |
| PR-8 | Missing accessibility labels | **Added**: `accessibilityRole`, `accessibilityLabel` on BalanceCard, BiometricGate, and key interactive elements |
| PR-9 | "NFC-ready" claim contradicts deferred NFC | **Fixed**: Removed "NFC-ready" from PoS feature description |

### Accepted Decisions (Design-Level)

| Question | Decision | Rationale |
|---|---|---|
| Q1: Single-address vs BIP44 | **Single-address for MVP** | Maintain compatibility. Store `derivation_version: 1` for future BIP44 migration. |
| Q2: Expo managed vs bare | **Managed for MVP** | NFC not in scope. Eject to bare if NFC becomes required. |
| Q3: SSE vs WebSocket | **SSE** | Sufficient for unidirectional events. WebSocket adds complexity with no MVP benefit. |
| Q4: Post-quantum on mobile | **Defer** | No @noble lattice crypto available. WASM Dilithium5 is Phase 3+. |
| Q5: Offline nonce | **Require network** | Too complex for MVP. Show "No network" error. Queue only for retry after temporary failure. |
| Q6: Token price oracle | **Central oracle acceptable** | Disclose single-source in UI. Multi-source aggregation is Phase 3+. |
| Q7: Push notification privacy | **Opt-in with disclosure** | FCM registration gated behind settings toggle + privacy explanation. |
| Q8: Minimum Android version | **API 23 (Android 6)** | Detect and warn on software-only Keystore. Stronger PIN-based KDF as fallback. |
| Q9: APK sideloading | **HTTPS + SHA-256 checksum** | Publish checksum on website for manual verification. |
| Q10: Multi-wallet | **Defer to post-MVP** | Design store with keyed `wallets: { [addr]: data }` to enable later. |

### Acknowledged but Deferred

| Finding | Deferral Reason |
|---|---|
| Certificate pinning with backup pins and rotation mechanism | Phase 4 hardening — requires EAS config update mechanism |
| Root detection on every auth attempt | Phase 4 — `isRootedExperimentalAsync()` is slow, per-launch is sufficient for MVP |
| Anti-screenshot on login screen (mnemonic entry) | Phase 4 — implement with `expo-screen-capture` |
| Detox/Maestro E2E test suite | Phase 4 — manual QA sufficient for internal APK, E2E for Play Store |
| CI/CD pipeline (GitHub Actions) | Phase 2 — EAS Build handles CI for now |
| Dynamic font size support | Phase 3 — react-native-paper respects system font scale by default |

---

*End of Technical Review Document*

