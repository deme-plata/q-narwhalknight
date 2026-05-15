# Handoff — DSM-Inspired Bluetooth Offline Transfer for Quillon Graph

**Date:** 2026-05-15
**Owner:** _unassigned — pick up after v10.9.27 sync stability lands_
**Status:** RESEARCH / DEFERRED. Not blocking mainnet. Captured here so the design isn't lost.

---

## 1. Goal

Enable two Quillon Graph wallets to **transfer QUG to each other without internet**, via Bluetooth, with on-chain settlement once either party reconnects. Practical scenarios:

- Festivals, transit, sub-Saharan / remote areas with intermittent connectivity
- Conferences (rapid attendee-to-attendee transfers)
- Censorship-resistant ad-hoc commerce
- Resilience theatre: a payment rail that survives wholesale internet outage

**Phased rollout:**

1. **Phase A — Browser (Web Bluetooth API)** — laptop/desktop wallet at `quillon.xyz` gains a "BLE pay" tab. Two open laptops in the same room can transact. Chrome/Edge only; Firefox/Safari don't ship Web Bluetooth yet.
2. **Phase B — Expo mobile app** — port the protocol to the React Native / Expo wallet for iOS + Android. Uses native BLE bridges (`expo-bluetooth-le-plx` or similar).

---

## 2. Why DSM (instead of building from scratch)

Brandon Ramsay's **Deterministic State Machine** (`deterministicstatemachine/dsm` on GitHub) has already solved the hard parts of offline bilateral transfer for Bitcoin Signet:

| Asset | Where in DSM | Value to us |
|---|---|---|
| **Bilateral hash-chain transaction model** | `dsm_client/.../dsm/src/bilateral/` and `core/bilateral_transaction_manager.rs` (78K LOC) | Lift directly; swap dBTC primitives for QUG |
| **Android BLE stack** | `dsm_client/android/.../bridge/ble/BleCoordinator.kt` (72K LOC), `GattClientSession.kt` (56K LOC), `BleAdvertiser.kt`, `BleScanner.kt` | Reusable for Expo Phase B (native bridge) |
| **Lean4 formal proofs** | `lean4/DSMOfflineFinality.lean`, `DSM_dBTC_Conservation.lean`, `DSMCertChain.lean`, `DSMCryptoBinding.lean`, `DSMNonInterference.lean`, `DSMCardinality.lean`, `DSM_dBTC_TrustReduction.lean` | Template for our own offline-finality proof — the math doesn't care which token's balance the deltas represent |
| **License** | Dual Apache-2.0 / MIT | Compatible with Q-NarwhalKnight |

**What it's NOT:** a competing blockchain. DSM explicitly is *"a cryptographic state and identity layer designed for sovereign coordination, offline-capable settlement, and deterministic application logic."* It sits on top of an anchor chain. Brandon uses Bitcoin Signet; we'd use Quillon Graph mainnet.

---

## 3. Architecture mapping — DSM → Quillon Graph

```
                       Two devices in BLE range (~10m)
                                      │
                                      ▼
   ┌───────────────────┐                              ┌───────────────────┐
   │  Alice's wallet   │ ◄────── GATT R/W ──────►    │   Bob's wallet    │
   │  (browser/Expo)   │      bilateral state         │  (browser/Expo)   │
   └─────────┬─────────┘      diff exchange            └─────────┬─────────┘
             │                                                   │
             │ relationship-local hash chain                     │
             │  step_n = H(step_{n-1} || alice_balance_delta     │
             │              || bob_balance_delta || tx_meta      │
             │              || alice_sig || bob_sig)             │
             │                                                   │
             ▼ (next online connection — anchor settle)           ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                  Quillon Graph mainnet                          │
   │  ────────────────────────────────────────────────────           │
   │  Bilateral-channel custody contract holds Alice's & Bob's       │
   │  collateral. Final net state = single QUG transaction.          │
   │  Dispute window: 24h to publish a more-recent signed state.     │
   └─────────────────────────────────────────────────────────────────┘
```

### Component table

| DSM concept | Quillon Graph adaptation |
|---|---|
| `dBTC` (Bitcoin Signet custody) | QUG balance under a bilateral-channel custody contract |
| Relationship-local hash chain | Same — chain of signed state diffs, each hashing the previous |
| Storage nodes (3rd-party state archives) | Not needed initially — each wallet stores its own chain; can add later for backup |
| Bitcoin anchor txns | QUG smart contract anchor: open / close / dispute |
| `BleCoordinator.kt` dual-role advertiser+scanner | Same — Quillon BLE wallet alternates advertise/scan |
| GATT service UUID `<dsm specific>` | Define `qnk-ble-v1` UUID |
| Post-quantum signing | Dilithium5 (already in Q-NarwhalKnight) — reuse |

### Smart contract anchor (Quillon Graph side, sketch)

```rust
// crates/q-vm-runtime/src/contracts/bilateral_channel.rs   (new)
pub struct BilateralChannel {
    party_a: Address,
    party_b: Address,
    collateral_a: u128,                  // QUG escrowed by Alice on open
    collateral_b: u128,                  // QUG escrowed by Bob
    open_block_height: u64,
    dispute_window_blocks: u64,          // e.g., 14400 ≈ 24h at 1.6s/block
    latest_state: Option<SignedState>,   // most-recent on-chain claim
}

pub struct SignedState {
    chain_tip_hash: [u8; 32],            // last hash in the bilateral chain
    nonce: u64,                          // monotonic; reject older nonces
    alice_balance: u128,
    bob_balance: u128,                   // alice + bob == collateral_a + collateral_b
    sig_a: DilithiumSignature,           // Alice's sig over (chain_tip_hash, nonce, balances)
    sig_b: DilithiumSignature,           // Bob's sig over same
}

// Three entrypoints:
// open_channel(party_b, collateral)         -> escrows collateral_a
// settle_channel(SignedState)               -> starts dispute window
// dispute(SignedState_higher_nonce)         -> if challenger has newer nonce, slashes
// close_channel()                           -> after window, releases per latest_state
```

---

## 4. Phase A — Web Bluetooth API (browser)

### What Web Bluetooth gives us

Chrome / Edge desktop expose `navigator.bluetooth.requestDevice(...)` → returns a `BluetoothDevice` → `gatt.connect()` → read/write characteristics. **Pairing UX** is browser-native ("Pair this device" prompt).

### What Web Bluetooth does NOT give us

1. **No advertising.** Browsers can scan + connect, but cannot become a GATT *server*. So one of the two devices must be a non-browser (native app) advertising the service.
2. **No background operation.** The page must be in foreground.
3. **Foreground tab requirement** + user-gesture requirement for `requestDevice()`.

**Implication for Phase A:** the browser side is **the client (Alice)**. The peer must be either:
- A native dongle (USB BLE peripheral running custom firmware)
- A second laptop running a native helper app that advertises a GATT server
- A phone with the Expo app (Phase B device acting as peripheral)

For initial validation, the simplest path is: **Phase A target = browser-to-browser via a USB BLE peripheral co-located on one of the laptops.** Or wait for Phase B and validate browser-to-mobile first.

### Phase A deliverable

- `gui/quantum-wallet/src/ble/` — TypeScript module
  - `BleClient.ts` — wraps `navigator.bluetooth`
  - `bilateral_chain.ts` — TS port of `bilateral_transaction_manager` core logic (state hashing + Dilithium signing — can use `pqcrypto-js` or wasm-compiled version of our q-types Dilithium)
  - `channel_anchor.ts` — calls our QUG smart contract API
- React component `<BLEPayTab />` in `DexScreen.tsx` or new `<BLEScreen.tsx>`
  - "Scan for nearby wallets" → device picker
  - "Request payment" form
  - "Settle channel on-chain" action

---

## 5. Phase B — Expo mobile app

### What Expo offers

The Expo wallet (planned, not yet shipped) will use **react-native-ble-plx** or **expo-bluetooth-le** for native BLE. Unlike the browser, mobile apps can BOTH advertise and scan, so device-to-device works without a peripheral.

### Reuse from DSM

The Android Kotlin code in `dsm_client/android/.../bridge/ble/` is **directly reusable** if we choose to ship a native module rather than pure-TypeScript. Estimated 4 files / ~150K LOC of working code:

- `BleCoordinator.kt` — state machine for advertise/scan roles
- `GattClientSession.kt` — message framing, write-budget management
- `BleAdvertiser.kt`, `BleScanner.kt` — primitives
- `BlePermissionsGate.kt` — Android 12+ runtime permission flow

For iOS we'd need to implement an equivalent in Swift (no upstream equivalent in DSM since DSM is Android-first).

### Phase B deliverable

- Expo native module `quillon-ble` wrapping react-native-ble-plx
- Shared TS layer (`bilateral_chain.ts`, `channel_anchor.ts`) reused from Phase A
- iOS Swift native module — new code, ~2 weeks
- Android module — adapt DSM's `BleCoordinator.kt`, ~1 week

---

## 6. Risks + open questions

| Risk | Mitigation |
|---|---|
| Two consensus models in parallel (Quillon DAG-BFT + bilateral channel) — bridge code is where bugs hide | Smart contract anchor is the single source of truth; bilateral state only "wins" if dispute-window passes without challenge |
| Balance integrity (CLAUDE.md Rule 1: max-wins) — offline double-spend window | Collateral lock-up at open; dispute window allows the wronged party to publish the higher-nonce state |
| Range / Reliability | BLE 5.0 ≈ 10-50m practical. Acceptable for use cases (festival, conference). NOT for cross-continent. |
| iOS BLE peripheral quirks (background advertising disabled when locked) | Document the limitation; payments require foreground app |
| Web Bluetooth in Safari / Firefox | Phase A is Chrome/Edge only. Document. |
| Brandon's DSM is on Bitcoin Signet — does the bilateral logic actually depend on Bitcoin-specific primitives? | **Open question.** Need to read `bilateral_transaction_manager.rs` (78K LOC) and confirm the anchor is abstracted. If not, more work to port. |
| Lean4 proof effort to adapt to QUG primitives | **Open question.** Likely 1-2 weeks of Lean work to re-state for our types. Worth doing for the "we have formal proofs" marketing + actual safety. |
| Brandon's beta is `v0.1.0-beta.3` — is it production-quality? | "EARLY BETA RELEASE" per README. Not production-ready upstream. We'd lift the design + structure, not the actual binaries. |

---

## 7. Concrete next steps (when this picks back up)

### Step 0 — Spike (1-2 weeks)
1. Clone `deterministicstatemachine/dsm`. Build the Android wallet. Install on two phones. Run a real dBTC transfer between them via BLE. Verify the protocol works in practice.
2. Read `bilateral_transaction_manager.rs` end-to-end. Confirm Bitcoin coupling is abstracted (or note where it isn't).
3. Read the 7 Lean4 proofs. Catalog which theorems are reusable as-is vs need re-stating for QUG.

### Step 1 — Architecture decision doc
1. Smart contract anchor design (sketch in §3 above)
2. Bilateral chain data structure (port from DSM, swap dBTC → QUG)
3. Dispute resolution flow (DSM has this; copy and adapt)
4. Where the offline-state lives (per-device IndexedDB / SQLite)

### Step 2 — Reach out to Brandon
Email `info@irrefutablelabs.org`. He explicitly invited contributors. Offer:
- Code review of our adaptation (catches bugs early)
- Co-authorship on a paper if we extend the formal proofs to non-Bitcoin anchors
- Whatever's mutually useful

### Step 3 — Phase A prototype
Browser-only PoC. Two browsers + a USB BLE peripheral OR one browser + one Expo prototype.

### Step 4 — Phase B mobile
After Phase A validates the design end-to-end, port to Expo. Reuse the TS / WASM Dilithium layer; add native BLE bridges.

### Step 5 — Audit + mainnet
- Security audit of the anchor contract
- Soak test on testnet with hostile users (try to double-spend)
- Mainnet activation via `q-consensus-guard::Upgrade::BluetoothChannels` at a future height

---

## 8. Reading list

- DSM repository: https://github.com/deterministicstatemachine/dsm
- DSM project site: https://deterministicstatemachine.org/
- DSM onboarding walkthrough (11 chapters, 8 demos): https://www.deterministicstatemachine.org/onboarding.html
- Brandon Ramsay's metzdowd announcement (2025-12-09)
- Lightning Network BOLTs (for comparison — DSM removes the need for them but the dispute design is analogous)
- Web Bluetooth API spec: https://webbluetoothcg.github.io/web-bluetooth/
- react-native-ble-plx docs (Phase B mobile candidate)

---

## 9. Decision framework — when to actually do this

**Do it when:**
- v10.9.27+ sync stability is verified for ≥ 1 week on real mainnet
- The expo mobile app has shipped a v1 (basic send/receive online)
- There's a concrete user demand (operator, partner, conference deal)
- We have ≥ 1 engineer with 4-6 weeks of focus

**Don't do it when:**
- Mainnet sync is still flaky (i.e., right now)
- We're firefighting a different production issue
- We can't get Lean4 expertise for the proof port
- Brandon's DSM beta exposes a critical flaw that hasn't been fixed

---

**Last note:** Brandon's pitch line — *"There is no global ledger. There is no intermediary. There is no global witness set."* — describes *DSM's* primitive. Our integration ADDS a global ledger (Quillon Graph) as the **anchor of last resort**. That's the right move: pure DSM is great until you need dispute resolution against a malicious counterparty. The hybrid (relationship-local for speed, global-ledger for safety) is strictly better.
