# Technical Review: Leveraging Block Inc's Bitcoin Ecosystem for Q-NarwhalKnight

**Version**: 1.0 | **Date**: 2026-04-04 | **Status**: RFC (Request for Comments)
**Authors**: QNK Core Team | **Review Targets**: DeepSeek, Nemotron, QNK Contributors

---

## Executive Summary

This document evaluates the most impactful quick-win integrations between Q-NarwhalKnight (QNK) and Jack Dorsey's Block Inc Bitcoin ecosystem. We identify **five high-value, technically feasible integration paths** ranked by impact-to-effort ratio, with concrete implementation proposals grounded in QNK's existing Rust/libp2p/post-quantum architecture.

The central thesis: QNK already has 80% of the infrastructure needed to interoperate with Block's open-source Bitcoin stack. The missing 20% is protocol-level glue — and most of it is well-documented, Rust-native, and designed for exactly this kind of integration.

---

## BREAKING: Block Inc 2026 Developments (Critical Context)

The following developments from Q1 2026 dramatically change the integration landscape:

### 1. Square Auto-Enables Bitcoin Payments for 3M+ US Merchants (March 30, 2026)
- Bitcoin payments **enabled by default** for all eligible US Square sellers
- Lightning Network for instant settlement, **zero processing fees through 2026**
- Customers pay via QR code, merchants receive USD (instant conversion)
- **QNK Implication**: Square's 3M merchant network is now a potential QUG acceptance layer. If QNK implements Lightning bridge (Quick Win #4), any Square merchant could accept QUG via Lightning→BTC atomic swap, settled instantly to USD.

### 2. Block's "Goose" AI Agent — 90% of Code Submissions (2026)
- Block built "Goose", an open-source AI agent using Anthropic's **Model Context Protocol (MCP)**
- Now handles ~90% of Block's code submissions internally
- 27K+ GitHub stars, connects to 3,000+ MCP servers
- Donated to Linux Foundation's **Agentic AI Foundation (AAIF)** alongside Anthropic's MCP and OpenAI's AGENTS.md
- GitHub: `block/goose`
- **QNK Implication**: QNK could build an MCP server for its API, enabling Goose (and any MCP-compatible AI agent) to interact with QNK — deploy contracts, manage wallets, execute swaps, monitor nodes.

### 3. Block 40% Workforce Reduction → Bitcoin-Only Pivot (Early 2026)
- Headcount: 10,000+ → ~6,000 (40% cut)
- **TBD division wound down** — Web5/tbDEX transitioned to open-source community
- All resources concentrated on: Cash App Bitcoin, Bitkey, Proto mining, Spiral (BDK/LDK)
- **QNK Implication**: tbDEX is now community-maintained, not Block-backed. Integration is still viable (protocol is open-source) but QNK would be self-reliant. BDK/LDK remain actively maintained via Spiral.

### 4. Proto 3nm Bitcoin Mining ASIC (Production 2026)
- Block's Proto team completed a **3nm ASIC mining chip** tapeout
- First customer: Core Scientific (~15 EH/s hashrate)
- 1.5x better power per rack foot vs Bitmain 5nm
- Modular mining platform design
- **QNK Implication**: Proto's modular mining platform concept could inspire QNK miner hardware. The open-source mining system design is reference material for dedicated QUG mining devices.

### 5. SEC/CFTC Clarity — Self-Custody & Mining Hardware Are NOT Securities (March 2026)
- Joint SEC/CFTC interpretation: Bitcoin mining and self-custody hardware are not securities
- Directly validates Bitkey and Proto's business model
- **QNK Implication**: Regulatory green light for QNK to ship hardware wallets and mining devices without securities registration concerns.

### 6. Bitcoin Faucet Revival (April 6, 2026)
- Block relaunching Bitcoin faucet to drive adoption
- Free BTC distribution via CAPTCHA
- **QNK Implication**: QNK could partner or create a similar QUG faucet integrated with Block's ecosystem for cross-promotion.

---

## Table of Contents

1. [Architecture Compatibility Matrix](#1-architecture-compatibility-matrix)
2. [Quick Win #1: tbDEX Protocol for Fiat On/Off-Ramps](#2-quick-win-1-tbdex-protocol-for-fiat-onoff-ramps)
3. [Quick Win #2: DID:DHT Identity Layer on Existing Kademlia](#3-quick-win-2-diddht-identity-layer-on-existing-kademlia)
4. [Quick Win #3: BDK-Powered BTC Bridge Wallet](#4-quick-win-3-bdk-powered-btc-bridge-wallet)
5. [Quick Win #4: LDK Lightning Channels for Instant QUG/BTC](#5-quick-win-4-ldk-lightning-channels-for-instant-qugbtc)
6. [Quick Win #5: Bitkey Multisig Model for QNK Wallet Security](#6-quick-win-5-bitkey-multisig-model-for-qnk-wallet-security)
7. [Integration Priority Matrix](#7-integration-priority-matrix)
8. [Risk Analysis](#8-risk-analysis)
9. [Open Questions for Reviewer Models](#9-open-questions-for-reviewer-models)

---

## 1. Architecture Compatibility Matrix

Before diving into integrations, here is how QNK's existing stack maps to Block's:

| Layer | QNK Current | Block Equivalent | Compatibility |
|-------|-------------|------------------|---------------|
| **Language** | Rust (workspace, 15+ crates) | Rust (BDK, LDK, Bitkey firmware, Web5-rs, tbDEX-rs) | Native — zero FFI overhead |
| **P2P Transport** | libp2p 0.56 (TCP, QUIC, WebSocket, WebRTC) | libp2p (LDK uses custom, BDK uses Electrum/Esplora) | Shared stack, gossipsub compatible |
| **DHT** | Kademlia (peer discovery, height proofs) | Mainline DHT (did:dht identity resolution) | Same algorithm family, bridgeable |
| **Crypto — Classical** | Ed25519 (Phase Q0) | Ed25519 (Bitcoin Schnorr is related) | Direct interop |
| **Crypto — Post-Quantum** | Dilithium5, Kyber1024, SPHINCS+ | None (Bitcoin has no PQ yet) | QNK advantage — can offer PQ wrapper |
| **Serialization** | MessagePack, Postcard, Bincode | JSON (tbDEX), CBOR (DIDs), Bitcoin Script | Adapter layer needed |
| **DEX Model** | AMM pools (Uniswap-style) + order book via gossipsub | tbDEX (RFQ/Quote message protocol, no on-chain AMM) | Complementary — AMM for on-chain, tbDEX for fiat |
| **Bridge Tokens** | wBTC, wETH, wZEC, wIRON (mint/burn custody model) | BDK (native BTC), LDK (Lightning BTC) | Replace custodial bridge with trustless BDK/LDK |
| **Identity** | `qnk` + 64-char hex, X-Wallet-Auth headers | DIDs (did:dht, did:web, did:jwk), Verifiable Credentials | DID layer can wrap QNK addresses |
| **Wallet** | Slint desktop + web wallet | Bitkey (2-of-3 multisig), Cash App | Multisig model directly applicable |

**Key Insight**: Both ecosystems are Rust-native with libp2p DNA. Integration is library-level linking, not cross-language bridging.

---

## 2. Quick Win #1: tbDEX Protocol for Fiat On/Off-Ramps

### Impact: CRITICAL | Effort: MEDIUM | Timeline: 4-6 weeks

### Problem Statement

QNK currently has no native fiat on/off-ramp. Users must find QUG through the DEX (QUG/QUGUSD pool) or mine it. There is no way for a new user to go from USD/EUR → QUG without an intermediary.

### Solution: Implement tbDEX PFI (Participating Financial Institution) Node

tbDEX is a message-based protocol for discovering liquidity and exchanging assets. QNK would become a **PFI** — a liquidity provider that quotes and settles QUG/fiat trades.

### Protocol Flow

```
┌──────────────┐                    ┌──────────────────┐
│  User Wallet │                    │  QNK PFI Node    │
│  (any tbDEX  │                    │  (q-api-server)  │
│   client)    │                    │                  │
└──────┬───────┘                    └────────┬─────────┘
       │                                     │
       │  1. GET /offerings                  │
       │────────────────────────────────────>│  Returns: QUG/USD, QUG/EUR
       │                                     │  with exchange rate from
       │  2. POST /exchanges (RFQ)           │  QUG/QUGUSD pool price
       │────────────────────────────────────>│
       │                                     │  Validate DID, check KYC VC
       │  3. Quote (with expiry)             │
       │<────────────────────────────────────│  Price locked for 60s
       │                                     │
       │  4. Order (accept quote)            │
       │────────────────────────────────────>│
       │                                     │  Lock QUG from reserve pool
       │  5. OrderStatus: PAYIN_PENDING      │
       │<────────────────────────────────────│  Waiting for fiat payment
       │                                     │
       │  [User sends fiat via bank/card]    │
       │                                     │
       │  6. OrderStatus: PAYOUT_SETTLED     │
       │<────────────────────────────────────│  QUG sent to user's QNK address
       │                                     │
       │  7. Close                           │
       │<────────────────────────────────────│  Exchange complete
       └─────────────────────────────────────┘
```

### Implementation Architecture

```rust
// New crate: crates/q-tbdex/
// Dependencies: tbdex-rs (Block's Rust SDK), q-dex, q-storage, q-types

/// QNK acts as a Participating Financial Institution (PFI)
pub struct QnkPfiNode {
    /// DID for this PFI (did:dht resolved via our existing Kademlia)
    pub did: DidDht,
    
    /// Offerings: what this PFI can trade
    /// Derived from active QUG/QUGUSD pool + bridge token pools
    pub offerings: Vec<Offering>,
    
    /// Connection to QNK's DEX engine for real-time pricing
    pub dex_pricer: Arc<DexPricer>,
    
    /// Settlement engine — executes QUG transfers on-chain
    pub settler: Arc<QugSettler>,
    
    /// KYC/compliance via Verifiable Credentials
    pub vc_verifier: Arc<VcVerifier>,
}

/// Offering: "I will sell QUG for USD at market rate + 0.5% spread"
pub struct QugOffering {
    pub payin_currency: Currency,     // USD, EUR, GBP
    pub payout_currency: String,      // "QUG"
    pub rate_source: RateSource,      // QUG/QUGUSD AMM pool mid-price
    pub spread_bps: u32,              // 50 = 0.5%
    pub min_amount: u64,              // Minimum $10
    pub max_amount: u64,              // Maximum $10,000
    pub payin_methods: Vec<PayinMethod>, // Bank transfer, card
    pub required_credentials: Vec<String>, // KYC VCs required
}
```

### Why This Works for QNK

1. **QNK already has the QUG/QUGUSD AMM pool** — provides real-time pricing oracle
2. **QNK already has bridge tokens** (wBTC, wETH) — can extend to fiat bridges
3. **tbDEX-rs is Rust** — native integration into q-api-server
4. **No on-chain changes needed** — tbDEX is off-chain messaging, settlement uses existing QUG transfer
5. **DID identity maps to qnk addresses** — `did:dht` key → `qnk` address derivation

### New API Endpoints

```
GET  /api/v1/tbdex/offerings              List available QUG/fiat pairs
POST /api/v1/tbdex/exchanges              Create exchange (submit RFQ)
GET  /api/v1/tbdex/exchanges/:id          Get exchange status
PUT  /api/v1/tbdex/exchanges/:id/order    Accept quote → create order
GET  /api/v1/tbdex/exchanges/:id/status   Settlement status
POST /api/v1/tbdex/exchanges/:id/close    Close exchange
```

### Pricing Engine Integration

```rust
/// Real-time QUG/USD price from the on-chain AMM
fn get_qug_usd_price(&self) -> Decimal {
    // Pool: pool-qug-qugusd-bootstrap
    // reserve0 (QUG): 73,941.99 QUG
    // reserve1 (QUGUSD): 213,555,663.97 QUGUSD
    // Price = reserve1/reserve0 = ~2,888.05 QUGUSD per QUG
    let pool = self.dex.get_pool("pool-qug-qugusd-bootstrap");
    pool.reserve1 / pool.reserve0  // QUGUSD/QUG mid-price
}

/// tbDEX quote with spread
fn generate_quote(&self, rfq: &Rfq) -> Quote {
    let mid_price = self.get_qug_usd_price();
    let spread = mid_price * Decimal::from(self.spread_bps) / Decimal::from(10000);
    let offer_price = if rfq.is_buy() {
        mid_price + spread  // User buying QUG → pay more
    } else {
        mid_price - spread  // User selling QUG → receive less
    };
    Quote {
        payin_amount: rfq.amount * offer_price,
        payout_amount: rfq.amount,
        expires_at: Utc::now() + Duration::seconds(60),
    }
}
```

### Compliance & KYC

tbDEX uses **Verifiable Credentials** for compliance. QNK's existing auth system (X-Wallet-Auth with Dilithium5 signatures) can issue VCs:

```rust
/// Issue a KYC Verifiable Credential after identity verification
/// This VC is reusable across all tbDEX PFIs
fn issue_kyc_vc(&self, wallet: &QnkAddress, identity_proof: &IdentityProof) -> VerifiableCredential {
    VerifiableCredential {
        context: vec!["https://www.w3.org/2018/credentials/v1"],
        type_: vec!["VerifiableCredential", "KYCCredential"],
        issuer: self.did.to_string(),  // QNK PFI's DID
        credential_subject: CredentialSubject {
            id: format!("did:dht:{}", wallet.to_hex()),
            kyc_level: "basic",  // or "enhanced" for higher limits
        },
        proof: self.sign_vc_dilithium5(wallet),  // Post-quantum signed!
    }
}
```

### Competitive Advantage

QNK's tbDEX PFI would be the **first post-quantum-secured fiat on-ramp** in the tbDEX network. Every other PFI uses classical Ed25519/secp256k1. QNK offers Dilithium5-signed VCs and AEGIS-QL authenticated settlements.

---

## 3. Quick Win #2: DID:DHT Identity Layer on Existing Kademlia

### Impact: HIGH | Effort: LOW | Timeline: 2-3 weeks

### Problem Statement

QNK uses raw hex addresses (`qnk1a2b3c...`) with X-Wallet-Auth header authentication. This works but:
- No interoperability with external identity systems
- No credential portability (KYC done once, used nowhere else)
- No human-readable identity resolution

### Solution: Map QNK Addresses to DID:DHT on Existing Kademlia

`did:dht` stores Decentralized Identifiers on the Mainline DHT (same Kademlia algorithm QNK already runs). Since QNK already has a Kademlia DHT for peer discovery, we can **dual-purpose it** for identity resolution.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                  QNK Kademlia DHT                   │
│                                                     │
│  Existing Keys:                                     │
│    /qnk/peers/{peer_id}  → PeerInfo (IP, port)     │
│    /qnk/heights/{peer_id} → HeightProof            │
│                                                     │
│  NEW Keys (DID:DHT):                                │
│    /did/dht/{z-base-32-key} → DID Document (DNS)    │
│    /qnk/did/{qnk_address}  → DID:DHT URI           │
│                                                     │
│  Resolution Flow:                                   │
│    qnk_address → DHT lookup → DID Document →        │
│    public keys, service endpoints, VCs               │
└─────────────────────────────────────────────────────┘
```

### DID Document for a QNK Wallet

```json
{
  "@context": ["https://www.w3.org/ns/did/v1"],
  "id": "did:dht:qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723",
  "verificationMethod": [
    {
      "id": "#key-0",
      "type": "Ed25519VerificationKey2020",
      "publicKeyMultibase": "z6Mk..."
    },
    {
      "id": "#key-1",
      "type": "Dilithium5VerificationKey2024",
      "publicKeyMultibase": "zDi5..."
    }
  ],
  "authentication": ["#key-0", "#key-1"],
  "service": [
    {
      "id": "#qnk-wallet",
      "type": "QNKWallet",
      "serviceEndpoint": "qnk://qnkefca1e8c1f46e91..."
    },
    {
      "id": "#tbdex-pfi",
      "type": "PFI",
      "serviceEndpoint": "https://quillon.xyz/api/v1/tbdex"
    }
  ]
}
```

### Implementation

```rust
// Extend crates/q-network/src/peer_discovery.rs

use did_dht::{DidDht, DidDocument};

impl UnifiedNetworkManager {
    /// Register a QNK wallet as a DID:DHT identity on the existing Kademlia DHT
    pub async fn register_did(&self, wallet: &QnkAddress, keypair: &Keypair) -> Result<DidDht> {
        let did_doc = DidDocument::builder()
            .id(format!("did:dht:{}", wallet.to_hex()))
            .verification_method_ed25519(keypair.ed25519_public())
            .verification_method_dilithium5(keypair.dilithium5_public())
            .service("QNKWallet", format!("qnk://{}", wallet.to_hex()))
            .build()?;
        
        // Encode as DNS packet (did:dht spec uses DNS TXT records in DHT values)
        let dns_packet = did_doc.to_dns_packet()?;
        
        // Store on QNK's existing Kademlia DHT
        let key = RecordKey::new(&format!("/did/dht/{}", wallet.to_z_base_32()));
        self.kademlia.put_record(Record::new(key, dns_packet), Quorum::One)?;
        
        Ok(did_doc.did())
    }
    
    /// Resolve a DID:DHT to its DID Document via Kademlia lookup
    pub async fn resolve_did(&self, did: &str) -> Result<DidDocument> {
        let key = RecordKey::new(&format!("/did/dht/{}", did.strip_prefix("did:dht:")?));
        let record = self.kademlia.get_record(key).await?;
        DidDocument::from_dns_packet(&record.value)
    }
}
```

### Why This Is a Quick Win

1. **Zero new infrastructure** — reuses existing Kademlia DHT
2. **did:dht spec is simple** — DID Documents encoded as DNS TXT records in DHT values
3. **Enables tbDEX** — tbDEX requires DIDs; this unlocks Quick Win #1
4. **Enables Verifiable Credentials** — KYC, reputation, mining proofs become portable
5. **Human-readable aliases** — `did:dht:alice` → `qnk1a2b...` via DHT resolution
6. **Post-quantum identity** — DID Documents include Dilithium5 keys (unique to QNK)

---

## 4. Quick Win #3: BDK-Powered BTC Bridge Wallet

### Impact: HIGH | Effort: MEDIUM | Timeline: 3-4 weeks

### Problem Statement

QNK's current wBTC bridge uses a custodial mint/burn model (`bridge_tokens.rs`). A user deposits BTC into a bridge address, and wrapped tokens are minted. This requires trust in the bridge operator.

### Solution: Replace Custodial Bridge with BDK-Powered Trustless HTLC

BDK (Bitcoin Development Kit) is a Rust library for building Bitcoin wallets. It supports descriptors, PSBTs, and Miniscript — everything needed for trustless atomic swaps.

### Current vs Proposed Architecture

```
CURRENT (Custodial):
  User BTC ──deposit──> Bridge Custody ──mint──> wBTC on QNK
  Trust assumption: Bridge operator is honest

PROPOSED (Trustless HTLC via BDK):
  User BTC ──HTLC lock──> Bitcoin Script (time-locked, hash-locked)
                              │
  QNK QUG ──HTLC lock──> QNK Contract (same hash, shorter timelock)
                              │
  User reveals secret ──> Claims QUG on QNK
  Counterparty uses secret ──> Claims BTC on Bitcoin
  
  Trust assumption: Cryptographic (hash preimage security)
```

### Implementation

```rust
// New crate: crates/q-btc-bridge/
// Dependencies: bdk = "1.0", bdk_electrum, bdk_esplora

use bdk_wallet::{Wallet, KeychainKind, SignOptions};
use bdk_electrum::ElectrumClient;

/// BDK-powered Bitcoin bridge for trustless QUG/BTC atomic swaps
pub struct BdkBtcBridge {
    /// BDK wallet instance with HTLC-capable descriptors
    wallet: Wallet,
    
    /// Electrum backend for Bitcoin chain queries
    electrum: ElectrumClient,
    
    /// QNK storage for tracking swap state
    storage: Arc<StorageEngine>,
    
    /// HTLC parameters
    btc_timelock_blocks: u32,    // 144 blocks (~24 hours)
    qnk_timelock_seconds: u64,  // 43200 seconds (12 hours, shorter!)
}

/// Atomic swap HTLC using BDK Miniscript
/// 
/// Bitcoin Script (Miniscript descriptor):
///   wsh(or_d(
///     and_v(v:pk(user_pubkey), sha256(secret_hash)),     // User claims with secret
///     and_v(v:pk(refund_pubkey), older(144))              // Refund after 144 blocks
///   ))
fn create_btc_htlc_descriptor(
    user_pubkey: &PublicKey,
    refund_pubkey: &PublicKey,
    secret_hash: &[u8; 32],
    timelock_blocks: u32,
) -> String {
    format!(
        "wsh(or_d(and_v(v:pk({}),sha256({})),and_v(v:pk({}),older({}))))",
        user_pubkey, hex::encode(secret_hash), refund_pubkey, timelock_blocks
    )
}

impl BdkBtcBridge {
    /// Initiate atomic swap: lock BTC in HTLC, wait for QUG lock on QNK side
    pub async fn initiate_swap(&self, params: SwapParams) -> Result<SwapState> {
        // 1. Generate cryptographic secret (32 bytes from QRNG if available)
        let secret = generate_secret();
        let secret_hash = sha256(&secret);
        
        // 2. Create BDK wallet with HTLC descriptor
        let htlc_desc = create_btc_htlc_descriptor(
            &params.user_btc_pubkey,
            &params.refund_pubkey,
            &secret_hash,
            self.btc_timelock_blocks,
        );
        
        // 3. Build PSBT (Partially Signed Bitcoin Transaction)
        let mut psbt = self.wallet.build_tx()
            .add_recipient(htlc_desc.script_pubkey(), params.btc_amount)
            .fee_rate(FeeRate::from_sat_per_vb(10))
            .finish()?;
        
        // 4. Sign and broadcast BTC HTLC
        self.wallet.sign(&mut psbt, SignOptions::default())?;
        let tx = psbt.extract_tx()?;
        self.electrum.broadcast(&tx)?;
        
        // 5. Lock QUG on QNK side with matching hash (shorter timelock!)
        let qnk_lock = QnkHtlcLock {
            amount_qug: params.qug_amount,
            secret_hash,
            timelock_seconds: self.qnk_timelock_seconds,
            recipient: params.user_qnk_address,
        };
        self.storage.create_htlc_lock(&qnk_lock)?;
        
        Ok(SwapState::BtcLocked { btc_txid: tx.txid(), secret_hash })
    }
}
```

### BDK Integration Points

| BDK Component | QNK Usage | Benefit |
|---|---|---|
| `bdk_wallet` | HTLC wallet management | Descriptor-based, no raw script handling |
| `bdk_electrum` | Bitcoin chain monitoring | Watch for HTLC claims/refunds |
| `bdk_esplora` | Alternative backend | Fallback if Electrum down |
| Miniscript | HTLC spending conditions | Formally verifiable, composable |
| PSBT | Transaction construction | Multi-party signing, hardware wallet support |

### Why BDK Over Raw Bitcoin Libraries

1. **Miniscript** — formally verifiable spending conditions, no hand-rolled Script
2. **PSBT workflow** — enables Bitkey hardware wallet signing of bridge transactions
3. **Descriptor wallets** — modern standard, compatible with all BIP-compliant wallets
4. **Chain backends** — Electrum or Esplora, no need to run a full Bitcoin node
5. **Maintained by Spiral** — active development, security audits, battle-tested

---

## 5. Quick Win #4: LDK Lightning Channels for Instant QUG/BTC

### Impact: VERY HIGH | Effort: HIGH | Timeline: 8-12 weeks

### Problem Statement

Atomic swaps via HTLC (Quick Win #3) require on-chain Bitcoin transactions — slow (10-60 min confirmation) and expensive ($1-20 fees). For sub-$100 trades, this is impractical.

### Solution: LDK Payment Channels for Instant, Near-Free QUG/BTC Exchange

LDK (Lightning Development Kit) is an embeddable Rust Lightning implementation. Unlike LND (Go daemon), LDK is a library that integrates directly into QNK's Tokio runtime.

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    QNK Node (q-api-server)                     │
│                                                                │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────┐  │
│  │  QNK Engine   │   │  LDK Node    │   │  Submarine Swap   │  │
│  │  (blocks,     │   │  (Lightning  │   │  Engine           │  │
│  │   balances,   │◄─►│   channels,  │◄─►│  (QUG ↔ LN-BTC)  │  │
│  │   DEX, P2P)   │   │   invoices)  │   │                   │  │
│  └──────────────┘   └──────────────┘   └───────────────────┘  │
│         │                   │                    │              │
│         ▼                   ▼                    ▼              │
│    QNK gossipsub      Lightning P2P       Swap Coordination    │
│    (blocks, DEX)      (BOLT messages)     (hash-locked both)   │
└────────────────────────────────────────────────────────────────┘
```

### Submarine Swap: QUG ↔ Lightning BTC

A submarine swap exchanges an on-chain asset (QUG) for a Lightning payment (BTC) atomically:

```
1. User wants to buy 1 QUG with Lightning BTC
2. QNK generates invoice: "Pay 0.00035 BTC via Lightning"
3. QNK creates hash-locked QUG hold: "1 QUG locked, released when LN payment settles"
4. User pays Lightning invoice → LDK receives BTC
5. LDK payment preimage reveals secret → QUG released to user
6. Atomic: either both happen or neither happens

Latency: ~3 seconds (Lightning hop + QNK block)
Fee: ~10 sats Lightning + 0 QUG network fee
```

### LDK Integration

```rust
// Extend crates/q-btc-bridge/ with LDK
// Dependencies: lightning = "0.0.125", lightning-net-tokio, lightning-persister

use lightning::ln::channelmanager::ChannelManager;
use lightning::ln::peer_handler::PeerManager;
use lightning::chain::chaininterface::BroadcasterInterface;

/// LDK Lightning node embedded in QNK
pub struct QnkLightningNode {
    channel_manager: Arc<ChannelManager>,
    peer_manager: Arc<PeerManager>,
    
    /// QNK-side swap engine
    swap_engine: Arc<SubmarineSwapEngine>,
    
    /// Persister backed by QNK's RocksDB
    persister: Arc<RocksDbLdkPersister>,
}

/// Submarine swap: Lightning BTC → QUG
pub struct SubmarineSwapEngine {
    /// Pending swaps indexed by payment hash
    pending: DashMap<PaymentHash, PendingSwap>,
    
    /// QNK storage for HTLC locks
    storage: Arc<StorageEngine>,
}

impl SubmarineSwapEngine {
    /// User wants to buy QUG with Lightning BTC
    pub async fn create_buy_qug_swap(&self, amount_qug: u128) -> Result<SwapOffer> {
        // 1. Price QUG in BTC via AMM: QUG/QUGUSD * QUGUSD/wBTC
        let qug_btc_price = self.get_qug_btc_price();
        let btc_amount_sats = (amount_qug as f64 * qug_btc_price * 1e8) as u64;
        
        // 2. Generate Lightning invoice via LDK
        let (payment_hash, payment_secret) = self.channel_manager
            .create_inbound_payment(Some(btc_amount_sats * 1000), 3600, None)?;
        
        // 3. Lock QUG in hash-locked hold (released when LN payment settles)
        let hold = QugHashLock {
            amount: amount_qug,
            payment_hash,
            expires_at: Utc::now() + Duration::hours(1),
            recipient: swap_request.user_qnk_address,
        };
        self.storage.create_hash_lock(&hold)?;
        
        // 4. Return Lightning invoice to user
        Ok(SwapOffer {
            invoice: self.create_bolt11_invoice(btc_amount_sats, payment_hash)?,
            qug_amount: amount_qug,
            expires_in_seconds: 3600,
        })
    }
    
    /// Called by LDK when Lightning payment is received
    fn on_payment_received(&self, payment_hash: PaymentHash, preimage: PaymentPreimage) {
        // Payment settled → release QUG to user
        if let Some(hold) = self.pending.remove(&payment_hash) {
            self.storage.release_hash_lock(&payment_hash, &hold.recipient);
            // QUG now in user's balance, BTC in our Lightning channel
        }
    }
}
```

### Performance Comparison

| Method | Latency | Fee (for $50 trade) | Trust Model |
|--------|---------|---------------------|-------------|
| Current wBTC bridge | Minutes (manual) | 0 QNK fee | Custodial trust |
| BDK atomic swap (QW#3) | 10-60 min (BTC conf) | $1-20 BTC fee | Trustless (HTLC) |
| **LDK submarine swap** | **~3 seconds** | **~$0.01** | **Trustless (hash-lock)** |

---

## 6. Quick Win #5: Bitkey Multisig Model for QNK Wallet Security

### Impact: HIGH | Effort: LOW-MEDIUM | Timeline: 2-3 weeks

### Problem Statement

QNK wallets are single-key (Ed25519 or Dilithium5). If the private key is compromised or lost, funds are gone. The Slint wallet stores keys locally with no recovery mechanism beyond seed backup.

### Solution: Adapt Bitkey's 2-of-3 Multisig Architecture

Bitkey uses 3 keys where any 2 can authorize a transaction:

```
┌─────────────────────────────────────────────────────┐
│               QNK 2-of-3 Multisig                   │
│                                                     │
│  Key 1: MOBILE (Slint Wallet)                       │
│    ├── Ed25519 + Dilithium5 keypair                 │
│    ├── Stored encrypted on device                   │
│    └── Used for daily transactions                  │
│                                                     │
│  Key 2: HARDWARE (NFC device or air-gapped)         │
│    ├── Ed25519 + Dilithium5 keypair                 │
│    ├── Never leaves hardware                        │
│    └── Required for large transactions (>threshold) │
│                                                     │
│  Key 3: RECOVERY (Server-held, time-delayed)        │
│    ├── Ed25519 + Dilithium5 keypair                 │
│    ├── Held by QNK bootstrap server (encrypted)     │
│    ├── 48-hour delay before signing                 │
│    └── User can cancel during delay window          │
│                                                     │
│  Authorization:                                     │
│    Mobile + Hardware  = Instant (daily use)          │
│    Mobile + Recovery  = 48h delay (lost hardware)    │
│    Hardware + Recovery = 48h delay (lost phone)      │
└─────────────────────────────────────────────────────┘
```

### Implementation Using QNK's Existing Crypto

```rust
// Extend crates/q-types/ with multisig support

/// 2-of-3 multisig wallet using QNK's existing crypto-agile framework
pub struct MultisigWallet {
    /// The multisig address (derived from all 3 public keys)
    pub address: [u8; 32],
    
    /// Three key slots
    pub keys: [MultisigKey; 3],
    
    /// Threshold: 2 of 3
    pub threshold: u8,
    
    /// Spending policy
    pub policy: SpendingPolicy,
}

pub struct MultisigKey {
    pub role: KeyRole,  // Mobile, Hardware, Recovery
    pub ed25519_pubkey: [u8; 32],
    pub dilithium5_pubkey: Vec<u8>,  // ~2.5 KB
    pub weight: u8,  // 1 each, threshold = 2
}

pub struct SpendingPolicy {
    /// Transactions below this amount: Mobile key alone (convenience)
    pub single_sig_limit: u128,  // e.g., 10 QUG
    
    /// Transactions above limit: require 2-of-3
    pub multisig_threshold_amount: u128,
    
    /// Recovery key delay (prevents immediate theft if server compromised)
    pub recovery_delay_seconds: u64,  // 172800 = 48 hours
}

/// Verify a 2-of-3 multisig transaction
pub fn verify_multisig_tx(
    tx: &Transaction,
    wallet: &MultisigWallet,
    signatures: &[MultisigSignature],
) -> Result<bool> {
    // Must have at least 2 valid signatures from different keys
    let mut valid_keys = HashSet::new();
    
    for sig in signatures {
        let key = &wallet.keys[sig.key_index as usize];
        let valid = match sig.scheme {
            AuthScheme::Ed25519 => verify_ed25519(&key.ed25519_pubkey, &tx.hash(), &sig.signature),
            AuthScheme::Dilithium5 => verify_dilithium5(&key.dilithium5_pubkey, &tx.hash(), &sig.signature),
            AuthScheme::Hybrid => {
                verify_ed25519(&key.ed25519_pubkey, &tx.hash(), &sig.ed25519_sig)
                && verify_dilithium5(&key.dilithium5_pubkey, &tx.hash(), &sig.dilithium5_sig)
            }
        };
        if valid {
            valid_keys.insert(sig.key_index);
        }
    }
    
    Ok(valid_keys.len() >= wallet.threshold as usize)
}
```

### Post-Quantum Advantage Over Bitkey

Bitkey uses secp256k1 (Bitcoin's curve). QNK's multisig would use **Dilithium5 + Ed25519 hybrid**, making it the first consumer multisig wallet with post-quantum security on all 3 keys.

---

## 7. Quick Win #6: MCP Server for AI Agent Interoperability (Goose/Claude)

### Impact: HIGH | Effort: LOW | Timeline: 1-2 weeks

### Problem Statement

QNK's API is REST-only. The emerging standard for AI agent interoperability is the **Model Context Protocol (MCP)**, co-developed by Anthropic and Block. Block's Goose agent (27K GitHub stars) and Claude Code both use MCP. Without an MCP server, QNK is invisible to the fastest-growing developer tooling ecosystem.

### Solution: Build an MCP Server Exposing QNK Operations

```rust
// New crate: crates/q-mcp-server/
// Exposes QNK operations as MCP tools for AI agents

/// MCP Tool: Get wallet balance
#[mcp_tool(name = "qnk_get_balance", description = "Get QUG balance for a wallet address")]
async fn get_balance(address: String) -> McpResult {
    let balance = storage.get_balance(&parse_qnk_address(&address)?)?;
    McpResult::json(json!({ "address": address, "balance_qug": format_qug(balance) }))
}

/// MCP Tool: Execute DEX swap
#[mcp_tool(name = "qnk_swap", description = "Swap tokens on QNK DEX")]
async fn swap(from_token: String, to_token: String, amount: String, wallet: String) -> McpResult {
    // ... execute swap via existing DEX engine
}

/// MCP Tool: Deploy smart contract
#[mcp_tool(name = "qnk_deploy_contract", description = "Deploy a token contract on QNK")]
async fn deploy_contract(name: String, symbol: String, supply: String, wallet: String) -> McpResult {
    // ... deploy via existing contract engine
}

/// MCP Tool: Get network status
#[mcp_tool(name = "qnk_status", description = "Get QNK network status, height, peers")]
async fn network_status() -> McpResult {
    // ... return from /api/v1/status
}

/// MCP Tool: Get DEX pools
#[mcp_tool(name = "qnk_dex_pools", description = "List all DEX liquidity pools with reserves")]
async fn dex_pools() -> McpResult {
    // ... return from /api/v1/dex/pools
}

/// MCP Resource: Live block stream
#[mcp_resource(uri = "qnk://blocks/stream", description = "Real-time block stream via SSE")]
async fn block_stream() -> McpResourceStream {
    // ... bridge existing SSE to MCP resource stream
}
```

### Why This Is the Fastest Win

1. **MCP spec is simple** — JSON-RPC over stdio/HTTP, well-documented
2. **QNK already has all the endpoints** — MCP tools just wrap existing REST API
3. **Goose has 27K stars** — instant developer reach
4. **Claude Code uses MCP** — QNK becomes accessible from this very tool
5. **Block donated MCP to Linux Foundation AAIF** — industry standard, not proprietary
6. **1-2 weeks** because it's a thin adapter layer over existing functionality

### Developer Experience After MCP

```bash
# Any AI agent with MCP support can now interact with QNK:
goose "Check the QNK network status and list all DEX pools"
goose "Deploy a token called ROCKET with 1B supply on QNK"
goose "Swap 10 QUG for QUGUSD on the DEX"

# Claude Code with QNK MCP server:
claude "Monitor QNK block production and alert if blocks stop"
```

---

## 8. Integration Priority Matrix (UPDATED with 2026 Intelligence)

| # | Integration | Impact | Effort | Dependencies | Timeline | ROI Score |
|---|---|---|---|---|---|---|
| **6** | **MCP Server (Goose/Claude)** | High | **Very Low** | None | **1-2 wk** | **9.8/10** |
| **2** | **DID:DHT Identity** | High | Low | None (uses existing DHT) | 2-3 wk | **9.0/10** |
| **5** | **Multisig Wallets** | High | Low-Med | None | 2-3 wk | **8.5/10** |
| **3** | **BDK BTC Bridge** | High | Medium | None | 3-4 wk | **8.0/10** |
| **1** | **tbDEX Fiat Ramp** | Critical | Medium | Needs DID:DHT (#2) | 4-6 wk | **8.0/10** |
| **4** | **LDK Lightning→Square** | **VERY HIGH** | High | Needs BDK (#3) | 8-12 wk | **9.5/10** |

**NOTE on tbDEX**: TBD (Block's identity division) was **wound down in early 2026**. The tbDEX protocol is now community-maintained open source. Integration is still viable and the protocol is sound, but QNK would be self-reliant for maintenance. ROI adjusted from 9.5 to 8.0 due to reduced ecosystem momentum.

**NOTE on LDK Lightning**: ROI **upgraded from 7.5 to 9.5** because Square auto-enabled Bitcoin Lightning payments for 3M+ US merchants in March 2026. LDK Lightning bridge now means QUG holders can pay at ANY Square merchant via Lightning→BTC atomic swap. This is the single highest-impact long-term integration.

### Recommended Execution Order (REVISED)

```
Week 1-2:   MCP Server (#6)                                   [fastest win, immediate developer reach]
Week 2-4:   DID:DHT Identity (#2) + Multisig Wallets (#5)     [parallel]
Week 4-8:   BDK BTC Bridge (#3) + tbDEX Fiat Ramp (#1)        [parallel]
Week 8-14:  LDK Lightning → Square Merchant Network (#4)       [the big prize]
```

### The Square Lightning Endgame

```
┌──────────────┐     Lightning      ┌──────────────┐     Square POS     ┌──────────────┐
│  QNK Wallet  │────(LDK atomic)───>│  BTC/LN      │────(auto-convert)─>│  3M+ US      │
│  (QUG holder)│     swap           │  Network      │     to USD         │  Merchants   │
│              │     ~3 seconds     │              │     ~instant        │              │
│  Pays 100    │     ~$0.01 fee     │  QUG→BTC     │     zero fee       │  Receives    │
│  QUG         │                    │  via LDK     │     (thru 2026)    │  $289,000    │
└──────────────┘                    └──────────────┘                    └──────────────┘
```

This is the killer use case: **QUG becomes spendable at 3 million US businesses** through a trustless Lightning bridge, with zero merchant integration required (Square already handles it).

---

## 9. Risk Analysis

### Technical Risks

| Risk | Severity | Mitigation |
|---|---|---|
| tbDEX ecosystem — TBD division **wound down** (early 2026) | Medium-High | Protocol is open-source and community-maintained; QNK self-hosts |
| BDK/LDK version churn (pre-1.0 LDK) | Low | Pin versions, BDK 1.0 is stable |
| Kademlia DHT dual-use (peers + DIDs) could cause routing table bloat | Low | Separate DHT namespace prefixes, TTL on DID records |
| Lightning channel liquidity management is operationally complex | High | Start with single well-capitalized channel to LSP |
| Multisig increases transaction size (2x signatures) | Low | Only for high-value transactions; PQ sigs already large |
| Regulatory risk on fiat ramp (tbDEX PFI = money transmitter?) | High | Start with crypto-to-crypto only; add fiat after legal review |

### Security Risks

| Risk | Severity | Mitigation |
|---|---|---|
| HTLC timelock mismatch (BTC confirms slower than expected) | High | Conservative timelocks: BTC 144 blocks, QNK 12 hours |
| Lightning channel force-close during swap | Medium | Submarine swap protocol handles this (HTLC on both sides) |
| Recovery key server compromise (multisig) | Medium | 48-hour delay + user notification + encrypted at rest |
| DID:DHT poisoning (malicious DID Documents) | Low | Verify DID Document signatures, cache with TTL |

---

## 10. Open Questions for Reviewer Models

### For DeepSeek

1. **tbDEX Message Optimization**: The tbDEX protocol uses JSON-LD for message encoding. QNK uses MessagePack/Postcard internally. What is the optimal serialization strategy for a bridge — convert at the API boundary, or carry JSON-LD through the internal pipeline?

2. **DID:DHT vs DID:QNK**: Should QNK create its own DID method (`did:qnk`) that natively understands QNK addresses and post-quantum keys, or is adapting `did:dht` with PQ extensions the better path? Trade-offs between ecosystem compatibility and native optimization.

3. **AMM Price Oracle for tbDEX**: The QUG/QUGUSD AMM pool is the only on-chain price source. How should we handle price manipulation risk when using this as the quote source for tbDEX trades? TWAP? Multi-block average? External oracle cross-reference?

4. **Lightning Liquidity Bootstrap**: LDK requires pre-funded payment channels. What is the optimal strategy for bootstrapping Lightning liquidity for a new chain's bridge? LSP partnership? Dual-funded channels? Liquidity ads?

### For Nemotron

1. **Post-Quantum HTLC Security**: QNK's atomic swaps would use Dilithium5 on the QNK side but secp256k1/Schnorr on the Bitcoin side. In a quantum threat model, the Bitcoin HTLC is the weak link. Is there a practical way to add PQ protection to the Bitcoin side of the swap without Bitcoin protocol changes?

2. **Multisig Key Derivation**: Bitkey derives all 3 keys from a single seed with different derivation paths. For QNK's hybrid Ed25519+Dilithium5 scheme, what is the recommended key derivation strategy? Separate seeds per key type, or unified seed with algorithm-specific derivation?

3. **DHT Scalability**: If every QNK wallet publishes a DID Document to the Kademlia DHT, and the network grows to 100K+ wallets, what is the expected DHT storage overhead per node? Should we implement lazy resolution (resolve on demand) vs eager replication?

4. **Submarine Swap Atomic Guarantee**: In the LDK submarine swap design, the QUG hash-lock and Lightning payment use the same preimage. If QNK's block time is ~1 second but Lightning payment can take 1-30 seconds across multiple hops, what is the failure window where QUG is locked but Lightning hasn't settled? How should the timeout cascade be structured?

5. **Cross-Chain MEV**: With a QUG/BTC bridge, arbitrageurs could exploit price differences between QNK's AMM and Bitcoin DEXes (Bisq, etc.). Is this beneficial (price alignment) or harmful (value extraction)? Should the bridge include MEV protection (commit-reveal, batch auctions)?

6. **Square Lightning Capacity Planning**: If QNK builds an LDK Lightning bridge and 0.1% of Square's 3M merchants see QUG-via-Lightning payments, that's 3,000 merchants. At 10 transactions/day average, that's 30,000 Lightning payments/day requiring QUG→BTC atomic swaps. What channel capacity (in BTC) is needed to sustain this volume? What is the optimal number of Lightning channels, and should QNK operate its own LSP (Lightning Service Provider)?

7. **MCP Server Security Model**: An MCP server exposing wallet operations (swap, deploy, transfer) to AI agents creates a new attack surface. What is the recommended auth model? Per-tool permissions? Spending limits? Human-in-the-loop confirmation for high-value operations? How does this interact with QNK's existing X-Wallet-Auth system?

8. **Goose + QNK Developer Onboarding**: If Goose (Block's AI agent, 27K GitHub stars) can interact with QNK via MCP, it becomes a developer onboarding tool. A new developer says "goose, deploy a token on QNK" and Goose handles everything. What MCP tools are minimum-viable for this flow? What guardrails prevent abuse (rate limiting, testnet-only for unsigned agents)?

---

## Appendix A: Repository References

| Block Project | GitHub | Language | License | Status (2026) |
|---|---|---|---|---|
| BDK | `bitcoindevkit/bdk` | Rust | MIT/Apache-2.0 | Active (Spiral-funded) |
| LDK | `lightningdevkit/rust-lightning` | Rust | MIT/Apache-2.0 | Active (Spiral-funded) |
| Bitkey | `proto-at-block/bitkey` | Rust/Kotlin/Swift | Various | Shipping (95 countries) |
| **Goose** | **`block/goose`** | **Rust/Python** | **Apache-2.0** | **Active (27K stars, AAIF)** |
| Proto Mining | `proto.xyz` | Hardware/Rust | Open-source design | 3nm tapeout complete |
| tbDEX (Rust) | `TBD54566975/tbdex-rs` | Rust | Apache-2.0 | Community-maintained (TBD wound down) |
| Web5 (Rust) | `TBD54566975/web5-rs` | Rust | Apache-2.0 | Community-maintained |
| did:dht | `TBD54566975/did-dht` | Multiple | Apache-2.0 | Community-maintained |
| Fedimint | `fedimint/fedimint` | Rust | MIT | Active |
| **MCP** | **`modelcontextprotocol`** | **TypeScript/Rust** | **MIT** | **Active (Linux Foundation AAIF)** |

## Appendix B: QNK Codebase Integration Points

| QNK Component | File Path | Integration Target |
|---|---|---|
| DEX AMM Pools | `crates/q-dex/src/lib.rs` | tbDEX pricing oracle |
| Bridge Tokens | `crates/q-api-server/src/bridge_tokens.rs` | Replace with BDK HTLC |
| Wallet Auth | `crates/q-api-server/src/wallet_auth.rs` | Add DID verification path |
| Kademlia DHT | `crates/q-network/src/peer_discovery.rs` | Dual-purpose for DID:DHT |
| P2P Manager | `crates/q-network/src/unified_network_manager.rs` | LDK peer integration |
| Atomic Swaps | `crates/q-api-server/src/main.rs` (BTC swap routes) | Upgrade to BDK/LDK |
| Slint Wallet | `gui/slint-wallet/src/main.rs` | Multisig key management |
| Gossipsub | `crates/q-network/src/distributed_dex.rs` | tbDEX message relay |

---

---

## Appendix C: Sources (Web Research, April 4, 2026)

- [Square Auto-Enables Bitcoin Payments for US Sellers](https://www.coindesk.com/business/2026/03/30/jack-dorsey-s-square-auto-enables-bitcoin-payments-for-millions-of-u-s-businesses) — CoinDesk, March 30, 2026
- [Block Open Source Introduces "Codename Goose"](https://block.xyz/inside/block-open-source-introduces-codename-goose) — Block.xyz
- [Goose AI Agent — 27K Stars](https://github.com/block/goose) — GitHub
- [The Lean Bitcoin Machine: How Dorsey's Radical Pivot Sent Block Soaring](https://markets.financialcontent.com/stocks/article/marketminute-2026-3-25-the-lean-bitcoin-machine-how-jack-dorseys-radical-pivot-sent-block-inc-soaring-in-early-2026) — FinancialContent, March 25, 2026
- [Block's 3nm ASIC Technology for Mining](https://corescientific.com/resources/blog/the-potential-of-blocks-3nm-asic-technology-to-maximize-mining/) — Core Scientific
- [Proto Mining: 3nm Chip Headed to Foundry](https://proto.xyz/blog/posts/latest-updates-3nm-system/) — Proto.xyz
- [Block Revives Bitcoin Faucet — April 6 Launch](https://www.ainvest.com/news/jack-dorsey-block-revives-bitcoin-faucet-april-6-launch-2604/) — AInvest, April 2026
- [Bitkey Review 2026](https://blockdyor.com/bitkey-review/) — BlockDYOR
- [Linux Foundation AAIF — MCP, Goose, AGENTS.md](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation) — Linux Foundation
- [Jack Dorsey on Stablecoins](https://www.indexbox.io/blog/block-ceo-jack-dorsey-reluctantly-accepts-stablecoin-demand/) — IndexBox
- [Spiral — Bitcoin Open Source](https://spiral.xyz/) — Spiral.xyz
- [Block Bitcoin Overview](https://block.xyz/bitcoin) — Block.xyz
- [Block Open Source Tools for Bitcoin Treasury Management](https://block.xyz/inside/block-releases-open-source-tools-for-bitcoin-treasury-management) — Block.xyz

---

*This document is intended for technical review by AI collaborators (DeepSeek, Nemotron) and QNK core contributors. All code samples are illustrative — production implementation requires security audit and test coverage per QNK's mandatory testing protocol (125+ critical tests).*

*Q-NarwhalKnight v10.2.5 | Block Integration RFC v1.0 | 2026-04-04*
