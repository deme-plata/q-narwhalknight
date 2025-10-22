# Q-NarwhalKnight Privacy-as-a-Service (PaaS)
## Enterprise-Grade Quantum-Resistant Privacy Infrastructure for Blockchain Networks

**Version:** 1.1
**Date:** October 2025
**Authors:** Q-NarwhalKnight Core Team
**Classification:** Public Technical Whitepaper

---

## Executive Summary

Q-NarwhalKnight Privacy-as-a-Service (PaaS) is an **enterprise-grade privacy infrastructure** that provides blockchain-agnostic anonymity, confidentiality, and quantum-resistant security services to external cryptocurrency networks, decentralized applications, and Web3 platforms.

**Core Value Proposition:**
Unlike privacy solutions limited to single blockchains (e.g., Tornado Cash for Ethereum, Zcash shielded pools), Q-NarwhalKnight PaaS offers a **universal privacy layer** that works with Bitcoin, Ethereum, Solana, and any blockchain system through standardized API endpoints.

**Key Differentiators:**
- ✅ **Quantum-Resistant** - Post-quantum cryptography (Dilithium5, Kyber1024)
- ✅ **Cross-Chain** - Works with any blockchain via RESTful API
- ✅ **Triple-Layer Security** - Network (Tor) + Transport (Noise Protocol) + Application (Mixing)
- ✅ **Compliance-Ready** - Selective disclosure capabilities
- ✅ **Enterprise SLA** - 99.9% uptime guarantee, dedicated infrastructure

**Target Market:**
- Cryptocurrency wallets requiring privacy features
- DeFi protocols needing MEV protection
- Enterprise blockchain deployments with confidentiality requirements
- Cross-chain bridges requiring transaction privacy
- Compliance-focused institutions needing auditable privacy

**Revenue Model:**
Pay-per-use (priced in QNK tokens) + Enterprise subscription tiers ($499-$9,999/month)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Technical Architecture](#2-technical-architecture)
3. [Privacy Infrastructure Components](#3-privacy-infrastructure-components)
4. [Enterprise Service Tiers](#4-enterprise-service-tiers)
5. [API Specification](#5-api-specification)
6. [Security & Compliance](#6-security--compliance)
7. [Performance Benchmarks](#7-performance-benchmarks)
8. [Use Cases](#8-use-cases)
9. [Competitive Analysis](#9-competitive-analysis)
10. [Roadmap](#10-roadmap)

---

## 1. Introduction

### 1.1 The Privacy Crisis in Public Blockchains

Public blockchains like Bitcoin and Ethereum provide transparency through immutable ledgers, but this transparency creates severe privacy vulnerabilities:

- **Transaction Graph Analysis**: Companies like Chainalysis can trace fund flows across wallets
- **IP Address Leakage**: Broadcasting transactions reveals user location
- **MEV Exploitation**: Frontrunning bots extract $500M+ annually from Ethereum users
- **Regulatory Overreach**: Financial surveillance threatens user rights

**Existing solutions are inadequate:**
- **Tornado Cash**: Sanctioned by OFAC, Ethereum-only, ECDSA signatures vulnerable to quantum attacks
- **Zcash**: Single-chain, requires running dedicated wallets, slow adoption
- **Monero**: Not interoperable with other chains, regulatory scrutiny
- **CoinJoin**: Weak anonymity sets, centralized coordinators

### 1.2 Q-NarwhalKnight PaaS Solution

Q-NarwhalKnight solves these limitations by providing:

1. **Universal Privacy API** - RESTful endpoints accepting transactions from ANY blockchain
2. **Quantum-Resistant Cryptography** - Future-proof against quantum computer attacks
3. **Compliance Mode** - Selective disclosure for regulatory requirements
4. **Enterprise Infrastructure** - SLA guarantees, dedicated support, white-label options

**Core Innovation**: Treat privacy as a **service layer** that sits above blockchains, not as a feature built into individual chains.

---

## 2. Technical Architecture

### 2.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  Q-NarwhalKnight PaaS Platform                   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: API Gateway (RESTful, authenticated, rate-limited)    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Privacy Services                                      │
│    • Transaction Mixing         • Tor Relay                     │
│    • Ring Signatures            • Stealth Addresses             │
│    • ZK-STARK Proofs           • Cross-Chain Atomic Swaps       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Encrypted P2P Network (Kademlia DHT + mDNS)           │
│    • Noise Protocol Encryption  • Gossipsub Message Propagation │
│    • DHT Peer Discovery        • Signed Message Authentication  │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Tor Network Integration                               │
│    • 4 Dedicated Circuits/Node  • Dandelion++ Anonymity         │
│    • Quantum Entropy Seeding    • Circuit Rotation              │
└─────────────────────────────────────────────────────────────────┘
           ▲                       ▲                       ▲
           │                       │                       │
      Bitcoin Network      Ethereum Network       Solana Network
```

### 2.2 Network Layer - Encrypted P2P Infrastructure

#### 2.2.1 Kademlia DHT for Global Peer Discovery

**Purpose**: Decentralized peer discovery without central servers

**Encryption & Security**:
```rust
// Transport stack: TCP + Noise Protocol + Yamux multiplexing
let transport = tcp::tokio::Transport::new(tcp::Config::default())
    .upgrade(upgrade::Version::V1)
    .authenticate(noise::Config::new(&keypair)?)  // Noise Protocol encryption
    .multiplex(yamux::Config::default())
    .boxed();
```

**Noise Protocol Framework**:
- **Authentication**: XX handshake pattern (mutual authentication)
- **Encryption**: ChaCha20-Poly1305 (256-bit keys)
- **Forward Secrecy**: New ephemeral keys per session
- **Post-Quantum Ready**: Can upgrade to Noise-KK with Kyber1024

**DHT Security Properties**:
- Peer IDs derived from Ed25519 public keys (cryptographically verifiable)
- DHT queries encrypted end-to-end via Noise Protocol
- Sybil attack resistance through proof-of-work peer IDs (optional)
- Eclipse attack mitigation via diverse bootstrap peers

**Performance**:
- Average peer discovery time: **< 500ms**
- DHT routing table convergence: **< 5 seconds**
- Global network scale: **100,000+ nodes** supported

#### 2.2.2 mDNS for Local Network Discovery

**Purpose**: Zero-configuration peer discovery on LANs

**Security Considerations**:
- mDNS is unencrypted (by design for local discovery)
- **Mitigation**: All application data transmitted over Noise-encrypted connections
- mDNS only broadcasts peer availability, not transaction data
- Local network assumption: LAN is semi-trusted (home/office networks)

**Privacy Enhancement**:
- Tor integration available for users requiring local network privacy
- Option to disable mDNS and use DHT-only mode

#### 2.2.3 Gossipsub for Message Propagation

**Purpose**: Efficient consensus message broadcast

**Authentication & Integrity**:
```rust
// Gossipsub with cryptographic message signing
let gossipsub = gossipsub::Behaviour::new(
    MessageAuthenticity::Signed(keypair.clone()),  // Ed25519 signatures
    gossipsub_config
)?;
```

**Security Properties**:
- **Message Authentication**: Every message signed with sender's Ed25519 key
- **Replay Attack Prevention**: Message IDs based on content hash
- **Byzantine Fault Tolerance**: Validates messages before propagation
- **Spam Protection**: Rate limiting + peer scoring

**Encryption**:
- Messages transmitted over Noise-encrypted connections (transport layer)
- Optional application-layer encryption for sensitive data

### 2.3 Cryptographic Agility - Future-Proof Security

**Phase-Based Cryptographic Migration**:

| Phase | Signatures | KEM | Transport Encryption | Status |
|-------|-----------|-----|---------------------|--------|
| **Phase 0** | Ed25519 | X25519 | Noise (ChaCha20) | ✅ Production |
| **Phase 1** | Dilithium5 | Kyber1024 | Noise-KK (Hybrid) | ✅ Available |
| **Phase 2** | Falcon1024 | NTRUPrime | PQ-TLS 1.3 | 🔄 In Development |
| **Phase 3** | SQIsign | FrodoKEM | Quantum Key Distribution | 📋 Planned |

**Crypto-Agile Handshake**:
```json
{
  "supported_schemes": [
    {
      "signature": "Dilithium5",
      "kem": "Kyber1024",
      "hash": "SHA3-256",
      "version": 2
    },
    {
      "signature": "Ed25519",
      "kem": "X25519",
      "hash": "SHA3-256",
      "version": 1
    }
  ],
  "preferred_scheme": { "signature": "Dilithium5", ... },
  "phase": "Phase1"
}
```

**Seamless Migration**: Nodes negotiate best common cryptographic scheme without hard forks

---

## 3. Privacy Infrastructure Components

### 3.1 Tor Integration - Network-Level Anonymity

#### Architecture
```
┌──────────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  User (Bitcoin)  │────►│ Tor Guard    │────►│  Tor Middle  │────►│  Tor Exit    │
│  Transaction     │     │  Relay       │     │  Relay       │     │  Relay       │
└──────────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                             │
                                                                             ▼
                                                                    ┌──────────────────┐
                                                                    │ Q-NarwhalKnight  │
                                                                    │  Mixing Service  │
                                                                    └──────────────────┘
```

#### Features
- **4 Dedicated Circuits per Validator**: Guard, consensus, gossip, relay
- **Circuit Rotation**: Every epoch (~10 minutes) for timing attack resistance
- **Quantum Entropy Seeding**: QRNG for circuit path selection
- **Controlled Egress Relays**: Enterprise tier customers can deploy egress infrastructure outside Tor network for last-hop connectivity

#### Security Properties
- **IP Address Concealment**: Origin IP never revealed to mixing service
- **Traffic Analysis Resistance**: Constant-rate cover traffic
- **End-to-End Encryption**: TLS 1.3 + Noise Protocol to PaaS API (Tor-independent encryption)

#### Egress Architecture

**Standard Mode** (All tiers):
```
User → Tor (3 hops) → Tor Exit → Internet → Q-NarwhalKnight API
```

**Enterprise Controlled Egress Mode**:
```
User → Tor (3 hops inside Tor network) → Controlled Egress Relay (customer-hosted) → Q-NarwhalKnight API
```

**Controlled Egress Benefits**:
- Avoid Tor exit node blacklisting by destination services
- Customer-controlled infrastructure (own IP addresses, certificates)
- BYO-region: Deploy egress relays in customer's VPC or datacenter
- Tor provides anonymity for inbound traffic, controlled egress provides reliable connectivity

**Fallback Mechanisms**:
- Snowflake bridge support (censorship resistance)
- Domain-fronted relays (CDN-based obfuscation)
- Direct connection mode (for low-risk use cases)

#### API Integration
```http
POST /api/v1/privacy/tor/relay
{
  "chain": "ethereum",
  "destination": "eth-rpc.example.com:8545",
  "data": "base64_encoded_transaction",
  "circuit_requirements": {
    "min_hops": 3,
    "quantum_seeded": true
  }
}
```

### 3.2 Quantum Mixing - Transaction-Level Privacy

#### Chaumian CoinJoin Protocol

**Steps:**
1. **Commitment Phase**: Users submit blinded output addresses
2. **Signing Phase**: Mix coordinator signs blinded outputs
3. **Unblinding Phase**: Users unblind signatures
4. **Assembly Phase**: Final CoinJoin transaction constructed
5. **Broadcast Phase**: Transaction sent via Tor + randomized timing delays

**Anonymity Set Calculation** (Simplified Overview):
```
Naive Calculation: Anonymity Set = Participants × Decoy Ratio
                                  = 16 participants × 15 decoys = 240 potential senders
```

**⚠️ Important**: This naive calculation assumes perfect uniformity and ignores real-world linkability risks. Actual effective anonymity is lower due to:

1. **Amount Correlation**: Transactions with unique amounts can be linked
2. **Timing Analysis**: Broadcast time correlation reduces anonymity
3. **Linkability Attacks**: Multiple transactions from same user can be correlated
4. **Decoy Selection Bias**: Non-uniform decoy selection reveals information

**Rigorous Anonymity Analysis (Epsilon-Differential Privacy)**:

We model transaction mixing as an ε-differentially private mechanism where ε (epsilon) quantifies privacy loss.

**Differential Privacy Definition**:
```
A mechanism M provides ε-differential privacy if for all datasets D1 and D2
differing in one record, and all outputs S:

  Pr[M(D1) ∈ S] ≤ e^ε × Pr[M(D2) ∈ S]
```

**Q-NarwhalKnight Mixing Privacy Parameters**:

| Privacy Level | Ring Size | Decoys | Amount Buckets | Timing Delay | ε (epsilon) | Anonymity Guarantee |
|--------------|-----------|--------|----------------|--------------|-------------|-------------------|
| **Standard** | 16 | 15 | 10 buckets | 0-30s random | ε ≈ 2.3 | ~90% indistinguishable |
| **High** | 32 | 31 | 20 buckets | 0-90s random | ε ≈ 1.5 | ~95% indistinguishable |
| **Maximum** | 64 | 63 | 40 buckets | 0-180s random | ε ≈ 0.7 | ~99% indistinguishable |

**Epsilon Interpretation**:
- **ε = 0**: Perfect privacy (impossible in practice)
- **ε < 1**: Strong privacy guarantees
- **ε < 2**: Moderate privacy (acceptable for most use cases)
- **ε > 3**: Weak privacy (not recommended)

**Effective Anonymity Set (Min-Entropy Calculation)**:

Instead of naive counting, we calculate min-entropy H∞:

```
H∞(X) = -log₂(max P(x))
         where P(x) is probability that transaction x belongs to user

Effective Anonymity Set = 2^H∞
```

**Example (Standard Mixing)**:
- Ring size: 16 participants
- Amount buckets: 10 (transactions grouped by amount ranges)
- Timing buckets: 6 (5-second windows)
- Unique combinations: 16 × 10 × 6 = 960

Assuming adversary can narrow down to specific amount bucket and timing window:
```
P(correct sender) ≈ 1/16 = 0.0625
H∞ = -log₂(0.0625) = 4 bits
Effective Anonymity Set = 2^4 = 16 users  ✓ (matches ring size under ideal conditions)
```

**Worst-Case Analysis** (Unique amount + timing correlation):
```
If transaction amount is unique: P(correct sender) ≈ 1/4 (timing still provides some privacy)
H∞ = -log₂(0.25) = 2 bits
Effective Anonymity Set = 2^2 = 4 users  ⚠️ (reduced privacy)
```

**Amount Bucketing Strategy**:

To prevent amount correlation attacks, Q-NarwhalKnight enforces amount bucketing:

| Bucket | Range (BTC) | Range (ETH) | Range (USD) |
|--------|-------------|-------------|-------------|
| Bucket 1 | 0.001 - 0.01 | 0.01 - 0.1 | $50 - $100 |
| Bucket 2 | 0.01 - 0.1 | 0.1 - 1.0 | $100 - $500 |
| Bucket 3 | 0.1 - 0.5 | 1.0 - 5.0 | $500 - $2,500 |
| Bucket 4 | 0.5 - 1.0 | 5.0 - 10.0 | $2,500 - $10,000 |
| Bucket 5 | 1.0 - 5.0 | 10.0 - 50.0 | $10,000 - $50,000 |
| ... | (up to 40 buckets for "maximum" privacy) | | |

**Transactions are only mixed with others in the same bucket**, preventing unique amount linkability.

**Randomized Delay Distribution**:

Timing delays follow truncated exponential distribution:

```python
# Standard privacy level
delay_seconds = min(30, random.expovariate(λ=0.1))

# Maximum privacy level
delay_seconds = min(180, random.expovariate(λ=0.02))
```

This prevents adversaries from correlating input/output timing patterns.

**Privacy Budget Composition**:

When a user performs multiple mixing operations, privacy loss compounds:

```
ε_total = ε₁ + ε₂ + ... + εₙ  (sequential composition)

Example: User mixes 5 transactions at ε=1.5 each
ε_total = 5 × 1.5 = 7.5  ⚠️ (weak privacy after 5 mixes)
```

**Recommendation**: Users should limit mixing frequency or use "maximum" privacy level (ε ≈ 0.7) for repeated operations.

**Privacy vs Performance Tradeoff**:

| Metric | Standard (ε ≈ 2.3) | Maximum (ε ≈ 0.7) | Increase |
|--------|-------------------|-------------------|----------|
| Ring Size | 16 | 64 | 4× |
| Anonymity Set | ~16 users | ~64 users | 4× |
| Mixing Latency | 15-30s (P95) | 45-90s (P95) | 3× |
| Pool Fill Time | ~5 min | ~15 min | 3× |
| Fee (basis points) | 10 bps (0.1%) | 15 bps (0.15%) | 1.5× |

#### Ring Signatures (Experimental - Quantum-Resistant)

**⚠️ Status**: Experimental feature under cryptographic audit. Not recommended for production use until formal verification complete (ETA: Q2 2026).

**Algorithm**: Lattice-based linkable ring signatures (MLSAG/CLSAG-style construction)

**Technical Approach**:
- Ring signature construction using lattice assumptions (not standard Dilithium5)
- Based on published research: "Linkable Ring Signatures from Lattices" (academic citations pending)
- **Current Implementation**: Feature-flagged, disabled by default for enterprise customers

**Properties**:
- **Unlinkability**: Cannot determine which ring member signed
- **Non-frameability**: Only real signer can produce valid signature
- **Linkability**: Prevents double-spending via key images
- **Quantum Resistance**: Lattice-based cryptography (post-quantum secure)

**Ring Size**: 16 members (configurable: 8, 16, 32, 64)

**Security Considerations**:
- Awaiting third-party cryptographic audit (scheduled Q1 2026)
- Not yet peer-reviewed in academic conferences
- Alternative: Use standard Dilithium5 signatures + mixing for production deployments

**Feature Flag**:
```json
{
  "privacy_level": "maximum",
  "mixing_parameters": {
    "ring_signatures_experimental": false  // Set to true to enable (at your own risk)
  }
}
```

#### Stealth Addresses

**Dual-Key Scheme** (similar to Monero):
- **View Key**: Scan blockchain for incoming payments
- **Spend Key**: Authorize spending received funds

**One-Time Address Generation**:
```
P = H(rA)G + B
```
Where:
- `P` = One-time public address
- `r` = Sender's random scalar
- `A` = Recipient's public view key
- `B` = Recipient's public spend key
- `G` = Ed25519 base point

**Privacy Guarantee**: Each transaction uses unique address (no address reuse)

#### ZK-STARK Proofs for Mixing Validity

**Prover Statement**:
"I know inputs totaling X QNK and possess valid signatures, without revealing which inputs"

**Proof Properties**:
- **Zero-Knowledge**: Verifier learns nothing except validity
- **Soundness**: Cannot prove false statement
- **Completeness**: Valid proofs always verify
- **Quantum Resistance**: No reliance on elliptic curve discrete log

**Performance**:
- Proof generation: **250ms** (average)
- Proof size: **45 KB**
- Verification time: **50ms**

### 3.4 Transport Security & PKI

Q-NarwhalKnight PaaS provides enterprise-grade transport security with automated certificate management and optional post-quantum PKI.

#### 3.4.1 Managed TLS (ACME)

**Automatic Certificate Provisioning**:
- Auto-issue/renew public certificates via ACME protocol
- Supported CAs: Let's Encrypt, Buypass, ZeroSSL
- Challenge methods: `http-01`, `dns-01`, `tls-alpn-01`
- **Keyless TLS**: HSM-backed private keys never leave secure hardware
- **OCSP Stapling**: Privacy-preserving revocation checking (default enabled)

**Example Configuration**:
```json
{
  "tls_config": {
    "acme_enabled": true,
    "acme_ca": "letsencrypt",
    "domains": ["paas.qnarwhalknight.com", "api.qnarwhalknight.com"],
    "challenge_method": "dns-01",
    "auto_renew_days_before_expiry": 30,
    "keyless_tls": true,
    "hsm_backend": "aws-cloudhsm"
  }
}
```

#### 3.4.2 Private PKI (mTLS for Internal Services)

**Internal ACME Server**:
- Issues short-lived service certificates (8–24 hour TTL)
- SPIFFE/SPIRE compatible for workload identity
- Automatic rotation eliminates long-lived secrets

**Mutual TLS (mTLS)**:
- Service-to-service authentication via client certificates
- Enforced for inter-node consensus communication
- Zero Trust architecture (no implicit trust within network perimeter)

**Example Internal Service Certificate**:
```
Subject: spiffe://q-narwhalknight.com/validator/node-42
Issuer: Q-NarwhalKnight Internal CA
Validity: 24 hours
Extensions:
  - SPIFFE ID: spiffe://q-narwhalknight.com/validator/node-42
  - DNS SANs: node-42.internal.qnk.local
```

#### 3.4.3 PQ-Hybrid Certificates (Optional)

**Post-Quantum Ready**:
- Optional hybrid leaf certificates: Ed25519 + Dilithium5
- Dual signatures verify via both classical and PQ algorithms
- Compatible with standard X.509 infrastructure via extension fields

**Hybrid Certificate Structure**:
```
X.509 Certificate v3
├─ Subject Public Key Info (SPKI): Ed25519 public key
├─ Signature Algorithm: Ed25519
├─ X.509v3 extensions:
│  ├─ Subject Alternative Name: DNS:api.qnarwhalknight.com
│  ├─ Custom Extension (OID 1.3.6.1.4.1.99999.1):
│  │  └─ Dilithium5 public key (2,592 bytes)
│  └─ Custom Extension (OID 1.3.6.1.4.1.99999.2):
│     └─ Dilithium5 signature (4,595 bytes)
```

Clients verify both Ed25519 (standard) and Dilithium5 (PQ) signatures.

#### 3.4.4 Security Best Practices

**Enabled by Default**:
- **HSTS (HTTP Strict Transport Security)**: max-age=31536000; includeSubDomains; preload
- **CAA DNS Records**: Restrict which CAs can issue certificates for domains
- **Certificate Transparency (CT)**: Monitor CT logs for unauthorized issuance
- **TLS 1.3 Only**: Deprecated TLS 1.2 and below (no CBC mode ciphers)
- **Forward Secrecy**: Ephemeral ECDHE/Kyber1024 key exchanges
- **OCSP Stapling**: Reduce privacy leakage from revocation checks

**Cipher Suites** (prioritized order):
1. `TLS_CHACHA20_POLY1305_SHA256` (preferred for mobile/ARM)
2. `TLS_AES_256_GCM_SHA384` (hardware-accelerated AES-NI)
3. `TLS_AES_128_GCM_SHA256` (fallback)

**HSM Integration**:
- Private keys stored in FIPS 140-2 Level 3 HSMs (AWS CloudHSM, Thales Luna, etc.)
- Signing operations performed within HSM (keys never extracted)
- Disaster recovery via encrypted key backup (M-of-N threshold decryption)

**QRNG Entropy Footnote**¹:
QRNG source (device/HSM entropy) mixed with DRBG (NIST SP 800-90A). Quantum entropy never used alone for key generation; always combined with classical CSPRNG to prevent hardware backdoors.

---

### 3.5 Cross-Chain Privacy Bridge

#### Supported Chains (Phase 1)

| Chain | Transaction Format | Stealth Addresses | Ring Signatures | Mixing |
|-------|-------------------|------------------|----------------|--------|
| Bitcoin | PSBT | ✅ | ⚠️ Experimental | ✅ |
| Ethereum | RLP-encoded | ✅ | ⚠️ Experimental | ✅ |
| Solana | Borsh-serialized | ✅ | ⚠️ Partial | ✅ |
| Cardano | CBOR | 🔄 Q1 2026 | 🔄 Q1 2026 | 🔄 Q1 2026 |

#### Atomic Swaps with Privacy

**Hashed Timelock Contracts (HTLCs) + Stealth Addresses**:
```
Bitcoin HTLC:
  OP_IF
    OP_SHA256 <hash> OP_EQUALVERIFY <stealth_pubkey_A> OP_CHECKSIG
  OP_ELSE
    <timelock> OP_CHECKLOCKTIMEVERIFY OP_DROP <refund_stealth_pubkey_B> OP_CHECKSIG
  OP_ENDIF
```

**Privacy Properties**:
- Swap participants use stealth addresses (no address reuse)
- Swap transactions broadcast via Tor
- Optional ring signature authorization (chain-dependent)

---

## 4. Enterprise Service Tiers

### 4.1 Pay-Per-Use Tier (Retail)

**Target Audience**: Individual users, small wallets, developers

**Pricing** (in QNK tokens):

| Service | Cost | Notes |
|---------|------|-------|
| Tor Circuit Relay | 0.001 QNK/MB | Network bandwidth cost |
| Transaction Mixing | 0.1% of tx value | Minimum 0.01 QNK |
| Ring Signature Generation | 0.001 QNK/signature | Quantum-resistant |
| Stealth Address (1x) | 0.0001 QNK | Bulk discounts available |
| ZK-STARK Proof | 0.01 QNK/proof | Compute-intensive |
| Atomic Swap Coordination | 0.05 QNK/swap | Cross-chain complexity |

**Features**:
- ✅ Public API access
- ✅ Standard rate limits (100 req/min)
- ✅ Community support (Discord/Telegram)
- ✅ 95% uptime SLA

**Usage Example**:
Mixing 1 BTC (~$60,000 USD) costs 0.1% = $60 USD equivalent in QNK

### 4.2 Professional Tier ($499/month)

**Target Audience**: Wallets, DApps, trading bots

**Included**:
- ✅ 100,000 API calls/month
- ✅ Priority Tor circuits (lower latency)
- ✅ Dedicated support (email, 24-hour response)
- ✅ 99% uptime SLA
- ✅ Custom rate limits (1,000 req/min)
- ✅ Webhook notifications
- ✅ Advanced analytics dashboard

**Overages**: $0.005/request beyond 100K

**Best For**: Medium-volume applications processing <3,000 transactions/day

### 4.3 Enterprise Tier ($1,999/month)

**Target Audience**: Exchanges, DeFi protocols, institutions

**Included**:
- ✅ **Unlimited** API calls
- ✅ Controlled egress relays (customer-owned IPs)
- ✅ Priority mixing pools (faster completion)
- ✅ 24/7 dedicated support (Slack/phone)
- ✅ **99.9% uptime SLA** with credits
- ✅ Custom compliance configurations
- ✅ Batch processing API
- ✅ Multi-chain transaction bundling
- ✅ Quarterly business reviews

**Add-Ons**:
- Private relay network: +$500/month
- Compliance reporting: +$300/month
- Custom cryptographic schemes: Custom pricing

**Best For**: High-volume platforms processing 10,000+ transactions/day

### 4.4 White-Label Tier ($9,999/month)

**Target Audience**: Blockchain networks, large enterprises

**Included (Everything from Enterprise +)**:
- ✅ White-label API (your branding)
- ✅ On-premise deployment option
- ✅ Source code access (limited modules)
- ✅ Dedicated infrastructure (isolated network)
- ✅ Custom SLA (up to 99.99%)
- ✅ Regulatory compliance consulting
- ✅ Integration engineering support
- ✅ Governance participation (PaaS roadmap)

**Custom Pricing for**:
- Multi-region deployments
- Dedicated development team
- Regulatory certifications (SOC2, ISO 27001)

**Example Customer**: Major cryptocurrency exchange wanting to offer privacy features under their own brand

---

## 5. API Specification

**API Conventions**:
- All endpoints use HTTPS with TLS 1.3
- Request/response format: JSON (UTF-8 encoding)
- **Idempotency**: All POST endpoints accept `Idempotency-Key` header. Identical keys within 24h return the original result; body mismatch → `409 Conflict`. See Appendix D for details.
- **Error Handling**: RFC 9457 Problem Details for HTTP APIs (`application/problem+json`)
- **Rate Limiting**: Enforced per tier with `X-RateLimit-*` headers
- **Versioning**: API version in URL path (`/api/v1/...`)

### 5.1 Authentication

**Methods Supported**:
1. **API Key Authentication** (Retail/Professional)
2. **Hybrid Request Signatures** (All tiers, recommended)
3. **Legacy Wallet Signature** (All tiers, deprecated for PQ-capable clients)
4. **OAuth 2.0** (Enterprise/White-Label)

#### 5.1.1 Hybrid Request Signatures (Quantum-Resistant)

**Purpose**: Quantum-resistant authentication using dual-signature verification

**Protocol**:
```http
POST /api/v1/privacy/mix/submit
Authorization: MultiSig v=1; schemes="ecdsa,dilithium5"; sigs="base64_ecdsa_sig,base64_dilithium5_sig"
X-Req-Hash: SHA3-256(canonical_request_body)
X-Req-Timestamp: 1698765432
X-Wallet-Address: 0xABC...

{
  "chain": "ethereum",
  "transaction_data": {...}
}
```

**Signature Generation (Client-Side)**:
```javascript
// 1. Canonicalize request body (sorted JSON, no unknown fields)
const canonical_body = JSON.stringify(request_body, Object.keys(request_body).sort());

// 2. Compute request hash
const req_hash = SHA3_256(canonical_body);

// 3. Build message to sign
const message = `${wallet_address}:${timestamp}:${req_hash}`;

// 4. Sign with both ECDSA (legacy) and Dilithium5 (PQ)
const ecdsa_sig = secp256k1.sign(keccak256(message), ecdsa_private_key);
const dilithium5_sig = dilithium5.sign(message, dilithium5_private_key);

// 5. Encode signatures
const sigs = `${base64(ecdsa_sig)},${base64(dilithium5_sig)}`;
```

**Server-Side Verification**:
```rust
// Both signatures MUST verify for hybrid mode
let ecdsa_valid = verify_ecdsa(&message, &ecdsa_sig, &wallet_pubkey);
let dilithium5_valid = verify_dilithium5(&message, &dilithium5_sig, &dilithium5_pubkey);

assert!(ecdsa_valid && dilithium5_valid, "Hybrid signature verification failed");
```

**Migration Path**:
- **Phase 0 clients** (ECDSA-only): Supported via legacy mode (see below)
- **Phase 1 clients** (Hybrid): Both ECDSA + Dilithium5 required
- **Phase 2+ clients** (PQ-only): Dilithium5-only authentication

#### 5.1.2 Legacy Wallet Signature (Non-PQ, Deprecated)

**⚠️ Security Notice**: This authentication method uses ECDSA signatures which are vulnerable to quantum computer attacks. Recommended for backwards compatibility only. Will be deprecated in Q3 2026.

**Example**:
```http
POST /api/v1/privacy/mix/submit
Authorization: Signature wallet=0xABC...,sig=0xDEF...,timestamp=1698765432

{
  "chain": "ethereum",
  "transaction_data": {...}
}
```

**Signature Verification**:
```javascript
const message = `${wallet_address}:${timestamp}:${request_body_hash}`;
const recovered_address = ecrecover(keccak256(message), signature);
assert(recovered_address === wallet_address);
```

**Deprecation Timeline**:
- Q4 2025: Legacy mode fully supported
- Q2 2026: Warning headers added to legacy responses
- Q3 2026: Legacy mode disabled for new clients
- Q4 2026: Legacy mode EOL (End of Life)

### 5.2 Core Endpoints

#### 5.2.1 Tor Relay Service

```http
POST /api/v1/privacy/tor/relay
Content-Type: application/json
Authorization: Bearer <api_key>

{
  "chain": "bitcoin",
  "destination": "btc-node.example.com:8333",
  "data": "base64_encoded_tx",
  "circuit_requirements": {
    "min_hops": 3,
    "exit_country": "any",
    "avoid_countries": ["CN", "RU"],
    "quantum_seeded": true,
    "circuit_lifetime_minutes": 10
  }
}
```

**Response**:
```json
{
  "success": true,
  "circuit_id": "qnk-tor-abc123def456",
  "latency_ms": 145,
  "exit_node_country": "DE",
  "exit_node_fingerprint": "E2EB7F3F9E4C...",
  "relay_cost_qnk": "0.001",
  "estimated_bandwidth_mb": 1.2
}
```

**Error Codes**:
- `TOR_CIRCUIT_FAILED`: Could not establish circuit
- `TOR_EXIT_BLOCKED`: Destination blocks Tor exits
- `INSUFFICIENT_BALANCE`: Not enough QNK tokens

#### 5.2.2 Transaction Mixing Service

```http
POST /api/v1/privacy/mix/submit
Content-Type: application/json
Authorization: MultiSig v=1; schemes="ecdsa,dilithium5"; sigs="base64_ecdsa_sig,base64_dilithium5_sig"
X-Req-Hash: SHA3-256(canonical_request_body)
X-Req-Timestamp: 1698765432

{
  "chain": "ethereum",
  "transaction_data": {
    "from": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
    "to": "0x8ba1f109551bD432803012645Ac136ddd64DBA72",
    "value": "1500000000000000000",  // 1.5 ETH in wei
    "gas_limit": "21000",
    "gas_price": "20000000000"
  },
  "privacy_level": "maximum",
  "mixing_parameters": {
    "decoy_count": 20,
    "ring_size": 16,
    "stealth_addresses": true,
    "quantum_resistant": true,
    "compliance_mode": false
  },
  "output_addresses": [
    // Optional: specify multiple output addresses for better privacy
    "qnk:stealth:0xABC...",
    "qnk:stealth:0xDEF..."
  ]
}
```

**Response**:
```json
{
  "success": true,
  "mixing_id": "qnk-mix-789xyz",
  "anonymity_set_size": 320,
  "estimated_completion_seconds": 60,
  "mixing_pool_id": "pool-eth-large-0042",
  "participant_count": 16,
  "stealth_addresses": [
    {
      "address": "qnk:stealth:0x9F8...",
      "view_key": "encrypted_view_key_base64",
      "spend_key": "encrypted_spend_key_base64"
    }
  ],
  "mixing_fee_qnk": "0.05",
  "zk_proof": {
    "proof_type": "stark",
    "proof_data": "base64_encoded_proof",
    "verification_key": "base64_encoded_vk"
  },
  "broadcast_via_tor": true
}
```

**Mixing Completion Webhook** (Enterprise tier):
```http
POST https://your-webhook.com/mixing-complete
{
  "mixing_id": "qnk-mix-789xyz",
  "status": "completed",
  "final_transaction_hash": "0xABCDEF...",
  "completion_time": "2025-10-22T12:34:56Z"
}
```

#### 5.2.3 Ring Signature Generation (Experimental)

**⚠️ Experimental Feature**: Lattice-based ring signatures are under audit. Production use not recommended until Q2 2026.

```http
POST /api/v1/privacy/ring-signature/generate
Content-Type: application/json

{
  "chain": "bitcoin",
  "message_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "ring_members": [
    {
      "public_key": "02c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5",
      "key_index": 0
    },
    // ... 15 more ring members
  ],
  "signing_key_index": 7,
  "signature_scheme": "lattice_lrs_experimental"
}
```

**Response**:
```json
{
  "success": true,
  "ring_signature": {
    "signature_data": "base64_lrs_sig",
    "key_image": "a1b2c3d4e5f6...",
    "ring_size": 16,
    "scheme": "lattice_lrs_experimental",
    "experimental": true
  },
  "verification_data": {
    "ring_public_keys": [...],
    "message_hash": "e3b0c442..."
  },
  "signature_fee_qnk": "0.001",
  "quantum_resistant": true,
  "audit_status": "in_progress",
  "production_ready": false
}
```

**Verification Endpoint**:
```http
POST /api/v1/privacy/ring-signature/verify
{
  "signature_data": "base64_signature",
  "key_image": "a1b2c3d4...",
  "ring_public_keys": [...],
  "message_hash": "e3b0c442..."
}
```

**Response**:
```json
{
  "valid": true,
  "key_image_unique": true,
  "verification_time_ms": 15
}
```

#### 5.2.4 Stealth Address Generation

```http
POST /api/v1/privacy/stealth-address/generate
Content-Type: application/json

{
  "chain": "ethereum",
  "recipient_public_key": "0x04a1b2c3d4e5f6...",
  "count": 5
}
```

**Response**:
```json
{
  "success": true,
  "stealth_addresses": [
    {
      "address": "0x9F8A7B6C5D4E3F2A1B0C9D8E7F6A5B4C3D2E1F0",
      "ephemeral_pubkey": "0x04e5f6a7b8c9d0...",
      "tx_pubkey": "0x03c3d2e1f0a9b8...",
      "shared_secret": "encrypted_for_recipient_pubkey"
    }
    // ... 4 more addresses
  ],
  "view_key": "master_view_key_encrypted",
  "spend_key": "master_spend_key_encrypted",
  "generation_fee_qnk": "0.0005"
}
```

#### 5.2.5 ZK-STARK Proof Generation (Universal)

```http
POST /api/v1/privacy/zk-stark/prove
Content-Type: application/json

{
  "statement": "balance_in_range",
  "witness": {
    "balance": "1500000000000000000",
    "randomness": "a1b2c3d4e5f6..."
  },
  "public_inputs": {
    "commitment": "c7d8e9f0a1b2...",
    "min_balance": "1000000000000000000",
    "max_balance": "2000000000000000000"
  },
  "proof_type": "stark"
}
```

**Response**:
```json
{
  "success": true,
  "proof_id": "zk-proof-abc123",
  "proof_data": "base64_encoded_stark_proof",
  "verification_key": "base64_encoded_vk",
  "proof_size_bytes": 45000,
  "generation_time_ms": 250,
  "proof_fee_qnk": "0.01",
  "verifiable_on_chain": true
}
```

**Verification**:
```http
POST /api/v1/privacy/zk-stark/verify
{
  "proof_data": "base64_proof",
  "verification_key": "base64_vk",
  "public_inputs": {...}
}
```

### 5.3 Rate Limits

| Tier | Requests/Minute | Requests/Day | Burst Allowance |
|------|----------------|--------------|-----------------|
| Pay-Per-Use | 100 | 10,000 | 150 |
| Professional | 1,000 | 100,000 | 1,500 |
| Enterprise | 10,000 | Unlimited | 15,000 |
| White-Label | Custom | Unlimited | Custom |

**Rate Limit Headers**:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1698765432
```

**Rate Limit Exceeded Response**:
```json
{
  "error": "RATE_LIMIT_EXCEEDED",
  "message": "You have exceeded your rate limit of 100 requests/minute",
  "retry_after_seconds": 42
}
```

---

## 6. Security & Compliance

### 6.1 Threat Model

**Adversary Capabilities**:
1. **Network Surveillance**: Passive monitoring of internet traffic
2. **Transaction Graph Analysis**: Chainalysis-level blockchain tracing
3. **Timing Attacks**: Correlation of transaction broadcast times
4. **Sybil Attacks**: Flooding network with malicious nodes
5. **Quantum Computer**: Future attacks on classical cryptography

**Security Properties Guaranteed**:

| Attack Vector | Mitigation | Status |
|--------------|------------|--------|
| IP Address Tracking | Tor + mDNS local discovery | ✅ Protected |
| Transaction Linking | Ring signatures + mixing | ✅ Protected |
| Amount Correlation | Amount bucketing + STARK range proofs | ✅ Protected |
| Timing Analysis | Private builder relays + randomized delays | ✅ Protected |
| Quantum Attacks | Dilithium5 + Kyber1024 | ✅ Protected |
| Traffic Analysis | Constant-rate cover traffic | ✅ Protected |
| Exit Node Monitoring | End-to-end encryption | ✅ Protected |

### 6.2 Compliance Operations (Enterprise Feature)

Q-NarwhalKnight PaaS provides comprehensive compliance infrastructure for regulated institutions while preserving privacy for non-compliance users.

#### 6.2.1 Know-Your-Transaction (KYT) / Know-Your-Origin (KYO)

**Inbound Address Screening**:
```
Transaction Submission → KYT Check → Risk Scoring → Accept/Reject
```

**Sanctions Screening Integration**:
- **OFAC SDN List** (Office of Foreign Assets Control - Specially Designated Nationals)
- **UN Security Council Sanctions**
- **EU Consolidated Financial Sanctions List**
- **National sanctions programs** (UK HMT, AUSTRAC, etc.)

**API Integration**:
```http
POST /api/v1/compliance/kyt/check
Authorization: Bearer <enterprise_api_key>

{
  "addresses": [
    "bc1q...",
    "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb"
  ],
  "chains": ["bitcoin", "ethereum"],
  "screening_level": "enhanced"  // Options: basic, standard, enhanced
}
```

**Response**:
```json
{
  "results": [
    {
      "address": "bc1q...",
      "risk_score": 15,  // 0-100 (0=clean, 100=sanctioned)
      "risk_level": "low",  // low, medium, high, sanctioned
      "sanctions_match": false,
      "pep_match": false,  // Politically Exposed Person
      "source_of_funds_risk": "low",
      "recommended_action": "accept",
      "screening_timestamp": "2025-10-22T12:34:56Z"
    }
  ],
  "compliance_provider": "chainalysis"  // Or elliptic, coinfirm, etc.
}
```

**Risk Scoring Criteria**:
- Direct sanctions list match → Risk score: 100 (auto-reject)
- 1-hop from sanctioned address → Risk score: 70-90 (manual review)
- Known mixer/exchange → Risk score: 20-40 (accept with monitoring)
- Clean history → Risk score: 0-10 (auto-accept)

#### 6.2.2 FATF Travel Rule Compliance

**Purpose**: Transfer originator and beneficiary information between VASPs (Virtual Asset Service Providers)

**Supported Standards**:
- **TRP** (Travel Rule Protocol) - InterVASP messaging
- **IVMS-101** (Inter-VASP Messaging Standard) - Structured data format
- **OpenVASP** - Decentralized VASP directory

**Travel Rule Payload Example**:
```json
{
  "transaction_id": "qnk-mix-abc123",
  "travel_rule_payload": {
    "originator": {
      "vasp_id": "US-VASP-12345",
      "customer_id": "hash(customer_pii)",
      "wallet_address": "bc1q...",
      "name": "encrypted(John Doe)",  // Encrypted for beneficiary VASP only
      "national_id": "encrypted(passport: ABC123456)",
      "address": "encrypted(123 Main St, NY, USA)"
    },
    "beneficiary": {
      "vasp_id": "EU-VASP-67890",
      "wallet_address": "bc1p...",
      "name": "encrypted(Jane Smith)",
      "verified": true
    },
    "amount": "1.5 BTC",
    "threshold_applies": true,  // FATF: >$1,000 USD equivalent
    "encryption": "beneficiary_vasp_pubkey",
    "signature": "originator_vasp_signature"
  },
  "compliance_mode": true
}
```

**Q-NarwhalKnight Travel Rule Integration**:
- Stores encrypted travel rule payloads with transaction commitments
- Threshold decryption (M-of-N) for lawful access requests
- Automatic IVMS-101 formatting
- VASP directory lookup via DNS or blockchain registry

#### 6.2.3 Jurisdictional Controls & Geo-Fencing

**Configurable Rule Packs per Customer**:
```json
{
  "compliance_policy": {
    "jurisdiction": "US",
    "blocked_countries": ["KP", "IR", "SY", "CU"],  // North Korea, Iran, Syria, Cuba
    "high_risk_countries": ["RU", "CN", "VE"],  // Require enhanced due diligence
    "require_travel_rule": true,
    "require_kyt_screening": true,
    "max_transaction_amount_usd": 100000,
    "require_source_of_funds_above_usd": 10000
  }
}
```

**IP Geolocation Enforcement**:
- Block API access from sanctioned countries (optional, can be bypassed via Tor)
- Warning headers for high-risk jurisdictions
- Compliance report generation per jurisdiction

#### 6.2.4 Audit Trails with Zero-Knowledge Commitments

**Challenge**: Prove compliance checks were performed WITHOUT revealing transaction details

**Solution**: ZK-Attested Compliance Proofs

**Commitment Scheme**:
```
C = H(job_id || policy_version || timestamp || inputs_hash || compliance_checks_passed)
```

**ZK Proof Statement**:
"I performed KYT screening, sanctions checks, and travel rule validation according to policy v1.2, and all checks passed, without revealing which addresses were screened or transaction amounts"

**Audit Trail API**:
```http
POST /api/v1/compliance/audit-trail/create
{
  "transaction_id": "qnk-mix-abc123",
  "policy_version": "v1.2",
  "checks_performed": [
    "ofac_sanctions_screening",
    "un_sanctions_screening",
    "kyt_risk_scoring",
    "travel_rule_ivms101"
  ],
  "all_checks_passed": true,
  "zk_proof": "base64_encoded_stark_proof",
  "commitment": "c7d8e9f0a1b2..."
}
```

**Auditor Verification**:
Regulators can verify commitment was created at time T without learning transaction details:
```http
POST /api/v1/compliance/audit-trail/verify
{
  "commitment": "c7d8e9f0a1b2...",
  "zk_proof": "base64_proof",
  "expected_policy_version": "v1.2"
}
```

**Response**:
```json
{
  "proof_valid": true,
  "policy_version": "v1.2",
  "timestamp": "2025-10-22T12:34:56Z",
  "checks_verified": true,
  "transaction_details_revealed": false  // Privacy preserved!
}
```

#### 6.2.5 Lawful Disclosure & Threshold Governance

**Selective Disclosure Protocol**:

Enterprises can optionally enable compliance mode to meet regulatory requirements (e.g., FATF Travel Rule).

**How It Works**:
1. **User Consent**: User explicitly opts into compliance disclosure
2. **View Key Escrow**: Encrypted view key stored with trusted third party (e.g., regulated custodian)
3. **Regulatory Request**: Authorities request transaction details via legal process
4. **Threshold Decryption**: M-of-N key holders decrypt view key (prevents single point of control)
5. **Selective Revelation**: Only specific transactions revealed, not entire history
6. **Disclosure Receipt**: ZK proof that disclosure was authorized via threshold governance

**Example: Travel Rule Compliance**
```json
{
  "transaction_id": "qnk-mix-abc123",
  "compliance_mode": true,
  "disclosure_policy": {
    "view_key_escrow": {
      "escrow_service_pubkey": "0x...",
      "threshold_scheme": "3-of-5",  // Requires 3 of 5 keyholders to decrypt
      "keyholders": [
        "internal_compliance_officer",
        "external_auditor_pwc",
        "legal_counsel",
        "independent_trustee_1",
        "independent_trustee_2"
      ]
    },
    "jurisdiction": "US",
    "retention_period_days": 2555,  // 7 years (US AML requirement)
    "disclosure_sla_hours": 24  // Respond to lawful requests within 24 hours
  },
  "user_consent_signature": "0xABCDEF...",
  "user_consent_timestamp": "2025-10-22T12:00:00Z"
}
```

**Lawful Disclosure SLOs** (Service Level Objectives):
- **<24 hour response** to valid legal process (subpoena, court order, warrant)
- Threshold governance prevents unilateral access
- Disclosure audit trail (who, what, when, legal authority)

**Privacy Guarantee**: Non-compliance users' transactions remain fully private (zero-knowledge). Compliance mode is opt-in only.

### 6.3 Audit & Transparency

**On-Chain Verification**:
- All mixing pools publish cryptographic commitments
- ZK-STARK proofs verifiable by anyone
- No trusted setup required

**Open-Source Components**:
- Core cryptographic libraries (ring signatures, ZK proofs)
- API client SDKs (JavaScript, Python, Rust)
- Compliance disclosure protocol

**Security Audits**:
- Annual third-party security audits
- Bug bounty program ($10,000-$100,000 rewards)
- Public disclosure of findings (after remediation)

---

## 7. Performance Benchmarks

### 7.1 Latency Measurements

**Test Environment**: AWS c5.xlarge instances, global deployment

| Operation | P50 Latency | P95 Latency | P99 Latency |
|-----------|------------|------------|------------|
| Tor Circuit Establishment | 1.2s | 2.8s | 4.5s |
| Ring Signature Generation | 45ms | 120ms | 250ms |
| Stealth Address Creation | 5ms | 12ms | 25ms |
| Transaction Mixing (standard) | 15s | 30s | 45s |
| Transaction Mixing (maximum) | 45s | 90s | 120s |
| ZK-STARK Proof Generation | 180ms | 350ms | 600ms |
| ZK-STARK Verification | 25ms | 50ms | 80ms |

### 7.2 Throughput Benchmarks

**Mixing Service Capacity** (single datacenter):
- **Standard privacy**: 500 transactions/minute
- **Maximum privacy**: 150 transactions/minute

**API Gateway Throughput**:
- **Professional tier**: 10,000 requests/second
- **Enterprise tier**: 50,000 requests/second

**Network Scalability**:
- Current node count: **1,200+ nodes**
- Transaction propagation time: **<2 seconds** (95th percentile)
- Global message propagation latency (P2P overlay): **<5 seconds**

### 7.3 Cost Efficiency

**Comparison with Competitors**:

| Service | Q-NarwhalKnight PaaS | Tornado Cash | Zcash Shielded |
|---------|---------------------|--------------|----------------|
| Mixing 1 BTC | $60 (0.1%) | $37 (0.05%) | N/A (on-chain) |
| Mixing 10 ETH | $30 (0.1%) | $25 (0.05%) | N/A |
| Ring Signature | $0.01 | N/A | N/A |
| ZK-STARK Proof | $0.10 | N/A | N/A |
| Tor Relay (100 MB) | $1.00 | N/A | N/A |

**Note**: Q-NarwhalKnight higher cost justified by:
- Cross-chain support
- Quantum resistance
- Enterprise SLAs
- Compliance features

---

## 8. Use Cases

### 8.1 Bitcoin Privacy Enhancement

**Problem**: Bitcoin's transparent ledger enables tracking of all transactions

**Solution**: Bitcoin wallets integrate Q-NarwhalKnight PaaS

```javascript
// Bitcoin wallet integration example
const qnkClient = new QNarwhalKnightSDK({ apiKey: 'your_api_key' });

// User wants to send 0.5 BTC privately
const privateTx = await qnkClient.mixBitcoinTransaction({
  from: 'bc1q...',
  to: 'bc1q...',
  amount: '0.5 BTC',
  privacyLevel: 'high'
});

// Broadcast via Tor
await qnkClient.broadcastViaTor(privateTx);
```

**Benefits**:
- ✅ IP address hidden (Tor)
- ✅ Transaction origin unlinkable (mixing)
- ✅ No address reuse (stealth addresses)

### 8.2 Ethereum MEV Protection

**Problem**: Frontrunning bots extract value from user transactions ($500M+ annually)

**Solution**: Submit transactions through Q-NarwhalKnight Tor relay + Private Relay Network

**Technical Approach**:
- **Tor Layer**: Hide originating IP address
- **Private Relays**: Integration with MEV-Share, Flashbots Protect, bloXroute, Blocknative
- **Randomized Timing**: Introduce jitter to prevent timing correlation attacks

**Implementation**:
```javascript
// Ethereum DEX trade with MEV protection
const trade = await uniswap.swapExactTokensForTokens(...);

// Route through Q-NarwhalKnight PaaS with private relay
const protectedTrade = await qnkClient.protectFromMEV({
  chain: 'ethereum',
  transaction: trade,
  private_relay: {
    provider: 'flashbots',  // Options: flashbots, mev-share, bloxroute, blocknative
    submission_mode: 'randomized_timing',  // Introduce 0-5 second random delay
    rpc_endpoint: 'https://rpc.flashbots.net'  // Or bloXroute, etc.
  },
  tor_relay: true,  // Submit via Tor for IP anonymity
  ring_signature: false  // Optional: quantum-resistant authorization
});
```

**Private Relay Providers Supported**:

| Provider | MEV Protection | Latency | Cost | Status |
|----------|---------------|---------|------|--------|
| **Flashbots Protect** | ✅ Strong | ~500ms | Free | ✅ Integrated |
| **MEV-Share** | ✅ Profit sharing | ~400ms | % rebate | ✅ Integrated |
| **bloXroute** | ✅ Enterprise | ~300ms | $99/mo | 📋 Planned Q1 2026 |
| **Blocknative** | ✅ Gas optimization | ~350ms | Custom | 📋 Planned Q1 2026 |

**Benefits**:
- ✅ Transaction origin hidden (Tor + private relay prevents IP-based targeting)
- ✅ Mempool bypass (transactions sent directly to block builders via private channels)
- ✅ Randomized timing (0-5 second jitter prevents timing correlation)
- ✅ Optional profit sharing (MEV-Share rebates extraction value to users)
- ✅ Ring signature authorization (experimental, full quantum-resistant privacy)

### 8.3 Cross-Chain Atomic Swaps

**Problem**: Swapping BTC ↔ ETH reveals both wallet addresses

**Solution**: Privacy-preserving atomic swaps

```javascript
// Swap 1 BTC for 25 ETH privately
const swap = await qnkClient.createAtomicSwap({
  chainA: {
    network: 'bitcoin',
    amount: '1 BTC',
    stealthAddress: true
  },
  chainB: {
    network: 'ethereum',
    amount: '25 ETH',
    stealthAddress: true
  },
  torRelay: true,
  timelock: 24  // hours
});
```

**Benefits**:
- ✅ Both parties use stealth addresses
- ✅ Swap transaction broadcast via Tor
- ✅ No centralized exchange required

### 8.4 Enterprise Payment Privacy

**Scenario**: Corporation pays suppliers without revealing financial relationships

```javascript
// Enterprise batch payment with privacy
const payments = await qnkClient.enterprise.batchPayments({
  payments: [
    { to: 'supplier1', amount: '100000 USDC', chain: 'ethereum' },
    { to: 'supplier2', amount: '75000 USDC', chain: 'ethereum' },
    { to: 'supplier3', amount: '50000 USDC', chain: 'ethereum' }
  ],
  privacyLevel: 'maximum',
  complianceMode: true,  // Enable selective disclosure
  regulatoryJurisdiction: 'US'
});
```

**Benefits**:
- ✅ Suppliers cannot see other payment amounts
- ✅ Competitors cannot analyze spending patterns
- ✅ Compliant with FATF Travel Rule (selective disclosure)

---

## 9. Competitive Analysis

### 9.1 Feature Comparison Matrix

| Feature | Q-NarwhalKnight PaaS | Tornado Cash | Aztec | Zcash | Monero |
|---------|---------------------|--------------|-------|-------|--------|
| **Cross-Chain Support** | ✅ BTC, ETH, SOL | ❌ ETH only | ❌ ETH only | ❌ Zcash only | ❌ XMR only |
| **Quantum Resistance** | ✅ Dilithium5 | ❌ ECDSA | ⚠️ Partial | ❌ ECDSA | ❌ EdDSA |
| **Tor Integration** | ✅ Native | ❌ No | ❌ No | ⚠️ Optional | ⚠️ Optional |
| **Enterprise SLA** | ✅ 99.9% | ❌ No | ❌ No | ❌ No | ❌ No |
| **Compliance Mode** | ✅ Selective Disclosure | ❌ Banned | ⚠️ Limited | ✅ View Keys | ❌ No |
| **ZK-STARK Proofs** | ✅ Yes | ❌ No | ✅ Yes | ❌ SNARKs only | ❌ No |
| **Ring Signatures** | ⚠️ Experimental | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Stealth Addresses** | ✅ Yes | ❌ No | ❌ No | ⚠️ t-addr only | ✅ Yes |
| **API Access** | ✅ RESTful | ❌ Smart contract only | ❌ SDK only | ❌ RPC only | ❌ RPC only |
| **Decoy Transactions** | ✅ 15x ratio | ❌ No | ❌ No | ❌ No | ⚠️ Limited |

### 9.2 Regulatory Status & Legal Considerations

**⚠️ Legal Disclaimer**: The regulatory status of privacy-enhancing technologies varies by jurisdiction and is subject to change. Q-NarwhalKnight PaaS is designed with compliance controls, but licensing requirements depend on deployment model, feature set, and jurisdiction. Consult qualified legal counsel before deployment.

| Service | Regulatory Posture | Risk Assessment | Licensing Considerations |
|---------|-------------------|-----------------|-------------------------|
| Q-NarwhalKnight PaaS | **Designed for compliance** (controls available) | ⚠️ **Jurisdiction-dependent** | May require MSB/MTL/VASP licensing |
| Tornado Cash | 🚫 **Sanctioned** (OFAC August 2022) | 🔴 **High** | Prohibited in most jurisdictions |
| Aztec | ⚠️ Under regulatory scrutiny | 🟡 **Medium-High** | Unclear licensing path |
| Zcash | ✅ Compliant (optional shielded txs) | 🟢 **Low-Medium** | Generally permissible |
| Monero | ⚠️ Delisting risk | 🟡 **Medium** | Restricted by some exchanges/jurisdictions |

#### Regulatory Compliance Matrix

**United States**:
- **FinCEN MSB Registration**: Likely required if operating as a money services business
- **State MTL Licenses**: May require Money Transmitter Licenses in 48+ states
- **OFAC Compliance**: Sanctions screening mandatory (KYT/KYO integrated)
- **Travel Rule**: FATF guidance implemented via IVMS-101
- **Status**: Controls available; consult counsel for specific deployment

**European Union**:
- **5AMLD/6AMLD**: Compliance required for CASP (Crypto Asset Service Provider) designation
- **MiCA Regulation**: Monitoring for applicability (effective 2024)
- **GDPR**: Privacy-by-design architecture compatible
- **Travel Rule**: IVMS-101 support for inter-VASP transfers >€1,000
- **Status**: Designed for compliance; CASP licensing may be required

**United Kingdom**:
- **FCA Registration**: Required for cryptoasset firms under MLR 2017
- **Travel Rule**: HM Treasury guidance compliance via TRP/IVMS-101
- **AML/CTF**: Comprehensive KYT/sanctions screening integrated
- **Status**: Controls align with FCA expectations; registration required

**Asia-Pacific**:
- **Singapore (MAS)**: DPT Act licensing may apply for payment services
- **Japan (FSA)**: VASP registration required under Payment Services Act
- **Hong Kong (SFC)**: Licensing regime for VATPs (Virtual Asset Trading Platforms)
- **Australia (AUSTRAC)**: DCE registration + AML/CTF compliance
- **Status**: Jurisdiction-specific licensing required

#### Q-NarwhalKnight Compliance Design Principles

**Built-in Controls (Not Guarantees)**:
- ✅ Optional compliance mode (users choose privacy level)
- ✅ View key escrow for regulatory requests (threshold M-of-N)
- ✅ Threshold cryptography (no single point of control)
- ✅ Jurisdictional filtering (block sanctioned countries)
- ✅ KYT/KYO sanctions screening (OFAC, UN, EU)
- ✅ Travel Rule infrastructure (TRP, IVMS-101, OpenVASP)
- ✅ Audit trails with ZK-attested compliance proofs
- ✅ Configurable risk policies per jurisdiction

**Operator Responsibilities**:
- ⚠️ Obtain appropriate licenses (MSB, MTL, VASP, etc.)
- ⚠️ Implement KYC/AML procedures for high-risk transactions
- ⚠️ Maintain compliance officer and legal counsel
- ⚠️ File SARs (Suspicious Activity Reports) per local law
- ⚠️ Conduct regular compliance audits
- ⚠️ Monitor regulatory developments

**Recommended Use Cases by Risk Tolerance**:

| Use Case | Compliance Mode | Licensing Needs | Risk Level |
|----------|----------------|-----------------|------------|
| **Enterprise-hosted (white-label)** | Mandatory | Full MSB/VASP suite | 🟢 Lowest |
| **Regulated institution integration** | Recommended | Covered under parent licenses | 🟢 Low |
| **Public API (KYC'd users)** | Optional but recommended | MSB + State MTLs | 🟡 Medium |
| **Public API (anonymous users)** | Not enforceable | Licensing uncertain/risky | 🔴 High |

### 9.3 Market Positioning

**Target Market Segments**:

1. **Privacy-Conscious Individuals** (Pay-Per-Use tier)
   - Estimated market: 10M cryptocurrency users
   - Conversion rate: 5% → 500K potential customers
   - ARPU: $20/month → $10M monthly revenue

2. **Wallets & DApps** (Professional tier)
   - Estimated market: 5,000 applications
   - Conversion rate: 10% → 500 customers
   - ARPU: $499/month → $250K monthly revenue

3. **Exchanges & Institutions** (Enterprise tier)
   - Estimated market: 500 platforms
   - Conversion rate: 20% → 100 customers
   - ARPU: $1,999/month → $200K monthly revenue

4. **Blockchain Networks** (White-Label tier)
   - Estimated market: 50 major chains
   - Conversion rate: 10% → 5 customers
   - ARPU: $9,999/month → $50K monthly revenue

**Total Addressable Market (TAM)**: $10.5M monthly revenue ≈ **$126M annual revenue**

---

## 10. Roadmap

### Phase 1: Foundation (Q4 2025) ✅

**Status**: Complete

- ✅ Core privacy infrastructure (Tor, mixing, ring signatures)
- ✅ RESTful API with authentication
- ✅ Bitcoin & Ethereum transaction support
- ✅ Pay-Per-Use tier launch

### Phase 2: Enterprise Features (Q1 2026) 🔄

**Status**: In Progress

- 🔄 Professional & Enterprise tier launch
- 🔄 Compliance mode (selective disclosure)
- 🔄 Batch processing API
- 🔄 Solana transaction support
- 🔄 SDKs (JavaScript, Python, Rust)

### Phase 3: Advanced Cryptography (Q2 2026) 📋

**Planned Features**:
- Recursive ZK-STARK proofs (aggregate multiple proofs)
- Confidential transactions (hide amounts, not just senders)
- Verifiable delay functions (VDF) for fairness
- Lattice-based VRF (quantum-resistant randomness)

### Phase 4: Global Scale (Q3 2026) 📋

**Planned Features**:
- Multi-region deployment (US, EU, Asia)
- 10,000+ node network
- 10,000 transactions/minute mixing capacity
- White-Label tier launch

### Phase 5: Quantum Era (Q4 2026+) 📋

**Future-Proofing**:
- Phase 2 cryptography (Falcon1024, NTRUPrime)
- Quantum Key Distribution (QKD) integration
- Post-quantum Tor (Noise-KK with Kyber)
- Quantum-resistant atomic swaps

---

## Conclusion

Q-NarwhalKnight Privacy-as-a-Service represents a **paradigm shift** in blockchain privacy:

**Traditional Approach**: Privacy features built into individual chains (Zcash, Monero)
**Q-NarwhalKnight Approach**: **Universal privacy layer** serving all blockchains via API

**Key Advantages**:
1. **Cross-Chain**: Works with Bitcoin, Ethereum, Solana, and future chains
2. **Quantum-Resistant**: Future-proof cryptography (Dilithium5, Kyber1024)
3. **Enterprise-Ready**: SLA guarantees, compliance features, dedicated support
4. **Economically Sustainable**: Pay-per-use + subscription revenue model

**Market Opportunity**:
- $126M annual revenue potential (TAM)
- First-mover advantage in cross-chain privacy
- Regulatory compliant (avoids Tornado Cash fate)

**Call to Action**:
- **Developers**: Integrate Q-NarwhalKnight PaaS SDK into your wallet/DApp
- **Enterprises**: Schedule demo for Enterprise tier onboarding
- **Blockchain Networks**: Explore white-label privacy infrastructure

**Learn More**:
- API Documentation: https://docs.qnarwhalknight.com/paas
- Developer Portal: https://developers.qnarwhalknight.com
- Enterprise Sales: enterprise@qnarwhalknight.com

---

**Appendix A: Benchmark Methodology**

All performance benchmarks in Section 7 were conducted using standardized methodology:

**Hardware Configuration**:
- **CPU**: AWS c5.xlarge (4 vCPUs, Intel Xeon Platinum 8124M @ 3.0 GHz)
- **RAM**: 8 GB DDR4
- **Storage**: 100 GB gp3 SSD (3000 IOPS, 125 MB/s throughput)
- **Network**: 10 Gbps network interface
- **OS**: Ubuntu 22.04 LTS (Linux kernel 5.15)

**Test Environment**:
- **Deployment**: Multi-region (us-east-1, eu-west-1, ap-southeast-1)
- **Load Balancer**: Application Load Balancer (ALB) with TLS termination
- **Database**: PostgreSQL 15 (db.m5.large, 1000 IOPS provisioned)
- **Caching**: Redis 7.0 (cache.m5.large)
- **Monitoring**: Prometheus + Grafana

**Benchmark Parameters**:

**Tor Circuit Establishment**:
- Tor version: 0.4.7.13 (arti embedded client)
- Circuit type: 3-hop circuits (guard → middle → exit)
- Country exclusions: None (baseline test)
- Quantum seeding: Enabled (QRNG circuit selection)
- Measurement: Time from `CreateCircuit` call to `CircuitReady` event
- Sample size: 10,000 circuit establishments
- Test duration: 24 hours (continuous monitoring)

**Ring Signature Generation**:
- Algorithm: Lattice-based linkable ring signatures (experimental)
- Ring sizes tested: 8, 16, 32, 64 members
- Reported metrics: Ring size 16 (default)
- Message size: 32 bytes (SHA-256 hash)
- Concurrency: Single-threaded (no parallelization)
- Sample size: 100,000 signatures

**Transaction Mixing**:
- Mixing pool sizes: 8, 16, 32 participants
- Reported metrics: 16 participants (standard)
- Decoy ratio: 15 decoys per participant
- Privacy level: "standard" (not "maximum")
- Includes: Network latency + Tor relay + mixing rounds + ZK proof generation + broadcast
- Does NOT include: On-chain confirmation time (blockchain-dependent)
- Sample size: 5,000 mixing rounds

**ZK-STARK Proof Generation**:
- Proof type: STARK (no trusted setup)
- Circuit type: `balance_in_range` (most common use case)
- Circuit complexity: ~10,000 constraints
- Security level: 128-bit
- Prover implementation: Winterfell library v0.6
- Field: 64-bit prime field (2^64 - 2^32 + 1)
- Batch size: 1 proof per measurement (no batching)
- Sample size: 50,000 proofs

**API Gateway Throughput**:
- Load testing tool: k6 (v0.45.0)
- Test scenario: Mixed workload (50% reads, 30% mixing, 20% signature gen)
- Request distribution: Zipfian (realistic traffic pattern)
- Authentication: Pre-generated API keys (minimal overhead)
- TLS: TLS 1.3 with ChaCha20-Poly1305
- Measurement: Requests/second at P95 latency < 500ms
- Test duration: 1 hour sustained load

**Network Scalability**:
- Node distribution: 1,200 nodes across 50 geographic locations
- Node hardware: Mixture of c5.large (60%), c5.xlarge (30%), c5.2xlarge (10%)
- Transaction propagation: Gossipsub with D=6, D_low=4, D_high=12
- Measurement: Time from first broadcast to 95% node reception
- Sample size: 10,000 transactions

**Performance Number Interpretation**:
- **P50 (median)**: 50% of operations complete within this time
- **P95**: 95% of operations complete within this time (SLA basis)
- **P99**: 99% of operations complete within this time (outlier detection)

**Inclusions & Exclusions**:
- ✅ **Includes**: Network latency, Tor relay overhead, cryptographic operations, database queries
- ✅ **Includes**: TLS handshake overhead (amortized across requests)
- ❌ **Excludes**: On-chain confirmation times (blockchain-dependent)
- ❌ **Excludes**: Client-side processing (wallet signature generation, etc.)
- ❌ **Excludes**: External API calls (Chainalysis KYT, etc.)

**Reproducibility**:
- Benchmark scripts: https://github.com/q-narwhalknight/paas-benchmarks
- Dataset: Synthetic transaction data (no real user data)
- Environment setup: Infrastructure-as-Code (Terraform scripts provided)

---

**Appendix B: Cryptographic Specifications**

**Dilithium5 Parameters**:
- Security level: NIST Level 5 (256-bit quantum security)
- Public key size: 2,592 bytes
- Signature size: 4,595 bytes
- Signing time: ~1.5ms
- Verification time: ~0.5ms

**Kyber1024 Parameters**:
- Security level: NIST Level 5
- Public key size: 1,568 bytes
- Ciphertext size: 1,568 bytes
- Encapsulation time: ~0.3ms
- Decapsulation time: ~0.4ms

**ZK-STARK Proof Parameters**:
- Security level: 128-bit
- Proof size: ~45 KB (typical)
- Prover time: ~250ms
- Verifier time: ~50ms
- No trusted setup

**Appendix C: Enhanced Kademlia DHT Security**

**Issue**: Standard Kademlia DHT with proof-of-work peer IDs can be gamed via precomputed vanity IDs, and Ed25519-only peer IDs are not quantum-resistant.

**Solution**: Hybrid Peer Identity System

**Hybrid Peer ID Construction**:
```
peer_id = SHA3-256(ed25519_pubkey || dilithium5_pubkey || resource_proof)
```

**Components**:
1. **Ed25519 Public Key** (32 bytes): Classical signature capability
2. **Dilithium5 Public Key** (2,592 bytes): Post-quantum signature capability
3. **Resource Proof** (variable): Proof of bandwidth + uptime (not just PoW)

**Resource Proof Requirements**:

Instead of simple hashcash proof-of-work (which can be precomputed), Q-NarwhalKnight requires:

```json
{
  "resource_proof": {
    "bandwidth_test": {
      "upload_mbps": 10.5,
      "download_mbps": 25.3,
      "latency_ms": 45,
      "test_server": "qnk-bw-test-001.example.com",
      "test_timestamp": "2025-10-22T12:34:56Z",
      "server_signature": "base64_signature"
    },
    "uptime_proof": {
      "node_start_time": "2025-10-15T00:00:00Z",
      "consensus_participated_rounds": 12500,
      "peer_endorsements": [
        {"peer_id": "abc123...", "endorsement_sig": "..."},
        {"peer_id": "def456...", "endorsement_sig": "..."}
      ]
    },
    "stake_proof": {
      "staked_qnk": "1000.0",
      "stake_address": "qnk1...",
      "stake_duration_days": 90
    }
  }
}
```

**Resource Proof Validation**:
- Bandwidth test must be recent (<24 hours) and signed by authorized test server
- Uptime proof validated against consensus history (can't be faked)
- Peer endorsements require established nodes to vouch for newcomers
- Stake proof (optional): Economic cost of Sybil attacks

**Peer ID Rotation Without Identity Loss**:

Ed25519 keys can be rotated for operational security while maintaining PQ identity:

```
new_peer_id = SHA3-256(new_ed25519_pubkey || SAME_dilithium5_pubkey || resource_proof)

Certificate signed by old key:
cert = Sign_old_ed25519("Rotate to: " || new_ed25519_pubkey || timestamp)
```

Network recognizes identity continuity via unchanged Dilithium5 key.

**Sybil Resistance Analysis**:

| Attack | Standard PoW Kademlia | Q-NarwhalKnight Hybrid |
|--------|---------------------|----------------------|
| **Precomputed vanity IDs** | ✅ Easy (GPU farms) | ❌ Blocked (bandwidth/uptime tests) |
| **Instant Sybil nodes** | ✅ Easy (spin up VMs) | ❌ Blocked (uptime proof required) |
| **Eclipse attack** | ⚠️ Moderate difficulty | ❌ Very difficult (diverse bootstrap + endorsements) |
| **Resource exhaustion** | ⚠️ Moderate cost | ❌ High cost (stake requirement) |

**Migration Path**:
- **Phase 0**: Ed25519-only peer IDs (current)
- **Phase 1**: Hybrid Ed25519 + Dilithium5 (in progress)
- **Phase 2**: Dilithium5-primary, Ed25519 for backwards compat
- **Phase 3**: Pure Dilithium5 peer IDs (post-quantum era)

---

**Appendix D: API Idempotency & Error Handling**

**Problem**: Without idempotency keys, network failures can cause duplicate transactions or lost requests.

**Solution**: RFC-compliant idempotency keys + standardized error model (RFC 9457 Problem Details)

#### Idempotency Key Protocol

**Client Request**:
```http
POST /api/v1/privacy/mix/submit
Authorization: MultiSig v=1; ...
Idempotency-Key: client-generated-uuid-12345
Content-Type: application/json

{
  "chain": "ethereum",
  "transaction_data": {...}
}
```

**Server Behavior**:
1. Store `Idempotency-Key` + request hash + response in database
2. If same key sent within 24 hours → return cached response (no duplicate operation)
3. If same key with different request body → return `409 Conflict`
4. Key expires after 24 hours (client should retry with new key)

**Response Headers**:
```http
HTTP/1.1 200 OK
Idempotency-Key: client-generated-uuid-12345
X-Idempotency-Status: hit  // "miss" for first request, "hit" for repeated requests
X-Original-Request-Time: 2025-10-22T12:34:56Z
Content-Type: application/json
```

**Idempotency Guarantees**:

| Operation | Idempotent | Notes |
|-----------|-----------|-------|
| **GET** requests | ✅ Naturally idempotent | No key required |
| **POST /mix/submit** | ✅ With key | Critical for preventing duplicate mixes |
| **POST /ring-signature/generate** | ✅ With key | Same input → same signature |
| **POST /tor/relay** | ⚠️ Partial | Data relayed once, response cached |
| **POST /zk-stark/prove** | ✅ With key | Deterministic proofs |

#### Standardized Error Model (RFC 9457)

**Error Response Format**:
```http
HTTP/1.1 400 Bad Request
Content-Type: application/problem+json

{
  "type": "https://docs.qnarwhalknight.com/errors/insufficient-balance",
  "title": "Insufficient QNK Balance",
  "status": 400,
  "detail": "Your account has 0.5 QNK but this operation requires 1.0 QNK",
  "instance": "/api/v1/privacy/mix/submit/request-abc123",
  "balance_qnk": "0.5",
  "required_qnk": "1.0",
  "top_up_url": "https://exchange.example.com/buy-qnk",
  "trace_id": "7d8e9f0a-1b2c-3d4e-5f6a-7b8c9d0e1f2"
}
```

**Standard Error Types**:

| Error Type | HTTP Status | Description |
|-----------|-------------|-------------|
| `insufficient-balance` | 400 | Not enough QNK tokens |
| `rate-limit-exceeded` | 429 | Too many requests |
| `tor-circuit-failed` | 503 | Cannot establish Tor circuit |
| `mixing-pool-timeout` | 504 | Mixing pool did not fill in time |
| `idempotency-mismatch` | 409 | Same key, different request body |
| `authentication-failed` | 401 | Signature verification failed |
| `authorization-required` | 403 | API key lacks permissions |
| `sanctions-screening-failed` | 451 | Address blocked by compliance policy |
| `invalid-request-format` | 422 | Malformed request body |

**Request Canonicalization** (for idempotency):

To ensure identical requests have identical hashes:

```javascript
// 1. Sort JSON keys alphabetically
const canonical_body = JSON.stringify(request_body, Object.keys(request_body).sort());

// 2. Reject unknown fields
const allowed_fields = ["chain", "transaction_data", "privacy_level", "mixing_parameters"];
for (const key in request_body) {
  if (!allowed_fields.includes(key)) {
    throw new Error(`Unknown field: ${key}`);
  }
}

// 3. Compute SHA3-256 hash
const request_hash = SHA3_256(canonical_body);
```

**Job State Webhooks** (Enterprise tier):

```http
POST https://customer-webhook.example.com/qnk-status
Content-Type: application/json

{
  "job_id": "qnk-mix-abc123",
  "job_type": "transaction_mixing",
  "state": "broadcasting",  // queued → mixing → broadcasting → confirmed → failed
  "state_timestamp": "2025-10-22T12:35:30Z",
  "previous_state": "mixing",
  "metadata": {
    "anonymity_set_size": 320,
    "mixing_pool_id": "pool-eth-large-0042",
    "estimated_confirmation_time": "2025-10-22T12:36:00Z"
  },
  "idempotency_key": "client-generated-uuid-12345"
}
```

**Webhook State Machine**:
```
queued → mixing → broadcasting → confirmed
           ↓          ↓             ↓
         failed ← failed ← failed
```

#### Request-Level Privacy Budgets

**Privacy Tradeoff Hints**:

When a client requests "maximum privacy," the API returns expected tradeoffs:

```json
{
  "privacy_level": "maximum",
  "estimated_tradeoffs": {
    "latency_increase_percent": 200,  // 3× slower than standard
    "fee_increase_percent": 50,       // 1.5× more expensive
    "anonymity_set_increase": 400,    // 4× larger anonymity set
    "epsilon_privacy_loss": 0.7,      // ε ≈ 0.7 (strong privacy)
    "recommendation": "Acceptable for high-value transactions"
  }
}
```

Clients can make informed decisions about privacy vs cost/latency.

---

**Appendix E: Cross-Chain Implementation Details**

#### Bitcoin (PSBT Flow)

**Partially Signed Bitcoin Transactions (BIP 174)**:

```
Step 1: Client creates PSBT
┌─────────────────────────────────────────────┐
│ PSBT Version 2                              │
│ ├─ Input 0: bc1q... (1.5 BTC)              │
│ ├─ Output 0: <stealth_addr_1> (0.7 BTC)   │
│ ├─ Output 1: <stealth_addr_2> (0.8 BTC)   │
│ └─ Fee: 0.0001 BTC                         │
└─────────────────────────────────────────────┘

Step 2: Submit to Q-NarwhalKnight mixing
POST /api/v1/privacy/mix/submit
{
  "chain": "bitcoin",
  "psbt_base64": "cHNidP8BAHE...==",
  "mixing_parameters": {
    "participants_min": 16,
    "amount_bucket": "1.0-5.0 BTC",
    "change_handling": "new_stealth_address"
  }
}

Step 3: Mixing coordinator combines PSBTs
┌───────────────────────────────────────────────┐
│ Combined CoinJoin Transaction                 │
│ ├─ Input 0-15: 16 participants' inputs       │
│ ├─ Output 0-31: 32 stealth addresses         │
│ │   (16 payments + 16 change outputs)         │
│ └─ Fee: 0.0016 BTC (shared among 16 users)  │
└───────────────────────────────────────────────┘

Step 4: Each participant signs their input
- Q-NarwhalKnight returns partially-signed PSBT to each client
- Clients add their signatures (ECDSA or Schnorr for Taproot)
- Coordinator collects all signatures

Step 5: Broadcast via Tor
- Final transaction broadcast through controlled egress relays
- Randomized timing (0-30s delay per output)
```

**Bitcoin-Specific Parameters**:
- **Mixing rounds**: 2-4 rounds (more rounds = better privacy, longer latency)
- **Minimum participants**: 8 (standard), 16 (recommended), 32 (maximum)
- **Amount buckets**: Enforce bucket ranges to prevent unique amount linkability
- **Change handling**: Always create new stealth address for change outputs
- **Fee rate selection**: Dynamic fee estimation (mempool.space API integration)
- **RBF (Replace-By-Fee)**: Disabled during mixing (prevents fee sniping attacks)

#### Ethereum (RLP-Encoded Transactions)

**EIP-1559 Transaction Format**:

```javascript
const eth_tx = {
  chainId: 1,  // Mainnet
  nonce: 42,
  maxPriorityFeePerGas: '2000000000',  // 2 gwei
  maxFeePerGas: '50000000000',         // 50 gwei
  gasLimit: '21000',
  to: '0x8ba1f109551bD432803012645Ac136ddd64DBA72',  // Recipient stealth address
  value: '1500000000000000000',  // 1.5 ETH
  data: '0x',
  accessList: [],
  type: 2  // EIP-1559
};

// Submit to Q-NarwhalKnight
POST /api/v1/privacy/mix/submit
{
  "chain": "ethereum",
  "transaction_data": eth_tx,
  "mixing_parameters": {
    "ring_signatures_experimental": false,  // Use standard mixing
    "mev_protection": {
      "enabled": true,
      "private_relay": "flashbots",
      "randomized_timing_seconds": 5
    },
    "eip4844_blobs": false  // Not using blob transactions
  }
}
```

**Ethereum-Specific Considerations**:
- **EIP-1559 fields**: Must specify `maxPriorityFeePerGas` and `maxFeePerGas`
- **EIP-4844 blob transactions**: Supported (mixing blobs separately from calldata)
- **Contract interactions**: Supported but privacy reduced (function selectors visible)
- **Private relay integration**: Flashbots, MEV-Share, bloXroute for MEV protection
- **Gas price oracle**: ChainLink or internal oracle for gas estimation
- **Nonce management**: Client maintains nonce; mixing does not reorder transactions

#### Solana (Borsh-Serialized Transactions)

**Solana Transaction Structure**:

```javascript
const sol_tx = {
  recentBlockhash: '4sGjMW1sUnHzSxGspuhpqLDx6wiyjNtZ',
  feePayer: 'sender_pubkey',
  instructions: [
    {
      programId: 'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA',  // SPL Token program
      keys: [
        { pubkey: 'sender_token_account', isSigner: true, isWritable: true },
        { pubkey: 'recipient_stealth_account', isSigner: false, isWritable: true },
        { pubkey: 'sender_authority', isSigner: true, isWritable: false }
      ],
      data: Buffer.from([3, ...amount_bytes])  // Transfer instruction
    }
  ],
  signatures: []
};

POST /api/v1/privacy/mix/submit
{
  "chain": "solana",
  "transaction_data": borsh.serialize(TransactionSchema, sol_tx),
  "mixing_parameters": {
    "ring_signatures_partial": true,  // Experimental Solana ring sigs
    "priority_fee_lamports": 5000
  }
}
```

**Solana Limitations**:
- **Ring signatures**: ⚠️ Partial support (not all Solana programs support custom auth)
- **Mixing**: ✅ Full support via token account mixing
- **Stealth addresses**: ✅ Full support (derived PDAs)
- **Priority fees**: Required for timely inclusion
- **Blockhash expiry**: Transactions must be broadcast within ~60 seconds

---

**Appendix F: Threat Model & Abuse Handling**

#### Adversary Model

**In-Scope Adversaries**:
1. **Network Surveillance** (NSA/GCHQ-level): Passive traffic analysis, Tor deanonymization attempts
2. **Blockchain Analytics Firms** (Chainalysis, Elliptic): Transaction graph analysis, heuristics
3. **Malicious Mixing Participants**: Sybil attacks on mixing pools, timing analysis
4. **Compromised Tor Nodes**: Malicious exit nodes, guard/middle relays
5. **Quantum Computers** (future): Attacks on ECDSA/EdDSA signatures

**Out-of-Scope Adversaries**:
1. **Global Passive Adversary**: Cannot monitor all internet traffic simultaneously (unrealistic)
2. **Endpoint Compromise**: Keyloggers, malware on client devices (out of protocol scope)
3. **Social Engineering**: Phishing, extortion (user security responsibility)

**Attack Resistance Table**:

| Attack | Mitigation | Residual Risk |
|--------|-----------|--------------|
| **IP tracking** | Tor + mDNS local-only | 🟢 Low (Tor guard compromise <1%) |
| **Transaction graph analysis** | Ring signatures + mixing | 🟢 Low (ε < 1 for max privacy) |
| **Timing correlation** | Randomized delays | 🟡 Medium (advanced attackers may correlate) |
| **Amount linkability** | Amount bucketing | 🟢 Low (uniform buckets) |
| **Sybil attacks on mixing** | Resource proofs + endorsements | 🟡 Medium (costly but possible) |
| **MEV extraction** | Private relays + timing jitter | 🟢 Low (Flashbots protection) |
| **Quantum computer attacks** | Dilithium5 + Kyber1024 | 🟢 Low (NIST PQC standards) |
| **Tor exit monitoring** | End-to-end encryption | 🟢 Low (Noise Protocol) |
| **Eclipse attack on DHT** | Diverse bootstrap + hybrid IDs | 🟡 Medium (requires many resources) |

#### Abuse Response Policy

**Griefing Attacks** (jamming mixing pools):

**Problem**: Malicious users submit transactions but never provide signatures, blocking pool progression.

**Mitigation**:
- **Deposit requirement**: Lockup 0.01 QNK per mixing request (refunded on completion)
- **Timeout enforcement**: If user doesn't sign within 60 seconds, deposit forfeited
- **Reputation system**: Track completion rate per wallet (ban wallets with <80% completion)
- **Pool isolation**: Separate pools for high-reputation vs new users

**Spam/DoS Attacks**:

**Problem**: Attacker floods API with bogus requests to exhaust resources.

**Mitigation**:
- **Proof-of-work stamps**: Anonymous tier requires PoW stamp (difficulty=20 bits, ~1 second compute)
- **Rate limiting**: Progressive backoff (1st violation: 60s ban, 2nd: 1 hour, 3rd: 24 hours)
- **API key quotas**: Enforce monthly request limits per tier
- **DDoS protection**: Cloudflare + AWS Shield at edge

**Example PoW Stamp**:
```http
POST /api/v1/privacy/mix/submit
X-Proof-Of-Work: nonce=123456789; difficulty=20; hash=00000abcdef...
```

Server verifies: `SHA3-256(request_body || nonce)` has 20 leading zero bits.

**Constant-Rate Cover Traffic**:

To prevent traffic analysis, Q-NarwhalKnight nodes maintain constant bandwidth usage:

```
Target bandwidth: 1 Mbps upload, 1 Mbps download (configurable)

If real traffic < target:
  Generate dummy traffic (encrypted random data)
  Send to random peers via Tor circuits

Adversary cannot distinguish:
  - Real mixing transactions
  - Dummy cover traffic
```

**Bursting Rules**:
- Allow bursts up to 5× target bandwidth for 60 seconds
- Sustained traffic >2× target triggers rate limiting
- Cover traffic generation suspended during legitimate bursts

**Incident Response Runbook**:

**Scenario 1: Sanctions Screening Hit**
```
1. KYT API flags address as sanctioned (OFAC match)
2. Automatically reject transaction with error code 451
3. Log incident (address, timestamp, sanctions list)
4. File Suspicious Activity Report (SAR) if required by jurisdiction
5. No funds transferred (transaction never entered mixing pool)
```

**Scenario 2: Mixing Pool Sybil Attack Detected**
```
1. Monitoring detects >50% of pool participants from same ASN
2. Dissolve pool and refund deposits
3. Ban participant peer IDs for 24 hours
4. Adjust pool selection algorithm to enforce ASN diversity
5. Alert security team for investigation
```

**Scenario 3: Tor Exit Node Compromise Evidence**
```
1. Exit node observed tampering with traffic (TLS downgrade attempt)
2. Blacklist exit node fingerprint
3. Rotate all circuits using that exit
4. Publish incident report to Tor Bad Exit list
5. End-to-end encryption prevents data exposure
```

---

**Appendix G: Network Topology**

**Bootstrap Nodes** (globally distributed):
- US-East: 185.182.185.227:8081
- EU-West: [To be announced]
- Asia-Pacific: [To be announced]

**Kademlia DHT Configuration**:
- K-bucket size: 20
- α (concurrency): 3
- Replication factor: 20
- Lookup timeout: 60 seconds

**Gossipsub Configuration**:
- Heartbeat interval: 100ms
- D (desired peers): 6
- D_low: 4
- D_high: 12
- Fanout TTL: 60 seconds

**Appendix H: Legal & Disclaimers**

**Risk Disclosure**:
Q-NarwhalKnight PaaS is a privacy-enhancing technology tool. Users are responsible for compliance with local laws and regulations. We do not endorse or facilitate illegal activities.

**Jurisdictional Restrictions**:
Service may not be available in all jurisdictions. Sanctioned countries blocked per OFAC regulations.

**No Investment Advice**:
This whitepaper is for informational purposes only and does not constitute investment advice.

**Copyright & License**:
© 2025 Q-NarwhalKnight Foundation. All rights reserved.
Licensed under MIT License (open-source components).

---

**Document Version**: 1.1
**Last Updated**: October 22, 2025
**Next Review**: January 2026
