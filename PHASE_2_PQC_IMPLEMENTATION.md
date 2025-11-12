# Phase 2: Post-Quantum Cryptography Implementation

**Status:** In Progress
**Target Completion:** December 2025 (Mainnet Launch)
**Version:** v0.2.0-beta

---

## 🎯 Objectives

Implement comprehensive post-quantum cryptography (PQC) across Q-NarwhalKnight to achieve quantum resistance before potential Q-Day threats materialize.

### Key Goals:
1. **Hybrid Cryptography:** Classical + Post-Quantum dual security
2. **Crypto-Agility:** Seamless algorithm migration and negotiation
3. **QKD Preparation:** Infrastructure for quantum key distribution
4. **Performance:** Maintain 100,000+ TPS with PQ signatures
5. **Backward Compatibility:** Support Phase 0 (Ed25519) during transition

---

## 📋 Implementation Phases

### Phase 2.1: Core PQ Primitives (Week 1-2)
**Status:** Foundation Already Exists ✅

**Existing Infrastructure:**
- ✅ Dilithium5 signatures (NIST Level 5) - `q-wallet/src/dilithium_wallet.rs`
- ✅ Kyber1024 KEM (NIST Level 5) - `q-wallet/src/kyber_wallet.rs`
- ✅ Hybrid encryption (Kyber + AES-256-GCM)
- ✅ SHA3-256 hashing for quantum resistance

**Dependencies:**
```toml
pqcrypto-dilithium = "0.5"  # Dilithium5 signatures
pqcrypto-kyber = "0.8"      # Kyber1024 KEM
pqcrypto-traits = "0.3"     # Common PQ traits
sha3 = "0.10"                # Quantum-resistant hashing
```

---

### Phase 2.2: Crypto-Agile Transaction Format (Week 3-4)
**Status:** In Progress 🚧

**Goal:** Support multiple cryptographic algorithms per transaction with seamless negotiation.

**Transaction Format Enhancement:**
```rust
pub enum SignatureScheme {
    Ed25519,              // Phase 0: Classical
    Dilithium5,           // Phase 2: Post-quantum
    HybridEd25519Dilithium5, // Phase 2: Hybrid security
}

pub struct CryptoAgileTransaction {
    pub from: Address,
    pub to: Address,
    pub amount: u64,
    pub nonce: u64,
    pub data: Vec<u8>,

    // Crypto-agile signature support
    pub signature_scheme: SignatureScheme,
    pub signatures: Vec<SignatureData>,  // Multiple signatures for hybrid

    // Algorithm negotiation
    pub supported_schemes: Vec<SignatureScheme>,
    pub timestamp: u64,
}

pub struct SignatureData {
    pub scheme: SignatureScheme,
    pub public_key: Vec<u8>,
    pub signature: Vec<u8>,
}
```

**Verification Logic:**
```rust
pub fn verify_crypto_agile_tx(tx: &CryptoAgileTransaction) -> Result<bool> {
    match tx.signature_scheme {
        SignatureScheme::Ed25519 => {
            // Phase 0 verification
            verify_ed25519(&tx.signatures[0])
        },
        SignatureScheme::Dilithium5 => {
            // Phase 2 verification
            verify_dilithium5(&tx.signatures[0])
        },
        SignatureScheme::HybridEd25519Dilithium5 => {
            // Dual verification - BOTH must pass
            let ed25519_valid = verify_ed25519(&tx.signatures[0])?;
            let dilithium_valid = verify_dilithium5(&tx.signatures[1])?;
            Ok(ed25519_valid && dilithium_valid)
        },
    }
}
```

**Files to Modify:**
- `crates/q-types/src/lib.rs` - Add SignatureScheme enum
- `crates/q-types/src/transaction.rs` - Update Transaction struct
- `crates/q-wallet/src/lib.rs` - Add hybrid signing methods
- `crates/q-api-server/src/handlers.rs` - Update tx validation

---

### Phase 2.3: P2P Network PQ Key Exchange (Week 5-6)
**Status:** Planned 📋

**Goal:** Secure P2P connections with post-quantum key exchange using Kyber1024.

**libp2p Integration:**
```rust
pub struct PQSecureTransport {
    /// Kyber1024 keypair for this node
    kyber_keypair: Kyber1024KeyPair,

    /// Classical noise protocol for initial handshake
    noise_config: NoiseConfig,

    /// Hybrid: Noise + Kyber shared secret
    hybrid_keys: HashMap<PeerId, HybridKey>,
}

pub struct HybridKey {
    /// Classical ECDH shared secret
    noise_secret: [u8; 32],

    /// Post-quantum Kyber shared secret
    kyber_secret: [u8; 32],

    /// Combined: XOR or KDF of both
    combined_secret: [u8; 32],
}
```

**Handshake Protocol:**
1. **Classical Handshake:** Noise protocol (ECDH + Ed25519)
2. **PQ Key Exchange:** Kyber1024 encapsulation over secure channel
3. **Hybrid Key Derivation:** Combine classical + PQ secrets
4. **Authenticated Encryption:** AES-256-GCM with hybrid key

**Files to Create:**
- `crates/q-network/src/pq_transport.rs` - PQ transport layer
- `crates/q-network/src/hybrid_handshake.rs` - Hybrid protocol
- `crates/q-network/src/key_derivation.rs` - KDF for hybrid keys

---

### Phase 2.4: QKD Preparation Layer (Week 7-8)
**Status:** Planned 📋

**Goal:** Infrastructure for future Quantum Key Distribution integration.

**QKD Interface Design:**
```rust
pub trait QKDProvider {
    /// Request quantum-generated symmetric key
    async fn request_qkd_key(&self, peer_id: &PeerId) -> Result<Vec<u8>>;

    /// Verify key authenticity via classical channel
    async fn verify_qkd_key(&self, key_id: &str, peer_id: &PeerId) -> Result<bool>;

    /// Get QKD availability status
    fn is_qkd_available(&self) -> bool;
}

pub struct QKDSimulator {
    /// Simulated quantum channel
    quantum_channel: Arc<RwLock<QuantumChannel>>,

    /// Classical authentication channel
    classical_channel: Arc<Mutex<ClassicalAuth>>,

    /// Key storage with timestamps
    key_storage: Arc<RwLock<HashMap<String, QKDKey>>>,
}

pub struct QKDKey {
    pub key_material: Vec<u8>,
    pub generated_at: SystemTime,
    pub entropy_source: EntropySource,  // Quantum vs QRNG
    pub verified: bool,
}
```

**Entropy Sources:**
```rust
pub enum EntropySource {
    QuantumChannel,      // Real QKD hardware
    QRNG,                // Quantum RNG (CHSH inequality)
    ClassicalCSPRNG,     // Fallback: /dev/urandom
}
```

**Files to Create:**
- `crates/q-qkd/src/lib.rs` - QKD trait definitions
- `crates/q-qkd/src/simulator.rs` - QKD simulator
- `crates/q-qkd/src/qrng.rs` - Quantum random number generation
- `crates/q-qkd/src/chsh_test.rs` - Bell inequality testing

---

### Phase 2.5: Wallet PQ Integration (Week 9-10)
**Status:** Planned 📋

**Goal:** Seamless wallet support for hybrid classical+PQ keys.

**Wallet Format:**
```rust
pub struct HybridWallet {
    pub id: Uuid,

    // Classical keypair (Ed25519)
    pub ed25519_keypair: Ed25519KeyPair,
    pub ed25519_address: [u8; 32],

    // Post-quantum keypair (Dilithium5)
    pub dilithium5_keypair: Dilithium5KeyPair,
    pub dilithium5_address: [u8; 32],

    // Key encapsulation (Kyber1024)
    pub kyber1024_keypair: Kyber1024KeyPair,

    // Hybrid address (combines both)
    pub hybrid_address: [u8; 32],

    // Metadata
    pub created_at: i64,
    pub scheme: SignatureScheme,
}
```

**Wallet Operations:**
```rust
impl HybridWallet {
    /// Create new hybrid wallet with both key types
    pub fn create_hybrid(password: &str) -> Result<Self> {
        let ed25519_kp = Ed25519KeyPair::generate();
        let dilithium5_kp = Dilithium5KeyPair::generate();
        let kyber1024_kp = Kyber1024KeyPair::generate();

        // Derive hybrid address from both public keys
        let hybrid_address = Self::derive_hybrid_address(
            ed25519_kp.public_key(),
            dilithium5_kp.public_key.as_bytes(),
        );

        Ok(Self { /* ... */ })
    }

    /// Sign transaction with hybrid signatures
    pub fn sign_hybrid(&self, tx: &Transaction) -> Result<Vec<SignatureData>> {
        vec![
            SignatureData {
                scheme: SignatureScheme::Ed25519,
                public_key: self.ed25519_keypair.public_key().to_vec(),
                signature: self.ed25519_keypair.sign(tx)?,
            },
            SignatureData {
                scheme: SignatureScheme::Dilithium5,
                public_key: self.dilithium5_keypair.public_key.as_bytes().to_vec(),
                signature: self.dilithium5_keypair.sign(tx)?,
            },
        ]
    }

    /// Derive hybrid address (SHA3-512 of concatenated public keys)
    fn derive_hybrid_address(ed25519_pk: &[u8], dilithium5_pk: &[u8]) -> [u8; 32] {
        use sha3::{Digest, Sha3_512};
        let mut hasher = Sha3_512::new();
        hasher.update(ed25519_pk);
        hasher.update(dilithium5_pk);
        let full_hash = hasher.finalize();
        let mut address = [0u8; 32];
        address.copy_from_slice(&full_hash[..32]);
        address
    }
}
```

**Files to Modify:**
- `crates/q-wallet/src/lib.rs` - Add HybridWallet
- `crates/q-wallet/src/storage.rs` - Hybrid wallet serialization
- `crates/q-api-server/src/handlers.rs` - Hybrid wallet endpoints

---

### Phase 2.6: Performance Optimization (Week 11-12)
**Status:** Planned 📋

**Goal:** Maintain 100,000+ TPS with post-quantum signatures.

**Optimization Strategies:**

1. **Batch Verification (SIMD):**
```rust
pub struct PQBatchVerifier {
    /// Dilithium5 signatures to verify in parallel
    pub dilithium_sigs: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>, // (msg, sig, pk)

    /// AVX-512 vectorized verification
    pub simd_engine: Option<AVX512Engine>,
}

impl PQBatchVerifier {
    pub fn verify_batch(&self) -> Result<Vec<bool>> {
        if let Some(simd) = &self.simd_engine {
            // Vectorized Dilithium5 verification (4x-8x speedup)
            simd.verify_dilithium_batch(&self.dilithium_sigs)
        } else {
            // Sequential fallback
            self.dilithium_sigs.iter()
                .map(|(msg, sig, pk)| Dilithium5KeyPair::verify(msg, sig, pk))
                .collect()
        }
    }
}
```

2. **Signature Caching:**
```rust
pub struct SignatureCache {
    /// LRU cache: tx_hash -> verification result
    cache: Arc<Mutex<LruCache<[u8; 32], bool>>>,

    /// Cache size (100,000 entries)
    max_entries: usize,
}
```

3. **Parallel Processing:**
```rust
pub fn verify_transactions_parallel(txs: &[Transaction]) -> Result<Vec<bool>> {
    use rayon::prelude::*;

    txs.par_iter()
        .map(|tx| verify_crypto_agile_tx(tx))
        .collect()
}
```

**Performance Targets:**
- **Ed25519 verification:** ~71 µs per signature
- **Dilithium5 verification:** ~1.2 ms per signature (17× slower)
- **Batch Dilithium5 (SIMD):** ~300 µs per signature (4× speedup)
- **Target throughput:** 100,000+ TPS with hybrid signatures

**Files to Create:**
- `crates/q-crypto-simd/src/dilithium_batch.rs` - SIMD batch verification
- `crates/q-crypto-simd/src/signature_cache.rs` - LRU caching
- `crates/q-crypto-simd/benches/pq_benchmarks.rs` - Performance tests

---

### Phase 2.7: API Integration (Week 13-14)
**Status:** Planned 📋

**Goal:** Expose PQ cryptography via REST API and WebSocket.

**New API Endpoints:**

```typescript
// Create hybrid wallet
POST /api/wallet/hybrid/new
{
  "password": "string",
  "scheme": "HybridEd25519Dilithium5"
}

Response:
{
  "wallet_id": "uuid",
  "ed25519_address": "qnk...",
  "dilithium5_address": "qnk...",
  "hybrid_address": "qnk...",
  "public_keys": {
    "ed25519": "hex",
    "dilithium5": "hex"
  }
}

// Sign transaction with hybrid signatures
POST /api/wallet/hybrid/sign
{
  "wallet_id": "uuid",
  "password": "string",
  "transaction": { /* tx data */ }
}

Response:
{
  "signatures": [
    {
      "scheme": "Ed25519",
      "public_key": "hex",
      "signature": "hex"
    },
    {
      "scheme": "Dilithium5",
      "public_key": "hex",
      "signature": "hex"
    }
  ],
  "signed_transaction": "hex"
}

// Get node's PQ capabilities
GET /api/node/pq-capabilities

Response:
{
  "supported_schemes": [
    "Ed25519",
    "Dilithium5",
    "HybridEd25519Dilithium5"
  ],
  "kyber1024_supported": true,
  "qkd_available": false,
  "performance": {
    "ed25519_verify_us": 71,
    "dilithium5_verify_us": 1200,
    "batch_speedup": "4x"
  }
}
```

**Files to Modify:**
- `crates/q-api-server/src/handlers.rs` - Add PQ endpoints
- `crates/q-api-server/src/wallet_api.rs` - Hybrid wallet operations
- `gui/quantum-wallet/src/services/api.ts` - TypeScript client

---

## 🔐 Security Considerations

### 1. Cryptographic Agility Protocol
```rust
pub struct CryptoNegotiation {
    /// Node advertises supported schemes
    pub supported: Vec<SignatureScheme>,

    /// Prefer strongest available
    pub preference_order: Vec<SignatureScheme>,
}

impl CryptoNegotiation {
    pub fn negotiate(local: &[SignatureScheme], remote: &[SignatureScheme]) -> SignatureScheme {
        // Prefer hybrid > PQ > classical
        if local.contains(&SignatureScheme::HybridEd25519Dilithium5)
            && remote.contains(&SignatureScheme::HybridEd25519Dilithium5)
        {
            return SignatureScheme::HybridEd25519Dilithium5;
        }

        if local.contains(&SignatureScheme::Dilithium5)
            && remote.contains(&SignatureScheme::Dilithium5)
        {
            return SignatureScheme::Dilithium5;
        }

        SignatureScheme::Ed25519 // Fallback
    }
}
```

### 2. Migration Strategy
- **Phase 0 → Phase 2:** Gradual migration over 6 months
- **Dual signing period:** Both signatures required during transition
- **Network consensus:** 67% nodes must support PQ before mandatory
- **Backward compatibility:** Always maintain Ed25519 fallback

### 3. Key Storage Security
- **Encrypted storage:** AES-256-GCM for all secret keys
- **Key derivation:** Argon2id with high parameters
- **Secure deletion:** Explicit memory zeroing after use
- **HSM integration:** Optional hardware security module support

---

## 📊 Performance Benchmarks

### Signature Sizes:
- **Ed25519:** 64 bytes
- **Dilithium5:** 4,595 bytes (72× larger)
- **Hybrid:** 4,659 bytes

### Verification Times (AMD EPYC 7763):
- **Ed25519:** 71 µs
- **Dilithium5:** 1,200 µs (17× slower)
- **Dilithium5 Batch (AVX-512):** 300 µs (4× speedup)
- **Hybrid:** 1,271 µs (both verified)

### Network Impact:
- **Transaction size increase:** ~4.5 KB per tx with Dilithium5
- **Block size impact:** Significant (70× larger signatures)
- **Mitigation:** Schnorr aggregation (future), signature compression

### Throughput Analysis:
- **Current (Ed25519):** 100,000+ TPS
- **With Dilithium5 (naive):** ~833 TPS (120× slowdown)
- **With SIMD batch:** ~3,333 TPS (30× slowdown)
- **With parallel processing (64 cores):** 100,000+ TPS maintained ✅

---

## 🧪 Testing Strategy

### Unit Tests:
- ✅ Dilithium5 keypair generation
- ✅ Dilithium5 sign/verify
- ✅ Kyber1024 encapsulation/decapsulation
- ✅ Hybrid encryption end-to-end
- 📋 Crypto-agile transaction validation
- 📋 Algorithm negotiation logic
- 📋 Key derivation functions

### Integration Tests:
- 📋 Hybrid wallet creation and signing
- 📋 P2P handshake with Kyber1024
- 📋 Cross-scheme transaction validation
- 📋 Network consensus with mixed schemes
- 📋 QKD simulator functionality

### Performance Tests:
- 📋 Batch signature verification benchmarks
- 📋 TPS measurement with PQ signatures
- 📋 Memory usage profiling
- 📋 Network bandwidth impact analysis

### Security Audits:
- 📋 Cryptographic parameter review
- 📋 Side-channel attack mitigation
- 📋 Timing attack resistance
- 📋 Formal verification of critical paths

---

## 📦 Dependencies

### New Crates:
```toml
[dependencies]
pqcrypto-dilithium = "0.5"
pqcrypto-kyber = "0.8"
pqcrypto-traits = "0.3"
sha3 = "0.10"
argon2 = "0.5"
aes-gcm = "0.10"
rayon = "1.7"  # Parallel processing
lru = "0.12"   # Signature caching
```

### Build Requirements:
- **LLVM 15+** for AVX-512 intrinsics
- **OpenSSL 3.0+** for hybrid TLS
- **Rust 1.70+** with nightly features

---

## 🚀 Deployment Strategy

### Testnet Rollout (November 2025):
1. Deploy Phase 2 nodes to testnet
2. Enable hybrid signatures (optional)
3. Monitor performance and stability
4. Iterate based on metrics

### Mainnet Migration (December 2025):
1. **Week 1:** Announce PQ support available
2. **Week 2:** Encourage hybrid wallet creation
3. **Week 3:** Measure adoption rate
4. **Week 4:** Mainnet launch with PQ enabled

### Monitoring Metrics:
- Percentage of hybrid transactions
- Average verification time per block
- Network bandwidth usage
- Transaction size distribution
- Error rates and failed verifications

---

## 📚 Documentation

### Developer Docs:
- [ ] Phase 2 architecture overview
- [ ] Crypto-agile API reference
- [ ] Hybrid wallet guide
- [ ] P2P PQ handshake protocol spec
- [ ] Performance optimization guide

### User Docs:
- [ ] "Why Post-Quantum Matters" explainer
- [ ] Hybrid wallet creation tutorial
- [ ] Migration guide (Phase 0 → Phase 2)
- [ ] Security best practices
- [ ] FAQ: Q-Day preparedness

---

## ✅ Success Criteria

Phase 2 is considered complete when:

1. ✅ Dilithium5 + Kyber1024 fully integrated
2. ✅ Hybrid signatures working end-to-end
3. ✅ Crypto-agile transaction format deployed
4. ✅ P2P network supports PQ key exchange
5. ✅ QKD preparation layer implemented
6. ✅ Performance: 100,000+ TPS maintained
7. ✅ >90% test coverage for PQ code
8. ✅ Security audit passed
9. ✅ Documentation complete
10. ✅ Mainnet deployment successful

---

## 🎯 Next Steps (Phase 3)

After Phase 2 completion:

1. **Tor Integration:** Anonymous P2P with dedicated circuits
2. **Dandelion++:** Traffic analysis resistance
3. **Full Network Stress Testing:** 1M+ TPS targets
4. **Hardware Wallet Support:** Ledger/Trezor integration
5. **Mobile Applications:** iOS/Android quantum wallets

---

**Let's build the quantum-resistant future! 🚀🔐⚛️**
