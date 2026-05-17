# Q-NarwhalKnight Advanced Cryptography Implementation Plan

## Overview

This document outlines the complete implementation plan for Phase 2-4 cryptographic improvements.

---

## Phase 2: Lattice Aggregate Signatures + Genus-2 VDF

### Timeline: Weeks 2-5

---

### 2.1 Lattice-Based Aggregate Signatures

**Paper Reference**: IACR 2025/1056

**Duration**: 2-3 weeks

#### Week 2: Core Implementation

**Day 1-2: Research & Design**
```
Tasks:
├── Study IACR 2025/1056 paper in detail
├── Analyze lattice structure requirements
├── Design Rust module architecture
└── Create type definitions and traits

Deliverables:
├── lattice_aggregate.rs skeleton
├── Type definitions for LatticeSignature, AggregateSignature
└── Trait definitions for Aggregatable
```

**Day 3-4: Lattice Arithmetic**
```
Tasks:
├── Implement polynomial ring operations Rq = Zq[X]/(X^n + 1)
├── Implement NTT (Number Theoretic Transform) for fast multiplication
├── Implement rejection sampling for signature generation
└── Unit tests for arithmetic operations

Deliverables:
├── lattice_ring.rs - Ring arithmetic
├── ntt.rs - NTT transforms
└── sampling.rs - Gaussian and rejection sampling
```

**Day 5-7: Signature Aggregation**
```
Tasks:
├── Implement individual signature verification
├── Implement signature aggregation algorithm
├── Implement aggregated verification
└── Handle edge cases (empty set, single sig, etc.)

Deliverables:
├── aggregate.rs - Core aggregation logic
├── verify_aggregate.rs - Verification routines
└── Integration tests
```

#### Week 3: Optimization & Integration

**Day 1-2: Performance Optimization**
```
Tasks:
├── SIMD vectorization for lattice operations
├── Parallel NTT using rayon
├── Memory optimization (in-place operations)
└── Benchmark against individual signatures

Deliverables:
├── Optimized NTT with AVX2/AVX-512
├── Parallel aggregation
└── Benchmark results
```

**Day 3-4: Integration with q-consensus**
```
Tasks:
├── Integrate with block signature aggregation
├── Modify BlockHeader to support aggregate sigs
├── Update transaction verification to use aggregation
└── Add feature flag for backwards compatibility

Deliverables:
├── Updated crates/q-types/src/block.rs
├── Updated crates/q-consensus/src/verify.rs
└── Feature flag: lattice-aggregate
```

**Day 5-7: Testing & Documentation**
```
Tasks:
├── Comprehensive unit tests (>90% coverage)
├── Integration tests with consensus
├── Fuzz testing for edge cases
├── API documentation
└── Update technical review document

Deliverables:
├── tests/lattice_aggregate_tests.rs
├── Fuzz harness
├── Updated ADVANCED_CRYPTOGRAPHY_TECHNICAL_REVIEW.md
```

#### API Design

```rust
// crates/q-crypto-advanced/src/lattice_aggregate.rs

/// Lattice-based signature compatible with aggregation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatticeSignature {
    /// Commitment component z
    z: PolynomialVec,
    /// Hint for verification
    h: HintVec,
}

/// Aggregated signature from multiple signers
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregateSignature {
    /// Aggregated z component
    z_agg: PolynomialVec,
    /// Combined hints
    h_agg: HintVec,
    /// Number of aggregated signatures
    count: u32,
    /// Hash of public keys (for binding)
    pk_hash: [u8; 32],
}

/// Key pair for lattice signatures
pub struct LatticeKeyPair {
    pub public_key: LatticePublicKey,
    secret_key: LatticeSecretKey,
}

impl LatticeKeyPair {
    /// Generate new key pair
    pub fn generate() -> Self;

    /// Sign a message
    pub fn sign(&self, message: &[u8]) -> LatticeSignature;
}

impl LatticeSignature {
    /// Verify single signature
    pub fn verify(&self, pk: &LatticePublicKey, message: &[u8]) -> bool;
}

impl AggregateSignature {
    /// Aggregate multiple signatures on SAME message
    pub fn aggregate_same_message(
        signatures: &[(LatticePublicKey, LatticeSignature)],
        message: &[u8],
    ) -> Result<Self, CryptoError>;

    /// Aggregate signatures on DIFFERENT messages
    pub fn aggregate_different_messages(
        signatures: &[(LatticePublicKey, LatticeSignature, Vec<u8>)],
    ) -> Result<Self, CryptoError>;

    /// Verify aggregated signature (same message)
    pub fn verify_same_message(
        &self,
        public_keys: &[LatticePublicKey],
        message: &[u8],
    ) -> bool;

    /// Verify aggregated signature (different messages)
    pub fn verify_different_messages(
        &self,
        public_keys: &[LatticePublicKey],
        messages: &[&[u8]],
    ) -> bool;

    /// Incrementally add a signature
    pub fn add(&mut self, pk: &LatticePublicKey, sig: &LatticeSignature, msg: &[u8])
        -> Result<(), CryptoError>;
}
```

---

### 2.2 Genus-2 Curve VDF

**Paper Reference**: IACR 2025/1050

**Duration**: 3-4 weeks

#### Week 4: Curve Arithmetic

**Day 1-2: Hyperelliptic Curve Definition**
```
Tasks:
├── Define genus-2 curve structure y² = x⁵ + ax³ + bx² + cx + d
├── Implement affine point representation
├── Implement projective coordinates for efficiency
└── Define curve parameters for 128-bit security

Deliverables:
├── hyperelliptic.rs - Curve definition
├── point.rs - Point representations
└── params.rs - Secure parameter sets
```

**Day 3-4: Jacobian Variety Operations**
```
Tasks:
├── Implement divisor representation
├── Implement divisor addition (Cantor's algorithm)
├── Implement scalar multiplication
└── Implement Frobenius endomorphism

Deliverables:
├── jacobian.rs - Jacobian variety
├── divisor.rs - Divisor arithmetic
└── endomorphism.rs - Frobenius computation
```

**Day 5-7: Field Arithmetic Optimization**
```
Tasks:
├── Implement Montgomery multiplication
├── Implement fast field inversion
├── Implement efficient square root
└── Benchmark against GMP/OpenSSL

Deliverables:
├── field.rs - Optimized field ops
├── montgomery.rs - Montgomery form
└── Benchmarks
```

#### Week 5: VDF Protocol

**Day 1-2: VDF Evaluation**
```
Tasks:
├── Implement repeated Frobenius for VDF
├── Add progress callbacks for long evaluations
├── Implement checkpointing for interruption recovery
└── Unit tests for various time parameters

Deliverables:
├── vdf_eval.rs - Core VDF evaluation
├── checkpoint.rs - State checkpointing
└── Tests
```

**Day 3-4: VDF Proofs**
```
Tasks:
├── Implement Wesolowski-style proofs adapted for genus-2
├── Implement proof verification
├── Optimize proof generation
└── Security analysis

Deliverables:
├── vdf_proof.rs - Proof generation
├── vdf_verify.rs - Proof verification
└── Security documentation
```

**Day 5-7: Integration**
```
Tasks:
├── Integrate with leader election
├── Replace RSA-based VDF in q-vdf crate
├── Add migration path from old VDF
├── Comprehensive testing

Deliverables:
├── Updated crates/q-vdf/
├── Migration guide
├── Integration tests
└── Benchmark comparison with old VDF
```

#### API Design

```rust
// crates/q-crypto-advanced/src/genus2_vdf.rs

/// Genus-2 hyperelliptic curve parameters
pub struct Genus2Curve {
    /// Curve coefficients y² = x⁵ + a₃x³ + a₂x² + a₁x + a₀
    a3: FieldElement,
    a2: FieldElement,
    a1: FieldElement,
    a0: FieldElement,
    /// Field modulus
    modulus: BigUint,
}

/// Point on the Jacobian variety
#[derive(Clone, Debug)]
pub struct JacobianPoint {
    /// Mumford representation (u, v) where u = x² + u₁x + u₀
    u: [FieldElement; 3],
    v: [FieldElement; 2],
}

/// VDF output with proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VDFOutput {
    /// Result point on Jacobian
    pub result: JacobianPoint,
    /// Wesolowski proof
    pub proof: VDFProof,
    /// Number of iterations
    pub iterations: u64,
    /// Evaluation time in milliseconds
    pub eval_time_ms: u64,
}

/// VDF proof (compact)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VDFProof {
    /// Quotient point
    quotient: JacobianPoint,
    /// Challenge hash
    challenge: [u8; 32],
}

/// Quantum-safe VDF evaluator
pub struct Genus2VDF {
    curve: Genus2Curve,
    generator: JacobianPoint,
}

impl Genus2VDF {
    /// Create VDF with default secure parameters
    pub fn new() -> Self;

    /// Create VDF with custom curve
    pub fn with_curve(curve: Genus2Curve) -> Self;

    /// Evaluate VDF (slow, sequential)
    /// Returns after `iterations` Frobenius applications
    pub fn evaluate(
        &self,
        input: &[u8],
        iterations: u64
    ) -> VDFOutput;

    /// Evaluate with progress callback
    pub fn evaluate_with_progress<F>(
        &self,
        input: &[u8],
        iterations: u64,
        callback: F,
    ) -> VDFOutput
    where
        F: Fn(u64, u64); // (current, total)

    /// Verify VDF output (fast, O(log n))
    pub fn verify(&self, input: &[u8], output: &VDFOutput) -> bool;

    /// Calculate time parameter for target delay
    pub fn iterations_for_delay(&self, target_ms: u64) -> u64;
}

/// Integration with consensus
pub trait VDFLeaderElection {
    /// Compute VDF for a specific slot
    fn compute_for_slot(&self, slot: u64, prev_hash: &[u8]) -> VDFOutput;

    /// Verify a leader's VDF proof
    fn verify_leader(&self, slot: u64, prev_hash: &[u8], proof: &VDFOutput) -> bool;
}

impl VDFLeaderElection for Genus2VDF {
    fn compute_for_slot(&self, slot: u64, prev_hash: &[u8]) -> VDFOutput {
        let input = sha3_256(&[prev_hash, &slot.to_le_bytes()].concat());
        let iterations = self.iterations_for_slot();
        self.evaluate(&input, iterations)
    }

    fn verify_leader(&self, slot: u64, prev_hash: &[u8], proof: &VDFOutput) -> bool {
        let input = sha3_256(&[prev_hash, &slot.to_le_bytes()].concat());
        self.verify(&input, proof)
    }
}
```

---

## Phase 3: SQIsign + Improved Bulletproofs

### Timeline: Weeks 6-10

---

### 3.1 SQIsign Compact Signatures

**Paper Reference**: IACR 2025/847

**Duration**: 4-5 weeks

#### Week 6-7: Isogeny Foundations

**Day 1-3: Supersingular Curves**
```
Tasks:
├── Implement supersingular elliptic curves over Fp²
├── Implement curve arithmetic (add, double, scalar mult)
├── Implement j-invariant computation
└── Implement curve isomorphism detection

Deliverables:
├── supersingular.rs - SS curve implementation
├── fp2.rs - Fp² arithmetic
└── j_invariant.rs - Invariant computation
```

**Day 4-5: Isogeny Computation**
```
Tasks:
├── Implement kernel computation
├── Implement Vélu's formulas for isogeny evaluation
├── Implement isogeny composition
└── Optimize for SQIsign-specific degree structures

Deliverables:
├── isogeny.rs - Core isogeny operations
├── velu.rs - Vélu's formulas
└── kernel.rs - Kernel algorithms
```

**Day 6-7: SIDH Structure**
```
Tasks:
├── Implement torsion point generation
├── Implement public key computation
├── Implement shared secret derivation
└── Unit tests for key exchange

Deliverables:
├── sidh.rs - SIDH protocol
├── torsion.rs - Torsion points
└── Tests
```

#### Week 8-9: SQIsign Protocol

**Day 1-3: Key Generation**
```
Tasks:
├── Implement secret isogeny path generation
├── Implement public key (end curve) computation
├── Implement key validation
└── Performance optimization

Deliverables:
├── keygen.rs - Key generation
├── validation.rs - Key validation
└── Benchmarks
```

**Day 4-5: Signing**
```
Tasks:
├── Implement commitment scheme
├── Implement challenge generation
├── Implement response computation
└── Implement signature compression

Deliverables:
├── sign.rs - Signing algorithm
├── compress.rs - Signature compression
└── Tests
```

**Day 6-7: Verification**
```
Tasks:
├── Implement signature decompression
├── Implement verification equation checking
├── Optimize verification path
└── Security analysis

Deliverables:
├── verify.rs - Verification algorithm
├── decompress.rs - Decompression
└── Security docs
```

#### Week 10: Integration & Hybrid Mode

**Day 1-3: Integration**
```
Tasks:
├── Integrate with transaction signing
├── Add hybrid mode (SQIsign + Dilithium)
├── Update wallet to support SQIsign
└── Migration path for existing keys

Deliverables:
├── Updated q-types for hybrid signatures
├── Wallet integration
├── Migration tools
```

**Day 4-7: Testing & Optimization**
```
Tasks:
├── Comprehensive test suite
├── Fuzz testing
├── Performance optimization
├── Documentation

Deliverables:
├── Complete test coverage
├── Optimized implementation
├── API documentation
```

#### API Design

```rust
// crates/q-crypto-advanced/src/sqisign/mod.rs

/// SQIsign public key (64 bytes - smallest PQ signature!)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SQIsignPublicKey([u8; 64]);

/// SQIsign secret key
pub struct SQIsignSecretKey {
    /// Secret isogeny path
    path: IsogenyPath,
    /// Cached intermediate values
    cache: SecretCache,
}

/// SQIsign signature (204 bytes)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SQIsignSignature([u8; 204]);

/// SQIsign key pair
pub struct SQIsignKeyPair {
    pub public_key: SQIsignPublicKey,
    secret_key: SQIsignSecretKey,
}

impl SQIsignKeyPair {
    /// Generate new key pair
    /// Note: This is slow (~850ms) due to isogeny computation
    pub fn generate() -> Self;

    /// Sign a message
    /// Note: Slower than Dilithium (~450ms) but produces tiny signatures
    pub fn sign(&self, message: &[u8]) -> SQIsignSignature;
}

impl SQIsignSignature {
    /// Verify signature
    /// Note: Slower than Dilithium (~75ms) but smallest PQ signature
    pub fn verify(&self, public_key: &SQIsignPublicKey, message: &[u8]) -> bool;

    /// Batch verify multiple signatures (more efficient)
    pub fn batch_verify(
        signatures: &[(SQIsignSignature, SQIsignPublicKey, Vec<u8>)],
    ) -> bool;
}

/// Hybrid signature for transition period
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HybridSignature {
    /// SQIsign signature (always present)
    pub sqisign: SQIsignSignature,
    /// Dilithium signature (optional, for backwards compat)
    pub dilithium: Option<DilithiumSignature>,
}

impl HybridSignature {
    /// Create hybrid signature
    pub fn sign(
        sqisign_key: &SQIsignKeyPair,
        dilithium_key: Option<&DilithiumKeyPair>,
        message: &[u8],
    ) -> Self;

    /// Verify hybrid (requires SQIsign, optionally checks Dilithium)
    pub fn verify(
        &self,
        sqisign_pk: &SQIsignPublicKey,
        dilithium_pk: Option<&DilithiumPublicKey>,
        message: &[u8],
    ) -> bool;
}
```

---

### 3.2 Improved Bulletproofs

**Paper Reference**: IACR 2024/1756

**Duration**: 2-3 weeks (Week 10)

```rust
// crates/q-crypto-advanced/src/bulletproofs_v2.rs

/// Aggregated range proof for multiple values
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregatedRangeProof {
    /// Proof data
    data: Vec<u8>,
    /// Number of values proven
    count: usize,
    /// Bit length of each value
    bit_length: usize,
}

/// Improved Bulletproof prover
pub struct BulletproofProver {
    /// Generators
    g_vec: Vec<RistrettoPoint>,
    h_vec: Vec<RistrettoPoint>,
}

impl BulletproofProver {
    /// Create prover with generators for n values of m bits each
    pub fn new(n: usize, m: usize) -> Self;

    /// Prove that all values are in range [0, 2^bits)
    pub fn prove_range(
        &self,
        values: &[u64],
        blindings: &[Scalar],
        bit_length: usize,
    ) -> Result<AggregatedRangeProof, CryptoError>;
}

/// Batch verifier for multiple proofs
pub struct BatchVerifier {
    proofs: Vec<AggregatedRangeProof>,
    commitments: Vec<Vec<PedersenCommitment>>,
}

impl BatchVerifier {
    /// Add proof to batch
    pub fn add(&mut self, proof: AggregatedRangeProof, commitments: Vec<PedersenCommitment>);

    /// Verify all proofs in batch (10x faster than individual)
    pub fn verify_batch(self) -> bool;
}
```

---

## Phase 4: Integration & Testing

### Timeline: Weeks 11-12

#### Week 11: Full Integration

```
Tasks:
├── Integrate all crypto modules with consensus
├── Update block structure for new signature types
├── Add configuration for crypto algorithm selection
├── Database migration for new key types
└── Network protocol updates for new message types
```

#### Week 12: Testing & Audit Prep

```
Tasks:
├── End-to-end integration tests
├── Performance benchmarks on production hardware
├── Security review and documentation
├── Prepare materials for external audit
└── Final documentation updates
```

---

## Dependency Updates

```toml
# crates/q-crypto-advanced/Cargo.toml additions

[dependencies]
# Phase 2
num-bigint = "0.4"      # Big integer for genus-2 curves
num-traits = "0.2"
rayon = "1.8"           # Parallel processing

# Phase 3
# SQIsign will likely need custom implementation or bindings
# Bulletproofs v2
curve25519-dalek = "4.1"
merlin = "3.0"          # Transcript for Fiat-Shamir
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SQIsign performance too slow | Medium | Medium | Hybrid mode, batch verification |
| Genus-2 curve implementation bugs | Low | High | Extensive testing, formal verification |
| Lattice aggregate incompatibility | Low | Medium | Fallback to individual sigs |
| Timeline slippage | Medium | Low | Prioritize critical features |

---

## Success Criteria

### Phase 2
- [ ] Lattice aggregate reduces block size by >80%
- [ ] Genus-2 VDF achieves >128-bit quantum security
- [ ] All unit tests pass with >90% coverage
- [ ] Integration tests with consensus pass

### Phase 3
- [ ] SQIsign produces <250 byte signatures
- [ ] Bulletproofs v2 achieves 10x batch speedup
- [ ] Hybrid mode maintains backwards compatibility
- [ ] Security review complete

### Phase 4
- [ ] Full system integration complete
- [ ] Performance benchmarks meet targets
- [ ] Documentation complete
- [ ] Ready for external audit

---

*Document Version: 1.0.0 | Last Updated: 2025-11-28*
