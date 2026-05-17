# Phase 3: Consensus Security Development Plan

**Duration**: Weeks 9-12
**Version Target**: v1.2.0-beta
**Status**: Planning

---

## Executive Summary

Phase 3 focuses on securing the balance update mechanism by:
1. **Removing gossipsub balance updates** - All balance changes go through DAG-Knight consensus
2. **Mandatory cryptographic signatures** - Every balance-affecting operation must be signed
3. **Byzantine Fault Tolerance enforcement** - 2/3+ validators must agree on state changes
4. **Security hardening** - Audits, bug bounties, formal verification preparation

---

## Current Architecture Analysis

### What Works Well (Keep)
- **Block-based balance distribution**: Coinbase transactions in blocks already handle mining rewards
- **BalanceConsensusEngine**: Deterministically processes blocks on all nodes
- **DAG-Knight finality**: Committed blocks are immutable after δ rounds
- **Height monotonicity**: Prevents sync-down data loss

### What Needs Fixing (Remove/Replace)
- **P2PBalanceUpdate gossipsub messages**: Currently optional, can be spoofed
- **Rate limiting band-aids**: 50k/min limit is a workaround, not a solution
- **Signature optional for some operations**: Some legacy paths skip verification

---

## Development Phases

### Week 9: Remove Gossipsub Balance Updates

#### Step 1: Deprecate P2PBalanceUpdate Gossipsub Topic
**Files to modify:**
- `crates/q-api-server/src/main.rs` (lines 3600-3750)
- `crates/q-network/src/unified_network_manager.rs`

**Tasks:**
```
[ ] Add deprecation warning for /balance-updates topic
[ ] Create feature flag: ENABLE_LEGACY_BALANCE_GOSSIP (default: false)
[ ] Remove subscription to /qnk/{network}/balance-updates topic
[ ] Remove balance update broadcasting code
[ ] Update SSE to use block-based updates only
```

**Code changes:**
```rust
// Before: Subscribing to balance-updates gossipsub
swarm.behaviour_mut().subscribe(&IdentTopic::new(format!("/qnk/{}/balance-updates", network_id)))?;

// After: Remove entirely or gate behind feature flag
#[cfg(feature = "legacy_balance_gossip")]
swarm.behaviour_mut().subscribe(&IdentTopic::new(format!("/qnk/{}/balance-updates", network_id)))?;
```

#### Step 2: Enhance Block-Based Balance Notifications
**Files to modify:**
- `crates/q-api-server/src/streaming.rs`
- `crates/q-storage/src/balance_consensus.rs`

**Tasks:**
```
[ ] Add SSE events when BalanceConsensusEngine processes coinbase tx
[ ] Include block hash and height in balance update notifications
[ ] Add "pending" vs "confirmed" balance tracking
[ ] Create balance update receipts with merkle proofs
```

#### Step 3: Remove Rate Limiting Code (No Longer Needed)
**Files to modify:**
- `crates/q-api-server/src/main.rs` (lines 107-200)

**Tasks:**
```
[ ] Remove BALANCE_UPDATE_RATE_LIMITS static
[ ] Remove BALANCE_UPDATE_DEDUP_CACHE static
[ ] Remove check_balance_update_rate_limit() function
[ ] Remove is_duplicate_balance_update() function
[ ] Clean up related helper functions
```

---

### Week 10: Mandatory Signatures & Transaction Security

#### Step 4: Enforce Transaction Signatures
**Files to modify:**
- `crates/q-types/src/lib.rs` (Transaction struct)
- `crates/q-api-server/src/handlers.rs`
- `crates/q-storage/src/transaction.rs`

**Tasks:**
```
[ ] Make Transaction.signature field mandatory (not Option)
[ ] Add Transaction.verify_signature() -> Result<(), SignatureError>
[ ] Reject unsigned transactions at API boundary
[ ] Add signature verification in mempool insertion
[ ] Update all transaction creation paths
```

**Signature verification flow:**
```rust
impl Transaction {
    pub fn verify_signature(&self) -> Result<(), SignatureError> {
        // 1. Reconstruct signing payload
        let payload = self.signing_payload();

        // 2. Verify Ed25519 signature
        let public_key = VerifyingKey::from_bytes(&self.sender)?;
        let signature = Signature::from_bytes(&self.signature)?;
        public_key.verify(&payload, &signature)?;

        // 3. Optional: Verify Dilithium5 if present (Phase Q1+)
        if let Some(pq_sig) = &self.pq_signature {
            self.verify_dilithium5(pq_sig)?;
        }

        Ok(())
    }
}
```

#### Step 5: Enforce Block Producer Signatures
**Files to modify:**
- `crates/q-types/src/block.rs`
- `crates/q-api-server/src/block_producer.rs`
- `crates/q-dag-knight/src/lib.rs`

**Tasks:**
```
[ ] Add QBlock.producer_signature field
[ ] Sign blocks in BlockProducer.produce_block()
[ ] Verify producer signature before adding to DAG
[ ] Reject unsigned blocks from P2P
[ ] Add slashing conditions for double-signing
```

**Block signing:**
```rust
impl QBlock {
    pub fn sign(&mut self, keypair: &ValidatorKeypair) -> Result<()> {
        let payload = self.header.signing_payload();
        self.header.producer_signature = Some(keypair.sign(&payload)?);
        Ok(())
    }

    pub fn verify_producer_signature(&self) -> Result<()> {
        let sig = self.header.producer_signature
            .as_ref()
            .ok_or(ConsensusError::MissingBlockSignature)?;

        let producer_key = self.header.producer_public_key()?;
        producer_key.verify(&self.header.signing_payload(), sig)?;
        Ok(())
    }
}
```

#### Step 6: Coinbase Transaction Security
**Files to modify:**
- `crates/q-api-server/src/block_producer.rs` (create_coinbase_transactions)
- `crates/q-storage/src/balance_consensus.rs`

**Tasks:**
```
[ ] Sign coinbase transactions with block producer key
[ ] Verify coinbase signatures in BalanceConsensusEngine
[ ] Add coinbase amount validation (matches emission schedule)
[ ] Add merkle root of all coinbase outputs to block header
```

---

### Week 11: Byzantine Fault Tolerance Enforcement

#### Step 7: Validator Registry
**New files to create:**
- `crates/q-consensus/src/validator_registry.rs`
- `crates/q-consensus/src/stake_manager.rs`

**Tasks:**
```
[ ] Create ValidatorRegistry struct
[ ] Track active validators and their stakes
[ ] Implement validator set changes (add/remove)
[ ] Add minimum stake requirement (testnet: 0, mainnet: TBD)
[ ] Create validator rotation mechanism
```

**ValidatorRegistry design:**
```rust
pub struct ValidatorRegistry {
    /// Active validators with their stakes
    validators: HashMap<PeerId, ValidatorInfo>,

    /// Total staked amount for BFT threshold calculation
    total_stake: u64,

    /// Validator set epoch (increments on changes)
    epoch: u64,
}

pub struct ValidatorInfo {
    pub peer_id: PeerId,
    pub public_key: [u8; 32],
    pub pq_public_key: Option<Vec<u8>>,  // Dilithium5
    pub stake: u64,
    pub registered_at: u64,
}

impl ValidatorRegistry {
    /// Returns true if 2/3+ stake has voted for this block
    pub fn has_supermajority(&self, votes: &[Vote]) -> bool {
        let voted_stake: u64 = votes.iter()
            .filter_map(|v| self.validators.get(&v.validator).map(|vi| vi.stake))
            .sum();

        voted_stake * 3 > self.total_stake * 2
    }
}
```

#### Step 8: BFT Commit Protocol
**Files to modify:**
- `crates/q-dag-knight/src/commit_logic.rs`
- `crates/q-dag-knight/src/voting_coordinator.rs`

**Tasks:**
```
[ ] Add vote collection for block commits
[ ] Require 2/3+ validator signatures for finality
[ ] Implement view change protocol for liveness
[ ] Add commit certificates (aggregated signatures)
[ ] Store commit proofs in block headers
```

**Commit certificate:**
```rust
pub struct CommitCertificate {
    /// Block being committed
    pub block_hash: [u8; 32],

    /// Height of the block
    pub height: u64,

    /// Aggregated validator signatures (BLS or multi-sig)
    pub signatures: Vec<ValidatorSignature>,

    /// Bitmap of which validators signed
    pub signer_bitmap: BitVec,

    /// Epoch of the validator set
    pub validator_epoch: u64,
}

impl CommitCertificate {
    pub fn verify(&self, registry: &ValidatorRegistry) -> Result<()> {
        // 1. Check we have 2/3+ stake
        ensure!(registry.has_supermajority(&self.signatures), "Insufficient votes");

        // 2. Verify each signature
        for (idx, sig) in self.signatures.iter().enumerate() {
            let validator = self.get_signer(idx)?;
            validator.verify_signature(&self.block_hash, sig)?;
        }

        Ok(())
    }
}
```

#### Step 9: Equivocation Detection & Slashing
**New files to create:**
- `crates/q-consensus/src/equivocation.rs`
- `crates/q-consensus/src/slashing.rs`

**Tasks:**
```
[ ] Detect double-signing (same height, different blocks)
[ ] Detect conflicting votes
[ ] Implement slashing evidence submission
[ ] Create slashing transaction type
[ ] Add slashed validator list
```

---

### Week 12: Security Audits & Bug Bounty

#### Step 10: Security Documentation
**New files to create:**
- `docs/SECURITY_MODEL.md`
- `docs/THREAT_MODEL.md`
- `docs/CRYPTOGRAPHIC_SPEC.md`

**Tasks:**
```
[ ] Document all security assumptions
[ ] List attack vectors and mitigations
[ ] Specify cryptographic primitives used
[ ] Define trust boundaries
[ ] Create incident response plan
```

#### Step 11: Automated Security Testing
**New files to create:**
- `tests/security/fuzzing.rs`
- `tests/security/invariants.rs`
- `tests/security/attack_scenarios.rs`

**Tasks:**
```
[ ] Add fuzz testing for transaction parsing
[ ] Add fuzz testing for block deserialization
[ ] Test Byzantine scenarios (33% malicious)
[ ] Test network partition healing
[ ] Test double-spend attempts
[ ] Test replay attacks
```

#### Step 12: Bug Bounty Program
**Tasks:**
```
[ ] Define scope (smart contracts, consensus, P2P)
[ ] Set reward tiers:
    - Critical (consensus break): $50,000-$100,000
    - High (fund loss risk): $10,000-$50,000
    - Medium (DoS, spam): $1,000-$10,000
    - Low (info leak): $100-$1,000
[ ] Create submission process
[ ] Set up secure disclosure channel
[ ] Engage external auditors
```

---

## Implementation Order

```
Week 9:
├── Day 1-2: Deprecate gossipsub balance updates (Step 1)
├── Day 3-4: Enhance block-based notifications (Step 2)
└── Day 5: Remove rate limiting code (Step 3)

Week 10:
├── Day 1-2: Enforce transaction signatures (Step 4)
├── Day 3-4: Enforce block producer signatures (Step 5)
└── Day 5: Coinbase transaction security (Step 6)

Week 11:
├── Day 1-2: Validator registry (Step 7)
├── Day 3-4: BFT commit protocol (Step 8)
└── Day 5: Equivocation detection (Step 9)

Week 12:
├── Day 1-2: Security documentation (Step 10)
├── Day 3-4: Automated security testing (Step 11)
└── Day 5: Bug bounty setup (Step 12)
```

---

## Success Criteria

### Functionality
- [ ] No balance updates via gossipsub (100% through consensus)
- [ ] All transactions signed and verified
- [ ] All blocks signed by producer
- [ ] Commit certificates require 2/3+ validators

### Security
- [ ] Zero unsigned transactions accepted
- [ ] Zero unsigned blocks in DAG
- [ ] Double-signing detected and punished
- [ ] No replay attacks possible

### Performance
- [ ] Block finality < 3 seconds
- [ ] Transaction throughput > 1000 TPS
- [ ] P2P message overhead reduced by 50%

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking change for existing nodes | Feature flags, gradual rollout |
| Validator liveness issues | View change protocol, backup validators |
| Signature verification overhead | Batch verification, caching |
| Key compromise | Multi-sig support, key rotation |

---

## Dependencies

- **Phase 2 completion**: Stable block sync required
- **Cryptography libraries**: ed25519-dalek, pqcrypto-dilithium
- **BFT libraries**: Consider tendermint-rs for proven BFT

---

## Open Questions

1. **Validator incentives**: How are validators rewarded for signing?
2. **Stake source**: Where does stake come from on testnet?
3. **Key management**: How do validators securely store keys?
4. **Upgrade path**: How do existing balances migrate?

---

*Document created: 2025-12-10*
*Last updated: 2025-12-10*
