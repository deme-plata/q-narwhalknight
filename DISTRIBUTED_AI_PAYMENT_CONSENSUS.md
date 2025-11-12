# Distributed AI Payment Consensus System

**Date**: October 29, 2025
**Status**: Design Phase
**Goal**: Decentralized payment verification where all nodes reach consensus before processing AI requests

---

## Problem Statement

For true decentralization, we cannot have a single node controlling payment verification. Instead:
- **All validator nodes** must agree on payment validity
- **Byzantine fault tolerance** must prevent malicious nodes from approving invalid payments
- **Payment state** must be synchronized across the network
- **Double-spending** must be prevented through distributed consensus

---

## Architecture Overview

```
User submits AI request with payment
         ↓
    Node receives request
         ↓
    Broadcasts payment proposal to validators
         ↓
    ┌─────────────────────────────────────┐
    │  Distributed Payment Consensus       │
    │  - Each validator verifies payment   │
    │  - Checks wallet balance             │
    │  - Validates signature               │
    │  - Votes on payment validity         │
    └─────────────────────────────────────┘
         ↓
    ≥67% validators approve? (BFT threshold)
         ↓
    YES: Payment locked, AI inference starts
         ↓
    AI generation completes
         ↓
    Payment settlement broadcasted
         ↓
    Validators update balances
         ↓
    User receives response + payment proof
```

---

## Consensus Protocol: Payment-BFT

### Phase 1: Payment Proposal
```rust
struct PaymentProposal {
    request_id: String,
    wallet_address: String,
    estimated_tokens: u32,
    estimated_cost_qnk: u64,
    payment_token: PaymentToken, // QNK or QUGUSD
    signature: Vec<u8>,
    timestamp: u64,
    proposer_node_id: String,
}
```

**Process**:
1. User sends AI request with wallet signature
2. Receiving node creates `PaymentProposal`
3. Node broadcasts proposal to all validators via gossip
4. Validators have 2 seconds to respond

### Phase 2: Validator Verification
Each validator independently verifies:
```rust
async fn verify_payment_proposal(proposal: &PaymentProposal) -> Result<bool> {
    // 1. Verify wallet signature
    if !verify_signature(&proposal.wallet_address, &proposal.signature) {
        return Ok(false);
    }

    // 2. Check wallet balance (from local state)
    let balance = get_wallet_balance(&proposal.wallet_address).await?;
    if balance < proposal.estimated_cost_qnk {
        return Ok(false);
    }

    // 3. Check for double-spend (pending payments)
    if has_pending_payment(&proposal.wallet_address).await? {
        return Ok(false);
    }

    // 4. Verify timestamp (within 30 seconds)
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    if now - proposal.timestamp > 30 {
        return Ok(false);
    }

    // 5. Verify estimated cost calculation
    let oracle_price = get_oracle_price().await?;
    let expected_cost = calculate_cost(proposal.estimated_tokens, oracle_price);
    if (expected_cost as i64 - proposal.estimated_cost_qnk as i64).abs() > 100 {
        return Ok(false); // Price mismatch
    }

    Ok(true)
}
```

### Phase 3: Voting
```rust
struct PaymentVote {
    request_id: String,
    validator_node_id: String,
    vote: bool, // true = approve, false = reject
    reason: Option<String>, // rejection reason
    signature: Vec<u8>, // validator signature
}
```

**Voting Rules**:
- Each validator broadcasts their vote
- Votes are collected for 2 seconds
- **BFT Threshold**: ≥67% of validators must approve (2f+1 out of 3f+1)
- If threshold met: Payment approved
- If not met: Request rejected

### Phase 4: Payment Locking
```rust
struct PaymentLock {
    request_id: String,
    wallet_address: String,
    locked_amount_qnk: u64,
    locked_at: u64,
    validator_signatures: Vec<ValidatorSignature>, // Proof of consensus
}
```

**Process**:
1. Once consensus reached, payment is "locked"
2. Balance deducted from wallet (atomic operation)
3. Lock recorded with validator signatures as proof
4. AI generation begins
5. Lock prevents double-spending

### Phase 5: Settlement
```rust
struct PaymentSettlement {
    request_id: String,
    actual_tokens_generated: u32,
    actual_cost_qnk: u64,
    refund_amount_qnk: u64, // locked - actual
    generation_node_id: String,
    validator_signatures: Vec<ValidatorSignature>,
}
```

**Process**:
1. AI generation completes
2. Actual cost calculated
3. Settlement broadcasted to validators
4. Validators verify and sign
5. Refund issued if applicable
6. Transaction finalized on-chain

---

## Byzantine Fault Tolerance

### Assumptions:
- Network has `n = 3f + 1` validators
- At most `f` validators are Byzantine (malicious or faulty)
- Requires `2f + 1` honest validators for consensus

### Attack Scenarios & Defenses:

#### 1. **Malicious Node Approves Invalid Payment**
- **Defense**: Requires 67% approval, so single malicious node cannot approve alone
- **Example**: 10 validators, ≥7 must approve. 3 malicious nodes cannot force approval.

#### 2. **Validator Collusion (Sybil Attack)**
- **Defense**: Validators are staked (economic security)
- **Penalty**: Slashing for approving invalid payments
- **Detection**: On-chain balance verification post-settlement

#### 3. **Double-Spend Attempt**
- **Defense**: Payment locks prevent concurrent requests
- **Example**: User tries two AI requests simultaneously
  1. First request gets locked
  2. Second request fails verification (pending payment detected)

#### 4. **Price Manipulation**
- **Defense**: Oracle consensus (multiple price sources)
- **Example**: Malicious validator reports fake price
  1. Other validators use oracle price
  2. Price mismatch detected in verification
  3. Request rejected

#### 5. **Validator Censorship**
- **Defense**: Failover to other nodes
- **Example**: User's request ignored by validator
  1. User broadcasts to multiple nodes
  2. Any honest node can propose payment
  3. Network processes request

---

## Implementation Components

### 1. Payment Consensus Module (`q-payment-consensus`)
```rust
pub struct PaymentConsensusEngine {
    validator_set: Arc<RwLock<ValidatorSet>>,
    pending_proposals: Arc<RwLock<HashMap<String, PaymentProposal>>>,
    votes: Arc<RwLock<HashMap<String, Vec<PaymentVote>>>>,
    locked_payments: Arc<RwLock<HashMap<String, PaymentLock>>>,
    network: Arc<P2PNetwork>,
}

impl PaymentConsensusEngine {
    pub async fn propose_payment(&self, proposal: PaymentProposal) -> Result<()> {
        // Broadcast to validators
        self.network.gossip("payment-proposal", &proposal).await?;
        Ok(())
    }

    pub async fn vote_on_payment(&self, request_id: &str) -> Result<PaymentVote> {
        let proposal = self.get_proposal(request_id).await?;
        let valid = verify_payment_proposal(&proposal).await?;

        let vote = PaymentVote {
            request_id: request_id.to_string(),
            validator_node_id: self.node_id.clone(),
            vote: valid,
            reason: if !valid { Some("Invalid balance".to_string()) } else { None },
            signature: self.sign_vote(&request_id, valid).await?,
        };

        self.network.gossip("payment-vote", &vote).await?;
        Ok(vote)
    }

    pub async fn check_consensus(&self, request_id: &str) -> Result<ConsensusResult> {
        let votes = self.votes.read().await.get(request_id).cloned().unwrap_or_default();
        let total_validators = self.validator_set.read().await.count();
        let approvals = votes.iter().filter(|v| v.vote).count();

        // BFT threshold: ≥67%
        let threshold = (total_validators * 2) / 3 + 1;

        if approvals >= threshold {
            Ok(ConsensusResult::Approved)
        } else if votes.len() - approvals > total_validators - threshold {
            Ok(ConsensusResult::Rejected)
        } else {
            Ok(ConsensusResult::Pending)
        }
    }

    pub async fn lock_payment(&self, request_id: &str) -> Result<PaymentLock> {
        let proposal = self.get_proposal(request_id).await?;
        let votes = self.votes.read().await.get(request_id).cloned().unwrap();

        // Collect validator signatures from approving votes
        let signatures: Vec<ValidatorSignature> = votes
            .iter()
            .filter(|v| v.vote)
            .map(|v| ValidatorSignature {
                node_id: v.validator_node_id.clone(),
                signature: v.signature.clone(),
            })
            .collect();

        let lock = PaymentLock {
            request_id: request_id.to_string(),
            wallet_address: proposal.wallet_address.clone(),
            locked_amount_qnk: proposal.estimated_cost_qnk,
            locked_at: current_timestamp(),
            validator_signatures: signatures,
        };

        // Deduct from wallet balance (atomic)
        self.deduct_balance(&proposal.wallet_address, lock.locked_amount_qnk).await?;

        // Store lock
        self.locked_payments.write().await.insert(request_id.to_string(), lock.clone());

        Ok(lock)
    }
}
```

### 2. Payment State Synchronization
```rust
pub struct PaymentStateSync {
    state: Arc<RwLock<PaymentState>>,
    network: Arc<P2PNetwork>,
}

impl PaymentStateSync {
    pub async fn sync_from_peers(&self) -> Result<()> {
        // Request latest payment state from peers
        let peers = self.network.get_peers().await?;
        let mut states = Vec::new();

        for peer in peers {
            let state = self.network.request_payment_state(&peer).await?;
            states.push(state);
        }

        // Consensus on state (majority)
        let canonical_state = self.resolve_state_conflicts(states)?;
        *self.state.write().await = canonical_state;

        Ok(())
    }

    fn resolve_state_conflicts(&self, states: Vec<PaymentState>) -> Result<PaymentState> {
        // Use merkle root voting
        let mut state_votes: HashMap<String, usize> = HashMap::new();

        for state in &states {
            let merkle_root = state.compute_merkle_root();
            *state_votes.entry(merkle_root).or_insert(0) += 1;
        }

        // Pick state with most votes
        let majority_root = state_votes
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(root, _)| root)
            .ok_or(anyhow!("No majority state"))?;

        states
            .into_iter()
            .find(|s| &s.compute_merkle_root() == majority_root)
            .ok_or(anyhow!("State not found"))
    }
}
```

### 3. On-Chain Finalization
```rust
pub struct PaymentFinalization {
    blockchain: Arc<Blockchain>,
}

impl PaymentFinalization {
    pub async fn finalize_payment(&self, settlement: PaymentSettlement) -> Result<TxHash> {
        // Create on-chain transaction
        let tx = Transaction {
            from: settlement.wallet_address.clone(),
            to: AI_TREASURY_ADDRESS.to_string(),
            amount: settlement.actual_cost_qnk,
            data: bincode::serialize(&settlement)?,
            nonce: self.get_nonce(&settlement.wallet_address).await?,
        };

        // Submit to blockchain
        let tx_hash = self.blockchain.submit_transaction(tx).await?;

        // Wait for confirmation
        self.blockchain.wait_for_confirmation(&tx_hash, 3).await?;

        Ok(tx_hash)
    }
}
```

---

## Network Protocol

### Gossip Topics:
- `payment-proposal`: Broadcast new payment proposals
- `payment-vote`: Broadcast validator votes
- `payment-lock`: Broadcast payment locks
- `payment-settlement`: Broadcast final settlements

### Message Flow:
```
Node A (User Request)
  ↓ gossip: payment-proposal
  ├→ Validator 1 ─→ gossip: payment-vote (approve)
  ├→ Validator 2 ─→ gossip: payment-vote (approve)
  ├→ Validator 3 ─→ gossip: payment-vote (approve)
  └→ Validator 4 ─→ gossip: payment-vote (reject - low balance on their view)

Consensus: 3/4 = 75% approval ≥ 67% threshold ✓

Node A locks payment, starts AI generation
  ↓
AI completes
  ↓ gossip: payment-settlement
  ├→ Validator 1 ─→ Verify & sign
  ├→ Validator 2 ─→ Verify & sign
  ├→ Validator 3 ─→ Verify & sign
  └→ Validator 4 ─→ Verify & sign

Settlement finalized, refund issued
```

---

## Performance Optimization

### 1. Fast Path for High-Trust Users
- Users with history of valid payments get fast-tracked
- Skip full consensus for small amounts (<$0.01)
- Probabilistic verification (sample validators)

### 2. Batching
- Batch multiple small payments into one consensus round
- Reduce network overhead

### 3. Payment Channels
- Users can open payment channels with prepaid balance
- Only channel open/close requires consensus
- Intermediate payments are instant

---

## Security Guarantees

| Attack | Defense | Security Level |
|--------|---------|----------------|
| Invalid payment approval | BFT (67% threshold) | **High** |
| Double-spend | Payment locking + consensus | **Very High** |
| Price manipulation | Oracle consensus | **High** |
| Validator collusion | Economic slashing | **Medium-High** |
| Censorship | Multi-node submission | **High** |
| State inconsistency | Merkle root voting | **Very High** |

---

## Integration with Existing Systems

### DAG-Knight Consensus:
- Payment transactions included in DAG vertices
- Final settlement confirmed via DAG consensus
- Payment state is part of blockchain state

### AI Inference:
- Payment consensus runs parallel to AI generation
- If payment fails mid-generation, refund issued
- Generation results only released after payment confirmed

### Oracle System:
- Price feeds use existing q-oracle consensus
- Multiple price sources prevent manipulation
- Validators cross-verify prices

---

## Testing Strategy

1. **Unit Tests**: Individual component verification
2. **Integration Tests**: End-to-end payment flow
3. **Byzantine Tests**: Malicious validator scenarios
4. **Load Tests**: 1000+ concurrent payments
5. **Network Partition Tests**: Split-brain scenarios
6. **Economic Tests**: Slashing and incentive verification

---

## Deployment Phases

### Phase 1: Testnet (2 weeks)
- Deploy payment consensus on testnet
- 10 validator nodes
- Simulated Byzantine attacks
- Performance benchmarking

### Phase 2: Mainnet Soft Launch (1 week)
- Deploy with conservative thresholds
- Monitor for issues
- Gradual rollout to users

### Phase 3: Full Production (Ongoing)
- Enable all features
- Continuous monitoring
- Optimize based on real-world data

---

## Success Metrics

- **Consensus Latency**: <500ms for payment approval
- **False Positive Rate**: <0.01% (invalid payments approved)
- **False Negative Rate**: <0.1% (valid payments rejected)
- **Byzantine Tolerance**: Withstand up to 33% malicious nodes
- **State Sync Time**: <5 seconds
- **Throughput**: 10,000 payments/second network-wide

---

*This distributed payment consensus system ensures true decentralization while maintaining security and performance.*
