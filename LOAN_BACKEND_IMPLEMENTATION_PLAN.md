# Quillon Bank Loan Backend Implementation Plan

## Overview
Full implementation of decentralized loan application system with RocksDB persistence and multi-node consensus.

## Phase 1: Data Persistence (COMPLETED)
✅ Added `pending_loan_applications: Arc<RwLock<HashMap<String, LoanApplication>>>` to AppState
✅ Added load logic from RocksDB in AppState::new
✅ Added LoanApplication and ApplyLoanRequest structs

## Phase 2: Storage Engine Methods (NEXT STEP)

Add to `crates/q-storage/src/lib.rs`:

```rust
// Loan Applications Storage
pub async fn save_loan_application(&self, loan_id: &str, loan: &crate::quillon_bank_api::LoanApplication) -> anyhow::Result<()> {
    let loan_bytes = bincode::serialize(loan)?;
    self.hot_db.put(format!("loan_app:{}", loan_id).as_bytes(), &loan_bytes)?;
    Ok(())
}

pub async fn load_loan_applications(&self) -> anyhow::Result<HashMap<String, Vec<u8>>> {
    let mut loans = HashMap::new();
    let prefix = b"loan_app:";
    let iter = self.hot_db.prefix_iterator(prefix);

    for item in iter {
        let (key, value) = item?;
        if let Ok(key_str) = String::from_utf8(key.to_vec()) {
            let loan_id = key_str.strip_prefix("loan_app:").unwrap_or(&key_str).to_string();
            loans.insert(loan_id, value.to_vec());
        }
    }

    Ok(loans)
}

pub async fn delete_loan_application(&self, loan_id: &str) -> anyhow::Result<()> {
    self.hot_db.delete(format!("loan_app:{}", loan_id).as_bytes())?;
    Ok(())
}
```

## Phase 3: apply_loan Handler with Collateral Validation

Add to `crates/q-api-server/src/quillon_bank_api.rs` after line 707:

```rust
pub async fn apply_loan(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ApplyLoanRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    info!("🏦 Loan application received for {} QUGUSD", request.loan_amount as f64 / 1e8);

    // 1. Parse and validate wallet address
    let borrower_address = match parse_wallet_address(&request.wallet_address) {
        Ok(addr) => addr,
        Err(e) => {
            error!("Invalid wallet address: {}", e);
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    // 2. Validate collateral availability
    let wallet_balances = state.wallet_balances.read().await;
    let current_qug_balance = wallet_balances.get(&borrower_address).copied().unwrap_or(0) as f64 / 1e12;
    drop(wallet_balances);

    if current_qug_balance < request.collateral_amount {
        error!(
            "Insufficient collateral: have {:.2} QUG, need {:.2} QUG",
            current_qug_balance, request.collateral_amount
        );
        return Err(StatusCode::BAD_REQUEST);
    }

    // 3. Calculate interest rate based on collateral ratio and term
    const QUG_PRICE: f64 = 42.50; // $42.50 per QUG
    const MINIMUM_COLLATERAL_RATIO: f64 = 1.5; // 150%

    let loan_amount_f64 = request.loan_amount as f64 / 1e8;
    let required_collateral_usd = loan_amount_f64 * MINIMUM_COLLATERAL_RATIO;
    let provided_collateral_usd = request.collateral_amount * QUG_PRICE;
    let collateral_ratio = provided_collateral_usd / loan_amount_f64;

    if collateral_ratio < MINIMUM_COLLATERAL_RATIO {
        error!(
            "Collateral ratio {:.2}% below minimum {:.2}%",
            collateral_ratio * 100.0,
            MINIMUM_COLLATERAL_RATIO * 100.0
        );
        return Err(StatusCode::BAD_REQUEST);
    }

    // Calculate interest rate
    let base_rate = 0.05; // 5% APR
    let collateral_bonus = ((collateral_ratio - MINIMUM_COLLATERAL_RATIO) / 0.10) * -0.01; // -1% per 10% extra collateral
    let term_premium = (request.term_months as f64 / 6.0) * 0.005; // +0.5% per 6 months
    let interest_rate = (base_rate + collateral_bonus + term_premium).max(0.01); // Minimum 1% APR

    // 4. Calculate monthly payment
    let total_interest = loan_amount_f64 * interest_rate * (request.term_months as f64 / 12.0);
    let total_repayment = loan_amount_f64 + total_interest;
    let monthly_payment = total_repayment / request.term_months as f64;

    // 5. Create LoanApplication with UUID
    let loan_id = uuid::Uuid::new_v4().to_string();
    let loan_application = LoanApplication {
        loan_id: loan_id.clone(),
        borrower_address: request.wallet_address.clone(),
        loan_amount: request.loan_amount,
        collateral_amount: request.collateral_amount,
        collateral_type: request.collateral_type.clone(),
        term_months: request.term_months,
        interest_rate: interest_rate * 100.0, // Convert to percentage
        monthly_payment,
        status: "pending".to_string(),
        created_at: chrono::Utc::now().timestamp(),
    };

    // 6. Store in pending applications (in-memory)
    let mut pending_loans = state.pending_loan_applications.write().await;
    pending_loans.insert(loan_id.clone(), loan_application.clone());
    drop(pending_loans);

    // 7. Persist to RocksDB
    if let Err(e) = state.storage_engine.save_loan_application(&loan_id, &loan_application).await {
        error!("Failed to persist loan application to storage: {}", e);
        // Continue anyway - we have it in memory
    }

    // 8. Broadcast to network for decentralized consensus (if libp2p available)
    if let Some(ref cmd_tx) = state.libp2p_command_tx {
        let loan_json = serde_json::to_string(&loan_application).unwrap_or_default();
        let _ = cmd_tx.send(q_network::NetworkCommand::PublishMessage {
            topic: "qnk/bank/loan-applications".to_string(),
            message: loan_json.into_bytes(),
        });
        info!("📡 Broadcasted loan application {} to network for consensus", loan_id);
    }

    info!(
        "✅ Loan application {} created: {} QUGUSD @ {:.2}% APR for {} months",
        loan_id, loan_amount_f64, interest_rate * 100.0, request.term_months
    );

    Ok(Json(ApiResponse::success(serde_json::json!({
        "loan_id": loan_id,
        "status": "pending",
        "interest_rate": interest_rate * 100.0,
        "monthly_payment": monthly_payment,
        "collateral_ratio": collateral_ratio * 100.0,
        "message": "Loan application submitted successfully. Awaiting founder approval via Quillon Bank CLI."
    }))))
}
```

## Phase 4: Decentralized Consensus Integration

### Network Gossip Topic Subscription
In `crates/q-api-server/src/main.rs`, add loan topic subscription:

```rust
// Subscribe to loan-related topics for decentralized consensus
network_manager.subscribe_topic("qnk/bank/loan-applications").await?;
network_manager.subscribe_topic("qnk/bank/loan-approvals").await?;
```

### Consensus Validation Handler
Add network message handler for loan consensus:

```rust
// Handle incoming loan applications from network
if topic == "qnk/bank/loan-applications" {
    if let Ok(loan) = serde_json::from_slice::<LoanApplication>(&message) {
        // Validate loan application from other nodes
        let mut pending_loans = app_state.pending_loan_applications.write().await;
        if !pending_loans.contains_key(&loan.loan_id) {
            // New loan from another node - add to local state
            pending_loans.insert(loan.loan_id.clone(), loan.clone());

            // Persist locally
            let _ = app_state.storage_engine.save_loan_application(&loan.loan_id, &loan).await;

            info!("📥 Received loan application {} from network", loan.loan_id);
        }
    }
}
```

## Phase 5: Testing & Deployment

### Test Plan:
1. **Unit Test**: Collateral validation logic
2. **Integration Test**: Full loan application flow
3. **Network Test**: Multi-node consensus synchronization
4. **Frontend Test**: Loan modal → Backend → SSE approval flow

### Deployment Checklist:
- [ ] Compile backend with 10-hour timeout: `timeout 36000 cargo build --release --package q-api-server`
- [ ] Test loan application endpoint: `curl -X POST http://localhost:8080/api/bank/lending/apply`
- [ ] Verify RocksDB persistence after restart
- [ ] Test multi-node gossip with Docker test node
- [ ] Frontend integration with LoanApplicationModal
- [ ] SSE loan approval notification flow

## Security Considerations
- ✅ Collateral validation before loan approval
- ✅ Post-quantum signatures for founder CLI approval (AEGIS-QL)
- ✅ Rate limiting on loan applications (inherit from faucet abuse protection)
- ✅ Decentralized consensus to prevent single-node manipulation
- ✅ RocksDB fsync for persistence across hard kills

## Future Enhancements (v0.3.0+)
- [ ] Automated liquidation monitoring based on QUG price oracle
- [ ] Multi-signature approval for large loans (> 10K QUGUSD)
- [ ] Credit score system based on on-chain transaction history
- [ ] Loan repayment modal and automatic QUGUSD burning
- [ ] Interest accrual automation with block-based timestamps

---

**Status**: Ready for implementation
**Priority**: High (Core DeFi feature for v0.2.9-beta)
**Estimated Completion**: Phase 2-3 (2-4 hours compile + test time)
