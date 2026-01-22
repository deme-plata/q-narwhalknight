// PaaS Billing Atomicity System
// Implements pre-charge, reserve, and rollback for atomic billing operations
//
// Transaction Flow:
// 1. Check balance (sufficient funds?)
// 2. Reserve funds (lock amount)
// 3. Execute service
// 4a. Success: Finalize (debit wallet, credit Quillon Bank)
// 4b. Failure: Release (unlock reserved funds)
//
// Features:
// - Atomic reserve/finalize/release operations
// - 5-minute reservation timeout with auto-release
// - Complete audit trail
// - Deadlock-free design with fine-grained locking

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::privacy_service_api::PaaSService;
use crate::AppState;

/// Reservation status state machine
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReservationStatus {
    /// Funds reserved, awaiting service execution
    Pending,
    /// Service succeeded, funds debited
    Finalized,
    /// Service failed or timeout, funds released
    Released,
    /// Reservation expired (auto-released after 5 minutes)
    Expired,
}

/// Balance reservation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceReservation {
    /// Unique reservation ID
    pub reservation_id: String,

    /// Wallet address holding reserved funds
    pub wallet_address: [u8; 32],

    /// Amount reserved (atomic units)
    pub amount_qug: u64,

    /// Service being paid for
    pub service: PaaSService,

    /// Additional service metadata
    pub metadata: serde_json::Value,

    /// Reservation creation timestamp
    pub created_at: DateTime<Utc>,

    /// Reservation expiration (5 minutes from creation)
    pub expires_at: DateTime<Utc>,

    /// Current status
    pub status: ReservationStatus,

    /// Finalization timestamp (if finalized)
    pub finalized_at: Option<DateTime<Utc>>,

    /// Release timestamp (if released)
    pub released_at: Option<DateTime<Utc>>,

    /// Billing transaction ID (if finalized)
    pub billing_tx_id: Option<String>,

    /// Error message (if failed)
    pub error_message: Option<String>,
}

impl BalanceReservation {
    /// Check if reservation has expired
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }

    /// Check if reservation is still active
    pub fn is_active(&self) -> bool {
        self.status == ReservationStatus::Pending && !self.is_expired()
    }
}

/// Billing manager with atomic operations
pub struct PaaSBillingManager {
    /// Active reservations (reservation_id -> BalanceReservation)
    reservations: Arc<RwLock<HashMap<String, BalanceReservation>>>,

    /// Reserved balance per wallet (wallet_address -> total_reserved)
    wallet_reserved_balances: Arc<RwLock<HashMap<[u8; 32], u64>>>,

    /// Reservation timeout (seconds)
    reservation_timeout: u64,
}

impl PaaSBillingManager {
    pub fn new() -> Self {
        let manager = Self {
            reservations: Arc::new(RwLock::new(HashMap::new())),
            wallet_reserved_balances: Arc::new(RwLock::new(HashMap::new())),
            reservation_timeout: 300, // 5 minutes
        };

        // Start background task to cleanup expired reservations
        manager.start_expiration_cleanup();

        manager
    }

    /// Reserve funds for a service (Step 1: Pre-charge)
    pub async fn reserve_funds(
        &self,
        wallet_address: [u8; 32],
        amount_qug: u64,
        service: PaaSService,
        metadata: serde_json::Value,
    ) -> Result<String, String> {
        // Generate reservation ID
        let reservation_id = Uuid::new_v4().to_string();
        let now = Utc::now();
        let expires_at = now + chrono::Duration::seconds(self.reservation_timeout as i64);

        // Create reservation record
        let reservation = BalanceReservation {
            reservation_id: reservation_id.clone(),
            wallet_address,
            amount_qug,
            service: service.clone(),
            metadata,
            created_at: now,
            expires_at,
            status: ReservationStatus::Pending,
            finalized_at: None,
            released_at: None,
            billing_tx_id: None,
            error_message: None,
        };

        // Add to reservations
        let mut reservations = self.reservations.write().await;
        reservations.insert(reservation_id.clone(), reservation);
        drop(reservations);

        // Update wallet reserved balance
        let mut wallet_reserved = self.wallet_reserved_balances.write().await;
        let current_reserved = wallet_reserved.entry(wallet_address).or_insert(0);
        *current_reserved += amount_qug;
        drop(wallet_reserved);

        info!(
            "💳 Reserved {} QUG for wallet {} (service: {:?}, reservation: {})",
            amount_qug as f64 / 1e24,
            hex::encode(&wallet_address[..8]),
            service,
            &reservation_id[..8]
        );

        Ok(reservation_id)
    }

    /// Check if wallet has sufficient available balance (total - reserved)
    pub async fn check_available_balance(
        &self,
        state: &AppState,
        wallet_address: &[u8; 32],
        required_amount: u64,
    ) -> Result<bool, String> {
        // Get total balance from Quillon Bank
        let quillon_bank = state.quillon_bank.read().await;
        let accounts = quillon_bank.accounts.read().await;

        // Convert wallet address to Quillon Bank Address
        use q_quillon_bank::{Address, AssetType};
        let address = Address::from_public_key(&{
            let mut key = [0u8; 33];
            key[0] = 0x02; // Compressed public key prefix
            key[1..33].copy_from_slice(wallet_address);
            key
        });

        let total_balance = accounts
            .get(&address)
            .and_then(|acc| acc.balances.get(&AssetType::ORB))
            .map(|bal| bal.available as u64)
            .unwrap_or(0);

        drop(accounts);
        drop(quillon_bank);

        // Get reserved balance
        let wallet_reserved = self.wallet_reserved_balances.read().await;
        let reserved_balance = wallet_reserved.get(wallet_address).copied().unwrap_or(0);
        drop(wallet_reserved);

        // Calculate available balance
        let available_balance = total_balance.saturating_sub(reserved_balance);

        info!(
            "💰 Wallet {} balance check: total={} QUG, reserved={} QUG, available={} QUG, required={} QUG",
            hex::encode(&wallet_address[..8]),
            total_balance as f64 / 1e24,
            reserved_balance as f64 / 1e24,
            available_balance as f64 / 1e24,
            required_amount as f64 / 1e24
        );

        Ok(available_balance >= required_amount)
    }

    /// Finalize reservation after successful service execution (Step 2a: Success)
    pub async fn finalize_reservation(
        &self,
        state: &AppState,
        reservation_id: &str,
    ) -> Result<String, String> {
        // Get reservation
        let mut reservations = self.reservations.write().await;
        let reservation = reservations
            .get_mut(reservation_id)
            .ok_or_else(|| format!("Reservation {} not found", reservation_id))?;

        // Verify reservation is still active
        if reservation.status != ReservationStatus::Pending {
            return Err(format!(
                "Reservation {} is not pending (status: {:?})",
                reservation_id, reservation.status
            ));
        }

        if reservation.is_expired() {
            reservation.status = ReservationStatus::Expired;
            return Err(format!("Reservation {} has expired", reservation_id));
        }

        let wallet_address = reservation.wallet_address;
        let amount_qug = reservation.amount_qug;
        let service = reservation.service.clone();

        // Generate billing transaction ID
        let billing_tx_id = Uuid::new_v4().to_string();

        // Update reservation status
        reservation.status = ReservationStatus::Finalized;
        reservation.finalized_at = Some(Utc::now());
        reservation.billing_tx_id = Some(billing_tx_id.clone());

        drop(reservations);

        // Debit customer wallet (actual charge)
        self.debit_wallet(state, &wallet_address, amount_qug)
            .await?;

        // Credit Quillon Bank master account
        self.credit_quillon_bank(state, amount_qug, service, wallet_address)
            .await?;

        // Update reserved balance
        let mut wallet_reserved = self.wallet_reserved_balances.write().await;
        if let Some(reserved) = wallet_reserved.get_mut(&wallet_address) {
            *reserved = reserved.saturating_sub(amount_qug);
            if *reserved == 0 {
                wallet_reserved.remove(&wallet_address);
            }
        }
        drop(wallet_reserved);

        info!(
            "✅ Finalized reservation {} - charged {} QUG from wallet {} (tx: {})",
            &reservation_id[..8],
            amount_qug as f64 / 1e24,
            hex::encode(&wallet_address[..8]),
            &billing_tx_id[..8]
        );

        Ok(billing_tx_id)
    }

    /// Release reservation after service failure (Step 2b: Failure)
    pub async fn release_reservation(
        &self,
        reservation_id: &str,
        error_message: Option<String>,
    ) -> Result<(), String> {
        // Get reservation
        let mut reservations = self.reservations.write().await;
        let reservation = reservations
            .get_mut(reservation_id)
            .ok_or_else(|| format!("Reservation {} not found", reservation_id))?;

        // Verify reservation is still pending
        if reservation.status != ReservationStatus::Pending {
            return Err(format!(
                "Reservation {} is not pending (status: {:?})",
                reservation_id, reservation.status
            ));
        }

        let wallet_address = reservation.wallet_address;
        let amount_qug = reservation.amount_qug;

        // Update reservation status
        reservation.status = ReservationStatus::Released;
        reservation.released_at = Some(Utc::now());
        reservation.error_message = error_message.clone();

        drop(reservations);

        // Update reserved balance
        let mut wallet_reserved = self.wallet_reserved_balances.write().await;
        if let Some(reserved) = wallet_reserved.get_mut(&wallet_address) {
            *reserved = reserved.saturating_sub(amount_qug);
            if *reserved == 0 {
                wallet_reserved.remove(&wallet_address);
            }
        }
        drop(wallet_reserved);

        warn!(
            "🔄 Released reservation {} - returned {} QUG to wallet {} (reason: {})",
            &reservation_id[..8],
            amount_qug as f64 / 1e24,
            hex::encode(&wallet_address[..8]),
            error_message.unwrap_or_else(|| "Service failed".to_string())
        );

        Ok(())
    }

    /// Get reservation details
    pub async fn get_reservation(&self, reservation_id: &str) -> Option<BalanceReservation> {
        let reservations = self.reservations.read().await;
        reservations.get(reservation_id).cloned()
    }

    /// Get all reservations for a wallet
    pub async fn get_wallet_reservations(
        &self,
        wallet_address: &[u8; 32],
    ) -> Vec<BalanceReservation> {
        let reservations = self.reservations.read().await;
        reservations
            .values()
            .filter(|r| r.wallet_address == *wallet_address)
            .cloned()
            .collect()
    }

    /// Debit wallet balance
    async fn debit_wallet(
        &self,
        state: &AppState,
        wallet_address: &[u8; 32],
        amount: u64,
    ) -> Result<(), String> {
        use q_quillon_bank::{Address, AssetType};

        let quillon_bank = state.quillon_bank.read().await;
        let mut accounts = quillon_bank.accounts.write().await;

        // Convert wallet address to Quillon Bank Address
        let address = Address::from_public_key(&{
            let mut key = [0u8; 33];
            key[0] = 0x02;
            key[1..33].copy_from_slice(wallet_address);
            key
        });

        let account = accounts
            .get_mut(&address)
            .ok_or_else(|| format!("Wallet {} not found", hex::encode(&wallet_address[..8])))?;

        let orb_balance = account
            .balances
            .get_mut(&AssetType::ORB)
            .ok_or_else(|| "ORB balance not found".to_string())?;

        // Check sufficient balance
        if orb_balance.available < amount as u128 {
            return Err(format!(
                "Insufficient balance: have {} QUG, need {} QUG",
                orb_balance.available as f64 / 1e24,
                amount as f64 / 1e24
            ));
        }

        // Debit balance
        orb_balance.available -= amount as u128;
        orb_balance.last_updated = Utc::now().timestamp() as u64;

        info!(
            "💸 Debited {} QUG from wallet {} (new balance: {} QUG)",
            amount as f64 / 1e24,
            hex::encode(&wallet_address[..8]),
            orb_balance.available as f64 / 1e24
        );

        Ok(())
    }

    /// Credit Quillon Bank master account
    async fn credit_quillon_bank(
        &self,
        state: &AppState,
        amount: u64,
        service: PaaSService,
        customer_wallet: [u8; 32],
    ) -> Result<(), String> {
        use q_quillon_bank::{Address, AssetType};
        use sha2::{Digest, Sha256};

        let quillon_bank = state.quillon_bank.read().await;
        let mut accounts = quillon_bank.accounts.write().await;

        // Get Quillon Bank master account
        let master_address = Address::from_public_key(&{
            let mut key = [0u8; 33];
            key[0] = 0x02;
            let mut hasher = Sha256::new();
            hasher.update(b"quillon_bank_master");
            let hash = hasher.finalize();
            key[1..33].copy_from_slice(&hash[..32]);
            key
        });

        let master_account = accounts.entry(master_address.clone()).or_insert_with(|| {
            use q_quillon_bank::{
                BankAccount, CreditScore, QuantumAccountFeatures, QuantumCreditData, RiskTier,
            };
            BankAccount {
                address: master_address.clone(),
                balances: std::collections::HashMap::new(),
                credit_score: CreditScore {
                    score: 850,
                    risk_tier: RiskTier::Excellent,
                    factors: vec![],
                    history: vec![],
                    quantum_enhancement: QuantumCreditData {
                        quantum_transaction_patterns: 1.0,
                        post_quantum_security_usage: 1.0,
                        vault_utilization_score: 1.0,
                        consensus_participation: 1.0,
                    },
                    last_calculated: Utc::now().timestamp() as u64,
                },
                identity: q_quillon_bank::identity::VerifiedIdentity {
                    id: "quillon_bank_master".to_string(),
                    verified: true,
                },
                privacy_tier: q_quillon_bank::PrivacyTier::Standard,
                wealth_agent: None,
                transaction_history: vec![],
                created_at: Utc::now().timestamp() as u64,
                last_activity: Utc::now().timestamp() as u64,
                quantum_features: QuantumAccountFeatures::default(),
            }
        });

        // Credit ORB balance
        let orb_balance = master_account
            .balances
            .entry(AssetType::ORB)
            .or_insert_with(|| q_quillon_bank::Balance {
                available: 0,
                locked: 0,
                staked: 0,
                borrowed: 0,
                lending: 0,
                quantum_secured: 0,
                last_updated: Utc::now().timestamp() as u64,
            });

        orb_balance.available += amount as u128;
        orb_balance.last_updated = Utc::now().timestamp() as u64;

        info!(
            "🏦 Credited {} QUG to Quillon Bank master account (service: {:?}, customer: {}, new balance: {} QUG)",
            amount as f64 / 1e24,
            service,
            hex::encode(&customer_wallet[..8]),
            orb_balance.available as f64 / 1e24
        );

        Ok(())
    }

    /// Background task to cleanup expired reservations
    fn start_expiration_cleanup(&self) {
        let reservations = self.reservations.clone();
        let wallet_reserved = self.wallet_reserved_balances.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));

            loop {
                interval.tick().await;

                let mut reservations_write = reservations.write().await;
                let mut wallet_reserved_write = wallet_reserved.write().await;

                let mut expired_count = 0;

                // Find and release expired reservations
                for reservation in reservations_write.values_mut() {
                    if reservation.status == ReservationStatus::Pending && reservation.is_expired()
                    {
                        // Release expired reservation
                        reservation.status = ReservationStatus::Expired;
                        reservation.released_at = Some(Utc::now());
                        reservation.error_message = Some("Reservation timeout".to_string());

                        // Update reserved balance
                        if let Some(reserved) =
                            wallet_reserved_write.get_mut(&reservation.wallet_address)
                        {
                            *reserved = reserved.saturating_sub(reservation.amount_qug);
                            if *reserved == 0 {
                                wallet_reserved_write.remove(&reservation.wallet_address);
                            }
                        }

                        expired_count += 1;

                        warn!(
                            "⏱️  Auto-released expired reservation {} (wallet: {}, amount: {} QUG)",
                            &reservation.reservation_id[..8],
                            hex::encode(&reservation.wallet_address[..8]),
                            reservation.amount_qug as f64 / 1e24
                        );
                    }
                }

                if expired_count > 0 {
                    info!("🧹 Cleaned up {} expired reservations", expired_count);
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reservation_lifecycle() {
        let manager = PaaSBillingManager::new();
        let wallet = [1u8; 32];
        let amount = 100_000_000; // 1 QUG

        // Reserve funds
        let reservation_id = manager
            .reserve_funds(
                wallet,
                amount,
                PaaSService::TorRelay,
                serde_json::json!({"data_size_mb": 1.0}),
            )
            .await
            .unwrap();

        // Check reservation exists
        let reservation = manager.get_reservation(&reservation_id).await;
        assert!(reservation.is_some());
        assert_eq!(reservation.unwrap().status, ReservationStatus::Pending);

        // Release reservation
        manager
            .release_reservation(&reservation_id, Some("Test release".to_string()))
            .await
            .unwrap();

        // Check reservation is released
        let reservation = manager.get_reservation(&reservation_id).await;
        assert_eq!(reservation.unwrap().status, ReservationStatus::Released);
    }

    #[tokio::test]
    async fn test_wallet_reserved_balance_tracking() {
        let manager = PaaSBillingManager::new();
        let wallet = [2u8; 32];

        // Reserve 1 QUG
        let _res1 = manager
            .reserve_funds(
                wallet,
                100_000_000,
                PaaSService::TorRelay,
                serde_json::json!({}),
            )
            .await
            .unwrap();

        // Reserve another 0.5 QUG
        let res2 = manager
            .reserve_funds(
                wallet,
                50_000_000,
                PaaSService::TransactionMixing,
                serde_json::json!({}),
            )
            .await
            .unwrap();

        // Check total reserved balance
        let reserved = manager.wallet_reserved_balances.read().await;
        assert_eq!(*reserved.get(&wallet).unwrap(), 150_000_000); // 1.5 QUG
        drop(reserved);

        // Release second reservation
        manager.release_reservation(&res2, None).await.unwrap();

        // Check reserved balance decreased
        let reserved = manager.wallet_reserved_balances.read().await;
        assert_eq!(*reserved.get(&wallet).unwrap(), 100_000_000); // 1 QUG
    }
}
