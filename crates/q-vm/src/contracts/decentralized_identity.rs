//! Decentralized Identity Contract - Q-NarwhalKnight
//!
//! v3.9.1-beta: On-chain identity management with inheritance support
//!
//! Features:
//! - Self-sovereign identity registration
//! - KYC level tracking (0-4)
//! - Beneficiary designation for inheritance
//! - Death certificate verification
//! - Automated inheritance execution
//! - Bank integration for institutional KYC
//!
//! This contract works in conjunction with the Quillon Bank API
//! to provide both on-chain and off-chain identity services.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{debug, info, warn, error};

/// KYC verification levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KycLevel {
    /// Level 0: Unverified - basic wallet only
    Unverified = 0,
    /// Level 1: Email verified
    EmailVerified = 1,
    /// Level 2: Phone verified
    PhoneVerified = 2,
    /// Level 3: Document verified (ID/Passport)
    DocumentVerified = 3,
    /// Level 4: Full KYC (institutional/bank verified)
    FullKyc = 4,
}

impl Default for KycLevel {
    fn default() -> Self {
        KycLevel::Unverified
    }
}

impl From<u8> for KycLevel {
    fn from(level: u8) -> Self {
        match level {
            0 => KycLevel::Unverified,
            1 => KycLevel::EmailVerified,
            2 => KycLevel::PhoneVerified,
            3 => KycLevel::DocumentVerified,
            4 => KycLevel::FullKyc,
            _ => KycLevel::Unverified,
        }
    }
}

/// On-chain identity record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnChainIdentity {
    /// Wallet address (32-byte public key hash)
    pub wallet_address: [u8; 32],
    /// Display name (optional, hash only stored on-chain)
    pub name_hash: Option<[u8; 32]>,
    /// Email hash (for privacy, only hash stored)
    pub email_hash: Option<[u8; 32]>,
    /// KYC verification level
    pub kyc_level: KycLevel,
    /// Block height when identity was created
    pub created_at_height: u64,
    /// Block height of last update
    pub updated_at_height: u64,
    /// Whether identity is verified by an authorized verifier
    pub is_verified: bool,
    /// Verifier address (bank or authorized entity)
    pub verifier_address: Option<[u8; 32]>,
    /// Is the account holder deceased?
    pub is_deceased: bool,
    /// Designated beneficiary wallet address
    pub beneficiary_address: Option<[u8; 32]>,
    /// Death certificate ID (if deceased)
    pub death_certificate_id: Option<String>,
    /// Recovery addresses for account recovery
    pub recovery_addresses: Vec<[u8; 32]>,
    /// Metadata hash for off-chain extended data
    pub metadata_ipfs_hash: Option<String>,
}

impl OnChainIdentity {
    pub fn new(wallet_address: [u8; 32], height: u64) -> Self {
        Self {
            wallet_address,
            name_hash: None,
            email_hash: None,
            kyc_level: KycLevel::Unverified,
            created_at_height: height,
            updated_at_height: height,
            is_verified: false,
            verifier_address: None,
            is_deceased: false,
            beneficiary_address: None,
            death_certificate_id: None,
            recovery_addresses: Vec::new(),
            metadata_ipfs_hash: None,
        }
    }
}

/// On-chain death certificate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnChainDeathCertificate {
    /// Unique certificate ID
    pub id: String,
    /// Deceased wallet address
    pub deceased_wallet: [u8; 32],
    /// Beneficiary wallet address
    pub beneficiary_wallet: [u8; 32],
    /// Issuer address (authorized entity)
    pub issuer_address: [u8; 32],
    /// Block height when issued
    pub issued_at_height: u64,
    /// Whether approved by governance/bank
    pub is_approved: bool,
    /// Approver address
    pub approver_address: Option<[u8; 32]>,
    /// Block height when approved
    pub approved_at_height: Option<u64>,
    /// Whether inheritance has been executed
    pub is_executed: bool,
    /// Execution transaction hash
    pub execution_tx_hash: Option<[u8; 32]>,
    /// Block height when executed
    pub executed_at_height: Option<u64>,
    /// Reason/notes
    pub reason: String,
    /// Evidence hash (off-chain documentation)
    pub evidence_hash: Option<[u8; 32]>,
}

/// Inheritance transfer record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InheritanceTransfer {
    /// Certificate ID that authorized this transfer
    pub certificate_id: String,
    /// From address (deceased)
    pub from_address: [u8; 32],
    /// To address (beneficiary)
    pub to_address: [u8; 32],
    /// Amount transferred (in base units)
    pub amount: u128,
    /// Token type (native or token address)
    pub token: TokenType,
    /// Block height of execution
    pub executed_at_height: u64,
    /// Transaction hash
    pub tx_hash: [u8; 32],
}

/// Token types for inheritance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenType {
    /// Native QUG token
    Native,
    /// Custom token by contract address
    Token([u8; 32]),
}

/// Contract events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IdentityEvent {
    /// Identity registered
    IdentityRegistered {
        wallet: [u8; 32],
        height: u64,
    },
    /// Identity updated
    IdentityUpdated {
        wallet: [u8; 32],
        height: u64,
        field: String,
    },
    /// KYC level changed
    KycLevelChanged {
        wallet: [u8; 32],
        old_level: KycLevel,
        new_level: KycLevel,
        verifier: [u8; 32],
    },
    /// Beneficiary designated
    BeneficiarySet {
        wallet: [u8; 32],
        beneficiary: [u8; 32],
        height: u64,
    },
    /// Death certificate issued
    DeathCertificateIssued {
        certificate_id: String,
        deceased: [u8; 32],
        beneficiary: [u8; 32],
        issuer: [u8; 32],
    },
    /// Death certificate approved
    DeathCertificateApproved {
        certificate_id: String,
        approver: [u8; 32],
    },
    /// Inheritance executed
    InheritanceExecuted {
        certificate_id: String,
        from: [u8; 32],
        to: [u8; 32],
        total_amount: u128,
    },
    /// Recovery address added
    RecoveryAddressAdded {
        wallet: [u8; 32],
        recovery: [u8; 32],
    },
}

/// Error types for identity contract
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IdentityError {
    /// Identity already exists
    IdentityExists,
    /// Identity not found
    IdentityNotFound,
    /// Not authorized
    Unauthorized,
    /// Invalid KYC level transition
    InvalidKycTransition,
    /// Already deceased
    AlreadyDeceased,
    /// Certificate not found
    CertificateNotFound,
    /// Certificate not approved
    CertificateNotApproved,
    /// Certificate already executed
    CertificateAlreadyExecuted,
    /// Insufficient balance for inheritance
    InsufficientBalance,
    /// Invalid beneficiary
    InvalidBeneficiary,
    /// Recovery address limit exceeded
    RecoveryAddressLimitExceeded,
}

impl std::fmt::Display for IdentityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IdentityError::IdentityExists => write!(f, "Identity already exists for this wallet"),
            IdentityError::IdentityNotFound => write!(f, "Identity not found"),
            IdentityError::Unauthorized => write!(f, "Unauthorized operation"),
            IdentityError::InvalidKycTransition => write!(f, "Invalid KYC level transition"),
            IdentityError::AlreadyDeceased => write!(f, "Account is already marked as deceased"),
            IdentityError::CertificateNotFound => write!(f, "Death certificate not found"),
            IdentityError::CertificateNotApproved => write!(f, "Death certificate not approved"),
            IdentityError::CertificateAlreadyExecuted => write!(f, "Inheritance already executed"),
            IdentityError::InsufficientBalance => write!(f, "Insufficient balance for inheritance"),
            IdentityError::InvalidBeneficiary => write!(f, "Invalid beneficiary address"),
            IdentityError::RecoveryAddressLimitExceeded => write!(f, "Recovery address limit exceeded"),
        }
    }
}

impl std::error::Error for IdentityError {}

/// Maximum recovery addresses per identity
const MAX_RECOVERY_ADDRESSES: usize = 5;

/// Decentralized Identity Contract
///
/// This contract manages on-chain identity data including:
/// - Identity registration and updates
/// - KYC verification
/// - Beneficiary designation
/// - Death certificates and inheritance
pub struct DecentralizedIdentityContract {
    /// Identities by wallet address
    identities: RwLock<HashMap<[u8; 32], OnChainIdentity>>,
    /// Death certificates by ID
    certificates: RwLock<HashMap<String, OnChainDeathCertificate>>,
    /// Inheritance transfers
    transfers: RwLock<Vec<InheritanceTransfer>>,
    /// Authorized verifiers (banks, institutions)
    authorized_verifiers: RwLock<Vec<[u8; 32]>>,
    /// Authorized issuers (for death certificates)
    authorized_issuers: RwLock<Vec<[u8; 32]>>,
    /// Event log
    events: RwLock<Vec<IdentityEvent>>,
    /// Contract owner
    owner: [u8; 32],
    /// Current block height (updated by VM)
    current_height: RwLock<u64>,
}

impl DecentralizedIdentityContract {
    /// Create a new identity contract
    pub fn new(owner: [u8; 32]) -> Self {
        info!("🪪 Initializing DecentralizedIdentityContract");
        Self {
            identities: RwLock::new(HashMap::new()),
            certificates: RwLock::new(HashMap::new()),
            transfers: RwLock::new(Vec::new()),
            authorized_verifiers: RwLock::new(vec![owner]),
            authorized_issuers: RwLock::new(vec![owner]),
            events: RwLock::new(Vec::new()),
            owner,
            current_height: RwLock::new(0),
        }
    }

    /// Update current block height
    pub fn set_block_height(&self, height: u64) {
        let mut h = self.current_height.write().unwrap();
        *h = height;
    }

    fn get_height(&self) -> u64 {
        *self.current_height.read().unwrap()
    }

    fn emit_event(&self, event: IdentityEvent) {
        let mut events = self.events.write().unwrap();
        events.push(event);
    }

    // ============================================================================
    // Identity Management
    // ============================================================================

    /// Register a new identity
    pub fn register_identity(
        &self,
        wallet_address: [u8; 32],
        caller: [u8; 32],
    ) -> Result<OnChainIdentity, IdentityError> {
        // Caller must be the wallet owner
        if wallet_address != caller {
            return Err(IdentityError::Unauthorized);
        }

        let mut identities = self.identities.write().unwrap();

        if identities.contains_key(&wallet_address) {
            return Err(IdentityError::IdentityExists);
        }

        let height = self.get_height();
        let identity = OnChainIdentity::new(wallet_address, height);
        identities.insert(wallet_address, identity.clone());

        info!("🪪 Identity registered for wallet: {:?}", hex::encode(&wallet_address[..4]));

        self.emit_event(IdentityEvent::IdentityRegistered {
            wallet: wallet_address,
            height,
        });

        Ok(identity)
    }

    /// Get identity by wallet address
    pub fn get_identity(&self, wallet: &[u8; 32]) -> Option<OnChainIdentity> {
        let identities = self.identities.read().unwrap();
        identities.get(wallet).cloned()
    }

    /// Update identity name hash
    pub fn update_name_hash(
        &self,
        wallet: [u8; 32],
        name_hash: [u8; 32],
        caller: [u8; 32],
    ) -> Result<(), IdentityError> {
        if wallet != caller {
            return Err(IdentityError::Unauthorized);
        }

        let mut identities = self.identities.write().unwrap();
        let identity = identities.get_mut(&wallet).ok_or(IdentityError::IdentityNotFound)?;

        identity.name_hash = Some(name_hash);
        identity.updated_at_height = self.get_height();

        self.emit_event(IdentityEvent::IdentityUpdated {
            wallet,
            height: identity.updated_at_height,
            field: "name_hash".to_string(),
        });

        Ok(())
    }

    /// Update identity email hash
    pub fn update_email_hash(
        &self,
        wallet: [u8; 32],
        email_hash: [u8; 32],
        caller: [u8; 32],
    ) -> Result<(), IdentityError> {
        if wallet != caller {
            return Err(IdentityError::Unauthorized);
        }

        let mut identities = self.identities.write().unwrap();
        let identity = identities.get_mut(&wallet).ok_or(IdentityError::IdentityNotFound)?;

        identity.email_hash = Some(email_hash);
        identity.updated_at_height = self.get_height();

        self.emit_event(IdentityEvent::IdentityUpdated {
            wallet,
            height: identity.updated_at_height,
            field: "email_hash".to_string(),
        });

        Ok(())
    }

    // ============================================================================
    // KYC Verification
    // ============================================================================

    /// Add authorized verifier (owner only)
    pub fn add_authorized_verifier(
        &self,
        verifier: [u8; 32],
        caller: [u8; 32],
    ) -> Result<(), IdentityError> {
        if caller != self.owner {
            return Err(IdentityError::Unauthorized);
        }

        let mut verifiers = self.authorized_verifiers.write().unwrap();
        if !verifiers.contains(&verifier) {
            verifiers.push(verifier);
            info!("🔐 Added authorized verifier: {:?}", hex::encode(&verifier[..4]));
        }
        Ok(())
    }

    /// Check if address is authorized verifier
    pub fn is_authorized_verifier(&self, address: &[u8; 32]) -> bool {
        let verifiers = self.authorized_verifiers.read().unwrap();
        verifiers.contains(address)
    }

    /// Verify identity and set KYC level (verifier only)
    pub fn verify_identity(
        &self,
        wallet: [u8; 32],
        new_level: KycLevel,
        verifier: [u8; 32],
    ) -> Result<(), IdentityError> {
        // Check verifier authorization
        if !self.is_authorized_verifier(&verifier) {
            return Err(IdentityError::Unauthorized);
        }

        let mut identities = self.identities.write().unwrap();
        let identity = identities.get_mut(&wallet).ok_or(IdentityError::IdentityNotFound)?;

        let old_level = identity.kyc_level;
        identity.kyc_level = new_level;
        identity.is_verified = true;
        identity.verifier_address = Some(verifier);
        identity.updated_at_height = self.get_height();

        info!("🔐 Identity verified for wallet {:?}: KYC level {} -> {}",
            hex::encode(&wallet[..4]), old_level as u8, new_level as u8);

        self.emit_event(IdentityEvent::KycLevelChanged {
            wallet,
            old_level,
            new_level,
            verifier,
        });

        Ok(())
    }

    // ============================================================================
    // Beneficiary & Inheritance
    // ============================================================================

    /// Set beneficiary for inheritance
    pub fn set_beneficiary(
        &self,
        wallet: [u8; 32],
        beneficiary: [u8; 32],
        caller: [u8; 32],
    ) -> Result<(), IdentityError> {
        if wallet != caller {
            return Err(IdentityError::Unauthorized);
        }

        // Cannot set self as beneficiary
        if wallet == beneficiary {
            return Err(IdentityError::InvalidBeneficiary);
        }

        let mut identities = self.identities.write().unwrap();
        let identity = identities.get_mut(&wallet).ok_or(IdentityError::IdentityNotFound)?;

        if identity.is_deceased {
            return Err(IdentityError::AlreadyDeceased);
        }

        identity.beneficiary_address = Some(beneficiary);
        identity.updated_at_height = self.get_height();

        info!("👥 Beneficiary set for wallet {:?} -> {:?}",
            hex::encode(&wallet[..4]), hex::encode(&beneficiary[..4]));

        self.emit_event(IdentityEvent::BeneficiarySet {
            wallet,
            beneficiary,
            height: identity.updated_at_height,
        });

        Ok(())
    }

    /// Add authorized issuer for death certificates (owner only)
    pub fn add_authorized_issuer(
        &self,
        issuer: [u8; 32],
        caller: [u8; 32],
    ) -> Result<(), IdentityError> {
        if caller != self.owner {
            return Err(IdentityError::Unauthorized);
        }

        let mut issuers = self.authorized_issuers.write().unwrap();
        if !issuers.contains(&issuer) {
            issuers.push(issuer);
            info!("🏦 Added authorized death certificate issuer: {:?}", hex::encode(&issuer[..4]));
        }
        Ok(())
    }

    /// Check if address is authorized issuer
    pub fn is_authorized_issuer(&self, address: &[u8; 32]) -> bool {
        let issuers = self.authorized_issuers.read().unwrap();
        issuers.contains(address)
    }

    /// Issue death certificate (authorized issuer only)
    pub fn issue_death_certificate(
        &self,
        deceased_wallet: [u8; 32],
        beneficiary_wallet: [u8; 32],
        reason: String,
        evidence_hash: Option<[u8; 32]>,
        issuer: [u8; 32],
    ) -> Result<String, IdentityError> {
        // Check issuer authorization
        if !self.is_authorized_issuer(&issuer) {
            return Err(IdentityError::Unauthorized);
        }

        // Check if identity exists
        let identities = self.identities.read().unwrap();
        let identity = identities.get(&deceased_wallet).ok_or(IdentityError::IdentityNotFound)?;

        if identity.is_deceased {
            return Err(IdentityError::AlreadyDeceased);
        }

        drop(identities);

        // Generate certificate ID
        let height = self.get_height();
        let cert_id = format!("DC-{}-{}", hex::encode(&deceased_wallet[..4]), height);

        let certificate = OnChainDeathCertificate {
            id: cert_id.clone(),
            deceased_wallet,
            beneficiary_wallet,
            issuer_address: issuer,
            issued_at_height: height,
            is_approved: false,
            approver_address: None,
            approved_at_height: None,
            is_executed: false,
            execution_tx_hash: None,
            executed_at_height: None,
            reason,
            evidence_hash,
        };

        let mut certificates = self.certificates.write().unwrap();
        certificates.insert(cert_id.clone(), certificate);

        info!("💀 Death certificate issued: {} for wallet {:?}",
            cert_id, hex::encode(&deceased_wallet[..4]));

        self.emit_event(IdentityEvent::DeathCertificateIssued {
            certificate_id: cert_id.clone(),
            deceased: deceased_wallet,
            beneficiary: beneficiary_wallet,
            issuer,
        });

        Ok(cert_id)
    }

    /// Get death certificate by ID
    pub fn get_death_certificate(&self, cert_id: &str) -> Option<OnChainDeathCertificate> {
        let certificates = self.certificates.read().unwrap();
        certificates.get(cert_id).cloned()
    }

    /// Approve death certificate (authorized verifier/owner only)
    pub fn approve_death_certificate(
        &self,
        cert_id: &str,
        approver: [u8; 32],
    ) -> Result<(), IdentityError> {
        // Must be verifier or owner
        if !self.is_authorized_verifier(&approver) && approver != self.owner {
            return Err(IdentityError::Unauthorized);
        }

        let mut certificates = self.certificates.write().unwrap();
        let certificate = certificates.get_mut(cert_id).ok_or(IdentityError::CertificateNotFound)?;

        if certificate.is_executed {
            return Err(IdentityError::CertificateAlreadyExecuted);
        }

        let height = self.get_height();
        certificate.is_approved = true;
        certificate.approver_address = Some(approver);
        certificate.approved_at_height = Some(height);

        // Mark identity as deceased
        drop(certificates);
        let mut identities = self.identities.write().unwrap();
        if let Some(identity) = identities.get_mut(&self.get_death_certificate(cert_id).unwrap().deceased_wallet) {
            identity.is_deceased = true;
            identity.death_certificate_id = Some(cert_id.to_string());
            identity.updated_at_height = height;
        }

        info!("✅ Death certificate approved: {}", cert_id);

        self.emit_event(IdentityEvent::DeathCertificateApproved {
            certificate_id: cert_id.to_string(),
            approver,
        });

        Ok(())
    }

    /// Execute inheritance transfer
    ///
    /// This function creates the inheritance transfer record.
    /// The actual balance transfer must be handled by the VM executor
    /// which has access to the balance state.
    pub fn execute_inheritance(
        &self,
        cert_id: &str,
        amount: u128,
        token: TokenType,
        tx_hash: [u8; 32],
        executor: [u8; 32],
    ) -> Result<InheritanceTransfer, IdentityError> {
        // Must be authorized
        if !self.is_authorized_verifier(&executor) && executor != self.owner {
            return Err(IdentityError::Unauthorized);
        }

        let mut certificates = self.certificates.write().unwrap();
        let certificate = certificates.get_mut(cert_id).ok_or(IdentityError::CertificateNotFound)?;

        if !certificate.is_approved {
            return Err(IdentityError::CertificateNotApproved);
        }

        if certificate.is_executed {
            return Err(IdentityError::CertificateAlreadyExecuted);
        }

        let height = self.get_height();
        certificate.is_executed = true;
        certificate.execution_tx_hash = Some(tx_hash);
        certificate.executed_at_height = Some(height);

        let transfer = InheritanceTransfer {
            certificate_id: cert_id.to_string(),
            from_address: certificate.deceased_wallet,
            to_address: certificate.beneficiary_wallet,
            amount,
            token,
            executed_at_height: height,
            tx_hash,
        };

        let from = certificate.deceased_wallet;
        let to = certificate.beneficiary_wallet;

        drop(certificates);

        let mut transfers = self.transfers.write().unwrap();
        transfers.push(transfer.clone());

        info!("💰 Inheritance executed: {} - {} QUG from {:?} to {:?}",
            cert_id, amount as f64 / 1e24, hex::encode(&from[..4]), hex::encode(&to[..4]));

        self.emit_event(IdentityEvent::InheritanceExecuted {
            certificate_id: cert_id.to_string(),
            from,
            to,
            total_amount: amount,
        });

        Ok(transfer)
    }

    // ============================================================================
    // Recovery Addresses
    // ============================================================================

    /// Add recovery address
    pub fn add_recovery_address(
        &self,
        wallet: [u8; 32],
        recovery: [u8; 32],
        caller: [u8; 32],
    ) -> Result<(), IdentityError> {
        if wallet != caller {
            return Err(IdentityError::Unauthorized);
        }

        let mut identities = self.identities.write().unwrap();
        let identity = identities.get_mut(&wallet).ok_or(IdentityError::IdentityNotFound)?;

        if identity.recovery_addresses.len() >= MAX_RECOVERY_ADDRESSES {
            return Err(IdentityError::RecoveryAddressLimitExceeded);
        }

        if !identity.recovery_addresses.contains(&recovery) {
            identity.recovery_addresses.push(recovery);
            identity.updated_at_height = self.get_height();

            self.emit_event(IdentityEvent::RecoveryAddressAdded {
                wallet,
                recovery,
            });
        }

        Ok(())
    }

    // ============================================================================
    // Queries
    // ============================================================================

    /// Get all pending death certificates
    pub fn get_pending_death_certificates(&self) -> Vec<OnChainDeathCertificate> {
        let certificates = self.certificates.read().unwrap();
        certificates.values()
            .filter(|c| !c.is_approved && !c.is_executed)
            .cloned()
            .collect()
    }

    /// Get all approved but unexecuted death certificates
    pub fn get_approved_death_certificates(&self) -> Vec<OnChainDeathCertificate> {
        let certificates = self.certificates.read().unwrap();
        certificates.values()
            .filter(|c| c.is_approved && !c.is_executed)
            .cloned()
            .collect()
    }

    /// Get inheritance transfers for a beneficiary
    pub fn get_inheritance_transfers(&self, beneficiary: &[u8; 32]) -> Vec<InheritanceTransfer> {
        let transfers = self.transfers.read().unwrap();
        transfers.iter()
            .filter(|t| &t.to_address == beneficiary)
            .cloned()
            .collect()
    }

    /// Get events (for indexing)
    pub fn get_events(&self, from_index: usize) -> Vec<IdentityEvent> {
        let events = self.events.read().unwrap();
        events.iter().skip(from_index).cloned().collect()
    }

    /// Get contract statistics
    pub fn get_stats(&self) -> IdentityContractStats {
        let identities = self.identities.read().unwrap();
        let certificates = self.certificates.read().unwrap();
        let transfers = self.transfers.read().unwrap();

        let verified_count = identities.values().filter(|i| i.is_verified).count();
        let deceased_count = identities.values().filter(|i| i.is_deceased).count();
        let pending_certs = certificates.values().filter(|c| !c.is_approved).count();
        let executed_certs = certificates.values().filter(|c| c.is_executed).count();

        IdentityContractStats {
            total_identities: identities.len(),
            verified_identities: verified_count,
            deceased_identities: deceased_count,
            total_certificates: certificates.len(),
            pending_certificates: pending_certs,
            executed_certificates: executed_certs,
            total_transfers: transfers.len(),
            total_transferred: transfers.iter().map(|t| t.amount).sum(),
        }
    }
}

/// Contract statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityContractStats {
    pub total_identities: usize,
    pub verified_identities: usize,
    pub deceased_identities: usize,
    pub total_certificates: usize,
    pub pending_certificates: usize,
    pub executed_certificates: usize,
    pub total_transfers: usize,
    pub total_transferred: u128,
}

// ============================================================================
// Contract Call Interface (for VM integration)
// ============================================================================

/// Contract call methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IdentityContractMethod {
    // Identity
    RegisterIdentity,
    GetIdentity { wallet: [u8; 32] },
    UpdateNameHash { name_hash: [u8; 32] },
    UpdateEmailHash { email_hash: [u8; 32] },

    // KYC
    AddAuthorizedVerifier { verifier: [u8; 32] },
    VerifyIdentity { wallet: [u8; 32], level: u8 },

    // Beneficiary
    SetBeneficiary { beneficiary: [u8; 32] },

    // Death Certificates
    AddAuthorizedIssuer { issuer: [u8; 32] },
    IssueDeathCertificate {
        deceased: [u8; 32],
        beneficiary: [u8; 32],
        reason: String
    },
    ApproveDeathCertificate { cert_id: String },
    ExecuteInheritance {
        cert_id: String,
        amount: u128,
        tx_hash: [u8; 32]
    },

    // Recovery
    AddRecoveryAddress { recovery: [u8; 32] },

    // Queries
    GetPendingCertificates,
    GetApprovedCertificates,
    GetInheritanceTransfers { beneficiary: [u8; 32] },
    GetStats,
}

/// Contract call result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IdentityContractResult {
    Success,
    Identity(Option<OnChainIdentity>),
    Certificate(Option<OnChainDeathCertificate>),
    Certificates(Vec<OnChainDeathCertificate>),
    CertificateId(String),
    Transfers(Vec<InheritanceTransfer>),
    Transfer(InheritanceTransfer),
    Stats(IdentityContractStats),
    Error(String),
}

impl DecentralizedIdentityContract {
    /// Execute a contract call
    pub fn execute_call(
        &self,
        method: IdentityContractMethod,
        caller: [u8; 32],
    ) -> IdentityContractResult {
        match method {
            IdentityContractMethod::RegisterIdentity => {
                match self.register_identity(caller, caller) {
                    Ok(identity) => IdentityContractResult::Identity(Some(identity)),
                    Err(e) => IdentityContractResult::Error(e.to_string()),
                }
            }

            IdentityContractMethod::GetIdentity { wallet } => {
                IdentityContractResult::Identity(self.get_identity(&wallet))
            }

            IdentityContractMethod::UpdateNameHash { name_hash } => {
                match self.update_name_hash(caller, name_hash, caller) {
                    Ok(()) => IdentityContractResult::Success,
                    Err(e) => IdentityContractResult::Error(e.to_string()),
                }
            }

            IdentityContractMethod::UpdateEmailHash { email_hash } => {
                match self.update_email_hash(caller, email_hash, caller) {
                    Ok(()) => IdentityContractResult::Success,
                    Err(e) => IdentityContractResult::Error(e.to_string()),
                }
            }

            IdentityContractMethod::AddAuthorizedVerifier { verifier } => {
                match self.add_authorized_verifier(verifier, caller) {
                    Ok(()) => IdentityContractResult::Success,
                    Err(e) => IdentityContractResult::Error(e.to_string()),
                }
            }

            IdentityContractMethod::VerifyIdentity { wallet, level } => {
                match self.verify_identity(wallet, KycLevel::from(level), caller) {
                    Ok(()) => IdentityContractResult::Success,
                    Err(e) => IdentityContractResult::Error(e.to_string()),
                }
            }

            IdentityContractMethod::SetBeneficiary { beneficiary } => {
                match self.set_beneficiary(caller, beneficiary, caller) {
                    Ok(()) => IdentityContractResult::Success,
                    Err(e) => IdentityContractResult::Error(e.to_string()),
                }
            }

            IdentityContractMethod::AddAuthorizedIssuer { issuer } => {
                match self.add_authorized_issuer(issuer, caller) {
                    Ok(()) => IdentityContractResult::Success,
                    Err(e) => IdentityContractResult::Error(e.to_string()),
                }
            }

            IdentityContractMethod::IssueDeathCertificate { deceased, beneficiary, reason } => {
                match self.issue_death_certificate(deceased, beneficiary, reason, None, caller) {
                    Ok(cert_id) => IdentityContractResult::CertificateId(cert_id),
                    Err(e) => IdentityContractResult::Error(e.to_string()),
                }
            }

            IdentityContractMethod::ApproveDeathCertificate { cert_id } => {
                match self.approve_death_certificate(&cert_id, caller) {
                    Ok(()) => IdentityContractResult::Success,
                    Err(e) => IdentityContractResult::Error(e.to_string()),
                }
            }

            IdentityContractMethod::ExecuteInheritance { cert_id, amount, tx_hash } => {
                match self.execute_inheritance(&cert_id, amount, TokenType::Native, tx_hash, caller) {
                    Ok(transfer) => IdentityContractResult::Transfer(transfer),
                    Err(e) => IdentityContractResult::Error(e.to_string()),
                }
            }

            IdentityContractMethod::AddRecoveryAddress { recovery } => {
                match self.add_recovery_address(caller, recovery, caller) {
                    Ok(()) => IdentityContractResult::Success,
                    Err(e) => IdentityContractResult::Error(e.to_string()),
                }
            }

            IdentityContractMethod::GetPendingCertificates => {
                IdentityContractResult::Certificates(self.get_pending_death_certificates())
            }

            IdentityContractMethod::GetApprovedCertificates => {
                IdentityContractResult::Certificates(self.get_approved_death_certificates())
            }

            IdentityContractMethod::GetInheritanceTransfers { beneficiary } => {
                IdentityContractResult::Transfers(self.get_inheritance_transfers(&beneficiary))
            }

            IdentityContractMethod::GetStats => {
                IdentityContractResult::Stats(self.get_stats())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_wallet(seed: u8) -> [u8; 32] {
        let mut wallet = [0u8; 32];
        wallet[0] = seed;
        wallet
    }

    #[test]
    fn test_register_identity() {
        let owner = create_test_wallet(1);
        let contract = DecentralizedIdentityContract::new(owner);
        contract.set_block_height(100);

        let user = create_test_wallet(2);
        let identity = contract.register_identity(user, user).unwrap();

        assert_eq!(identity.wallet_address, user);
        assert_eq!(identity.kyc_level, KycLevel::Unverified);
        assert!(!identity.is_verified);
        assert!(!identity.is_deceased);
    }

    #[test]
    fn test_verify_identity() {
        let owner = create_test_wallet(1);
        let contract = DecentralizedIdentityContract::new(owner);
        contract.set_block_height(100);

        let user = create_test_wallet(2);
        contract.register_identity(user, user).unwrap();

        // Owner is automatically an authorized verifier
        contract.verify_identity(user, KycLevel::FullKyc, owner).unwrap();

        let identity = contract.get_identity(&user).unwrap();
        assert_eq!(identity.kyc_level, KycLevel::FullKyc);
        assert!(identity.is_verified);
        assert_eq!(identity.verifier_address, Some(owner));
    }

    #[test]
    fn test_set_beneficiary() {
        let owner = create_test_wallet(1);
        let contract = DecentralizedIdentityContract::new(owner);
        contract.set_block_height(100);

        let user = create_test_wallet(2);
        let beneficiary = create_test_wallet(3);

        contract.register_identity(user, user).unwrap();
        contract.set_beneficiary(user, beneficiary, user).unwrap();

        let identity = contract.get_identity(&user).unwrap();
        assert_eq!(identity.beneficiary_address, Some(beneficiary));
    }

    #[test]
    fn test_death_certificate_flow() {
        let owner = create_test_wallet(1);
        let contract = DecentralizedIdentityContract::new(owner);
        contract.set_block_height(100);

        let deceased = create_test_wallet(2);
        let beneficiary = create_test_wallet(3);

        contract.register_identity(deceased, deceased).unwrap();

        // Issue death certificate (owner is authorized issuer)
        let cert_id = contract.issue_death_certificate(
            deceased,
            beneficiary,
            "Natural causes".to_string(),
            None,
            owner
        ).unwrap();

        // Check certificate is pending
        let pending = contract.get_pending_death_certificates();
        assert_eq!(pending.len(), 1);

        // Approve certificate
        contract.approve_death_certificate(&cert_id, owner).unwrap();

        // Check identity is marked deceased
        let identity = contract.get_identity(&deceased).unwrap();
        assert!(identity.is_deceased);

        // Check approved certificates
        let approved = contract.get_approved_death_certificates();
        assert_eq!(approved.len(), 1);

        // Execute inheritance
        let tx_hash = create_test_wallet(99);
        let transfer = contract.execute_inheritance(
            &cert_id,
            1_000_000_000_000_000_000_000_000u128, // 1 QUG
            TokenType::Native,
            tx_hash,
            owner
        ).unwrap();

        assert_eq!(transfer.from_address, deceased);
        assert_eq!(transfer.to_address, beneficiary);
    }

    #[test]
    fn test_cannot_execute_unapproved_certificate() {
        let owner = create_test_wallet(1);
        let contract = DecentralizedIdentityContract::new(owner);
        contract.set_block_height(100);

        let deceased = create_test_wallet(2);
        let beneficiary = create_test_wallet(3);

        contract.register_identity(deceased, deceased).unwrap();

        let cert_id = contract.issue_death_certificate(
            deceased, beneficiary, "Test".to_string(), None, owner
        ).unwrap();

        let tx_hash = create_test_wallet(99);
        let result = contract.execute_inheritance(&cert_id, 100, TokenType::Native, tx_hash, owner);

        assert!(matches!(result, Err(IdentityError::CertificateNotApproved)));
    }

    #[test]
    fn test_contract_stats() {
        let owner = create_test_wallet(1);
        let contract = DecentralizedIdentityContract::new(owner);
        contract.set_block_height(100);

        // Register some identities
        for i in 2..5 {
            let user = create_test_wallet(i);
            contract.register_identity(user, user).unwrap();
        }

        // Verify one
        let user2 = create_test_wallet(2);
        contract.verify_identity(user2, KycLevel::DocumentVerified, owner).unwrap();

        let stats = contract.get_stats();
        assert_eq!(stats.total_identities, 3);
        assert_eq!(stats.verified_identities, 1);
    }
}
