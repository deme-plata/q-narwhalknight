use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
/// Smart Contract Security Module
///
/// This module provides comprehensive security features for Orobit smart contracts,
/// including reentrancy protection, overflow checks, access control, and more.
/// Modeled after OpenZeppelin's security patterns but implemented in Rust/WASM.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// Type alias for U256 (using u128 for compatibility)
type U256 = u128;

/// Reentrancy Guard - Prevents reentrancy attacks similar to OpenZeppelin's ReentrancyGuard
#[derive(Debug, Clone)]
pub struct ReentrancyGuard {
    /// Maps contract address to current execution state
    execution_state: Arc<Mutex<HashMap<[u8; 32], ExecutionState>>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ExecutionState {
    NotEntered = 1,
    Entered = 2,
}

impl ReentrancyGuard {
    pub fn new() -> Self {
        Self {
            execution_state: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Begin a protected function call - equivalent to OpenZeppelin's nonReentrant modifier
    pub fn non_reentrant_start(&self, contract_address: [u8; 32]) -> Result<ReentrancyLock> {
        let mut state = self.execution_state.lock().unwrap();

        let current_state = state
            .get(&contract_address)
            .unwrap_or(&ExecutionState::NotEntered);

        if *current_state == ExecutionState::Entered {
            return Err(anyhow!("ReentrancyGuard: reentrant call"));
        }

        state.insert(contract_address, ExecutionState::Entered);

        Ok(ReentrancyLock {
            guard: self.clone(),
            contract_address,
        })
    }

    /// End a protected function call
    fn non_reentrant_end(&self, contract_address: [u8; 32]) {
        let mut state = self.execution_state.lock().unwrap();
        state.insert(contract_address, ExecutionState::NotEntered);
    }
}

/// RAII lock for reentrancy protection - automatically releases when dropped
pub struct ReentrancyLock {
    guard: ReentrancyGuard,
    contract_address: [u8; 32],
}

impl Drop for ReentrancyLock {
    fn drop(&mut self) {
        self.guard.non_reentrant_end(self.contract_address);
    }
}

/// Access Control - Role-based permissions similar to OpenZeppelin's AccessControl
#[derive(Debug, Clone)]
pub struct AccessControl {
    /// Maps (contract_address, role, account) -> bool
    role_members: Arc<Mutex<HashMap<([u8; 32], RoleId, [u8; 32]), bool>>>,
    /// Maps (contract_address, role) -> admin_role
    role_admins: Arc<Mutex<HashMap<([u8; 32], RoleId), RoleId>>>,
}

pub type RoleId = [u8; 32];

/// Standard roles
pub struct Roles;

impl Roles {
    pub const DEFAULT_ADMIN_ROLE: RoleId = [0u8; 32];
    pub const MINTER_ROLE: RoleId = {
        let mut role = [0u8; 32];
        role[0] = 1;
        role
    };
    pub const PAUSER_ROLE: RoleId = {
        let mut role = [0u8; 32];
        role[0] = 2;
        role
    };
    pub const UPGRADER_ROLE: RoleId = {
        let mut role = [0u8; 32];
        role[0] = 3;
        role
    };
}

impl AccessControl {
    pub fn new() -> Self {
        Self {
            role_members: Arc::new(Mutex::new(HashMap::new())),
            role_admins: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Check if account has role
    pub fn has_role(&self, contract_address: [u8; 32], role: RoleId, account: [u8; 32]) -> bool {
        let members = self.role_members.lock().unwrap();
        members
            .get(&(contract_address, role, account))
            .unwrap_or(&false)
            .clone()
    }

    /// Grant role to account (only by role admin)
    pub fn grant_role(
        &self,
        contract_address: [u8; 32],
        role: RoleId,
        account: [u8; 32],
        granter: [u8; 32],
    ) -> Result<()> {
        let admin_role = self.get_role_admin(contract_address, role);

        if !self.has_role(contract_address, admin_role, granter) {
            return Err(anyhow!("AccessControl: account is missing role"));
        }

        let mut members = self.role_members.lock().unwrap();
        members.insert((contract_address, role, account), true);

        Ok(())
    }

    /// Revoke role from account
    pub fn revoke_role(
        &self,
        contract_address: [u8; 32],
        role: RoleId,
        account: [u8; 32],
        revoker: [u8; 32],
    ) -> Result<()> {
        let admin_role = self.get_role_admin(contract_address, role);

        if !self.has_role(contract_address, admin_role, revoker) {
            return Err(anyhow!("AccessControl: account is missing role"));
        }

        let mut members = self.role_members.lock().unwrap();
        members.insert((contract_address, role, account), false);

        Ok(())
    }

    /// Get the admin role for a given role
    pub fn get_role_admin(&self, contract_address: [u8; 32], role: RoleId) -> RoleId {
        let admins = self.role_admins.lock().unwrap();
        admins
            .get(&(contract_address, role))
            .unwrap_or(&Roles::DEFAULT_ADMIN_ROLE)
            .clone()
    }

    /// Setup default admin
    pub fn setup_admin(&self, contract_address: [u8; 32], admin: [u8; 32]) {
        let mut members = self.role_members.lock().unwrap();
        members.insert((contract_address, Roles::DEFAULT_ADMIN_ROLE, admin), true);
    }
}

/// Pausable functionality - similar to OpenZeppelin's Pausable
#[derive(Debug, Clone)]
pub struct Pausable {
    /// Maps contract_address -> is_paused
    paused_state: Arc<Mutex<HashMap<[u8; 32], bool>>>,
}

impl Pausable {
    pub fn new() -> Self {
        Self {
            paused_state: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Check if contract is paused
    pub fn paused(&self, contract_address: [u8; 32]) -> bool {
        let state = self.paused_state.lock().unwrap();
        state.get(&contract_address).unwrap_or(&false).clone()
    }

    /// Pause the contract
    pub fn pause(&self, contract_address: [u8; 32]) -> Result<()> {
        if self.paused(contract_address) {
            return Err(anyhow!("Pausable: paused"));
        }

        let mut state = self.paused_state.lock().unwrap();
        state.insert(contract_address, true);

        Ok(())
    }

    /// Unpause the contract
    pub fn unpause(&self, contract_address: [u8; 32]) -> Result<()> {
        if !self.paused(contract_address) {
            return Err(anyhow!("Pausable: not paused"));
        }

        let mut state = self.paused_state.lock().unwrap();
        state.insert(contract_address, false);

        Ok(())
    }

    /// Modifier to require not paused
    pub fn when_not_paused(&self, contract_address: [u8; 32]) -> Result<()> {
        if self.paused(contract_address) {
            return Err(anyhow!("Pausable: paused"));
        }
        Ok(())
    }

    /// Modifier to require paused
    pub fn when_paused(&self, contract_address: [u8; 32]) -> Result<()> {
        if !self.paused(contract_address) {
            return Err(anyhow!("Pausable: not paused"));
        }
        Ok(())
    }
}

/// Safe Math operations - prevents overflow/underflow
pub struct SafeMath;

impl SafeMath {
    /// Safe addition
    pub fn safe_add(a: U256, b: U256) -> Result<U256> {
        let result = a
            .checked_add(b)
            .ok_or_else(|| anyhow!("SafeMath: addition overflow"))?;
        Ok(result)
    }

    /// Safe subtraction
    pub fn safe_sub(a: U256, b: U256) -> Result<U256> {
        let result = a
            .checked_sub(b)
            .ok_or_else(|| anyhow!("SafeMath: subtraction underflow"))?;
        Ok(result)
    }

    /// Safe multiplication
    pub fn safe_mul(a: U256, b: U256) -> Result<U256> {
        if a == 0 {
            return Ok(0);
        }
        let result = a
            .checked_mul(b)
            .ok_or_else(|| anyhow!("SafeMath: multiplication overflow"))?;
        Ok(result)
    }

    /// Safe division
    pub fn safe_div(a: U256, b: U256) -> Result<U256> {
        if b == 0 {
            return Err(anyhow!("SafeMath: division by zero"));
        }
        Ok(a / b)
    }

    /// Safe modulo
    pub fn safe_mod(a: U256, b: U256) -> Result<U256> {
        if b == 0 {
            return Err(anyhow!("SafeMath: modulo by zero"));
        }
        Ok(a % b)
    }
}

// U256 type already defined above

/// Pull Payment - secure payment withdrawal pattern
#[derive(Debug, Clone)]
pub struct PullPayment {
    /// Maps (contract_address, payee) -> amount_owed
    escrow: Arc<Mutex<HashMap<([u8; 32], [u8; 32]), U256>>>,
}

impl PullPayment {
    pub fn new() -> Self {
        Self {
            escrow: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Deposit payment for a payee (internal function)
    pub fn async_transfer(&self, contract_address: [u8; 32], dest: [u8; 32], amount: U256) {
        let mut escrow = self.escrow.lock().unwrap();
        let current = *escrow.get(&(contract_address, dest)).unwrap_or(&0);
        escrow.insert((contract_address, dest), current + amount);
    }

    /// Withdraw accumulated balance
    pub fn withdraw_payments(&self, contract_address: [u8; 32], payee: [u8; 32]) -> Result<U256> {
        let mut escrow = self.escrow.lock().unwrap();
        let amount = escrow.get(&(contract_address, payee)).unwrap_or(&0).clone();

        if amount == 0 {
            return Err(anyhow!("PullPayment: no payments to withdraw"));
        }

        escrow.insert((contract_address, payee), 0);

        // Here would be the actual transfer logic
        // For now, just return the amount that would be transferred
        Ok(amount)
    }

    /// Check payments owed to address
    pub fn payments(&self, contract_address: [u8; 32], dest: [u8; 32]) -> U256 {
        let escrow = self.escrow.lock().unwrap();
        escrow.get(&(contract_address, dest)).unwrap_or(&0).clone()
    }
}

/// Complete Security Suite - combines all security features
#[derive(Debug, Clone)]
pub struct SecuritySuite {
    pub reentrancy_guard: ReentrancyGuard,
    pub access_control: AccessControl,
    pub pausable: Pausable,
    pub pull_payment: PullPayment,
}

impl SecuritySuite {
    pub fn new() -> Self {
        Self {
            reentrancy_guard: ReentrancyGuard::new(),
            access_control: AccessControl::new(),
            pausable: Pausable::new(),
            pull_payment: PullPayment::new(),
        }
    }

    /// Initialize security for a new contract
    pub fn initialize_contract(&self, contract_address: [u8; 32], owner: [u8; 32]) -> Result<()> {
        // Set up default admin role
        self.access_control.setup_admin(contract_address, owner);

        // Contract starts unpaused
        // (Pausable starts with false by default)

        Ok(())
    }

    /// Execute a function with full security checks
    pub fn secure_execute<F, R>(
        &self,
        contract_address: [u8; 32],
        caller: [u8; 32],
        required_role: Option<RoleId>,
        f: F,
    ) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
        // 1. Check if contract is paused
        self.pausable.when_not_paused(contract_address)?;

        // 2. Check access control if role is required
        if let Some(role) = required_role {
            if !self.access_control.has_role(contract_address, role, caller) {
                return Err(anyhow!("AccessControl: account is missing role"));
            }
        }

        // 3. Apply reentrancy protection
        let _lock = self
            .reentrancy_guard
            .non_reentrant_start(contract_address)?;

        // 4. Execute the function
        f()
    }
}

/// Security Configuration for contract templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub reentrancy_protection: bool,
    pub overflow_protection: bool,
    pub access_control: bool,
    pub pausable: bool,
    pub pull_payments: bool,
    pub timelock_enabled: bool,
    pub multisig_required: bool,
    pub audit_status: AuditStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditStatus {
    NotAudited,
    InProgress,
    Audited,
    CertifiedSecure,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            reentrancy_protection: true,
            overflow_protection: true,
            access_control: true,
            pausable: true,
            pull_payments: true,
            timelock_enabled: false,
            multisig_required: false,
            audit_status: AuditStatus::NotAudited,
        }
    }
}

/// Security analyzer for contracts
pub struct SecurityAnalyzer;

impl SecurityAnalyzer {
    /// Analyze contract for security vulnerabilities
    pub fn analyze_contract(_bytecode: &[u8], config: &SecurityConfig) -> SecurityReport {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        // Check for reentrancy protection
        if !config.reentrancy_protection {
            issues.push(SecurityIssue {
                severity: Severity::High,
                issue_type: "Missing Reentrancy Protection".to_string(),
                description: "Contract lacks reentrancy guards on state-changing functions"
                    .to_string(),
                recommendation: "Add ReentrancyGuard and use nonReentrant modifier".to_string(),
            });
        }

        // Check for overflow protection
        if !config.overflow_protection {
            issues.push(SecurityIssue {
                severity: Severity::High,
                issue_type: "Missing Overflow Protection".to_string(),
                description: "Contract may be vulnerable to integer overflow/underflow".to_string(),
                recommendation: "Use SafeMath for all arithmetic operations".to_string(),
            });
        }

        // Check for access control
        if !config.access_control {
            issues.push(SecurityIssue {
                severity: Severity::Medium,
                issue_type: "Missing Access Control".to_string(),
                description: "Contract lacks proper role-based access control".to_string(),
                recommendation: "Implement AccessControl for sensitive functions".to_string(),
            });
        }

        // Generate recommendations based on contract type
        if config.pausable {
            recommendations.push("Consider implementing emergency pause functionality".to_string());
        }

        if config.pull_payments {
            recommendations.push("Use pull payment pattern for external transfers".to_string());
        }

        SecurityReport {
            overall_score: calculate_security_score(&issues),
            issues,
            recommendations,
            audit_status: config.audit_status.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityReport {
    pub overall_score: u8, // 0-100
    pub issues: Vec<SecurityIssue>,
    pub recommendations: Vec<String>,
    pub audit_status: AuditStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityIssue {
    pub severity: Severity,
    pub issue_type: String,
    pub description: String,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

fn calculate_security_score(issues: &[SecurityIssue]) -> u8 {
    let mut score = 100u8;

    for issue in issues {
        match issue.severity {
            Severity::Critical => score = score.saturating_sub(30),
            Severity::High => score = score.saturating_sub(20),
            Severity::Medium => score = score.saturating_sub(10),
            Severity::Low => score = score.saturating_sub(5),
        }
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reentrancy_guard() {
        let guard = ReentrancyGuard::new();
        let contract_addr = [1u8; 32];

        // First call should succeed
        let _lock1 = guard.non_reentrant_start(contract_addr).unwrap();

        // Second call should fail (reentrancy)
        assert!(guard.non_reentrant_start(contract_addr).is_err());

        // After first lock is dropped, should work again
        drop(_lock1);
        let _lock2 = guard.non_reentrant_start(contract_addr).unwrap();
    }

    #[test]
    fn test_access_control() {
        let ac = AccessControl::new();
        let contract_addr = [1u8; 32];
        let admin = [2u8; 32];
        let user = [3u8; 32];

        // Setup admin
        ac.setup_admin(contract_addr, admin);

        // Admin should have admin role
        assert!(ac.has_role(contract_addr, Roles::DEFAULT_ADMIN_ROLE, admin));

        // User should not have admin role
        assert!(!ac.has_role(contract_addr, Roles::DEFAULT_ADMIN_ROLE, user));

        // Admin can grant minter role to user
        ac.grant_role(contract_addr, Roles::MINTER_ROLE, user, admin)
            .unwrap();
        assert!(ac.has_role(contract_addr, Roles::MINTER_ROLE, user));
    }

    #[test]
    fn test_safe_math() {
        // Test normal operations
        assert_eq!(SafeMath::safe_add(10, 20).unwrap(), 30);
        assert_eq!(SafeMath::safe_sub(30, 10).unwrap(), 20);
        assert_eq!(SafeMath::safe_mul(5, 6).unwrap(), 30);
        assert_eq!(SafeMath::safe_div(30, 6).unwrap(), 5);

        // Test overflow detection
        assert!(SafeMath::safe_add(u128::MAX, 1).is_err());
        assert!(SafeMath::safe_sub(5, 10).is_err());
        assert!(SafeMath::safe_div(10, 0).is_err());
    }
}
