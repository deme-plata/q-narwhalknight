// ORB Currency with 18 Decimal Precision for DAGKnight VM
// Identical to the main implementation for consistency

use serde::{Deserialize, Serialize};
use std::fmt;

/// ORB currency with 18 decimal precision (1 ORB = 10^18 wei-ORB)
/// This makes ORB as divisible as Ethereum but with ultra-cheap fees
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Orb {
    /// Value stored in wei-ORB (smallest unit)
    /// 1 ORB = 1,000,000,000,000,000,000 wei-ORB (18 zeros)
    wei_orb: u128,
}

impl Orb {
    /// Number of decimal places for ORB (18, same as ETH)
    pub const DECIMALS: u8 = 18;

    /// One ORB in wei-ORB units
    pub const ONE_ORB: u128 = 1_000_000_000_000_000_000; // 10^18

    /// Ultra-cheap VM execution fee: 0.0000001 ORB per operation
    pub const VM_EXECUTION_FEE: u128 = 100_000_000_000; // 10^11 wei-ORB

    /// Smart contract deployment fee: 0.000001 ORB
    pub const CONTRACT_DEPLOYMENT_FEE: u128 = 1_000_000_000_000; // 10^12 wei-ORB

    /// Storage operation fee: 0.00000001 ORB per byte
    pub const STORAGE_FEE_PER_BYTE: u128 = 10_000_000_000; // 10^10 wei-ORB

    /// Create ORB from wei-ORB units
    pub fn from_wei_orb(wei_orb: u128) -> Self {
        Self { wei_orb }
    }

    /// Create ORB from floating point value
    pub fn from_orb(orb: f64) -> Self {
        let wei_orb = (orb * Self::ONE_ORB as f64) as u128;
        Self { wei_orb }
    }

    /// Get wei-ORB value
    pub fn wei_orb(&self) -> u128 {
        self.wei_orb
    }

    /// Convert to ORB as floating point
    pub fn to_orb(&self) -> f64 {
        self.wei_orb as f64 / Self::ONE_ORB as f64
    }

    /// Calculate VM execution cost
    pub fn vm_execution_cost(operations: u64) -> Self {
        Self::from_wei_orb(Self::VM_EXECUTION_FEE * operations as u128)
    }

    /// Calculate smart contract deployment cost
    pub fn contract_deployment_cost(bytecode_size: usize) -> Self {
        let base_fee = Self::CONTRACT_DEPLOYMENT_FEE;
        let storage_fee = Self::STORAGE_FEE_PER_BYTE * bytecode_size as u128;
        Self::from_wei_orb(base_fee + storage_fee)
    }

    /// Calculate storage cost
    pub fn storage_cost(bytes: usize) -> Self {
        Self::from_wei_orb(Self::STORAGE_FEE_PER_BYTE * bytes as u128)
    }
}

impl fmt::Display for Orb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.18} ORB", self.to_orb())
    }
}

impl std::ops::Add for Orb {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::from_wei_orb(self.wei_orb + other.wei_orb)
    }
}

impl std::ops::Sub for Orb {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::from_wei_orb(self.wei_orb - other.wei_orb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vm_costs() {
        let execution_cost = Orb::vm_execution_cost(1000);
        assert_eq!(execution_cost.to_orb(), 0.0001); // 1000 * 0.0000001

        let deployment_cost = Orb::contract_deployment_cost(1024);
        let expected = 0.000001 + (0.00000001 * 1024.0);
        assert!((deployment_cost.to_orb() - expected).abs() < 0.000000001);
    }
}
