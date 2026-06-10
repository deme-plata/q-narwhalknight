//! Fee calculation module for QNK-INDEX
//!
//! Handles mint/redeem fees, management fees, and performance fees.

use crate::types::IndexError;

/// Mint fee in basis points (10 = 0.1%)
pub const MINT_FEE_BPS: u64 = 10;

/// Redeem fee in basis points (10 = 0.1%)
pub const REDEEM_FEE_BPS: u64 = 10;

/// Maximum management fee in basis points (500 = 5%)
pub const MAX_MANAGEMENT_FEE_BPS: u16 = 500;

/// Maximum performance fee in basis points (2000 = 20%)
pub const MAX_PERFORMANCE_FEE_BPS: u16 = 2000;

/// Protocol fee in basis points (5 = 0.05%)
pub const PROTOCOL_FEE_BPS: u64 = 5;

/// Apply mint fee and return (net_amount, fee_amount)
pub fn apply_mint_fee(gross_amount: u64) -> Result<(u64, u64), IndexError> {
    let fee = gross_amount
        .checked_mul(MINT_FEE_BPS)
        .ok_or(IndexError::ArithmeticOverflow)?
        / 10000;

    let net = gross_amount
        .checked_sub(fee)
        .ok_or(IndexError::ArithmeticUnderflow)?;

    Ok((net, fee))
}

/// Apply redeem fee and return (net_amount, fee_amount)
pub fn apply_redeem_fee(gross_amount: u64) -> Result<(u64, u64), IndexError> {
    let fee = gross_amount
        .checked_mul(REDEEM_FEE_BPS)
        .ok_or(IndexError::ArithmeticOverflow)?
        / 10000;

    let net = gross_amount
        .checked_sub(fee)
        .ok_or(IndexError::ArithmeticUnderflow)?;

    Ok((net, fee))
}

/// Calculate management fee for a period
/// Returns fee amount in QUG (8 decimals)
pub fn calculate_management_fee(
    nav: u64,
    management_fee_bps: u16,
    blocks_elapsed: u64,
) -> Result<u64, IndexError> {
    if management_fee_bps > MAX_MANAGEMENT_FEE_BPS {
        return Err(IndexError::FeeTooHigh);
    }

    // Blocks per year (assuming 6-second blocks)
    const BLOCKS_PER_YEAR: u64 = 365 * 24 * 60 * 60 / 6;

    // Pro-rata fee for the period
    let fee = (nav as u128)
        .checked_mul(management_fee_bps as u128)
        .ok_or(IndexError::ArithmeticOverflow)?
        .checked_mul(blocks_elapsed as u128)
        .ok_or(IndexError::ArithmeticOverflow)?
        / (BLOCKS_PER_YEAR as u128 * 10000);

    Ok(fee as u64)
}

/// Calculate performance fee based on high water mark
/// Returns fee amount in QUG (8 decimals)
pub fn calculate_performance_fee(
    current_nav_per_share: u64,
    high_water_mark: u64,
    total_supply: u64,
    performance_fee_bps: u16,
) -> Result<u64, IndexError> {
    if performance_fee_bps > MAX_PERFORMANCE_FEE_BPS {
        return Err(IndexError::FeeTooHigh);
    }

    // Only charge if above high water mark
    if current_nav_per_share <= high_water_mark {
        return Ok(0);
    }

    let gain_per_share = current_nav_per_share - high_water_mark;

    // Fee = gain * total_supply * fee_rate
    let fee = (gain_per_share as u128)
        .checked_mul(total_supply as u128)
        .ok_or(IndexError::ArithmeticOverflow)?
        .checked_mul(performance_fee_bps as u128)
        .ok_or(IndexError::ArithmeticOverflow)?
        / (100_000_000_u128 * 10000); // Normalize for 8 decimals and bps

    Ok(fee as u64)
}

/// Calculate protocol fee from a fee amount
pub fn calculate_protocol_fee(fee_amount: u64) -> u64 {
    fee_amount * PROTOCOL_FEE_BPS / 10000
}

/// Split fee between manager and protocol
pub fn split_fee(total_fee: u64) -> (u64, u64) {
    let protocol_portion = calculate_protocol_fee(total_fee);
    let manager_portion = total_fee.saturating_sub(protocol_portion);
    (manager_portion, protocol_portion)
}

/// Fee summary for display
#[derive(Debug, Clone)]
pub struct FeeSummary {
    pub mint_fee_percent: f64,
    pub redeem_fee_percent: f64,
    pub management_fee_percent: f64,
    pub performance_fee_percent: f64,
    pub protocol_fee_percent: f64,
}

impl FeeSummary {
    pub fn from_index(management_fee_bps: u16, performance_fee_bps: u16) -> Self {
        Self {
            mint_fee_percent: MINT_FEE_BPS as f64 / 100.0,
            redeem_fee_percent: REDEEM_FEE_BPS as f64 / 100.0,
            management_fee_percent: management_fee_bps as f64 / 100.0,
            performance_fee_percent: performance_fee_bps as f64 / 100.0,
            protocol_fee_percent: PROTOCOL_FEE_BPS as f64 / 100.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mint_fee() {
        let (net, fee) = apply_mint_fee(1_000_000_000).unwrap();
        assert_eq!(fee, 1_000_000); // 0.1% of 10 QUG = 0.01 QUG
        assert_eq!(net, 999_000_000);
    }

    #[test]
    fn test_redeem_fee() {
        let (net, fee) = apply_redeem_fee(1_000_000_000).unwrap();
        assert_eq!(fee, 1_000_000); // 0.1%
        assert_eq!(net, 999_000_000);
    }

    #[test]
    fn test_management_fee() {
        // 1% annual fee on 1000 QUG for ~1 year
        let nav = 1000_00000000u64; // 1000 QUG
        let blocks_per_year = 365 * 24 * 60 * 60 / 6;

        let fee = calculate_management_fee(nav, 100, blocks_per_year).unwrap();
        // Should be approximately 1% = 10 QUG
        assert!(fee > 9_00000000 && fee < 11_00000000);
    }

    #[test]
    fn test_performance_fee() {
        // 20% performance fee, NAV up 10%
        let hwm = 100_000_000; // 1.00 QUG
        let current = 110_000_000; // 1.10 QUG (10% gain)
        let supply = 1000_00000000; // 1000 shares

        let fee = calculate_performance_fee(current, hwm, supply, 2000).unwrap();
        // 10% gain * 1000 shares * 20% fee = 20 QUG
        assert!(fee > 19_00000000 && fee < 21_00000000);
    }

    #[test]
    fn test_no_performance_fee_below_hwm() {
        let hwm = 100_000_000;
        let current = 90_000_000; // Below HWM

        let fee = calculate_performance_fee(current, hwm, 1000_00000000, 2000).unwrap();
        assert_eq!(fee, 0);
    }

    #[test]
    fn test_fee_split() {
        let total = 100_000_000; // 1 QUG
        let (manager, protocol) = split_fee(total);

        // Protocol gets 0.05% = 0.05% of 1 QUG = 0.0005 QUG = 50000
        assert_eq!(protocol, 500);
        assert_eq!(manager, 99_999_500);
    }
}
