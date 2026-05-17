//! Comprehensive U128 Migration Tests (v3.0.4)
//!
//! This module tests all u128 monetary value migrations to ensure:
//! 1. 24-decimal precision (10^24 base units per 1 token)
//! 2. Correct arithmetic operations (no overflow/underflow)
//! 3. Backward compatibility with legacy u64 values
//! 4. JSON serialization as strings (to avoid JS 2^53 overflow)
//! 5. Proper conversion between display and base units

use std::collections::HashMap;

/// ONE_QNK constant: 1 QUG = 10^24 base units
pub const ONE_QNK: u128 = 1_000_000_000_000_000_000_000_000;

/// ONE_QUG constant (alias for ONE_QNK)
pub const ONE_QUG: u128 = ONE_QNK;

/// ONE_QUGUSD constant: 1 QUGUSD = 10^24 base units
pub const ONE_QUGUSD: u128 = ONE_QNK;

/// Max supply (21 million QUG)
pub const MAX_SUPPLY_QNK: u128 = 21_000_000 * ONE_QNK;

/// Legacy ONE_QNK for backward compatibility (8 decimals)
pub const LEGACY_ONE_QNK: u64 = 100_000_000; // 10^8

/// Conversion factor from legacy 8-decimal to 24-decimal
pub const LEGACY_TO_NEW_FACTOR: u128 = 10_000_000_000_000_000; // 10^16 (24-8 = 16 zeros)

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== PRECISION TESTS ====================

    #[test]
    fn test_one_qnk_constant() {
        // 1 QUG should equal 10^24 base units
        assert_eq!(ONE_QNK, 1_000_000_000_000_000_000_000_000u128);
        assert_eq!(ONE_QNK.to_string(), "1000000000000000000000000");

        // Verify it's the correct power of 10
        let expected = 10u128.pow(24);
        assert_eq!(ONE_QNK, expected);
    }

    #[test]
    fn test_max_supply() {
        // 21 million QUG max supply
        let expected = 21_000_000u128 * ONE_QNK;
        assert_eq!(MAX_SUPPLY_QNK, expected);

        // Verify no overflow
        assert!(MAX_SUPPLY_QNK < u128::MAX);

        // Max supply should be about 2.1 * 10^31
        assert!(MAX_SUPPLY_QNK > 10u128.pow(31));
        assert!(MAX_SUPPLY_QNK < 10u128.pow(32));
    }

    #[test]
    fn test_u128_range_sufficient() {
        // u128 max is about 3.4 * 10^38
        // Our max supply (2.1 * 10^31) has plenty of headroom
        let headroom = u128::MAX / MAX_SUPPLY_QNK;
        assert!(headroom > 1_000_000, "Should have >1M headroom factor");

        // Even 1000x max supply won't overflow
        let large_amount = MAX_SUPPLY_QNK * 1000;
        assert!(large_amount < u128::MAX);
    }

    // ==================== ARITHMETIC TESTS ====================

    #[test]
    fn test_basic_addition() {
        let balance1: u128 = 100 * ONE_QNK;  // 100 QUG
        let balance2: u128 = 50 * ONE_QNK;   // 50 QUG
        let total = balance1 + balance2;
        assert_eq!(total, 150 * ONE_QNK);    // 150 QUG
    }

    #[test]
    fn test_basic_subtraction() {
        let balance: u128 = 100 * ONE_QNK;   // 100 QUG
        let spend: u128 = 30 * ONE_QNK;      // 30 QUG
        let remaining = balance - spend;
        assert_eq!(remaining, 70 * ONE_QNK); // 70 QUG
    }

    #[test]
    fn test_saturating_sub_prevents_underflow() {
        let balance: u128 = 10 * ONE_QNK;    // 10 QUG
        let spend: u128 = 50 * ONE_QNK;      // 50 QUG
        let remaining = balance.saturating_sub(spend);
        assert_eq!(remaining, 0);            // Should be 0, not panic
    }

    #[test]
    fn test_multiplication() {
        let price: u128 = ONE_QNK / 10;      // 0.1 QUG per token
        let quantity: u128 = 50;
        let total = price * quantity;
        assert_eq!(total, 5 * ONE_QNK);      // 5 QUG total
    }

    #[test]
    fn test_division() {
        let total: u128 = 100 * ONE_QNK;     // 100 QUG
        let shares: u128 = 4;
        let per_share = total / shares;
        assert_eq!(per_share, 25 * ONE_QNK); // 25 QUG per share
    }

    #[test]
    fn test_precision_division() {
        // Test that we don't lose precision with small divisions
        let amount: u128 = ONE_QNK;          // 1 QUG
        let divisor: u128 = 3;
        let result = amount / divisor;

        // Should be approximately 0.333... QUG
        let expected = ONE_QNK / 3;
        assert_eq!(result, expected);

        // Verify we keep significant precision
        assert!(result > ONE_QNK / 4);  // Greater than 0.25 QUG
        assert!(result < ONE_QNK / 2);  // Less than 0.5 QUG
    }

    #[test]
    fn test_fractional_amounts() {
        // Test 1.5 QUG
        let one_and_half: u128 = ONE_QNK + ONE_QNK / 2;
        assert_eq!(one_and_half, 1_500_000_000_000_000_000_000_000u128);

        // Test 0.001 QUG (1 milli-QUG)
        let milli_qug: u128 = ONE_QNK / 1000;
        assert_eq!(milli_qug, 1_000_000_000_000_000_000_000u128);  // 10^21

        // Test 0.000001 QUG (1 micro-QUG)
        let micro_qug: u128 = ONE_QNK / 1_000_000;
        assert_eq!(micro_qug, 1_000_000_000_000_000_000u128);      // 10^18

        // Test 0.000000000000000001 QUG (1 atto-QUG) - still valid
        let atto_qug: u128 = ONE_QNK / 1_000_000_000_000_000_000;
        assert_eq!(atto_qug, 1_000_000u128);                       // 10^6

        // Test smallest representable unit (1 base unit)
        let smallest: u128 = 1;
        assert!(smallest > 0);
    }

    // ==================== CONVERSION TESTS ====================

    #[test]
    fn test_display_to_base_conversion() {
        // Convert "1.5" QUG display to base units
        // Note: f64 can't precisely represent 10^24, so we use approximate comparison
        let display = 1.5f64;
        let base_units: u128 = (display * 1e24) as u128;
        let expected: u128 = 1_500_000_000_000_000_000_000_000u128;
        // Allow 0.01% precision loss due to f64 limitations
        let tolerance = expected / 10000;
        assert!(
            (base_units as i128 - expected as i128).unsigned_abs() < tolerance,
            "Expected ~{}, got {} (tolerance {})",
            expected, base_units, tolerance
        );

        // Convert "0.000001" QUG display to base units
        let micro = 0.000001f64;
        let micro_base: u128 = (micro * 1e24) as u128;
        let expected_micro: u128 = 1_000_000_000_000_000_000u128;
        let micro_tolerance = expected_micro / 10000;
        assert!(
            (micro_base as i128 - expected_micro as i128).unsigned_abs() < micro_tolerance,
            "Expected ~{}, got {}",
            expected_micro, micro_base
        );
    }

    #[test]
    fn test_base_to_display_conversion() {
        // Convert 1.5 QUG in base units to display
        let base_units: u128 = 1_500_000_000_000_000_000_000_000u128;
        let display = base_units as f64 / 1e24;
        assert!((display - 1.5).abs() < 0.0001);

        // Large amounts
        let large_base: u128 = 1_000_000 * ONE_QNK;  // 1M QUG
        let large_display = large_base as f64 / 1e24;
        assert!((large_display - 1_000_000.0).abs() < 0.001);
    }

    #[test]
    fn test_legacy_migration() {
        // Legacy value: 100 QUG with 8 decimals = 100 * 10^8
        let legacy_value: u64 = 100 * LEGACY_ONE_QNK;

        // Convert to new 24-decimal format
        let new_value: u128 = (legacy_value as u128) * LEGACY_TO_NEW_FACTOR;

        // Should equal 100 QUG in new format
        assert_eq!(new_value, 100 * ONE_QNK);
    }

    #[test]
    fn test_legacy_fractional_migration() {
        // Legacy value: 1.5 QUG with 8 decimals = 150_000_000
        let legacy_value: u64 = 150_000_000;  // 1.5 * 10^8

        // Convert to new 24-decimal format
        let new_value: u128 = (legacy_value as u128) * LEGACY_TO_NEW_FACTOR;

        // Should equal 1.5 QUG in new format
        assert_eq!(new_value, ONE_QNK + ONE_QNK / 2);
    }

    // ==================== SERIALIZATION TESTS ====================

    #[test]
    fn test_u128_string_serialization() {
        // Large u128 values must be serialized as strings for JSON
        // to avoid JavaScript's 2^53 precision limit

        let large_value: u128 = 12_345_678_901_234_567_890_123_456u128;
        let serialized = large_value.to_string();
        assert_eq!(serialized, "12345678901234567890123456");

        // Parse back
        let parsed: u128 = serialized.parse().unwrap();
        assert_eq!(parsed, large_value);
    }

    #[test]
    fn test_json_string_roundtrip() {
        use serde::{Deserialize, Serialize};

        #[derive(Serialize, Deserialize)]
        struct TestBalance {
            #[serde(serialize_with = "serialize_u128_as_string")]
            #[serde(deserialize_with = "deserialize_u128_from_string")]
            balance: u128,
        }

        fn serialize_u128_as_string<S>(value: &u128, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            serializer.serialize_str(&value.to_string())
        }

        fn deserialize_u128_from_string<'de, D>(deserializer: D) -> Result<u128, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let s: String = serde::Deserialize::deserialize(deserializer)?;
            s.parse().map_err(serde::de::Error::custom)
        }

        let test = TestBalance {
            balance: 100 * ONE_QNK,
        };

        let json = serde_json::to_string(&test).unwrap();
        assert!(json.contains("\"100000000000000000000000000\""));

        let parsed: TestBalance = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.balance, 100 * ONE_QNK);
    }

    // ==================== COLLATERAL VAULT TESTS ====================

    #[test]
    fn test_collateral_vault_amounts() {
        // Simulate CollateralVault operations
        struct MockVault {
            locked_qug: u128,
            minted_qugusd: u128,
        }

        impl MockVault {
            fn mint(&mut self, qug_amount: u128) -> u128 {
                // 1:1 collateral ratio for simplicity
                self.locked_qug += qug_amount;
                self.minted_qugusd += qug_amount;
                qug_amount
            }

            fn redeem(&mut self, qugusd_amount: u128) -> u128 {
                self.minted_qugusd = self.minted_qugusd.saturating_sub(qugusd_amount);
                let qug_returned = qugusd_amount;
                self.locked_qug = self.locked_qug.saturating_sub(qug_returned);
                qug_returned
            }
        }

        let mut vault = MockVault {
            locked_qug: 0,
            minted_qugusd: 0,
        };

        // Mint 100 QUGUSD with 100 QUG collateral
        let minted = vault.mint(100 * ONE_QNK);
        assert_eq!(minted, 100 * ONE_QUGUSD);
        assert_eq!(vault.locked_qug, 100 * ONE_QNK);
        assert_eq!(vault.minted_qugusd, 100 * ONE_QUGUSD);

        // Redeem 50 QUGUSD
        let returned = vault.redeem(50 * ONE_QUGUSD);
        assert_eq!(returned, 50 * ONE_QNK);
        assert_eq!(vault.locked_qug, 50 * ONE_QNK);
        assert_eq!(vault.minted_qugusd, 50 * ONE_QUGUSD);
    }

    // ==================== MINING REWARDS TESTS ====================

    #[test]
    fn test_mining_rewards_precision() {
        // Block reward: 6.25 QUG (like Bitcoin halving)
        // Use integer arithmetic instead of f64 to avoid precision loss
        // 6.25 = 6 + 1/4 = 6 + 0.25
        let block_reward: u128 = ONE_QNK * 6 + ONE_QNK / 4;
        assert_eq!(block_reward, 6_250_000_000_000_000_000_000_000u128);

        // Verify it's 6.25 * ONE_QNK
        assert_eq!(block_reward, ONE_QNK * 6 + ONE_QNK / 4);
    }

    #[test]
    fn test_emission_calculation() {
        // Simulate total supply accumulation
        let block_reward: u128 = 6 * ONE_QNK + ONE_QNK / 4;  // 6.25 QUG
        let blocks_per_day: u128 = 24 * 60;  // ~1 block per minute
        let daily_emission = block_reward * blocks_per_day;

        // Daily emission should be about 9000 QUG
        let expected_daily = 9000 * ONE_QNK;
        assert!((daily_emission as i128 - expected_daily as i128).abs() < ONE_QNK as i128);
    }

    // ==================== STAKING TESTS ====================

    #[test]
    fn test_staking_amounts() {
        struct MinerStake {
            staked_amount: u128,
            rewards_earned: u128,
        }

        let stake = MinerStake {
            staked_amount: 1000 * ONE_QNK,  // 1000 QUG staked
            rewards_earned: 50 * ONE_QNK,   // 50 QUG rewards
        };

        // Verify amounts
        assert_eq!(stake.staked_amount / ONE_QNK, 1000);
        assert_eq!(stake.rewards_earned / ONE_QNK, 50);

        // Total value
        let total = stake.staked_amount + stake.rewards_earned;
        assert_eq!(total, 1050 * ONE_QNK);
    }

    #[test]
    fn test_stake_accumulation() {
        let mut total_staked: u128 = 0;

        // Simulate multiple stakers
        let stakes = vec![
            100 * ONE_QNK,
            250 * ONE_QNK,
            500 * ONE_QNK,
            1000 * ONE_QNK,
        ];

        for stake in &stakes {
            total_staked += stake;
        }

        assert_eq!(total_staked, 1850 * ONE_QNK);
    }

    // ==================== PRIVATE TRANSACTION TESTS ====================

    #[test]
    fn test_private_transaction_amounts() {
        struct PrivateTx {
            amount: u128,
            sender_balance: u128,
            fee: u128,
        }

        let tx = PrivateTx {
            amount: 100 * ONE_QNK,
            sender_balance: 500 * ONE_QNK,
            fee: ONE_QNK / 1000,  // 0.001 QUG fee
        };

        // Validate sufficient balance
        assert!(tx.sender_balance >= tx.amount + tx.fee);

        // Calculate remaining
        let remaining = tx.sender_balance - tx.amount - tx.fee;
        let expected = 500 * ONE_QNK - 100 * ONE_QNK - ONE_QNK / 1000;
        assert_eq!(remaining, expected);
    }

    // ==================== DEX POOL TESTS ====================

    #[test]
    fn test_dex_pool_liquidity() {
        struct LiquidityPool {
            reserve_qug: u128,
            reserve_qugusd: u128,
        }

        impl LiquidityPool {
            fn get_price(&self) -> f64 {
                self.reserve_qugusd as f64 / self.reserve_qug as f64
            }

            fn add_liquidity(&mut self, qug: u128, qugusd: u128) {
                self.reserve_qug += qug;
                self.reserve_qugusd += qugusd;
            }
        }

        let mut pool = LiquidityPool {
            reserve_qug: 10000 * ONE_QNK,
            reserve_qugusd: 10000 * ONE_QUGUSD,
        };

        // Initial price should be 1:1
        assert!((pool.get_price() - 1.0).abs() < 0.001);

        // Add liquidity
        pool.add_liquidity(5000 * ONE_QNK, 5000 * ONE_QUGUSD);

        // Price should still be 1:1
        assert!((pool.get_price() - 1.0).abs() < 0.001);
        assert_eq!(pool.reserve_qug, 15000 * ONE_QNK);
    }

    // ==================== AI CREDITS TESTS ====================

    #[test]
    fn test_ai_credits_balance() {
        struct AICredits {
            balance_qnk: u128,
            balance_qugusd: u128,
            total_spent_qnk: u128,
            total_spent_qugusd: u128,
        }

        let mut credits = AICredits {
            balance_qnk: 100 * ONE_QNK,
            balance_qugusd: 50 * ONE_QUGUSD,
            total_spent_qnk: 0,
            total_spent_qugusd: 0,
        };

        // Spend 10 QNK on AI inference
        let cost: u128 = 10 * ONE_QNK;
        credits.balance_qnk = credits.balance_qnk.saturating_sub(cost);
        credits.total_spent_qnk += cost;

        assert_eq!(credits.balance_qnk, 90 * ONE_QNK);
        assert_eq!(credits.total_spent_qnk, 10 * ONE_QNK);
    }

    // ==================== TREASURY TESTS ====================

    #[test]
    fn test_treasury_accumulation() {
        struct AITreasury {
            total_revenue_qnk: u128,
            total_revenue_qugusd: u128,
            total_requests: u64,
        }

        let mut treasury = AITreasury {
            total_revenue_qnk: 0,
            total_revenue_qugusd: 0,
            total_requests: 0,
        };

        // Simulate many AI requests
        let cost_per_request: u128 = ONE_QNK / 100;  // 0.01 QUG per request
        let num_requests = 10_000u64;

        treasury.total_revenue_qnk += cost_per_request * num_requests as u128;
        treasury.total_requests += num_requests;

        // Should have 100 QUG in revenue
        assert_eq!(treasury.total_revenue_qnk, 100 * ONE_QNK);

        // Average cost per request
        let avg_cost = treasury.total_revenue_qnk / treasury.total_requests as u128;
        assert_eq!(avg_cost, cost_per_request);
    }

    // ==================== EDGE CASES ====================

    #[test]
    fn test_zero_amounts() {
        let zero: u128 = 0;
        assert_eq!(zero + ONE_QNK, ONE_QNK);
        assert_eq!(ONE_QNK.saturating_sub(ONE_QNK), 0);
        assert_eq!(zero / 1, 0);
    }

    #[test]
    fn test_very_small_amounts() {
        // 1 base unit (smallest possible)
        let smallest: u128 = 1;
        assert!(smallest > 0);
        assert!(smallest < ONE_QNK);

        // Can still do arithmetic
        let doubled = smallest * 2;
        assert_eq!(doubled, 2);
    }

    #[test]
    fn test_very_large_amounts() {
        // Max supply check
        let total = MAX_SUPPLY_QNK;
        assert!(total < u128::MAX);

        // Can still add amounts within supply
        let amount1 = 10_000_000 * ONE_QNK;
        let amount2 = 11_000_000 * ONE_QNK;
        let sum = amount1 + amount2;
        assert_eq!(sum, MAX_SUPPLY_QNK);
    }

    #[test]
    fn test_percentage_calculations() {
        let amount: u128 = 100 * ONE_QNK;

        // 5% fee
        let fee_percent = 5u128;
        let fee = amount * fee_percent / 100;
        assert_eq!(fee, 5 * ONE_QNK);

        // 0.1% fee
        let small_fee_percent = 1u128;  // 0.1% = 1/1000
        let small_fee = amount / 1000;
        assert_eq!(small_fee, ONE_QNK / 10);
    }

    // ==================== HASH/COMMITMENT TESTS ====================

    #[test]
    fn test_amount_to_bytes_for_hashing() {
        // When creating balance commitments, we need to convert u128 to bytes
        let amount: u128 = 12345 * ONE_QNK;
        let bytes = amount.to_be_bytes();

        assert_eq!(bytes.len(), 16);  // u128 = 16 bytes

        // Can reconstruct
        let reconstructed = u128::from_be_bytes(bytes);
        assert_eq!(reconstructed, amount);
    }

    #[test]
    fn test_stark_trace_u128_split() {
        // STARK proofs work with u64 field elements
        // Split u128 into (lower_64, upper_64)
        let amount: u128 = 12_345_678_901_234_567_890_123_456u128;

        let amount_lo = (amount & u64::MAX as u128) as u64;
        let amount_hi = (amount >> 64) as u64;

        // Reconstruct
        let reconstructed = amount_lo as u128 | ((amount_hi as u128) << 64);
        assert_eq!(reconstructed, amount);
    }

    // ==================== BACKWARD COMPATIBILITY ====================

    #[test]
    fn test_detect_legacy_storage() {
        // Legacy values are stored as 8 bytes (u64)
        // New values are stored as 16 bytes (u128)

        let legacy_bytes = 100u64.to_be_bytes();
        assert_eq!(legacy_bytes.len(), 8);

        let new_bytes = 100u128.to_be_bytes();
        assert_eq!(new_bytes.len(), 16);

        // Can detect format by byte length
        fn parse_amount(bytes: &[u8]) -> u128 {
            match bytes.len() {
                8 => {
                    let legacy = u64::from_be_bytes(bytes.try_into().unwrap());
                    (legacy as u128) * LEGACY_TO_NEW_FACTOR
                }
                16 => u128::from_be_bytes(bytes.try_into().unwrap()),
                _ => panic!("Invalid byte length"),
            }
        }

        // Test legacy
        let legacy_100 = 100u64 * LEGACY_ONE_QNK;  // 100 QUG in old format
        let legacy_bytes = legacy_100.to_be_bytes();
        let parsed_legacy = parse_amount(&legacy_bytes);
        assert_eq!(parsed_legacy, 100 * ONE_QNK);

        // Test new
        let new_100 = 100 * ONE_QNK;
        let new_bytes = new_100.to_be_bytes();
        let parsed_new = parse_amount(&new_bytes);
        assert_eq!(parsed_new, 100 * ONE_QNK);
    }
}
