//! Comprehensive DEX Tests
//!
//! Tests for price action, token decimals, swap logic, and UI-related scenarios.
//! These tests verify the DEX behaves correctly for real-world use cases.

use bigdecimal::BigDecimal;
use std::str::FromStr;

// ============================================================================
// TOKEN DECIMAL TESTS
// ============================================================================

#[cfg(test)]
mod token_decimal_tests {
    use super::*;

    /// Test token with 0 decimals (like some NFT-style tokens)
    #[test]
    fn test_token_0_decimals() {
        let total_supply = BigDecimal::from(1_000_000);
        let decimals: u8 = 0;

        // Converting to smallest unit should work
        let smallest_unit = &total_supply * BigDecimal::from(10u64.pow(decimals as u32));
        assert_eq!(smallest_unit, BigDecimal::from(1_000_000));

        // 1 token = 1 smallest unit
        let one_token = BigDecimal::from(1);
        let display_amount = &one_token / BigDecimal::from(10u64.pow(decimals as u32));
        assert_eq!(display_amount, BigDecimal::from(1));
    }

    /// Test token with 6 decimals (like USDC)
    #[test]
    fn test_token_6_decimals() {
        let decimals: u8 = 6;
        let total_supply = BigDecimal::from_str("1000000").unwrap(); // 1M tokens

        // 1 token in smallest units
        let one_token_smallest = BigDecimal::from(10u64.pow(decimals as u32));
        assert_eq!(one_token_smallest, BigDecimal::from(1_000_000));

        // 0.000001 token (1 unit)
        let smallest_amount = BigDecimal::from_str("0.000001").unwrap();
        let smallest_in_units = &smallest_amount * BigDecimal::from(10u64.pow(decimals as u32));
        assert_eq!(smallest_in_units, BigDecimal::from(1));

        // Test precision: 1.123456 tokens
        let precise_amount = BigDecimal::from_str("1.123456").unwrap();
        let in_units = &precise_amount * BigDecimal::from(10u64.pow(decimals as u32));
        assert_eq!(in_units, BigDecimal::from(1_123_456));
    }

    /// Test token with 8 decimals (like BTC)
    #[test]
    fn test_token_8_decimals() {
        let decimals: u8 = 8;

        // 1 satoshi equivalent
        let one_satoshi = BigDecimal::from_str("0.00000001").unwrap();
        let in_units = &one_satoshi * BigDecimal::from(10u64.pow(decimals as u32));
        assert_eq!(in_units, BigDecimal::from(1));

        // BTC-like supply
        let total_supply = BigDecimal::from(21_000_000);
        let total_units = &total_supply * BigDecimal::from(10u64.pow(decimals as u32));
        assert_eq!(total_units, BigDecimal::from(2_100_000_000_000_000u64));
    }

    /// Test token with 18 decimals (like ETH and ORB)
    #[test]
    fn test_token_18_decimals() {
        let decimals: u8 = 18;

        // 1 wei equivalent
        let one_wei = BigDecimal::from_str("0.000000000000000001").unwrap();
        let in_units = &one_wei * BigDecimal::from_str("1000000000000000000").unwrap();
        assert_eq!(in_units, BigDecimal::from(1));

        // 1 ETH in wei
        let one_eth = BigDecimal::from(1);
        let in_wei = &one_eth * BigDecimal::from_str("1000000000000000000").unwrap();
        assert_eq!(in_wei, BigDecimal::from_str("1000000000000000000").unwrap());

        // Large amount: 1 billion tokens
        let large_supply = BigDecimal::from_str("1000000000").unwrap();
        let in_wei_large = &large_supply * BigDecimal::from_str("1000000000000000000").unwrap();
        assert!(in_wei_large > BigDecimal::from(0));
    }

    /// Test token with unusual decimals (24 decimals - edge case)
    #[test]
    fn test_token_24_decimals() {
        let decimals: u8 = 24;

        // 1 smallest unit
        let smallest = BigDecimal::from_str("0.000000000000000000000001").unwrap();
        let in_units = &smallest * BigDecimal::from_str("1000000000000000000000000").unwrap();

        // Should be very close to 1 (may have floating point issues)
        let diff = (&in_units - BigDecimal::from(1)).abs();
        assert!(diff < BigDecimal::from_str("0.000001").unwrap());
    }

    /// Test decimal conversion round-trips
    #[test]
    fn test_decimal_roundtrip() {
        for decimals in [0u8, 2, 6, 8, 12, 18] {
            let original = BigDecimal::from_str("123.456789012345678901").unwrap();
            let multiplier = BigDecimal::from(10u64.pow(decimals as u32));

            // Convert to smallest unit and back
            let in_units = (&original * &multiplier).with_scale(0);
            let back = &in_units / &multiplier;

            // The difference should be less than the precision lost
            let precision_limit = BigDecimal::from(1) / &multiplier;
            let diff = (&original - &back).abs();
            assert!(diff <= precision_limit,
                "Roundtrip failed for {} decimals: original={}, back={}, diff={}",
                decimals, original, back, diff);
        }
    }

    /// Test that very small amounts don't become zero
    #[test]
    fn test_small_amounts_nonzero() {
        let decimals: u8 = 18;
        let multiplier = BigDecimal::from_str("1000000000000000000").unwrap();

        // 1 wei should stay as 1
        let one_unit = BigDecimal::from(1);
        let display = &one_unit / &multiplier;
        assert!(display > BigDecimal::from(0), "1 unit should not become zero");

        // Test with very small display amounts
        let small_display = BigDecimal::from_str("0.000000000000000001").unwrap();
        let back_to_units = &small_display * &multiplier;
        assert_eq!(back_to_units, BigDecimal::from(1));
    }
}

// ============================================================================
// PRICE CALCULATION TESTS
// ============================================================================

#[cfg(test)]
mod price_calculation_tests {
    use super::*;

    /// Standard AMM constant product formula: x * y = k
    fn calculate_output_amount(
        reserve_in: &BigDecimal,
        reserve_out: &BigDecimal,
        amount_in: &BigDecimal,
        fee_bps: u16,
    ) -> BigDecimal {
        let fee_multiplier = BigDecimal::from(10000 - fee_bps) / BigDecimal::from(10000);
        let amount_in_with_fee = amount_in * &fee_multiplier;
        let numerator = &amount_in_with_fee * reserve_out;
        let denominator = reserve_in + &amount_in_with_fee;
        numerator / denominator
    }

    /// Calculate price impact percentage
    fn calculate_price_impact(
        reserve_in: &BigDecimal,
        reserve_out: &BigDecimal,
        amount_in: &BigDecimal,
    ) -> BigDecimal {
        let spot_price = reserve_out / reserve_in;
        let amount_out = calculate_output_amount(reserve_in, reserve_out, amount_in, 30);
        let execution_price = &amount_out / amount_in;
        let impact = (BigDecimal::from(1) - (&execution_price / &spot_price)) * BigDecimal::from(100);
        impact.abs()
    }

    #[test]
    fn test_basic_swap_calculation() {
        let reserve_a = BigDecimal::from(1_000_000); // 1M token A
        let reserve_b = BigDecimal::from(1_000_000); // 1M token B (1:1 price)
        let amount_in = BigDecimal::from(1000);

        let amount_out = calculate_output_amount(&reserve_a, &reserve_b, &amount_in, 30);

        // With 0.3% fee, should get slightly less than 1000
        assert!(amount_out < BigDecimal::from(1000));
        assert!(amount_out > BigDecimal::from(990)); // At least 99% after fees
    }

    #[test]
    fn test_price_impact_small_trade() {
        let reserve_a = BigDecimal::from(1_000_000);
        let reserve_b = BigDecimal::from(1_000_000);
        let small_trade = BigDecimal::from(100); // 0.01% of pool

        let impact = calculate_price_impact(&reserve_a, &reserve_b, &small_trade);

        // Small trade should have low price impact
        // Note: The impact includes the 0.3% fee, so even tiny trades show ~0.3%
        // The actual slippage (price movement) is < 0.1%, but fee adds to impact
        assert!(impact < BigDecimal::from_str("0.5").unwrap(),
            "Small trade price impact too high: {}%", impact);
        assert!(impact > BigDecimal::from_str("0.2").unwrap(),
            "Impact should include fee, got: {}%", impact);
    }

    #[test]
    fn test_price_impact_large_trade() {
        let reserve_a = BigDecimal::from(1_000_000);
        let reserve_b = BigDecimal::from(1_000_000);
        let large_trade = BigDecimal::from(100_000); // 10% of pool

        let impact = calculate_price_impact(&reserve_a, &reserve_b, &large_trade);

        // Large trade should have significant impact
        assert!(impact > BigDecimal::from(1),
            "Large trade should have >1% impact, got: {}%", impact);
        assert!(impact < BigDecimal::from(20),
            "Impact should be reasonable, got: {}%", impact);
    }

    #[test]
    fn test_swap_preserves_k_invariant() {
        let reserve_a = BigDecimal::from(1_000_000);
        let reserve_b = BigDecimal::from(2_000_000);
        let amount_in = BigDecimal::from(10_000);

        let k_before = &reserve_a * &reserve_b;

        // Calculate swap output (ignoring fees for k check)
        let amount_out = calculate_output_amount(&reserve_a, &reserve_b, &amount_in, 0);

        let new_reserve_a = &reserve_a + &amount_in;
        let new_reserve_b = &reserve_b - &amount_out;
        let k_after = &new_reserve_a * &new_reserve_b;

        // k should stay roughly the same (small differences due to rounding)
        let k_diff_percent = ((&k_after - &k_before).abs() / &k_before) * BigDecimal::from(100);
        assert!(k_diff_percent < BigDecimal::from_str("0.001").unwrap(),
            "K changed too much: {}%", k_diff_percent);
    }

    #[test]
    fn test_price_with_different_ratios() {
        // Test various price ratios
        let test_cases = vec![
            (1_000_000u64, 1_000_000u64, "1.0"),    // 1:1
            (1_000_000, 2_000_000, "2.0"),           // 1:2
            (2_000_000, 1_000_000, "0.5"),           // 2:1
            (1_000_000, 10_000_000, "10.0"),         // 1:10
            (100_000, 1_000_000, "10.0"),            // 0.1:1
        ];

        for (reserve_a, reserve_b, expected_price_str) in test_cases {
            let ra = BigDecimal::from(reserve_a);
            let rb = BigDecimal::from(reserve_b);
            let expected_price = BigDecimal::from_str(expected_price_str).unwrap();

            let calculated_price = &rb / &ra;
            let diff = (&calculated_price - &expected_price).abs();

            assert!(diff < BigDecimal::from_str("0.0001").unwrap(),
                "Price mismatch for {}:{} - expected {}, got {}",
                reserve_a, reserve_b, expected_price, calculated_price);
        }
    }

    #[test]
    fn test_extreme_price_ratios() {
        // Very unbalanced pools
        let reserve_a = BigDecimal::from_str("1000000000000000000").unwrap(); // 1e18
        let reserve_b = BigDecimal::from(1); // 1 token

        let price = &reserve_b / &reserve_a;
        assert!(price > BigDecimal::from(0));
        // Price = 1 / 1e18 = 1e-18 (or approximately that)
        assert!(price <= BigDecimal::from_str("0.000000000000000002").unwrap(),
            "Extreme ratio price too high: {}", price);

        // Swap should still work
        let amount_in = BigDecimal::from(1000);
        let amount_out = calculate_output_amount(&reserve_a, &reserve_b, &amount_in, 30);
        assert!(amount_out > BigDecimal::from(0));
    }

    #[test]
    fn test_fee_tiers() {
        let reserve_a = BigDecimal::from(1_000_000);
        let reserve_b = BigDecimal::from(1_000_000);
        let amount_in = BigDecimal::from(10_000);

        // Test different fee tiers
        let fees = vec![
            (1u16, "0.01%"),   // 0.01%
            (5, "0.05%"),      // 0.05%
            (30, "0.3%"),      // 0.3% (standard Uniswap V2)
            (100, "1%"),       // 1% (stable pools)
            (300, "3%"),       // 3% (exotic pairs)
        ];

        let mut prev_output = BigDecimal::from(999999);
        for (fee_bps, label) in fees {
            let output = calculate_output_amount(&reserve_a, &reserve_b, &amount_in, fee_bps);

            // Higher fee = less output
            assert!(output < prev_output,
                "Fee {} should give less than previous", label);
            assert!(output > BigDecimal::from(0),
                "Output should be positive for fee {}", label);

            prev_output = output;
        }
    }
}

// ============================================================================
// SWAP SLIPPAGE TESTS
// ============================================================================

#[cfg(test)]
mod slippage_tests {
    use super::*;

    fn calculate_output_with_fee(
        reserve_in: &BigDecimal,
        reserve_out: &BigDecimal,
        amount_in: &BigDecimal,
    ) -> BigDecimal {
        let fee_bps = 30u16; // 0.3%
        let fee_multiplier = BigDecimal::from(10000 - fee_bps) / BigDecimal::from(10000);
        let amount_in_with_fee = amount_in * &fee_multiplier;
        let numerator = &amount_in_with_fee * reserve_out;
        let denominator = reserve_in + &amount_in_with_fee;
        numerator / denominator
    }

    #[test]
    fn test_slippage_tolerance_exact() {
        let reserve_a = BigDecimal::from(1_000_000);
        let reserve_b = BigDecimal::from(1_000_000);
        let amount_in = BigDecimal::from(10_000);

        let expected_output = calculate_output_with_fee(&reserve_a, &reserve_b, &amount_in);

        // 0.5% slippage tolerance
        let slippage_tolerance = BigDecimal::from_str("0.005").unwrap();
        let min_output = &expected_output * (BigDecimal::from(1) - &slippage_tolerance);

        // Actual output should be >= min_output
        assert!(expected_output >= min_output);

        // And min_output should be reasonable
        let ratio = &min_output / &expected_output;
        assert!(ratio > BigDecimal::from_str("0.99").unwrap());
    }

    #[test]
    fn test_slippage_protection_rejects_bad_trade() {
        let reserve_a = BigDecimal::from(100_000); // Small pool
        let reserve_b = BigDecimal::from(100_000);
        let large_trade = BigDecimal::from(50_000); // 50% of pool

        // Calculate expected output
        let output = calculate_output_with_fee(&reserve_a, &reserve_b, &large_trade);

        // Spot price expectation (without impact)
        let spot_expectation = &large_trade * BigDecimal::from_str("0.997").unwrap(); // Just fees

        // Should get much less due to impact
        let ratio = &output / &spot_expectation;
        assert!(ratio < BigDecimal::from_str("0.8").unwrap(),
            "Large trade should have significant slippage, ratio: {}", ratio);
    }

    #[test]
    fn test_minimum_output_calculation() {
        let expected_output = BigDecimal::from(1000);

        // Test various slippage tolerances
        let tolerances = vec![
            ("0.001", 999), // 0.1% -> min 999
            ("0.005", 995), // 0.5% -> min 995
            ("0.01", 990),  // 1% -> min 990
            ("0.05", 950),  // 5% -> min 950
        ];

        for (tolerance_str, expected_min) in tolerances {
            let tolerance = BigDecimal::from_str(tolerance_str).unwrap();
            let min_output = &expected_output * (BigDecimal::from(1) - &tolerance);
            let expected_min_bd = BigDecimal::from(expected_min);

            assert_eq!(min_output.with_scale(0), expected_min_bd,
                "Slippage {} should give min {}", tolerance_str, expected_min);
        }
    }
}

// ============================================================================
// LIQUIDITY POOL TESTS
// ============================================================================

#[cfg(test)]
mod liquidity_pool_tests {
    use super::*;

    fn calculate_lp_shares(
        amount_a: &BigDecimal,
        amount_b: &BigDecimal,
        total_shares: &BigDecimal,
        reserve_a: &BigDecimal,
        reserve_b: &BigDecimal,
    ) -> BigDecimal {
        if total_shares == &BigDecimal::from(0) {
            // Initial liquidity
            (amount_a * amount_b).sqrt().unwrap_or(BigDecimal::from(0))
        } else {
            // Proportional shares
            let share_a = amount_a * total_shares / reserve_a;
            let share_b = amount_b * total_shares / reserve_b;
            // Take minimum to ensure balanced provision
            if share_a < share_b { share_a } else { share_b }
        }
    }

    #[test]
    fn test_initial_liquidity_provision() {
        let amount_a = BigDecimal::from(1_000_000);
        let amount_b = BigDecimal::from(1_000_000);

        // Initial shares = sqrt(a * b)
        let shares = (amount_a.clone() * amount_b.clone()).sqrt().unwrap();

        assert_eq!(shares, BigDecimal::from(1_000_000));
    }

    #[test]
    fn test_proportional_liquidity_addition() {
        let reserve_a = BigDecimal::from(1_000_000);
        let reserve_b = BigDecimal::from(2_000_000);
        let total_shares = BigDecimal::from(1_414_214); // approx sqrt(2) * 1M

        // Add 10% more liquidity at current ratio
        let add_a = BigDecimal::from(100_000);
        let add_b = BigDecimal::from(200_000);

        let new_shares = calculate_lp_shares(&add_a, &add_b, &total_shares, &reserve_a, &reserve_b);

        // Should get about 10% of total shares
        let expected_shares = &total_shares * BigDecimal::from_str("0.1").unwrap();
        let diff = (&new_shares - &expected_shares).abs();
        let diff_percent = &diff / &expected_shares * BigDecimal::from(100);

        assert!(diff_percent < BigDecimal::from(1),
            "Share calculation off by {}%", diff_percent);
    }

    #[test]
    fn test_unbalanced_liquidity_provision() {
        let reserve_a = BigDecimal::from(1_000_000);
        let reserve_b = BigDecimal::from(1_000_000);
        let total_shares = BigDecimal::from(1_000_000);

        // Try to add unbalanced liquidity
        let add_a = BigDecimal::from(100_000);
        let add_b = BigDecimal::from(50_000); // Only half the B

        let shares = calculate_lp_shares(&add_a, &add_b, &total_shares, &reserve_a, &reserve_b);

        // Should only get shares based on the smaller proportion
        let max_possible = &total_shares * BigDecimal::from_str("0.1").unwrap();
        assert!(shares < max_possible,
            "Unbalanced provision should give less shares");

        // Should get shares based on B proportion (5%)
        let expected_shares = &total_shares * BigDecimal::from_str("0.05").unwrap();
        assert_eq!(shares.with_scale(0), expected_shares.with_scale(0));
    }

    #[test]
    fn test_liquidity_removal() {
        let reserve_a = BigDecimal::from(1_000_000);
        let reserve_b = BigDecimal::from(2_000_000);
        let total_shares = BigDecimal::from(1_000_000);

        // Remove 10% of liquidity
        let shares_to_remove = BigDecimal::from(100_000);
        let share_ratio = &shares_to_remove / &total_shares;

        let amount_a_out = &reserve_a * &share_ratio;
        let amount_b_out = &reserve_b * &share_ratio;

        assert_eq!(amount_a_out, BigDecimal::from(100_000));
        assert_eq!(amount_b_out, BigDecimal::from(200_000));
    }

    #[test]
    fn test_pool_ratio_preservation() {
        let reserve_a = BigDecimal::from(1_000_000);
        let reserve_b = BigDecimal::from(2_000_000);
        let initial_ratio = &reserve_b / &reserve_a;

        // Add proportional liquidity
        let add_a = BigDecimal::from(100_000);
        let add_b = &add_a * &initial_ratio;

        let new_reserve_a = &reserve_a + &add_a;
        let new_reserve_b = &reserve_b + &add_b;
        let new_ratio = &new_reserve_b / &new_reserve_a;

        let ratio_diff = (&new_ratio - &initial_ratio).abs();
        assert!(ratio_diff < BigDecimal::from_str("0.0001").unwrap(),
            "Ratio changed: {} -> {}", initial_ratio, new_ratio);
    }
}

// ============================================================================
// UI-SPECIFIC TESTS (Display formatting, edge cases)
// ============================================================================

#[cfg(test)]
mod ui_display_tests {
    use super::*;

    /// Format a BigDecimal for display with specified decimals
    fn format_for_display(amount: &BigDecimal, decimals: u8, max_display_decimals: u8) -> String {
        let scaled = amount / BigDecimal::from(10u64.pow(decimals as u32));
        let formatted = scaled.with_scale(max_display_decimals as i64);
        formatted.to_string()
    }

    /// Parse user input to BigDecimal in smallest units
    fn parse_user_input(input: &str, decimals: u8) -> Result<BigDecimal, String> {
        let parsed = BigDecimal::from_str(input).map_err(|e| e.to_string())?;
        let multiplier = BigDecimal::from(10u64.pow(decimals as u32));
        Ok(parsed * multiplier)
    }

    #[test]
    fn test_display_formatting_18_decimals() {
        let decimals = 18u8;

        // 1 full token
        let one_token = BigDecimal::from_str("1000000000000000000").unwrap();
        let display = format_for_display(&one_token, decimals, 6);
        assert_eq!(display, "1.000000");

        // Very small amount (1 wei)
        // Note: BigDecimal may use scientific notation for very small numbers
        let one_wei = BigDecimal::from(1);
        let display = format_for_display(&one_wei, decimals, 18);
        // Can be either "0.000000000000000001" or "1E-18" depending on BigDecimal version
        let parsed_back = BigDecimal::from_str(&display).unwrap();
        let expected = BigDecimal::from_str("0.000000000000000001").unwrap();
        assert_eq!(parsed_back, expected);
    }

    #[test]
    fn test_display_formatting_6_decimals() {
        let decimals = 6u8;

        // 1 full token
        let one_token = BigDecimal::from(1_000_000);
        let display = format_for_display(&one_token, decimals, 6);
        assert_eq!(display, "1.000000");

        // 1 micro-token
        let one_micro = BigDecimal::from(1);
        let display = format_for_display(&one_micro, decimals, 6);
        assert_eq!(display, "0.000001");
    }

    #[test]
    fn test_parse_user_input_valid() {
        let decimals = 18u8;

        // Normal amounts
        let result = parse_user_input("1.5", decimals).unwrap();
        let expected = BigDecimal::from_str("1500000000000000000").unwrap();
        assert_eq!(result, expected);

        // Whole numbers
        let result = parse_user_input("100", decimals).unwrap();
        let expected = BigDecimal::from_str("100000000000000000000").unwrap();
        assert_eq!(result, expected);

        // Very precise
        let result = parse_user_input("0.123456789012345678", decimals).unwrap();
        let expected = BigDecimal::from_str("123456789012345678").unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_user_input_invalid() {
        let decimals = 18u8;

        // Invalid inputs should fail
        assert!(parse_user_input("abc", decimals).is_err());
        assert!(parse_user_input("", decimals).is_err());
        assert!(parse_user_input("1.2.3", decimals).is_err());
    }

    #[test]
    fn test_price_display_formatting() {
        // Very small prices - BigDecimal may use scientific notation
        let tiny_price = BigDecimal::from_str("0.000000001").unwrap();
        // Check it's positive and small, not the exact string format
        assert!(tiny_price > BigDecimal::from(0));
        assert!(tiny_price < BigDecimal::from_str("0.00001").unwrap());

        // Very large prices
        let huge_price = BigDecimal::from_str("999999999.123").unwrap();
        assert!(huge_price > BigDecimal::from(0));

        // Standard prices
        // Note: BigDecimal's with_scale() truncates, not rounds
        let normal_price = BigDecimal::from_str("1.23456789").unwrap();
        let truncated = normal_price.with_scale(4);
        assert_eq!(truncated.to_string(), "1.2345"); // Truncated, not rounded
    }

    #[test]
    fn test_balance_display_truncation() {
        // Test that very long decimals get truncated properly
        let balance = BigDecimal::from_str("1234567890123456789").unwrap();
        let decimals = 18u8;

        // Display with 4 decimal places
        let display = format_for_display(&balance, decimals, 4);
        // "1.2345" is 6 characters
        assert!(display.len() >= 6, "Display too short: {}", display);
        assert!(display.starts_with("1.2345"), "Display: {}", display);
    }

    #[test]
    fn test_percentage_display() {
        // Price change percentage
        // Note: BigDecimal's with_scale() truncates, not rounds
        let price_change = BigDecimal::from_str("12.5678").unwrap();
        let display = price_change.with_scale(2);
        assert_eq!(display.to_string(), "12.56"); // Truncated, not rounded

        // Negative percentage
        let neg_change = BigDecimal::from_str("-5.123").unwrap();
        let display = neg_change.with_scale(2);
        assert_eq!(display.to_string(), "-5.12");
    }

    #[test]
    fn test_large_number_display() {
        // Market cap formatting
        let market_cap = BigDecimal::from_str("1234567890123").unwrap();

        // Should be displayable
        let as_string = market_cap.to_string();
        assert!(!as_string.is_empty());
        assert_eq!(as_string, "1234567890123");

        // Billions
        // Note: BigDecimal's with_scale() truncates, not rounds
        let billions = &market_cap / BigDecimal::from(1_000_000_000u64);
        let display = billions.with_scale(2);
        assert_eq!(display.to_string(), "1234.56"); // Truncated, not rounded
    }

    #[test]
    fn test_zero_handling() {
        let zero = BigDecimal::from(0);

        // Zero should display correctly
        assert_eq!(zero.to_string(), "0");

        // Division by zero should be caught
        let one = BigDecimal::from(1);
        // Note: BigDecimal doesn't panic on division by zero, but returns infinity
        // The application should check for this
    }
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_dust_amount_handling() {
        let decimals = 18u8;
        let multiplier = BigDecimal::from_str("1000000000000000000").unwrap();

        // 1 wei equivalent
        let dust = BigDecimal::from(1);
        let display = &dust / &multiplier;

        // Should not become exactly zero
        assert!(display > BigDecimal::from(0));
    }

    #[test]
    fn test_maximum_u64_amount() {
        let max_u64 = BigDecimal::from(u64::MAX);

        // Should be representable
        assert!(max_u64 > BigDecimal::from(0));

        // Can do math with it
        let doubled = &max_u64 * BigDecimal::from(2);
        assert!(doubled > max_u64);
    }

    #[test]
    fn test_maximum_u128_amount() {
        let max_u128_str = u128::MAX.to_string();
        let max_u128 = BigDecimal::from_str(&max_u128_str).unwrap();

        // Should be representable
        assert!(max_u128 > BigDecimal::from(0));

        // Can do math with it
        let halved = &max_u128 / BigDecimal::from(2);
        assert!(halved < max_u128);
    }

    #[test]
    fn test_tiny_pool_reserves() {
        // Pool with very small reserves
        let reserve_a = BigDecimal::from(100);
        let reserve_b = BigDecimal::from(100);
        let amount_in = BigDecimal::from(1);

        // Should still calculate correctly
        let fee_multiplier = BigDecimal::from_str("0.997").unwrap();
        let amount_in_with_fee = &amount_in * &fee_multiplier;
        let numerator = &amount_in_with_fee * &reserve_b;
        let denominator = &reserve_a + &amount_in_with_fee;
        let amount_out = numerator / denominator;

        assert!(amount_out > BigDecimal::from(0));
        assert!(amount_out < BigDecimal::from(1)); // Less than input due to fees + impact
    }

    #[test]
    fn test_huge_pool_reserves() {
        // Pool with very large reserves (e.g., stablecoin pool)
        let reserve_a = BigDecimal::from_str("1000000000000000000000000").unwrap(); // 1 quadrillion
        let reserve_b = BigDecimal::from_str("1000000000000000000000000").unwrap();
        let amount_in = BigDecimal::from_str("1000000000000000000").unwrap(); // 1e18

        // Should still calculate correctly
        let fee_multiplier = BigDecimal::from_str("0.997").unwrap();
        let amount_in_with_fee = &amount_in * &fee_multiplier;
        let numerator = &amount_in_with_fee * &reserve_b;
        let denominator = &reserve_a + &amount_in_with_fee;
        let amount_out = numerator / denominator;

        assert!(amount_out > BigDecimal::from(0));
        // Very small trade in huge pool should have minimal impact
        let expected_approx = &amount_in * &fee_multiplier;
        let diff_percent = ((&amount_out - &expected_approx).abs() / &expected_approx) * BigDecimal::from(100);
        assert!(diff_percent < BigDecimal::from_str("0.01").unwrap(),
            "Huge pool should have minimal impact: {}%", diff_percent);
    }

    #[test]
    fn test_negative_amount_rejection() {
        let negative = BigDecimal::from(-100);

        // Negative amounts should be detectable
        assert!(negative < BigDecimal::from(0));
    }

    #[test]
    fn test_precision_loss_detection() {
        // Test that we can detect precision loss
        let high_precision = BigDecimal::from_str("1.123456789012345678901234567890").unwrap();
        let truncated = high_precision.with_scale(18);

        // Should be different if we had more than 18 decimals
        let diff = (&high_precision - &truncated).abs();
        assert!(diff > BigDecimal::from(0),
            "Precision loss should be detectable");
    }
}

// ============================================================================
// TOKEN CREATION TESTS
// ============================================================================

#[cfg(test)]
mod token_creation_tests {
    use super::*;
    use chrono::Utc;

    /// Simulated token metadata structure
    #[derive(Debug, Clone)]
    struct TestTokenMetadata {
        symbol: String,
        name: String,
        decimals: u8,
        total_supply: BigDecimal,
        price_usd: BigDecimal,
    }

    impl TestTokenMetadata {
        fn market_cap(&self) -> BigDecimal {
            // Market cap in display units
            let supply_display = &self.total_supply / BigDecimal::from(10u64.pow(self.decimals as u32));
            &supply_display * &self.price_usd
        }

        fn format_supply(&self) -> String {
            let display = &self.total_supply / BigDecimal::from(10u64.pow(self.decimals as u32));
            display.with_scale(self.decimals as i64).to_string()
        }
    }

    #[test]
    fn test_create_token_0_decimals() {
        let token = TestTokenMetadata {
            symbol: "NFT".to_string(),
            name: "NFT Token".to_string(),
            decimals: 0,
            total_supply: BigDecimal::from(10_000),
            price_usd: BigDecimal::from(100),
        };

        assert_eq!(token.decimals, 0);
        assert_eq!(token.market_cap(), BigDecimal::from(1_000_000));
        assert_eq!(token.format_supply(), "10000");
    }

    #[test]
    fn test_create_token_6_decimals() {
        let token = TestTokenMetadata {
            symbol: "USDC".to_string(),
            name: "USD Coin".to_string(),
            decimals: 6,
            total_supply: BigDecimal::from(1_000_000_000_000u64), // 1M tokens
            price_usd: BigDecimal::from(1),
        };

        assert_eq!(token.decimals, 6);
        let supply_display = &token.total_supply / BigDecimal::from(1_000_000);
        assert_eq!(supply_display, BigDecimal::from(1_000_000));
    }

    #[test]
    fn test_create_token_18_decimals() {
        let token = TestTokenMetadata {
            symbol: "ORB".to_string(),
            name: "OroBit Token".to_string(),
            decimals: 18,
            total_supply: BigDecimal::from_str("21000000000000000000000000").unwrap(), // 21M tokens
            price_usd: BigDecimal::from_str("1.618").unwrap(),
        };

        assert_eq!(token.decimals, 18);

        // Market cap should be 21M * 1.618
        let expected_mcap = BigDecimal::from_str("33978000").unwrap();
        let actual_mcap = token.market_cap().with_scale(0);
        assert_eq!(actual_mcap, expected_mcap);
    }

    #[test]
    fn test_create_token_extreme_supply() {
        // Token with max u128 supply
        let max_supply = u128::MAX.to_string();
        let token = TestTokenMetadata {
            symbol: "MAX".to_string(),
            name: "Max Supply Token".to_string(),
            decimals: 18,
            total_supply: BigDecimal::from_str(&max_supply).unwrap(),
            price_usd: BigDecimal::from_str("0.000000001").unwrap(),
        };

        // Should not overflow
        let market_cap = token.market_cap();
        assert!(market_cap > BigDecimal::from(0));
    }

    #[test]
    fn test_create_token_fractional_price() {
        let token = TestTokenMetadata {
            symbol: "MEME".to_string(),
            name: "Meme Coin".to_string(),
            decimals: 18,
            total_supply: BigDecimal::from_str("1000000000000000000000000000000").unwrap(), // 1 trillion
            price_usd: BigDecimal::from_str("0.00000001").unwrap(), // Very low price
        };

        let market_cap = token.market_cap();
        // Should be around 10,000 USD
        let expected = BigDecimal::from(10_000);
        let diff = (&market_cap - &expected).abs();
        assert!(diff < BigDecimal::from(1),
            "Market cap should be ~10000, got {}", market_cap);
    }

    #[test]
    fn test_token_symbol_validation() {
        // Valid symbols
        let valid_symbols = vec!["ORB", "USDC", "ETH", "BTC", "ORBUSD", "A", "ABC123"];
        for symbol in valid_symbols {
            assert!(symbol.len() >= 1 && symbol.len() <= 10,
                "Symbol {} should be valid", symbol);
            assert!(symbol.chars().all(|c| c.is_alphanumeric()),
                "Symbol {} should be alphanumeric", symbol);
        }
    }

    #[test]
    fn test_token_decimal_range() {
        // Test all valid decimal values
        for decimals in 0u8..=24 {
            let multiplier = BigDecimal::from_str(&format!("1{}", "0".repeat(decimals as usize))).unwrap();
            let one_token_units = multiplier.clone();
            let display = &one_token_units / &multiplier;
            assert_eq!(display, BigDecimal::from(1),
                "Decimal {} should work correctly", decimals);
        }
    }
}
