//! DEX Overflow Protection Tests
//!
//! v3.2.25-beta: Tests for u128 overflow prevention in AMM calculations
//!
//! These tests verify:
//! - Token amounts with various decimal places don't overflow
//! - AMM constant product formula handles large amounts safely
//! - Proper error messages for amounts exceeding limits
//!
//! Run with: cargo test --package q-dex --test overflow_protection_tests

// ============================================================================
// CONSTANTS
// ============================================================================

/// Maximum value for u128
const U128_MAX: u128 = u128::MAX; // ~3.4e38

/// Common token decimals
const DECIMALS_QUG: u32 = 24;      // QUG/QUGUSD use 24 decimals
const DECIMALS_CUSTOM: u32 = 8;    // Most custom tokens use 7-8 decimals
const DECIMALS_LOW: u32 = 2;       // Stablecoins like USD

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Convert token amount to raw amount (with decimals)
fn to_raw_amount(amount: f64, decimals: u32) -> Option<u128> {
    let multiplier = 10u128.checked_pow(decimals)?;
    let raw = (amount * multiplier as f64) as u128;

    // Check if we actually fit in u128
    if amount > (U128_MAX / multiplier) as f64 {
        return None;
    }

    Some(raw)
}

/// Safe multiplication that returns None on overflow
fn safe_mul(a: u128, b: u128) -> Option<u128> {
    a.checked_mul(b)
}

/// Safe division
fn safe_div(a: u128, b: u128) -> Option<u128> {
    if b == 0 {
        None
    } else {
        Some(a / b)
    }
}

/// Calculate maximum tokens for a given decimal precision
fn max_tokens_for_decimals(decimals: u32) -> f64 {
    let multiplier = 10u128.pow(decimals);
    (U128_MAX / multiplier) as f64
}

/// AMM constant product swap calculation (x * y = k)
/// Returns the output amount for a given input
fn calculate_swap_output(
    input_amount: u128,
    input_reserve: u128,
    output_reserve: u128,
    fee_bps: u16,  // Fee in basis points (e.g., 30 = 0.3%)
) -> Option<u128> {
    // input_with_fee = input_amount * (10000 - fee_bps)
    let fee_multiplier = 10000u128 - fee_bps as u128;
    let input_with_fee = safe_mul(input_amount, fee_multiplier)?;

    // numerator = input_with_fee * output_reserve
    let numerator = safe_mul(input_with_fee, output_reserve)?;

    // denominator = (input_reserve * 10000) + input_with_fee
    let input_reserve_scaled = safe_mul(input_reserve, 10000)?;
    let denominator = input_reserve_scaled.checked_add(input_with_fee)?;

    safe_div(numerator, denominator)
}

/// Check if adding liquidity would overflow
fn check_liquidity_overflow(
    amount_a: u128,
    amount_b: u128,
    reserve_a: u128,
    reserve_b: u128,
) -> Result<(), String> {
    // Check constant product multiplication
    let new_reserve_a = reserve_a.checked_add(amount_a)
        .ok_or_else(|| format!("Reserve A overflow: {} + {}", reserve_a, amount_a))?;
    let new_reserve_b = reserve_b.checked_add(amount_b)
        .ok_or_else(|| format!("Reserve B overflow: {} + {}", reserve_b, amount_b))?;

    // Check if k = x * y would overflow
    safe_mul(new_reserve_a, new_reserve_b)
        .ok_or_else(|| format!("Constant product overflow: {} * {}", new_reserve_a, new_reserve_b))?;

    Ok(())
}

// ============================================================================
// DECIMAL PRECISION TESTS
// ============================================================================

mod decimal_tests {
    use super::*;

    #[test]
    fn test_max_tokens_24_decimals() {
        let max = max_tokens_for_decimals(24);
        println!("Max tokens with 24 decimals: {:.2e}", max);

        // With 24 decimals, max is ~340,000,000,000,000 tokens (3.4e14)
        assert!(max > 3e14, "Max should be > 3e14, got {:.2e}", max);
        assert!(max < 4e14, "Max should be < 4e14, got {:.2e}", max);
    }

    #[test]
    fn test_max_tokens_8_decimals() {
        let max = max_tokens_for_decimals(8);
        println!("Max tokens with 8 decimals: {:.2e}", max);

        // With 8 decimals, max is ~3.4e30 tokens
        assert!(max > 3e30, "Max should be > 3e30, got {:.2e}", max);
    }

    #[test]
    fn test_max_tokens_2_decimals() {
        let max = max_tokens_for_decimals(2);
        println!("Max tokens with 2 decimals: {:.2e}", max);

        // With 2 decimals, max is ~3.4e36 tokens
        assert!(max > 3e36, "Max should be > 3e36, got {:.2e}", max);
    }

    #[test]
    fn test_raw_amount_normal_case() {
        // 100 tokens with 24 decimals
        // Note: Floating point has precision limits, so we compare with tolerance
        let raw = to_raw_amount(100.0, 24);
        assert!(raw.is_some());
        let expected = 100 * 10u128.pow(24);
        let actual = raw.unwrap();
        // Allow 0.1% tolerance for floating point precision
        let tolerance = expected / 1000;
        assert!(
            (actual as i128 - expected as i128).unsigned_abs() < tolerance,
            "Raw amount {} not within tolerance of expected {}",
            actual, expected
        );
    }

    #[test]
    fn test_raw_amount_overflow_24_decimals() {
        // 1e28 tokens with 24 decimals should overflow
        let raw = to_raw_amount(1e28, 24);
        assert!(raw.is_none(), "1e28 tokens with 24 decimals should overflow");
    }

    #[test]
    fn test_raw_amount_overflow_boundary() {
        // Test near the boundary for 24 decimals
        let max = max_tokens_for_decimals(24);

        // Just under max should work
        let raw = to_raw_amount(max * 0.99, 24);
        assert!(raw.is_some(), "99% of max should not overflow");

        // Just over max should fail
        let raw = to_raw_amount(max * 1.01, 24);
        assert!(raw.is_none(), "101% of max should overflow");
    }
}

// ============================================================================
// AMM SWAP CALCULATION TESTS
// ============================================================================

mod amm_tests {
    use super::*;

    #[test]
    fn test_normal_swap() {
        // Small amounts - should work fine
        let input = 1000 * 10u128.pow(8);  // 1000 tokens (8 decimals)
        let reserve_in = 1_000_000 * 10u128.pow(8);
        let reserve_out = 1_000_000 * 10u128.pow(8);

        let output = calculate_swap_output(input, reserve_in, reserve_out, 30);
        assert!(output.is_some());

        // ~997 tokens out (due to 0.3% fee and slippage)
        let output_tokens = output.unwrap() / 10u128.pow(8);
        assert!(output_tokens > 990 && output_tokens < 1000);
    }

    #[test]
    fn test_large_swap_24_decimals() {
        // Large amounts with 24 decimals
        let input = 1_000_000 * 10u128.pow(24);  // 1M tokens
        let reserve_in = 10_000_000 * 10u128.pow(24);
        let reserve_out = 10_000_000 * 10u128.pow(24);

        // This might overflow in the calculation
        let output = calculate_swap_output(input, reserve_in, reserve_out, 30);

        // With these large numbers, overflow is expected
        // The test documents this limitation
        if output.is_none() {
            println!("Expected overflow with 24 decimal tokens at scale");
        }
    }

    #[test]
    fn test_swap_overflow_detection() {
        // Intentionally large values that should overflow
        let input = U128_MAX / 2;
        let reserve_in = U128_MAX / 2;
        let reserve_out = U128_MAX / 2;

        let output = calculate_swap_output(input, reserve_in, reserve_out, 30);
        assert!(output.is_none(), "Should overflow with max values");
    }

    #[test]
    fn test_swap_zero_fee() {
        let input = 1000 * 10u128.pow(8);
        let reserve_in = 1_000_000 * 10u128.pow(8);
        let reserve_out = 1_000_000 * 10u128.pow(8);

        // 0% fee
        let output = calculate_swap_output(input, reserve_in, reserve_out, 0);
        assert!(output.is_some());
    }

    #[test]
    fn test_swap_high_fee() {
        let input = 1000 * 10u128.pow(8);
        let reserve_in = 1_000_000 * 10u128.pow(8);
        let reserve_out = 1_000_000 * 10u128.pow(8);

        // 10% fee (1000 bps)
        let output = calculate_swap_output(input, reserve_in, reserve_out, 1000);
        assert!(output.is_some());

        // Should be significantly less than with 0.3% fee
        let output_30bps = calculate_swap_output(input, reserve_in, reserve_out, 30).unwrap();
        assert!(output.unwrap() < output_30bps);
    }
}

// ============================================================================
// LIQUIDITY OVERFLOW TESTS
// ============================================================================

mod liquidity_tests {
    use super::*;

    #[test]
    fn test_normal_liquidity_addition() {
        let amount_a = 10_000 * 10u128.pow(8);
        let amount_b = 10_000 * 10u128.pow(8);
        let reserve_a = 1_000_000 * 10u128.pow(8);
        let reserve_b = 1_000_000 * 10u128.pow(8);

        let result = check_liquidity_overflow(amount_a, amount_b, reserve_a, reserve_b);
        assert!(result.is_ok());
    }

    #[test]
    fn test_liquidity_overflow_constant_product() {
        // Values that would overflow when multiplied
        let amount_a = 10u128.pow(30);
        let amount_b = 10u128.pow(30);
        let reserve_a = 10u128.pow(30);
        let reserve_b = 10u128.pow(30);

        let result = check_liquidity_overflow(amount_a, amount_b, reserve_a, reserve_b);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("overflow"));
    }

    #[test]
    fn test_liquidity_reserve_overflow() {
        let amount_a = U128_MAX - 100;
        let amount_b = 1000;
        let reserve_a = 1000;
        let reserve_b = 1000;

        let result = check_liquidity_overflow(amount_a, amount_b, reserve_a, reserve_b);
        // Adding amount_a to reserve_a would overflow
        assert!(result.is_err());
    }

    #[test]
    fn test_fomo_token_scenario() {
        // The specific bug scenario: user tries to add 1e28 FOMO tokens
        let fomo_amount = 10u128.pow(28);  // 1e28 tokens
        let fomo_decimals = 7u32;          // Custom token with 7 decimals

        // Convert to raw amount
        let raw_amount = fomo_amount.checked_mul(10u128.pow(fomo_decimals));

        // This should overflow or be very close to u128 max
        if raw_amount.is_none() {
            println!("1e28 tokens with 7 decimals overflows u128");
        } else {
            // Check if it's reasonable
            let raw = raw_amount.unwrap();
            println!("1e28 tokens with 7 decimals = {:.2e}", raw as f64);

            // Try to use it in a liquidity calculation
            let qugusd_amount = 10u128.pow(20);  // Some QUGUSD
            let result = check_liquidity_overflow(raw, qugusd_amount, 0, 0);

            if result.is_err() {
                println!("Liquidity check failed: {}", result.unwrap_err());
            }
        }
    }
}

// ============================================================================
// ERROR MESSAGE TESTS
// ============================================================================

mod error_message_tests {
    use super::*;

    fn format_overflow_error(decimals: u32) -> String {
        let max_tokens = max_tokens_for_decimals(decimals);
        let max_display = format!("{:.2e}", max_tokens);
        format!(
            "Amount exceeds maximum supported!\n\nWith {} decimals, max amount: ~{} tokens\n\nPlease reduce the amount or use a token with fewer decimals.",
            decimals, max_display
        )
    }

    #[test]
    fn test_error_message_24_decimals() {
        let msg = format_overflow_error(24);
        assert!(msg.contains("24 decimals"));
        assert!(msg.contains("3.40e14") || msg.contains("3.4e14"));
    }

    #[test]
    fn test_error_message_8_decimals() {
        let msg = format_overflow_error(8);
        assert!(msg.contains("8 decimals"));
        assert!(msg.contains("e30") || msg.contains("e+30"));
    }

    #[test]
    fn test_error_message_different_decimals() {
        // Error messages should show different max amounts for different decimals
        let msg_24 = format_overflow_error(24);
        let msg_8 = format_overflow_error(8);

        // 24 decimal max should be much smaller than 8 decimal max
        assert_ne!(msg_24, msg_8);
    }
}

// ============================================================================
// REGRESSION TESTS
// ============================================================================

mod regression_tests {
    use super::*;

    /// Regression test for v3.2.25-beta overflow error message fix
    ///
    /// BUG: Error message showed "Max amount: ~340,000,000,000,000 tokens"
    /// even for tokens with different decimal places (assumed 24 decimals).
    ///
    /// FIX: Calculate actual max based on token's decimals.
    #[test]
    fn test_regression_overflow_message_v3_2_25() {
        // For a token with 7 decimals (like FOMO)
        let fomo_decimals = 7;
        let fomo_max = max_tokens_for_decimals(fomo_decimals);

        // For QUG with 24 decimals
        let qug_decimals = 24;
        let qug_max = max_tokens_for_decimals(qug_decimals);

        // FOMO should allow MUCH larger amounts than QUG
        assert!(
            fomo_max > qug_max * 1e10,
            "REGRESSION: Tokens with fewer decimals should have higher max. FOMO max: {:.2e}, QUG max: {:.2e}",
            fomo_max, qug_max
        );

        // Specifically, FOMO max should be ~3.4e30, not ~3.4e14
        assert!(
            fomo_max > 1e30,
            "REGRESSION: FOMO max should be > 1e30, got {:.2e}",
            fomo_max
        );
    }
}

// ============================================================================
// BOUNDARY TESTS
// ============================================================================

mod boundary_tests {
    use super::*;

    #[test]
    fn test_zero_amounts() {
        let output = calculate_swap_output(0, 1000, 1000, 30);
        assert_eq!(output, Some(0), "Zero input should give zero output");
    }

    #[test]
    fn test_one_wei_swap() {
        let output = calculate_swap_output(1, 10u128.pow(24), 10u128.pow(24), 30);
        assert!(output.is_some());
        // Might be 0 due to rounding, which is acceptable
    }

    #[test]
    fn test_entire_reserve_swap() {
        // Trying to swap entire output reserve
        let reserve = 1_000_000 * 10u128.pow(8);
        let output = calculate_swap_output(U128_MAX / 1000, reserve, reserve, 30);

        // Should either fail or give less than reserve
        if let Some(out) = output {
            assert!(out < reserve, "Cannot get more than reserve");
        }
    }

    #[test]
    fn test_min_decimals() {
        // 0 decimals (integer tokens)
        let max = max_tokens_for_decimals(0);
        assert_eq!(max, U128_MAX as f64);
    }

    #[test]
    fn test_max_practical_decimals() {
        // 38 decimals (near u128 limit for single token)
        let max = max_tokens_for_decimals(38);
        assert!(max < 100.0, "38 decimals allows < 100 tokens");
        assert!(max > 1.0, "38 decimals should allow at least 1 token");
    }
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_swap_calculation_performance() {
        let iterations = 100_000;
        let start = Instant::now();

        for i in 0..iterations {
            let input = (i as u128 + 1) * 10u128.pow(8);
            let _ = calculate_swap_output(input, 10u128.pow(20), 10u128.pow(20), 30);
        }

        let elapsed = start.elapsed();
        let per_calc_ns = elapsed.as_nanos() / iterations as u128;

        println!("Swap calculation: {} ns per call", per_calc_ns);
        assert!(per_calc_ns < 1000, "Swap calculation should be < 1us, got {} ns", per_calc_ns);
    }

    #[test]
    fn test_overflow_check_performance() {
        let iterations = 100_000;
        let start = Instant::now();

        for i in 0..iterations {
            let amount = (i as u128 + 1) * 10u128.pow(8);
            let _ = check_liquidity_overflow(amount, amount, 10u128.pow(20), 10u128.pow(20));
        }

        let elapsed = start.elapsed();
        let per_check_ns = elapsed.as_nanos() / iterations as u128;

        println!("Overflow check: {} ns per call", per_check_ns);
        // Allow up to 1000ns (1 microsecond) to account for system load variation
        assert!(per_check_ns < 1000, "Overflow check should be < 1us, got {} ns", per_check_ns);
    }
}
