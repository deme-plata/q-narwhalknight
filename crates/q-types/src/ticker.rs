/// Ticker and branding constants for Quillon Graph
///
/// Official Name: Quillon Graph
/// Codename: Q-NarwhalKnight (internal use only)
/// Ticker: QUG
///
/// This module provides ticker-related constants with backward compatibility
/// for the legacy QNK ticker during the migration period.

/// Official ticker symbol displayed to users
pub const TICKER_SYMBOL: &str = "QUG";

/// Legacy ticker symbol (deprecated, maintained for backward compatibility)
pub const LEGACY_TICKER: &str = "QNK";

/// Official project display name
pub const DISPLAY_NAME: &str = "Quillon Graph";

/// Internal codename (used in comments and development)
pub const CODENAME: &str = "Q-NarwhalKnight";

/// Address prefix for new addresses
pub const ADDRESS_PREFIX: &str = "qug";

/// Legacy address prefixes (still accepted for compatibility)
pub const LEGACY_ADDRESS_PREFIXES: &[&str] = &["qnk"];

/// Satoshis per coin (100 million satoshis = 1 QUG)
pub const SATOSHIS_PER_COIN: u64 = 100_000_000;

/// Check if an address prefix is valid (accepts both new and legacy)
pub fn is_valid_address_prefix(prefix: &str) -> bool {
    prefix == ADDRESS_PREFIX || LEGACY_ADDRESS_PREFIXES.contains(&prefix)
}

/// Normalize address by removing any valid prefix
pub fn normalize_address(address: &str) -> String {
    let address_lower = address.to_lowercase();

    // Try new prefix first
    if let Some(stripped) = address_lower.strip_prefix(ADDRESS_PREFIX) {
        return stripped.to_string();
    }

    // Try legacy prefixes
    for legacy_prefix in LEGACY_ADDRESS_PREFIXES {
        if let Some(stripped) = address_lower.strip_prefix(legacy_prefix) {
            return stripped.to_string();
        }
    }

    // No prefix found, return as-is
    address.to_string()
}

/// Add the official prefix to a normalized address
pub fn add_address_prefix(normalized_address: &str) -> String {
    format!("{}{}", ADDRESS_PREFIX, normalized_address)
}

/// Convert satoshis to decimal amount with ticker symbol
pub fn format_balance(satoshis: u64) -> String {
    let coins = satoshis as f64 / SATOSHIS_PER_COIN as f64;
    format!("{:.8} {}", coins, TICKER_SYMBOL)
}

/// Convert satoshis to decimal amount (without ticker symbol)
pub fn satoshis_to_coins(satoshis: u64) -> f64 {
    satoshis as f64 / SATOSHIS_PER_COIN as f64
}

/// Convert decimal amount to satoshis
pub fn coins_to_satoshis(coins: f64) -> u64 {
    (coins * SATOSHIS_PER_COIN as f64) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_address() {
        assert_eq!(normalize_address("qug1234abcd"), "1234abcd");
        assert_eq!(normalize_address("qnk1234abcd"), "1234abcd");
        assert_eq!(normalize_address("1234abcd"), "1234abcd");
    }

    #[test]
    fn test_is_valid_prefix() {
        assert!(is_valid_address_prefix("qug"));
        assert!(is_valid_address_prefix("qnk"));
        assert!(!is_valid_address_prefix("btc"));
    }

    #[test]
    fn test_format_balance() {
        assert_eq!(format_balance(100_000_000), "1.00000000 QUG");
        assert_eq!(format_balance(50_000_000), "0.50000000 QUG");
    }

    #[test]
    fn test_satoshis_conversion() {
        assert_eq!(satoshis_to_coins(100_000_000), 1.0);
        assert_eq!(coins_to_satoshis(1.0), 100_000_000);
    }
}
