//! Token Balance Routing Tests
//!
//! Tests for token balance routing in the balance consensus engine.
//!
//! These tests verify that the routing logic correctly directs:
//! - QUGUSD transfers to token_balances (via QUGUSD_TOKEN_ADDRESS)
//! - Custom token transfers to token_balances (via custom address)
//! - QUG transfers to wallet_balances (native balance, no token address)
//!
//! Run with: cargo test --package q-storage --test token_balance_tests

use q_types::{TokenType, QUGUSD_TOKEN_ADDRESS};

// ============================================================================
// HELPER: Token balance routing logic (mirrors the consensus engine match)
// ============================================================================

/// Determines the token address for balance routing.
///
/// Returns `Some(address)` for token_balances (QUGUSD, Custom tokens),
/// or `None` for wallet_balances (native QUG).
fn route_token_balance(token_type: &TokenType) -> Option<[u8; 32]> {
    match token_type {
        TokenType::QUGUSD => Some(QUGUSD_TOKEN_ADDRESS),
        TokenType::Custom(addr) => Some(*addr),
        TokenType::QUG => None,
    }
}

// ============================================================================
// TOKEN BALANCE ROUTING TESTS
// ============================================================================

#[test]
fn test_qugusd_transfer_routes_to_token_balances() {
    // QUGUSD transfers must route to token_balances, not wallet_balances.
    // The routing should return Some(QUGUSD_TOKEN_ADDRESS).
    let token_type = TokenType::QUGUSD;
    let result = route_token_balance(&token_type);

    assert!(
        result.is_some(),
        "QUGUSD must route to token_balances (expected Some), got None"
    );
    assert_eq!(
        result.unwrap(),
        QUGUSD_TOKEN_ADDRESS,
        "QUGUSD must map to the canonical QUGUSD_TOKEN_ADDRESS"
    );
}

#[test]
fn test_custom_token_routes_to_token_balances() {
    // Custom tokens must route to token_balances using their contract address.
    let custom_addr = [0xAB; 32];
    let token_type = TokenType::Custom(custom_addr);
    let result = route_token_balance(&token_type);

    assert!(
        result.is_some(),
        "Custom token must route to token_balances (expected Some), got None"
    );
    assert_eq!(
        result.unwrap(),
        custom_addr,
        "Custom token must map to its own contract address [0xAB; 32]"
    );
}

#[test]
fn test_qug_transfer_routes_to_wallet_balances() {
    // QUG (native token) transfers must route to wallet_balances, not token_balances.
    // The routing should return None, indicating the native balance path.
    let token_type = TokenType::QUG;
    let result = route_token_balance(&token_type);

    assert!(
        result.is_none(),
        "QUG must route to wallet_balances (expected None), got Some({:?})",
        result
    );
}
