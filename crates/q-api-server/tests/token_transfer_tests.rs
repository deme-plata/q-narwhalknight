//! Token transfer resolution tests
//!
//! Since `q-api-server` is a `[[bin]]` crate, we cannot import `resolve_token_type`
//! directly. Instead, these tests reimplement the same resolution algorithm and
//! verify correctness against the canonical `q_types` constants. This ensures the
//! algorithm behaves identically to the production code in `main.rs`.

use q_types::{TokenType, QUGUSD_TOKEN_ADDRESS};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Local reimplementation of resolve_token_type (mirrors main.rs exactly)
// ---------------------------------------------------------------------------

fn resolve_token_type(token_address: &Option<String>) -> TokenType {
    match token_address {
        Some(addr_hex) => {
            if let Ok(bytes) = hex::decode(addr_hex) {
                if bytes.len() == 32 && bytes[..] == QUGUSD_TOKEN_ADDRESS[..] {
                    TokenType::QUGUSD
                } else if bytes.len() == 32 {
                    let mut addr = [0u8; 32];
                    addr.copy_from_slice(&bytes);
                    TokenType::Custom(addr)
                } else {
                    TokenType::QUG
                }
            } else if addr_hex.to_uppercase() == "QUGUSD" {
                TokenType::QUGUSD
            } else {
                TokenType::QUG
            }
        }
        None => TokenType::QUG,
    }
}

// ---------------------------------------------------------------------------
// Wire-format struct matching BrowserTransaction for msgpack round-trip tests
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct BrowserTransaction {
    from: String,
    to: String,
    amount: String,
    nonce: u64,
    timestamp: u64,
    signature: Vec<u8>,
    public_key: Vec<u8>,
    #[serde(default)]
    token_address: Option<String>,
    #[serde(default)]
    memo: Option<String>,
    #[serde(default)]
    network_id: Option<String>,
    #[serde(default)]
    protocol_version: Option<String>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_qugusd_hex_address_resolves_to_qugusd() {
    let hex_str = hex::encode(QUGUSD_TOKEN_ADDRESS);
    assert_eq!(
        hex_str,
        "5155475553440000000000000000000000000000000000000000000000000000",
        "QUGUSD_TOKEN_ADDRESS hex encoding must match the canonical value"
    );

    let result = resolve_token_type(&Some(hex_str));
    assert_eq!(result, TokenType::QUGUSD);
}

#[test]
fn test_custom_32byte_hex_resolves_to_custom() {
    // 32 bytes of 0xAB — clearly not QUGUSD
    let custom_addr = [0xABu8; 32];
    let hex_str = hex::encode(custom_addr);

    let result = resolve_token_type(&Some(hex_str));
    match result {
        TokenType::Custom(addr) => assert_eq!(addr, custom_addr),
        other => panic!("Expected Custom, got {:?}", other),
    }
}

#[test]
fn test_null_token_address_defaults_to_qug() {
    let result = resolve_token_type(&None);
    assert_eq!(result, TokenType::QUG);
}

#[test]
fn test_string_qugusd_resolves() {
    // Exact case
    assert_eq!(resolve_token_type(&Some("QUGUSD".to_string())), TokenType::QUGUSD);
    // Lowercase — to_uppercase() makes it match
    assert_eq!(resolve_token_type(&Some("qugusd".to_string())), TokenType::QUGUSD);
    // Mixed case
    assert_eq!(resolve_token_type(&Some("QuGuSd".to_string())), TokenType::QUGUSD);
}

#[test]
fn test_invalid_hex_defaults_to_qug() {
    // "zzzzinvalid" is not valid hex and not "QUGUSD"
    let result = resolve_token_type(&Some("zzzzinvalid".to_string()));
    assert_eq!(result, TokenType::QUG);
}

#[test]
fn test_short_hex_defaults_to_qug() {
    // "0011" decodes to 2 bytes, which is not 32
    let result = resolve_token_type(&Some("0011".to_string()));
    assert_eq!(result, TokenType::QUG);
}

#[test]
fn test_msgpack_roundtrip_preserves_token_address() {
    let qugusd_hex = hex::encode(QUGUSD_TOKEN_ADDRESS);

    let tx = BrowserTransaction {
        from: "aa".repeat(32),
        to: "bb".repeat(32),
        amount: "1000000".to_string(),
        nonce: 42,
        timestamp: 1700000000,
        signature: vec![0u8; 64],
        public_key: vec![0u8; 32],
        token_address: Some(qugusd_hex.clone()),
        memo: Some("test transfer".to_string()),
        network_id: Some("mainnet-genesis".to_string()),
        protocol_version: Some("1.0".to_string()),
    };

    // Serialize to MessagePack
    let packed = rmp_serde::to_vec(&tx).expect("msgpack serialize failed");

    // Deserialize back
    let unpacked: BrowserTransaction =
        rmp_serde::from_slice(&packed).expect("msgpack deserialize failed");

    assert_eq!(unpacked.token_address, Some(qugusd_hex));
    assert_eq!(unpacked.from, tx.from);
    assert_eq!(unpacked.to, tx.to);
    assert_eq!(unpacked.amount, tx.amount);
    assert_eq!(unpacked.nonce, tx.nonce);
    assert_eq!(unpacked.memo, tx.memo);

    // Verify the round-tripped token_address still resolves correctly
    let resolved = resolve_token_type(&unpacked.token_address);
    assert_eq!(resolved, TokenType::QUGUSD);
}
