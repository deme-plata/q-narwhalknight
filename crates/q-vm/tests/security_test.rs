/// Security Tests for VM Network Bridge
///
/// Tests all critical security features:
/// - Message signing and verification
/// - Rate limiting
/// - Resource quotas
/// - Bytecode validation
/// - Access control
/// - Replay attack prevention

use q_vm::network::{
    VmNetworkBridge, VmNetworkConfig, VmNetworkMessage,
    SignedVmMessage, PeerRateLimiter, ResourceQuotaManager, BytecodeValidator,
    AccessController, NonceTracker,
};
use q_vm::state::StateDB;
use std::sync::Arc;
use ed25519_dalek::SigningKey;
use rand::thread_rng;

#[tokio::test]
async fn test_message_signing_and_verification() {
    // Create signing keys
    let mut rng = thread_rng();
    let signing_key = SigningKey::generate(&mut rng);

    // Create a test message
    let message = VmNetworkMessage::VmCapabilities {
        vm_version: "0.1.0".to_string(),
        supported_features: vec!["wasm".to_string()],
        max_gas_limit: 15_000_000,
        tps_capacity: 150_000,
    };

    // Sign the message
    let signed_msg = SignedVmMessage::sign(message.clone(), &signing_key)
        .expect("Failed to sign message");

    // Verify the signature
    assert!(signed_msg.verify().is_ok(), "Signature verification should succeed");

    // Test that verification fails with tampered message
    let mut tampered = signed_msg.clone();
    tampered.timestamp += 1000;
    assert!(tampered.verify().is_err(), "Tampered message verification should fail");
}

#[tokio::test]
async fn test_rate_limiting() {
    let rate_limiter = Arc::new(PeerRateLimiter::new(2)); // 2 requests per second
    let peer_key = [1u8; 32];

    // First request should succeed
    assert!(rate_limiter.check_rate_limit(&peer_key).await.is_ok());

    // Second request should succeed
    assert!(rate_limiter.check_rate_limit(&peer_key).await.is_ok());

    // Third request should fail (rate limit exceeded)
    assert!(rate_limiter.check_rate_limit(&peer_key).await.is_err());

    // Wait for rate limit to reset
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Request should succeed again
    assert!(rate_limiter.check_rate_limit(&peer_key).await.is_ok());
}

#[tokio::test]
async fn test_resource_quota_management() {
    let quota_manager = Arc::new(ResourceQuotaManager::new(
        100_000, // total gas pool
        50_000,  // max per request
    ));

    // Acquire gas quota
    let permit1 = quota_manager.acquire_gas_quota(30_000).await;
    assert!(permit1.is_ok(), "First gas acquisition should succeed");

    let permit2 = quota_manager.acquire_gas_quota(40_000).await;
    assert!(permit2.is_ok(), "Second gas acquisition should succeed");

    // This should fail (would exceed total pool)
    let permit3 = quota_manager.acquire_gas_quota(50_000).await;
    assert!(permit3.is_err(), "Third gas acquisition should fail (pool exhausted)");

    // Drop permits to release gas
    drop(permit1);
    drop(permit2);

    // Wait briefly for semaphore to update
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    // Now it should succeed again
    let permit4 = quota_manager.acquire_gas_quota(50_000).await;
    assert!(permit4.is_ok(), "Gas acquisition should succeed after release");
}

#[tokio::test]
async fn test_resource_quota_per_request_limit() {
    let quota_manager = Arc::new(ResourceQuotaManager::new(
        1_000_000, // large total pool
        50_000,    // max per request
    ));

    // Request exceeding per-request limit should fail
    let permit = quota_manager.acquire_gas_quota(60_000).await;
    assert!(permit.is_err(), "Request exceeding per-request limit should fail");
}

#[tokio::test]
async fn test_bytecode_validation_size_limit() {
    let validator = Arc::new(BytecodeValidator::new(1024)); // 1 KB limit

    // Valid small bytecode
    let small_bytecode = vec![0u8; 512];
    assert!(validator.validate(&small_bytecode).is_ok());

    // Oversized bytecode
    let large_bytecode = vec![0u8; 2048];
    assert!(validator.validate(&large_bytecode).is_err());
}

#[tokio::test]
async fn test_bytecode_validation_wasm_format() {
    let validator = Arc::new(BytecodeValidator::new(5 * 1024 * 1024));

    // Valid WASM module (minimal)
    let valid_wasm = wat::parse_str(r#"
        (module
            (func (export "test") (result i32)
                i32.const 42
            )
        )
    "#).expect("Failed to parse WAT");

    assert!(validator.validate(&valid_wasm).is_ok(), "Valid WASM should pass validation");

    // Invalid bytecode
    let invalid = vec![0xFF, 0xFF, 0xFF, 0xFF];
    assert!(validator.validate(&invalid).is_err(), "Invalid bytecode should fail validation");
}

#[tokio::test]
async fn test_access_control_authorization() {
    let controller = Arc::new(AccessController::new());
    let peer_key = [1u8; 32];

    // Initially, peer should not be authorized
    assert!(!controller.is_peer_authorized(&peer_key).await);

    // Add peer to authorized list
    controller.add_authorized_peer(peer_key).await;

    // Now peer should be authorized
    assert!(controller.is_peer_authorized(&peer_key).await);

    // Remove authorization
    controller.remove_authorized_peer(&peer_key).await;

    // Peer should no longer be authorized
    assert!(!controller.is_peer_authorized(&peer_key).await);
}

#[tokio::test]
async fn test_access_control_ban() {
    let controller = Arc::new(AccessController::new());
    let peer_key = [1u8; 32];

    // Add peer to authorized list
    controller.add_authorized_peer(peer_key).await;
    assert!(controller.is_peer_authorized(&peer_key).await);

    // Ban the peer
    controller.ban_peer(peer_key).await;

    // Banned peer should not be authorized even if in authorized list
    assert!(!controller.is_peer_authorized(&peer_key).await);
}

#[tokio::test]
async fn test_contract_access_control() {
    let controller = Arc::new(AccessController::new());
    let peer_key = [1u8; 32];
    let contract = "0xcontract123".to_string();

    // Initially, peer should not have contract access
    assert!(!controller.check_contract_access(&peer_key, &contract).await);

    // Grant contract access
    controller.grant_contract_access(peer_key, contract.clone()).await;

    // Now peer should have access
    assert!(controller.check_contract_access(&peer_key, &contract).await);

    // Revoke access
    controller.revoke_contract_access(&peer_key, &contract).await;

    // Peer should no longer have access
    assert!(!controller.check_contract_access(&peer_key, &contract).await);
}

#[tokio::test]
async fn test_nonce_replay_protection() {
    let nonce_tracker = Arc::new(NonceTracker::new());
    let peer_key = [1u8; 32];
    let nonce1 = 12345u64;
    let nonce2 = 67890u64;

    // First use of nonce should succeed
    assert!(nonce_tracker.check_and_mark_nonce(&peer_key, nonce1).await.is_ok());

    // Reusing same nonce should fail (replay attack)
    assert!(nonce_tracker.check_and_mark_nonce(&peer_key, nonce1).await.is_err());

    // Different nonce should succeed
    assert!(nonce_tracker.check_and_mark_nonce(&peer_key, nonce2).await.is_ok());

    // Reusing second nonce should fail
    assert!(nonce_tracker.check_and_mark_nonce(&peer_key, nonce2).await.is_err());
}

#[tokio::test]
async fn test_vm_network_bridge_security_initialization() {
    let state_db = Arc::new(StateDB::new());
    let config = VmNetworkConfig::default();

    let bridge = VmNetworkBridge::new(config, state_db).await;
    assert!(bridge.is_ok(), "Bridge initialization should succeed");

    let bridge = bridge.unwrap();

    // Bridge should have a valid public key
    let public_key = bridge.get_public_key();
    assert_ne!(public_key, [0u8; 32], "Public key should not be all zeros");
}

#[tokio::test]
async fn test_vm_network_bridge_peer_management() {
    let state_db = Arc::new(StateDB::new());
    let config = VmNetworkConfig::default();

    let bridge = VmNetworkBridge::new(config, state_db).await.unwrap();

    let peer_key = [1u8; 32];
    let contract = "0xcontract456".to_string();

    // Add authorized peer
    assert!(bridge.add_authorized_peer(peer_key).await.is_ok());

    // Grant contract access
    assert!(bridge.grant_contract_access(peer_key, contract.clone()).await.is_ok());

    // Revoke contract access
    assert!(bridge.revoke_contract_access(&peer_key, &contract).await.is_ok());

    // Ban peer
    assert!(bridge.ban_peer(peer_key).await.is_ok());
}

#[tokio::test]
async fn test_message_size_limit() {
    let state_db = Arc::new(StateDB::new());
    let mut config = VmNetworkConfig::default();
    config.max_message_size = 1024; // 1 KB limit

    let bridge = VmNetworkBridge::new(config, state_db).await.unwrap();

    // Create a large deployment message
    let large_bytecode = vec![0u8; 2048]; // 2 KB
    let result = bridge.deploy_contract_to_network(large_bytecode, "deployer".to_string()).await;

    // Should fail due to size limit
    assert!(result.is_err(), "Oversized message should be rejected");
}

#[tokio::test]
async fn test_quota_statistics() {
    let quota_manager = Arc::new(ResourceQuotaManager::new(100_000, 50_000));

    // Acquire some gas
    let _permit = quota_manager.acquire_gas_quota(30_000).await.unwrap();

    // Check stats
    let stats = quota_manager.get_stats().await;
    assert_eq!(stats.total_gas_pool, 100_000);
    assert_eq!(stats.available_gas, 70_000);
    assert_eq!(stats.max_gas_per_request, 50_000);
}

#[tokio::test]
async fn test_integrated_security_flow() {
    // This test simulates a complete secure message flow:
    // 1. Create signed message
    // 2. Verify signature
    // 3. Check rate limit
    // 4. Check authorization
    // 5. Acquire gas quota
    // 6. Process message

    let mut rng = thread_rng();
    let signing_key = SigningKey::generate(&mut rng);
    let public_key = signing_key.verifying_key().to_bytes();

    let rate_limiter = Arc::new(PeerRateLimiter::new(10));
    let quota_manager = Arc::new(ResourceQuotaManager::new(150_000_000, 15_000_000));
    let access_controller = Arc::new(AccessController::new());
    let nonce_tracker = Arc::new(NonceTracker::new());

    // Setup: Authorize the peer
    access_controller.add_authorized_peer(public_key).await;

    // Step 1: Create and sign message
    let message = VmNetworkMessage::ContractExecutionRequest {
        contract_address: "0xcontract".to_string(),
        function: "transfer".to_string(),
        args: vec![1, 2, 3, 4],
        caller: "0xcaller".to_string(),
        gas_limit: 1_000_000,
        request_id: "req-123".to_string(),
    };

    let signed_msg = SignedVmMessage::sign(message, &signing_key).unwrap();

    // Step 2: Verify signature
    assert!(signed_msg.verify().is_ok());

    // Step 3: Check nonce (replay protection)
    assert!(nonce_tracker.check_and_mark_nonce(&public_key, signed_msg.nonce).await.is_ok());

    // Step 4: Check rate limit
    assert!(rate_limiter.check_rate_limit(&public_key).await.is_ok());

    // Step 5: Check authorization
    assert!(access_controller.is_peer_authorized(&public_key).await);

    // Step 6: Acquire gas quota
    let _permit = quota_manager.acquire_gas_quota(1_000_000).await;
    assert!(_permit.is_ok());

    // All security checks passed!
}

#[tokio::test]
async fn test_security_rejection_scenarios() {
    let mut rng = thread_rng();
    let signing_key = SigningKey::generate(&mut rng);
    let public_key = signing_key.verifying_key().to_bytes();

    let access_controller = Arc::new(AccessController::new());
    let nonce_tracker = Arc::new(NonceTracker::new());

    // Scenario 1: Unauthorized peer
    assert!(!access_controller.is_peer_authorized(&public_key).await);

    // Scenario 2: Replay attack
    let nonce = 12345u64;
    nonce_tracker.check_and_mark_nonce(&public_key, nonce).await.unwrap();
    assert!(nonce_tracker.check_and_mark_nonce(&public_key, nonce).await.is_err());

    // Scenario 3: Banned peer
    access_controller.add_authorized_peer(public_key).await;
    assert!(access_controller.is_peer_authorized(&public_key).await);

    access_controller.ban_peer(public_key).await;
    assert!(!access_controller.is_peer_authorized(&public_key).await);
}
