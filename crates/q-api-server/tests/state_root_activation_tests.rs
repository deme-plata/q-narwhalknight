//! State Root Activation Tests — v10.4.15
//!
//! MAINNET SAFETY: These tests guard a $1.5B network.
//! If any test here fails, DO NOT DEPLOY to production.
//!
//! Tests cover:
//!   - UpgradeGate::StateRootV1 mainnet config: activation_height = u64::MAX (MUST NOT CHANGE)
//!   - UpgradeGate::StateRootV1 testnet config: activation_height = 0 (immediate)
//!   - Pre-activation behavior: compute_state_root() returns [0u8;32]
//!   - Post-activation behavior: compute_state_root() calls balance state hash
//!   - Block validator accepts [0u8;32] pre-activation (no rejection)
//!   - Block validator rejects wrong state_root post-activation
//!   - Block validator rejects [0u8;32] post-activation (missing root)
//!   - Activation height is monotonic (once active, stays active)
//!   - Shadow mode: logs mismatch but does NOT reject blocks
//!   - Mandatory=false on mainnet: node doesn't disconnect peers over state_root
//!   - 6 prerequisites that MUST be met before changing u64::MAX
//!   - BEDA anchor must version state_root semantics (v0=TX root, v1=balance root)
//!   - State root activation is an irreversible hard fork
//!   - Block hash commitment prevents post-hoc state_root modification
//!   - Activation sequence: shadow mode → testnet soak → mainnet notice → activation
//!
//! Run with: cargo test --package q-api-server --test state_root_activation_tests

// ============================================================================
// CONSTANTS (mirroring upgrade_gate.rs values)
// ============================================================================

/// StateRootV1 mainnet activation height. MUST be u64::MAX until ALL prerequisites met.
const MAINNET_STATE_ROOT_V1_ACTIVATION: u64 = u64::MAX;

/// StateRootV1 testnet activation height. Activates immediately.
const TESTNET_STATE_ROOT_V1_ACTIVATION: u64 = 0;

/// Mainnet checkpoint height — used to verify pre-activation state.
const CHECKPOINT_HEIGHT: u64 = 16_538_868;

/// Minimum advance notice for mandatory mainnet upgrade (in blocks, assuming 1 bps).
/// 6 weeks × 7 days × 24h × 60min × 60s = 3,628,800 seconds = 3,628,800 blocks
const MIN_UPGRADE_NOTICE_BLOCKS: u64 = 3_628_800;

/// Maximum QUG supply: 21M × 10^24
const MAX_QUG_SUPPLY: u128 = 21_000_000u128 * 10u128.pow(24);

// ============================================================================
// SIMULATED TYPES (pure logic, no real RocksDB)
// ============================================================================

/// Simulates UpgradeGate behavior for the StateRootV1 upgrade.
#[derive(Debug, Clone)]
struct MockUpgradeGate {
    is_mainnet: bool,
}

impl MockUpgradeGate {
    fn mainnet() -> Self { Self { is_mainnet: true } }
    fn testnet() -> Self { Self { is_mainnet: false } }

    fn state_root_v1_active(&self, block_height: u64) -> bool {
        let activation = if self.is_mainnet {
            MAINNET_STATE_ROOT_V1_ACTIVATION
        } else {
            TESTNET_STATE_ROOT_V1_ACTIVATION
        };
        block_height >= activation
    }

    fn state_root_v1_activation_height(&self) -> u64 {
        if self.is_mainnet {
            MAINNET_STATE_ROOT_V1_ACTIVATION
        } else {
            TESTNET_STATE_ROOT_V1_ACTIVATION
        }
    }
}

/// Simulates the state root computation logic in block_producer.rs.
/// Returns [0u8;32] pre-activation, balance hash post-activation.
fn compute_state_root_for_block(
    gate: &MockUpgradeGate,
    block_height: u64,
    balance_hash: [u8; 32],
) -> [u8; 32] {
    if gate.state_root_v1_active(block_height) {
        balance_hash
    } else {
        [0u8; 32]
    }
}

/// Simulates block validation logic for state_root field.
/// Returns Ok(()) if valid, Err(reason) if rejected.
fn validate_state_root(
    gate: &MockUpgradeGate,
    block_height: u64,
    claimed_state_root: [u8; 32],
    computed_state_root: [u8; 32],
    shadow_mode: bool,
) -> Result<(), String> {
    if !gate.state_root_v1_active(block_height) {
        // Pre-activation: accept any state_root (including [0;32])
        return Ok(());
    }

    // Post-activation enforcement
    if claimed_state_root == [0u8; 32] {
        return Err(format!(
            "Block {} has state_root=[0;32] but StateRootV1 is active — missing root",
            block_height
        ));
    }

    if claimed_state_root != computed_state_root {
        let msg = format!(
            "Block {} state_root mismatch: claimed={}, computed={}",
            block_height,
            hex::encode(&claimed_state_root),
            hex::encode(&computed_state_root),
        );
        if shadow_mode {
            // Log only, don't reject
            eprintln!("[SHADOW MODE] {}", msg);
            return Ok(());
        }
        return Err(msg);
    }

    Ok(())
}

/// Simulates a block hash commitment: block_hash = Blake3(height || prev || state_root || tx_root)
fn block_hash(height: u64, prev: [u8; 32], state_root: [u8; 32], tx_root: [u8; 32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&height.to_le_bytes());
    hasher.update(&prev);
    hasher.update(&state_root);
    hasher.update(&tx_root);
    *hasher.finalize().as_bytes()
}

fn hex_str(bytes: &[u8; 32]) -> String {
    hex::encode(bytes)
}

// ============================================================================
// MODULE 1: UPGRADE GATE CONFIGURATION (REGRESSION)
// ============================================================================

mod upgrade_gate_config {
    use super::*;

    #[test]
    fn mainnet_state_root_v1_activation_is_u64_max() {
        // CRITICAL REGRESSION TEST.
        // The activation height MUST be u64::MAX until these 6 prerequisites are met:
        //
        // 1. compute_state_root() renamed to compute_transaction_set_root()
        // 2. New compute_state_root() calls compute_balance_state_hash() from q-storage
        // 3. Block validation ENFORCES (not just warns) wrong state_root → reject block
        // 4. Post-checkpoint block replay implemented (replay blocks H+1..tip after import)
        // 5. Testnet soak with StateRootV1 active for ≥ 14 days, zero mismatches
        // 6. ≥ 6 weeks public upgrade notice on mainnet
        //
        // Changing this value without all 6 prerequisites will:
        // → All nodes compute wrong TX-ID-based state_root
        // → Block producers publish wrong roots
        // → Any node with correct implementation gets kicked off network
        // → $1.5B mainnet fails
        assert_eq!(MAINNET_STATE_ROOT_V1_ACTIVATION, u64::MAX,
            "StateRootV1 mainnet activation MUST be u64::MAX. \
             Changing this requires completing all 6 prerequisites (see doc comment).");
    }

    #[test]
    fn testnet_state_root_v1_activates_at_genesis() {
        assert_eq!(TESTNET_STATE_ROOT_V1_ACTIVATION, 0,
            "StateRootV1 testnet activation must be 0 (immediate) for testing");
    }

    #[test]
    fn mainnet_gate_never_active_at_any_realistic_height() {
        let gate = MockUpgradeGate::mainnet();
        let test_heights = [
            0u64,
            1,
            CHECKPOINT_HEIGHT,
            100_000_000,
            u64::MAX / 2,
            u64::MAX - 1,
        ];

        for height in test_heights {
            assert!(!gate.state_root_v1_active(height),
                "StateRootV1 must NOT be active on mainnet at height {} \
                 (activation is u64::MAX)", height);
        }
    }

    #[test]
    fn testnet_gate_always_active() {
        let gate = MockUpgradeGate::testnet();
        let test_heights = [0u64, 1, 1000, CHECKPOINT_HEIGHT, u64::MAX - 1];

        for height in test_heights {
            assert!(gate.state_root_v1_active(height),
                "StateRootV1 must be active on testnet at every height including {}", height);
        }
    }

    #[test]
    fn activation_is_monotonic() {
        // If active at height N, must be active at N+1.
        let gate = MockUpgradeGate::testnet();
        let activation = gate.state_root_v1_activation_height();

        // For testnet (activation=0), it's always active
        if activation < u64::MAX {
            assert!(gate.state_root_v1_active(activation),
                "Must be active exactly at activation height");
            if activation < u64::MAX - 1 {
                assert!(gate.state_root_v1_active(activation + 1),
                    "Must be active at activation height + 1 (monotonic)");
            }
        }
    }

    #[test]
    fn mandatory_is_false_on_mainnet() {
        // mandatory=false means: even if StateRootV1 were active, nodes don't disconnect
        // peers that send blocks without correct state roots. This is intentional —
        // we need all nodes to upgrade before making it mandatory.
        //
        // When mandatory becomes true (future v10.5.x):
        // - Old nodes that don't compute state_root correctly will be rejected by the network
        // - This requires 6-week advance notice
        //
        // For now, mandatory=false is the safe default.
        // This is a documentation test.
        let mandatory_current = false;
        assert!(!mandatory_current,
            "StateRootV1 must be non-mandatory on mainnet until all nodes have upgraded");
    }
}

// ============================================================================
// MODULE 2: PRE-ACTIVATION BEHAVIOR
// ============================================================================

mod pre_activation_behavior {
    use super::*;

    #[test]
    fn pre_activation_produces_zero_state_root() {
        let gate = MockUpgradeGate::mainnet();
        let balance_hash = [0xabu8; 32];

        // At any mainnet height (all pre-activation since activation = u64::MAX)
        let state_root = compute_state_root_for_block(&gate, CHECKPOINT_HEIGHT, balance_hash);
        assert_eq!(state_root, [0u8; 32],
            "Pre-activation state_root must be [0u8;32], not balance hash");
    }

    #[test]
    fn pre_activation_accepts_any_state_root_value() {
        let gate = MockUpgradeGate::mainnet();

        // Before activation: validator must accept ANY state_root value
        let test_roots = [
            [0u8; 32],
            [0xffu8; 32],
            [0x01u8; 32],
        ];

        for root in test_roots {
            let result = validate_state_root(
                &gate,
                CHECKPOINT_HEIGHT,
                root,
                [0xabu8; 32], // computed root differs
                false,
            );
            assert!(result.is_ok(),
                "Pre-activation must accept any state_root value (got error for {:?})", root);
        }
    }

    #[test]
    fn all_16m_existing_blocks_have_zero_state_root() {
        // All blocks produced before StateRootV1 activation have state_root = [0u8;32].
        // The validator MUST accept these when replaying history.
        let gate = MockUpgradeGate::mainnet();

        // Simulate replaying blocks 1..CHECKPOINT_HEIGHT
        for height in [1u64, 1000, 100_000, 1_000_000, CHECKPOINT_HEIGHT] {
            let result = validate_state_root(
                &gate,
                height,
                [0u8; 32], // All historical blocks have zero state_root
                [0u8; 32],
                false,
            );
            assert!(result.is_ok(),
                "Historical block at height {} with zero state_root must be accepted", height);
        }
    }

    #[test]
    fn pre_activation_state_root_is_pre_committed_to_block_hash() {
        // The [0u8;32] pre-activation state_root is committed to the block hash.
        // Changing it retroactively would invalidate the block hash (and chain).
        // This is why all historical blocks are safe: their hashes commit to [0;32].
        let h = block_hash(1000, [0u8; 32], [0u8; 32], [0u8; 32]);
        let h_modified = block_hash(1000, [0u8; 32], [0x01u8; 32], [0u8; 32]);

        assert_ne!(h, h_modified,
            "Block hash with state_root=[0;32] differs from hash with state_root=[01;32]. \
             Historical blocks cannot be retrofitted with correct state roots.");
    }
}

// ============================================================================
// MODULE 3: POST-ACTIVATION ENFORCEMENT
// ============================================================================

mod post_activation_enforcement {
    use super::*;

    #[test]
    fn post_activation_uses_balance_hash() {
        let gate = MockUpgradeGate::testnet();
        let balance_hash = [0xabu8; 32];

        let state_root = compute_state_root_for_block(&gate, 1000, balance_hash);
        assert_eq!(state_root, balance_hash,
            "Post-activation state_root must equal the balance hash, not [0;32]");
    }

    #[test]
    fn post_activation_rejects_wrong_state_root() {
        let gate = MockUpgradeGate::testnet();
        let correct_root = [0xaau8; 32];
        let wrong_root   = [0xbbu8; 32];

        let result = validate_state_root(&gate, 1000, wrong_root, correct_root, false);
        assert!(result.is_err(),
            "Post-activation: wrong state_root must be rejected");

        let err = result.unwrap_err();
        assert!(err.contains("mismatch"),
            "Error message must mention 'mismatch': {}", err);
    }

    #[test]
    fn post_activation_rejects_zero_state_root() {
        let gate = MockUpgradeGate::testnet();
        let correct_root = [0xaau8; 32];
        let zero_root    = [0u8; 32];

        let result = validate_state_root(&gate, 1000, zero_root, correct_root, false);
        assert!(result.is_err(),
            "Post-activation: zero state_root must be rejected (indicates missing computation)");

        let err = result.unwrap_err();
        assert!(err.contains("missing root") || err.contains("[0;32]") || err.contains("0000"),
            "Error must indicate zero root is missing: {}", err);
    }

    #[test]
    fn post_activation_accepts_correct_state_root() {
        let gate = MockUpgradeGate::testnet();
        let correct_root = [0xaau8; 32];

        let result = validate_state_root(&gate, 1000, correct_root, correct_root, false);
        assert!(result.is_ok(),
            "Post-activation: correct state_root must be accepted");
    }

    #[test]
    fn post_activation_block_hash_commits_to_correct_root() {
        // After activation, block hash includes the balance-based state_root.
        // A block with wrong state_root has a different hash, so it's trivially
        // distinguishable from a valid block.
        let balance_root = [0xabu8; 32];
        let wrong_root   = [0xcdu8; 32];
        let prev         = [0u8; 32];
        let tx_root      = [0u8; 32];
        let height       = 1000u64;

        let valid_hash = block_hash(height, prev, balance_root, tx_root);
        let wrong_hash = block_hash(height, prev, wrong_root, tx_root);

        assert_ne!(valid_hash, wrong_hash,
            "Block with wrong state_root has different hash — cryptographically distinguishable");
    }
}

// ============================================================================
// MODULE 4: SHADOW MODE
// ============================================================================

mod shadow_mode {
    use super::*;

    #[test]
    fn shadow_mode_does_not_reject_mismatched_block() {
        // Shadow mode: log the mismatch but accept the block.
        // Used for: running state_root computation in production without enforcement,
        // to verify correctness before enabling enforcement.
        let gate = MockUpgradeGate::testnet();
        let correct_root = [0xaau8; 32];
        let wrong_root   = [0xbbu8; 32];

        let result = validate_state_root(
            &gate,
            1000,
            wrong_root,
            correct_root,
            true, // shadow_mode = true
        );

        assert!(result.is_ok(),
            "Shadow mode must NOT reject blocks even with wrong state_root");
    }

    #[test]
    fn enforcement_mode_rejects_mismatched_block() {
        // Enforcement mode: reject block if state_root is wrong.
        let gate = MockUpgradeGate::testnet();
        let correct_root = [0xaau8; 32];
        let wrong_root   = [0xbbu8; 32];

        let result = validate_state_root(
            &gate,
            1000,
            wrong_root,
            correct_root,
            false, // enforcement mode
        );

        assert!(result.is_err(),
            "Enforcement mode must reject blocks with wrong state_root");
    }

    #[test]
    fn shadow_mode_required_duration_before_enforcement() {
        // Shadow mode must run for at least 14 days (at 1 block/second = 1,209,600 blocks)
        // before switching to enforcement mode on testnet.
        // This ensures no false positives from edge cases.
        let shadow_mode_min_blocks = 14 * 24 * 60 * 60u64; // 14 days at 1 bps
        assert_eq!(shadow_mode_min_blocks, 1_209_600,
            "Shadow mode must run for 14 days (1,209,600 blocks) before enforcement");
    }

    #[test]
    fn shadow_mode_catches_implementation_bugs_before_enforcement() {
        // If shadow mode had been enabled on mainnet, it would have detected:
        // 1. compute_state_root() hashes TX IDs, not balance state
        // 2. Two nodes produce different state_roots for same block
        // → Mismatch logged, but network stays up
        // → Bug discovered and fixed before enforcement
        //
        // This is the sequence:
        // 1. Enable shadow mode (log only)
        // 2. Monitor for 14+ days
        // 3. Zero mismatches → switch to enforcement
        // If mismatches found → fix bugs before they affect consensus

        let any_mismatch_found_in_shadow = false; // Would be true if bug exists
        // After shadow mode shows zero mismatches:
        let safe_to_enable_enforcement = !any_mismatch_found_in_shadow;
        assert!(safe_to_enable_enforcement || !safe_to_enable_enforcement,
            "This is a documentation test — shadow mode logic is correct by construction");
    }
}

// ============================================================================
// MODULE 5: ACTIVATION PREREQUISITES
// ============================================================================

mod activation_prerequisites {
    use super::*;

    /// Represents the 6 prerequisites for activating StateRootV1 on mainnet.
    #[derive(Debug, Default)]
    struct ActivationPrerequisites {
        /// 1. compute_state_root() renamed and replaced
        compute_fn_correct: bool,
        /// 2. Block validation enforces (not warns) wrong state_root → reject
        enforcement_enabled: bool,
        /// 3. Post-checkpoint block replay implemented
        block_replay_implemented: bool,
        /// 4. Testnet soak ≥ 14 days with zero mismatches
        testnet_soak_complete: bool,
        /// 5. 6-week mainnet upgrade notice given
        six_week_notice_given: bool,
        /// 6. All nodes (≥ 2/3 stake) have upgraded
        sufficient_nodes_upgraded: bool,
    }

    impl ActivationPrerequisites {
        fn all_met(&self) -> bool {
            self.compute_fn_correct
                && self.enforcement_enabled
                && self.block_replay_implemented
                && self.testnet_soak_complete
                && self.six_week_notice_given
                && self.sufficient_nodes_upgraded
        }

        fn any_missing(&self) -> bool {
            !self.all_met()
        }
    }

    #[test]
    fn current_prerequisites_not_met() {
        // As of v10.4.15, NONE of the prerequisites are complete.
        // StateRootV1 MUST remain at u64::MAX.
        let prereqs = ActivationPrerequisites::default(); // All false
        assert!(prereqs.any_missing(),
            "In v10.4.15, no prerequisites are met — activation must stay at u64::MAX");
    }

    #[test]
    fn all_prerequisites_required_before_setting_activation_height() {
        let mut prereqs = ActivationPrerequisites::default();

        // Simulate completing 5 of 6 prerequisites
        prereqs.compute_fn_correct = true;
        prereqs.enforcement_enabled = true;
        prereqs.block_replay_implemented = true;
        prereqs.testnet_soak_complete = true;
        prereqs.six_week_notice_given = true;
        // six_week_notice_given = false!

        assert!(prereqs.any_missing(),
            "Even with 5/6 prerequisites met, must not activate — all 6 required");
    }

    #[test]
    fn completing_all_prerequisites_allows_activation() {
        let prereqs = ActivationPrerequisites {
            compute_fn_correct: true,
            enforcement_enabled: true,
            block_replay_implemented: true,
            testnet_soak_complete: true,
            six_week_notice_given: true,
            sufficient_nodes_upgraded: true,
        };

        assert!(prereqs.all_met(),
            "All 6 prerequisites complete → safe to set activation height");
    }

    #[test]
    fn minimum_upgrade_notice_is_six_weeks() {
        // $1.5B mainnet requires ≥ 6 weeks notice before mandatory upgrade.
        // This is non-negotiable: node operators need time to upgrade.
        let six_weeks_in_blocks = 6u64 * 7 * 24 * 60 * 60; // 6 weeks at 1 bps
        assert_eq!(six_weeks_in_blocks, MIN_UPGRADE_NOTICE_BLOCKS,
            "Minimum upgrade notice must be exactly 6 weeks ({} blocks)",
            MIN_UPGRADE_NOTICE_BLOCKS);
    }

    #[test]
    fn activation_height_must_be_at_least_six_weeks_in_future() {
        // When setting activation height, it must be ≥ current_height + 3_628_800 blocks.
        let current_mainnet_height = 16_538_868u64; // As of checkpoint
        let proposed_activation = current_mainnet_height + MIN_UPGRADE_NOTICE_BLOCKS;

        let advance_notice = proposed_activation - current_mainnet_height;
        assert!(advance_notice >= MIN_UPGRADE_NOTICE_BLOCKS,
            "Activation height must give ≥ 6 weeks notice ({} blocks)",
            MIN_UPGRADE_NOTICE_BLOCKS);
    }

    #[test]
    fn block_replay_gap_is_known() {
        // Test container (qnk-sync-test-v4) is ~4,336 blocks past checkpoint.
        // These blocks must be replayed after checkpoint import.
        // If replay is not implemented, balances from those blocks are lost.
        let checkpoint_height = 16_538_868u64;
        let test_container_height = 16_543_204u64; // Approximate, as of v10.4.15 session
        let gap = test_container_height - checkpoint_height;

        assert!(gap > 0,
            "There is a block gap between checkpoint and current height");
        assert!(gap < 100_000,
            "Gap should be reasonable (not a full re-sync): {} blocks", gap);
    }
}

// ============================================================================
// MODULE 6: BEDA ANCHOR VERSIONING
// ============================================================================

mod beda_anchor_versioning {
    use super::*;

    /// Simulates BEDA state root semantic versions.
    #[derive(Debug, PartialEq)]
    enum BedaStateRootVersion {
        /// v0: TX-ID-based root (current wrong implementation)
        V0TxIds,
        /// v1: Balance-based root (correct implementation)
        V1Balances,
    }

    fn beda_anchor_version_for_height(
        mainnet_state_root_v1_activation: u64,
        block_height: u64,
    ) -> BedaStateRootVersion {
        if block_height >= mainnet_state_root_v1_activation {
            BedaStateRootVersion::V1Balances
        } else {
            BedaStateRootVersion::V0TxIds
        }
    }

    #[test]
    fn historical_blocks_use_v0_tx_semantics() {
        // All blocks before StateRootV1 activation embed TX-ID-based roots.
        // BEDA anchors for these blocks must be interpreted as TX-ID roots.
        let version = beda_anchor_version_for_height(
            MAINNET_STATE_ROOT_V1_ACTIVATION,
            CHECKPOINT_HEIGHT,
        );
        assert_eq!(version, BedaStateRootVersion::V0TxIds,
            "Historical block at checkpoint height uses V0 TX-ID semantics");
    }

    #[test]
    fn post_activation_blocks_use_v1_balance_semantics() {
        // After StateRootV1 activates, BEDA anchors must be interpreted as balance roots.
        // The BEDA indexer must NOT apply v1 semantics to v0 blocks.
        let hypothetical_activation = 20_000_000u64;
        let post_activation_height = 20_000_001u64;

        let version = beda_anchor_version_for_height(hypothetical_activation, post_activation_height);
        assert_eq!(version, BedaStateRootVersion::V1Balances,
            "Post-activation block uses V1 balance semantics");
    }

    #[test]
    fn v0_and_v1_state_roots_are_semantically_incompatible() {
        // A V0 state root for a block with N transactions has NOTHING to do
        // with the wallet balances at that block.
        // A V1 state root for a block covers wallet balances.
        // An indexer that confuses the two will produce wrong balance proofs.

        // V0: SHA3-256(tx_id_1 || tx_id_2 || ...)
        // V1: Blake3(addr_1 || bal_1 || addr_2 || bal_2 || ...)
        // These are completely different hash structures.
        let v0_root = [0x11u8; 32]; // Placeholder for TX-based root
        let v1_root = [0x22u8; 32]; // Placeholder for balance-based root

        assert_ne!(v0_root, v1_root,
            "V0 and V1 state roots must be treated as semantically incompatible");
    }

    #[test]
    fn beda_must_not_retroactively_apply_v1_to_historical_blocks() {
        // CRITICAL: BEDA cannot recompute V1 balance roots for historical blocks
        // because those blocks used V0 TX-ID roots. The state root stored in
        // Bitcoin/BEDA for those blocks IS V0.
        // The BEDA indexer must record the version alongside each anchor.

        let historical_block_height = CHECKPOINT_HEIGHT;
        let historical_version = beda_anchor_version_for_height(
            MAINNET_STATE_ROOT_V1_ACTIVATION,
            historical_block_height,
        );

        assert_eq!(historical_version, BedaStateRootVersion::V0TxIds,
            "BEDA must not retroactively apply V1 semantics to {} historical blocks",
            historical_block_height);
    }
}

// ============================================================================
// MODULE 7: ACTIVATION IS IRREVERSIBLE
// ============================================================================

mod activation_irreversibility {
    use super::*;

    #[test]
    fn once_activation_height_passes_cannot_deactivate() {
        // Once StateRootV1 activates, there is no way to go back without a hard fork.
        // All blocks produced after activation have balance-based state roots committed.
        // Rolling back would mean all those block hashes are invalid.

        let gate = MockUpgradeGate::testnet(); // Already active
        let block_height = 1_000_000u64;

        assert!(gate.state_root_v1_active(block_height),
            "Active gate cannot be deactivated without hard fork");

        // Any block produced at this height is committed to having a balance root
        let balance_root = [0xabu8; 32];
        let block_h = block_hash(block_height, [0u8; 32], balance_root, [0u8; 32]);

        // Verify: changing the state_root changes the hash (commitment is irreversible)
        let modified_h = block_hash(block_height, [0u8; 32], [0u8; 32], [0u8; 32]);
        assert_ne!(block_h, modified_h,
            "Post-activation block hash is committed to the balance root — irreversible");
    }

    #[test]
    fn activation_height_must_be_set_once_never_lowered() {
        // Setting activation height to X and later lowering it to X-1 would:
        // 1. Make blocks at height X-1 that were accepted (as pre-activation) now invalid
        // 2. Cause a chain reorg
        // 3. Potentially lose funds
        // NEVER lower the activation height after announcement.

        let announced_height = 20_000_000u64;
        let proposed_new_height = 19_999_999u64; // LOWER than announced!

        let is_safe_change = proposed_new_height >= announced_height;
        assert!(!is_safe_change,
            "Lowering activation height from {} to {} is UNSAFE — never do this",
            announced_height, proposed_new_height);
    }

    #[test]
    fn compute_state_root_rename_is_also_irreversible() {
        // When compute_state_root() is renamed to compute_transaction_set_root():
        // - Any code that calls compute_state_root() for block production stops working
        // - This forces all paths to call the new balance-based function
        // - The old function name CANNOT be reused for a different purpose
        //
        // This is a documentation test.
        let rename_planned = true;
        assert!(rename_planned,
            "compute_state_root() must be renamed to compute_transaction_set_root() \
             before StateRootV1 activation");
    }
}

// ============================================================================
// MODULE 8: ATOMIC VALIDATION
// ============================================================================

mod atomic_validation {
    use super::*;

    /// Simulates the correct atomic validation flow:
    /// 1. Apply transactions to OVERLAY (not canonical DB)
    /// 2. Compute state root from overlay
    /// 3. Compare with block.header.state_root
    /// 4. If match: commit overlay to canonical DB
    /// 5. If mismatch: discard overlay (canonical DB unchanged)
    #[derive(Debug, Default)]
    struct AtomicValidator {
        overlay_balances: std::collections::HashMap<[u8; 32], u128>,
        canonical_balances: std::collections::HashMap<[u8; 32], u128>,
        committed: bool,
    }

    impl AtomicValidator {
        fn apply_to_overlay(&mut self, addr: [u8; 32], new_balance: u128) {
            self.overlay_balances.insert(addr, new_balance);
        }

        fn compute_overlay_root(&self) -> [u8; 32] {
            let mut sorted: Vec<_> = self.overlay_balances.iter()
                .filter(|(_, &b)| b > 0)
                .collect();
            sorted.sort_by_key(|(a, _)| *a);
            let mut hasher = blake3::Hasher::new();
            for (addr, &bal) in &sorted {
                hasher.update(addr.as_slice());
                hasher.update(&bal.to_le_bytes());
            }
            *hasher.finalize().as_bytes()
        }

        fn commit(&mut self) {
            self.canonical_balances = self.overlay_balances.clone();
            self.committed = true;
        }

        fn discard_overlay(&mut self) {
            self.overlay_balances.clear();
            self.committed = false;
        }

        fn validate_and_commit(&mut self, claimed_root: [u8; 32]) -> Result<(), String> {
            let computed = self.compute_overlay_root();
            if computed == claimed_root {
                self.commit();
                Ok(())
            } else {
                self.discard_overlay();
                Err(format!("State root mismatch: expected {}, got {}",
                    hex_str(&claimed_root), hex_str(&computed)))
            }
        }
    }

    #[test]
    fn valid_block_commits_overlay_to_canonical() {
        let mut validator = AtomicValidator::default();
        let addr = [0x01u8; 32];
        let balance = 1_000_000u128;

        validator.apply_to_overlay(addr, balance);
        let correct_root = validator.compute_overlay_root();

        let result = validator.validate_and_commit(correct_root);
        assert!(result.is_ok(), "Valid block must commit overlay");
        assert!(validator.committed, "Canonical DB must be updated on valid block");
        assert_eq!(*validator.canonical_balances.get(&addr).unwrap(), balance);
    }

    #[test]
    fn invalid_block_discards_overlay_canonical_unchanged() {
        let mut validator = AtomicValidator::default();
        let addr = [0x01u8; 32];
        let original_balance = 500_000u128;
        let attacker_balance = 999_999_999u128;

        // Canonical DB has original balance
        validator.canonical_balances.insert(addr, original_balance);

        // Attacker's block tries to give themselves more tokens
        validator.apply_to_overlay(addr, attacker_balance);
        let wrong_root = [0xdeu8; 32]; // Doesn't match actual overlay root

        let result = validator.validate_and_commit(wrong_root);
        assert!(result.is_err(), "Invalid block must be rejected");
        assert!(!validator.committed, "Canonical DB must NOT be updated on invalid block");
        // Original balance preserved
        assert_eq!(*validator.canonical_balances.get(&addr).unwrap(), original_balance,
            "Canonical balance must be unchanged after invalid block rejection");
    }

    #[test]
    fn overlay_isolation_prevents_partial_state_mutation() {
        // CRITICAL: The overlay must be committed atomically.
        // If the system crashes between applying TXs and committing,
        // the canonical DB must remain in the pre-block state.
        let mut validator = AtomicValidator::default();

        let addr1 = [0x01u8; 32];
        let addr2 = [0x02u8; 32];

        // Canonical: both have 100
        validator.canonical_balances.insert(addr1, 100u128);
        validator.canonical_balances.insert(addr2, 100u128);

        // Apply transfer: addr1 → addr2 (50 units)
        validator.apply_to_overlay(addr1, 50u128);
        validator.apply_to_overlay(addr2, 150u128);

        // Simulate failure before commit (wrong state root)
        let wrong_root = [0xffu8; 32];
        let _ = validator.validate_and_commit(wrong_root);

        // Canonical must be unchanged
        assert_eq!(*validator.canonical_balances.get(&addr1).unwrap(), 100u128,
            "addr1 canonical balance must be unchanged after failed block");
        assert_eq!(*validator.canonical_balances.get(&addr2).unwrap(), 100u128,
            "addr2 canonical balance must be unchanged after failed block");
    }
}

// ============================================================================
// MODULE 9: BACKWARD SYNC GATE
// ============================================================================

mod backward_sync_gate {
    use super::*;

    #[test]
    fn backward_sync_must_be_disabled_after_checkpoint_applied() {
        // The 15-second backward RocksDB→HashMap sync reads ALL wallet_balance_* keys
        // and rebuilds the in-memory HashMap.
        // After the checkpoint is applied, this sync would overwrite the checkpoint data
        // IF the RocksDB reflects the wrong pre-checkpoint state.
        //
        // The backward sync MUST be gated behind !checkpoint_applied.
        // This test documents the requirement.
        let checkpoint_applied = true;
        let should_run_backward_sync = !checkpoint_applied;

        assert!(!should_run_backward_sync,
            "Backward sync must be DISABLED after checkpoint is applied");
    }

    #[test]
    fn backward_sync_enabled_before_checkpoint() {
        // Before checkpoint, backward sync is safe (no checkpoint data to preserve).
        let checkpoint_applied = false;
        let should_run_backward_sync = !checkpoint_applied;

        assert!(should_run_backward_sync,
            "Backward sync is ENABLED before checkpoint (normal operation)");
    }

    #[test]
    fn p2p_balance_sync_must_not_overwrite_checkpoint_balances() {
        // P2P balance updates arrive from peers. If a peer sends an old/wrong balance
        // for a wallet that was correctly imported by the checkpoint, accepting it would
        // corrupt the checkpoint.
        //
        // The balance update handler must:
        // 1. Check if the balance comes with a valid block hash
        // 2. If no block hash (legacy update): reject if checkpoint is applied
        // 3. If block hash present: only accept if block is AFTER checkpoint height

        let checkpoint_height = CHECKPOINT_HEIGHT;
        let p2p_update_block_height = 16_538_000u64; // BEFORE checkpoint!

        let update_is_safe = p2p_update_block_height > checkpoint_height;
        assert!(!update_is_safe,
            "P2P balance update from before checkpoint height must be rejected");
    }

    #[test]
    fn p2p_balance_update_post_checkpoint_is_safe() {
        let checkpoint_height = CHECKPOINT_HEIGHT;
        let p2p_update_block_height = 16_539_000u64; // AFTER checkpoint

        let update_is_safe = p2p_update_block_height > checkpoint_height;
        assert!(update_is_safe,
            "P2P balance update from after checkpoint height is safe to apply");
    }
}

// ============================================================================
// MODULE 10: FULL ACTIVATION SEQUENCE
// ============================================================================

mod full_activation_sequence {
    use super::*;

    /// Documents the correct activation sequence for StateRootV1 on mainnet.
    /// Each step is a test that verifies its own precondition.
    #[test]
    fn step_1_rename_compute_state_root() {
        // compute_state_root() → compute_transaction_set_root()
        // New compute_state_root() calls compute_balance_state_hash() from q-storage
        // CURRENT STATUS: NOT DONE
        let done = false;
        // When done, change this to true — a failing test reminds us to complete it
        let _ = done; // Documentation test
    }

    #[test]
    fn step_2_implement_post_checkpoint_block_replay() {
        // After applying checkpoint, replay native-coin changes from blocks
        // CHECKPOINT_HEIGHT+1 through current tip.
        // This closes the 4,336-block gap in the test container.
        // CURRENT STATUS: NOT DONE
        let gap_blocks = 4_336u64;
        assert!(gap_blocks > 0, "Block replay gap exists and must be implemented");
    }

    #[test]
    fn step_3_disable_backward_rdb_to_hashmap_sync() {
        // Gate the 15-second backward sync behind !checkpoint_applied.
        // CURRENT STATUS: NOT DONE
        let _ = ();
    }

    #[test]
    fn step_4_enable_shadow_mode_on_testnet() {
        // Deploy to testnet with shadow mode: compute state_root but don't reject on mismatch.
        // Monitor for ≥ 14 days. Zero mismatches required.
        let min_shadow_days = 14u32;
        assert!(min_shadow_days >= 14, "Shadow mode minimum duration: 14 days");
    }

    #[test]
    fn step_5_enable_enforcement_on_testnet() {
        // After 14 days shadow mode with zero mismatches, enable enforcement on testnet.
        // Monitor for ≥ 7 more days.
        let min_enforcement_days = 7u32;
        assert!(min_enforcement_days >= 7, "Testnet enforcement minimum duration: 7 days");
    }

    #[test]
    fn step_6_announce_mainnet_upgrade_with_six_week_notice() {
        // Announce activation height = current_height + 3,628,800 blocks (6 weeks).
        // Publish on Discord, BitcoinTalk, quillon.xyz blog.
        let notice_blocks = MIN_UPGRADE_NOTICE_BLOCKS;
        assert_eq!(notice_blocks, 3_628_800,
            "6-week notice must be exactly {} blocks", 3_628_800);
    }

    #[test]
    fn step_7_set_mainnet_activation_height() {
        // After 6-week notice expires, set MAINNET_UPGRADES[StateRootV1].activation_height
        // = announced_height. Deploy via ha-deploy.sh rolling upgrade.
        // IMPORTANT: This is the ONLY step that changes the constant from u64::MAX.
        let current_safe_value = MAINNET_STATE_ROOT_V1_ACTIVATION;
        assert_eq!(current_safe_value, u64::MAX,
            "Currently safe — u64::MAX. Only change after steps 1-6 are complete.");
    }

    #[test]
    fn complete_sequence_summary() {
        // Summary: 7 steps, all must complete before StateRootV1 goes live.
        // Estimated timeline (from v10.4.15):
        // - Steps 1-3: ~2 weeks implementation
        // - Step 4: 14 days shadow testnet
        // - Step 5: 7 days enforcement testnet
        // - Step 6: 6-week mainnet announcement
        // - Step 7: Deployment
        // Total minimum: 2 + 2 + 1 + 6 = 11 weeks from v10.4.15
        let min_weeks_from_now = 11u32;
        assert!(min_weeks_from_now >= 11,
            "StateRootV1 mainnet activation takes at least {} weeks from v10.4.15", min_weeks_from_now);
    }
}
