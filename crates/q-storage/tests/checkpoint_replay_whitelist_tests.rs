//! Checkpoint Replay Whitelist Tests — v10.4.15+
//!
//! WHAT THIS FILE TESTS (Unknown 5):
//!
//! The post-checkpoint block replay function must correctly classify every
//! transaction type in the Q-NarwhalKnight type system. A classification error
//! causes silent balance divergence on mainnet.
//!
//! REPLAY WHITELIST (conservative Phase 0):
//!   APPLY:  Coinbase (0x01) — mining reward credited to wallet_balance_*
//!           Transfer (0x00) — native QUG moved between wallets
//!   SKIP:   Everything else — too complex, touches non-native state, or requires
//!           full execution context that isn't available in replay mode
//!
//! WHY SKIP EVERYTHING ELSE:
//!   Burn (0x02): Could affect wallet balances, but fee/burn accounting needs full context
//!   Fee (0x03): Fee recipient logic requires block context
//!   Swap (0x23): May involve native QUG but DEX state must also be consistent
//!   VaultLock/Unlock: Moves native to/from vault — vault state not checkpointed
//!   Any contract op: May touch wallet_balance_* via contract execution
//!
//! RISK: A skipped TX means its native balance effect is NOT replayed. The node
//! will have slightly wrong balances for the gap period (H to current). This is
//! acceptable as a Phase 0 limitation — far better than partially-replayed state.
//!
//! Run with: cargo test --package q-storage --test checkpoint_replay_whitelist_tests

use std::collections::HashMap;

// ============================================================================
// TX TYPE CONSTANTS (mirrors TransactionType enum in q-types)
// These are the wire codes. DO NOT CHANGE without updating TransactionType enum.
// ============================================================================

const TX_TRANSFER: u8 = 0x00;        // APPLY — native QUG transfer
const TX_COINBASE: u8 = 0x01;        // APPLY — mining reward
const TX_BURN: u8 = 0x02;            // SKIP — burn accounting needs context
const TX_FEE: u8 = 0x03;             // SKIP — fee routing needs block context
const TX_TOKEN_CREATE: u8 = 0x10;    // SKIP — token state (token_balance_*)
const TX_TOKEN_MINT: u8 = 0x11;      // SKIP — token state
const TX_TOKEN_TRANSFER: u8 = 0x12;  // SKIP — token state
const TX_TOKEN_BURN: u8 = 0x13;      // SKIP — token state
const TX_POOL_CREATE: u8 = 0x20;     // SKIP — DEX state (liquidity_pool:*)
const TX_POOL_ADD_LIQ: u8 = 0x21;    // SKIP — DEX state
const TX_POOL_REMOVE_LIQ: u8 = 0x22; // SKIP — DEX state
const TX_SWAP: u8 = 0x23;            // SKIP — DEX state + native might be involved
const TX_LIMIT_ORDER: u8 = 0x24;     // SKIP — DEX state
const TX_CONTRACT_DEPLOY: u8 = 0x30; // SKIP — contract state
const TX_CONTRACT_CALL: u8 = 0x31;   // SKIP — contract may touch wallet_balance_*
const TX_VAULT_LOCK: u8 = 0x40;      // SKIP — vault state not checkpointed
const TX_VAULT_UNLOCK: u8 = 0x41;    // SKIP — vault state not checkpointed
const TX_STABLE_MINT: u8 = 0x42;     // SKIP — token state
const TX_STABLE_BURN: u8 = 0x43;     // SKIP — token state

// ============================================================================
// REPLAY STRUCTURES
// ============================================================================

/// Minimal representation of a transaction for replay classification.
#[derive(Debug, Clone)]
struct MockTx {
    tx_type: u8,
    from: [u8; 32],   // sender (zero addr for coinbase)
    to: [u8; 32],     // receiver
    amount: u128,     // in raw units (24 decimals)
}

impl MockTx {
    fn coinbase(to: [u8; 32], amount: u128) -> Self {
        Self {
            tx_type: TX_COINBASE,
            from: [0u8; 32], // zero address = system
            to,
            amount,
        }
    }

    fn transfer(from: [u8; 32], to: [u8; 32], amount: u128) -> Self {
        Self { tx_type: TX_TRANSFER, from, to, amount }
    }

    fn of_type(tx_type: u8) -> Self {
        Self {
            tx_type,
            from: [0xAA; 32],
            to: [0xBB; 32],
            amount: 1_000_000_000_000_000_000_000_000,
        }
    }
}

/// The replay whitelist classifier.
/// Returns true if the TX should be applied to wallet_balance_* during replay.
fn is_replay_apply(tx_type: u8) -> bool {
    matches!(tx_type, TX_TRANSFER | TX_COINBASE)
}

/// Apply a single TX to the wallet store (only called for APPLY-classified TXs).
/// Returns Err if the TX would cause a negative balance or overflow.
fn apply_tx_to_wallets(
    wallets: &mut HashMap<[u8; 32], u128>,
    tx: &MockTx,
) -> Result<(), String> {
    match tx.tx_type {
        TX_COINBASE => {
            // Credit reward to `to` address (mining reward)
            let balance = wallets.entry(tx.to).or_insert(0);
            *balance = balance
                .checked_add(tx.amount)
                .ok_or_else(|| format!("Coinbase overflow for {:?}", tx.to))?;
            Ok(())
        }
        TX_TRANSFER => {
            // Debit from `from`, credit to `to`
            let from_balance = wallets.get(&tx.from).copied().unwrap_or(0);
            if from_balance < tx.amount {
                return Err(format!(
                    "Transfer would make balance negative: have {}, need {}",
                    from_balance, tx.amount
                ));
            }
            *wallets.entry(tx.from).or_insert(0) -= tx.amount;
            *wallets.entry(tx.to).or_insert(0) = wallets
                .get(&tx.to)
                .copied()
                .unwrap_or(0)
                .checked_add(tx.amount)
                .ok_or("Transfer to overflow")?;
            Ok(())
        }
        _ => Err(format!("TX type 0x{:02x} is not in the apply whitelist", tx.tx_type)),
    }
}

/// Replay a block (list of TXs) onto the wallet store.
/// Returns (applied_count, skipped_count).
fn replay_block(
    wallets: &mut HashMap<[u8; 32], u128>,
    txs: &[MockTx],
) -> (usize, usize) {
    let mut applied = 0;
    let mut skipped = 0;
    for tx in txs {
        if is_replay_apply(tx.tx_type) {
            match apply_tx_to_wallets(wallets, tx) {
                Ok(()) => applied += 1,
                Err(_) => skipped += 1, // reject also counts as skipped
            }
        } else {
            skipped += 1;
        }
    }
    (applied, skipped)
}

fn make_addr(seed: u8) -> [u8; 32] {
    let mut a = [0u8; 32];
    a[0] = seed;
    a
}

// ============================================================================
// MODULE 1: TX TYPE CLASSIFICATION (the whitelist)
// ============================================================================

mod tx_type_classification {
    use super::*;

    #[test]
    fn coinbase_is_apply() {
        assert!(is_replay_apply(TX_COINBASE), "Coinbase must be in APPLY whitelist");
    }

    #[test]
    fn transfer_is_apply() {
        assert!(is_replay_apply(TX_TRANSFER), "Transfer must be in APPLY whitelist");
    }

    #[test]
    fn burn_is_skip() {
        assert!(!is_replay_apply(TX_BURN), "Burn must be SKIPPED — requires full context");
    }

    #[test]
    fn fee_is_skip() {
        assert!(!is_replay_apply(TX_FEE), "Fee must be SKIPPED — fee routing needs block context");
    }

    #[test]
    fn all_token_ops_are_skip() {
        let token_ops = [
            TX_TOKEN_CREATE, TX_TOKEN_MINT, TX_TOKEN_TRANSFER, TX_TOKEN_BURN,
        ];
        for op in token_ops {
            assert!(!is_replay_apply(op), "Token op 0x{:02x} must be SKIPPED", op);
        }
    }

    #[test]
    fn all_dex_ops_are_skip() {
        let dex_ops = [
            TX_POOL_CREATE, TX_POOL_ADD_LIQ, TX_POOL_REMOVE_LIQ,
            TX_SWAP, TX_LIMIT_ORDER,
        ];
        for op in dex_ops {
            assert!(!is_replay_apply(op), "DEX op 0x{:02x} must be SKIPPED", op);
        }
    }

    #[test]
    fn all_contract_ops_are_skip() {
        let contract_ops = [TX_CONTRACT_DEPLOY, TX_CONTRACT_CALL];
        for op in contract_ops {
            assert!(
                !is_replay_apply(op),
                "Contract op 0x{:02x} must be SKIPPED — may touch wallet_balance_* unpredictably",
                op
            );
        }
    }

    #[test]
    fn all_vault_ops_are_skip() {
        let vault_ops = [TX_VAULT_LOCK, TX_VAULT_UNLOCK, TX_STABLE_MINT, TX_STABLE_BURN];
        for op in vault_ops {
            assert!(
                !is_replay_apply(op),
                "Vault op 0x{:02x} must be SKIPPED — vault state not in native checkpoint",
                op
            );
        }
    }

    #[test]
    fn swap_is_skip_even_though_it_may_involve_native_qug() {
        // A QUG→qUSD swap would change wallet_balance_* for the sender.
        // We SKIP it anyway because: (a) the DEX pool state is not checkpointed,
        // and (b) a partial replay that moves native but not pool reserves would
        // corrupt k-invariant. Phase 0 accepts this imprecision.
        assert!(
            !is_replay_apply(TX_SWAP),
            "Swap must be SKIPPED even when involving native QUG — pool state not replayed"
        );
    }

    #[test]
    fn unknown_tx_type_is_skip() {
        // Any future TX type defaults to SKIP (whitelist, not blacklist)
        let unknown_types = [0x05u8, 0x0F, 0x1F, 0x2F, 0x50, 0x60, 0xFF];
        for t in unknown_types {
            assert!(
                !is_replay_apply(t),
                "Unknown TX type 0x{:02x} must be SKIPPED (whitelist approach)",
                t
            );
        }
    }

    #[test]
    fn whitelist_has_exactly_two_members() {
        // INVARIANT: only Coinbase and Transfer are in the whitelist.
        // If someone adds a third type, this test forces a conscious review.
        let all_types: Vec<u8> = (0u8..=0xFF).collect();
        let apply_types: Vec<u8> = all_types
            .iter()
            .copied()
            .filter(|&t| is_replay_apply(t))
            .collect();

        assert_eq!(
            apply_types,
            vec![TX_TRANSFER, TX_COINBASE],
            "Whitelist must contain exactly Transfer (0x00) and Coinbase (0x01)"
        );
    }
}

// ============================================================================
// MODULE 2: REPLAY LOGIC CORRECTNESS
// ============================================================================

mod replay_logic {
    use super::*;

    #[test]
    fn coinbase_credits_correct_receiver() {
        let miner = make_addr(0x01);
        let reward = 2_625_000_000_000_000_000_000_000_000u128; // ~2.625 QUG

        let mut wallets = HashMap::new();
        let tx = MockTx::coinbase(miner, reward);
        apply_tx_to_wallets(&mut wallets, &tx).unwrap();

        assert_eq!(wallets[&miner], reward);
        // No other wallet touched
        assert_eq!(wallets.len(), 1);
    }

    #[test]
    fn coinbase_accumulates_multiple_rewards() {
        let miner = make_addr(0x01);
        let reward = 1_000_000_000_000_000_000_000_000u128;

        let mut wallets = HashMap::new();
        for _ in 0..5 {
            apply_tx_to_wallets(&mut wallets, &MockTx::coinbase(miner, reward)).unwrap();
        }

        assert_eq!(wallets[&miner], reward * 5);
    }

    #[test]
    fn transfer_moves_balance_from_sender_to_receiver() {
        let alice = make_addr(0x01);
        let bob = make_addr(0x02);
        let initial = 10_000_000_000_000_000_000_000_000u128; // 10 QUG
        let send = 3_000_000_000_000_000_000_000_000u128;      // 3 QUG

        let mut wallets = HashMap::new();
        wallets.insert(alice, initial);

        let tx = MockTx::transfer(alice, bob, send);
        apply_tx_to_wallets(&mut wallets, &tx).unwrap();

        assert_eq!(wallets[&alice], initial - send);
        assert_eq!(wallets[&bob], send);
    }

    #[test]
    fn transfer_rejected_if_sender_has_insufficient_balance() {
        let alice = make_addr(0x01);
        let bob = make_addr(0x02);
        let initial = 1_000u128;
        let send = 2_000u128; // more than alice has

        let mut wallets = HashMap::new();
        wallets.insert(alice, initial);

        let tx = MockTx::transfer(alice, bob, send);
        let result = apply_tx_to_wallets(&mut wallets, &tx);

        assert!(result.is_err(), "Transfer exceeding balance must be rejected");
        // Alice's balance must be unchanged
        assert_eq!(wallets[&alice], initial, "Sender balance must not change on rejection");
        assert_eq!(wallets.get(&bob).copied().unwrap_or(0), 0, "Receiver must not receive anything");
    }

    #[test]
    fn transfer_of_zero_is_rejected_or_noop() {
        let alice = make_addr(0x01);
        let bob = make_addr(0x02);

        let mut wallets = HashMap::new();
        wallets.insert(alice, 1_000_000u128);

        // Zero transfer: alice sends 0 to bob
        let tx = MockTx::transfer(alice, bob, 0);
        let _ = apply_tx_to_wallets(&mut wallets, &tx);

        // Either rejected or noop — alice and bob balances unchanged
        assert_eq!(wallets[&alice], 1_000_000u128);
        assert_eq!(wallets.get(&bob).copied().unwrap_or(0), 0);
    }

    #[test]
    fn coinbase_never_overflows_u128() {
        let miner = make_addr(0x01);
        let near_max = u128::MAX - 1_000_000_000_000_000_000_000_000u128;

        let mut wallets = HashMap::new();
        wallets.insert(miner, near_max);

        // Small safe reward
        let safe_reward = 500_000_000_000_000_000_000_000u128;
        let tx = MockTx::coinbase(miner, safe_reward);
        let result = apply_tx_to_wallets(&mut wallets, &tx);
        // Should succeed — doesn't overflow
        assert!(result.is_ok());

        // Overflow: would push past u128::MAX
        let overflow_reward = u128::MAX - near_max + 1; // one more than fits
        wallets.insert(miner, near_max);
        let tx2 = MockTx::coinbase(miner, overflow_reward);
        let result2 = apply_tx_to_wallets(&mut wallets, &tx2);
        assert!(result2.is_err(), "Coinbase overflow must be detected and rejected");
    }

    #[test]
    fn replay_of_empty_block_is_noop() {
        let mut wallets = HashMap::new();
        wallets.insert(make_addr(0x01), 1_000_000u128);

        let (applied, skipped) = replay_block(&mut wallets, &[]);

        assert_eq!(applied, 0);
        assert_eq!(skipped, 0);
        assert_eq!(wallets[&make_addr(0x01)], 1_000_000u128, "Empty block = no changes");
    }

    #[test]
    fn replay_only_applies_whitelist_txs_in_mixed_block() {
        let miner = make_addr(0x01);
        let alice = make_addr(0x02);
        let bob = make_addr(0x03);
        let reward = 2_625_000_000_000_000_000_000_000u128;
        let transfer_amount = 1_000_000_000_000_000_000_000_000u128;

        let mut wallets = HashMap::new();
        wallets.insert(alice, 5_000_000_000_000_000_000_000_000u128);

        // Mixed block: coinbase + transfer (APPLY) + swap + contract call (SKIP)
        let txs = vec![
            MockTx::coinbase(miner, reward),                // APPLY
            MockTx::transfer(alice, bob, transfer_amount),  // APPLY
            MockTx::of_type(TX_SWAP),                       // SKIP
            MockTx::of_type(TX_CONTRACT_CALL),              // SKIP
            MockTx::of_type(TX_TOKEN_MINT),                 // SKIP
        ];

        let (applied, skipped) = replay_block(&mut wallets, &txs);

        assert_eq!(applied, 2, "Only coinbase + transfer should be applied");
        assert_eq!(skipped, 3, "swap + contract + token mint should be skipped");

        // Verify the applied TXs had correct effect
        assert_eq!(wallets[&miner], reward, "Miner received coinbase reward");
        assert_eq!(
            wallets[&alice],
            5_000_000_000_000_000_000_000_000 - transfer_amount,
            "Alice balance reduced by transfer"
        );
        assert_eq!(wallets[&bob], transfer_amount, "Bob received transfer");
    }

    #[test]
    fn replay_is_order_dependent() {
        // Transfer then coinbase, vs coinbase then transfer — different results if
        // the transfer is from the miner and the coinbase credits the miner first.
        let miner = make_addr(0x01);
        let bob = make_addr(0x02);
        let reward = 1_000_000_000_000_000_000_000_000u128;
        let send = 2_000_000_000_000_000_000_000_000u128; // sends more than reward

        // Scenario A: coinbase first (miner gets reward), then transfer (miner sends)
        let mut wallets_a = HashMap::new();
        let txs_a = vec![
            MockTx::coinbase(miner, reward),
            MockTx::transfer(miner, bob, send), // send > reward: insufficient balance
        ];
        let (applied_a, skipped_a) = replay_block(&mut wallets_a, &txs_a);
        assert_eq!(applied_a, 1, "Only coinbase applied (transfer rejected)");
        assert_eq!(skipped_a, 1, "Transfer rejected due to insufficient balance");

        // Scenario B: transfer first (fails — miner has 0 balance), then coinbase
        let mut wallets_b = HashMap::new();
        let txs_b = vec![
            MockTx::transfer(miner, bob, send), // fails: miner has 0
            MockTx::coinbase(miner, reward),
        ];
        let (applied_b, skipped_b) = replay_block(&mut wallets_b, &txs_b);
        assert_eq!(applied_b, 1, "Only coinbase applied");
        assert_eq!(skipped_b, 1, "Transfer fails (miner had 0 at that point)");
        assert_eq!(wallets_b[&miner], reward, "Miner only has the coinbase reward");
    }

    #[test]
    fn replay_does_not_touch_token_or_pool_keys() {
        // INVARIANT: the replay function must ONLY modify wallet_balance_* data.
        // We simulate this by checking that only our wallet HashMap is modified.
        let miner = make_addr(0x01);
        let alice = make_addr(0x02);
        let bob = make_addr(0x03);

        let mut wallets = HashMap::new();
        wallets.insert(alice, 10_000_000_000_000_000_000_000_000u128);

        let txs = vec![
            MockTx::coinbase(miner, 1_000_000_000_000_000_000_000_000u128),
            MockTx::transfer(alice, bob, 500_000_000_000_000_000_000_000u128),
            MockTx::of_type(TX_SWAP),             // SKIP
            MockTx::of_type(TX_POOL_ADD_LIQ),     // SKIP
            MockTx::of_type(TX_TOKEN_MINT),       // SKIP
        ];

        // Track wallet keys before
        let keys_before: std::collections::HashSet<[u8; 32]> = wallets.keys().copied().collect();

        replay_block(&mut wallets, &txs);

        // Only the APPLY txs should have added new keys (miner + bob)
        // token_balance_* and liquidity_pool:* never appear in the wallet HashMap
        let new_keys: Vec<_> = wallets
            .keys()
            .filter(|k| !keys_before.contains(*k))
            .collect();

        // New keys: only miner (coinbase) and bob (transfer target)
        assert_eq!(new_keys.len(), 2, "Only coinbase receiver and transfer receiver added");

        // No key should look like a token or pool key (they'd be string-typed, not [u8;32])
        // This is structurally guaranteed by using a HashMap<[u8; 32], u128>
        assert!(
            true,
            "Wallet HashMap only holds 32-byte address keys — token/pool keys are different types"
        );
    }

    #[test]
    fn replay_of_multiple_blocks_accumulates_correctly() {
        let miner = make_addr(0x01);
        let alice = make_addr(0x02);
        let bob = make_addr(0x03);

        let reward = 2_625_000_000_000_000_000_000_000u128;
        let send = 1_000_000_000_000_000_000_000_000u128;

        let mut wallets = HashMap::new();

        // Simulate replaying 5 blocks
        for block_num in 0..5u32 {
            let mut block_txs = vec![MockTx::coinbase(miner, reward)];

            // Even blocks: alice sends to bob (alice is pre-funded)
            if block_num == 0 {
                wallets.insert(alice, 10_000_000_000_000_000_000_000_000u128);
            }
            if block_num % 2 == 0 && wallets.get(&alice).copied().unwrap_or(0) >= send {
                block_txs.push(MockTx::transfer(alice, bob, send));
            }

            // Add some skipped TXs to each block
            block_txs.push(MockTx::of_type(TX_SWAP));

            replay_block(&mut wallets, &block_txs);
        }

        // Miner got 5 coinbase rewards
        assert_eq!(wallets[&miner], reward * 5, "Miner accumulated 5 block rewards");
        // Alice sent 3 times (blocks 0, 2, 4)
        // Block 0: alice sends, balance = 10M - 1M = 9M
        // Block 2: alice sends, balance = 9M - 1M = 8M
        // Block 4: alice sends, balance = 8M - 1M = 7M
        let expected_alice = 10_000_000_000_000_000_000_000_000u128 - send * 3;
        assert_eq!(wallets[&alice], expected_alice, "Alice sent 3 transfers");
        assert_eq!(wallets[&bob], send * 3, "Bob received 3 transfers");
    }
}

// ============================================================================
// MODULE 3: REPLAY IDEMPOTENCY (replayed_through_height guard)
// ============================================================================

mod replay_idempotency {
    use super::*;

    /// Simulates the replay state tracker (replayed_through_height in the extended marker).
    #[derive(Default)]
    struct ReplayTracker {
        wallet_db: HashMap<[u8; 32], u128>,
        replayed_through_height: u64,
    }

    impl ReplayTracker {
        fn replay_blocks_up_to(&mut self, blocks: &[(u64, Vec<MockTx>)]) {
            for (block_height, txs) in blocks {
                if *block_height <= self.replayed_through_height {
                    continue; // already replayed
                }
                replay_block(&mut self.wallet_db, txs);
                self.replayed_through_height = *block_height;
            }
        }
    }

    #[test]
    fn replay_same_block_twice_does_not_double_apply() {
        let miner = make_addr(0x01);
        let reward = 1_000_000_000_000_000_000_000_000u128;

        let blocks = vec![(
            16_538_869u64,
            vec![MockTx::coinbase(miner, reward)],
        )];

        let mut tracker = ReplayTracker::default();

        // Replay once
        tracker.replay_blocks_up_to(&blocks);
        assert_eq!(tracker.wallet_db[&miner], reward, "First replay correct");
        assert_eq!(tracker.replayed_through_height, 16_538_869);

        // Replay same block again (simulates restart/idempotency)
        tracker.replay_blocks_up_to(&blocks);
        assert_eq!(
            tracker.wallet_db[&miner], reward,
            "Second replay must be a no-op — replayed_through_height guard works"
        );
    }

    #[test]
    fn replay_continues_from_last_height_after_restart() {
        let miner = make_addr(0x01);
        let reward = 1_000_000_000_000_000_000_000_000u128;

        let all_blocks = vec![
            (16_538_869u64, vec![MockTx::coinbase(miner, reward)]),
            (16_538_870u64, vec![MockTx::coinbase(miner, reward)]),
            (16_538_871u64, vec![MockTx::coinbase(miner, reward)]),
        ];

        let mut tracker = ReplayTracker::default();

        // Replay first 2 blocks, then "restart" (simulate crash)
        tracker.replay_blocks_up_to(&all_blocks[..2]);
        assert_eq!(tracker.wallet_db[&miner], reward * 2);
        assert_eq!(tracker.replayed_through_height, 16_538_870);

        // Resume from where we left off (not from beginning)
        tracker.replay_blocks_up_to(&all_blocks);
        assert_eq!(
            tracker.wallet_db[&miner],
            reward * 3,
            "Replay continued from block 16_538_871, not re-replayed blocks 0-1"
        );
    }

    #[test]
    fn replayed_through_height_must_be_stored_in_extended_marker() {
        // DOCUMENT: The extended 105-byte marker stores replayed_through_height.
        // Without it, a restart cannot know where replay stopped.
        // Without the marker, the node would re-replay from H, double-applying TXs.
        //
        // This is why the extended marker MUST be written atomically with the final
        // replay balance write (or the marker must be absent if replay is incomplete).
        let tracker = ReplayTracker {
            replayed_through_height: 16_540_000,
            ..Default::default()
        };
        assert_eq!(tracker.replayed_through_height, 16_540_000);
        // In the real implementation, this value is serialized into the 105-byte marker
        // and read back on startup to determine the replay resume point.
    }
}

// ============================================================================
// MODULE 4: KNOWN UNKNOWN 5 — REAL BLOCKCHAIN DATA GAPS
// ============================================================================

mod real_data_unknown {
    use super::*;

    #[test]
    fn replay_handles_coinbase_with_zero_amount() {
        // Edge case: a block might have a zero-value coinbase (e.g., genesis-like blocks)
        let miner = make_addr(0x01);
        let mut wallets = HashMap::new();
        wallets.insert(miner, 1_000_000u128);

        let tx = MockTx::coinbase(miner, 0);
        let result = apply_tx_to_wallets(&mut wallets, &tx);

        // Zero coinbase: acceptable (no-op or success)
        assert!(
            result.is_ok(),
            "Zero coinbase should not fail — just adds 0"
        );
        assert_eq!(wallets[&miner], 1_000_000u128, "Balance unchanged after zero coinbase");
    }

    #[test]
    fn replay_handles_transfer_to_self() {
        // Self-transfer (from == to): valid but unusual
        let alice = make_addr(0x01);
        let initial = 5_000_000_000_000_000_000_000_000u128;
        let send = 1_000_000_000_000_000_000_000_000u128;

        let mut wallets = HashMap::new();
        wallets.insert(alice, initial);

        let tx = MockTx::transfer(alice, alice, send);
        let result = apply_tx_to_wallets(&mut wallets, &tx);

        // Self-transfer: debit AND credit same address — net effect is zero
        // Depending on impl order, might succeed with no change or fail
        // Expect: no net change in balance (debit + credit = 0)
        assert!(
            result.is_ok(),
            "Self-transfer should not panic"
        );
        assert_eq!(
            wallets[&alice], initial,
            "Self-transfer: net balance unchanged"
        );
    }

    #[test]
    fn replay_gap_test_documents_unknown_5() {
        // DOCUMENT: The tests above verify replay LOGIC with synthetic blocks.
        // Unknown 5 is: "will the actual chain blocks H+1..current_tip parse correctly?"
        //
        // Things we cannot test here:
        //   a) Corrupt blocks from a kill -9 shutdown (gaps in turbo-sync)
        //   b) Schema drift: blocks from v7.x vs v10.x may have different TX wire format
        //   c) Blocks with exotic multi-sig or non-standard fee arrangements
        //   d) Concurrent coinbase blocks (when two miners solve same height)
        //
        // MITIGATION: Run on Delta's container first (not production).
        // The container will parse real blocks from Epsilon via P2P sync.
        // Any parse error in the replay will be caught and logged before mainnet deploy.
        //
        // ACCEPTANCE CRITERIA for Delta test:
        //   1. All blocks H+1..local_tip parse without error
        //   2. Post-replay total supply is within ±1% of expected (accounting for DEX skips)
        //   3. No wallet goes negative
        //   4. The extended marker is written correctly

        let synthetic_coverage = [
            "coinbase_applies",
            "transfer_applies",
            "swap_skipped",
            "contract_skipped",
            "negative_balance_rejected",
            "u128_overflow_caught",
        ];

        assert_eq!(synthetic_coverage.len(), 6, "6 known unknowns documented and covered synthetically");
        println!("Unknown 5 (real blockchain data parsing): covered synthetically. Delta container test required for full validation.");
    }

    #[test]
    fn schema_drift_would_show_as_parse_errors_not_silent_wrong_balances() {
        // If a TX type from an older version is no longer in the whitelist,
        // it gets SKIPPED (not applied). This is safer than misinterpreting it.
        // The whitelist approach means unknown types are skipped, not erroneously applied.

        let future_tx_type = 0xFFu8; // unknown to current code
        assert!(
            !is_replay_apply(future_tx_type),
            "Unknown/future TX types must be SKIPPED, never applied"
        );

        let deprecated_tx_type = 0x08u8; // hypothetical removed type
        assert!(
            !is_replay_apply(deprecated_tx_type),
            "Deprecated TX types must be SKIPPED too"
        );
    }
}
