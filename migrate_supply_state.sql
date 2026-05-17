-- 🔒 Q-NarwhalKnight Max Supply Enforcement Migration
-- Version: 1.0
-- Date: 2025-10-23
-- Purpose: Fix unlimited minting vulnerability and cap affected balances

-- ============================================================================
-- STEP 1: Create supply consensus tracking table
-- ============================================================================

CREATE TABLE IF NOT EXISTS chain_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- Singleton row
    total_minted_supply INTEGER NOT NULL DEFAULT 0,
    last_halving_block INTEGER NOT NULL DEFAULT 0,
    last_consensus_timestamp INTEGER NOT NULL,
    consensus_node_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chain_state_supply ON chain_state(total_minted_supply);
CREATE INDEX IF NOT EXISTS idx_chain_state_timestamp ON chain_state(last_consensus_timestamp);

-- ============================================================================
-- STEP 2: Calculate current total supply from wallet balances
-- ============================================================================

-- Insert or update chain state with current supply
-- NOTE: Replace this with actual calculated supply from wallet_balances table
INSERT OR REPLACE INTO chain_state (
    id,
    total_minted_supply,
    last_halving_block,
    last_consensus_timestamp,
    consensus_node_count
) VALUES (
    1,
    (SELECT COALESCE(SUM(balance), 0) FROM wallet_balances),  -- Sum all balances
    0,  -- Starting halving count
    strftime('%s', 'now'),  -- Current Unix timestamp
    0   -- Initial peer count
);

-- ============================================================================
-- STEP 3: Cap affected user balance (184T QNK → 1M QNK)
-- ============================================================================

-- Find wallets with astronomical balances (> 21M QNK max supply)
-- These are clearly erroneous due to the bug
SELECT
    address,
    balance,
    balance / 100000000.0 AS balance_qnk,
    'NEEDS CAPPING' AS status
FROM wallet_balances
WHERE balance > 21000000000000000  -- More than 21M QNK
ORDER BY balance DESC;

-- Cap affected wallets to 1M QNK (100B atomic units) as compensation
-- This is generous considering the bug allowed unlimited minting
UPDATE wallet_balances
SET balance = 100000000000000,  -- 1,000,000 QNK in atomic units
    updated_at = CURRENT_TIMESTAMP
WHERE balance > 21000000000000000;

-- Log the capping action for audit trail
CREATE TABLE IF NOT EXISTS balance_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    address BLOB NOT NULL,
    old_balance INTEGER NOT NULL,
    new_balance INTEGER NOT NULL,
    reason TEXT NOT NULL,
    capped_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO balance_audit_log (address, old_balance, new_balance, reason)
SELECT
    address,
    balance AS old_balance,
    100000000000000 AS new_balance,
    'Max supply enforcement migration - capped due to unlimited minting bug'
FROM wallet_balances
WHERE balance > 21000000000000000;

-- ============================================================================
-- STEP 4: Recalculate total supply after capping
-- ============================================================================

UPDATE chain_state
SET total_minted_supply = (SELECT COALESCE(SUM(balance), 0) FROM wallet_balances),
    updated_at = CURRENT_TIMESTAMP
WHERE id = 1;

-- ============================================================================
-- STEP 5: Validation queries
-- ============================================================================

-- Verify total supply is within bounds
SELECT
    'Total Supply Validation' AS check_name,
    total_minted_supply,
    total_minted_supply / 100000000.0 AS supply_qnk,
    CASE
        WHEN total_minted_supply <= 21000000000000000 THEN '✅ PASS'
        ELSE '❌ FAIL - Exceeds 21M QNK'
    END AS status
FROM chain_state
WHERE id = 1;

-- Verify no wallet exceeds max supply
SELECT
    'Individual Balance Validation' AS check_name,
    COUNT(*) AS wallets_exceeding_max,
    CASE
        WHEN COUNT(*) = 0 THEN '✅ PASS'
        ELSE '❌ FAIL - ' || COUNT(*) || ' wallets exceed 21M QNK'
    END AS status
FROM wallet_balances
WHERE balance > 21000000000000000;

-- Show top 10 richest wallets for verification
SELECT
    'Top 10 Richest Wallets' AS report_name,
    hex(address) AS address_hex,
    balance,
    balance / 100000000.0 AS balance_qnk,
    balance / (SELECT total_minted_supply FROM chain_state) * 100.0 AS supply_percentage
FROM wallet_balances
ORDER BY balance DESC
LIMIT 10;

-- ============================================================================
-- STEP 6: Create supply update triggers
-- ============================================================================

-- Trigger to update chain_state.updated_at on supply change
CREATE TRIGGER IF NOT EXISTS trg_chain_state_updated
AFTER UPDATE ON chain_state
FOR EACH ROW
BEGIN
    UPDATE chain_state
    SET updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;
END;

-- Audit trigger for balance changes
CREATE TRIGGER IF NOT EXISTS trg_balance_audit
AFTER UPDATE OF balance ON wallet_balances
FOR EACH ROW
WHEN NEW.balance != OLD.balance
BEGIN
    INSERT INTO balance_audit_log (address, old_balance, new_balance, reason)
    VALUES (
        NEW.address,
        OLD.balance,
        NEW.balance,
        'Balance update - mining or transaction'
    );
END;

-- ============================================================================
-- STEP 7: Create monitoring views
-- ============================================================================

CREATE VIEW IF NOT EXISTS v_supply_status AS
SELECT
    total_minted_supply / 100000000.0 AS current_supply_qnk,
    21000000.0 AS max_supply_qnk,
    (total_minted_supply / 21000000000000000.0 * 100.0) AS supply_percentage,
    last_halving_block,
    consensus_node_count,
    datetime(last_consensus_timestamp, 'unixepoch') AS last_consensus_time,
    datetime(updated_at) AS last_updated
FROM chain_state
WHERE id = 1;

CREATE VIEW IF NOT EXISTS v_wallet_distribution AS
SELECT
    CASE
        WHEN balance < 100000000 THEN '< 1 QNK'
        WHEN balance < 1000000000 THEN '1-10 QNK'
        WHEN balance < 10000000000 THEN '10-100 QNK'
        WHEN balance < 100000000000 THEN '100-1K QNK'
        WHEN balance < 1000000000000 THEN '1K-10K QNK'
        WHEN balance < 10000000000000 THEN '10K-100K QNK'
        WHEN balance < 100000000000000 THEN '100K-1M QNK'
        ELSE '> 1M QNK'
    END AS balance_range,
    COUNT(*) AS wallet_count,
    SUM(balance) / 100000000.0 AS total_qnk_in_range,
    (SUM(balance) / (SELECT total_minted_supply FROM chain_state) * 100.0) AS percentage_of_supply
FROM wallet_balances
GROUP BY balance_range
ORDER BY MIN(balance);

-- ============================================================================
-- STEP 8: Migration verification report
-- ============================================================================

-- Final migration summary
SELECT '🔒 MIGRATION SUMMARY' AS section;

SELECT
    'Current Total Supply' AS metric,
    total_minted_supply / 100000000.0 || ' QNK' AS value,
    '21,000,000 QNK max' AS limit,
    ROUND((total_minted_supply / 21000000000000000.0 * 100.0), 2) || '% of max' AS status
FROM chain_state WHERE id = 1;

SELECT
    'Affected Wallets Capped' AS metric,
    COUNT(*) || ' wallets' AS value,
    'Capped to 1M QNK each' AS action,
    'Compensation for bug' AS reason
FROM balance_audit_log
WHERE reason LIKE '%unlimited minting%';

SELECT
    'Total Wallets' AS metric,
    COUNT(*) || ' addresses' AS value,
    'Active balances' AS status,
    '' AS notes
FROM wallet_balances
WHERE balance > 0;

SELECT
    'Smallest Balance' AS metric,
    MIN(balance) / 100000000.0 || ' QNK' AS value,
    'Minimum mined' AS status,
    '' AS notes
FROM wallet_balances
WHERE balance > 0;

SELECT
    'Largest Balance' AS metric,
    MAX(balance) / 100000000.0 || ' QNK' AS value,
    'After capping' AS status,
    CASE
        WHEN MAX(balance) <= 100000000000000 THEN '✅ Within 1M QNK cap'
        ELSE '❌ Still exceeds cap'
    END AS verification
FROM wallet_balances;

-- ============================================================================
-- STEP 9: Post-migration recommendations
-- ============================================================================

-- Check if any balances still look suspicious
SELECT
    '⚠️ SUSPICIOUS BALANCES' AS warning,
    COUNT(*) AS wallet_count
FROM wallet_balances
WHERE balance > 10000000000000  -- > 100K QNK (might be legitimate, but worth checking)
  AND balance <= 100000000000000;  -- <= 1M QNK

-- Export data for manual review if needed
SELECT
    '📊 EXPORT FOR REVIEW' AS note,
    'Run this query to export all capped wallets for community disclosure:' AS instruction,
    'SELECT hex(address), old_balance/1e8 AS old_qnk, new_balance/1e8 AS new_qnk, capped_at FROM balance_audit_log WHERE reason LIKE "%unlimited%";' AS query;

-- ============================================================================
-- STEP 10: Enable max supply enforcement
-- ============================================================================

-- Create a flag table to enable/disable supply enforcement
CREATE TABLE IF NOT EXISTS feature_flags (
    feature_name TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 0,  -- 0 = disabled, 1 = enabled
    enabled_at TIMESTAMP,
    notes TEXT
);

INSERT OR REPLACE INTO feature_flags (feature_name, enabled, enabled_at, notes)
VALUES (
    'max_supply_enforcement',
    1,
    CURRENT_TIMESTAMP,
    'Enabled after migration v1.0 - Prevents unlimited minting'
);

SELECT
    '🎉 MIGRATION COMPLETE' AS status,
    'Max supply enforcement is now active' AS message,
    'Next: Restart q-api-server to load new supply state' AS next_step;

-- ============================================================================
-- ROLLBACK SCRIPT (Keep for emergency)
-- ============================================================================

-- To rollback this migration (DANGEROUS - only use if absolutely necessary):
-- 1. Restore database backup from before migration
-- 2. Update codebase to previous commit
-- 3. Restart services
--
-- Emergency rollback commands (commented out for safety):
-- -- DROP TABLE IF EXISTS chain_state;
-- -- DROP TABLE IF EXISTS balance_audit_log;
-- -- DROP VIEW IF EXISTS v_supply_status;
-- -- DROP VIEW IF EXISTS v_wallet_distribution;
-- -- DELETE FROM feature_flags WHERE feature_name = 'max_supply_enforcement';
