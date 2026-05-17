# Mainnet-Safe Upgrade System - Technical Review

**Version**: v1.4.1-beta
**Date**: December 2025
**Status**: Implementation Ready

## Executive Summary

This document describes Q-NarwhalKnight's block-height activated upgrade system, which allows safe software updates on mainnet without coordinated restarts or data loss risk.

**Key Principle**: Old blocks must always validate the same way they did when created.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [The Upgrade Framework](#2-the-upgrade-framework)
3. [Adding a New Upgrade](#3-adding-a-new-upgrade)
4. [Code Patterns](#4-code-patterns)
5. [Deployment Process](#5-deployment-process)
6. [Testing Requirements](#6-testing-requirements)
7. [Emergency Procedures](#7-emergency-procedures)
8. [Examples](#8-examples)

---

## 1. Architecture Overview

### Problem: Testnet vs Mainnet

```
TESTNET (current approach):
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Compile │────▶│ Restart │────▶│ New code│
│         │     │ service │     │ runs NOW│
└─────────┘     └─────────┘     └─────────┘
                                     │
                                     ▼
                              If broken: Reset to new phase
                              (acceptable on testnet)

MAINNET (required approach):
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ Compile │────▶│ Deploy  │────▶│ Wait for│────▶│ New code│
│         │     │ binary  │     │ height X│     │ activates│
└─────────┘     └─────────┘     └─────────┘     └─────────┘
                                     │
                                     ▼
                              If broken: Delay activation
                              (no data loss ever)
```

### Solution: Height-Activated Upgrades

All nodes run the same code, but new features only activate at a predetermined block height:

```rust
fn validate_block(&self, block: &Block) -> Result<()> {
    if block.height >= UPGRADE_HEIGHT {
        self.validate_v2(block)  // New rules
    } else {
        self.validate_v1(block)  // Old rules (immutable)
    }
}
```

### Benefits

| Benefit | Description |
|---------|-------------|
| **No coordination** | Nodes upgrade at their own pace |
| **Atomic activation** | All nodes activate at same block |
| **Safe rollback** | Run old binary before activation height |
| **History preserved** | Old blocks always validate with old rules |
| **No downtime** | No network pause for upgrades |

---

## 2. The Upgrade Framework

### File Structure

```
crates/q-types/src/upgrades.rs    # Upgrade definitions
scripts/safe-deploy.sh            # Deployment automation
```

### Core Types

```rust
/// Network upgrade definition
pub struct NetworkUpgrade {
    pub name: &'static str,           // Unique identifier
    pub activation_height: u64,       // When it activates
    pub description: &'static str,    // What it does
}

/// Upgrade manager
pub struct UpgradeManager {
    current_height: AtomicU64,        // Current chain height
    is_mainnet: bool,                 // Network type
}
```

### Defined Upgrades

```rust
// crates/q-types/src/upgrades.rs

pub mod upgrades {
    // Genesis - always active
    pub const GENESIS: NetworkUpgrade = NetworkUpgrade {
        name: "genesis",
        activation_height: 0,
        description: "Network launch",
    };

    // Current testnet phase
    pub const PHASE_16: NetworkUpgrade = NetworkUpgrade {
        name: "phase_16",
        activation_height: 0,
        description: "Testnet Phase 16",
    };

    // Future upgrade (not yet activated)
    pub const PQ_SIGNATURES_REQUIRED: NetworkUpgrade = NetworkUpgrade {
        name: "pq_signatures_required",
        activation_height: u64::MAX,  // Set when ready
        description: "Require Dilithium signatures",
    };
}
```

### Usage

```rust
use q_types::upgrades::{UpgradeManager, upgrades};

let manager = UpgradeManager::new(is_mainnet);
manager.set_height(current_block_height);

// Check if upgrade is active
if manager.is_active(&upgrades::PQ_SIGNATURES_REQUIRED) {
    // Use new logic
} else {
    // Use old logic
}
```

---

## 3. Adding a New Upgrade

### Step-by-Step Process

#### Step 1: Define the Upgrade

```rust
// In crates/q-types/src/upgrades.rs

pub const MY_NEW_FEATURE: NetworkUpgrade = NetworkUpgrade {
    name: "my_new_feature",
    activation_height: u64::MAX,  // Start with MAX (not activated)
    description: "Description of what this upgrade does",
};

// Add to the all_upgrades array in upgrades_between()
```

#### Step 2: Implement Height-Gated Code

```rust
// In your validation/consensus code

use q_types::upgrades::upgrades;

fn my_validation_function(&self, block: &Block) -> Result<()> {
    if block.height >= upgrades::MY_NEW_FEATURE.activation_height {
        // NEW behavior
        self.new_validation_logic(block)?;
    } else {
        // OLD behavior (must match original implementation exactly!)
        self.old_validation_logic(block)?;
    }
    Ok(())
}
```

#### Step 3: Test Both Paths

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_old_blocks_still_validate() {
        let old_block = create_block_at_height(100);  // Before activation
        assert!(validate_block(&old_block).is_ok());
    }

    #[test]
    fn test_new_blocks_use_new_rules() {
        let new_block = create_block_at_height(1_000_000);  // After activation
        // This should use new validation rules
        assert!(validate_block(&new_block).is_ok());
    }
}
```

#### Step 4: Set Activation Height

Once tested and ready for mainnet:

```rust
// Change from:
pub const MY_NEW_FEATURE: NetworkUpgrade = NetworkUpgrade {
    activation_height: u64::MAX,
    // ...
};

// To specific height (e.g., current + 20,000 = ~2 weeks):
pub const MY_NEW_FEATURE: NetworkUpgrade = NetworkUpgrade {
    activation_height: 500_000,  // Activates at block 500,000
    // ...
};
```

---

## 4. Code Patterns

### Pattern 1: Simple Validation Change

```rust
// ✅ CORRECT
fn check_transaction_size(&self, tx: &Transaction, block_height: u64) -> Result<()> {
    let max_size = if block_height >= upgrades::LARGER_TXS.activation_height {
        1_000_000  // 1MB after upgrade
    } else {
        100_000    // 100KB before upgrade
    };

    if tx.size() > max_size {
        return Err(Error::TransactionTooLarge);
    }
    Ok(())
}
```

### Pattern 2: Signature Verification

```rust
// ✅ CORRECT
fn verify_block_signature(&self, block: &Block) -> Result<()> {
    if block.height >= upgrades::PQ_SIGNATURES_REQUIRED.activation_height {
        // New: Require post-quantum signature
        self.verify_dilithium_signature(block)?;
    } else {
        // Old: Accept Ed25519 (for historical blocks)
        self.verify_ed25519_signature(block)?;
    }
    Ok(())
}
```

### Pattern 3: Reward Calculation

```rust
// ✅ CORRECT
fn calculate_block_reward(&self, block_height: u64) -> u64 {
    if block_height >= upgrades::HALVING_V2.activation_height {
        // New halving schedule
        self.calculate_reward_v2(block_height)
    } else {
        // Original halving schedule
        self.calculate_reward_v1(block_height)
    }
}
```

### Pattern 4: Using the Macro

```rust
use q_types::if_upgrade_active;

fn process_block(&self, block: &Block) -> Result<()> {
    if_upgrade_active!(self.upgrade_manager, PQ_SIGNATURES_REQUIRED, {
        // New code path
        self.require_pq_signatures(block)?;
    } else {
        // Old code path
        self.allow_classical_signatures(block)?;
    });

    Ok(())
}
```

### Anti-Patterns (WRONG)

```rust
// ❌ WRONG: No height check - breaks old blocks
fn verify_signature(&self, block: &Block) -> Result<()> {
    self.verify_dilithium_signature(block)?;  // Old blocks didn't have this!
    Ok(())
}

// ❌ WRONG: Height check on current time, not block height
fn verify_signature(&self, block: &Block) -> Result<()> {
    if SystemTime::now() > UPGRADE_DATE {  // Wrong! Use block.height
        self.verify_dilithium_signature(block)?;
    }
    Ok(())
}

// ❌ WRONG: Modifying old validation logic
fn verify_signature(&self, block: &Block) -> Result<()> {
    if block.height >= UPGRADE_HEIGHT {
        self.verify_dilithium_signature(block)?;
    } else {
        // This was changed from the original! Must match exactly.
        self.verify_ed25519_signature_v2(block)?;  // WRONG
    }
    Ok(())
}
```

---

## 5. Deployment Process

### Infrastructure

```
┌─────────────────────────────────────────────────────────────┐
│                    TWO-SERVER DEPLOYMENT                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  SERVER ALPHA                      SERVER BETA               │
│  (161.35.219.10)                   (185.182.185.227)         │
│  ┌─────────────────┐               ┌─────────────────┐       │
│  │ Test/Staging    │               │ Production      │       │
│  │ Docker tests    │               │ Bootstrap node  │       │
│  │ Canary deploy   │               │ User-facing     │       │
│  └─────────────────┘               └─────────────────┘       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Deployment Script Usage

```bash
# Full automated pipeline
./scripts/safe-deploy.sh full

# Or step by step:
./scripts/safe-deploy.sh build        # Compile + basic tests
./scripts/safe-deploy.sh test-docker  # Docker canary (fresh sync)
./scripts/safe-deploy.sh deploy-beta  # Production + 5-min soak test

# If issues:
./scripts/safe-deploy.sh rollback     # Instant rollback

# Check status:
./scripts/safe-deploy.sh status
```

### Deployment Timeline

```
Week 1: Development
├── Day 1-3: Write code with height checks
├── Day 4-5: Unit tests + integration tests
└── Day 6-7: Docker canary testing

Week 2: Staging
├── Day 1-3: Server Alpha 48-hour soak test
├── Day 4: Review logs for errors
└── Day 5-7: Fix any issues found

Week 3: Announcement
├── Day 1: Announce upgrade on Discord/Twitter
├── Day 2: Set activation height (current + 20,000)
└── Day 3-7: Deploy to Server Beta, users upgrade

Week 4+: Activation
├── Monitor activation block
├── Watch for consensus splits
└── Post-activation monitoring
```

### Pre-Deployment Checklist

```markdown
## Before Deploying to Production

### Code Quality
- [ ] All height checks use block.height, not timestamps
- [ ] Old validation logic is UNCHANGED
- [ ] New logic only runs at/after activation height
- [ ] Unit tests pass for both old and new paths
- [ ] Integration tests pass

### Testing
- [ ] Docker fresh-sync test passes
- [ ] Server Alpha ran 24+ hours without crashes
- [ ] No "CORRUPTION", "panic", or "CRITICAL" in logs
- [ ] Old blocks still validate correctly
- [ ] P2P sync still works

### Documentation
- [ ] Upgrade added to upgrades.rs
- [ ] Activation height documented
- [ ] User announcement prepared
- [ ] Rollback procedure tested
```

---

## 6. Testing Requirements

### Unit Tests

```rust
#[cfg(test)]
mod upgrade_tests {
    use super::*;

    #[test]
    fn test_upgrade_not_active_before_height() {
        let manager = UpgradeManager::new(false);
        manager.set_height(499_999);

        assert!(!manager.is_active_at_height(
            &upgrades::MY_UPGRADE,
            499_999
        ));
    }

    #[test]
    fn test_upgrade_active_at_height() {
        let manager = UpgradeManager::new(false);

        assert!(manager.is_active_at_height(
            &upgrades::MY_UPGRADE,
            500_000
        ));
    }

    #[test]
    fn test_historical_blocks_validate() {
        // Create a block from before the upgrade
        let old_block = Block {
            height: 100,
            signature: old_style_signature(),
            // ...
        };

        // Must still validate even after upgrade is active
        assert!(validate_block(&old_block).is_ok());
    }
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_sync_across_upgrade_boundary() {
    // Start node at height 0
    let node = TestNode::new().await;

    // Sync to before upgrade
    node.sync_to_height(499_999).await;
    assert!(node.is_healthy());

    // Sync across upgrade boundary
    node.sync_to_height(500_100).await;
    assert!(node.is_healthy());

    // Verify all blocks still valid
    for height in 0..=500_100 {
        assert!(node.validate_block_at_height(height).await.is_ok());
    }
}
```

### Docker Canary Test

```bash
#!/bin/bash
# Automated by safe-deploy.sh test-docker

# 1. Start fresh container
docker run -d --name test-node ...

# 2. Wait for sync
sleep 60

# 3. Check for errors
errors=$(docker logs test-node 2>&1 | grep -c "CRITICAL\|panic")
if [ "$errors" -gt 0 ]; then
    echo "FAILED: Found critical errors"
    exit 1
fi

# 4. Check sync progress
height=$(docker logs test-node 2>&1 | grep "height" | tail -1)
echo "Synced to: $height"

# 5. Cleanup
docker rm -f test-node
```

---

## 7. Emergency Procedures

### Scenario 1: Bug Found Before Activation

```
Timeline:
- Day 0: Deploy new binary (activation at height 500,000)
- Day 5: Bug discovered (current height: 490,000)
- Day 5: 10,000 blocks until activation = ~1.5 days

Action:
1. Announce on Discord: "DO NOT UPGRADE - bug found"
2. Users running old binary are safe
3. Users running new binary are safe (bug not triggered yet)
4. Fix bug, deploy new binary with later activation height
```

### Scenario 2: Bug Found After Activation

```
Timeline:
- Activation happened at height 500,000
- Bug discovered at height 500,500
- Network is split or producing invalid blocks

Action:
1. EMERGENCY: Announce rollback
2. Users downgrade to pre-upgrade binary
3. Nodes on old binary will reject blocks >= 500,000 with new rules
4. This creates a HARD FORK - requires coordination
5. Community decides: fix forward or rollback

Prevention:
- Extended testing before activation
- Conservative activation heights (2+ weeks out)
- Gradual rollout to small percentage first
```

### Rollback Command

```bash
# Instant rollback to last known-good binary
./scripts/safe-deploy.sh rollback

# Manual rollback if script fails
systemctl stop q-api-server
cp /opt/orobit/shared/q-narwhalknight/backups/q-api-server-rollback-* \
   /opt/orobit/shared/q-narwhalknight/target/release/q-api-server
systemctl start q-api-server
```

---

## 8. Examples

### Example 1: Increasing Transaction Size Limit

**Requirement**: Increase max transaction size from 100KB to 1MB at block 600,000.

```rust
// Step 1: Define upgrade in upgrades.rs
pub const LARGER_TRANSACTIONS: NetworkUpgrade = NetworkUpgrade {
    name: "larger_transactions",
    activation_height: 600_000,
    description: "Increase max tx size to 1MB",
};

// Step 2: Implement height-gated logic
fn validate_transaction_size(&self, tx: &Transaction, block_height: u64) -> Result<()> {
    let max_size = if block_height >= upgrades::LARGER_TRANSACTIONS.activation_height {
        1_000_000  // 1MB
    } else {
        100_000    // 100KB (original limit)
    };

    if tx.serialized_size() > max_size {
        return Err(ValidationError::TransactionTooLarge {
            size: tx.serialized_size(),
            max: max_size,
        });
    }
    Ok(())
}
```

### Example 2: Requiring Post-Quantum Signatures

**Requirement**: Require Dilithium signatures starting block 1,000,000.

```rust
// Step 1: Define upgrade
pub const REQUIRE_PQ_SIGNATURES: NetworkUpgrade = NetworkUpgrade {
    name: "require_pq_signatures",
    activation_height: 1_000_000,
    description: "Require Dilithium5 signatures on all transactions",
};

// Step 2: Implement in signature verification
fn verify_transaction_signature(&self, tx: &Transaction, block_height: u64) -> Result<()> {
    if block_height >= upgrades::REQUIRE_PQ_SIGNATURES.activation_height {
        // After upgrade: Only Dilithium accepted
        match &tx.signature {
            Signature::Dilithium(sig) => {
                verify_dilithium(tx.hash(), sig, &tx.sender_pubkey)?;
            }
            _ => {
                return Err(ValidationError::InvalidSignatureType {
                    expected: "Dilithium",
                    got: tx.signature.type_name(),
                });
            }
        }
    } else {
        // Before upgrade: Ed25519 or Dilithium both valid
        match &tx.signature {
            Signature::Ed25519(sig) => {
                verify_ed25519(tx.hash(), sig, &tx.sender_pubkey)?;
            }
            Signature::Dilithium(sig) => {
                verify_dilithium(tx.hash(), sig, &tx.sender_pubkey)?;
            }
        }
    }
    Ok(())
}
```

### Example 3: Changing Block Reward

**Requirement**: Implement new emission curve at block 800,000.

```rust
// Step 1: Define upgrade
pub const NEW_EMISSION_CURVE: NetworkUpgrade = NetworkUpgrade {
    name: "new_emission_curve",
    activation_height: 800_000,
    description: "Switch to logarithmic emission curve",
};

// Step 2: Implement
fn calculate_block_reward(&self, block_height: u64) -> u64 {
    if block_height >= upgrades::NEW_EMISSION_CURVE.activation_height {
        // New: Logarithmic curve
        let base_reward = 100_000_000; // 1 QNK in smallest units
        let decay = (block_height as f64 / 100_000.0).ln();
        (base_reward as f64 / (1.0 + decay)) as u64
    } else {
        // Original: Linear halving
        let halvings = block_height / 210_000;
        100_000_000 >> halvings
    }
}
```

---

## Appendix A: Quick Reference

### The Two Questions

```
1. Does this change validation or consensus?
   NO  → Proceed normally
   YES → Continue to #2

2. Is it wrapped in a height check?
   YES → Safe to proceed
   NO  → STOP and wrap it first
```

### Key Files

| File | Purpose |
|------|---------|
| `crates/q-types/src/upgrades.rs` | Upgrade definitions |
| `scripts/safe-deploy.sh` | Deployment automation |
| `CLAUDE.md` | Development guidelines |
| `docs/MAINNET_UPGRADE_TECHNICAL_REVIEW.md` | This document |

### Commands

```bash
# Full deployment pipeline
./scripts/safe-deploy.sh full

# Check deployment status
./scripts/safe-deploy.sh status

# Emergency rollback
./scripts/safe-deploy.sh rollback
```

---

## Appendix B: Changelog

| Version | Date | Changes |
|---------|------|---------|
| v1.4.1-beta | Dec 2025 | Initial upgrade framework |

---

*This document is part of the Q-NarwhalKnight mainnet preparation. For questions, refer to CLAUDE.md or ask the development team.*
