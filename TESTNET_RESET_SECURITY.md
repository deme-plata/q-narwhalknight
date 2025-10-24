# 🔒 Testnet Reset Script - Security Documentation

## Overview

The `TESTNET_RESET.sh` script is designed to safely reset the Q-NarwhalKnight testnet while **preventing accidental or malicious use on mainnet**.

## 🛡️ Multi-Layer Security Protection

### Layer 1: Version Check
```bash
# Automatically checks Cargo.toml version
# BLOCKS execution if version doesn't end with -beta or -alpha
```

**Protection:**
- Script reads `version = "X.Y.Z-beta"` from `Cargo.toml`
- Exits with error if version is production (e.g., `0.1.0` without `-beta`)
- Only runs on development/testing versions

**Attack Resistance:**
- Hacker cannot run on mainnet unless they modify `Cargo.toml`
- Modification of `Cargo.toml` would break the build and be immediately obvious
- Community would notice version downgrade in git commits

### Layer 2: Explicit Typed Confirmation
```bash
# User must type exact phrase: "RESET TESTNET"
# Case sensitive - prevents accidental execution
```

**Protection:**
- Simple "y/n" prompts are dangerous (easy to type 'y' by mistake)
- Requires full phrase typing: `RESET TESTNET`
- Case sensitive - `reset testnet` won't work

**Attack Resistance:**
- Cannot be automated without modifying the script
- Prevents accidental execution from muscle memory
- Makes intent crystal clear

### Layer 3: Database Path Detection
```bash
# Refuses to run if production database folders exist
# Checks for: data-mainnet, data-production, prod-data
```

**Protection:**
- Script scans for production database directory names
- Exits immediately if production paths detected
- Assumes mainnet uses proper naming conventions

**Attack Resistance:**
- Even if version check bypassed, production paths are protected
- Standard deployment conventions create natural safety boundary
- Multiple naming patterns checked

### Layer 4: Backup Before Reset
```bash
# Creates timestamped backup before any deletion
# Stored in: /opt/orobit/backups/testnet-pre-v0.0.9-YYYYMMDD_HHMMSS/
```

**Protection:**
- All data backed up before reset
- Timestamped backups prevent overwriting
- Recovery possible if script misused

**Attack Resistance:**
- Even if malicious reset occurs, data can be restored
- Backup path displayed to user
- Creates audit trail of resets

## 🚨 What if Someone Tries to Hack It?

### Scenario 1: Direct Script Modification
**Attack:** Hacker removes security checks from script

**Defense:**
- Script is in git repository - changes are tracked
- Community would see malicious commit
- Official distributions include checksums
- systemd service doesn't use the script - manual run only

### Scenario 2: Version Spoofing
**Attack:** Change version to "-beta" on mainnet

**Defense:**
- Changing version in `Cargo.toml` requires rebuild
- Build would fail if binaries don't match version
- Git tags and releases prevent version confusion
- Community knows official mainnet version number

### Scenario 3: Social Engineering
**Attack:** Trick user into running script on mainnet

**Defense:**
- Database path check prevents execution
- Mainnet uses different database folders
- User must type full confirmation phrase
- Multiple warnings displayed before execution

### Scenario 4: Automated Execution
**Attack:** Run script via cron or automation

**Defense:**
- Requires interactive terminal input
- User must type "RESET TESTNET" phrase
- Cannot be automated without script modification
- Systemd service doesn't call this script

## ✅ Safe Usage Guidelines

### For Testnet Operators:
```bash
# 1. Ensure you're on testnet (check version)
grep version Cargo.toml
# Should show: version = "0.0.9-beta" or similar

# 2. Verify database is testnet
ls -la | grep data
# Should NOT show: data-mainnet, data-production, prod-data

# 3. Run the script
./TESTNET_RESET.sh

# 4. Type the confirmation phrase when prompted
RESET TESTNET

# 5. Verify reset success
curl http://localhost:8080/api/chain/supply
# Should show total_supply: 0
```

### For Mainnet Operators:
```bash
# DO NOT use this script on mainnet
# Mainnet should NEVER be reset
# If you need to reset mainnet, you need a new genesis block and community consensus
```

## 📊 Security Trade-offs

### Why Not Use Stronger Protection?
**Could we:**
- Require GPG signatures? → Too complex for testnet
- Use hardware tokens? → Overkill for development
- Require 2FA? → Not available in bash scripts
- Network checks? → Can be circumvented

**Design Choice:**
Balance between:
- ✅ Easy to use for legitimate testnet resets
- ✅ Hard to misuse on mainnet
- ✅ Transparent security (no hidden checks)
- ✅ Open source and auditable

### Known Limitations
1. **Root user bypass:** Root can modify anything - mitigated by multi-layer checks
2. **Script modification:** Can edit security checks - mitigated by git tracking
3. **Social engineering:** User could be tricked - mitigated by clear warnings

## 🎯 Mainnet Safety Recommendations

### When Launching Mainnet:

1. **Version Management:**
   - Remove `-beta` suffix from version
   - Use semantic versioning (e.g., `1.0.0`)
   - Create git tag for release

2. **Database Naming:**
   - Use `data-mainnet` or `data-production` folder
   - Configure systemd: `Q_DB_PATH=/opt/q/mainnet-data`
   - Document mainnet paths in deployment guide

3. **Script Handling:**
   - Include `TESTNET_RESET.sh` in releases (for documentation)
   - Script will automatically refuse to run
   - Consider adding warning banner in mainnet README

4. **Backup Strategy:**
   - Regular mainnet backups via proper tools
   - Never rely on this script for mainnet
   - Use professional backup solutions

## 🔐 Security Audit Checklist

- [x] Version check prevents mainnet execution
- [x] Typed confirmation prevents accidents
- [x] Database path check adds redundancy
- [x] Backup created before reset
- [x] Multiple warnings displayed
- [x] No silent failures
- [x] Git tracking of script changes
- [x] Clear security documentation
- [x] Transparent implementation
- [x] Community review possible

## 📝 Changelog

### v0.0.9-beta (2025-10-23)
- Initial secure testnet reset script
- 4-layer security protection
- Comprehensive documentation
- Mainnet protection verified

---

**Security Note:** This script is designed for testnet safety, not mainnet. Mainnet should NEVER be reset via scripts. Any mainnet reset requires community consensus and a new genesis block.

**Questions?** Review the script source code - all security checks are transparent and auditable.
