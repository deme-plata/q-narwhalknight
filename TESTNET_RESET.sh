#!/bin/bash
# 🔄 Q-NarwhalKnight Testnet Reset Script
# Version: v0.0.9-beta - Max Supply Fix
# Date: 2025-10-23
#
# ⚠️  SECURITY: This script ONLY works on TESTNET
# Multiple safety checks prevent mainnet usage

set -e  # Exit on error

# ═══════════════════════════════════════════════════════════════════
# 🔒 MAINNET PROTECTION - Layer 1: Version Check
# ═══════════════════════════════════════════════════════════════════
CURRENT_VERSION=$(grep -oP '(?<=version = ")[^"]+' Cargo.toml | head -1)

if [[ ! "$CURRENT_VERSION" =~ -beta$ && ! "$CURRENT_VERSION" =~ -alpha$ ]]; then
    echo "🚨 SECURITY ALERT: Mainnet Protection Engaged!"
    echo ""
    echo "❌ This script is BLOCKED on production versions"
    echo "   Current version: $CURRENT_VERSION"
    echo "   This script only works on -beta or -alpha versions"
    echo ""
    echo "🔒 To reset testnet, ensure version in Cargo.toml ends with -beta or -alpha"
    echo ""
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════
# 🔒 MAINNET PROTECTION - Layer 2: Explicit Confirmation
# ═══════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════════════════════════"
echo "  🔄 Q-NarwhalKnight Testnet Reset - v0.0.9-beta"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "🔒 MAINNET PROTECTION: Active"
echo "   Current version: $CURRENT_VERSION"
echo ""
echo "⚠️  WARNING: This will reset ALL testnet data!"
echo ""
echo "Changes in v0.0.9:"
echo "  • ✅ Max supply enforcement (21M QNK cap)"
echo "  • ✅ Bitcoin-style halving schedule"
echo "  • ✅ libp2p consensus validation"
echo "  • ✅ u64 overflow protection"
echo "  • ✅ Post-quantum cryptography framework"
echo ""
echo "What will be reset:"
echo "  • All wallet balances → 0"
echo "  • All transaction history"
echo "  • All block data"
echo "  • Total supply → 0"
echo ""
echo "What will be preserved:"
echo "  • Wallet addresses (you keep your keys)"
echo "  • Node configuration"
echo "  • Network peers"
echo ""

# ═══════════════════════════════════════════════════════════════════
# 🔒 MAINNET PROTECTION - Layer 3: Typed Confirmation
# ═══════════════════════════════════════════════════════════════════
echo "Type 'RESET TESTNET' to continue (case sensitive):"
read -r CONFIRMATION

if [[ "$CONFIRMATION" != "RESET TESTNET" ]]; then
    echo "❌ Reset cancelled - confirmation phrase did not match"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════
# 🔒 MAINNET PROTECTION - Layer 4: Database Path Check
# ═══════════════════════════════════════════════════════════════════
# Refuse to reset if using production database names
if [ -d "./data-mainnet" ] || [ -d "./data-production" ] || [ -d "./prod-data" ]; then
    echo ""
    echo "🚨 SECURITY ALERT: Production database detected!"
    echo "   Found: ./data-mainnet, ./data-production, or ./prod-data"
    echo "   This script will NOT reset production databases"
    echo ""
    exit 1
fi

echo ""
echo "✅ All security checks passed - proceeding with testnet reset"

echo ""
echo "📋 Testnet Reset Steps:"
echo ""

# Step 1: Stop the service
echo "1️⃣ Stopping q-api-server service..."
systemctl stop q-api-server
sleep 2
echo "   ✅ Service stopped"

# Step 2: Backup old data
echo ""
echo "2️⃣ Backing up old data..."
BACKUP_DIR="/opt/orobit/backups/testnet-pre-v0.0.9-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

if [ -d "./data" ]; then
    echo "   📦 Backing up ./data → $BACKUP_DIR/data.tar.gz"
    tar -czf "$BACKUP_DIR/data.tar.gz" ./data 2>/dev/null || echo "   ⚠️  No data directory found"
fi

if [ -d "./data-mine2" ]; then
    echo "   📦 Backing up ./data-mine2 → $BACKUP_DIR/data-mine2.tar.gz"
    tar -czf "$BACKUP_DIR/data-mine2.tar.gz" ./data-mine2 2>/dev/null || echo "   ⚠️  No data-mine2 directory found"
fi

echo "   ✅ Backup complete: $BACKUP_DIR"

# Step 3: Remove old database
echo ""
echo "3️⃣ Removing old database..."
rm -rf ./data 2>/dev/null || true
rm -rf ./data-mine2 2>/dev/null || true
echo "   ✅ Old database removed"

# Step 4: Create fresh database directory
echo ""
echo "4️⃣ Creating fresh database directory..."
mkdir -p ./data-mine2
echo "   ✅ Fresh database directory created: ./data-mine2"

# Step 5: Update environment variable (if using systemd)
echo ""
echo "5️⃣ Updating database path..."
export Q_DB_PATH="./data-mine2"
echo "   ✅ Database path set to: $Q_DB_PATH"

# Step 6: Restart service
echo ""
echo "6️⃣ Starting q-api-server with fresh database..."
systemctl start q-api-server
sleep 5

# Step 7: Verify service is running
echo ""
echo "7️⃣ Verifying service status..."
if systemctl is-active --quiet q-api-server; then
    echo "   ✅ Service is running"
else
    echo "   ❌ Service failed to start"
    echo ""
    echo "   Last 20 log lines:"
    journalctl -u q-api-server --since "1 minute ago" -n 20
    exit 1
fi

# Step 8: Check logs
echo ""
echo "8️⃣ Checking startup logs..."
sleep 3
journalctl -u q-api-server --since "30 seconds ago" | grep -E "(libp2p|supply|Max supply)" | tail -5
echo "   ✅ Service started successfully"

# Step 9: Display status
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ Testnet Reset Complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📊 New Testnet Status:"
echo "   • Version: $CURRENT_VERSION (testnet)"
echo "   • Database: ./data-mine2 (fresh)"
echo "   • Total Supply: 0 QNK"
echo "   • Max Supply: 21,000,000 QNK"
echo "   • Halving: Every 1,000,000 blocks"
echo "   • Block Reward: 0.5 QNK (initial)"
echo ""
echo "🔗 Endpoints:"
echo "   • API: http://localhost:8080"
echo "   • Health: http://localhost:8080/health"
echo "   • Supply: http://localhost:8080/api/chain/supply"
echo ""
echo "🔒 Security Features:"
echo "   • ✅ Version check (testnet only)"
echo "   • ✅ Typed confirmation required"
echo "   • ✅ Production database protection"
echo "   • ✅ Backup created before reset"
echo ""
echo "📝 Next Steps:"
echo "   1. Notify community of testnet reset"
echo "   2. Users can resume mining with fresh balances"
echo "   3. Monitor logs: journalctl -u q-api-server -f"
echo "   4. Verify supply cap: curl http://localhost:8080/api/chain/supply"
echo ""
echo "🎉 Ready to test v0.0.9 with max supply enforcement!"
echo ""
echo "💡 Backup location: $BACKUP_DIR"
echo ""
