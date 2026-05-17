#!/bin/bash
# Q-NarwhalKnight Encryption Error Fix Script
# Fixes: "AES-GCM decryption failed (wrong passphrase?): aead::Error"

set -e

echo "============================================"
echo "Q-NarwhalKnight Encryption Error Fix"
echo "v1.0.44-beta"
echo "============================================"
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "❌ This script must be run as root (use sudo)"
   exit 1
fi

# Stop the service
echo "1️⃣ Stopping q-api-server service..."
systemctl stop q-api-server 2>/dev/null || echo "   Service not running or not installed"

# Backup old encryption keys (if they exist)
echo ""
echo "2️⃣ Backing up old encryption keys..."
if [ -f /opt/encryption.keys ]; then
    mv /opt/encryption.keys /opt/encryption.keys.backup.$(date +%s)
    echo "   ✅ Backed up /opt/encryption.keys"
fi
if [ -f /opt/encryption.zkstark ]; then
    mv /opt/encryption.zkstark /opt/encryption.zkstark.backup.$(date +%s)
    echo "   ✅ Backed up /opt/encryption.zkstark"
fi
if [ ! -f /opt/encryption.keys ]; then
    echo "   ℹ️  No old keys found - this is normal for new installations"
fi

# Update systemd service file
echo ""
echo "3️⃣ Updating systemd service configuration..."
SERVICE_FILE="/etc/systemd/system/q-api-server.service"

if [ -f "$SERVICE_FILE" ]; then
    # Create backup
    cp "$SERVICE_FILE" "${SERVICE_FILE}.backup.$(date +%s)"
    echo "   ✅ Backed up existing service file"

    # Comment out hardcoded encryption lines
    sed -i 's/^Environment="Q_ENCRYPTION_KEYS_FILE=/#Environment="Q_ENCRYPTION_KEYS_FILE=/' "$SERVICE_FILE"
    sed -i 's/^Environment="Q_ENCRYPTION_PASSPHRASE=/#Environment="Q_ENCRYPTION_PASSPHRASE=/' "$SERVICE_FILE"
    echo "   ✅ Removed hardcoded encryption configuration"
else
    echo "   ⚠️  Service file not found at $SERVICE_FILE"
    echo "   You'll need to create one manually"
fi

# Reload systemd
echo ""
echo "4️⃣ Reloading systemd configuration..."
systemctl daemon-reload
echo "   ✅ Systemd reloaded"

# Start the service
echo ""
echo "5️⃣ Starting q-api-server with auto-generated encryption..."
systemctl start q-api-server
echo "   ✅ Service started"

# Wait for initialization
echo ""
echo "6️⃣ Waiting for encryption initialization (5 seconds)..."
sleep 5

# Check status and logs
echo ""
echo "7️⃣ Checking encryption status..."
echo ""

# Get the database path from service file
DB_PATH=$(grep "Q_DB_PATH" "$SERVICE_FILE" | head -1 | cut -d'"' -f2 | sed 's/Environment=//')
if [ -z "$DB_PATH" ]; then
    DB_PATH="./data"
fi

# Check for passphrase file
PASSPHRASE_FILE="${DB_PATH}/encryption_passphrase.txt"
if [ -f "$PASSPHRASE_FILE" ]; then
    echo "✅ SUCCESS! Encryption passphrase auto-generated:"
    echo "   Location: $PASSPHRASE_FILE"
    echo "   ⚠️  IMPORTANT: Backup this file!"
    echo ""
fi

# Check logs for encryption messages
echo "📋 Recent encryption logs:"
journalctl -u q-api-server --since "30 seconds ago" --no-pager | grep -E "(encryption|passphrase|MANDATORY)" | tail -10 || echo "   ℹ️  No encryption messages yet (node may still be starting)"

echo ""
echo "============================================"
echo "✅ Fix complete!"
echo ""
echo "Next steps:"
echo "1. Verify service is running: systemctl status q-api-server"
echo "2. Check logs: journalctl -u q-api-server -n 100"
echo "3. BACKUP your passphrase file: $PASSPHRASE_FILE"
echo "============================================"
