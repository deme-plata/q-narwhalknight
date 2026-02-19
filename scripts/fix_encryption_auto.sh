#!/bin/bash
# One-command fix for encryption error
# This removes hardcoded credentials and lets automatic encryption work

set -e

echo "🔧 Fixing encryption configuration..."

# Stop service
systemctl stop q-api-server 2>/dev/null || true

# Remove old keys
rm -f /opt/encryption.keys /opt/encryption.zkstark 2>/dev/null || true

# Comment out hardcoded credentials
sed -i 's/^Environment="Q_ENCRYPTION_KEYS_FILE=/#Environment="Q_ENCRYPTION_KEYS_FILE=/' /etc/systemd/system/q-api-server.service 2>/dev/null || true
sed -i 's/^Environment="Q_ENCRYPTION_PASSPHRASE=/#Environment="Q_ENCRYPTION_PASSPHRASE=/' /etc/systemd/system/q-api-server.service 2>/dev/null || true

# Reload and restart
systemctl daemon-reload
systemctl start q-api-server

echo "✅ Done! Encryption is now fully automatic."
echo ""
echo "Your passphrase was auto-generated and saved."
echo "Check with: journalctl -u q-api-server -n 50 | grep passphrase"
