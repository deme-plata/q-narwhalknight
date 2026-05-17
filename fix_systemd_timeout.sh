#!/bin/bash
# fix_systemd_timeout.sh
# v0.9.76-beta: Fix systemd timeout to prevent SIGKILL during graceful shutdown

set -e

SERVICE_FILE="/etc/systemd/system/q-api-server.service"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 Systemd Timeout Fix for q-api-server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if service file exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo "❌ Error: Service file not found: $SERVICE_FILE"
    exit 1
fi

echo "📂 Service file: $SERVICE_FILE"
echo ""

# Backup original
BACKUP_FILE="${SERVICE_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
echo "💾 Creating backup: $BACKUP_FILE"
cp "$SERVICE_FILE" "$BACKUP_FILE"
echo "✅ Backup created"
echo ""

# Check current timeout
CURRENT_TIMEOUT=$(grep "^TimeoutStopSec=" "$SERVICE_FILE" || echo "Not set")
echo "Current TimeoutStopSec: $CURRENT_TIMEOUT"
echo ""

# Update service file
echo "🔧 Applying fixes..."

# Remove existing TimeoutStopSec if present
sed -i '/^TimeoutStopSec=/d' "$SERVICE_FILE"

# Add timeout settings under [Service] section
sed -i '/^\[Service\]/a TimeoutStopSec=300' "$SERVICE_FILE"
sed -i '/^\[Service\]/a KillMode=mixed' "$SERVICE_FILE"
sed -i '/^\[Service\]/a SendSIGKILL=yes' "$SERVICE_FILE"

echo "✅ Applied the following settings:"
echo "   • TimeoutStopSec=300 (5 minutes instead of 90 seconds)"
echo "   • KillMode=mixed (try SIGTERM first, then SIGKILL)"
echo "   • SendSIGKILL=yes (allow SIGKILL if needed)"
echo ""

# Show updated [Service] section
echo "📄 Updated [Service] section:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sed -n '/^\[Service\]/,/^\[/p' "$SERVICE_FILE" | head -20
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Reload systemd
echo "🔄 Reloading systemd daemon..."
systemctl daemon-reload
echo "✅ Systemd reloaded"
echo ""

# Show status
echo "📊 Service status:"
systemctl status q-api-server --no-pager | head -10
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Systemd timeout fix applied successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  1. Restart service: systemctl restart q-api-server"
echo "  2. Monitor graceful shutdown: journalctl -u q-api-server -f"
echo "  3. Verify no more SIGKILL: journalctl | grep SIGKILL"
echo ""
echo "Backup file: $BACKUP_FILE"
echo "(Restore with: cp $BACKUP_FILE $SERVICE_FILE && systemctl daemon-reload)"
echo ""
