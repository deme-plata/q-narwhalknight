#!/bin/bash
# Quillon Graph — Node Quick Install
# Run as root: curl -fsSL https://quillon.xyz/setup-node.sh | bash
set -e

INSTALL_DIR="/opt/quillon"
DATA_DIR="/opt/quillon/data"
BINARY_URL="https://quillon.xyz/downloads/q-api-server-linux-x86_64"
API_PORT=8080
P2P_PORT=9001

echo ""
echo "  Quillon Graph — Node Setup"
echo "  =========================="
echo ""

# Check root
if [ "$EUID" -ne 0 ]; then
  echo "  Please run as root: sudo bash setup-node.sh"
  exit 1
fi

# 1. Create directories
mkdir -p "$INSTALL_DIR" "$DATA_DIR"
echo "  ✓ Directories: $INSTALL_DIR"

# 2. Download latest binary
echo "  Downloading node binary..."
curl -fSL "$BINARY_URL" -o "$INSTALL_DIR/q-api-server"
chmod +x "$INSTALL_DIR/q-api-server"
echo "  ✓ Binary downloaded"

# 3. Install systemd service
cat > /etc/systemd/system/quillon-node.service << SVCEOF
[Unit]
Description=Quillon Graph Node
Documentation=https://quillon.xyz
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
Environment="Q_DB_PATH=${DATA_DIR}"
Environment="Q_NETWORK_ID=mainnet-genesis"
Environment="RUST_LOG=warn"
ExecStart=${INSTALL_DIR}/q-api-server --port ${API_PORT}
Restart=on-failure
RestartSec=10
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
SVCEOF

echo "  ✓ Systemd service installed"

# 4. Enable and start
systemctl daemon-reload
systemctl enable quillon-node
systemctl start quillon-node

sleep 2

echo ""
echo "  ╔═══════════════════════════════════════════════╗"
echo "  ║        Node is Running!                       ║"
echo "  ╠═══════════════════════════════════════════════╣"
echo "  ║                                               ║"
echo "  ║  API:   http://localhost:8080                 ║"
echo "  ║  Data:  $DATA_DIR                  ║"
echo "  ║                                               ║"
echo "  ║  Useful commands:                             ║"
echo "  ║    systemctl status quillon-node              ║"
echo "  ║    journalctl -u quillon-node -f              ║"
echo "  ║                                               ║"
echo "  ║  Check sync progress (after 30s):             ║"
echo "  ║    curl -s http://localhost:8080/api/v1/node/status | python3 -m json.tool"
echo "  ║                                               ║"
echo "  ║  Full sync takes 2-6 hours (turbo-sync).     ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo ""
echo "  quillon.xyz | Post-Quantum Electronic Cash"
echo ""
