#!/bin/bash
# Q-NarwhalKnight Node - Linux Installation Script
# Version: 0.1.0-alpha
# Description: One-click installer for Q-NarwhalKnight quantum consensus node

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="/usr/local/bin"
DATA_DIR="$HOME/.q-narwhalknight"
BINARY_NAME="q-api-server"
DOWNLOAD_URL="https://quillon.xyz/downloads/q-api-server-linux-x86_64"
SERVICE_NAME="q-narwhalknight"

echo -e "${CYAN}"
cat << "EOF"
  ___    _   _                 _           _ _  __      _       _     _
 / _ \  | \ | | __ _ _ ____      _____  _| | |/ /_ __ (_) __ _| |__ | |_
| | | | |  \| |/ _` | '__\ \ /\ / / _ \| | | ' /| '_ \| |/ _` | '_ \| __|
| |_| | | |\  | (_| | |   \ V  V |  __/| | | . \| | | | | (_| | | | | |_
 \__\_\ |_| \_|\__,_|_|    \_/\_/ \___||_|_|_|\_|_| |_|_|\__, |_| |_|\__|
                                                          |___/
   Quantum Consensus Network - Node Installer v0.1.0-alpha
EOF
echo -e "${NC}"

# Check if running as root for system-wide installation
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Running as root - will install system-wide${NC}"
    INSTALL_DIR="/usr/local/bin"
    DATA_DIR="/var/lib/q-narwhalknight"
    SYSTEMD_INSTALL=true
else
    echo -e "${BLUE}ℹ️  Running as user - will install to user directory${NC}"
    INSTALL_DIR="$HOME/.local/bin"
    DATA_DIR="$HOME/.q-narwhalknight"
    SYSTEMD_INSTALL=false
fi

# Detect OS and architecture
echo -e "${CYAN}🔍 Detecting system information...${NC}"
OS=$(uname -s)
ARCH=$(uname -m)

echo -e "   OS: ${GREEN}$OS${NC}"
echo -e "   Architecture: ${GREEN}$ARCH${NC}"

# Check if Linux x86_64
if [ "$OS" != "Linux" ] || [ "$ARCH" != "x86_64" ]; then
    echo -e "${RED}❌ This installer only supports Linux x86_64${NC}"
    echo -e "${YELLOW}   Your system: $OS $ARCH${NC}"
    exit 1
fi

# Create directories
echo -e "${CYAN}📁 Creating directories...${NC}"
mkdir -p "$INSTALL_DIR"
mkdir -p "$DATA_DIR"
echo -e "   Install directory: ${GREEN}$INSTALL_DIR${NC}"
echo -e "   Data directory: ${GREEN}$DATA_DIR${NC}"

# Download binary
echo -e "${CYAN}📥 Downloading Q-NarwhalKnight node binary...${NC}"
TEMP_FILE=$(mktemp)
if command -v curl &> /dev/null; then
    curl -fSL --progress-bar "$DOWNLOAD_URL" -o "$TEMP_FILE"
elif command -v wget &> /dev/null; then
    wget --show-progress -O "$TEMP_FILE" "$DOWNLOAD_URL"
else
    echo -e "${RED}❌ Neither curl nor wget found. Please install one of them.${NC}"
    exit 1
fi

# Verify download
if [ ! -s "$TEMP_FILE" ]; then
    echo -e "${RED}❌ Download failed or file is empty${NC}"
    rm -f "$TEMP_FILE"
    exit 1
fi

echo -e "${GREEN}✅ Download complete ($(du -h "$TEMP_FILE" | cut -f1))${NC}"

# Install binary
echo -e "${CYAN}📦 Installing binary...${NC}"
chmod +x "$TEMP_FILE"
mv "$TEMP_FILE" "$INSTALL_DIR/$BINARY_NAME"
echo -e "${GREEN}✅ Binary installed to $INSTALL_DIR/$BINARY_NAME${NC}"

# Create systemd service (if running as root)
if [ "$SYSTEMD_INSTALL" = true ]; then
    echo -e "${CYAN}⚙️  Creating systemd service...${NC}"

    cat > /etc/systemd/system/$SERVICE_NAME.service << EOF
[Unit]
Description=Q-NarwhalKnight Quantum Consensus Node
After=network.target
Documentation=https://github.com/deme-plata/q-narwhalknight

[Service]
Type=simple
User=root
WorkingDirectory=$DATA_DIR
Environment="Q_DB_PATH=$DATA_DIR"
Environment="DISABLE_GPU_STARK=1"
ExecStart=$INSTALL_DIR/$BINARY_NAME --port 8080
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$DATA_DIR

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable $SERVICE_NAME

    echo -e "${GREEN}✅ Systemd service created and enabled${NC}"
    echo -e "${YELLOW}   Start with: sudo systemctl start $SERVICE_NAME${NC}"
    echo -e "${YELLOW}   Status: sudo systemctl status $SERVICE_NAME${NC}"
    echo -e "${YELLOW}   Logs: sudo journalctl -u $SERVICE_NAME -f${NC}"
fi

# Create config file
echo -e "${CYAN}⚙️  Creating configuration file...${NC}"
cat > "$DATA_DIR/config.toml" << 'EOF'
# Q-NarwhalKnight Node Configuration
# Edit this file to customize your node

[node]
port = 8080
node_id = "auto-generated"

[network]
max_peers = 50
enable_tor = true
enable_bitcoin_bridge = true
enable_dns_phantom = true

[consensus]
validator_mode = false  # Set to true to participate in consensus
enable_mining = false   # Set to true to enable GPU mining

[database]
path = "auto"  # Uses Q_DB_PATH environment variable

[logging]
level = "info"  # Options: trace, debug, info, warn, error
EOF

echo -e "${GREEN}✅ Configuration created at $DATA_DIR/config.toml${NC}"

# Display usage instructions
echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Installation complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

if [ "$SYSTEMD_INSTALL" = true ]; then
    echo -e "${CYAN}🚀 Quick Start (System Service):${NC}"
    echo -e "   ${YELLOW}sudo systemctl start $SERVICE_NAME${NC}      - Start the node"
    echo -e "   ${YELLOW}sudo systemctl status $SERVICE_NAME${NC}     - Check status"
    echo -e "   ${YELLOW}sudo journalctl -u $SERVICE_NAME -f${NC}     - View logs"
else
    echo -e "${CYAN}🚀 Quick Start (Manual):${NC}"
    echo -e "   ${YELLOW}DISABLE_GPU_STARK=1 $INSTALL_DIR/$BINARY_NAME --port 8080${NC}"
    echo -e "\n${CYAN}💡 Tip: Add $INSTALL_DIR to your PATH:${NC}"
    echo -e "   ${YELLOW}export PATH=\"$INSTALL_DIR:\$PATH\"${NC}"
    echo -e "   ${YELLOW}echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.bashrc${NC}"
    echo -e "\n${CYAN}💡 GPU Compatibility:${NC}"
    echo -e "   ${YELLOW}Set DISABLE_GPU_STARK=1 to avoid GPU crashes on some systems${NC}"
fi

echo -e "\n${CYAN}📝 Configuration:${NC}"
echo -e "   ${YELLOW}nano $DATA_DIR/config.toml${NC}"

echo -e "\n${CYAN}🌐 Web Interface:${NC}"
echo -e "   ${YELLOW}http://localhost:8080${NC}"

echo -e "\n${CYAN}📊 API Documentation:${NC}"
echo -e "   ${YELLOW}http://localhost:8080/api/v1/health${NC}"

echo -e "\n${CYAN}💎 Wallet:${NC}"
echo -e "   ${YELLOW}https://quillon.xyz${NC}"

echo -e "\n${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${PURPLE}⚛️  Join the quantum consensus revolution!${NC}"
echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
