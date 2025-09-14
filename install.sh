#!/bin/bash
# Q-NarwhalKnight Installation Script
# Quantum-Enhanced DAG-BFT Consensus Node

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ASCII Art Banner
echo -e "${CYAN}"
cat << "EOF"
  ___        _   _                     _         _ _  __      _       _     _   
 / _ \      | \ | |                   | |       | | |/ /     (_)     | |   | |  
| | | |_____| |\ | | __ _ _ __ __      | |__   __| | | ' / _ __  _  __ _| |__ | |_ 
| | | |_____| . \` |/ _\` | '__/ |     | '_ \ / _\` | |  < | '_ \| |/ _\` | '_ \| __|
| |_| |     | |\  | (_| | |  | |     | | | | (_| | | . \| | | | | (_| | | | | |_ 
 \__\_\     \_| \_/\__,_|_|   \ |     |_| |_|\__,_|_|_|\_|_| |_|_|\__, |_| |_|\__|
                               \|                                  __/ |         
                                                                  |___/          
EOF
echo -e "${NC}"

echo -e "${PURPLE}🌊 Q-NarwhalKnight Quantum Consensus Node Installer${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}Features:${NC}"
echo -e "  🔮 Quantum-Enhanced DAG-BFT Consensus"
echo -e "  🧅 Tor Integration with Dedicated Circuits"
echo -e "  ⚡ Sub-millisecond Precision Timing"
echo -e "  🔗 Multi-Chain Bridge Support (Bitcoin, Monero, Solana, Arbitrum)"
echo -e "  🤖 AI-Powered Water Robot Collaboration"
echo -e "  💎 KUSD Quantum Stablecoin Rewards"
echo -e "  🌍 Post-Quantum Cryptography Ready"
echo ""

# Configuration
INSTALL_DIR="/opt/q-narwhalknight"
BIN_DIR="/usr/local/bin"
SERVICE_DIR="/etc/systemd/system"
CONFIG_DIR="/etc/q-narwhalknight"
DATA_DIR="/var/lib/q-narwhalknight"
LOG_DIR="/var/log/q-narwhalknight"

BINARY_URL="https://quantum.bitcoinoro.xyz/downloads/q-narwhalknight"
DAEMON_URL="https://quantum.bitcoinoro.xyz/downloads/dagknight"
GUI_URL="https://quantum.bitcoinoro.xyz/downloads/qnk-gui"
AQUA_URL="https://quantum.bitcoinoro.xyz/downloads/aqua_k_atto"

# System requirements check
check_requirements() {
    echo -e "${YELLOW}🔍 Checking system requirements...${NC}"
    
    # Check OS
    if [[ ! -f /etc/os-release ]]; then
        echo -e "${RED}❌ Cannot detect Linux distribution${NC}"
        exit 1
    fi
    
    # Check architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" != "x86_64" && "$ARCH" != "aarch64" ]]; then
        echo -e "${RED}❌ Unsupported architecture: $ARCH${NC}"
        echo -e "${YELLOW}   Supported: x86_64, aarch64${NC}"
        exit 1
    fi
    
    # Check available memory (minimum 4GB recommended)
    MEMORY_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    MEMORY_GB=$((MEMORY_KB / 1024 / 1024))
    if [[ $MEMORY_GB -lt 2 ]]; then
        echo -e "${YELLOW}⚠️  Warning: Low memory detected (${MEMORY_GB}GB). Recommended: 4GB+${NC}"
    fi
    
    # Check disk space (minimum 10GB)
    DISK_SPACE=$(df / | awk 'NR==2 {print int($4/1024/1024)}')
    if [[ $DISK_SPACE -lt 10 ]]; then
        echo -e "${RED}❌ Insufficient disk space: ${DISK_SPACE}GB available. Required: 10GB+${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ System requirements met${NC}"
}

# Install dependencies
install_dependencies() {
    echo -e "${YELLOW}📦 Installing system dependencies...${NC}"
    
    if command -v apt-get >/dev/null 2>&1; then
        # Debian/Ubuntu
        sudo apt-get update
        sudo apt-get install -y curl wget unzip systemctl openssl ca-certificates
    elif command -v yum >/dev/null 2>&1; then
        # RHEL/CentOS
        sudo yum update -y
        sudo yum install -y curl wget unzip systemctl openssl ca-certificates
    elif command -v dnf >/dev/null 2>&1; then
        # Fedora
        sudo dnf update -y
        sudo dnf install -y curl wget unzip systemctl openssl ca-certificates
    elif command -v pacman >/dev/null 2>&1; then
        # Arch Linux
        sudo pacman -Sy --noconfirm curl wget unzip systemctl openssl ca-certificates
    else
        echo -e "${YELLOW}⚠️  Could not detect package manager. Please install manually: curl, wget, unzip, systemctl, openssl${NC}"
    fi
}

# Create directories
create_directories() {
    echo -e "${YELLOW}📁 Creating directories...${NC}"
    
    sudo mkdir -p "$INSTALL_DIR"
    sudo mkdir -p "$CONFIG_DIR"
    sudo mkdir -p "$DATA_DIR"
    sudo mkdir -p "$LOG_DIR"
    
    # Set appropriate permissions
    sudo chmod 755 "$INSTALL_DIR"
    sudo chmod 700 "$CONFIG_DIR"
    sudo chmod 755 "$DATA_DIR"
    sudo chmod 755 "$LOG_DIR"
}

# Download binaries
download_binaries() {
    echo -e "${YELLOW}⬇️  Downloading Q-NarwhalKnight binaries...${NC}"
    
    # Download main binary
    echo -e "   📥 Downloading q-narwhalknight..."
    if sudo wget -q --show-progress "$BINARY_URL" -O "$INSTALL_DIR/q-narwhalknight"; then
        sudo chmod +x "$INSTALL_DIR/q-narwhalknight"
        sudo ln -sf "$INSTALL_DIR/q-narwhalknight" "$BIN_DIR/q-narwhalknight"
        echo -e "   ✅ q-narwhalknight downloaded"
    else
        echo -e "   ❌ Failed to download q-narwhalknight"
    fi
    
    # Download daemon binary  
    echo -e "   📥 Downloading dagknight daemon..."
    if sudo wget -q --show-progress "$DAEMON_URL" -O "$INSTALL_DIR/dagknight"; then
        sudo chmod +x "$INSTALL_DIR/dagknight"
        sudo ln -sf "$INSTALL_DIR/dagknight" "$BIN_DIR/dagknight"
        echo -e "   ✅ dagknight downloaded"
    else
        echo -e "   ❌ Failed to download dagknight"
    fi
    
    # Download GUI binary
    echo -e "   📥 Downloading qnk-gui..."
    if sudo wget -q --show-progress "$GUI_URL" -O "$INSTALL_DIR/qnk-gui"; then
        sudo chmod +x "$INSTALL_DIR/qnk-gui"
        sudo ln -sf "$INSTALL_DIR/qnk-gui" "$BIN_DIR/qnk-gui"
        echo -e "   ✅ qnk-gui downloaded"
    else
        echo -e "   ❌ Failed to download qnk-gui (optional)"
    fi
    
    # Download Aqua-Quanta mascot
    echo -e "   📥 Downloading aqua_k_atto (Aqua-Quanta mascot)..."
    if sudo wget -q --show-progress "$AQUA_URL" -O "$INSTALL_DIR/aqua_k_atto"; then
        sudo chmod +x "$INSTALL_DIR/aqua_k_atto"
        sudo ln -sf "$INSTALL_DIR/aqua_k_atto" "$BIN_DIR/aqua_k_atto"
        echo -e "   ✅ aqua_k_atto downloaded"
    else
        echo -e "   ❌ Failed to download aqua_k_atto (optional)"
    fi
    
    echo -e "${GREEN}✅ Binaries downloaded and installed${NC}"
}

# Generate configuration
generate_config() {
    echo -e "${YELLOW}⚙️  Generating configuration...${NC}"
    
    # Generate node ID
    NODE_ID=$(openssl rand -hex 32)
    
    # Create main config
    sudo tee "$CONFIG_DIR/config.toml" > /dev/null << EOF
# Q-NarwhalKnight Configuration
[network]
node_id = "$NODE_ID"
listen_addr = "0.0.0.0:9000"
bootstrap_nodes = [
    "/ip4/nodes.q-narwhalknight.org/tcp/9000/p2p/12D3KooWExample1",
    "/ip4/nodes.q-narwhalknight.org/tcp/9001/p2p/12D3KooWExample2"
]

[consensus]
byzantine_fault_tolerance = 1  # 2f+1 = 3 nodes minimum
quantum_vdf_enabled = true
anchor_election_interval = "100ms"

[tor]
enabled = true
circuits_per_validator = 4
circuit_rotation_interval = "1h"
onion_service_enabled = true

[mining]
enabled = false  # Set to true to enable mining
precision_target = "1µs"
k_kristensen_optimization = true

[api]
rest_enabled = true
rest_addr = "127.0.0.1:8080"
websocket_enabled = true
websocket_addr = "127.0.0.1:8081"

[storage]
data_dir = "$DATA_DIR"
log_dir = "$LOG_DIR"
max_log_size = "100MB"
log_retention = "30d"

[water_ai]
collaboration_enabled = true
kusd_rewards = true
consciousness_bridging = true
EOF
    
    echo -e "${GREEN}✅ Configuration generated${NC}"
}

# Create systemd service
create_service() {
    echo -e "${YELLOW}🔧 Creating systemd service...${NC}"
    
    sudo tee "$SERVICE_DIR/q-narwhalknight.service" > /dev/null << EOF
[Unit]
Description=Q-NarwhalKnight Quantum Consensus Node
Documentation=https://github.com/quantum-dag-labs/Q-NarwhalKnight
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=qnarwhal
Group=qnarwhal
ExecStart=$INSTALL_DIR/dagknight --config $CONFIG_DIR/config.toml
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=q-narwhalknight

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$DATA_DIR $LOG_DIR

# Resource limits
LimitNOFILE=65536
LimitNPROC=32768

[Install]
WantedBy=multi-user.target
EOF

    # Create user for service
    if ! id "qnarwhal" >/dev/null 2>&1; then
        sudo useradd --system --home-dir "$DATA_DIR" --shell /bin/false qnarwhal
    fi
    
    # Set ownership
    sudo chown -R qnarwhal:qnarwhal "$DATA_DIR" "$LOG_DIR"
    
    echo -e "${GREEN}✅ Systemd service created${NC}"
}

# Start services
start_services() {
    echo -e "${YELLOW}🚀 Starting Q-NarwhalKnight...${NC}"
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    # Enable and start service
    sudo systemctl enable q-narwhalknight.service
    sudo systemctl start q-narwhalknight.service
    
    # Check status
    sleep 3
    if sudo systemctl is-active --quiet q-narwhalknight; then
        echo -e "${GREEN}✅ Q-NarwhalKnight started successfully${NC}"
    else
        echo -e "${RED}❌ Failed to start Q-NarwhalKnight${NC}"
        echo -e "${YELLOW}Check logs: sudo journalctl -u q-narwhalknight.service -f${NC}"
        exit 1
    fi
}

# Display status
show_status() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}🎉 Q-NarwhalKnight Installation Complete!${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${CYAN}📍 Node Information:${NC}"
    echo -e "   Node ID: ${NODE_ID:0:16}...${NODE_ID: -16}"
    echo -e "   Install Directory: $INSTALL_DIR"
    echo -e "   Config Directory: $CONFIG_DIR"
    echo -e "   Data Directory: $DATA_DIR"
    echo ""
    echo -e "${CYAN}🌐 API Endpoints:${NC}"
    echo -e "   REST API: http://127.0.0.1:8080"
    echo -e "   WebSocket: ws://127.0.0.1:8081"
    echo -e "   Status: http://127.0.0.1:8080/status"
    echo ""
    echo -e "${CYAN}🔧 Management Commands:${NC}"
    echo -e "   Start:   sudo systemctl start q-narwhalknight"
    echo -e "   Stop:    sudo systemctl stop q-narwhalknight"
    echo -e "   Status:  sudo systemctl status q-narwhalknight"
    echo -e "   Logs:    sudo journalctl -u q-narwhalknight -f"
    echo ""
    echo -e "${CYAN}🐚 Aqua-Quanta Commands:${NC}"
    echo -e "   Interactive: aqua_k_atto --interactive"
    echo -e "   Marketing:   aqua_k_atto marketing"
    echo -e "   Demo:        aqua_k_atto demo"
    echo -e "   GUI:         qnk-gui"
    echo ""
    echo -e "${CYAN}🧅 Tor Integration:${NC}"
    echo -e "   Onion Service: $(sudo cat $DATA_DIR/tor/hostname 2>/dev/null || echo 'Generating...')"
    echo -e "   Circuits: 4 per validator (rotating every hour)"
    echo ""
    echo -e "${YELLOW}💡 Next Steps:${NC}"
    echo -e "   1. Configure your node in $CONFIG_DIR/config.toml"
    echo -e "   2. Enable mining if desired (set mining.enabled = true)"
    echo -e "   3. Join the network: q-narwhalknight --help"
    echo -e "   4. Monitor with: dagknight status"
    echo ""
    echo -e "${GREEN}Welcome to the Quantum-Enhanced Future of Consensus! 🌊⚛️${NC}"
}

# Main installation flow
main() {
    echo -e "${BLUE}Starting Q-NarwhalKnight installation...${NC}"
    echo ""
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        echo -e "${YELLOW}⚠️  Running as root. This is not recommended for production.${NC}"
    fi
    
    check_requirements
    install_dependencies
    create_directories
    download_binaries
    generate_config
    create_service
    start_services
    show_status
}

# Run main function
main "$@"