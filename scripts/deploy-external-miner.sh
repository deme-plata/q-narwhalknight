#!/bin/bash
#
# Q-NarwhalKnight External Miner Deployment Script
#
# Purpose: Automated deployment of external miners on VPS instances
# Usage: ./deploy-external-miner.sh [wallet_address]
#
# This script will:
# 1. Download the latest miner binary
# 2. Create systemd service
# 3. Start and enable the miner
# 4. Verify connectivity to bootstrap node
#
# Date: 2025-11-13
# Version: 1.0.0

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BOOTSTRAP_API="https://quillon.xyz/api/v1"
MINER_DIR="/opt/q-miner"
MINER_BINARY_URL="https://dl.quillon.xyz/downloads/q-miner-linux-x64"
SERVICE_NAME="q-miner"
MINER_THREADS=4

# Function: Print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║     Q-NarwhalKnight External Miner Deployment Script          ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Function: Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "This script must be run as root"
        echo "Please run: sudo $0"
        exit 1
    fi
}

# Function: Check system requirements
check_requirements() {
    print_info "Checking system requirements..."

    # Check CPU cores
    CPU_CORES=$(nproc)
    if [ "$CPU_CORES" -lt 2 ]; then
        print_warning "Only $CPU_CORES CPU core(s) detected. Recommended: 2+ cores"
    else
        print_success "CPU cores: $CPU_CORES"
    fi

    # Check RAM
    TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    TOTAL_RAM_GB=$((TOTAL_RAM_KB / 1024 / 1024))
    if [ "$TOTAL_RAM_GB" -lt 2 ]; then
        print_warning "Only ${TOTAL_RAM_GB}GB RAM detected. Recommended: 2GB+"
    else
        print_success "RAM: ${TOTAL_RAM_GB}GB"
    fi

    # Check disk space
    AVAILABLE_SPACE=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ "$AVAILABLE_SPACE" -lt 10 ]; then
        print_warning "Only ${AVAILABLE_SPACE}GB disk space available. Recommended: 10GB+"
    else
        print_success "Disk space: ${AVAILABLE_SPACE}GB available"
    fi

    # Check required commands
    for cmd in curl wget systemctl; do
        if ! command -v $cmd &> /dev/null; then
            print_error "$cmd is not installed"
            exit 1
        fi
    done

    print_success "All requirements met"
}

# Function: Test API connectivity
test_api() {
    print_info "Testing connectivity to bootstrap node..."

    if curl -s --max-time 10 "$BOOTSTRAP_API/node/status" > /dev/null 2>&1; then
        print_success "Successfully connected to $BOOTSTRAP_API"
    else
        print_error "Cannot connect to $BOOTSTRAP_API"
        print_info "Please check your internet connection and firewall settings"
        exit 1
    fi
}

# Function: Get or create wallet
setup_wallet() {
    if [ -n "$MINER_WALLET" ]; then
        print_success "Using provided wallet: $MINER_WALLET"
        return
    fi

    print_info "No wallet address provided"
    print_info "You can either:"
    echo "  1. Provide an existing wallet address"
    echo "  2. Create a new wallet (will be saved to $MINER_DIR/wallet.txt)"

    read -p "Enter choice (1 or 2): " choice

    if [ "$choice" = "1" ]; then
        read -p "Enter your wallet address (qnk...): " MINER_WALLET
        if [[ ! $MINER_WALLET =~ ^qnk[a-zA-Z0-9]+ ]]; then
            print_error "Invalid wallet address format. Must start with 'qnk'"
            exit 1
        fi
    elif [ "$choice" = "2" ]; then
        print_info "Creating new wallet..."
        # TODO: Implement wallet creation via API
        print_warning "Wallet creation not yet implemented"
        print_info "Please create a wallet manually and run this script again with:"
        echo "  $0 YOUR_WALLET_ADDRESS"
        exit 1
    else
        print_error "Invalid choice"
        exit 1
    fi
}

# Function: Download miner binary
download_miner() {
    print_info "Downloading miner binary..."

    # Create directory
    mkdir -p "$MINER_DIR"
    cd "$MINER_DIR"

    # Download binary
    if wget -q --show-progress "$MINER_BINARY_URL" -O q-miner; then
        print_success "Miner binary downloaded"
    else
        print_error "Failed to download miner binary from $MINER_BINARY_URL"
        exit 1
    fi

    # Make executable
    chmod +x q-miner

    # Verify binary
    if ./q-miner --version > /dev/null 2>&1; then
        MINER_VERSION=$(./q-miner --version 2>&1 | head -1)
        print_success "Miner binary verified: $MINER_VERSION"
    else
        print_error "Downloaded binary is not valid"
        exit 1
    fi
}

# Function: Create systemd service
create_service() {
    print_info "Creating systemd service..."

    cat > "/etc/systemd/system/$SERVICE_NAME.service" <<EOF
[Unit]
Description=Q-NarwhalKnight External Miner
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$MINER_DIR
ExecStart=$MINER_DIR/q-miner \\
  --api-url $BOOTSTRAP_API \\
  --wallet $MINER_WALLET \\
  --threads $MINER_THREADS \\
  --max-retries 10 \\
  --retry-delay 5

# Restart policy
Restart=always
RestartSec=10

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

# Security hardening
NoNewPrivileges=true
ProtectSystem=full
ProtectHome=true
PrivateTmp=true

# Resource limits
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
EOF

    print_success "Systemd service created"
}

# Function: Start and enable service
start_service() {
    print_info "Starting miner service..."

    # Reload systemd
    systemctl daemon-reload

    # Enable service
    systemctl enable "$SERVICE_NAME"

    # Start service
    if systemctl start "$SERVICE_NAME"; then
        print_success "Miner service started"
    else
        print_error "Failed to start miner service"
        print_info "Check logs with: journalctl -u $SERVICE_NAME -f"
        exit 1
    fi

    # Wait for service to stabilize
    sleep 5

    # Check status
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        print_success "Miner is running"
    else
        print_error "Miner service failed to start"
        print_info "Check logs with: journalctl -u $SERVICE_NAME -n 50"
        exit 1
    fi
}

# Function: Verify mining activity
verify_mining() {
    print_info "Verifying mining activity (waiting 30 seconds)..."

    sleep 30

    # Check logs for mining solutions
    if journalctl -u "$SERVICE_NAME" --since "30 seconds ago" | grep -qi "solution\|mining"; then
        print_success "Miner is actively mining!"
    else
        print_warning "No mining activity detected yet"
        print_info "This may be normal if difficulty is high"
        print_info "Monitor logs with: journalctl -u $SERVICE_NAME -f"
    fi
}

# Function: Display summary
display_summary() {
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                                ║${NC}"
    echo -e "${GREEN}║                  Deployment Successful! 🎉                     ║${NC}"
    echo -e "${GREEN}║                                                                ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Miner Configuration:${NC}"
    echo "  • API URL: $BOOTSTRAP_API"
    echo "  • Wallet: $MINER_WALLET"
    echo "  • Threads: $MINER_THREADS"
    echo "  • Service: $SERVICE_NAME"
    echo ""
    echo -e "${BLUE}Useful Commands:${NC}"
    echo "  • Check status: systemctl status $SERVICE_NAME"
    echo "  • View logs: journalctl -u $SERVICE_NAME -f"
    echo "  • Restart: systemctl restart $SERVICE_NAME"
    echo "  • Stop: systemctl stop $SERVICE_NAME"
    echo ""
    echo -e "${BLUE}Monitoring:${NC}"
    echo "  • Hash rate: journalctl -u $SERVICE_NAME --since '10 min ago' | grep 'hash rate'"
    echo "  • Solutions: journalctl -u $SERVICE_NAME --since '10 min ago' | grep 'solution'"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "  1. Deploy miners on 2-4 more VPS instances for redundancy"
    echo "  2. Monitor network hashrate: curl -s https://quillon.xyz/api/v1/network/supply | jq"
    echo "  3. Verify bootstrap node receives solutions"
    echo ""
}

# Main execution
main() {
    print_header

    # Parse arguments
    if [ $# -ge 1 ]; then
        MINER_WALLET="$1"
    fi

    # Run deployment steps
    check_root
    check_requirements
    test_api
    setup_wallet
    download_miner
    create_service
    start_service
    verify_mining
    display_summary

    print_success "Deployment complete!"
}

# Run main function
main "$@"
