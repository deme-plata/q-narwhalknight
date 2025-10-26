#!/bin/bash
#
# Q-NarwhalKnight Easy Mining Script
# Version: 1.1.0
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
cat << "EOF"
██████╗     ███╗   ██╗ █████╗ ██████╗ ██╗    ██╗██╗  ██╗ █████╗ ██╗
██╔═══██╗    ████╗  ██║██╔══██╗██╔══██╗██║    ██║██║  ██║██╔══██╗██║
██║   ██║    ██╔██╗ ██║███████║██████╔╝██║ █╗ ██║███████║███████║██║
██║▄▄ ██║    ██║╚██╗██║██╔══██║██╔══██╗██║███╗██║██╔══██║██╔══██║██║
╚██████╔╝    ██║ ╚████║██║  ██║██║  ██║╚███╔███╔╝██║  ██║██║  ██║███████╗
 ╚══▀▀═╝     ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
                    MINER v1.1.0 - OPTIMIZED FOR MAXIMUM PERFORMANCE
EOF
echo -e "${NC}"

# Default configuration
DEFAULT_SERVER="http://185.182.185.227:8080/"
DEFAULT_INTENSITY=10
THREADS=0  # Auto-detect

# Check if wallet address provided
if [ -z "$1" ]; then
    echo -e "${RED}ERROR: Wallet address required!${NC}"
    echo ""
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0 YOUR_WALLET_ADDRESS [threads] [intensity]"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 qnk1234567890abcdef..."
    echo "  $0 qnk1234567890abcdef... 8"
    echo "  $0 qnk1234567890abcdef... 8 10"
    echo ""
    exit 1
fi

WALLET_ADDRESS=$1
THREADS=${2:-0}  # Use provided threads or auto-detect
INTENSITY=${3:-$DEFAULT_INTENSITY}

echo -e "${GREEN}🚀 Starting Q-NarwhalKnight Miner${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo "  • Wallet:    $WALLET_ADDRESS"
echo "  • Server:    $DEFAULT_SERVER"
echo "  • Threads:   $THREADS (0 = auto-detect)"
echo "  • Intensity: $INTENSITY (1-10)"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Make sure miner is executable
chmod +x q-miner 2>/dev/null || true

# Detect CPU info
if command -v lscpu &> /dev/null; then
    CPU_MODEL=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
    CPU_CORES=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
    echo -e "${YELLOW}💻 CPU Detected:${NC}"
    echo "  • Model: $CPU_MODEL"
    echo "  • Cores: $CPU_CORES"
    echo ""
fi

# Check for AVX2 support
if grep -q avx2 /proc/cpuinfo 2>/dev/null; then
    echo -e "${GREEN}✅ AVX2 support detected - optimal performance!${NC}"
else
    echo -e "${YELLOW}⚠️  AVX2 not detected - performance may be limited${NC}"
fi
echo ""

# Performance recommendations
if [ "$INTENSITY" -eq 10 ]; then
    echo -e "${YELLOW}⚡ Running at MAXIMUM intensity (99% CPU usage)${NC}"
    echo -e "${YELLOW}   Close other applications for best performance${NC}"
    echo ""
fi

echo -e "${GREEN}🔥 Starting miner... Press Ctrl+C to stop${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Start the miner
./q-miner \
    --mode solo \
    --server "$DEFAULT_SERVER" \
    --wallet "$WALLET_ADDRESS" \
    --threads "$THREADS" \
    --intensity "$INTENSITY"
