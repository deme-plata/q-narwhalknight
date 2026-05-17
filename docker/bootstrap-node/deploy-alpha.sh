#!/bin/bash
# Q-NarwhalKnight Bootstrap Node Deployment Script
# v2.9.0-beta: Deploy secondary bootstrap on Server Alpha (161.35.219.10)
#
# Usage:
#   ./deploy-alpha.sh [build|start|stop|logs|status]
#
# This script:
# 1. Builds the bootstrap node Docker image
# 2. Starts the container with proper configuration
# 3. Syncs from Server Beta (185.182.185.227)
# 4. Becomes a secondary bootstrap node

set -e

# Configuration
CONTAINER_NAME="q-bootstrap-alpha"
IMAGE_NAME="q-narwhalknight-bootstrap:latest"
DATA_DIR="/var/lib/q-narwhalknight/bootstrap"

# Server Beta as primary bootstrap source
PRIMARY_BOOTSTRAP="http://185.182.185.227:8080"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    log_success "Docker is installed"
}

# Check connectivity to primary bootstrap
check_primary_bootstrap() {
    log_info "Checking connectivity to primary bootstrap (Server Beta)..."
    if curl -sf "${PRIMARY_BOOTSTRAP}/api/v1/status" > /dev/null 2>&1; then
        local height=$(curl -s "${PRIMARY_BOOTSTRAP}/api/v1/status" | jq -r '.data.height // 0')
        log_success "Primary bootstrap reachable (height: $height)"
    else
        log_error "Cannot reach primary bootstrap at ${PRIMARY_BOOTSTRAP}"
        log_error "Ensure Server Beta is running and accessible"
        exit 1
    fi
}

# Build the Docker image
build_image() {
    log_info "Building Docker image..."

    # Navigate to project root
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

    cd "$PROJECT_ROOT"

    docker build \
        -t "$IMAGE_NAME" \
        -f docker/bootstrap-node/Dockerfile \
        .

    log_success "Docker image built: $IMAGE_NAME"
}

# Start the bootstrap node
start_node() {
    log_info "Starting bootstrap node..."

    # Create data directory
    mkdir -p "$DATA_DIR"

    # Stop existing container if running
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

    # Start container
    docker run -d \
        --name "$CONTAINER_NAME" \
        --network host \
        --restart unless-stopped \
        -e RUST_LOG=info,q_api_server=info,q_network=info \
        -e Q_NETWORK_ID=testnet-phase16 \
        -e Q_BOOTSTRAP_URL="$PRIMARY_BOOTSTRAP" \
        -e Q_BOOTSTRAP_URLS="$PRIMARY_BOOTSTRAP" \
        -e Q_API_PORT=8080 \
        -e Q_P2P_PORT=9001 \
        -e Q_DISABLE_MINING=true \
        -v "${DATA_DIR}:/data" \
        "$IMAGE_NAME"

    log_success "Bootstrap node started: $CONTAINER_NAME"
    log_info "Syncing from primary bootstrap..."

    # Wait for initial sync
    sleep 10
    show_status
}

# Stop the bootstrap node
stop_node() {
    log_info "Stopping bootstrap node..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    log_success "Bootstrap node stopped"
}

# Show logs
show_logs() {
    log_info "Showing logs (Ctrl+C to exit)..."
    docker logs -f "$CONTAINER_NAME"
}

# Show status
show_status() {
    log_info "Checking bootstrap node status..."

    if ! docker ps --filter "name=$CONTAINER_NAME" --format "{{.Status}}" | grep -q "Up"; then
        log_error "Container is not running"
        return 1
    fi

    local status=$(curl -s "http://localhost:8080/api/v1/status" 2>/dev/null)
    if [ -z "$status" ]; then
        log_warn "Node is starting up, API not ready yet..."
        return 0
    fi

    local height=$(echo "$status" | jq -r '.data.height // 0')
    local peers=$(echo "$status" | jq -r '.data.connected_peers // 0')
    local peer_id=$(echo "$status" | jq -r '.data.peer_id // "unknown"')

    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Q-NarwhalKnight Bootstrap Node (Server Alpha)"
    echo "═══════════════════════════════════════════════════════════"
    echo "  Status:    $(docker ps --filter "name=$CONTAINER_NAME" --format "{{.Status}}")"
    echo "  Height:    $height blocks"
    echo "  Peers:     $peers connected"
    echo "  Peer ID:   $peer_id"
    echo ""
    echo "  API URL:   http://161.35.219.10:8080"
    echo "  P2P:       /ip4/161.35.219.10/tcp/9001/p2p/$peer_id"
    echo "═══════════════════════════════════════════════════════════"
    echo ""

    # Check sync status with primary
    local primary_height=$(curl -s "${PRIMARY_BOOTSTRAP}/api/v1/status" 2>/dev/null | jq -r '.data.height // 0')
    local behind=$((primary_height - height))

    if [ "$behind" -gt 0 ]; then
        log_warn "Behind primary by $behind blocks (syncing...)"
    else
        log_success "Fully synced with network!"
    fi
}

# Print usage
usage() {
    echo "Q-NarwhalKnight Bootstrap Node Deployment"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build   - Build the Docker image"
    echo "  start   - Start the bootstrap node"
    echo "  stop    - Stop the bootstrap node"
    echo "  restart - Restart the bootstrap node"
    echo "  logs    - Show container logs"
    echo "  status  - Show node status"
    echo "  all     - Build and start (full deployment)"
    echo ""
}

# Main
case "${1:-all}" in
    build)
        check_docker
        build_image
        ;;
    start)
        check_docker
        check_primary_bootstrap
        start_node
        ;;
    stop)
        stop_node
        ;;
    restart)
        stop_node
        start_node
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    all)
        check_docker
        check_primary_bootstrap
        build_image
        start_node
        ;;
    *)
        usage
        exit 1
        ;;
esac
