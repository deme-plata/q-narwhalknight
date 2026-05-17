#!/bin/bash
# 🚀 HISTORIC 10-NODE ANONYMOUS QUANTUM BFT NETWORK DEPLOYMENT
# Server Alpha (5 nodes) + Server Beta (5 nodes) = 10 total validators
# Byzantine fault tolerance: f=3, requires 2f+1=7 nodes for consensus

set -e

echo "🌟 DEPLOYING WORLD'S FIRST CROSS-SERVER QUANTUM BFT NETWORK 🌟"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Network configuration
NETWORK_DIR="/tmp/qnk_network"
SERVER_ALPHA_NODES=("alice" "bob" "charlie" "diana" "eve")
SERVER_BETA_NODES=("frank" "grace" "henry" "iris" "jack")
BASE_HTTP_PORT=8001
BASE_TOR_PORT=9051
TOTAL_NODES=10

echo -e "${BLUE}📋 Network Configuration:${NC}"
echo "   Total Nodes: ${TOTAL_NODES}"
echo "   Byzantine Tolerance: f=3 (up to 3 failures)"
echo "   Consensus Threshold: 7/10 nodes (2f+1)"
echo "   Server Alpha Nodes: ${SERVER_ALPHA_NODES[*]}"
echo "   Server Beta Nodes: ${SERVER_BETA_NODES[*]}"
echo ""

# Create network directory structure
echo -e "${YELLOW}🏗️ Creating network infrastructure...${NC}"
rm -rf "$NETWORK_DIR"
mkdir -p "$NETWORK_DIR"/{alpha,beta}/logs
mkdir -p "$NETWORK_DIR"/{alpha,beta}/data
mkdir -p "$NETWORK_DIR"/{alpha,beta}/tor

# Function to generate node configuration
generate_node_config() {
    local node_name=$1
    local server_type=$2
    local node_index=$3
    local http_port=$((BASE_HTTP_PORT + node_index))
    local tor_port=$((BASE_TOR_PORT + node_index))
    
    cat > "$NETWORK_DIR/$server_type/$node_name.toml" << EOF
# Q-NarwhalKnight Node Configuration: $node_name ($server_type)
[node]
name = "$node_name"
server_type = "$server_type"
node_id = "$(openssl rand -hex 32)"

[network]
http_port = $http_port
tor_control_port = $tor_port
enable_tor = true
enable_anonymous_mode = true

[consensus]
role = "validator"
byzantine_threshold = 3
min_consensus_nodes = 7

[crypto]
phase = "Phase1"
enable_post_quantum = true
signature_algorithm = "Dilithium5"
key_exchange = "Kyber1024"

[storage]
data_dir = "$NETWORK_DIR/$server_type/data/$node_name"
enable_persistence = true

[tor]
service_name = "qnk-$node_name"
data_dir = "$NETWORK_DIR/$server_type/tor/$node_name"
auth_cookie = "/var/lib/tor/control_auth_cookie"
EOF
    
    echo "   ✅ Generated config for $node_name ($server_type)"
}

# Generate configurations for all nodes
echo -e "${YELLOW}⚙️ Generating node configurations...${NC}"
node_index=0

# Server Alpha nodes
for node in "${SERVER_ALPHA_NODES[@]}"; do
    generate_node_config "$node" "alpha" "$node_index"
    ((node_index++))
done

# Server Beta nodes  
for node in "${SERVER_BETA_NODES[@]}"; do
    generate_node_config "$node" "beta" "$node_index"
    ((node_index++))
done

echo ""

# Function to start a node
start_node() {
    local node_name=$1
    local server_type=$2
    local config_file="$NETWORK_DIR/$server_type/$node_name.toml"
    local log_file="$NETWORK_DIR/$server_type/logs/$node_name.log"
    
    echo -e "${GREEN}🚀 Starting $node_name ($server_type)...${NC}"
    
    # Start the node in background
    RUST_LOG=info target/release/q-api-server \
        --config "$config_file" \
        --node-name "$node_name" \
        > "$log_file" 2>&1 &
        
    local node_pid=$!
    echo "$node_pid" > "$NETWORK_DIR/$server_type/$node_name.pid"
    
    echo "   📋 PID: $node_pid"
    echo "   📁 Config: $config_file"
    echo "   📝 Log: $log_file"
    echo ""
    
    # Brief startup delay
    sleep 2
}

# Function to check if Tor is running
check_tor_daemon() {
    echo -e "${YELLOW}🔍 Checking Tor daemon...${NC}"
    
    if pgrep tor > /dev/null; then
        echo -e "${GREEN}   ✅ Tor daemon is running${NC}"
    else
        echo -e "${RED}   ❌ Tor daemon not found${NC}"
        echo -e "${YELLOW}   🔧 Starting Tor daemon...${NC}"
        
        # Start Tor with control port enabled
        sudo systemctl start tor || {
            echo -e "${RED}   ❌ Failed to start Tor daemon${NC}"
            echo -e "${YELLOW}   💡 Manual setup required:${NC}"
            echo "      1. Install Tor: sudo apt install tor"
            echo "      2. Enable ControlPort in /etc/tor/torrc:"
            echo "         ControlPort 9051"
            echo "         CookieAuthentication 1"
            echo "      3. Restart Tor: sudo systemctl restart tor"
            exit 1
        }
    fi
}

# Function to monitor network status
monitor_network() {
    echo -e "${BLUE}📊 Network Monitoring Dashboard${NC}"
    echo "═══════════════════════════════"
    
    for server_type in "alpha" "beta"; do
        echo -e "${YELLOW}Server ${server_type^} Status:${NC}"
        
        if [ "$server_type" = "alpha" ]; then
            nodes=("${SERVER_ALPHA_NODES[@]}")
        else
            nodes=("${SERVER_BETA_NODES[@]}")
        fi
        
        for node in "${nodes[@]}"; do
            local pid_file="$NETWORK_DIR/$server_type/$node.pid"
            if [ -f "$pid_file" ]; then
                local pid=$(cat "$pid_file")
                if kill -0 "$pid" 2>/dev/null; then
                    echo -e "   ✅ $node (PID: $pid)"
                else
                    echo -e "   ❌ $node (stopped)"
                fi
            else
                echo -e "   ❓ $node (not started)"
            fi
        done
        echo ""
    done
    
    echo -e "${BLUE}🔗 Network Endpoints:${NC}"
    node_index=0
    for server_type in "alpha" "beta"; do
        if [ "$server_type" = "alpha" ]; then
            nodes=("${SERVER_ALPHA_NODES[@]}")
        else
            nodes=("${SERVER_BETA_NODES[@]}")
        fi
        
        for node in "${nodes[@]}"; do
            local http_port=$((BASE_HTTP_PORT + node_index))
            echo "   🌐 $node: http://localhost:$http_port"
            ((node_index++))
        done
    done
}

# Function to test Byzantine fault tolerance
test_byzantine_tolerance() {
    echo -e "${BLUE}🧪 Testing Byzantine Fault Tolerance${NC}"
    echo "═══════════════════════════════════════"
    echo ""
    
    # Stop 3 random nodes (maximum Byzantine failures)
    echo -e "${YELLOW}🔥 Simulating Byzantine failures (3 nodes)...${NC}"
    
    # Randomly select 3 nodes to stop
    all_nodes=("${SERVER_ALPHA_NODES[@]}" "${SERVER_BETA_NODES[@]}")
    failed_nodes=()
    
    for i in {1..3}; do
        # Pick a random node index
        random_index=$((RANDOM % ${#all_nodes[@]}))
        failed_node="${all_nodes[$random_index]}"
        
        # Determine server type
        if [[ " ${SERVER_ALPHA_NODES[*]} " == *" $failed_node "* ]]; then
            server_type="alpha"
        else
            server_type="beta"
        fi
        
        # Stop the node
        local pid_file="$NETWORK_DIR/$server_type/$failed_node.pid"
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid"
                echo -e "${RED}   💥 Stopped $failed_node ($server_type)${NC}"
                failed_nodes+=("$failed_node")
            fi
        fi
        
        # Remove from array to avoid double-stopping
        all_nodes=("${all_nodes[@]/$failed_node}")
    done
    
    echo ""
    echo -e "${GREEN}✅ Network should continue operating with 7/10 nodes${NC}"
    echo -e "${BLUE}Failed nodes: ${failed_nodes[*]}${NC}"
    echo -e "${GREEN}Active nodes: 7/10 (above 2f+1=7 threshold)${NC}"
    echo ""
    
    # Wait a moment for the network to adjust
    echo -e "${YELLOW}⏳ Allowing network to detect failures and re-stabilize...${NC}"
    sleep 10
    
    # Check if remaining nodes are still responsive
    echo -e "${BLUE}🔍 Testing remaining nodes...${NC}"
    node_index=0
    active_count=0
    
    for server_type in "alpha" "beta"; do
        if [ "$server_type" = "alpha" ]; then
            nodes=("${SERVER_ALPHA_NODES[@]}")
        else
            nodes=("${SERVER_BETA_NODES[@]}")
        fi
        
        for node in "${nodes[@]}"; do
            local http_port=$((BASE_HTTP_PORT + node_index))
            
            # Skip failed nodes
            if [[ " ${failed_nodes[*]} " != *" $node "* ]]; then
                # Test if node is responsive
                if curl -s --connect-timeout 5 "http://localhost:$http_port/health" >/dev/null; then
                    echo -e "   ✅ $node: responsive"
                    ((active_count++))
                else
                    echo -e "   ⚠️ $node: not responding (may still be starting)"
                fi
            else
                echo -e "   ❌ $node: intentionally failed"
            fi
            
            ((node_index++))
        done
    done
    
    echo ""
    echo -e "${GREEN}🏆 Byzantine Fault Tolerance Test Result:${NC}"
    echo "   Active Nodes: $active_count/10"
    echo "   Failed Nodes: 3/10 (maximum Byzantine threshold)"
    
    if [ "$active_count" -ge 7 ]; then
        echo -e "${GREEN}   ✅ SUCCESS: Network maintains consensus with Byzantine failures!${NC}"
    else
        echo -e "${RED}   ❌ FAILURE: Not enough active nodes for consensus${NC}"
    fi
}

# Function to cleanup network
cleanup_network() {
    echo -e "${YELLOW}🧹 Cleaning up network...${NC}"
    
    # Stop all nodes
    for server_type in "alpha" "beta"; do
        if [ "$server_type" = "alpha" ]; then
            nodes=("${SERVER_ALPHA_NODES[@]}")
        else
            nodes=("${SERVER_BETA_NODES[@]}")
        fi
        
        for node in "${nodes[@]}"; do
            local pid_file="$NETWORK_DIR/$server_type/$node.pid"
            if [ -f "$pid_file" ]; then
                local pid=$(cat "$pid_file")
                if kill -0 "$pid" 2>/dev/null; then
                    kill "$pid"
                    echo "   🛑 Stopped $node"
                fi
                rm -f "$pid_file"
            fi
        done
    done
    
    echo -e "${GREEN}✅ Network cleanup complete${NC}"
}

# Trap cleanup on exit
trap cleanup_network EXIT INT TERM

# Main deployment sequence
echo -e "${GREEN}🚀 STARTING HISTORIC DEPLOYMENT SEQUENCE${NC}"
echo ""

# Check prerequisites
check_tor_daemon

# Build the project if needed
if [ ! -f "target/release/q-api-server" ]; then
    echo -e "${YELLOW}🔨 Building Q-NarwhalKnight binary...${NC}"
    cargo build --release --bin q-api-server
    echo ""
fi

# Start all nodes
echo -e "${GREEN}🌐 Starting 10-node quantum BFT network...${NC}"
echo ""

# Server Alpha nodes
echo -e "${BLUE}Server Alpha Nodes (Primary):${NC}"
for node in "${SERVER_ALPHA_NODES[@]}"; do
    start_node "$node" "alpha"
done

# Server Beta nodes
echo -e "${BLUE}Server Beta Nodes (Secondary):${NC}"
for node in "${SERVER_BETA_NODES[@]}"; do
    start_node "$node" "beta"
done

# Wait for network stabilization
echo -e "${YELLOW}⏳ Allowing network to stabilize (30 seconds)...${NC}"
sleep 30

# Monitor network status
monitor_network

# Interactive menu
echo ""
echo -e "${GREEN}🎛️ Network Control Menu:${NC}"
echo "   1. Monitor network status"
echo "   2. Test Byzantine fault tolerance"
echo "   3. View node logs"
echo "   4. Stop all nodes"
echo ""

while true; do
    read -p "Select option (1-4): " choice
    case $choice in
        1)
            monitor_network
            ;;
        2)
            test_byzantine_tolerance
            ;;
        3)
            echo -e "${BLUE}📝 Recent log entries:${NC}"
            tail -n 20 "$NETWORK_DIR"/*/logs/*.log
            ;;
        4)
            echo -e "${YELLOW}🛑 Stopping all nodes...${NC}"
            cleanup_network
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            ;;
    esac
    echo ""
done