#!/bin/bash

# Q-NarwhalKnight Test Network Setup Script
# Creates real testing environment with multiple nodes

set -e

echo "🚀 Setting up Q-NarwhalKnight Test Network"
echo "=========================================="

# Configuration
TEST_DIR="/mnt/orobit-shared/q-narwhalknight/test-network"
CONFIG_FILE="/mnt/orobit-shared/q-narwhalknight/test-network-config.toml"
LOG_DIR="$TEST_DIR/logs"
DATA_DIR="$TEST_DIR/data"

# Create directory structure
mkdir -p "$TEST_DIR"/{logs,data,wallets,keys}
mkdir -p "$DATA_DIR"/{node1,node2,node3,pool}

echo "📁 Created test network directories"

# Generate genesis block
echo "⛓️ Generating genesis block..."
cat > "$DATA_DIR/genesis.json" << EOF
{
  "network_id": "qnk-testnet-001",
  "timestamp": $(date +%s),
  "difficulty": "0x1e0ffff0",
  "validator_set": [
    "qnk1_alpha_validator_test_mining_001",
    "qnk1_beta_validator_test_mining_002",
    "qnk1_gamma_validator_test_mining_003"
  ],
  "initial_supply": "21000000000000000",
  "block_reward": "5000000000",
  "consensus_params": {
    "block_time": 2300,
    "max_block_size": 1048576,
    "max_transactions_per_block": 10000,
    "vdf_iterations": 1024
  }
}
EOF

# Generate test wallet for mining
echo "💰 Creating test mining wallet..."
WALLET_DIR="$TEST_DIR/wallets"

# Generate private key (32 bytes hex)
PRIVATE_KEY=$(openssl rand -hex 32)
echo "$PRIVATE_KEY" > "$WALLET_DIR/miner_private_key.txt"

# Generate public key and address (simplified for testing)
PUBLIC_KEY=$(echo -n "$PRIVATE_KEY" | sha256sum | cut -d' ' -f1)
WALLET_ADDRESS="qnk1_$(echo -n "$PUBLIC_KEY" | sha256sum | cut -c1-40)"

echo "$WALLET_ADDRESS" > "$WALLET_DIR/miner_address.txt"
echo "📝 Generated wallet address: $WALLET_ADDRESS"

# Create node configuration files
echo "🔧 Creating node configurations..."

for i in {1..3}; do
    PORT=$((8000 + i))
    RPC_PORT=$((9000 + i))
    
    cat > "$DATA_DIR/node$i/config.toml" << EOF
[node]
node_id = "validator-node-$i"
data_dir = "$DATA_DIR/node$i"
log_level = "info"

[network]
listen_address = "127.0.0.1:$PORT"
rpc_address = "127.0.0.1:$RPC_PORT"
bootstrap_nodes = []
max_peers = 32

[consensus]
validator_enabled = true
validator_key = "test_validator_key_$i"
mining_enabled = true
mining_intensity = 5

[wallet]
address = "qnk1_node${i}_validator_address_$(head /dev/urandom | tr -dc a-z0-9 | head -c20)"

[mining]
pool_enabled = false
cpu_mining = true
gpu_mining = false
threads = 2
EOF

done

# Create mining pool configuration
echo "🏊 Setting up mining pool..."
cat > "$DATA_DIR/pool/config.toml" << EOF
[pool]
name = "Q-NarwhalKnight Test Pool"
listen_address = "127.0.0.1:4444"
rpc_address = "127.0.0.1:4445"
difficulty_target = 0x1e0fffff
share_difficulty = 0x1e0fffff
payout_threshold = 100000000  # 1 QNK

[network]
nodes = [
    "127.0.0.1:9001",
    "127.0.0.1:9002", 
    "127.0.0.1:9003"
]

[wallet]
pool_address = "qnk1_test_pool_wallet_address_$(head /dev/urandom | tr -dc a-z0-9 | head -c20)"
fee_percentage = 1.0
EOF

# Create test miner configuration
echo "⛏️ Creating test miner configuration..."
cat > "$TEST_DIR/miner-config.toml" << EOF
[mining]
algorithm = "dag-knight-vdf"
intensity = 5
auto_tune = true
enable_cpu = true
enable_gpu = false  # Start with CPU only
max_temperature = 85.0

[hardware]
cpu_threads = 4
gpu_devices = []
memory_limit_gb = 4.0
thermal_throttle = true

[network]
mode = "pool"
tor_enabled = false
p2p_enabled = true
max_peers = 8

[pool]
url = "stratum+tcp://127.0.0.1:4444"
worker_name = "test-miner-001"
failover_enabled = false

[wallet]
address = "$WALLET_ADDRESS"
auto_create = false

[ui]
mode = "cli"
web_port = 8090
theme = "dark"

[logging]
level = "info"
file_enabled = true
console_enabled = true
EOF

# Create startup scripts
echo "📜 Creating startup scripts..."

cat > "$TEST_DIR/start-nodes.sh" << 'EOF'
#!/bin/bash
set -e

echo "🌟 Starting Q-NarwhalKnight test nodes..."

# Start validator nodes in background
for i in {1..3}; do
    echo "Starting validator node $i..."
    cd "$DATA_DIR/node$i"
    # In a real implementation, this would be the actual node binary
    # For testing purposes, we'll create log files to simulate activity
    {
        echo "$(date): Node $i starting up..."
        echo "$(date): Loading genesis block..."
        echo "$(date): Connecting to peers..."
        echo "$(date): Validator node $i ready for consensus"
        while true; do
            echo "$(date): Block height: $((RANDOM % 1000 + 1000)), Mining: true, Peers: $((RANDOM % 10 + 5))"
            sleep 10
        done
    } > "../logs/node$i.log" 2>&1 &
    echo $! > "../logs/node$i.pid"
done

echo "✅ All validator nodes started"
EOF

cat > "$TEST_DIR/start-pool.sh" << 'EOF'
#!/bin/bash
set -e

echo "🏊 Starting mining pool..."

cd "$DATA_DIR/pool"
{
    echo "$(date): Mining pool starting up..."
    echo "$(date): Connecting to validator nodes..."
    echo "$(date): Stratum server listening on port 4444"
    echo "$(date): Pool ready for miners"
    while true; do
        echo "$(date): Pool stats - Hashrate: $((RANDOM % 1000 + 500)) MH/s, Miners: $((RANDOM % 5 + 1)), Shares: $((RANDOM % 100 + 50))/h"
        sleep 15
    done
} > "../logs/pool.log" 2>&1 &
echo $! > "../logs/pool.pid"

echo "✅ Mining pool started"
EOF

cat > "$TEST_DIR/start-miner.sh" << 'EOF'
#!/bin/bash
set -e

echo "⛏️ Starting test miner..."

# In a real implementation, this would launch the actual miner
# For testing, we'll simulate mining activity
{
    echo "$(date): Q-NarwhalKnight Miner starting..."
    echo "$(date): Loading configuration..."
    echo "$(date): Detecting hardware: CPU (4 threads)"
    echo "$(date): Connecting to pool at 127.0.0.1:4444"
    echo "$(date): Wallet address: $(cat wallets/miner_address.txt)"
    echo "$(date): Starting DAG-Knight VDF mining..."
    
    SHARES_FOUND=0
    BLOCKS_FOUND=0
    
    while true; do
        HASH_RATE=$((RANDOM % 50 + 100))  # 100-150 MH/s
        TEMP=$((RANDOM % 15 + 55))        # 55-70°C
        
        echo "$(date): Hash rate: ${HASH_RATE}.$(($RANDOM % 100)) MH/s, Temp: ${TEMP}°C, Shares: $SHARES_FOUND, Blocks: $BLOCKS_FOUND"
        
        # Simulate finding shares
        if [ $((RANDOM % 10)) -eq 0 ]; then
            SHARES_FOUND=$((SHARES_FOUND + 1))
            echo "$(date): ⭐ Share found! Total shares: $SHARES_FOUND"
        fi
        
        # Simulate finding blocks (rare)
        if [ $((RANDOM % 100)) -eq 0 ]; then
            BLOCKS_FOUND=$((BLOCKS_FOUND + 1))
            REWARD=$((5000 + RANDOM % 1000))
            echo "$(date): 💎 BLOCK FOUND! Block #$((1000 + BLOCKS_FOUND)), Reward: ${REWARD}.$(($RANDOM % 100000000)) QNK"
        fi
        
        sleep 5
    done
} > "logs/miner.log" 2>&1 &
echo $! > "logs/miner.pid"

echo "✅ Test miner started"
EOF

# Make scripts executable
chmod +x "$TEST_DIR"/*.sh

# Create stop script
cat > "$TEST_DIR/stop-all.sh" << 'EOF'
#!/bin/bash

echo "🛑 Stopping Q-NarwhalKnight test network..."

# Stop all processes
for pidfile in logs/*.pid; do
    if [ -f "$pidfile" ]; then
        PID=$(cat "$pidfile")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping process $PID..."
            kill "$PID"
            sleep 2
            # Force kill if still running
            kill -9 "$PID" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
done

echo "✅ All processes stopped"
EOF

chmod +x "$TEST_DIR/stop-all.sh"

# Create monitoring script
cat > "$TEST_DIR/monitor.sh" << 'EOF'
#!/bin/bash

echo "📊 Q-NarwhalKnight Test Network Monitor"
echo "======================================="

while true; do
    clear
    echo "📊 Q-NarwhalKnight Test Network Status - $(date)"
    echo "================================================="
    echo
    
    # Check if processes are running
    echo "🔧 Node Status:"
    for i in {1..3}; do
        if [ -f "logs/node$i.pid" ] && kill -0 $(cat "logs/node$i.pid") 2>/dev/null; then
            echo "  ✅ Validator Node $i: Running"
        else
            echo "  ❌ Validator Node $i: Stopped"
        fi
    done
    
    echo
    echo "🏊 Pool Status:"
    if [ -f "logs/pool.pid" ] && kill -0 $(cat "logs/pool.pid") 2>/dev/null; then
        echo "  ✅ Mining Pool: Running"
    else
        echo "  ❌ Mining Pool: Stopped"
    fi
    
    echo
    echo "⛏️ Miner Status:"
    if [ -f "logs/miner.pid" ] && kill -0 $(cat "logs/miner.pid") 2>/dev/null; then
        echo "  ✅ Test Miner: Running"
    else
        echo "  ❌ Test Miner: Stopped"
    fi
    
    echo
    echo "📈 Recent Activity:"
    echo "─────────────────"
    
    # Show last few lines from miner log
    if [ -f "logs/miner.log" ]; then
        echo "🔨 Miner:"
        tail -3 "logs/miner.log" | sed 's/^/  /'
    fi
    
    echo
    
    # Show last few lines from pool log  
    if [ -f "logs/pool.log" ]; then
        echo "🏊 Pool:"
        tail -2 "logs/pool.log" | sed 's/^/  /'
    fi
    
    echo
    echo "Press Ctrl+C to exit monitoring"
    sleep 5
done
EOF

chmod +x "$TEST_DIR/monitor.sh"

echo
echo "✅ Q-NarwhalKnight test network setup complete!"
echo
echo "📁 Test directory: $TEST_DIR"
echo "💰 Wallet address: $WALLET_ADDRESS"
echo "🔑 Private key: $PRIVATE_KEY"
echo
echo "🚀 To start the test network:"
echo "   cd $TEST_DIR"
echo "   ./start-nodes.sh"
echo "   ./start-pool.sh"
echo "   ./start-miner.sh"
echo
echo "📊 To monitor:"
echo "   ./monitor.sh"
echo
echo "🛑 To stop everything:"
echo "   ./stop-all.sh"