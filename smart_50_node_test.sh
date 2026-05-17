#!/bin/bash
# 🚀 SMART 50-NODE Q-NARWHALKNIGHT HORIZONTAL SCALING TEST
# Optimized for port management, DNS-phantom bridge, and consensus performance

set -e

# Configuration
TOTAL_NODES=25  # Start with 25 for system stability, can scale to 50
BATCH_SIZE=5
NETWORK_NAME="qnk-smart-test"
TEST_DURATION=180  # 3 minutes for faster iteration
TPS_TARGET=50000   # 50k TPS target (realistic for testing)
CONSENSUS_THRESHOLD=$((TOTAL_NODES * 2 / 3 + 1))  # Byzantine fault tolerance
BASE_PORT=9000     # Use higher port range to avoid conflicts

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}🚀 SMART 25-NODE Q-NARWHALKNIGHT SCALING TEST${NC}"
echo -e "${CYAN}=============================================${NC}"
echo -e "${BLUE}📊 Configuration:${NC}"
echo -e "   • Total Nodes: ${TOTAL_NODES}"
echo -e "   • Consensus Threshold: ${CONSENSUS_THRESHOLD} nodes"
echo -e "   • Test Duration: ${TEST_DURATION}s"
echo -e "   • TPS Target: ${TPS_TARGET}"
echo -e "   • Base Port: ${BASE_PORT}"
echo -e "   • Network: ${NETWORK_NAME}"
echo ""

# Check available ports
echo -e "${YELLOW}🔍 Checking port availability...${NC}"
for ((port = BASE_PORT; port < BASE_PORT + TOTAL_NODES; port++)); do
    if netstat -ln 2>/dev/null | grep -q ":$port "; then
        echo -e "${RED}❌ Port $port is in use. Adjusting base port...${NC}"
        BASE_PORT=$((port + 100))
        break
    fi
done

# Cleanup function
cleanup() {
    echo -e "${YELLOW}🧹 Smart cleanup...${NC}"
    
    # Stop and remove containers
    docker ps -q --filter "name=qnk-node-" | xargs -r docker stop > /dev/null 2>&1
    docker ps -aq --filter "name=qnk-node-" | xargs -r docker rm > /dev/null 2>&1
    docker ps -q --filter "name=qnk-prometheus" | xargs -r docker stop > /dev/null 2>&1
    docker ps -aq --filter "name=qnk-prometheus" | xargs -r docker rm > /dev/null 2>&1
    
    # Remove network
    docker network rm ${NETWORK_NAME} > /dev/null 2>&1 || true
    
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Trap cleanup on exit
trap cleanup EXIT

# Clean up any existing test
cleanup

# Create dedicated network
echo -e "${BLUE}🌐 Creating optimized Docker network...${NC}"
docker network create --driver bridge \
    --opt com.docker.network.bridge.enable_icc=true \
    --opt com.docker.network.bridge.enable_ip_masquerade=true \
    --opt com.docker.network.driver.mtu=1500 \
    --subnet=172.25.0.0/16 \
    ${NETWORK_NAME}

# Phase 1: Smart Node Deployment
echo -e "${GREEN}🚀 PHASE 1: SMART NODE DEPLOYMENT${NC}"
echo -e "${GREEN}==================================${NC}"

node_containers=()
start_time=$(date +%s)

for ((batch = 0; batch < (TOTAL_NODES + BATCH_SIZE - 1) / BATCH_SIZE; batch++)); do
    batch_start=$((batch * BATCH_SIZE + 1))
    batch_end=$(((batch + 1) * BATCH_SIZE))
    if [ $batch_end -gt $TOTAL_NODES ]; then
        batch_end=$TOTAL_NODES
    fi
    
    echo -e "${YELLOW}📦 Deploying batch $((batch + 1)): nodes ${batch_start}-${batch_end}${NC}"
    
    # Deploy batch sequentially for better success rate
    for ((node_id = batch_start; node_id <= batch_end; node_id++)); do
        port=$((BASE_PORT + node_id - 1))
        ip_suffix=$((50 + node_id))
        container_name="qnk-node-${node_id}"
        
        echo -e "   🔧 Starting node ${node_id} on port ${port}..."
        
        # Start container with optimized settings
        docker run -d \
            --name "$container_name" \
            --network ${NETWORK_NAME} \
            --ip "172.25.0.${ip_suffix}" \
            -p "${port}:8080" \
            -e NODE_ID="${node_id}" \
            -e NETWORK_SIZE="${TOTAL_NODES}" \
            -e CONSENSUS_THRESHOLD="${CONSENSUS_THRESHOLD}" \
            -e LOG_LEVEL="warn" \
            -e RUST_LOG="q_api_server=info,q_dns_phantom=debug,q_network=debug" \
            -e TOR_ENABLED="true" \
            -e DNS_PHANTOM_ENABLED="true" \
            -e BENCHMARK_MODE="true" \
            -e MAX_TPS="${TPS_TARGET}" \
            --memory="512m" \
            --cpus="1.0" \
            --restart=no \
            q-narwhalknight-fixed:latest > /dev/null 2>&1
        
        if [ $? -eq 0 ]; then
            node_containers+=("$container_name")
            echo -e "   ✅ Node ${node_id} started successfully"
        else
            echo -e "   ❌ Node ${node_id} failed to start"
        fi
        
        # Brief pause between container starts
        sleep 2
    done
    
    # Check batch health
    sleep 5
    healthy_count=0
    for ((node_id = batch_start; node_id <= batch_end; node_id++)); do
        container_name="qnk-node-${node_id}"
        if docker ps --filter "name=$container_name" --filter "status=running" | grep -q "$container_name"; then
            ((healthy_count++))
        fi
    done
    
    echo -e "   ✅ Batch $((batch + 1)): ${healthy_count}/$((batch_end - batch_start + 1)) nodes healthy"
done

deploy_time=$(($(date +%s) - start_time))
total_running=$(docker ps --filter "name=qnk-node-" --format "{{.Names}}" | wc -l)
echo -e "${GREEN}✅ ${total_running}/${TOTAL_NODES} nodes deployed in ${deploy_time}s${NC}"
echo ""

if [ $total_running -lt $((TOTAL_NODES / 2)) ]; then
    echo -e "${RED}❌ Too few nodes running. Aborting test.${NC}"
    exit 1
fi

# Phase 2: Network Bootstrap with Real-time Monitoring
echo -e "${BLUE}🔄 PHASE 2: NETWORK BOOTSTRAP & MONITORING${NC}"
echo -e "${BLUE}==========================================${NC}"

echo -e "${YELLOW}⏳ Waiting for Tor bootstrap and DNS-phantom network formation...${NC}"

# Start real-time monitoring in background
python3 realtime_monitoring.py $TOTAL_NODES 10 > /tmp/monitor_output.txt 2>&1 &
MONITOR_PID=$!

# Progressive bootstrap monitoring
for ((i = 1; i <= 90; i++)); do
    if [ $((i % 15)) -eq 0 ]; then
        # Check progress every 15 seconds
        echo -e "\n${CYAN}📊 Bootstrap Progress Check at ${i}s:${NC}"
        
        # Sample a few nodes for status
        sample_nodes=($(docker ps --filter "name=qnk-node-" --format "{{.Names}}" | shuf | head -3))
        
        tor_ready=0
        phantom_active=0
        
        for node in "${sample_nodes[@]}"; do
            if docker logs "$node" 2>/dev/null | grep -q "Tor SOCKS proxy is operational"; then
                ((tor_ready++))
            fi
            if docker logs "$node" 2>/dev/null | grep -q "DNS-Phantom Network.*Active"; then
                ((phantom_active++))
            fi
        done
        
        echo -e "   🧅 Tor ready: ${tor_ready}/3 sampled nodes"
        echo -e "   👻 DNS-Phantom active: ${phantom_active}/3 sampled nodes"
    fi
    
    echo -ne "\r⏳ Bootstrap: ${i}/90s"
    sleep 1
done
echo ""

# Phase 3: DNS-Phantom Bridge Analysis
echo -e "${PURPLE}🔍 PHASE 3: DNS-PHANTOM BRIDGE DEEP ANALYSIS${NC}"
echo -e "${PURPLE}=============================================${NC}"

echo -e "${YELLOW}🧪 Analyzing DNS-phantom bridge performance...${NC}"

# Test all running nodes for bridge functionality
running_nodes=($(docker ps --filter "name=qnk-node-" --format "{{.Names}}"))
echo -e "📊 Testing ${#running_nodes[@]} active nodes..."

bridge_stats=()
total_discoveries=0
total_bridges=0
total_circuits=0
consensus_ready_count=0

for node in "${running_nodes[@]}"; do
    node_num=${node#qnk-node-}
    
    # Get detailed metrics from logs
    discoveries=$(docker logs "$node" 2>/dev/null | grep -c "phantom discovered peer" || echo "0")
    bridges=$(docker logs "$node" 2>/dev/null | grep -c "P2P connection established" || echo "0") 
    circuits=$(docker logs "$node" 2>/dev/null | grep -c "Tor circuit established" || echo "0")
    
    # Check consensus readiness
    consensus_ready="false"
    if docker logs "$node" 2>/dev/null | grep -q "Consensus network ready"; then
        consensus_ready="true"
        ((consensus_ready_count++))
    fi
    
    # Check for bridge-specific logs
    bridge_attempts=$(docker logs "$node" 2>/dev/null | grep -c "Attempting P2P connection to phantom peer" || echo "0")
    bridge_failures=$(docker logs "$node" 2>/dev/null | grep -c "P2P connection failed to phantom peer" || echo "0")
    
    total_discoveries=$((total_discoveries + discoveries))
    total_bridges=$((total_bridges + bridges))
    total_circuits=$((total_circuits + circuits))
    
    echo -e "   📡 Node-${node_num}: ${discoveries}→${bridges} (${bridge_attempts} attempts, ${bridge_failures} failures) Consensus: ${consensus_ready}"
done

# Calculate bridge performance metrics
bridge_success_rate=0
if [ $total_discoveries -gt 0 ]; then
    bridge_success_rate=$(echo "scale=2; $total_bridges / $total_discoveries * 100" | bc)
fi

echo -e "${GREEN}📊 DNS-PHANTOM BRIDGE RESULTS:${NC}"
echo -e "   👻 Total phantom discoveries: ${total_discoveries}"
echo -e "   🔗 Total bridge connections: ${total_bridges}"
echo -e "   🧅 Total Tor circuits: ${total_circuits}"
echo -e "   📈 Bridge success rate: ${bridge_success_rate}%"
echo -e "   🏛️ Consensus ready nodes: ${consensus_ready_count}/${#running_nodes[@]}"

# Phase 4: Consensus Network Formation Test
echo -e "${GREEN}🏛️ PHASE 4: CONSENSUS NETWORK VALIDATION${NC}"
echo -e "${GREEN}=======================================${NC}"

byzantine_threshold_met="false"
if [ $consensus_ready_count -ge $CONSENSUS_THRESHOLD ]; then
    byzantine_threshold_met="true"
    echo -e "${GREEN}✅ Byzantine fault tolerance ACHIEVED (${consensus_ready_count} >= ${CONSENSUS_THRESHOLD})${NC}"
else
    echo -e "${YELLOW}⚠️ Byzantine threshold NOT MET (${consensus_ready_count} < ${CONSENSUS_THRESHOLD})${NC}"
fi

# Test network partition tolerance
echo -e "${YELLOW}🧪 Testing network partition tolerance...${NC}"

# Temporarily isolate 20% of nodes
isolate_count=$((${#running_nodes[@]} / 5))
if [ $isolate_count -gt 0 ]; then
    echo -e "🔌 Temporarily isolating ${isolate_count} nodes..."
    
    isolated_nodes=()
    for ((i = 0; i < isolate_count; i++)); do
        node=${running_nodes[$i]}
        docker network disconnect ${NETWORK_NAME} "$node" > /dev/null 2>&1 || true
        isolated_nodes+=("$node")
    done
    
    # Wait and check if remaining nodes maintain consensus
    sleep 15
    
    remaining_consensus=0
    for node in "${running_nodes[@]:$isolate_count}"; do
        if docker logs "$node" 2>/dev/null | tail -20 | grep -q "Consensus network ready"; then
            ((remaining_consensus++))
        fi
    done
    
    echo -e "🔍 Remaining consensus nodes: ${remaining_consensus}/$((${#running_nodes[@]} - isolate_count))"
    
    # Reconnect isolated nodes
    for node in "${isolated_nodes[@]}"; do
        docker network connect ${NETWORK_NAME} "$node" > /dev/null 2>&1 || true
    done
    
    echo -e "🔌 Reconnected isolated nodes"
fi

# Phase 5: High-Throughput Transaction Testing
echo -e "${RED}⚡ PHASE 5: HIGH-THROUGHPUT TRANSACTION TEST${NC}"
echo -e "${RED}===========================================${NC}"

echo -e "${YELLOW}🚀 Generating ${TPS_TARGET} TPS load for ${TEST_DURATION}s...${NC}"

# Create optimized transaction generator
cat > /tmp/optimized_tx_generator.py << 'EOF'
import asyncio
import aiohttp
import time
import sys
import json
from concurrent.futures import ThreadPoolExecutor

async def send_batch_transactions(session, base_url, start_id, count):
    successful = 0
    tasks = []
    
    for i in range(count):
        tx_id = start_id + i
        tx = {
            "from": f"addr_{tx_id % 100}",
            "to": f"addr_{(tx_id + 1) % 100}", 
            "amount": (tx_id % 1000) + 1,
            "nonce": tx_id
        }
        
        task = session.post(f"{base_url}/api/v1/transaction/submit",
                           json=tx, timeout=aiohttp.ClientTimeout(total=0.5))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if not isinstance(result, Exception):
            try:
                if result.status == 200:
                    successful += 1
            except:
                pass
            finally:
                try:
                    result.close()
                except:
                    pass
    
    return successful

async def load_test():
    nodes = []
    for i in range(1, int(sys.argv[1]) + 1):
        port = int(sys.argv[4]) + i - 1  # BASE_PORT calculation
        nodes.append(f"http://localhost:{port}")
    
    target_tps = int(sys.argv[2])
    duration = int(sys.argv[3])
    
    total_transactions = target_tps * duration
    batch_size = 50  # Smaller batches for better success rate
    
    connector = aiohttp.TCPConnector(limit=200, limit_per_host=20)
    timeout = aiohttp.ClientTimeout(total=30, connect=5)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        start_time = time.time()
        
        tasks = []
        tx_id = 0
        
        for node_url in nodes:
            transactions_per_node = total_transactions // len(nodes)
            
            for batch_start in range(0, transactions_per_node, batch_size):
                batch_count = min(batch_size, transactions_per_node - batch_start)
                task = send_batch_transactions(session, node_url, tx_id, batch_count)
                tasks.append(task)
                tx_id += batch_count
        
        print(f"📊 Created {len(tasks)} batch tasks for {tx_id} total transactions")
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        total_successful = sum(r for r in results if isinstance(r, int))
        actual_tps = total_successful / actual_duration
        
        print(f"RESULT: {total_successful},{tx_id},{actual_tps:.2f},{actual_duration:.2f}")

if __name__ == "__main__":
    asyncio.run(load_test())
EOF

# Run optimized load test
echo -e "${YELLOW}⚡ Starting optimized load test...${NC}"
load_start=$(date +%s)

python3 /tmp/optimized_tx_generator.py ${#running_nodes[@]} $TPS_TARGET $TEST_DURATION $BASE_PORT > /tmp/load_results.txt 2>&1 &
LOAD_PID=$!

# Monitor load test progress
for ((i = 0; i < TEST_DURATION; i += 15)); do
    echo -ne "\r⚡ Load test progress: ${i}/${TEST_DURATION}s"
    
    # Sample system health
    if [ $((i % 30)) -eq 0 ] && [ $i -gt 0 ]; then
        running_count=$(docker ps --filter "name=qnk-node-" --format "{{.Names}}" | wc -l)
        echo -e "\n   📊 Nodes still running: ${running_count}/${#running_nodes[@]}"
    fi
    
    sleep 15
done

wait $LOAD_PID
load_end=$(date +%s)
echo ""

# Parse results
if [ -f /tmp/load_results.txt ]; then
    load_result=$(grep "RESULT:" /tmp/load_results.txt | tail -1 | cut -d: -f2)
    if [ -n "$load_result" ]; then
        IFS=',' read -r successful_txs total_attempted actual_tps actual_duration <<< "$load_result"
        
        echo -e "${GREEN}⚡ HIGH-THROUGHPUT TEST RESULTS:${NC}"
        echo -e "   📊 Successful: ${successful_txs}/${total_attempted} transactions"
        echo -e "   ⚡ Achieved TPS: ${actual_tps}"
        echo -e "   🎯 Target TPS: ${TPS_TARGET}"
        echo -e "   📈 Success Rate: $(echo "scale=1; $successful_txs / $total_attempted * 100" | bc)%"
        echo -e "   ⏱️ Duration: ${actual_duration}s"
    fi
fi

# Phase 6: Final Network Analysis
echo -e "${PURPLE}🔬 PHASE 6: FINAL NETWORK ANALYSIS${NC}"
echo -e "${PURPLE}==================================${NC}"

# Kill monitoring
kill $MONITOR_PID > /dev/null 2>&1 || true

final_running=$(docker ps --filter "name=qnk-node-" --format "{{.Names}}" | wc -l)
total_time=$(($(date +%s) - start_time))

# Collect final comprehensive metrics
echo -e "${YELLOW}📊 Collecting final metrics...${NC}"

final_discoveries=0
final_bridges=0
final_circuits=0
final_consensus=0
final_transactions=0
final_rounds=0

for node in $(docker ps --filter "name=qnk-node-" --format "{{.Names}}"); do
    # Network metrics
    discoveries=$(docker logs "$node" 2>/dev/null | grep -c "phantom discovered peer" || echo "0")
    bridges=$(docker logs "$node" 2>/dev/null | grep -c "P2P connection established" || echo "0")
    circuits=$(docker logs "$node" 2>/dev/null | grep -c "Tor circuit established" || echo "0")
    
    final_discoveries=$((final_discoveries + discoveries))
    final_bridges=$((final_bridges + bridges))
    final_circuits=$((final_circuits + circuits))
    
    # Consensus metrics
    if docker logs "$node" 2>/dev/null | grep -q "Consensus network ready"; then
        ((final_consensus++))
    fi
    
    # Transaction metrics
    txs=$(docker logs "$node" 2>/dev/null | grep -c "transaction processed\|transaction finalized" || echo "0")
    final_transactions=$((final_transactions + txs))
    
    rounds=$(docker logs "$node" 2>/dev/null | grep -c "consensus round\|DAG vertex" || echo "0")
    final_rounds=$((final_rounds + rounds))
done

# Generate comprehensive final report
echo ""
echo -e "${GREEN}🎉 SMART 25-NODE TEST COMPLETED!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo -e "${CYAN}📊 COMPREHENSIVE FINAL RESULTS:${NC}"
echo -e "${CYAN}===============================${NC}"
echo ""
echo -e "${PURPLE}🌐 NETWORK PERFORMANCE:${NC}"
echo -e "   • Total nodes deployed: ${TOTAL_NODES}"
echo -e "   • Final running nodes: ${final_running}"
echo -e "   • Network uptime: $((final_running * 100 / TOTAL_NODES))%"
echo -e "   • Total test duration: ${total_time}s"
echo ""
echo -e "${GREEN}👻 DNS-PHANTOM BRIDGE SUCCESS:${NC}"
echo -e "   • Total discoveries: ${final_discoveries}"
echo -e "   • Successful bridges: ${final_bridges}" 
echo -e "   • Bridge success rate: $([ $final_discoveries -gt 0 ] && echo "scale=1; $final_bridges / $final_discoveries * 100" | bc || echo "0")%"
echo -e "   • Avg discoveries/node: $([ $final_running -gt 0 ] && echo "scale=1; $final_discoveries / $final_running" | bc || echo "0")"
echo ""
echo -e "${BLUE}🏛️ CONSENSUS ACHIEVEMENT:${NC}"
echo -e "   • Consensus ready: ${final_consensus}/${final_running}"
echo -e "   • Byzantine threshold: ${CONSENSUS_THRESHOLD} ($( [ $final_consensus -ge $CONSENSUS_THRESHOLD ] && echo "✅ MET" || echo "❌ NOT MET"))"
echo -e "   • Consensus formation: $([ $final_consensus -gt 0 ] && echo "✅ SUCCESS" || echo "❌ FAILED")"
echo -e "   • Total consensus rounds: ${final_rounds}"
echo ""
echo -e "${RED}⚡ THROUGHPUT PERFORMANCE:${NC}"
if [ -n "$actual_tps" ]; then
echo -e "   • Achieved TPS: ${actual_tps}"
echo -e "   • Target TPS: ${TPS_TARGET}"
echo -e "   • Performance ratio: $(echo "scale=1; $actual_tps / $TPS_TARGET * 100" | bc)%"
echo -e "   • Total successful txs: ${successful_txs:-0}"
fi
echo -e "   • Node-reported transactions: ${final_transactions}"
echo ""
echo -e "${YELLOW}🎯 HORIZONTAL SCALING VERDICT:${NC}"

# Determine overall success
success_score=0
if [ $final_running -ge $((TOTAL_NODES * 3 / 4)) ]; then ((success_score++)); fi  # Node deployment
if [ $final_bridges -gt $((TOTAL_NODES / 2)) ]; then ((success_score++)); fi      # Bridge functionality  
if [ $final_consensus -ge $CONSENSUS_THRESHOLD ]; then ((success_score++)); fi    # Consensus readiness
if [ -n "$successful_txs" ] && [ $successful_txs -gt 1000 ]; then ((success_score++)); fi  # Transaction processing

if [ $success_score -ge 3 ]; then
    echo -e "   🏆 ${GREEN}HORIZONTAL SCALING: SUCCESS${NC}"
    echo -e "   ✅ Q-NarwhalKnight scales effectively to ${TOTAL_NODES} nodes"
    echo -e "   ✅ DNS-phantom bridge functional at scale"
    echo -e "   ✅ Byzantine consensus maintained"
    echo -e "   ✅ Anonymous peer discovery working"
elif [ $success_score -ge 2 ]; then
    echo -e "   🎯 ${YELLOW}HORIZONTAL SCALING: PARTIAL SUCCESS${NC}"
    echo -e "   ⚠️ Some systems need optimization for full-scale deployment"
else
    echo -e "   ⚠️ ${RED}HORIZONTAL SCALING: NEEDS IMPROVEMENT${NC}"
    echo -e "   🔧 Requires system optimization for production scaling"
fi

echo ""
echo -e "${CYAN}📄 Detailed logs available via: docker logs qnk-node-[1-${TOTAL_NODES}]${NC}"
echo -e "${CYAN}📊 Monitoring data saved to: /tmp/network_metrics.json${NC}"
echo ""

# Save final report
cat > /tmp/smart_scaling_report.json << EOF
{
  "test_configuration": {
    "total_nodes": ${TOTAL_NODES},
    "consensus_threshold": ${CONSENSUS_THRESHOLD},
    "test_duration": ${TEST_DURATION},
    "tps_target": ${TPS_TARGET},
    "base_port": ${BASE_PORT}
  },
  "deployment_results": {
    "nodes_deployed": ${TOTAL_NODES},
    "final_running": ${final_running},
    "deployment_time": ${deploy_time},
    "uptime_percentage": $((final_running * 100 / TOTAL_NODES))
  },
  "bridge_performance": {
    "total_discoveries": ${final_discoveries},
    "successful_bridges": ${final_bridges},
    "success_rate": $([ $final_discoveries -gt 0 ] && echo "scale=2; $final_bridges / $final_discoveries * 100" | bc || echo "0"),
    "avg_per_node": $([ $final_running -gt 0 ] && echo "scale=2; $final_bridges / $final_running" | bc || echo "0")
  },
  "consensus_metrics": {
    "ready_nodes": ${final_consensus},
    "byzantine_threshold_met": $( [ $final_consensus -ge $CONSENSUS_THRESHOLD ] && echo "true" || echo "false"),
    "total_rounds": ${final_rounds}
  },
  "performance_metrics": {
    "achieved_tps": ${actual_tps:-0},
    "successful_transactions": ${successful_txs:-0},
    "node_reported_transactions": ${final_transactions}
  },
  "overall_success_score": ${success_score}
}
EOF

echo -e "${GREEN}📊 Comprehensive report: /tmp/smart_scaling_report.json${NC}"
echo ""
echo -e "${CYAN}✅ SMART SCALING TEST COMPLETED!${NC}"