# 🌐 Cross-Server 50-Node DNS-Phantom Discovery Instructions

## For Server Alpha: Deploy 50 nodes that auto-discover Server Beta via DNS-Phantom

### Prerequisites
- Docker installed and running
- Network connectivity to DNS providers (Cloudflare, Google, Quad9)
- Server Beta running with DNS-Phantom enabled nodes

---

## 🚀 STEP 1: Build the Docker Image

```bash
# Navigate to Q-NarwhalKnight directory
cd /opt/orobit/shared/q-narwhalknight

# Build the Docker image with DNS-Phantom + Tor support
docker build -t q-narwhalknight-phantom:latest \
  --build-arg ENABLE_DNS_PHANTOM=true \
  --build-arg ENABLE_TOR=true \
  --build-arg ENABLE_BEP44=true \
  -f Dockerfile .

# Verify image built successfully
docker images | grep q-narwhalknight-phantom
```

---

## 🌐 STEP 2: Create Dedicated Network

```bash
# Create Docker network for 50 nodes
docker network create \
  --driver bridge \
  --subnet=172.60.0.0/16 \
  --opt com.docker.network.bridge.enable_icc=true \
  --opt com.docker.network.bridge.enable_ip_masquerade=true \
  qnk-phantom-discovery

# Verify network created
docker network ls | grep qnk-phantom-discovery
```

---

## 🎯 STEP 3: Deploy 50 Nodes with DNS-Phantom Auto-Discovery

Create deployment script: `/tmp/deploy_50_phantom_nodes.sh`

```bash
#!/bin/bash

# Configuration
NETWORK_NAME="qnk-phantom-discovery"
IMAGE_NAME="q-narwhalknight-phantom:latest"
TOTAL_NODES=50
BASE_PORT=9000
BATCH_SIZE=10

# Server Beta's DNS-Phantom seeds (replace with actual)
PHANTOM_SEEDS="
phantom-seed-001.qnk.network
phantom-seed-002.qnk.network
beta-validator.qnk.onion
"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🚀 Deploying 50 DNS-Phantom Discovery Nodes${NC}"
echo -e "${CYAN}===========================================${NC}"

# Clean up any existing containers
echo -e "${YELLOW}🧹 Cleaning up existing containers...${NC}"
docker ps -aq --filter "name=phantom-node-" | xargs -r docker rm -f 2>/dev/null

# Deploy nodes in batches
for ((batch=0; batch<$((TOTAL_NODES/BATCH_SIZE)); batch++)); do
    start=$((batch * BATCH_SIZE + 1))
    end=$(((batch + 1) * BATCH_SIZE))
    
    echo -e "${GREEN}📦 Deploying batch $((batch+1)): nodes $start-$end${NC}"
    
    for ((i=$start; i<=$end; i++)); do
        PORT=$((BASE_PORT + i))
        IP_SUFFIX=$((100 + i))
        
        docker run -d \
            --name "phantom-node-$i" \
            --hostname "alpha-validator-$i" \
            --network "$NETWORK_NAME" \
            --ip "172.60.0.$IP_SUFFIX" \
            -p "$PORT:8080" \
            -e NODE_ID="alpha-$i" \
            -e NODE_TYPE="validator" \
            -e NETWORK_SIZE="100" \
            -e DNS_PHANTOM_ENABLED="true" \
            -e DNS_PHANTOM_MODE="active" \
            -e DNS_PHANTOM_SEEDS="$PHANTOM_SEEDS" \
            -e TOR_ENABLED="true" \
            -e TOR_SOCKS_PROXY="127.0.0.1:9050" \
            -e BEP44_ENABLED="true" \
            -e DISCOVERY_MODE="multi-tier" \
            -e LOG_LEVEL="info" \
            -e RUST_LOG="q_dns_phantom=debug,q_network=info,q_api_server=info" \
            -e AUTO_CONNECT_PEERS="true" \
            -e PHANTOM_ANNOUNCE_INTERVAL="30" \
            -e PHANTOM_PROVIDERS="cloudflare,google,quad9,opendns" \
            -e BOOTSTRAP_MODE="dns-phantom-first" \
            --memory="256m" \
            --cpus="0.5" \
            "$IMAGE_NAME" &
    done
    
    # Wait for batch to start
    wait
    sleep 5
    
    # Check batch health
    healthy=0
    for ((i=$start; i<=$end; i++)); do
        if docker ps --filter "name=phantom-node-$i" --filter "status=running" -q | grep -q .; then
            ((healthy++))
        fi
    done
    
    echo -e "   ✅ Batch $((batch+1)): $healthy/$BATCH_SIZE nodes running"
done

echo -e "${GREEN}✅ All 50 nodes deployed!${NC}"
```

Make it executable and run:
```bash
chmod +x /tmp/deploy_50_phantom_nodes.sh
/tmp/deploy_50_phantom_nodes.sh
```

---

## 📊 STEP 4: Monitor DNS-Phantom Discovery

Create monitoring script: `/tmp/monitor_phantom_discovery.sh`

```bash
#!/bin/bash

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}📊 DNS-PHANTOM DISCOVERY MONITOR${NC}"
echo -e "${CYAN}=================================${NC}"

while true; do
    clear
    echo -e "${CYAN}📊 DNS-PHANTOM DISCOVERY MONITOR - $(date)${NC}"
    echo -e "${CYAN}================================================${NC}"
    echo ""
    
    # Count running nodes
    RUNNING=$(docker ps --filter "name=phantom-node-" --filter "status=running" -q | wc -l)
    echo -e "${GREEN}🟢 Nodes Running: $RUNNING/50${NC}"
    echo ""
    
    # Aggregate discovery metrics
    PHANTOM_DISCOVERIES=0
    DNS_QUERIES=0
    TOR_CIRCUITS=0
    P2P_CONNECTIONS=0
    BETA_PEERS=0
    
    # Sample 10 random nodes for detailed analysis
    SAMPLE_NODES=$(docker ps --filter "name=phantom-node-" --format "{{.Names}}" | shuf | head -10)
    
    echo -e "${YELLOW}📡 Sampling 10 nodes for discovery metrics...${NC}"
    for node in $SAMPLE_NODES; do
        # Count DNS-Phantom discoveries
        phantom=$(docker logs "$node" 2>&1 | grep -c "phantom discovered peer" || echo 0)
        PHANTOM_DISCOVERIES=$((PHANTOM_DISCOVERIES + phantom))
        
        # Count DNS queries sent
        queries=$(docker logs "$node" 2>&1 | grep -c "DNS query sent" || echo 0)
        DNS_QUERIES=$((DNS_QUERIES + queries))
        
        # Count Tor circuits
        circuits=$(docker logs "$node" 2>&1 | grep -c "Tor circuit established" || echo 0)
        TOR_CIRCUITS=$((TOR_CIRCUITS + circuits))
        
        # Count P2P connections
        p2p=$(docker logs "$node" 2>&1 | grep -c "P2P connection established" || echo 0)
        P2P_CONNECTIONS=$((P2P_CONNECTIONS + p2p))
        
        # Count Server Beta peers discovered
        beta=$(docker logs "$node" 2>&1 | grep -c "beta-validator\|server-beta" || echo 0)
        BETA_PEERS=$((BETA_PEERS + beta))
        
        echo -e "   • $node: ${phantom} discoveries, ${p2p} connections"
    done
    
    # Extrapolate to full network
    TOTAL_PHANTOM=$((PHANTOM_DISCOVERIES * 5))  # Sample was 10 nodes, multiply by 5
    TOTAL_P2P=$((P2P_CONNECTIONS * 5))
    TOTAL_BETA=$((BETA_PEERS * 5))
    
    echo ""
    echo -e "${PURPLE}🌐 DISCOVERY STATISTICS (Estimated)${NC}"
    echo -e "${PURPLE}====================================${NC}"
    echo -e "👻 DNS-Phantom Discoveries: ~$TOTAL_PHANTOM"
    echo -e "🔍 DNS Queries Sent: ~$((DNS_QUERIES * 5))"
    echo -e "🧅 Tor Circuits: ~$((TOR_CIRCUITS * 5))"
    echo -e "🔗 P2P Connections: ~$TOTAL_P2P"
    echo -e "🎯 Server Beta Peers Found: ~$TOTAL_BETA"
    echo ""
    
    # Check for cross-server discovery
    echo -e "${BLUE}🔍 CROSS-SERVER DISCOVERY STATUS${NC}"
    echo -e "${BLUE}================================${NC}"
    
    # Look for Server Beta nodes in logs
    BETA_DISCOVERED=false
    for node in $(docker ps --filter "name=phantom-node-" --format "{{.Names}}" | head -5); do
        if docker logs "$node" 2>&1 | grep -q "Connected to beta-\|Discovered server-beta"; then
            BETA_DISCOVERED=true
            echo -e "${GREEN}✅ Server Beta nodes discovered via DNS-Phantom!${NC}"
            
            # Show some discovered Beta nodes
            docker logs "$node" 2>&1 | grep "beta-validator\|server-beta" | head -3 | while read line; do
                echo -e "   • $line"
            done
            break
        fi
    done
    
    if [ "$BETA_DISCOVERED" = false ]; then
        echo -e "${YELLOW}⏳ Waiting for Server Beta discovery...${NC}"
        echo -e "   Ensure Server Beta nodes are:"
        echo -e "   • Running with DNS-Phantom enabled"
        echo -e "   • Announcing to same DNS providers"
        echo -e "   • Using compatible phantom seeds"
    fi
    
    echo ""
    echo -e "${GREEN}📈 NETWORK FORMATION PROGRESS${NC}"
    echo -e "${GREEN}=============================${NC}"
    
    # Calculate network formation percentage
    FORMATION_PCT=$((TOTAL_P2P * 100 / (RUNNING * RUNNING)))
    
    # Progress bar
    printf "Network Mesh: ["
    for ((i=0; i<50; i++)); do
        if [ $i -lt $((FORMATION_PCT / 2)) ]; then
            printf "█"
        else
            printf "░"
        fi
    done
    printf "] ${FORMATION_PCT}%%\n"
    
    # DNS-Phantom activity indicator
    echo ""
    echo -e "${CYAN}🌊 DNS-PHANTOM ACTIVITY${NC}"
    
    # Show recent phantom queries
    RECENT_QUERIES=$(docker logs phantom-node-1 2>&1 | grep "phantom query" | tail -5)
    if [ -n "$RECENT_QUERIES" ]; then
        echo "$RECENT_QUERIES" | while read line; do
            echo -e "   • $line"
        done
    else
        echo -e "   ⏳ Waiting for phantom query activity..."
    fi
    
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to exit. Refreshing in 10s...${NC}"
    sleep 10
done
```

Make it executable and run:
```bash
chmod +x /tmp/monitor_phantom_discovery.sh
/tmp/monitor_phantom_discovery.sh
```

---

## 🔍 STEP 5: Verify Cross-Server Discovery

Run verification script: `/tmp/verify_cross_server.sh`

```bash
#!/bin/bash

echo "🔍 Verifying Cross-Server DNS-Phantom Discovery..."
echo "=================================================="

# Check if Alpha nodes discovered Beta nodes
DISCOVERIES=0
CONNECTIONS=0

for i in {1..10}; do
    NODE="phantom-node-$i"
    
    # Check logs for Server Beta discovery
    if docker logs "$NODE" 2>&1 | grep -q "Discovered.*beta\|phantom.*beta\|Connected.*beta"; then
        ((DISCOVERIES++))
        echo "✅ Node $i: Discovered Server Beta peers"
        
        # Check for actual connections
        if docker logs "$NODE" 2>&1 | grep -q "P2P connection.*beta\|Connected to.*beta"; then
            ((CONNECTIONS++))
            echo "   🔗 Established connection to Beta!"
        fi
    else
        echo "⏳ Node $i: Still searching..."
    fi
done

echo ""
echo "📊 RESULTS:"
echo "   • Nodes that discovered Beta: $DISCOVERIES/10"
echo "   • Nodes connected to Beta: $CONNECTIONS/10"

if [ $DISCOVERIES -gt 5 ]; then
    echo "🎉 SUCCESS: Cross-server DNS-Phantom discovery is working!"
else
    echo "⚠️  Discovery in progress. Ensure Server Beta nodes are running."
fi
```

---

## 🎯 STEP 6: Test DNS-Phantom Bridge

Test that discovered peers can exchange data:

```bash
# From any Alpha node, send test message to Beta
docker exec phantom-node-1 curl -X POST \
  http://localhost:8080/api/v1/phantom/broadcast \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello from Server Alpha via DNS-Phantom!",
    "target": "server-beta",
    "discovery_method": "dns-phantom"
  }'

# Check if Beta received it (on Server Beta)
docker logs qnk-node-1 | grep "Hello from Server Alpha"
```

---

## 📊 Expected Results

After 2-3 minutes, you should see:

1. **DNS-Phantom Activity**:
   - Steganographic DNS queries being sent
   - "phantom discovered peer" events in logs
   - DNS cache anomalies detected

2. **Cross-Server Discovery**:
   - Alpha nodes discovering Beta nodes
   - Onion addresses extracted from DNS responses
   - P2P connections established

3. **Network Formation**:
   - 50 Alpha nodes + Beta nodes forming mesh
   - Multi-tier discovery fallback working
   - Tor circuits for anonymous connections

---

## 🛠️ Troubleshooting

If discovery isn't working:

1. **Check DNS-Phantom is enabled on both servers**:
   ```bash
   docker logs phantom-node-1 | grep "DNS-Phantom.*initialized"
   ```

2. **Verify DNS providers are accessible**:
   ```bash
   docker exec phantom-node-1 nslookup google.com 8.8.8.8
   ```

3. **Check for firewall blocking DNS-over-HTTPS**:
   ```bash
   docker exec phantom-node-1 curl https://cloudflare-dns.com/dns-query
   ```

4. **Ensure compatible phantom seeds**:
   ```bash
   docker exec phantom-node-1 env | grep PHANTOM_SEEDS
   ```

---

## 🎉 Success Criteria

The test is successful when:
- ✅ 50 Alpha nodes running
- ✅ DNS-Phantom queries active
- ✅ Beta nodes discovered via DNS
- ✅ P2P connections established
- ✅ Cross-server mesh network formed

This proves the DNS-Phantom + Tor discovery system works across independent server deployments!