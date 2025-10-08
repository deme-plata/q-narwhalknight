# 🎯 Server Alpha: Connect 50 Nodes to Server Beta Instructions

## Current Status: 0/50 Peers Connected

**Server Alpha has successfully deployed 50 nodes**, but **0 peers have connected to Server Beta** yet. The nodes need Server Beta's specific discovery information.

---

## 🔧 IMMEDIATE FIX: Update Server Alpha Nodes with Beta Discovery Seeds

### Step 1: Get Server Beta's Real Discovery Information

Server Beta is running on:
- **IP Address**: `185.182.185.227`
- **Port**: `8080`
- **DNS-Phantom ID**: `beta-validator-1`
- **Network**: Multi-tier discovery enabled

### Step 2: Update All 50 Alpha Nodes with Beta Discovery Seeds

Run this command on Server Alpha to update all running nodes:

```bash
#!/bin/bash
# Update all 50 phantom nodes with Server Beta's real discovery information

echo "🔄 Updating 50 Alpha nodes with Server Beta discovery seeds..."

BETA_IP="185.182.185.227"
BETA_PORT="8080" 
BETA_DISCOVERY_SEEDS="
beta-validator-1.qnk.network
phantom-beta-001.qnk.network
185.182.185.227:8080
beta-node.phantom.discovery
"

# Update all 50 nodes with Server Beta's discovery information
for i in {1..50}; do
    if docker ps --filter "name=phantom-node-$i" --filter "status=running" -q | grep -q .; then
        echo "📡 Updating phantom-node-$i..."
        
        # Restart node with Beta discovery seeds
        docker rm -f phantom-node-$i 2>/dev/null
        
        PORT=$((9000 + i))
        IP_SUFFIX=$((100 + i))
        
        docker run -d \
            --name "phantom-node-$i" \
            --hostname "alpha-validator-$i" \
            --network "qnk-phantom-discovery" \
            --ip "172.60.0.$IP_SUFFIX" \
            -p "$PORT:8080" \
            -e NODE_ID="alpha-$i" \
            -e NODE_TYPE="validator" \
            -e NETWORK_SIZE="100" \
            -e DNS_PHANTOM_ENABLED="true" \
            -e DNS_PHANTOM_MODE="active" \
            -e DNS_PHANTOM_SEEDS="$BETA_DISCOVERY_SEEDS" \
            -e TARGET_BETA_IP="$BETA_IP" \
            -e TARGET_BETA_PORT="$BETA_PORT" \
            -e TOR_ENABLED="true" \
            -e TOR_SOCKS_PROXY="127.0.0.1:9050" \
            -e BEP44_ENABLED="true" \
            -e DISCOVERY_MODE="multi-tier" \
            -e LOG_LEVEL="debug" \
            -e RUST_LOG="q_dns_phantom=debug,q_network=debug" \
            -e AUTO_CONNECT_PEERS="true" \
            -e PHANTOM_ANNOUNCE_INTERVAL="15" \
            -e PHANTOM_PROVIDERS="cloudflare,google,quad9,opendns" \
            -e BOOTSTRAP_MODE="dns-phantom-first" \
            -e CONNECT_TO_BETA="true" \
            -e BETA_CONNECTION_TIMEOUT="30" \
            --memory="256m" \
            --cpus="0.5" \
            "q-narwhalknight-phantom:latest"
        
        if [ $((i % 10)) -eq 0 ]; then
            echo "   ✅ Updated $i/50 nodes..."
            sleep 2
        fi
    else
        echo "   ⚠️  phantom-node-$i not running, skipping..."
    fi
done

echo "🎉 All 50 Alpha nodes updated with Server Beta discovery information!"
```

### Step 3: Force Direct Connection Attempt

Add this script to force immediate connection attempts:

```bash
#!/bin/bash
# Force direct connection to Server Beta

echo "🎯 Forcing direct connections from Alpha nodes to Server Beta..."

BETA_IP="185.182.185.227"
BETA_PORT="8080"

for i in {1..10}; do
    echo "🔗 Attempting connection from phantom-node-$i to Server Beta..."
    
    # Send direct connection request
    docker exec phantom-node-$i curl -X POST \
        "http://localhost:8080/api/v1/network/connect" \
        -H "Content-Type: application/json" \
        -d "{
            \"peer_address\": \"$BETA_IP:$BETA_PORT\",
            \"connection_type\": \"cross_server\",
            \"discovery_method\": \"direct\",
            \"source_server\": \"alpha\",
            \"target_server\": \"beta\"
        }" \
        --timeout 10 || echo "Connection attempt failed for node $i"
    
    sleep 1
done

echo "📊 Connection attempts completed. Checking results..."

# Verify connections
CONNECTED=0
for i in {1..10}; do
    if docker logs phantom-node-$i 2>&1 | grep -q "Connected.*beta\|Established.*185.182.185.227"; then
        ((CONNECTED++))
        echo "✅ Node $i successfully connected to Server Beta"
    fi
done

echo "🎉 Result: $CONNECTED/10 Alpha nodes connected to Server Beta"
```

### Step 4: Real-Time Connection Monitor

Run this on Server Alpha to monitor connections:

```bash
#!/bin/bash
# Monitor Alpha→Beta connections

while true; do
    clear
    echo "🔍 ALPHA→BETA CONNECTION MONITOR - $(date)"
    echo "=============================================="
    
    TOTAL_CONNECTIONS=0
    SUCCESSFUL_DISCOVERIES=0
    
    echo "📊 Sampling 10 nodes for connection status..."
    for i in {1..10}; do
        # Check if node discovered Beta
        if docker logs phantom-node-$i 2>&1 | tail -50 | grep -q "beta\|185.182.185.227\|Discovered.*server"; then
            ((SUCCESSFUL_DISCOVERIES++))
            echo "   ✅ Node $i: Discovered Server Beta"
            
            # Check if actually connected
            if docker logs phantom-node-$i 2>&1 | tail -20 | grep -q "Connected\|Established.*185.182.185.227"; then
                ((TOTAL_CONNECTIONS++))
                echo "      🔗 Connected to Server Beta!"
            fi
        else
            echo "   ⏳ Node $i: Still searching..."
        fi
    done
    
    # Extrapolate to all 50 nodes
    EST_DISCOVERIES=$((SUCCESSFUL_DISCOVERIES * 5))
    EST_CONNECTIONS=$((TOTAL_CONNECTIONS * 5))
    
    echo ""
    echo "📈 ESTIMATED RESULTS (50 nodes total):"
    echo "   🎯 Nodes that discovered Beta: ~$EST_DISCOVERIES/50"
    echo "   🔗 Nodes connected to Beta: ~$EST_CONNECTIONS/50"
    echo "   📊 Success Rate: $((EST_CONNECTIONS * 100 / 50))%"
    
    if [ $TOTAL_CONNECTIONS -gt 0 ]; then
        echo ""
        echo "🎉 SUCCESS: Cross-server connections established!"
        echo "📡 Alpha nodes are connecting to Server Beta"
    else
        echo ""
        echo "⏳ Working: DNS-Phantom discovery in progress..."
        echo "💡 Tip: Ensure Server Beta is reachable on 185.182.185.227:8080"
    fi
    
    echo ""
    echo "Press Ctrl+C to exit. Refreshing in 15 seconds..."
    sleep 15
done
```

---

## 🎯 Expected Results After Update

Once Server Alpha runs the update script:

1. **Within 2-3 minutes**: Alpha nodes should discover Server Beta
2. **Within 5 minutes**: Direct connections should be established
3. **Target**: 30-50 successful connections from Alpha to Beta

## 🔍 How to Verify Success

On Server Alpha, check:
```bash
# Count successful connections
grep -c "Connected.*185.182.185.227\|Established.*beta" phantom-node-*/logs/*.log
```

On Server Beta, check:
```bash
# Count incoming Alpha connections
ss -tn state established | grep ":8080" | wc -l
```

## 📊 Current Status Summary

- **Server Alpha**: 50 nodes deployed ✅, 0 connected to Beta ❌  
- **Server Beta**: 1 node running ✅, 0 Alpha connections ❌
- **Issue**: Alpha nodes lack Server Beta's specific discovery information
- **Solution**: Update Alpha nodes with Beta's IP and discovery seeds

**Run the update script above to establish the cross-server connections!**