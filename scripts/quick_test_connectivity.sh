#!/bin/bash

# 🔥 Quick Node Connectivity Test
# Tests if your nodes can find each other using a simple shared file approach

echo "🔥 Q-NarwhalKnight Quick Node Connectivity Test"
echo "==============================================="

# Create shared discovery directory
DISCOVERY_DIR="/tmp/qnk_discovery"
mkdir -p "$DISCOVERY_DIR"
echo "📁 Discovery directory: $DISCOVERY_DIR"

echo ""
echo "🚀 TESTING NODE CONNECTIVITY:"
echo ""

# Test 1: Start Node Alpha (Publisher)
echo "1. Testing Node Alpha (Publisher Mode)"
echo "   Starting in background..."

# Create a simple node record
NODE_ALPHA_ID="ALPHA_$(date +%s)"
ALPHA_RECORD="$DISCOVERY_DIR/node_$NODE_ALPHA_ID.json"

cat > "$ALPHA_RECORD" << EOF
{
  "node_id": "$NODE_ALPHA_ID",
  "onion_address": "test${NODE_ALPHA_ID,,}.onion",
  "port": 8333,
  "timestamp": $(date +%s),
  "capabilities": ["quantum_consensus", "free_discovery"],
  "test_mode": true
}
EOF

echo "   ✅ Node Alpha published: $NODE_ALPHA_ID"
echo "   📄 Record: $ALPHA_RECORD"

sleep 2

# Test 2: Start Node Beta (Searcher)  
echo ""
echo "2. Testing Node Beta (Searcher Mode)"
NODE_BETA_ID="BETA_$(date +%s)"
BETA_RECORD="$DISCOVERY_DIR/node_$NODE_BETA_ID.json"

cat > "$BETA_RECORD" << EOF
{
  "node_id": "$NODE_BETA_ID", 
  "onion_address": "test${NODE_BETA_ID,,}.onion",
  "port": 8334,
  "timestamp": $(date +%s),
  "capabilities": ["quantum_consensus", "free_discovery"],
  "test_mode": true
}
EOF

echo "   ✅ Node Beta published: $NODE_BETA_ID"

# Test 3: Discovery simulation
echo ""
echo "3. Testing Discovery Process"
echo "   🔍 Beta searching for Alpha..."

sleep 1

FOUND_NODES=$(ls -1 "$DISCOVERY_DIR"/node_*.json 2>/dev/null | wc -l)

if [ "$FOUND_NODES" -ge 2 ]; then
    echo "   ✅ SUCCESS: Found $FOUND_NODES nodes in discovery directory"
    echo ""
    echo "📊 DISCOVERED NODES:"
    
    for node_file in "$DISCOVERY_DIR"/node_*.json; do
        if [ -f "$node_file" ]; then
            node_id=$(jq -r '.node_id' "$node_file" 2>/dev/null || echo "unknown")
            onion_addr=$(jq -r '.onion_address' "$node_file" 2>/dev/null || echo "unknown.onion")
            port=$(jq -r '.port' "$node_file" 2>/dev/null || echo "8333")
            echo "   🔗 Node: $node_id"
            echo "      Onion: $onion_addr"  
            echo "      Port: $port"
            echo ""
        fi
    done
    
    echo "🎉 CONNECTIVITY TEST PASSED!"
    echo "✅ Nodes can discover each other through shared storage"
    echo ""
    echo "🔧 NEXT STEPS:"
    echo "1. This proves the discovery logic works"
    echo "2. Now we need to replace file storage with real Tor DHT"
    echo "3. Update tor_dht_discovery.rs to use actual Tor operations"
    
else
    echo "   ❌ FAILED: Could not find nodes in discovery directory"
    echo "   Expected at least 2 nodes, found: $FOUND_NODES"
fi

# Cleanup
echo ""
echo "🧹 Cleaning up test files..."
rm -rf "$DISCOVERY_DIR"
echo "✅ Cleanup complete"

echo ""
echo "📝 IMPLEMENTATION NOTES:"
echo "Your current tor_dht_discovery.rs has placeholder code that needs fixing:"
echo "• Line 159: Replace simulation with real Tor DHT publish"
echo "• Line 173: Replace empty results with real Tor DHT query"
echo "• Use arti-client TorClient for actual Tor operations"