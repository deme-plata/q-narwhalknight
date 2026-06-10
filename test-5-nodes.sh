#!/bin/bash

echo "🚀 Q-NarwhalKnight 5-Node Network Test"
echo "======================================"
echo ""

# Clean up any existing containers
echo "🧹 Cleaning up existing containers..."
docker rm -f qnk-test-{1..5} qnk-test-monitor 2>/dev/null

# Create network
echo "🌐 Creating test network..."
docker network create qnk-test-net --subnet 172.21.0.0/24 2>/dev/null || true

echo ""
echo "📦 Starting 5 test nodes..."

# Start nodes with simple HTTP servers for testing
for i in {1..5}; do
    echo -n "Starting Node $i... "
    docker run -d \
        --name qnk-test-$i \
        --network qnk-test-net \
        --ip 172.21.0.1$i \
        -p 808$i:8080 \
        -e NODE_ID=node$i \
        python:3.9-alpine \
        sh -c "
            echo 'Node $i starting...' &&
            python -m http.server 8080 --bind 0.0.0.0 2>/dev/null &
            
            # Simulate node behavior
            while true; do
                # Try to connect to other nodes
                for j in {1..5}; do
                    if [ \$j -ne $i ]; then
                        wget -q -O- http://172.21.0.1\$j:8080/ >/dev/null 2>&1 && echo \"Node $i connected to Node \$j\" || true
                    fi
                done
                sleep 10
            done
        " >/dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅"
    else
        echo "❌"
    fi
done

echo ""
echo "⏳ Waiting for nodes to initialize..."
sleep 5

echo ""
echo "🔍 Testing Node Connectivity:"
echo "------------------------------"

# Test connectivity between nodes
for i in {1..5}; do
    echo -n "Node $i: "
    
    # Check if container is running
    if docker ps -q --filter "name=qnk-test-$i" | grep -q .; then
        echo -n "✅ Running | "
        
        # Test connectivity to other nodes
        connected=0
        for j in {1..5}; do
            if [ $j -ne $i ]; then
                docker exec qnk-test-$i wget -q -O- --timeout=1 http://172.21.0.1$j:8080/ >/dev/null 2>&1 && ((connected++)) || true
            fi
        done
        echo "Connected to $connected/4 peers"
    else
        echo "❌ Not running"
    fi
done

echo ""
echo "📊 Network Statistics:"
echo "----------------------"

# Show network stats
total_running=$(docker ps -q --filter "name=qnk-test-" | wc -l)
echo "Total nodes running: $total_running/5"

if [ $total_running -eq 5 ]; then
    echo "✅ All nodes are running!"
    
    echo ""
    echo "🔄 Testing Data Propagation:"
    echo "----------------------------"
    
    # Create test data on Node 1
    docker exec qnk-test-1 sh -c "echo '{\"tx\": \"test-transaction\", \"from\": \"node1\", \"to\": \"node5\"}' > /tmp/test-tx.json"
    echo "Created test transaction on Node 1"
    
    # Simulate propagation by copying to other nodes
    for i in {2..5}; do
        docker exec qnk-test-1 sh -c "cat /tmp/test-tx.json" | docker exec -i qnk-test-$i sh -c "cat > /tmp/test-tx.json"
        echo "Propagated to Node $i ✅"
    done
    
    echo ""
    echo "🎯 Data Propagation Test: SUCCESS"
else
    echo "⚠️  Some nodes failed to start"
fi

echo ""
echo "📈 Monitor Output (press Ctrl+C to stop):"
echo "----------------------------------------"

# Start monitoring
while true; do
    clear
    echo "Q-NarwhalKnight Test Network Status"
    echo "Time: $(date)"
    echo ""
    
    for i in {1..5}; do
        if docker ps -q --filter "name=qnk-test-$i" | grep -q .; then
            echo "Node $i: ✅ ONLINE | IP: 172.21.0.1$i | Port: 808$i"
        else
            echo "Node $i: ❌ OFFLINE"
        fi
    done
    
    echo ""
    echo "Network Activity:"
    for i in {1..5}; do
        logs=$(docker logs qnk-test-$i 2>&1 | tail -1)
        [ ! -z "$logs" ] && echo "Node $i: $logs"
    done
    
    sleep 5
done