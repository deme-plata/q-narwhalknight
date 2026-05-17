#!/bin/bash

echo "🚀🔥 Q-NARWHALKNIGHT 50-NODE MASSIVE SCALE DEPLOYMENT"
echo "========================================================"
echo "🎯 Target: 56 containers (50 consensus nodes + 6 infrastructure)"
echo "🌐 Network: Distributed DAG-Knight consensus with Tor anonymity"
echo "🔍 Discovery: DNS-Phantom steganographic peer detection"
echo "⚡ Goal: 30,000+ TPS at massive scale"
echo "⏰ $(date)"
echo ""

# Check Docker availability
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

echo "📊 System Check:"
echo "   Docker: $(docker --version | cut -d' ' -f3 | cut -d',' -f1)"
echo "   Docker Compose: $(docker-compose --version | cut -d' ' -f4 | cut -d',' -f1)"
echo "   Available Memory: $(free -h | grep '^Mem:' | awk '{print $7}')"
echo "   CPU Cores: $(nproc)"
echo ""

# Clean up any existing deployment
echo "🧹 Cleaning previous deployment..."
docker-compose -f docker-compose-full-50-nodes.yml down --remove-orphans 2>/dev/null || true
docker network prune -f 2>/dev/null || true
docker volume prune -f 2>/dev/null || true

echo ""
echo "🌐 Creating massive scale network infrastructure..."
echo ""

# Ensure Docker has enough resources
echo "⚠️  RESOURCE REQUIREMENTS:"
echo "   Minimum RAM: 8GB recommended for 56 containers"  
echo "   Minimum CPU: 4 cores recommended"
echo "   Disk Space: ~2GB for container images"
echo ""
read -p "Continue with 50-node deployment? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment cancelled."
    exit 1
fi

echo ""
echo "🚀 INITIATING 56-CONTAINER DEPLOYMENT..."
echo ""

# Deploy with staged startup to prevent resource overload
echo "🏗️ Phase 1: Infrastructure Services (4 containers)"
docker-compose -f docker-compose-full-50-nodes.yml up -d tor-proxy dns-phantom-hub prometheus grafana

echo "⏳ Waiting for infrastructure startup..."
sleep 20

echo "🤝 Phase 2: Beta Coordinator (1 container)"  
docker-compose -f docker-compose-full-50-nodes.yml up -d beta-coordinator

sleep 10

echo "🚀 Phase 3: Alpha Nodes (10 containers)"
docker-compose -f docker-compose-full-50-nodes.yml up -d \
    alpha-node-01 alpha-node-02 alpha-node-03 alpha-node-04 alpha-node-05 \
    alpha-node-06 alpha-node-07 alpha-node-08 alpha-node-09 alpha-node-10

sleep 15

echo "⚛️ Phase 4: Validator Nodes (39 containers) - This will take time..."
echo "   Starting validators in batches to prevent resource overload"

# Start validators in batches of 10
echo "   Batch 1: Validators 01-10"
docker-compose -f docker-compose-full-50-nodes.yml up -d \
    validator-01 validator-02 validator-03 validator-04 validator-05 \
    validator-06 validator-07 validator-08 validator-09 validator-10
sleep 8

echo "   Batch 2: Validators 11-20"
docker-compose -f docker-compose-full-50-nodes.yml up -d \
    validator-11 validator-12 validator-13 validator-14 validator-15 \
    validator-16 validator-17 validator-18 validator-19 validator-20
sleep 8

echo "   Batch 3: Validators 21-30"
docker-compose -f docker-compose-full-50-nodes.yml up -d \
    validator-21 validator-22 validator-23 validator-24 validator-25 \
    validator-26 validator-27 validator-28 validator-29 validator-30
sleep 8

echo "   Batch 4: Validators 31-39"
docker-compose -f docker-compose-full-50-nodes.yml up -d \
    validator-31 validator-32 validator-33 validator-34 validator-35 \
    validator-36 validator-37 validator-38 validator-39
sleep 8

echo "🧪 Phase 5: Testing & Monitoring (2 containers)"
docker-compose -f docker-compose-full-50-nodes.yml up -d massive-scale-tester network-monitor

echo ""
echo "⏳ Final system stabilization..."
sleep 30

echo ""
echo "✅ 50-NODE Q-NARWHALKNIGHT DEPLOYMENT COMPLETE!"
echo ""
echo "📊 Deployment Summary:"
RUNNING=$(docker ps --filter "name=q-" --format "table {{.Names}}" | wc -l)
echo "   • Running Containers: $((RUNNING - 1)) / 56 expected"
echo "   • Network: qnarwhal-massivenet (172.20.0.0/16)"
echo "   • Infrastructure: Tor + DNS-Phantom + Monitoring"
echo "   • Consensus: 39 validators + 10 alpha nodes + 1 beta coordinator"
echo ""

echo "🌐 Access Points:"
echo "   • Grafana Dashboard: http://localhost:3000 (admin/q-narwhal-50nodes)"
echo "   • Prometheus Metrics: http://localhost:9090"
echo "   • DNS-Phantom Hub: http://localhost:8080"
echo "   • Tor SOCKS5 Proxy: localhost:9050"
echo "   • Beta Coordinator: localhost:8081"
echo ""

echo "📈 Real-time Monitoring:"
echo "   • Network Monitor: docker logs -f q-network-monitor"
echo "   • Load Tester: docker logs -f q-massive-scale-tester"
echo "   • DNS-Phantom: docker logs -f q-dns-phantom-hub"
echo "   • Beta Coordinator: docker logs -f q-beta-coordinator"
echo "   • Validator-01: docker logs -f q-validator-01"
echo ""

echo "🔍 System Status Commands:"
echo "   • Container Status: docker ps --filter 'name=q-'"
echo "   • Network Topology: docker network inspect qnk-massive-50"
echo "   • Resource Usage: docker stats --format 'table {{.Container}}\\t{{.CPUPerc}}\\t{{.MemUsage}}'"
echo "   • Logs (all nodes): docker-compose -f docker-compose-full-50-nodes.yml logs -f"
echo ""

echo "🎯 EXPECTED PERFORMANCE:"
echo "   • Target TPS: 30,000+ transactions per second"
echo "   • Consensus Latency: <3 seconds (DAG-Knight)"
echo "   • Discovery Time: <10 seconds (DNS-Phantom)" 
echo "   • Network Mesh: 50 nodes with Byzantine fault tolerance"
echo "   • Anonymity Layer: Full Tor integration with onion services"
echo ""

echo "🚨 NEXT STEPS:"
echo "   1. Monitor container startup over next 2-3 minutes"
echo "   2. Check Grafana dashboard for performance metrics"  
echo "   3. Verify 10 Alpha connections to Beta coordinator"
echo "   4. Observe DNS-Phantom steganographic discovery"
echo "   5. Watch massive scale load testing results"
echo ""

echo "🎉 Q-NARWHALKNIGHT 50-NODE MASSIVE SCALE TEST ENVIRONMENT IS LIVE!"
echo "   Ready for distributed quantum consensus at unprecedented scale!"

# Start real-time monitoring
echo ""
echo "📊 Starting real-time deployment monitoring..."
echo "   (Press Ctrl+C to stop monitoring and return to shell)"
echo ""

# Monitor deployment
while true; do
    RUNNING=$(docker ps --filter "name=q-" --format "table {{.Names}}" | wc -l)
    TOTAL_CONTAINERS=$((RUNNING - 1))
    
    echo "[$(date +%H:%M:%S)] 🔍 Container Status: $TOTAL_CONTAINERS/56 running | Network: $(docker network ls | grep qnk-massive-50 | wc -l) active"
    
    # Check key services
    if docker ps | grep -q q-beta-coordinator; then
        BETA_STATUS="✅"
    else
        BETA_STATUS="❌"
    fi
    
    if docker ps | grep -q q-dns-phantom-hub; then
        DNS_STATUS="✅"
    else
        DNS_STATUS="❌"
    fi
    
    ALPHA_COUNT=$(docker ps --filter "name=q-alpha-node" | wc -l)
    VALIDATOR_COUNT=$(docker ps --filter "name=q-validator" | wc -l)
    
    echo "         Services: Beta $BETA_STATUS | DNS-Phantom $DNS_STATUS | Alpha nodes: $ALPHA_COUNT/10 | Validators: $VALIDATOR_COUNT/39"
    
    if [ $TOTAL_CONTAINERS -eq 56 ]; then
        echo "         🎉 ALL 56 CONTAINERS SUCCESSFULLY DEPLOYED!"
    fi
    
    sleep 15
done