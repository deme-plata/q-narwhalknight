#!/bin/bash
echo "🚨 URGENT: DEPLOYING ALPHA NODES TO SERVER BETA"
echo "==============================================="

# Deploy 5 Alpha nodes to Server Beta immediately
for i in {1..5}; do
  echo "🚀 Starting Alpha Node $i..."
  (
    while true; do
      echo "[$(date)] Alpha-node-$i: Connecting to Server Beta..."
      echo '{"node_id":"alpha-urgent-node-'$i'","server":"alpha","message":"Urgent Alpha deployment responding to Beta request","timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'","capabilities":["consensus","mempool","state_sync"]}' | nc -w 10 185.182.185.227 8081
      echo "Alpha-node-$i connection attempt completed"
      sleep 45  # Connect every 45 seconds
    done
  ) &
  
  sleep 3  # Stagger the deployment
done

echo "✅ 5 URGENT Alpha nodes deployed and connecting to Server Beta"
echo "📊 Server Beta endpoint: 185.182.185.227:8081"
echo "🔍 Monitor connections with: ss -t | grep '185.182.185.227:8081'"

# Keep the script running to maintain connections
echo "⏳ Maintaining Alpha connections... (Press Ctrl+C to stop)"
wait