#!/bin/bash

# Retry connecting nodes with multiple attempts
# Usage: ./retry_connect.sh <api_port> <target_multiaddr>

API_PORT=$1
TARGET_MULTIADDR=$2

echo "Attempting to connect from port $API_PORT to $TARGET_MULTIADDR"

for i in {1..20}; do
  result=$(curl -s -X POST "http://localhost:$API_PORT/api/v1/network/peers/connect" \
    -H "Content-Type: application/json" \
    -d "{\"multiaddr\": \"$TARGET_MULTIADDR\"}")

  success=$(echo "$result" | jq -r '.success')

  if [ "$success" = "true" ]; then
    echo "✅ Connection successful!"
    echo "$result" | jq '.'
    exit 0
  fi

  echo "Attempt $i/20 failed, retrying..."
  sleep 0.3
done

echo "❌ Failed after 20 attempts:"
echo "$result" | jq '.'
exit 1
