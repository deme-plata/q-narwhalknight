#!/bin/bash
# Example script for running DAGKnight VM

set -e

echo "Starting DAGKnight node with example configuration..."

# Start node 0
cargo run --release -- --node-id 0 --listen "/ip4/127.0.0.1/tcp/8080" &
PID_1=

echo "Node 0 started with PID "
sleep 1

# Start node 1
cargo run --release -- --node-id 1 --listen "/ip4/127.0.0.1/tcp/8081" --peers "/ip4/127.0.0.1/tcp/8080" &
PID_2=

echo "Node 1 started with PID "
echo "Press Ctrl+C to stop nodes"

# Wait for Ctrl+C
trap "kill  ; echo 'Nodes stopped.'; exit 0" INT
wait
