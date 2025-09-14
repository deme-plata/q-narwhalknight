#!/bin/bash

echo "🛑 Shutting down Q-NarwhalKnight network..."

# Stop all nodes
for i in {1..20}; do
    if [ -f "logs/node-$i.pid" ]; then
        pid=$(cat logs/node-$i.pid)
        if kill -0 $pid > /dev/null 2>&1; then
            kill $pid
            echo "✅ Stopped Node-$i (PID: $pid)"
        else
            echo "⚠️  Node-$i was not running"
        fi
        rm -f logs/node-$i.pid
    fi
done

echo "🌟 All nodes stopped!"