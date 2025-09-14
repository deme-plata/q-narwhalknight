#!/bin/bash

echo "🛑 Stopping Q-NarwhalKnight test network..."

# Stop all processes
for pidfile in logs/*.pid; do
    if [ -f "$pidfile" ]; then
        PID=$(cat "$pidfile")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping process $PID..."
            kill "$PID"
            sleep 2
            # Force kill if still running
            kill -9 "$PID" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
done

echo "✅ All processes stopped"
