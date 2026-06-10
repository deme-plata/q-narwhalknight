#!/bin/bash
echo "🔍 Server Beta 10-Node Network Performance Monitor"
echo "$(date): Starting real-world performance measurement"

while true; do
    echo "==============================================="
    echo "📊 REAL-TIME PERFORMANCE METRICS - $(date)"
    echo "==============================================="
    
    for i in {1..5}; do
        PORT=$((9000 + i))
        echo "🏷️  Node B$i (Port $((8000 + i))) Metrics:"
        
        # Try to get metrics from API endpoint (simulated)
        if curl -s --connect-timeout 2 "http://localhost:$PORT/status" 2>/dev/null; then
            echo "   Status: ✅ OPERATIONAL"
        else
            echo "   Status: 🔄 STILL COMPILING/STARTING"
        fi
        
        # Check if process is running
        PID=$(ps aux | grep "validator-beta-$i" | grep -v grep | awk '{print $2}' | head -1)
        if [ ! -z "$PID" ]; then
            # Get CPU and memory usage
            CPU_MEM=$(ps -p $PID -o %cpu,%mem --no-headers 2>/dev/null)
            echo "   Process: Active (PID: $PID) - $CPU_MEM"
        else
            echo "   Process: 🚫 NOT FOUND"
        fi
    done
    
    echo "🌐 Network Connectivity Test:"
    LISTENING_PORTS=$(netstat -tlnp | grep ':800[1-5]' | wc -l)
    echo "   Listening ports: $LISTENING_PORTS/5"
    
    echo "⏱️  Next check in 10 seconds..."
    sleep 10
done
