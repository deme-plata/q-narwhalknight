#!/bin/bash

echo "📊 Q-NarwhalKnight Test Network Monitor"
echo "======================================="

while true; do
    clear
    echo "📊 Q-NarwhalKnight Test Network Status - $(date)"
    echo "================================================="
    echo
    
    # Check if processes are running
    echo "🔧 Node Status:"
    for i in {1..3}; do
        if [ -f "logs/node$i.pid" ] && kill -0 $(cat "logs/node$i.pid") 2>/dev/null; then
            echo "  ✅ Validator Node $i: Running"
        else
            echo "  ❌ Validator Node $i: Stopped"
        fi
    done
    
    echo
    echo "🏊 Pool Status:"
    if [ -f "logs/pool.pid" ] && kill -0 $(cat "logs/pool.pid") 2>/dev/null; then
        echo "  ✅ Mining Pool: Running"
    else
        echo "  ❌ Mining Pool: Stopped"
    fi
    
    echo
    echo "⛏️ Miner Status:"
    if [ -f "logs/miner.pid" ] && kill -0 $(cat "logs/miner.pid") 2>/dev/null; then
        echo "  ✅ Test Miner: Running"
    else
        echo "  ❌ Test Miner: Stopped"
    fi
    
    echo
    echo "📈 Recent Activity:"
    echo "─────────────────"
    
    # Show last few lines from miner log
    if [ -f "logs/miner.log" ]; then
        echo "🔨 Miner:"
        tail -3 "logs/miner.log" | sed 's/^/  /'
    fi
    
    echo
    
    # Show last few lines from pool log  
    if [ -f "logs/pool.log" ]; then
        echo "🏊 Pool:"
        tail -2 "logs/pool.log" | sed 's/^/  /'
    fi
    
    echo
    echo "Press Ctrl+C to exit monitoring"
    sleep 5
done
