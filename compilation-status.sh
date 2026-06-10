#!/bin/bash

echo "🔧 Q-NarwhalKnight Compilation Status"
echo "===================================="
echo "Time: $(date)"
echo

# Check if cargo build is running
BUILD_PID=$(ps aux | grep "cargo build --release" | grep -v grep | awk '{print $2}' | head -1)

if [ -n "$BUILD_PID" ]; then
    echo "✅ Compilation Status: ACTIVE (PID: $BUILD_PID)"
    
    # Get process info
    PROC_INFO=$(ps -p $BUILD_PID -o pid,pcpu,pmem,etime,cmd --no-headers 2>/dev/null)
    echo "📊 Process: $PROC_INFO"
    
    # Check compilation artifacts
    if [ -d "target/release" ]; then
        RECENT_FILES=$(find target/release -type f -newermt "5 minutes ago" 2>/dev/null | wc -l)
        echo "📁 Recent build artifacts: $RECENT_FILES files"
        
        # Check for executable binaries
        BINARIES=$(find target/release -maxdepth 1 -type f -executable -name "q-*" 2>/dev/null)
        if [ -n "$BINARIES" ]; then
            echo "🎯 Completed binaries:"
            echo "$BINARIES" | sed 's/^/   ✅ /'
        fi
    fi
    
    # Check memory usage
    MEM_USAGE=$(free -h | grep "Mem:" | awk '{print "Used: "$3" / "$2}')
    echo "💾 Memory: $MEM_USAGE"
    
    echo "🔄 Compilation in progress..."
    
else
    echo "❌ Compilation Status: NOT RUNNING"
    
    # Check if binaries were created
    if [ -d "target/release" ]; then
        BINARIES=$(find target/release -maxdepth 1 -type f -executable -name "q-*" 2>/dev/null)
        if [ -n "$BINARIES" ]; then
            echo "🎯 Available binaries:"
            echo "$BINARIES" | sed 's/^/   ✅ /'
            
            # Check binary timestamps
            echo "📅 Binary creation times:"
            ls -la target/release/q-* 2>/dev/null | sed 's/^/   /'
        else
            echo "❌ No Q-NarwhalKnight binaries found"
        fi
    else
        echo "❌ No release directory found"
    fi
fi

echo
echo "🏁 Status check complete"