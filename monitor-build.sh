#!/bin/bash

# Monitor Q-NarwhalKnight compilation progress
echo "🔧 Q-NarwhalKnight Build Monitor"
echo "==============================="
echo "Started: $(date)"
echo

BUILD_PID=$(ps aux | grep "cargo build --release" | grep -v grep | awk '{print $2}' | head -1)

if [ -n "$BUILD_PID" ]; then
    echo "📊 Found cargo build process (PID: $BUILD_PID)"
    echo "⏱️ Monitoring compilation progress..."
    echo
    
    # Monitor process and show periodic status
    while kill -0 "$BUILD_PID" 2>/dev/null; do
        echo "$(date): ⚙️ Compilation in progress (PID: $BUILD_PID)"
        
        # Show memory usage
        MEM_USAGE=$(ps -p $BUILD_PID -o pid,pcpu,pmem,cmd --no-headers 2>/dev/null)
        if [ -n "$MEM_USAGE" ]; then
            echo "$(date): 📊 $MEM_USAGE"
        fi
        
        # Check for recent compilation activity in target directory
        TARGET_FILES=$(find /mnt/orobit-shared/q-narwhalknight/target -type f -newer /tmp/build-start 2>/dev/null | wc -l)
        echo "$(date): 📁 Active compilation files: $TARGET_FILES"
        
        echo "$(date): 🔄 Checking for errors..."
        
        sleep 30
        echo "---"
    done
    
    echo "$(date): ✅ Build process completed or stopped"
    
    # Check if binaries were created
    if [ -d "/mnt/orobit-shared/q-narwhalknight/target/release" ]; then
        echo "$(date): 📦 Release binaries:"
        ls -la /mnt/orobit-shared/q-narwhalknight/target/release/q-* 2>/dev/null || echo "No Q-NarwhalKnight binaries found"
    fi
else
    echo "❌ No cargo build process found"
fi

echo
echo "🏁 Build monitoring completed: $(date)"