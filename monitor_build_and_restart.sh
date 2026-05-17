#!/bin/bash
# Automated build monitoring and service restart script
# Monitors the cargo build process and restarts service when complete

set -e

BUILD_PID_FILE="/tmp/q-api-server-build.pid"
BINARY_PATH="/opt/orobit/shared/q-narwhalknight/target/release/q-api-server"
LOG_FILE="/tmp/q-api-server-build-monitor.log"

echo "🔍 Monitoring cargo build process..." | tee -a "$LOG_FILE"
echo "   Checking for completion every 30 seconds..." | tee -a "$LOG_FILE"
echo "   Binary path: $BINARY_PATH" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Wait for the binary to be created and finalized
while true; do
    # Check if cargo build processes are still running
    BUILD_RUNNING=$(ps aux | grep -E "cargo build.*q-api-server" | grep -v grep | wc -l)

    if [ "$BUILD_RUNNING" -eq 0 ]; then
        # No build processes running - check if binary exists and is recent
        if [ -f "$BINARY_PATH" ]; then
            # Check if binary was modified in the last 5 minutes
            BINARY_AGE=$(find "$BINARY_PATH" -mmin -5 2>/dev/null | wc -l)

            if [ "$BINARY_AGE" -gt 0 ]; then
                echo "✅ Build complete! Binary found at $BINARY_PATH" | tee -a "$LOG_FILE"
                ls -lh "$BINARY_PATH" | tee -a "$LOG_FILE"
                echo "" | tee -a "$LOG_FILE"

                # Restart the service
                echo "🔄 Restarting q-api-server service..." | tee -a "$LOG_FILE"
                /opt/orobit/shared/q-narwhalknight/restart_api_service.sh 2>&1 | tee -a "$LOG_FILE"

                exit 0
            fi
        fi
    else
        TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
        echo "[$TIMESTAMP] 🔄 Build still running ($BUILD_RUNNING processes active)..." | tee -a "$LOG_FILE"
    fi

    sleep 30
done
