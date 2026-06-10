#!/bin/bash
# Q-NarwhalKnight API Server Watchdog
# Automatically restarts the service if block production stops (deadlock detected)

LOGFILE="/var/log/q-api-server-watchdog.log"
SERVICE="q-api-server.service"
CHECK_INTERVAL=60  # Check every 60 seconds

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

log "========================================="
log "Q-NarwhalKnight Watchdog Started"
log "========================================="

LAST_BLOCK_TIME=$(date +%s)
STALL_THRESHOLD=180  # If no blocks for 3 minutes, restart

while true; do
    sleep $CHECK_INTERVAL

    # Check if service is running
    if ! systemctl is-active --quiet $SERVICE; then
        log "⚠️  Service is not running, attempting start..."
        systemctl start $SERVICE
        LAST_BLOCK_TIME=$(date +%s)
        continue
    fi

    # Check for recent block production
    RECENT_BLOCKS=$(journalctl -u $SERVICE --since "2 minutes ago" | grep -c "TIME-BASED PARALLEL BLOCK PRODUCED")

    if [ "$RECENT_BLOCKS" -eq 0 ]; then
        CURRENT_TIME=$(date +%s)
        TIME_SINCE_LAST=$(( CURRENT_TIME - LAST_BLOCK_TIME ))

        log "⚠️  No blocks produced in last 2 minutes (stalled for ${TIME_SINCE_LAST}s)"

        if [ "$TIME_SINCE_LAST" -gt "$STALL_THRESHOLD" ]; then
            log "🚨 DEADLOCK DETECTED: No blocks for ${TIME_SINCE_LAST}s, forcing restart"

            # Get PID and check CPU
            PID=$(systemctl show -p MainPID $SERVICE | cut -d= -f2)
            if [ "$PID" != "0" ]; then
                CPU=$(ps -p $PID -o %cpu= | awk '{print int($1)}')
                log "   Process $PID CPU usage: ${CPU}%"

                # Force kill if stuck
                log "   Force killing stuck process..."
                kill -9 $PID 2>/dev/null
                sleep 2
            fi

            # Start service
            log "   Starting service..."
            systemctl start $SERVICE
            sleep 5

            if systemctl is-active --quiet $SERVICE; then
                log "✅ Service restarted successfully"
                LAST_BLOCK_TIME=$(date +%s)
            else
                log "❌ Service restart failed!"
            fi
        fi
    else
        # Blocks are being produced
        LAST_BLOCK_TIME=$(date +%s)
        log "✅ Healthy: $RECENT_BLOCKS blocks produced in last 2 minutes"
    fi
done
