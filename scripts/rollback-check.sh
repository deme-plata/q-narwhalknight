#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# rollback-check.sh — systemd ExecStartPre auto-rollback for q-api-server
#
# v8.5.0: Part of the Node Auto-Update System
#
# Called by systemd BEFORE starting q-api-server. Detects if the last
# auto-update caused a crash and restores the previous binary.
#
# Logic:
#   IF restart_marker exists
#   AND update_healthy marker does NOT exist
#   AND backup binary exists
#   THEN → the last auto-update crashed → restore backup
#
# The node writes restart_marker BEFORE swapping the binary.
# After successful start, a 60s watchdog writes update_healthy.
# If the process crashes before 60s, update_healthy is never written.
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# Determine DB path from systemd environment or default
DB_PATH="${Q_DB_PATH:-./data-mainnet2026.1.1}"
RESTART_MARKER="${DB_PATH}/restart_marker"
HEALTHY_MARKER="${DB_PATH}/update_healthy"
LOG_TAG="[AUTO-UPDATE-ROLLBACK]"

log_info()  { echo "$(date '+%Y-%m-%d %H:%M:%S') ${LOG_TAG} INFO:  $*"; }
log_warn()  { echo "$(date '+%Y-%m-%d %H:%M:%S') ${LOG_TAG} WARN:  $*" >&2; }
log_error() { echo "$(date '+%Y-%m-%d %H:%M:%S') ${LOG_TAG} ERROR: $*" >&2; }

# If no restart marker exists, this is a normal start — nothing to do
if [ ! -f "${RESTART_MARKER}" ]; then
    log_info "No restart marker found — normal startup"
    exit 0
fi

# If healthy marker exists, the last update was successful — clean up
if [ -f "${HEALTHY_MARKER}" ]; then
    log_info "Update healthy marker found — last auto-update succeeded"
    rm -f "${RESTART_MARKER}" "${HEALTHY_MARKER}"
    exit 0
fi

# restart_marker exists BUT update_healthy does NOT
# → The last auto-update caused a crash!
log_warn "CRASH DETECTED: restart_marker exists but update_healthy does not"
log_warn "The last auto-update likely crashed the node. Initiating rollback..."

# Parse the restart marker for backup path
BACKUP_PATH=$(python3 -c "
import json, sys
try:
    marker = json.load(open('${RESTART_MARKER}'))
    print(marker.get('binary_backup_path', ''))
except:
    print('')
" 2>/dev/null || echo "")

if [ -z "${BACKUP_PATH}" ]; then
    # Fallback: try jq
    BACKUP_PATH=$(jq -r '.binary_backup_path // empty' "${RESTART_MARKER}" 2>/dev/null || echo "")
fi

if [ -z "${BACKUP_PATH}" ] || [ ! -f "${BACKUP_PATH}" ]; then
    log_error "Cannot find backup binary at '${BACKUP_PATH}'"
    log_error "Manual intervention required!"
    # Remove marker to prevent infinite restart loop
    rm -f "${RESTART_MARKER}"
    exit 0  # Exit 0 so systemd still starts (even with potentially bad binary)
fi

# Get current binary path
CURRENT_BINARY=$(readlink -f /proc/$$/exe 2>/dev/null || which q-api-server 2>/dev/null || echo "")
if [ -z "${CURRENT_BINARY}" ]; then
    # Try to get it from systemd service file
    CURRENT_BINARY=$(grep -oP 'ExecStart=\K[^ ]+' /etc/systemd/system/q-api-server.service 2>/dev/null || echo "")
fi

PREVIOUS_VERSION=$(python3 -c "
import json
marker = json.load(open('${RESTART_MARKER}'))
print(marker.get('previous_version', 'unknown'))
" 2>/dev/null || echo "unknown")

NEW_VERSION=$(python3 -c "
import json
marker = json.load(open('${RESTART_MARKER}'))
print(marker.get('new_version', 'unknown'))
" 2>/dev/null || echo "unknown")

log_warn "Rolling back: v${NEW_VERSION} → v${PREVIOUS_VERSION}"
log_warn "Backup binary: ${BACKUP_PATH}"

if [ -n "${CURRENT_BINARY}" ] && [ -f "${CURRENT_BINARY}" ]; then
    # Restore the backup binary over the current one
    cp "${BACKUP_PATH}" "${CURRENT_BINARY}"
    chmod +x "${CURRENT_BINARY}"
    log_info "Restored backup binary to ${CURRENT_BINARY}"
else
    log_warn "Could not determine current binary path — backup is at ${BACKUP_PATH}"
fi

# Clean up markers
rm -f "${RESTART_MARKER}" "${HEALTHY_MARKER}"

log_info "Rollback complete. Node will start with v${PREVIOUS_VERSION}"

# Log to systemd journal for visibility
logger -t "q-api-server-rollback" "AUTO-UPDATE ROLLBACK: v${NEW_VERSION} crashed, restored v${PREVIOUS_VERSION}" 2>/dev/null || true

exit 0
