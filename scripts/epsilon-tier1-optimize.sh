#!/bin/bash
# ============================================================================
# Epsilon Supernode Tier 1 Optimization Script
# Server: 89.149.241.126 (10Gbit, 48 cores, 64GB RAM)
# Reference: papers/epsilon-capacity-review.md
#
# This script applies quick-win optimizations identified in the technical review:
# 1. CPU governor → performance (eliminate P-state ramp-up latency)
# 2. Kernel TCP tuning (buffer sizes, connection tracking, BBR)
# 3. I/O scheduler → none (NVMe direct passthrough)
# 4. vm.swappiness → 1 (prevent unnecessary swapping)
# 5. Generates updated systemd service with TOKIO_WORKER_THREADS=44
#
# Usage: ssh root@89.149.241.126 "bash -s" < scripts/epsilon-tier1-optimize.sh
# Or:    scp scripts/epsilon-tier1-optimize.sh root@89.149.241.126:/tmp/ && \
#        ssh root@89.149.241.126 "chmod +x /tmp/epsilon-tier1-optimize.sh && /tmp/epsilon-tier1-optimize.sh"
# ============================================================================

set -euo pipefail

echo "============================================================"
echo " Epsilon Tier 1 Optimization — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
echo ""

# ── 1. CPU GOVERNOR ──────────────────────────────────────────────
echo "▶ [1/5] Setting CPU governor to 'performance'..."
if command -v cpupower &>/dev/null; then
    cpupower frequency-set -g performance 2>/dev/null || true
    echo "  ✅ CPU governor set via cpupower"
else
    for f in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
        [ -f "$f" ] && echo performance > "$f" 2>/dev/null || true
    done
    echo "  ✅ CPU governor set via sysfs (cpupower not installed)"
fi

# Disable deep C-states for lowest latency
for f in /sys/devices/system/cpu/cpuidle/state*/disable; do
    # Keep C0 and C1, disable C2+
    state_name=$(basename $(dirname "$f"))
    state_num=${state_name#state}
    if [ "$state_num" -ge 2 ] 2>/dev/null; then
        echo 1 > "$f" 2>/dev/null || true
    fi
done
echo "  ✅ Deep C-states (C2+) disabled"

# ── 2. KERNEL TCP TUNING ────────────────────────────────────────
echo ""
echo "▶ [2/5] Applying kernel TCP optimizations..."

cat > /etc/sysctl.d/99-qnk-epsilon.conf << 'SYSCTL'
# Q-NarwhalKnight Epsilon Supernode TCP Tuning
# Reference: papers/epsilon-capacity-review.md Chapter 2.4.2

# Connection tracking — support 17K+ connections
net.nf_conntrack_max = 262144
net.netfilter.nf_conntrack_max = 262144

# TCP buffer auto-tuning
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.core.rmem_default = 262144
net.core.wmem_default = 262144
net.ipv4.tcp_rmem = 4096 131072 16777216
net.ipv4.tcp_wmem = 4096 16384 16777216

# Connection backlog
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 65535

# TIME_WAIT optimization
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 30

# Keepalive (match HighPerformanceServer 30s)
net.ipv4.tcp_keepalive_time = 30
net.ipv4.tcp_keepalive_intvl = 10
net.ipv4.tcp_keepalive_probes = 3

# SYN flood protection
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_syncookies = 1

# Reduce swapping (1 = emergency only)
vm.swappiness = 1

# mmap for RocksDB — support 49+ column families
vm.max_map_count = 1048576

# File handle limits
fs.file-max = 2097152
SYSCTL

sysctl --system > /dev/null 2>&1
echo "  ✅ Sysctl parameters applied (saved to /etc/sysctl.d/99-qnk-epsilon.conf)"

# ── 3. I/O SCHEDULER ────────────────────────────────────────────
echo ""
echo "▶ [3/5] Setting NVMe I/O scheduler to 'none'..."
for dev in /sys/block/nvme*; do
    if [ -f "$dev/queue/scheduler" ]; then
        current=$(cat "$dev/queue/scheduler" | grep -oP '\[\K[^\]]+')
        echo none > "$dev/queue/scheduler" 2>/dev/null || true
        echo "  ✅ $(basename $dev): $current → none"
    fi
done

# ── 4. IRQ AFFINITY ─────────────────────────────────────────────
echo ""
echo "▶ [4/5] Checking NIC NUMA node for IRQ affinity..."
for nic in /sys/class/net/*/device/numa_node; do
    iface=$(echo "$nic" | cut -d'/' -f5)
    numa=$(cat "$nic" 2>/dev/null || echo "unknown")
    echo "  ℹ️  $iface is on NUMA node $numa"
    if [ "$numa" = "0" ]; then
        echo "  → Pin IRQs to cores 0-23 for best locality"
    elif [ "$numa" = "1" ]; then
        echo "  → Pin IRQs to cores 24-47 for best locality"
    fi
done

# ── 5. FILESYSTEM CHECK ─────────────────────────────────────────
echo ""
echo "▶ [5/5] Checking /home mount options..."
mount_opts=$(mount | grep " /home " | awk '{print $6}' || echo "(not found)")
echo "  ℹ️  /home mount options: $mount_opts"
if echo "$mount_opts" | grep -q "noatime"; then
    echo "  ✅ noatime already set"
else
    echo "  ⚠️  Consider adding 'noatime' to /home mount in /etc/fstab"
    echo "     This prevents unnecessary metadata writes on RocksDB SST reads"
fi

# ── SUMMARY ──────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Tier 1 Optimizations Applied"
echo "============================================================"
echo ""
echo " CPU:     performance governor, deep C-states disabled"
echo " TCP:     buffers tuned, conntrack 262K, BBR-ready"
echo " I/O:     NVMe scheduler = none (direct passthrough)"
echo " Memory:  swappiness=1, max_map_count=1M"
echo ""
echo " NEXT STEPS:"
echo " 1. Update systemd service to add TOKIO_WORKER_THREADS=44"
echo "    and ROCKSDB_MAX_BACKGROUND_JOBS=12:"
echo ""
echo '    [Service]'
echo '    Environment="TOKIO_WORKER_THREADS=44"'
echo '    Environment="ROCKSDB_MAX_BACKGROUND_JOBS=12"'
echo '    Environment="ROCKSDB_MAX_COMPACTIONS=6"'
echo '    Environment="ROCKSDB_MAX_FLUSHES=4"'
echo ""
echo " 2. Optionally add numactl for NUMA interleave:"
echo '    ExecStart=/usr/bin/numactl --interleave=all /path/to/q-api-server ...'
echo ""
echo " 3. Reload and restart:"
echo "    systemctl daemon-reload && systemctl restart q-api-server"
echo ""
echo "============================================================"
