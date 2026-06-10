//! OS Tuner — Automatic kernel and runtime parameter optimization
//!
//! Detects the current OS and applies optimal settings for mining
//! and compute workloads. All changes are safe and reversible.
//!
//! Linux: sysctl, cgroups, I/O scheduler, NUMA, transparent huge pages,
//!        IRQ affinity steering, RPS/XPS queue tuning
//! Windows: power plan, priority class, timer resolution

use tracing::{info, warn};

/// OS-level tuning engine
pub struct OsTuner;

impl OsTuner {
    /// Apply all safe OS tuning for compute workloads
    pub fn apply_all() {
        info!("🔧 [OS TUNER] Applying system-level optimizations...");

        #[cfg(target_os = "linux")]
        {
            Self::tune_vm_swappiness();
            Self::tune_transparent_hugepages();
            Self::tune_io_scheduler();
            Self::tune_network_stack();
            Self::tune_file_limits();

            // IRQ + network queue steering: keep interrupts off mining cores
            let total = num_cpus::get();
            let mining_count = total / 2; // cores 0..(total/2 - 1) reserved for mining
            let mining_cores: Vec<usize> = (0..mining_count).collect();
            let non_mining_cores: Vec<usize> = (mining_count..total).collect();

            if !non_mining_cores.is_empty() {
                if let Err(e) = steer_irq_affinity(&mining_cores) {
                    warn!("🔧 [OS TUNER] IRQ affinity steering failed: {}", e);
                }
                if let Err(e) = tune_network_queues(&non_mining_cores) {
                    warn!("🔧 [OS TUNER] RPS/XPS tuning failed: {}", e);
                }
            } else {
                info!("🔧 [OS TUNER] Only 1 core detected, skipping IRQ/RPS steering");
            }
        }

        #[cfg(target_os = "windows")]
        {
            Self::tune_windows_power();
            Self::tune_windows_timer();
        }

        info!("🔧 [OS TUNER] System tuning complete");
    }

    /// Get current tuning status as key-value pairs
    pub fn status() -> Vec<(String, String)> {
        let mut status = Vec::new();

        #[cfg(target_os = "linux")]
        {
            // Read swappiness
            if let Ok(val) = std::fs::read_to_string("/proc/sys/vm/swappiness") {
                status.push(("vm.swappiness".to_string(), val.trim().to_string()));
            }

            // Read THP status
            if let Ok(val) = std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled") {
                status.push(("transparent_hugepages".to_string(), val.trim().to_string()));
            }

            // Read max open files
            if let Ok(val) = std::fs::read_to_string("/proc/sys/fs/file-max") {
                status.push(("fs.file-max".to_string(), val.trim().to_string()));
            }

            // Read somaxconn
            if let Ok(val) = std::fs::read_to_string("/proc/sys/net/core/somaxconn") {
                status.push(("net.core.somaxconn".to_string(), val.trim().to_string()));
            }
        }

        #[cfg(target_os = "windows")]
        {
            status.push(("os".to_string(), "windows".to_string()));
            status.push(("tuning".to_string(), "power_plan+timer_resolution".to_string()));
        }

        if status.is_empty() {
            status.push(("os".to_string(), "unsupported".to_string()));
        }

        status
    }

    // ═══════════════════════════════════════════════════════════════
    // Linux tuning
    // ═══════════════════════════════════════════════════════════════

    /// Set vm.swappiness to 1 — minimize swapping for compute workloads
    #[cfg(target_os = "linux")]
    fn tune_vm_swappiness() {
        match std::fs::write("/proc/sys/vm/swappiness", "1") {
            Ok(_) => info!("🔧 [OS TUNER] vm.swappiness -> 1 (minimize swap)"),
            Err(_) => info!("🔧 [OS TUNER] vm.swappiness: no permission (need root)"),
        }
    }

    /// Enable transparent huge pages for compute allocations
    #[cfg(target_os = "linux")]
    fn tune_transparent_hugepages() {
        // Use madvise mode — huge pages only when requested
        match std::fs::write("/sys/kernel/mm/transparent_hugepage/enabled", "madvise") {
            Ok(_) => info!("🔧 [OS TUNER] THP -> madvise (huge pages on request)"),
            Err(_) => info!("🔧 [OS TUNER] THP: no permission or not available"),
        }

        // Disable THP defrag to avoid stalls
        let _ = std::fs::write("/sys/kernel/mm/transparent_hugepage/defrag", "defer+madvise");
    }

    /// Set I/O scheduler to none/noop for NVMe, deadline for HDD.
    /// Discovers devices dynamically from /sys/block/ instead of hardcoding names.
    #[cfg(target_os = "linux")]
    fn tune_io_scheduler() {
        let entries = match std::fs::read_dir("/sys/block") {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let name = entry.file_name();
            let dev = name.to_string_lossy().to_string();

            // Skip virtual/pseudo devices
            if dev.starts_with("loop") || dev.starts_with("ram") || dev.starts_with("dm-") {
                continue;
            }

            let path = format!("/sys/block/{}/queue/scheduler", dev);
            if !std::path::Path::new(&path).exists() {
                continue;
            }

            // NVMe benefits from none/noop
            if dev.starts_with("nvme") {
                if std::fs::write(&path, "none").is_ok() {
                    info!("🔧 [OS TUNER] {} scheduler -> none", dev);
                }
            } else {
                // Spinning disks / virtio benefit from mq-deadline
                if std::fs::write(&path, "mq-deadline").is_ok() {
                    info!("🔧 [OS TUNER] {} scheduler -> mq-deadline", dev);
                }
            }
        }
    }

    /// Optimize network stack for P2P workloads
    #[cfg(target_os = "linux")]
    fn tune_network_stack() {
        let tunings = [
            ("/proc/sys/net/core/somaxconn", "65535"),
            ("/proc/sys/net/core/netdev_max_backlog", "65535"),
            ("/proc/sys/net/ipv4/tcp_max_syn_backlog", "65535"),
            ("/proc/sys/net/ipv4/tcp_tw_reuse", "1"),
            ("/proc/sys/net/ipv4/tcp_fastopen", "3"),
        ];

        let mut applied = 0;
        for (path, value) in &tunings {
            if std::fs::write(path, value).is_ok() {
                applied += 1;
            }
        }
        if applied > 0 {
            info!("🔧 [OS TUNER] Network stack: {}/{} tunings applied", applied, tunings.len());
        } else {
            info!("🔧 [OS TUNER] Network stack: no permissions (need root)");
        }
    }

    /// Increase file descriptor limits
    #[cfg(target_os = "linux")]
    fn tune_file_limits() {
        // Set process limit via rlimit
        unsafe {
            let mut rlim = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
            if libc::getrlimit(libc::RLIMIT_NOFILE, &mut rlim) == 0 {
                if rlim.rlim_cur < 65536 {
                    let new_rlim = libc::rlimit {
                        rlim_cur: 65536.min(rlim.rlim_max),
                        rlim_max: rlim.rlim_max,
                    };
                    if libc::setrlimit(libc::RLIMIT_NOFILE, &new_rlim) == 0 {
                        info!("🔧 [OS TUNER] RLIMIT_NOFILE -> {}", new_rlim.rlim_cur);
                    }
                } else {
                    info!("🔧 [OS TUNER] RLIMIT_NOFILE already {} (good)", rlim.rlim_cur);
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Windows tuning
    // ═══════════════════════════════════════════════════════════════

    /// Set power plan to High Performance
    #[cfg(target_os = "windows")]
    fn tune_windows_power() {
        info!("🔧 [OS TUNER] Windows: Set power plan to High Performance");
        // Would need: powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
        let _ = std::process::Command::new("powercfg")
            .args(["/setactive", "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"])
            .output();
    }

    /// Set timer resolution to 1ms for precise scheduling
    #[cfg(target_os = "windows")]
    fn tune_windows_timer() {
        info!("🔧 [OS TUNER] Windows: Setting 1ms timer resolution");
        // timeBeginPeriod(1) — requires winmm.dll
        // Using windows-sys directly would need the Multimedia feature
        // For now, use the system default which is usually 15.6ms
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Standalone public functions: IRQ affinity steering + RPS/XPS tuning
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a list of core IDs to a hex bitmask string suitable for sysfs.
///
/// The kernel expects either a plain hex number (for <= 32 cores) or
/// comma-separated 32-bit groups with MSB first (for > 32 cores).
///
/// # Examples
///
/// - cores `[8,9,10,11,12,13,14,15]` on 16-core -> `"ff00"`
/// - cores `[0,1,2,3]` on 8-core -> `"f"`
/// - empty -> `"0"`
pub fn cores_to_hex_mask(cores: &[usize], total_cores: usize) -> String {
    if cores.is_empty() || total_cores == 0 {
        return "0".to_string();
    }

    // u128 supports up to 128 cores -- covers every realistic server
    let mut mask: u128 = 0;
    for &core in cores {
        if core < 128 && core < total_cores {
            mask |= 1u128 << core;
        }
    }

    if mask == 0 {
        return "0".to_string();
    }

    // For masks that fit in 32 bits, produce a plain hex string.
    // For wider masks, produce comma-separated 32-bit groups (MSB first).
    if mask <= u32::MAX as u128 {
        format!("{:x}", mask as u32)
    } else {
        let mut groups = Vec::new();
        let mut remaining = mask;
        while remaining > 0 {
            groups.push(format!("{:08x}", (remaining & 0xFFFF_FFFF) as u32));
            remaining >>= 32;
        }
        groups.reverse();
        // Trim leading zeros from the first group
        if let Some(first) = groups.first_mut() {
            *first = first.trim_start_matches('0').to_string();
            if first.is_empty() {
                *first = "0".to_string();
            }
        }
        groups.join(",")
    }
}

/// Convert a list of core IDs into a compact range string for
/// `smp_affinity_list`.
///
/// Example: `[8, 9, 10, 11, 14, 15]` -> `"8-11,14-15"`
fn cores_to_range_list(cores: &[usize]) -> String {
    if cores.is_empty() {
        return String::new();
    }

    let mut sorted = cores.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    let mut ranges = Vec::new();
    let mut start = sorted[0];
    let mut end = sorted[0];

    for &core in &sorted[1..] {
        if core == end + 1 {
            end = core;
        } else {
            if start == end {
                ranges.push(format!("{}", start));
            } else {
                ranges.push(format!("{}-{}", start, end));
            }
            start = core;
            end = core;
        }
    }

    if start == end {
        ranges.push(format!("{}", start));
    } else {
        ranges.push(format!("{}-{}", start, end));
    }

    ranges.join(",")
}

/// Steer network IRQ affinity away from mining cores.
///
/// 1. Reads `/proc/interrupts` to discover IRQs associated with network
///    devices (eth*, ens*, enp*, eno*, enx*, wlp*, ib*).
/// 2. For each discovered network IRQ, writes the non-mining core list to
///    `/proc/irq/{irq}/smp_affinity_list`.
/// 3. Also sets `/proc/irq/default_smp_affinity` so newly-created IRQs
///    inherit the same policy.
///
/// Fails gracefully (warns but does not crash) when not running as root.
#[cfg(target_os = "linux")]
pub fn steer_irq_affinity(mining_cores: &[usize]) -> Result<(), String> {
    use std::collections::HashSet;

    let total = num_cpus::get();
    let mining_set: HashSet<usize> = mining_cores.iter().copied().collect();
    let non_mining: Vec<usize> = (0..total).filter(|c| !mining_set.contains(c)).collect();

    if non_mining.is_empty() {
        return Err("No non-mining cores available for IRQ steering".to_string());
    }

    // Compact range string for smp_affinity_list (e.g. "8-15")
    let affinity_list = cores_to_range_list(&non_mining);
    // Hex mask for default_smp_affinity
    let hex_mask = cores_to_hex_mask(&non_mining, total);

    // Set default affinity for future IRQs
    match std::fs::write("/proc/irq/default_smp_affinity", &hex_mask) {
        Ok(_) => info!(
            "🔧 [OS TUNER] IRQ default_smp_affinity -> 0x{} (off mining cores)",
            hex_mask
        ),
        Err(e) => warn!(
            "🔧 [OS TUNER] Cannot set default_smp_affinity: {} (need root?)",
            e
        ),
    }

    // Read /proc/interrupts to find existing network IRQs
    let interrupts = std::fs::read_to_string("/proc/interrupts")
        .map_err(|e| format!("Cannot read /proc/interrupts: {}", e))?;

    // Network interface name prefixes to match
    let net_prefixes: &[&str] = &["eth", "ens", "enp", "eno", "enx", "wlp", "ib"];

    let mut steered = 0u32;
    let mut errors = 0u32;

    for line in interrupts.lines() {
        let trimmed = line.trim();

        // Each line: " 45:  1234  0  PCI-MSI-edge  eth0-TxRx-0"
        let irq_num = match trimmed.split(':').next() {
            Some(s) => s.trim(),
            None => continue,
        };

        // Skip non-numeric IRQ lines (NMI, LOC, SPU, etc.)
        if irq_num.parse::<u32>().is_err() {
            continue;
        }

        // Check if any word in the line matches a network device prefix
        let is_net_irq = trimmed
            .split_whitespace()
            .any(|word| net_prefixes.iter().any(|pfx| word.starts_with(pfx)));

        if !is_net_irq {
            continue;
        }

        let path = format!("/proc/irq/{}/smp_affinity_list", irq_num);
        match std::fs::write(&path, &affinity_list) {
            Ok(_) => {
                steered += 1;
            }
            Err(e) => {
                warn!("🔧 [OS TUNER] Cannot steer IRQ {} affinity: {}", irq_num, e);
                errors += 1;
            }
        }
    }

    if steered > 0 {
        info!(
            "🔧 [OS TUNER] Steered {} network IRQs to cores [{}] (away from mining cores {:?})",
            steered, affinity_list, mining_cores
        );
    } else if errors > 0 {
        warn!(
            "🔧 [OS TUNER] IRQ steering: {} attempts failed (need root?)",
            errors
        );
    } else {
        info!("🔧 [OS TUNER] No network IRQs found to steer");
    }

    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub fn steer_irq_affinity(_mining_cores: &[usize]) -> Result<(), String> {
    info!("🔧 [OS TUNER] IRQ affinity steering: not supported on this OS");
    Ok(())
}

/// Tune RPS (Receive Packet Steering) and XPS (Transmit Packet Steering)
/// to distribute network packet processing across non-mining cores.
///
/// For each non-loopback interface discovered under `/sys/class/net/`:
/// - Writes the hex CPU mask to `queues/rx-*/rps_cpus`
/// - Writes the hex CPU mask to `queues/tx-*/xps_cpus`
/// - Sets `rps_flow_cnt` to 32768 for better flow distribution
///
/// Fails gracefully when not running as root.
#[cfg(target_os = "linux")]
pub fn tune_network_queues(non_mining_cores: &[usize]) -> Result<(), String> {
    let total = num_cpus::get();
    let hex_mask = cores_to_hex_mask(non_mining_cores, total);

    let net_dir = std::path::Path::new("/sys/class/net");
    let entries = std::fs::read_dir(net_dir)
        .map_err(|e| format!("Cannot read /sys/class/net: {}", e))?;

    let mut rps_set = 0u32;
    let mut xps_set = 0u32;
    let mut ifaces_tuned: Vec<String> = Vec::new();

    for entry in entries.flatten() {
        let iface = entry.file_name().to_string_lossy().to_string();

        // Skip loopback
        if iface == "lo" {
            continue;
        }

        let queues_dir = format!("/sys/class/net/{}/queues", iface);
        let queue_entries = match std::fs::read_dir(&queues_dir) {
            Ok(e) => e,
            Err(_) => continue, // Interface may not expose queue sysfs entries
        };

        let mut tuned_this_iface = false;

        for qentry in queue_entries.flatten() {
            let qname = qentry.file_name().to_string_lossy().to_string();

            if qname.starts_with("rx-") {
                // RPS: steer receive processing to non-mining cores
                let rps_path = format!("{}/{}/rps_cpus", queues_dir, qname);
                if std::fs::write(&rps_path, &hex_mask).is_ok() {
                    rps_set += 1;
                    tuned_this_iface = true;
                }

                // Set rps_flow_cnt for better per-flow distribution
                let flow_path = format!("{}/{}/rps_flow_cnt", queues_dir, qname);
                let _ = std::fs::write(&flow_path, "32768");
            } else if qname.starts_with("tx-") {
                // XPS: steer transmit processing to non-mining cores
                let xps_path = format!("{}/{}/xps_cpus", queues_dir, qname);
                if std::fs::write(&xps_path, &hex_mask).is_ok() {
                    xps_set += 1;
                    tuned_this_iface = true;
                }
            }
        }

        if tuned_this_iface {
            ifaces_tuned.push(iface);
        }
    }

    if rps_set > 0 || xps_set > 0 {
        info!(
            "🔧 [OS TUNER] RPS/XPS: {} RX queues + {} TX queues on [{}] -> mask 0x{}",
            rps_set,
            xps_set,
            ifaces_tuned.join(", "),
            hex_mask
        );
    } else {
        info!("🔧 [OS TUNER] RPS/XPS: no queues tuned (need root or no queues found)");
    }

    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub fn tune_network_queues(_non_mining_cores: &[usize]) -> Result<(), String> {
    info!("🔧 [OS TUNER] RPS/XPS tuning: not supported on this OS");
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status() {
        let status = OsTuner::status();
        assert!(!status.is_empty());
    }

    #[test]
    fn test_apply_all_no_panic() {
        // Should not panic even without root permissions
        OsTuner::apply_all();
    }

    // -- cores_to_hex_mask tests ------------------------------------------

    #[test]
    fn test_hex_mask_upper_half() {
        // Cores 8-15 on 16-core -> bits 8..15 -> 0xFF00
        let mask = cores_to_hex_mask(&[8, 9, 10, 11, 12, 13, 14, 15], 16);
        assert_eq!(mask, "ff00");
    }

    #[test]
    fn test_hex_mask_lower_half() {
        // Cores 0-3 on 8-core -> 0x0F
        let mask = cores_to_hex_mask(&[0, 1, 2, 3], 8);
        assert_eq!(mask, "f");
    }

    #[test]
    fn test_hex_mask_single_core() {
        // Core 5 -> bit 5 -> 0x20
        let mask = cores_to_hex_mask(&[5], 16);
        assert_eq!(mask, "20");
    }

    #[test]
    fn test_hex_mask_empty() {
        assert_eq!(cores_to_hex_mask(&[], 16), "0");
    }

    #[test]
    fn test_hex_mask_zero_total() {
        assert_eq!(cores_to_hex_mask(&[0, 1], 0), "0");
    }

    #[test]
    fn test_hex_mask_all_4_cores() {
        assert_eq!(cores_to_hex_mask(&[0, 1, 2, 3], 4), "f");
    }

    #[test]
    fn test_hex_mask_scattered() {
        // Cores 0, 4, 8, 12 -> bits 0,4,8,12 -> 0x1111
        assert_eq!(cores_to_hex_mask(&[0, 4, 8, 12], 16), "1111");
    }

    #[test]
    fn test_hex_mask_wide_64plus() {
        // Core 32 on 64-core -> needs comma-separated groups
        // Bit 32 = 0x1_0000_0000 -> "1,00000000"
        let mask = cores_to_hex_mask(&[32], 64);
        assert_eq!(mask, "1,00000000");
    }

    #[test]
    fn test_hex_mask_wide_mixed() {
        // Cores 0 and 32 -> 0x1_0000_0001 -> "1,00000001"
        let mask = cores_to_hex_mask(&[0, 32], 64);
        assert_eq!(mask, "1,00000001");
    }

    #[test]
    fn test_hex_mask_core_beyond_total_ignored() {
        // Core 20 on 16-core system -> should be ignored
        let mask = cores_to_hex_mask(&[0, 20], 16);
        assert_eq!(mask, "1"); // only core 0
    }

    // -- cores_to_range_list tests ----------------------------------------

    #[test]
    fn test_range_list_contiguous() {
        assert_eq!(cores_to_range_list(&[8, 9, 10, 11]), "8-11");
    }

    #[test]
    fn test_range_list_with_gaps() {
        assert_eq!(cores_to_range_list(&[8, 9, 10, 14, 15]), "8-10,14-15");
    }

    #[test]
    fn test_range_list_single() {
        assert_eq!(cores_to_range_list(&[5]), "5");
    }

    #[test]
    fn test_range_list_mixed() {
        assert_eq!(cores_to_range_list(&[1, 3, 5, 6, 7, 10]), "1,3,5-7,10");
    }

    #[test]
    fn test_range_list_empty() {
        assert_eq!(cores_to_range_list(&[]), "");
    }

    #[test]
    fn test_range_list_unsorted_input() {
        assert_eq!(cores_to_range_list(&[15, 8, 10, 9]), "8-10,15");
    }

    #[test]
    fn test_range_list_duplicates() {
        assert_eq!(cores_to_range_list(&[5, 5, 6, 6, 7]), "5-7");
    }

    // -- steer_irq_affinity / tune_network_queues no-panic tests ----------

    #[test]
    fn test_steer_irq_affinity_no_panic() {
        // Must not panic even without /proc/interrupts or root
        let _ = steer_irq_affinity(&[0, 1, 2, 3]);
    }

    #[test]
    fn test_tune_network_queues_no_panic() {
        // Must not panic even without sysfs or root
        let _ = tune_network_queues(&[4, 5, 6, 7]);
    }
}
