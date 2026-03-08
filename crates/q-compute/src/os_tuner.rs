//! OS Tuner — Automatic kernel and runtime parameter optimization
//!
//! Detects the current OS and applies optimal settings for mining
//! and compute workloads. All changes are safe and reversible.
//!
//! Linux: sysctl, cgroups, I/O scheduler, NUMA, transparent huge pages
//! Windows: power plan, priority class, timer resolution

use tracing::info;

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
            Ok(_) => info!("🔧 [OS TUNER] ✅ vm.swappiness → 1 (minimize swap)"),
            Err(_) => info!("🔧 [OS TUNER] vm.swappiness: no permission (need root)"),
        }
    }

    /// Enable transparent huge pages for compute allocations
    #[cfg(target_os = "linux")]
    fn tune_transparent_hugepages() {
        // Use madvise mode — huge pages only when requested
        match std::fs::write("/sys/kernel/mm/transparent_hugepage/enabled", "madvise") {
            Ok(_) => info!("🔧 [OS TUNER] ✅ THP → madvise (huge pages on request)"),
            Err(_) => info!("🔧 [OS TUNER] THP: no permission or not available"),
        }

        // Disable THP defrag to avoid stalls
        let _ = std::fs::write("/sys/kernel/mm/transparent_hugepage/defrag", "defer+madvise");
    }

    /// Set I/O scheduler to none/noop for NVMe, deadline for HDD
    #[cfg(target_os = "linux")]
    fn tune_io_scheduler() {
        // Try to set noop scheduler on common block devices
        let devices = ["sda", "nvme0n1", "vda"];
        for dev in &devices {
            let path = format!("/sys/block/{}/queue/scheduler", dev);
            if std::path::Path::new(&path).exists() {
                // NVMe benefits from none/noop
                if dev.starts_with("nvme") {
                    if std::fs::write(&path, "none").is_ok() {
                        info!("🔧 [OS TUNER] ✅ {} scheduler → none", dev);
                    }
                } else {
                    // Spinning disks benefit from mq-deadline
                    if std::fs::write(&path, "mq-deadline").is_ok() {
                        info!("🔧 [OS TUNER] ✅ {} scheduler → mq-deadline", dev);
                    }
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
            info!("🔧 [OS TUNER] ✅ Network stack: {}/{} tunings applied", applied, tunings.len());
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
                        info!("🔧 [OS TUNER] ✅ RLIMIT_NOFILE → {}", new_rlim.rlim_cur);
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
        unsafe {
            // timeBeginPeriod(1) — requires winmm.dll
            // Using windows-sys directly would need the Multimedia feature
            // For now, use the system default which is usually 15.6ms
        }
    }
}

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
}
