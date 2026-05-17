use serde::{Deserialize, Serialize};
use sysinfo::{System, Disks};
use tracing::{info, warn};

/// System resource profile detected from hardware
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SystemResourceProfile {
    /// Total available disk space in GB
    pub total_disk_gb: u64,
    /// Available disk space in GB
    pub available_disk_gb: u64,
    /// Total RAM in GB
    pub total_ram_gb: u64,
    /// Available RAM in GB
    pub available_ram_gb: u64,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Device classification based on resources
    pub device_class: DeviceClass,
}

/// Device classification based on available resources
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeviceClass {
    /// Mobile/embedded device: <10 GB disk, <4 GB RAM, 1-4 cores
    Mobile,
    /// Desktop/workstation: 10-200 GB disk, 4-16 GB RAM, 4-8 cores
    Desktop,
    /// Server/data center: 200+ GB disk, 16+ GB RAM, 8+ cores
    Server,
    /// IoT/minimal device: <5 GB disk, <2 GB RAM, 1-2 cores
    IoT,
}

impl SystemResourceProfile {
    /// Detect system resources automatically
    pub fn auto_detect() -> anyhow::Result<Self> {
        info!("🔍 Auto-detecting system resources...");

        let mut sys = System::new_all();
        sys.refresh_all();

        // Detect disk space
        let disks = Disks::new_with_refreshed_list();
        let (total_disk_bytes, available_disk_bytes) = disks
            .iter()
            .map(|disk| (disk.total_space(), disk.available_space()))
            .fold((0u64, 0u64), |(total_acc, avail_acc), (total, avail)| {
                (total_acc + total, avail_acc + avail)
            });

        let total_disk_gb = total_disk_bytes / 1_000_000_000;
        let available_disk_gb = available_disk_bytes / 1_000_000_000;

        // Detect RAM
        let total_ram_bytes = sys.total_memory();
        let available_ram_bytes = sys.available_memory();
        let total_ram_gb = total_ram_bytes / 1_000_000_000;
        let available_ram_gb = available_ram_bytes / 1_000_000_000;

        // Detect CPU cores
        let cpu_cores = sys.cpus().len();

        // Classify device
        let device_class = Self::classify_device(available_disk_gb, total_ram_gb, cpu_cores);

        info!("📊 System Resources Detected:");
        info!("   💾 Disk: {} GB total, {} GB available", total_disk_gb, available_disk_gb);
        info!("   🧠 RAM: {} GB total, {} GB available", total_ram_gb, available_ram_gb);
        info!("   ⚙️  CPU: {} cores", cpu_cores);
        info!("   🏷️  Device Class: {:?}", device_class);

        Ok(Self {
            total_disk_gb,
            available_disk_gb,
            total_ram_gb,
            available_ram_gb,
            cpu_cores,
            device_class,
        })
    }

    /// Classify device based on resources
    fn classify_device(disk_gb: u64, ram_gb: u64, cores: usize) -> DeviceClass {
        if disk_gb < 5 && ram_gb < 2 {
            DeviceClass::IoT
        } else if disk_gb < 10 && ram_gb < 4 {
            DeviceClass::Mobile
        } else if disk_gb < 200 && ram_gb < 16 && cores < 8 {
            DeviceClass::Desktop
        } else {
            DeviceClass::Server
        }
    }

    /// Check if resources meet minimum requirements for node operation
    pub fn meets_minimum_requirements(&self) -> bool {
        // Minimum: 5 GB disk, 1 GB RAM, 1 core
        if self.available_disk_gb < 5 {
            warn!("⚠️  Insufficient disk space: {} GB (need 5+ GB)", self.available_disk_gb);
            return false;
        }

        if self.available_ram_gb < 1 {
            warn!("⚠️  Insufficient RAM: {} GB (need 1+ GB)", self.available_ram_gb);
            return false;
        }

        if self.cpu_cores < 1 {
            warn!("⚠️  No CPU cores detected");
            return false;
        }

        true
    }

    /// Get storage health indicator (percentage available)
    pub fn storage_health_percentage(&self) -> u8 {
        if self.total_disk_gb == 0 {
            return 0;
        }
        ((self.available_disk_gb * 100) / self.total_disk_gb) as u8
    }

    /// Get RAM health indicator (percentage available)
    pub fn ram_health_percentage(&self) -> u8 {
        if self.total_ram_gb == 0 {
            return 0;
        }
        ((self.available_ram_gb * 100) / self.total_ram_gb) as u8
    }

    /// Check if system is under resource pressure
    pub fn is_under_pressure(&self) -> bool {
        self.storage_health_percentage() < 20 || self.ram_health_percentage() < 30
    }
}

impl DeviceClass {
    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            DeviceClass::IoT => "IoT/Minimal Device",
            DeviceClass::Mobile => "Mobile/Embedded Device",
            DeviceClass::Desktop => "Desktop/Workstation",
            DeviceClass::Server => "Server/Data Center",
        }
    }

    /// Get recommended storage allocation percentage
    pub fn recommended_storage_percentage(&self) -> u8 {
        match self {
            DeviceClass::IoT => 50,      // Use 50% of available disk
            DeviceClass::Mobile => 60,   // Use 60% of available disk
            DeviceClass::Desktop => 70,  // Use 70% of available disk
            DeviceClass::Server => 80,   // Use 80% of available disk
        }
    }

    /// Get recommended max concurrent P2P connections
    pub fn max_concurrent_syncs(&self) -> u32 {
        match self {
            DeviceClass::IoT => 1,
            DeviceClass::Mobile => 2,
            DeviceClass::Desktop => 4,
            DeviceClass::Server => 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_classification() {
        assert_eq!(
            SystemResourceProfile::classify_device(3, 1, 2),
            DeviceClass::IoT
        );
        assert_eq!(
            SystemResourceProfile::classify_device(8, 3, 4),
            DeviceClass::Mobile
        );
        assert_eq!(
            SystemResourceProfile::classify_device(100, 8, 6),
            DeviceClass::Desktop
        );
        assert_eq!(
            SystemResourceProfile::classify_device(500, 32, 16),
            DeviceClass::Server
        );
    }

    #[test]
    fn test_auto_detect() {
        let profile = SystemResourceProfile::auto_detect().unwrap();
        assert!(profile.cpu_cores > 0);
        assert!(profile.total_disk_gb > 0);
        assert!(profile.total_ram_gb > 0);
    }
}
