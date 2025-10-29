//! Device Capability Detection for Distributed AI
//!
//! This module detects the hardware capabilities of the local node:
//! - CPU: cores and RAM
//! - CUDA: NVIDIA GPUs with VRAM
//! - Metal: Apple GPUs with unified memory
//!
//! The detected capabilities are used to determine how many model layers
//! this node can handle during distributed inference.

use crate::types::DeviceCapability;
use anyhow::{anyhow, Result};
use sysinfo::System;
use tracing::{debug, info};

#[allow(unused_imports)]
use tracing::warn;

/// Capability detector for local hardware
pub struct CapabilityDetector {
    system: System,
}

impl CapabilityDetector {
    /// Create a new capability detector
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        info!("🔍 Initializing device capability detector");
        Self { system }
    }

    /// Detect the best available device capability
    pub fn detect(&mut self) -> Result<DeviceCapability> {
        info!("🔎 Detecting device capabilities...");

        // Refresh system info
        self.system.refresh_all();

        // Try to detect in order of preference: CUDA > Metal > CPU
        if let Ok(cuda_cap) = self.detect_cuda() {
            info!("✅ Detected CUDA GPU: {} GB VRAM", cuda_cap.vram_gb());
            return Ok(cuda_cap);
        }

        if let Ok(metal_cap) = self.detect_metal() {
            info!("✅ Detected Metal GPU: {} GB unified memory", metal_cap.vram_gb());
            return Ok(metal_cap);
        }

        // Fallback to CPU
        let cpu_cap = self.detect_cpu()?;
        info!("✅ Using CPU: {} cores, {} GB RAM", cpu_cap.cores(), cpu_cap.ram_gb());
        Ok(cpu_cap)
    }

    /// Detect CPU capabilities
    fn detect_cpu(&mut self) -> Result<DeviceCapability> {
        self.system.refresh_cpu();
        self.system.refresh_memory();

        let cores = self.system.cpus().len();
        let ram_gb = (self.system.total_memory() / 1024 / 1024 / 1024) as usize;

        if cores == 0 || ram_gb == 0 {
            return Err(anyhow!("Failed to detect CPU capabilities"));
        }

        debug!("CPU detection: {} cores, {} GB RAM", cores, ram_gb);
        Ok(DeviceCapability::CPU { cores, ram_gb })
    }

    /// Detect CUDA GPU capabilities
    fn detect_cuda(&self) -> Result<DeviceCapability> {
        // Check if CUDA is available by looking for nvidia-smi
        #[cfg(target_os = "linux")]
        {
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=memory.total,compute_cap")
                .arg("--format=csv,noheader,nounits")
                .output()
            {
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    let lines: Vec<&str> = output_str.trim().lines().collect();

                    if !lines.is_empty() {
                        // Parse first GPU (for now, we'll support single GPU)
                        let parts: Vec<&str> = lines[0].split(',').collect();
                        if parts.len() >= 2 {
                            let vram_mb: f64 = parts[0].trim().parse().unwrap_or(0.0);
                            let vram_gb = (vram_mb / 1024.0).ceil() as usize;
                            let compute_capability = parts[1].trim().to_string();

                            debug!("CUDA detection: {} GB VRAM, compute capability {}",
                                   vram_gb, compute_capability);

                            return Ok(DeviceCapability::CUDA {
                                vram_gb,
                                compute_capability,
                            });
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // Try to detect CUDA on Windows using nvidia-smi.exe
            if let Ok(output) = std::process::Command::new("nvidia-smi.exe")
                .arg("--query-gpu=memory.total,compute_cap")
                .arg("--format=csv,noheader,nounits")
                .output()
            {
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    let lines: Vec<&str> = output_str.trim().lines().collect();

                    if !lines.is_empty() {
                        let parts: Vec<&str> = lines[0].split(',').collect();
                        if parts.len() >= 2 {
                            let vram_mb: f64 = parts[0].trim().parse().unwrap_or(0.0);
                            let vram_gb = (vram_mb / 1024.0).ceil() as usize;
                            let compute_capability = parts[1].trim().to_string();

                            debug!("CUDA detection (Windows): {} GB VRAM, compute capability {}",
                                   vram_gb, compute_capability);

                            return Ok(DeviceCapability::CUDA {
                                vram_gb,
                                compute_capability,
                            });
                        }
                    }
                }
            }
        }

        Err(anyhow!("No CUDA GPU detected"))
    }

    /// Detect Metal GPU capabilities (macOS only)
    fn detect_metal(&self) -> Result<DeviceCapability> {
        #[cfg(target_os = "macos")]
        {
            // Check if Metal is available using system_profiler
            if let Ok(output) = std::process::Command::new("system_profiler")
                .arg("SPDisplaysDataType")
                .output()
            {
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);

                    // Look for Metal support and VRAM information
                    if output_str.contains("Metal") {
                        // Try to extract VRAM/unified memory info
                        // macOS typically shows "Chipset Model" and "VRAM"
                        for line in output_str.lines() {
                            if line.contains("VRAM") || line.contains("Memory") {
                                // Parse VRAM size (e.g., "VRAM (Dynamic, Max): 8192 MB")
                                if let Some(vram_str) = line.split(':').nth(1) {
                                    let vram_str = vram_str.trim();

                                    // Extract numeric value
                                    let vram_mb: usize = vram_str
                                        .split_whitespace()
                                        .find_map(|s| s.parse::<usize>().ok())
                                        .unwrap_or(0);

                                    if vram_mb > 0 {
                                        let vram_gb = (vram_mb as f64 / 1024.0).ceil() as usize;

                                        debug!("Metal detection: {} GB unified memory", vram_gb);

                                        return Ok(DeviceCapability::Metal { vram_gb });
                                    }
                                }
                            }
                        }

                        // Fallback: assume reasonable default based on system RAM
                        // Apple Silicon typically has unified memory
                        let total_ram_gb = (self.system.total_memory() / 1024 / 1024 / 1024) as usize;
                        let vram_gb = (total_ram_gb / 2).max(8); // Conservative: use half of RAM

                        warn!("⚠️ Could not detect exact Metal VRAM, assuming {} GB", vram_gb);

                        return Ok(DeviceCapability::Metal { vram_gb });
                    }
                }
            }
        }

        Err(anyhow!("No Metal GPU detected"))
    }

    /// Get a human-readable description of the detected capability
    pub fn capability_description(capability: &DeviceCapability) -> String {
        match capability {
            DeviceCapability::CPU { cores, ram_gb } => {
                format!("CPU: {} cores, {} GB RAM (score: {})", cores, ram_gb, capability.score())
            }
            DeviceCapability::CUDA { vram_gb, compute_capability } => {
                format!("CUDA GPU: {} GB VRAM, compute {} (score: {})",
                        vram_gb, compute_capability, capability.score())
            }
            DeviceCapability::Metal { vram_gb } => {
                format!("Metal GPU: {} GB unified memory (score: {})", vram_gb, capability.score())
            }
        }
    }
}

impl Default for CapabilityDetector {
    fn default() -> Self {
        Self::new()
    }
}

// Extension trait for DeviceCapability
trait DeviceCapabilityExt {
    fn cores(&self) -> usize;
    fn ram_gb(&self) -> usize;
    fn vram_gb(&self) -> usize;
}

impl DeviceCapabilityExt for DeviceCapability {
    fn cores(&self) -> usize {
        match self {
            DeviceCapability::CPU { cores, .. } => *cores,
            _ => 0,
        }
    }

    fn ram_gb(&self) -> usize {
        match self {
            DeviceCapability::CPU { ram_gb, .. } => *ram_gb,
            _ => 0,
        }
    }

    fn vram_gb(&self) -> usize {
        match self {
            DeviceCapability::CUDA { vram_gb, .. } => *vram_gb,
            DeviceCapability::Metal { vram_gb } => *vram_gb,
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_detector_creation() {
        let detector = CapabilityDetector::new();
        assert!(detector.system.cpus().len() > 0);
    }

    #[test]
    fn test_cpu_detection() {
        let mut detector = CapabilityDetector::new();
        let cpu_cap = detector.detect_cpu();

        assert!(cpu_cap.is_ok());
        let cap = cpu_cap.unwrap();

        match cap {
            DeviceCapability::CPU { cores, ram_gb } => {
                assert!(cores > 0, "Should detect at least 1 core");
                assert!(ram_gb > 0, "Should detect at least 1 GB RAM");
            }
            _ => panic!("Expected CPU capability"),
        }
    }

    #[test]
    fn test_capability_description() {
        let cpu_cap = DeviceCapability::CPU { cores: 8, ram_gb: 16 };
        let desc = CapabilityDetector::capability_description(&cpu_cap);
        assert!(desc.contains("CPU"));
        assert!(desc.contains("8 cores"));
        assert!(desc.contains("16 GB RAM"));

        let cuda_cap = DeviceCapability::CUDA {
            vram_gb: 12,
            compute_capability: "8.0".to_string(),
        };
        let desc = CapabilityDetector::capability_description(&cuda_cap);
        assert!(desc.contains("CUDA"));
        assert!(desc.contains("12 GB VRAM"));
    }

    #[test]
    fn test_detect_best_available() {
        let mut detector = CapabilityDetector::new();
        let capability = detector.detect();

        assert!(capability.is_ok());
        let cap = capability.unwrap();

        // Should always be able to detect at least CPU
        println!("Detected capability: {}", CapabilityDetector::capability_description(&cap));
        assert!(cap.score() > 0);
    }
}
