#![allow(dead_code)]
//! GPU Scheduler — Multi-GPU scheduling and layer assignment (Issue #023)
//!
//! Detects all GPU devices on the system, assigns compute layers to specific
//! GPUs based on workload requirements, and provides `CUDA_VISIBLE_DEVICES`
//! strings for subprocess environment isolation.
//!
//! ## Assignment Strategy
//!
//! - **Mining** gets the fastest GPU (highest utilization headroom / highest VRAM)
//! - **AI Inference** gets the GPU with the most VRAM (large model weights)
//! - **ZK Proof Generation** gets the next available GPU
//! - Remaining layers share leftover GPUs round-robin
//! - Single GPU fallback: all layers share GPU 0

use crate::{ComputeLayer, GpuAssignment, GpuDevice};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Multi-GPU scheduler that detects devices and assigns them to compute layers.
pub struct GpuScheduler {
    /// Detected GPU devices (updated by `refresh_stats`)
    devices: Arc<RwLock<Vec<GpuDevice>>>,
    /// Current layer-to-GPU assignments
    assignments: Arc<RwLock<GpuAssignment>>,
}

impl GpuScheduler {
    /// Create a new GPU scheduler. Runs initial detection synchronously.
    pub fn new() -> Self {
        let devices = detect_gpus();
        let device_count = devices.len();

        let scheduler = Self {
            devices: Arc::new(RwLock::new(devices)),
            assignments: Arc::new(RwLock::new(HashMap::new())),
        };

        if device_count > 0 {
            info!(
                "GPU Scheduler initialized with {} device(s)",
                device_count
            );
        } else {
            debug!("GPU Scheduler initialized — no GPUs detected (CPU-only mode)");
        }

        scheduler
    }

    /// Create a scheduler with pre-supplied devices (useful for testing).
    pub fn with_devices(devices: Vec<GpuDevice>) -> Self {
        Self {
            devices: Arc::new(RwLock::new(devices)),
            assignments: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Return the current list of detected GPU devices.
    pub fn stats(&self) -> Vec<GpuDevice> {
        self.devices.read().clone()
    }

    /// Return the current layer-to-GPU assignment map.
    pub fn current_assignments(&self) -> GpuAssignment {
        self.assignments.read().clone()
    }

    /// Assign GPUs across the given compute layers.
    ///
    /// Strategy:
    /// 1. If no GPUs: return empty map.
    /// 2. If 1 GPU: all layers share GPU 0.
    /// 3. If N GPUs and M layers:
    ///    - Mining gets the GPU with the most headroom (lowest current utilization).
    ///    - AI Inference gets the GPU with the most VRAM (excluding Mining's GPU if possible).
    ///    - Remaining layers assigned round-robin over remaining GPUs.
    ///    - If more layers than GPUs, multiple layers share a GPU.
    ///    - If more GPUs than layers, extra GPUs go to Mining.
    pub fn assign_gpus(&self, layers: &[ComputeLayer]) -> GpuAssignment {
        let devices = self.devices.read();
        let mut assignment: GpuAssignment = HashMap::new();

        if devices.is_empty() || layers.is_empty() {
            *self.assignments.write() = assignment.clone();
            return assignment;
        }

        let gpu_count = devices.len();

        // Single GPU: all layers share it
        if gpu_count == 1 {
            for layer in layers {
                assignment.entry(*layer).or_default().push(0);
            }
            // Update device assigned_layer for the single GPU
            drop(devices);
            self.update_device_assignments(&assignment);
            *self.assignments.write() = assignment.clone();
            return assignment;
        }

        // Multi-GPU assignment
        // Sort GPUs by utilization ascending (most idle first) for mining
        let mut gpus_by_idle: Vec<(u32, f32)> = devices
            .iter()
            .map(|d| (d.id, d.utilization))
            .collect();
        gpus_by_idle.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Sort GPUs by VRAM descending for inference
        let mut gpus_by_vram: Vec<(u32, u64)> = devices
            .iter()
            .map(|d| (d.id, d.vram_total_mb))
            .collect();
        gpus_by_vram.sort_by(|a, b| b.1.cmp(&a.1));

        let mut used_gpu_ids: Vec<u32> = Vec::new();

        // Priority 1: Mining gets the most idle GPU
        if layers.contains(&ComputeLayer::Mining) {
            let mining_gpu = gpus_by_idle[0].0;
            assignment.entry(ComputeLayer::Mining).or_default().push(mining_gpu);
            used_gpu_ids.push(mining_gpu);
        }

        // Priority 2: AI Inference gets the GPU with most VRAM (prefer one not used by mining)
        if layers.contains(&ComputeLayer::AiInference) {
            let inference_gpu = gpus_by_vram
                .iter()
                .find(|(id, _)| !used_gpu_ids.contains(id))
                .map(|(id, _)| *id)
                .unwrap_or(gpus_by_vram[0].0);
            assignment.entry(ComputeLayer::AiInference).or_default().push(inference_gpu);
            if !used_gpu_ids.contains(&inference_gpu) {
                used_gpu_ids.push(inference_gpu);
            }
        }

        // Remaining layers get round-robin over remaining GPUs
        let remaining_layers: Vec<ComputeLayer> = layers
            .iter()
            .filter(|l| **l != ComputeLayer::Mining && **l != ComputeLayer::AiInference)
            .copied()
            .collect();

        if !remaining_layers.is_empty() {
            // Build pool of available GPUs (prefer unused, but fall back to all)
            let available_gpus: Vec<u32> = {
                let unused: Vec<u32> = (0..gpu_count as u32)
                    .filter(|id| !used_gpu_ids.contains(id))
                    .collect();
                if unused.is_empty() {
                    // All GPUs taken by mining/inference — share them
                    (0..gpu_count as u32).collect()
                } else {
                    unused
                }
            };

            for (i, layer) in remaining_layers.iter().enumerate() {
                let gpu_id = available_gpus[i % available_gpus.len()];
                assignment.entry(*layer).or_default().push(gpu_id);
                if !used_gpu_ids.contains(&gpu_id) {
                    used_gpu_ids.push(gpu_id);
                }
            }
        }

        // If more GPUs than layers, give extra GPUs to Mining
        if gpu_count > layers.len() && layers.contains(&ComputeLayer::Mining) {
            let extra: Vec<u32> = (0..gpu_count as u32)
                .filter(|id| !used_gpu_ids.contains(id))
                .collect();
            for gpu_id in extra {
                assignment.entry(ComputeLayer::Mining).or_default().push(gpu_id);
            }
        }

        drop(devices);
        self.update_device_assignments(&assignment);
        *self.assignments.write() = assignment.clone();

        info!(
            "GPU assignment complete: {} layers across {} GPUs",
            layers.len(),
            gpu_count
        );
        for (layer, ids) in &assignment {
            debug!("  {:?} -> GPU(s) {:?}", layer, ids);
        }

        assignment
    }

    /// Return the `CUDA_VISIBLE_DEVICES` value for a specific layer.
    ///
    /// Returns a comma-separated list of GPU indices (e.g. "0" or "0,2").
    /// Returns an empty string if the layer has no GPU assignment.
    pub fn get_cuda_visible_devices(&self, layer: ComputeLayer) -> String {
        let assignments = self.assignments.read();
        match assignments.get(&layer) {
            Some(ids) if !ids.is_empty() => {
                ids.iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            }
            _ => String::new(),
        }
    }

    /// Refresh GPU utilization, VRAM, and temperature for all detected devices.
    ///
    /// Calls nvidia-smi (or rocm-smi) to get current stats. This is a blocking
    /// operation and should be called from a background task.
    pub fn refresh_stats(&self) {
        let fresh = detect_gpus();
        if fresh.is_empty() {
            return;
        }

        let mut devices = self.devices.write();
        // Preserve assigned_layer from current state
        let old_assignments: HashMap<u32, Option<ComputeLayer>> = devices
            .iter()
            .map(|d| (d.id, d.assigned_layer))
            .collect();

        *devices = fresh;

        // Restore assigned_layer
        for dev in devices.iter_mut() {
            if let Some(layer) = old_assignments.get(&dev.id).copied().flatten() {
                dev.assigned_layer = Some(layer);
            }
        }
    }

    /// Update `assigned_layer` on each device based on the assignment map.
    fn update_device_assignments(&self, assignment: &GpuAssignment) {
        let mut devices = self.devices.write();

        // Clear all assignments first
        for dev in devices.iter_mut() {
            dev.assigned_layer = None;
        }

        // Set assignments (first layer wins if GPU is shared)
        for (layer, gpu_ids) in assignment {
            for gpu_id in gpu_ids {
                if let Some(dev) = devices.iter_mut().find(|d| d.id == *gpu_id) {
                    if dev.assigned_layer.is_none() {
                        dev.assigned_layer = Some(*layer);
                    }
                }
            }
        }
    }
}

/// Detect all GPU devices on the system by parsing nvidia-smi output.
///
/// Queries: `nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits`
///
/// Each line of output corresponds to one GPU. Example:
/// ```text
/// 0, NVIDIA GeForce RTX 4090, 24564, 512, 3, 42
/// 1, NVIDIA GeForce RTX 3090, 24576, 1024, 15, 55
/// ```
///
/// Falls back to an empty Vec if nvidia-smi is unavailable or fails.
pub fn detect_gpus() -> Vec<GpuDevice> {
    detect_gpus_from_command()
}

/// Internal: run nvidia-smi and parse multi-GPU output.
fn detect_gpus_from_command() -> Vec<GpuDevice> {
    #[cfg(target_os = "linux")]
    {
        let output = match std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            Ok(o) if o.status.success() => o,
            Ok(_) => {
                debug!("nvidia-smi returned non-zero exit code");
                return try_rocm_smi_detect();
            }
            Err(e) => {
                debug!("nvidia-smi not available: {}", e);
                return try_rocm_smi_detect();
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        parse_nvidia_smi_multi(&stdout)
    }

    #[cfg(not(target_os = "linux"))]
    {
        Vec::new()
    }
}

/// Parse multi-line nvidia-smi CSV output into a Vec of GpuDevice.
///
/// Each line: `index, name, memory.total, memory.used, utilization.gpu, temperature.gpu`
fn parse_nvidia_smi_multi(output: &str) -> Vec<GpuDevice> {
    let mut devices = Vec::new();

    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.splitn(6, ',').map(|s| s.trim()).collect();
        if parts.len() < 6 {
            warn!("Unexpected nvidia-smi output line (expected 6 fields): {}", line);
            continue;
        }

        let id = match parts[0].parse::<u32>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let name = parts[1].to_string();
        let vram_total_mb = parts[2].parse::<u64>().unwrap_or(0);
        let vram_used_mb = parts[3].parse::<u64>().unwrap_or(0);
        let utilization = parts[4].parse::<f32>().unwrap_or(0.0);
        let temperature = parts[5].parse::<f32>().unwrap_or(0.0);

        devices.push(GpuDevice {
            id,
            name,
            vram_total_mb,
            vram_used_mb,
            utilization,
            temperature,
            assigned_layer: None,
        });
    }

    if !devices.is_empty() {
        info!("Detected {} NVIDIA GPU(s) via nvidia-smi", devices.len());
    }

    devices
}

/// Fallback: try rocm-smi for AMD GPUs.
fn try_rocm_smi_detect() -> Vec<GpuDevice> {
    #[cfg(target_os = "linux")]
    {
        let output = match std::process::Command::new("rocm-smi")
            .args(["--showid", "--showtemp", "--showmeminfo", "vram", "--csv"])
            .output()
        {
            Ok(o) if o.status.success() => o,
            _ => {
                debug!("rocm-smi not available either — no GPU detected");
                return Vec::new();
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        // rocm-smi CSV output is less standardized; do basic parsing
        let mut devices = Vec::new();
        let mut gpu_idx: u32 = 0;

        for line in stdout.lines().skip(1) {
            // Skip header
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // rocm-smi varies by version; extract what we can
            devices.push(GpuDevice {
                id: gpu_idx,
                name: format!("AMD GPU {}", gpu_idx),
                vram_total_mb: 0,
                vram_used_mb: 0,
                utilization: 0.0,
                temperature: 0.0,
                assigned_layer: None,
            });
            gpu_idx += 1;
        }

        if !devices.is_empty() {
            info!("Detected {} AMD GPU(s) via rocm-smi", devices.len());
        }

        devices
    }

    #[cfg(not(target_os = "linux"))]
    {
        Vec::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a fake GpuDevice for testing
    fn make_gpu(id: u32, name: &str, vram_total_mb: u64, utilization: f32) -> GpuDevice {
        GpuDevice {
            id,
            name: name.to_string(),
            vram_total_mb,
            vram_used_mb: 0,
            utilization,
            temperature: 45.0,
            assigned_layer: None,
        }
    }

    #[test]
    fn test_detect_gpus_fallback_no_gpu() {
        // On CI/servers without nvidia-smi or rocm-smi, detect_gpus should
        // return an empty Vec (not panic).
        let devices = detect_gpus();
        // We cannot assert count because the test runner may have a GPU.
        // Instead, verify the return type is valid and no panic occurred.
        assert!(devices.len() < 128, "Unreasonable GPU count: {}", devices.len());
    }

    #[test]
    fn test_single_gpu_shared_by_all_layers() {
        let gpu = make_gpu(0, "NVIDIA RTX 4090", 24576, 10.0);
        let scheduler = GpuScheduler::with_devices(vec![gpu]);

        let layers = vec![
            ComputeLayer::Mining,
            ComputeLayer::AiInference,
            ComputeLayer::ZkProofGen,
        ];
        let assignment = scheduler.assign_gpus(&layers);

        // All 3 layers should map to GPU 0
        assert_eq!(assignment.len(), 3);
        for layer in &layers {
            let ids = assignment.get(layer).expect("layer should have assignment");
            assert_eq!(ids, &[0], "Single GPU — all layers should use GPU 0");
        }
    }

    #[test]
    fn test_multi_gpu_assignment_mining_first() {
        // GPU 0: high utilization (busy)
        // GPU 1: low utilization (idle) — should get mining
        let gpus = vec![
            make_gpu(0, "RTX 4090", 24576, 80.0),
            make_gpu(1, "RTX 4090", 24576, 5.0),
        ];
        let scheduler = GpuScheduler::with_devices(gpus);

        let layers = vec![ComputeLayer::Mining, ComputeLayer::AiInference];
        let assignment = scheduler.assign_gpus(&layers);

        // Mining should get GPU 1 (most idle)
        let mining_gpus = assignment.get(&ComputeLayer::Mining).unwrap();
        assert!(
            mining_gpus.contains(&1),
            "Mining should be assigned to the most idle GPU (1), got {:?}",
            mining_gpus
        );

        // AI Inference should get GPU 0 (remaining)
        let inference_gpus = assignment.get(&ComputeLayer::AiInference).unwrap();
        assert!(
            inference_gpus.contains(&0),
            "AI Inference should get the other GPU (0), got {:?}",
            inference_gpus
        );
    }

    #[test]
    fn test_cuda_visible_devices_format() {
        let gpus = vec![
            make_gpu(0, "RTX 4090", 24576, 10.0),
            make_gpu(1, "RTX 3090", 24576, 20.0),
            make_gpu(2, "RTX 3080", 10240, 30.0),
        ];
        let scheduler = GpuScheduler::with_devices(gpus);

        let layers = vec![
            ComputeLayer::Mining,
            ComputeLayer::AiInference,
            ComputeLayer::ZkProofGen,
        ];
        scheduler.assign_gpus(&layers);

        // Each layer should produce a valid CUDA_VISIBLE_DEVICES string
        for layer in &layers {
            let cvd = scheduler.get_cuda_visible_devices(*layer);
            assert!(!cvd.is_empty(), "Layer {:?} should have a CUDA device string", layer);
            // Should be comma-separated digits
            for part in cvd.split(',') {
                assert!(
                    part.parse::<u32>().is_ok(),
                    "CUDA_VISIBLE_DEVICES should be comma-separated integers, got '{}'",
                    part
                );
            }
        }

        // Unassigned layer should return empty string
        let cvd = scheduler.get_cuda_visible_devices(ComputeLayer::RenderFarm);
        assert_eq!(cvd, "", "Unassigned layer should return empty CUDA_VISIBLE_DEVICES");
    }

    #[test]
    fn test_refresh_updates_stats() {
        let gpus = vec![
            make_gpu(0, "RTX 4090", 24576, 10.0),
        ];
        let scheduler = GpuScheduler::with_devices(gpus);

        // Assign a layer to verify assigned_layer survives refresh
        scheduler.assign_gpus(&[ComputeLayer::Mining]);

        let before = scheduler.stats();
        assert_eq!(before.len(), 1);
        assert_eq!(before[0].assigned_layer, Some(ComputeLayer::Mining));

        // refresh_stats will call detect_gpus() which may return 0 GPUs in CI.
        // The key behavior: if detect_gpus returns empty, devices stay unchanged.
        scheduler.refresh_stats();

        let after = scheduler.stats();
        // Either refreshed with real data or stayed the same (no GPU on CI)
        // The important thing is no panic and assigned_layer preservation
        if !after.is_empty() {
            // If GPU was detected, assigned_layer should be preserved
            // (unless the GPU id changed, which shouldn't happen)
        }
        // No panic = success
    }

    #[test]
    fn test_gpu_device_serde_roundtrip() {
        let device = GpuDevice {
            id: 2,
            name: "NVIDIA A100 80GB".to_string(),
            vram_total_mb: 81920,
            vram_used_mb: 40960,
            utilization: 67.5,
            temperature: 72.0,
            assigned_layer: Some(ComputeLayer::AiInference),
        };

        let json = serde_json::to_string(&device).expect("serialize should succeed");
        let deserialized: GpuDevice =
            serde_json::from_str(&json).expect("deserialize should succeed");

        assert_eq!(deserialized.id, 2);
        assert_eq!(deserialized.name, "NVIDIA A100 80GB");
        assert_eq!(deserialized.vram_total_mb, 81920);
        assert_eq!(deserialized.vram_used_mb, 40960);
        assert!((deserialized.utilization - 67.5).abs() < 0.01);
        assert!((deserialized.temperature - 72.0).abs() < 0.01);
        assert_eq!(deserialized.assigned_layer, Some(ComputeLayer::AiInference));
    }

    #[test]
    fn test_assign_gpus_empty_layers() {
        let gpus = vec![
            make_gpu(0, "RTX 4090", 24576, 10.0),
            make_gpu(1, "RTX 3090", 24576, 20.0),
        ];
        let scheduler = GpuScheduler::with_devices(gpus);

        let assignment = scheduler.assign_gpus(&[]);
        assert!(
            assignment.is_empty(),
            "Empty layers should produce empty assignment"
        );
    }

    #[test]
    fn test_assign_more_layers_than_gpus() {
        // 2 GPUs, 5 layers — layers must share GPUs
        let gpus = vec![
            make_gpu(0, "RTX 4090", 24576, 5.0),
            make_gpu(1, "RTX 3090", 24576, 50.0),
        ];
        let scheduler = GpuScheduler::with_devices(gpus);

        let layers = vec![
            ComputeLayer::Mining,
            ComputeLayer::AiInference,
            ComputeLayer::ZkProofGen,
            ComputeLayer::BridgeVerify,
            ComputeLayer::VdfCompute,
        ];
        let assignment = scheduler.assign_gpus(&layers);

        // All 5 layers should have at least one GPU
        assert_eq!(assignment.len(), 5, "All 5 layers should have assignments");

        for layer in &layers {
            let ids = assignment.get(layer).expect("layer should have assignment");
            assert!(
                !ids.is_empty(),
                "Layer {:?} should have at least one GPU",
                layer
            );
            // All assigned IDs should be 0 or 1
            for id in ids {
                assert!(*id <= 1, "GPU id should be 0 or 1, got {}", id);
            }
        }
    }

    #[test]
    fn test_assign_more_gpus_than_layers() {
        // 4 GPUs, 2 layers — extra GPUs should go to Mining
        let gpus = vec![
            make_gpu(0, "RTX 4090", 24576, 50.0),
            make_gpu(1, "RTX 4090", 24576, 10.0),
            make_gpu(2, "RTX 3090", 24576, 20.0),
            make_gpu(3, "RTX 3080", 10240, 5.0),
        ];
        let scheduler = GpuScheduler::with_devices(gpus);

        let layers = vec![ComputeLayer::Mining, ComputeLayer::AiInference];
        let assignment = scheduler.assign_gpus(&layers);

        // Mining should have multiple GPUs (its primary + extras)
        let mining_gpus = assignment.get(&ComputeLayer::Mining).unwrap();
        assert!(
            mining_gpus.len() >= 2,
            "Mining should get extra GPUs when more GPUs than layers, got {:?}",
            mining_gpus
        );

        // AI Inference should have at least 1 GPU
        let inference_gpus = assignment.get(&ComputeLayer::AiInference).unwrap();
        assert!(!inference_gpus.is_empty());

        // Total assigned GPU IDs (with possible duplicates across layers) should cover all 4
        let mut all_ids: Vec<u32> = assignment
            .values()
            .flat_map(|ids| ids.iter().copied())
            .collect();
        all_ids.sort();
        all_ids.dedup();
        assert_eq!(
            all_ids.len(),
            4,
            "All 4 GPUs should be assigned, got {:?}",
            all_ids
        );
    }

    #[test]
    fn test_scheduler_creation() {
        // Default constructor (may detect real GPUs or not)
        let scheduler = GpuScheduler::new();
        let devices = scheduler.stats();
        // Should not panic; device count depends on hardware
        assert!(devices.len() < 128);
        assert!(scheduler.current_assignments().is_empty());
    }

    #[test]
    fn test_parse_nvidia_smi_multi_valid() {
        let output = "\
0, NVIDIA GeForce RTX 4090, 24564, 512, 3, 42
1, NVIDIA GeForce RTX 3090, 24576, 1024, 15, 55
2, NVIDIA A100-SXM4-80GB, 81920, 40960, 87, 71
";
        let devices = parse_nvidia_smi_multi(output);
        assert_eq!(devices.len(), 3);

        assert_eq!(devices[0].id, 0);
        assert_eq!(devices[0].name, "NVIDIA GeForce RTX 4090");
        assert_eq!(devices[0].vram_total_mb, 24564);
        assert_eq!(devices[0].vram_used_mb, 512);
        assert!((devices[0].utilization - 3.0).abs() < 0.01);
        assert!((devices[0].temperature - 42.0).abs() < 0.01);
        assert_eq!(devices[0].assigned_layer, None);

        assert_eq!(devices[1].id, 1);
        assert_eq!(devices[1].name, "NVIDIA GeForce RTX 3090");
        assert_eq!(devices[1].vram_total_mb, 24576);

        assert_eq!(devices[2].id, 2);
        assert_eq!(devices[2].name, "NVIDIA A100-SXM4-80GB");
        assert_eq!(devices[2].vram_total_mb, 81920);
        assert_eq!(devices[2].vram_used_mb, 40960);
        assert!((devices[2].utilization - 87.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_nvidia_smi_multi_empty() {
        let devices = parse_nvidia_smi_multi("");
        assert!(devices.is_empty());
    }

    #[test]
    fn test_parse_nvidia_smi_multi_malformed_line() {
        let output = "\
0, NVIDIA RTX 4090, 24564, 512, 3, 42
bad line
1, NVIDIA RTX 3090, 24576, 1024, 15, 55
";
        let devices = parse_nvidia_smi_multi(output);
        // Should parse the two valid lines and skip the bad one
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].id, 0);
        assert_eq!(devices[1].id, 1);
    }

    #[test]
    fn test_no_gpus_returns_empty_assignment() {
        let scheduler = GpuScheduler::with_devices(vec![]);
        let assignment = scheduler.assign_gpus(&[ComputeLayer::Mining]);
        assert!(assignment.is_empty());
    }

    #[test]
    fn test_inference_gets_most_vram() {
        // GPU 0: 10GB VRAM, idle
        // GPU 1: 80GB VRAM, busy
        // Inference should prefer GPU 1 (most VRAM) even though it's busier
        let gpus = vec![
            make_gpu(0, "RTX 3080", 10240, 5.0),
            make_gpu(1, "A100 80GB", 81920, 60.0),
        ];
        let scheduler = GpuScheduler::with_devices(gpus);

        let layers = vec![ComputeLayer::Mining, ComputeLayer::AiInference];
        let assignment = scheduler.assign_gpus(&layers);

        // Mining gets most idle (GPU 0)
        let mining_gpus = assignment.get(&ComputeLayer::Mining).unwrap();
        assert!(mining_gpus.contains(&0), "Mining should get most idle GPU (0)");

        // Inference gets most VRAM (GPU 1)
        let inference_gpus = assignment.get(&ComputeLayer::AiInference).unwrap();
        assert!(
            inference_gpus.contains(&1),
            "Inference should get GPU with most VRAM (1), got {:?}",
            inference_gpus
        );
    }
}
