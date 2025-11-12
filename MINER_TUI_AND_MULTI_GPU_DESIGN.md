# Q-Miner TUI & Multi-GPU Design

## Overview
Design document for implementing a beautiful terminal UI (TUI) for the Q-Miner with multi-GPU mining support inspired by mistral.rs GPU compute architecture.

## Architecture

### 1. TUI Interface Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Q-Miner v0.9.90-beta - Quantum-Enhanced Mining Dashboard        ⛏️ MINING  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─ Hashrate Performance ─────────────────────────────────────────────┐    │
│  │  Current:  245.3 MH/s  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░ 87%         │    │
│  │  Average:  238.7 MH/s                                              │    │
│  │  Peak:     267.1 MH/s  (4m ago)                                    │    │
│  │                                                                     │    │
│  │  ╭───────────────────────────────────────────────────────────╮    │    │
│  │  │ 270 ┤                                          ╭─╮         │    │    │
│  │  │ 250 ┤                    ╭─╮  ╭─╮     ╭─╮  ╭─╯ ╰─╮       │    │    │
│  │  │ 230 ┤         ╭─╮  ╭─╮  ╭╯ ╰──╯ ╰─╮ ╭─╯ ╰──╯     ╰─╮     │    │    │
│  │  │ 210 ┤    ╭─╮ ╭╯ ╰──╯ ╰──╯        ╰─╯             ╰─╮   │    │    │
│  │  │ 190 ┤ ╭──╯ ╰─╯                                     ╰───│    │    │
│  │  │     └─────────────────────────────────────────────────────╯    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─ GPU Status ───────────────────────────────────────────────────────┐    │
│  │  GPU 0 (RTX 4090)     124.5 MH/s  ▓▓▓▓▓▓▓▓▓▓▓▓░ 92%  72°C  350W   │    │
│  │  GPU 1 (RTX 4090)     120.8 MH/s  ▓▓▓▓▓▓▓▓▓▓▓░░ 88%  69°C  340W   │    │
│  │  CPU (16 cores)         12.3 kH/s ▓▓░░░░░░░░░░ 15%  45°C   85W   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─ Mining Statistics ────────────────────────────────────────────────┐    │
│  │  Blocks Mined:         47           Dev Fee:      3.0%             │    │
│  │  Total QNK Earned:     94.0 QNK     Shares:       1,247           │    │
│  │  Quantum Utilization:  73.4%        Efficiency:   0.0021%         │    │
│  │  Uptime:               4h 32m       Last Block:   2m 18s ago      │    │
│  │  Current Difficulty:   18,432,547   Network:      48.2 TH/s       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─ Recent Blocks ────────────────────────────────────────────────────┐    │
│  │  #12,847  1m ago   2.00 QNK  0xf4a2...3b1c  145.2 MH/s  GPU 0     │    │
│  │  #12,825  8m ago   2.00 QNK  0x8bc1...7e9a  138.7 MH/s  GPU 1     │    │
│  │  #12,803 15m ago   2.00 QNK  0x2d9f...4c6b  141.3 MH/s  GPU 0     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─ Network Info ─────────────────────────────────────────────────────┐    │
│  │  Pool: solo-mining   Height: 12,847   Phase: 8   Peers: 24        │    │
│  │  Node: 185.182.185.227:8080   Status: ✓ Connected                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ [q] Quit  [p] Pause  [g] GPU Settings  [l] Logs  [h] Help                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Multi-GPU Architecture (Inspired by mistral.rs)

```rust
// GPU Device Mapping Strategy
pub struct MultiGPUMiner {
    // GPU devices mapped to mining tasks
    device_mapper: LayerDeviceMapper,

    // Individual GPU miners
    gpu_miners: Vec<GPUMiner>,

    // Load balancer for distributing work
    load_balancer: GPULoadBalancer,

    // Metrics collector
    metrics: Arc<RwLock<MultiGPUMetrics>>,
}

// Similar to mistral.rs DeviceMapMetadata
pub struct GPUDeviceMapMetadata {
    pub gpu_id: usize,
    pub hash_capacity: f64,  // Hashes per second
    pub memory_gb: f64,
    pub compute_capability: f64,
}

// GPU-specific miner (like mistral.rs per-device execution)
pub struct GPUMiner {
    device_id: usize,
    device: Device,
    context: OpenCLContext,
    kernel: SHA3Kernel,
    stats: Arc<RwLock<GPUStats>>,
}
```

### 3. Key Features

#### TUI Features:
- **Real-time hashrate graph** with sparklines
- **Per-GPU monitoring** (temperature, power, utilization)
- **Block discovery notifications** with sound/animation
- **Live logs** with color coding
- **Interactive GPU configuration**
- **Quantum enhancement visualization**
- **Network statistics** and peer info
- **Keyboard shortcuts** for all actions

#### Multi-GPU Features:
- **Automatic GPU detection** (CUDA, OpenCL, Metal)
- **Dynamic load balancing** across GPUs
- **Per-GPU performance tuning**
- **Failover support** (if GPU crashes, redistribute)
- **Memory-aware work distribution**
- **Temperature-based throttling**
- **Mixed GPU support** (different models/vendors)

### 4. Implementation Strategy

#### Phase 1: Core TUI Framework
```rust
// crates/q-miner/src/tui/mod.rs
pub mod app;      // Main TUI application state
pub mod ui;       // Rendering logic
pub mod events;   // Keyboard/mouse event handling
pub mod charts;   // Hashrate graphs and visualizations
pub mod widgets;  // Custom widgets (GPU bars, etc.)
```

#### Phase 2: Multi-GPU Mining
```rust
// crates/q-miner/src/gpu/multi_gpu.rs
pub struct MultiGPUCoordinator {
    // GPU discovery and initialization
    pub fn detect_gpus() -> Vec<GPUDeviceInfo>;

    // Create device mapper (like mistral.rs)
    pub fn create_device_mapper(gpus: &[GPUDeviceInfo]) -> DeviceMapper;

    // Distribute mining work
    pub fn distribute_work(&mut self, template: MiningTemplate);

    // Collect results from all GPUs
    pub fn collect_results(&mut self) -> Vec<MiningResult>;
}
```

#### Phase 3: Integration
- Connect TUI to mining engine
- Real-time metrics streaming
- GPU metrics polling
- Event-driven UI updates

### 5. Technical Details

#### TUI Stack:
- **ratatui**: Modern terminal UI framework
- **crossterm**: Cross-platform terminal manipulation
- **tui-textarea**: Text input widgets
- **unicode-width**: Proper text rendering

#### GPU Stack (from mistral.rs):
- **candle-core**: GPU tensor operations
- **cudarc**: CUDA bindings (NVIDIA)
- **opencl3**: OpenCL bindings (cross-vendor)
- **metal**: Metal bindings (Apple)

#### Data Flow:
```
Mining Engine → Metrics Channel → TUI Update Loop
     ↓                                    ↑
Multi-GPU Coordinator ←────────────── User Input
     ↓
GPU Miners (parallel execution)
```

### 6. Performance Targets

- **UI Updates**: 10 FPS (100ms refresh)
- **Metrics Collection**: Every 500ms
- **GPU Load Balancing**: Dynamic, every 5s
- **Memory Overhead**: <50MB for TUI
- **GPU Efficiency**: >95% utilization per GPU

### 7. Example GPU Mapping

```rust
// Automatically detect and map GPUs
let gpu_config = MultiGPUConfig {
    devices: vec![
        GPUDeviceMapMetadata {
            gpu_id: 0,
            hash_capacity: 120_000_000.0,  // 120 MH/s
            memory_gb: 24.0,
            compute_capability: 8.9,
        },
        GPUDeviceMapMetadata {
            gpu_id: 1,
            hash_capacity: 118_000_000.0,  // 118 MH/s
            memory_gb: 24.0,
            compute_capability: 8.9,
        },
    ],
    // Auto-distribute work based on capacity
    load_distribution: LoadDistribution::Auto,
};

// Create multi-GPU miner with device mapper
let miner = MultiGPUMiner::new(gpu_config)?;
```

### 8. Next Steps

1. Implement basic TUI framework with ratatui
2. Add GPU detection and device mapping (mistral.rs style)
3. Implement multi-GPU work distribution
4. Add real-time metrics collection
5. Create interactive GPU configuration UI
6. Add advanced features (graphs, animations, etc.)
7. Performance optimization and testing

## References

- mistral.rs DeviceMapper: `/root/.cargo/git/checkouts/mistral.rs-d7a5d833e16ad691/bc0384b/mistralrs-core/src/device_map.rs`
- Current miner: `crates/q-mining/src/miner.rs`
- Existing TUI: `crates/q-tui/` (for reference)
