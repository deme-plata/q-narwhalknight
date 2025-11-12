# Q-Miner Build Guide

## Quick Start

### Building the Miner with TUI

```bash
# 1. Build with TUI interface only (no GPU support) - fastest
timeout 600 cargo build --release --package q-miner --features "tui"

# 2. Build with TUI + CUDA support (NVIDIA GPUs)
timeout 600 cargo build --release --package q-miner --features "tui,cuda-mining"

# 3. Build with TUI + OpenCL support (AMD/Intel/NVIDIA GPUs)
timeout 600 cargo build --release --package q-miner --features "tui,opencl-mining"

# 4. Build with TUI + Both CUDA and OpenCL
timeout 600 cargo build --release --package q-miner --features "tui,cuda-mining,opencl-mining"
```

## Feature Flags Explained

### Core Features:
- `cpu-mining` - CPU mining support (included in `default`)
- `tui` - Terminal UI interface with ratatui
- `cuda-mining` - NVIDIA GPU support via CUDA
- `opencl-mining` - Cross-vendor GPU support (AMD, Intel, NVIDIA)
- `vulkan-mining` - Vulkan compute support
- `gui` - Desktop GUI with egui/eframe
- `network` - Network connectivity for pool mining
- `jemalloc` - Better memory allocator (included in `default`)

### Build Combinations:

```bash
# CPU mining only with TUI
cargo build --release --package q-miner --features "tui"

# NVIDIA GPUs with TUI
cargo build --release --package q-miner --features "tui,cuda-mining"

# AMD GPUs with TUI
cargo build --release --package q-miner --features "tui,opencl-mining"

# Multi-vendor GPUs with TUI
cargo build --release --package q-miner --features "tui,cuda-mining,opencl-mining"

# Everything enabled
cargo build --release --package q-miner --all-features
```

## Compilation Issues & Fixes

### Issue: Missing CUDA Toolkit

**Error:**
```
Could not find CUDA toolkit
```

**Fix:**
```bash
# Install NVIDIA CUDA Toolkit
# Ubuntu/Debian:
sudo apt-get install nvidia-cuda-toolkit

# Or download from: https://developer.nvidia.com/cuda-downloads
```

### Issue: Missing OpenCL Headers

**Error:**
```
Could not find OpenCL headers
```

**Fix:**
```bash
# Ubuntu/Debian:
sudo apt-get install ocl-icd-opencl-dev

# Fedora/RHEL:
sudo dnf install ocl-icd-devel
```

### Issue: Build Timeout

**Error:**
```
Command timed out after 120 seconds
```

**Fix:**
Use longer timeout (10 hours for complex builds):
```bash
timeout 36000 cargo build --release --package q-miner --features "tui,cuda-mining"
```

## Running the Miner

### Basic Usage

```bash
# Run with TUI (auto-detect GPUs)
./target/release/q-miner --tui

# Run with TUI and specific GPUs
./target/release/q-miner --tui --gpus=0,1

# Run with TUI and solo mining
./target/release/q-miner --tui --pool=solo --node=http://185.182.185.227:8080

# Run with TUI and pool mining
./target/release/q-miner --tui --pool=http://pool.quillon.xyz:3333
```

### TUI Keyboard Controls

Once the TUI is running:

- **q** or **Esc** - Quit application
- **p** - Pause/Resume mining
- **h** or **?** - Show help overlay
- **Tab** or **→** - Next tab
- **←** - Previous tab
- **↑/↓** - Select GPU (in GPU Details tab)

### TUI Tabs

1. **Overview** - Hash rate graph, GPU status, mining stats
2. **GPU Details** - Detailed per-GPU metrics and controls
3. **Events** - Recent mining events (blocks found, shares, etc.)
4. **Settings** - Configuration and preferences

## Troubleshooting

### TUI Not Displaying Correctly

```bash
# Make sure your terminal supports 256 colors
echo $TERM
# Should be: xterm-256color or similar

# If not, set it:
export TERM=xterm-256color
```

### GPU Not Detected

```bash
# Check CUDA devices
nvidia-smi

# Check OpenCL devices
clinfo

# Run with debug logging
RUST_LOG=debug ./target/release/q-miner --tui
```

### Low Hash Rate

1. Check GPU temperature (thermal throttling)
2. Increase mining intensity: `--intensity 9`
3. Adjust GPU settings in TUI (press 'g')
4. Check GPU power limit: `nvidia-smi -i 0 -pl 350` (set to 350W)

## Build Times (Approximate)

| Configuration | Build Time | Binary Size |
|--------------|------------|-------------|
| TUI only | ~5 minutes | ~15 MB |
| TUI + CUDA | ~8 minutes | ~25 MB |
| TUI + OpenCL | ~7 minutes | ~20 MB |
| TUI + CUDA + OpenCL | ~10 minutes | ~30 MB |
| All features | ~15 minutes | ~50 MB |

## Performance Expectations

### Hash Rates (Estimated):

| GPU Model | Expected Hash Rate |
|-----------|-------------------|
| RTX 4090 | 120-130 MH/s |
| RTX 4080 | 100-110 MH/s |
| RTX 3090 | 110-120 MH/s |
| RTX 3080 | 90-100 MH/s |
| RX 7900 XT | 80-95 MH/s |
| RX 6900 XT | 70-85 MH/s |

### Multi-GPU Scaling:

- 2 GPUs: ~190% of single GPU
- 4 GPUs: ~380% of single GPU
- 8 GPUs: ~750% of single GPU

(Some overhead from work distribution and synchronization)

## Advanced Configuration

### Custom GPU Settings

Create `~/.config/q-miner/config.toml`:

```toml
[mining]
intensity = 8
pool = "solo"

[[gpus]]
id = 0
power_limit = 350  # Watts
temp_limit = 75    # Celsius
core_clock = 1800  # MHz
memory_clock = 10000 # MHz

[[gpus]]
id = 1
power_limit = 320
temp_limit = 70
```

### Environment Variables

```bash
# Database path
export Q_DB_PATH=./miner-data

# Network
export Q_NETWORK_ID=testnet-phase9

# Logging
export RUST_LOG=info

# TUI refresh rate (FPS)
export Q_MINER_TUI_FPS=4
```

## Next Steps

After building successfully:

1. Test with CPU mining first: `./target/release/q-miner --tui --cpu-threads 4`
2. Test GPU detection: Check GPU tab in TUI
3. Start actual mining: Connect to pool or solo mine
4. Monitor performance: Watch hash rate graph and GPU metrics
5. Optimize: Adjust intensity and GPU settings for best performance

## Support

For issues:
- Check logs: `RUST_LOG=debug ./target/release/q-miner --tui 2>&1 | tee miner.log`
- Report bugs: https://github.com/deme-plata/q-narwhalknight/issues
- Discord: Join Q-NarwhalKnight community

## Example Session

```bash
# 1. Build
timeout 600 cargo build --release --package q-miner --features "tui,cuda-mining"

# 2. Run
./target/release/q-miner --tui --gpus=0,1 --pool=solo

# 3. In TUI:
# - Press Tab to view GPU details
# - Press 'p' to pause if needed
# - Press 'h' for help
# - Monitor hash rate in real-time

# 4. Stop gracefully
# - Press 'q' to quit
```

Enjoy mining with Q-NarwhalKnight! ⛏️
