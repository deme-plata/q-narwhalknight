# Q-Miner TUI Quick Start Guide

## What We Built

A **beautiful terminal UI** for Q-Miner with **multi-GPU support** inspired by mistral.rs GPU compute architecture!

### Features:
- 🎨 Real-time hashrate graph (sparklines)
- 📊 Per-GPU monitoring (temp, power, utilization)
- ⚡ Multi-GPU coordination (2, 4, 8+ GPUs)
- 🔄 Smart load balancing (Equal, Capacity-Based, Dynamic)
- ⌨️  Interactive controls (pause, GPU selection, help)
- 📈 Mining statistics (shares, efficiency, uptime)

## Step-by-Step Build Process

###  1. Install Dependencies (if needed)

```bash
# For CUDA support (NVIDIA GPUs)
sudo apt-get install nvidia-cuda-toolkit

# For OpenCL support (AMD/Intel/NVIDIA GPUs)
sudo apt-get install ocl-icd-opencl-dev
```

### 2. Build the Miner

The compilation can take 5-15 minutes depending on features enabled.

**Option A: TUI Only (CPU Mining)**
```bash
cd /opt/orobit/shared/q-narwhalknight

timeout 600 cargo build --release --package q-miner --features "tui"
```

**Option B: TUI + CUDA (NVIDIA GPUs)**
```bash
timeout 600 cargo build --release --package q-miner --features "tui,cuda-mining"
```

**Option C: TUI + OpenCL (AMD/Intel GPUs)**
```bash
timeout 600 cargo build --release --package q-miner --features "tui,opencl-mining"
```

**Option D: TUI + All GPU Support**
```bash
timeout 600 cargo build --release --package q-miner --features "tui,cuda-mining,opencl-mining"
```

### 3. Run the Miner

After building successfully:

```bash
# Run with TUI (auto-detect GPUs)
./target/release/q-miner --tui

# Run with specific GPUs
./target/release/q-miner --tui --gpus=0,1

# Run with solo mining
./target/release/q-miner --tui --pool=solo --node=http://185.182.185.227:8080
```

## TUI Interface Overview

```
┌────────────────────────────────────────────────────────────────┐
│ Q-Miner v1.0.0 - Quantum-Enhanced Mining Dashboard  ⛏️  MINING │
├────────────────────────────────────────────────────────────────┤
│ [Overview] [GPU Details] [Events] [Settings]                  │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─ Hashrate Performance ───────────────────────────────┐     │
│  │  Current:  245.3 MH/s  ▓▓▓▓▓▓▓▓▓▓▓▓░░ 87%           │     │
│  │  Average:  238.7 MH/s                                │     │
│  │  Peak:     267.1 MH/s  (4m ago)                      │     │
│  │  ╭───────────────────────────────────────────────╮   │     │
│  │  │ 270 ┤                    ╭─╮                  │   │     │
│  │  │ 250 ┤         ╭─╮  ╭─╮ ╭─╯ ╰─╮               │   │     │
│  │  │ 230 ┤    ╭─╮ ╭╯ ╰──╯ ╰─╯     ╰─╮             │   │     │
│  │  └──────────────────────────────────────────────────┘     │
│                                                                 │
│  ┌─ GPU Status ──────────────────────────────────────────┐    │
│  │  GPU 0 (RTX 4090)  124.5 MH/s  ▓▓▓▓▓▓▓▓▓▓▓▓░ 92%    │    │
│  │                    72°C  350W                          │    │
│  │                                                         │    │
│  │  GPU 1 (RTX 4090)  120.8 MH/s  ▓▓▓▓▓▓▓▓▓▓▓░░ 88%   │    │
│  │                    69°C  340W                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─ Mining Statistics ──────────────────────────────────┐     │
│  │  Shares Accepted: 47     Efficiency:  96.0%          │     │
│  │  Shares Rejected: 2      Power Usage: 690W           │     │
│  │  Uptime: 4h 32m 18s      Last Block:  2m 18s ago    │     │
│  └───────────────────────────────────────────────────────┘     │
│                                                                 │
├────────────────────────────────────────────────────────────────┤
│ [q] Quit  [p] Pause  [Tab] Next  [h] Help                     │
└────────────────────────────────────────────────────────────────┘
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` or `Esc` | Quit application |
| `p` | Pause/Resume mining |
| `h` or `?` | Show/Hide help overlay |
| `Tab` or `→` | Next tab |
| `←` | Previous tab |
| `↑` / `↓` | Select GPU (in GPU Details) |

## Multi-GPU Features

### Auto-Detection
The miner automatically detects all compatible GPUs:
```
🔍 Auto-detecting GPUs for mining...
✅ Detected 2 CUDA device(s)
📊 Total GPUs detected: 2
  GPU 0: NVIDIA GeForce RTX 4090 (24 GB, est. 120.00 MH/s)
  GPU 1: NVIDIA GeForce RTX 4090 (24 GB, est. 120.00 MH/s)
```

### Load Balancing Strategies

**1. Equal Distribution** (default for same GPUs)
```
GPU 0: 50% of work → 120 MH/s
GPU 1: 50% of work → 120 MH/s
Total: 240 MH/s
```

**2. Capacity-Based** (for mixed GPUs)
```
GPU 0 (RTX 4090): 66.7% of work → 120 MH/s
GPU 1 (RTX 3080): 33.3% of work →  60 MH/s
Total: 180 MH/s
```

**3. Dynamic** (adjusts based on real performance)
```
Auto-adjusts every 5 seconds based on:
- Actual hash rate achieved
- GPU temperature
- Power consumption
- Error rate
```

## Troubleshooting

### Build Fails

**Problem:** Compilation timeout or errors

**Solution:**
```bash
# 1. Clean build cache
cargo clean

# 2. Update dependencies
cargo update

# 3. Try with longer timeout
timeout 36000 cargo build --release --package q-miner --features "tui"

# 4. Check for specific errors
cargo check --package q-miner --features "tui" 2>&1 | grep error
```

### No GPUs Detected

**Problem:** TUI shows "No compatible GPU devices detected"

**Solution:**
```bash
# Check CUDA devices
nvidia-smi

# Check OpenCL devices
clinfo

# Rebuild with correct features
cargo build --release --package q-miner --features "tui,cuda-mining"
```

### TUI Not Displaying Correctly

**Problem:** Garbled or missing graphics

**Solution:**
```bash
# Set correct terminal type
export TERM=xterm-256color

# Or use a better terminal emulator
# - Alacritty
# - Kitty
# - iTerm2 (Mac)
```

### Low Hash Rate

**Possible Causes:**
1. **Thermal throttling** - GPU too hot (>85°C)
2. **Power limit** - GPU not getting enough power
3. **Low intensity** - Try `--intensity 9`
4. **CPU bottleneck** - Use fewer CPU threads

**Solutions:**
```bash
# Check GPU temperature in TUI (should be <80°C)

# Increase power limit (NVIDIA)
sudo nvidia-smi -i 0 -pl 350  # Set 350W limit

# Adjust mining intensity
./target/release/q-miner --tui --intensity 9
```

## Performance Tuning

### Optimal Settings by GPU

**RTX 4090:**
```bash
./target/release/q-miner --tui \
    --intensity 8 \
    --power-limit 450 \
    --temp-limit 75
```

**RTX 3080:**
```bash
./target/release/q-miner --tui \
    --intensity 7 \
    --power-limit 320 \
    --temp-limit 70
```

**AMD RX 7900 XT:**
```bash
./target/release/q-miner --tui \
    --intensity 7 \
    --opencl-platform 0 \
    --temp-limit 75
```

## Advanced Usage

### Configuration File

Create `~/.config/q-miner/config.toml`:
```toml
[mining]
intensity = 8
pool = "solo"
node_url = "http://185.182.185.227:8080"

[tui]
refresh_rate = 4  # FPS
show_graphs = true
color_scheme = "default"

[[gpus]]
id = 0
enabled = true
power_limit = 350
temp_limit = 75
load_percentage = 100

[[gpus]]
id = 1
enabled = true
power_limit = 320
temp_limit = 70
load_percentage = 100
```

### Monitoring Remotely

```bash
# Run in tmux/screen for persistence
tmux new -s miner
./target/release/q-miner --tui

# Detach: Ctrl+b, d
# Reattach: tmux attach -t miner
```

### Logging

```bash
# Enable debug logging
RUST_LOG=debug ./target/release/q-miner --tui 2>&1 | tee miner.log

# Filter logs
RUST_LOG=q_miner=info ./target/release/q-miner --tui
```

## What's Next?

After getting mining working:

1. **Optimize Settings**: Tune intensity, power limits for best efficiency
2. **Join Pool**: Connect to a mining pool for consistent rewards
3. **Monitor Performance**: Watch TUI for temperature and hash rate
4. **Scale Up**: Add more GPUs as needed
5. **Automate**: Set up systemd service for automatic start

## Files Created

- `crates/q-miner/src/gpu/multi_gpu.rs` - Multi-GPU coordinator (559 lines)
- `crates/q-miner/src/ui/tui_app.rs` - Terminal UI (710 lines)
- `MINER_TUI_AND_MULTI_GPU_DESIGN.md` - Design document
- `MINER_TUI_MULTI_GPU_IMPLEMENTATION.md` - Implementation details
- `MINER_BUILD_GUIDE.md` - Complete build guide
- `MINER_TUI_QUICKSTART.md` - This file

## Support

Need help?
- Check logs: `RUST_LOG=debug ./target/release/q-miner --tui`
- Discord: Q-NarwhalKnight community
- GitHub: Open an issue

Happy mining! ⛏️🚀
