# Q-NarwhalKnight Miner v1.1.0 - Windows (Optimized)

🚀 **Maximum Performance CPU Miner for Q-NarwhalKnight Blockchain**

## Performance Improvements

✨ **NEW in v1.1.0:**
- 🔥 **15% Higher Hash Rate** - Zero-allocation mining algorithm
- ⚡ **99% CPU Utilization** - Removed throttling for maximum performance
- 💪 **10x Larger Batches** - Fewer context switches, more hashing
- 🎯 **Optimized VDF** - In-place computation with cache efficiency

**Hash Rate:**
- Previous: ~144 KH/s
- **New: ~166 KH/s** (+15% improvement)

## Quick Start (Pre-built Binary Included!)

### 1. Extract the Package
Simply extract the zip file - the optimized `q-miner.exe` v1.1.0 is ready to use!

### 2. Start Mining (Easy Way)
Open PowerShell in the extracted folder and run:
```powershell
.\start-mining.ps1 YOUR_WALLET_ADDRESS
```

The script will auto-detect your CPU and start mining at maximum performance!

### 3. Start Mining (Manual)
```powershell
.\q-miner.exe --mode solo --server http://185.182.185.227:8080/ --wallet YOUR_WALLET_ADDRESS --intensity 10
```

## Windows Build Instructions (Optional)

Due to cross-compilation complexity, we recommend building the miner on your Windows machine directly.

### Option 1: Build from Source (Recommended)

#### Prerequisites
1. Install [Rust](https://www.rust-lang.org/tools/install) for Windows
2. Install [Git for Windows](https://git-scm.com/download/win)
3. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)

#### Build Steps
```powershell
# Clone the repository
git clone https://github.com/deme-plata/q-narwhalknight
cd q-narwhalknight

# Build the miner (this may take 5-10 minutes)
cargo build --release --package q-miner --bin q-miner

# The miner will be at: target\release\q-miner.exe
```

#### Start Mining
```powershell
.\target\release\q-miner.exe --mode solo --server http://185.182.185.227:8080/ --wallet YOUR_WALLET_ADDRESS --intensity 10
```

### Option 2: Use WSL2 (Windows Subsystem for Linux)

1. Enable WSL2: https://learn.microsoft.com/en-us/windows/wsl/install
2. Download the Linux package: `q-narwhalknight-miner-v1.1.0-linux-x86_64.tar.gz`
3. Extract and run in WSL2:

```bash
tar -xzf q-narwhalknight-miner-v1.1.0-linux-x86_64.tar.gz
cd linux
./start-mining.sh YOUR_WALLET_ADDRESS
```

## Command Line Options

```
--mode <MODE>              Mining mode: solo, pool, benchmark [default: benchmark]
--wallet <ADDRESS>         Your QNK wallet address (required for solo/pool)
--threads <N>              Number of CPU threads (0 = auto-detect) [default: 0]
--intensity <1-10>         Mining intensity (10 = maximum) [default: 7]
--server <URL>             API server URL [default: http://localhost:8080]
--benchmark                Run performance benchmark
```

## Performance Tips

### Maximum Performance
```powershell
q-miner.exe --mode solo --server http://185.182.185.227:8080/ --wallet YOUR_WALLET --intensity 10 --threads 0
```

### Balanced (Leave CPU for other tasks)
```powershell
q-miner.exe --mode solo --server http://185.182.185.227:8080/ --wallet YOUR_WALLET --intensity 7 --threads 6
```

## Troubleshooting

### Build Errors
- Make sure Visual Studio Build Tools are installed
- Restart your terminal after installing Rust
- Run `rustup update` to get the latest toolchain

### Antivirus Warnings
Some antivirus software may flag cryptocurrency miners. Add an exception for `q-miner.exe` if needed.

### Low Hash Rate
- Close other applications
- Use `--intensity 10` for maximum performance
- Ensure your CPU supports AVX2 for optimal performance

## Support

- **Discord:** [discord.gg/qnarwhalknight](https://discord.gg/qnarwhalknight)
- **GitHub:** [github.com/deme-plata/q-narwhalknight](https://github.com/deme-plata/q-narwhalknight)
- **Email:** support@quillonq.xyz

---

**Pre-built Windows binary coming soon!**
