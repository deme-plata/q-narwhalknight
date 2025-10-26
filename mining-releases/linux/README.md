# Q-NarwhalKnight Miner v1.1.0 - Linux (Optimized)

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

## Quick Start

### 1. Make Executable
```bash
chmod +x q-miner
```

### 2. Start Mining (Easy Way)
```bash
./start-mining.sh YOUR_WALLET_ADDRESS
```

### 3. Start Mining (Manual)
```bash
./q-miner --mode solo \
  --server http://185.182.185.227:8080/ \
  --threads 4 \
  --wallet YOUR_WALLET_ADDRESS \
  --intensity 10
```

## Command Line Options

```
--mode <MODE>              Mining mode: solo, pool, benchmark [default: benchmark]
--wallet <ADDRESS>         Your QNK wallet address (required for solo/pool)
--threads <N>              Number of CPU threads (0 = auto-detect) [default: 0]
--intensity <1-10>         Mining intensity (10 = maximum) [default: 7]
--server <URL>             API server URL [default: http://localhost:8080]
--gpu                      Enable GPU mining (experimental)
--benchmark                Run performance benchmark
--duration <SECS>          Benchmark duration in seconds [default: 30]
```

## Examples

### Solo Mining (Recommended)
```bash
# Auto-detect CPU cores, maximum intensity
./q-miner --mode solo \
  --server http://185.182.185.227:8080/ \
  --wallet qnk1234567890abcdef... \
  --intensity 10

# Use specific number of threads
./q-miner --mode solo \
  --server http://185.182.185.227:8080/ \
  --wallet qnk1234567890abcdef... \
  --threads 8 \
  --intensity 10
```

### Benchmark Mode
```bash
# Test your mining performance
./q-miner --benchmark --threads 4 --duration 60
```

## Performance Tips

### 🔥 Maximum Hash Rate
- Use `--intensity 10` for 99% CPU utilization
- Set `--threads` to your CPU's physical core count
- Close other applications while mining

### 💻 Balanced Performance
- Use `--intensity 7` for 70-80% CPU usage
- Leave some threads for system (e.g., if you have 8 cores, use `--threads 6`)

### ⚡ Multi-Core CPUs
```bash
# AMD Ryzen 9 5950X (16 cores)
./q-miner --threads 16 --intensity 10

# Intel Core i9-12900K (16 cores)
./q-miner --threads 16 --intensity 10

# Raspberry Pi (4 cores)
./q-miner --threads 4 --intensity 8
```

## Mining Rewards

- **Block Reward:** 0.5 QNK per valid solution
- **Average Time:** 5-10 solutions per minute (depends on hardware)
- **Network:** Testnet (rewards are real QNK tokens)

## System Requirements

- **OS:** Linux (Ubuntu 20.04+, Debian 11+, CentOS 8+, etc.)
- **CPU:** x86_64 with AVX2 support (recommended)
- **RAM:** 1 GB minimum
- **Network:** Stable internet connection

## Troubleshooting

### Connection Refused
```
Error: Connection refused (os error 111)
```
**Solution:** Make sure the API server is running at the specified URL.

### Low Hash Rate
- Increase `--intensity` to 10
- Make sure no other CPU-intensive processes are running
- Check CPU governor is set to "performance" mode:
  ```bash
  sudo cpupower frequency-set -g performance
  ```

### No Solutions Found
This is normal! Mining is probabilistic. Keep running and solutions will come.

## Monitoring

### Real-Time Hash Rate
The miner displays hash rate every 5 seconds:
```
📊 Hash Rate: 166023.46 H/s (166.02 KH/s) - Total: 3212455
```

### Solution Notifications
```
💎 Thread 0 found solution! Block #123, Nonce: 1234567
✅ Solution accepted! Earned 0.5 QNK
```

## Support

- **Discord:** [discord.gg/qnarwhalknight](https://discord.gg/qnarwhalknight)
- **Telegram:** [@qnarwhalknight](https://t.me/qnarwhalknight)
- **Email:** support@quillonq.xyz
- **GitHub:** [github.com/deme-plata/q-narwhalknight](https://github.com/deme-plata/q-narwhalknight)

## Version History

### v1.1.0 (2025-10-25) - Performance Update
- ✨ 15% hash rate improvement
- 🔥 99% CPU utilization (removed throttling)
- ⚡ Zero-allocation VDF algorithm
- 💪 10x increased batch sizes

### v1.0.0 (2025-10-20) - Initial Release
- Basic CPU mining support
- Solo and pool mining modes
- Real-time SSE rewards

---

**Happy Mining! ⛏️💎**
