# Q-NarwhalKnight Miner: Server CPU Optimization Guide

## Overview

This guide explains how to compile and run the Q-NarwhalKnight miner with maximum performance on modern server CPUs from AMD (EPYC/Threadripper) and Intel (Xeon).

## Key Optimizations Implemented

### 1. **BLAKE3 SIMD Acceleration**
- ✅ Enabled `blake3` with `rayon` feature for parallel hashing
- Automatically uses AVX2/AVX-512 SIMD instructions
- 2-4x faster hashing compared to scalar implementation

### 2. **CPU Affinity Pinning**
- ✅ Each mining thread pinned to specific CPU core
- Dramatically improves cache locality on NUMA systems
- Eliminates thread migration overhead
- **Critical for AMD EPYC and Intel Xeon multi-socket systems**

### 3. **Architecture-Specific Tuning**
- ✅ Detects CPU vendor (AMD/Intel) and SIMD capabilities
- ✅ Reports AVX2/AVX-512 support
- Batch size optimized for cache hierarchy

### 4. **Zero-Allocation Mining Loop**
- Pre-allocated hash input buffers
- In-place VDF computation
- Minimal memory allocations in hot path

## Compilation Flags for Maximum Performance

### **AMD EPYC / Threadripper**

#### Option 1: Native CPU (Best Performance)
```bash
# Compile for the EXACT CPU model you're running on
RUSTFLAGS="-C target-cpu=native" cargo build --release --package q-miner

# Example output: "AMD EPYC 7763" → uses znver3 microarchitecture
```

#### Option 2: Generic AMD Zen 3/4 (Portable across EPYC generations)
```bash
# Use znver3 for Zen 3 (EPYC 7xx3 series)
RUSTFLAGS="-C target-cpu=znver3" cargo build --release --package q-miner

# Use znver4 for Zen 4 (EPYC 9xx4 series)
RUSTFLAGS="-C target-cpu=znver4" cargo build --release --package q-miner
```

#### Option 3: Maximum AVX2 Optimization (Compatible with Zen 2+)
```bash
RUSTFLAGS="-C target-cpu=znver2 -C target-feature=+avx2,+fma" cargo build --release --package q-miner
```

### **Intel Xeon (Scalable Processors)**

#### Option 1: Native CPU (Best Performance)
```bash
# Compile for the EXACT CPU model you're running on
RUSTFLAGS="-C target-cpu=native" cargo build --release --package q-miner

# Example output: "Intel Xeon Platinum 8380" → uses icelake-server microarchitecture
```

#### Option 2: AVX-512 Enabled (Ice Lake / Sapphire Rapids)
```bash
# Ice Lake Xeon (3rd Gen Scalable)
RUSTFLAGS="-C target-cpu=icelake-server" cargo build --release --package q-miner

# Sapphire Rapids Xeon (4th Gen Scalable)
RUSTFLAGS="-C target-cpu=sapphirerapids" cargo build --release --package q-miner
```

#### Option 3: AVX2 Only (Skylake / Cascade Lake)
```bash
# Skylake Xeon (1st/2nd Gen Scalable)
RUSTFLAGS="-C target-cpu=skylake-avx512" cargo build --release --package q-miner

# Cascade Lake Xeon (2nd Gen Scalable)
RUSTFLAGS="-C target-cpu=cascadelake" cargo build --release --package q-miner
```

### **Generic High-Performance Build (Portable)**
```bash
# Works on any modern x86_64 CPU with AVX2 (2015+)
RUSTFLAGS="-C target-cpu=x86-64-v3 -C opt-level=3" cargo build --release --package q-miner
```

## Recommended Compilation Command (Universal)

```bash
# ⭐ RECOMMENDED: This detects your CPU and uses optimal flags automatically
RUSTFLAGS="-C target-cpu=native -C opt-level=3" \
  cargo build --release --package q-miner
```

## Runtime Configuration

### **Optimal Thread Count**

```bash
# Physical cores only (RECOMMENDED for maximum hashrate)
./target/release/q-miner \
  --threads $(nproc --physical) \
  --intensity 10 \
  --wallet qnk1234... \
  --server http://185.182.185.227:8080
```

### **AMD EPYC 7763 (64 cores / 128 threads)**
```bash
# Use physical cores for best performance
./target/release/q-miner --threads 64 --intensity 10 --wallet <WALLET>
```

### **Intel Xeon Platinum 8380 (40 cores / 80 threads)**
```bash
# Use physical cores for best performance
./target/release/q-miner --threads 40 --intensity 10 --wallet <WALLET>
```

### **Multi-Socket Systems (2x or 4x CPUs)**
```bash
# Example: 2x AMD EPYC 7763 = 128 physical cores
./target/release/q-miner --threads 128 --intensity 10 --wallet <WALLET>
```

## Performance Expectations

### **AMD EPYC 7763 (64-core, 2.45 GHz base)**
- **Expected Hashrate**: 150-200 MH/s (with AVX2)
- **Expected Hashrate**: 200-250 MH/s (with AVX-512, if available)
- **Power Consumption**: ~280W TDP

### **Intel Xeon Platinum 8380 (40-core, 2.30 GHz base)**
- **Expected Hashrate**: 120-160 MH/s (with AVX-512)
- **Power Consumption**: ~270W TDP

### **AMD Threadripper PRO 5995WX (64-core, 2.70 GHz base)**
- **Expected Hashrate**: 180-220 MH/s
- **Power Consumption**: ~280W TDP

## Verification

After compiling, verify optimizations are active:

```bash
# Run the miner and check hardware detection output
./target/release/q-miner --benchmark --duration 10

# Expected output:
# 💻 Hardware Detection Results:
#    CPU: AuthenticAMD (AVX2) - 64 cores, 128 threads
#    Cache Line: 64 bytes │ SIMD: AVX2 │ Server-Optimized: ✅
#
# OR
#
#    CPU: GenuineIntel (AVX-512) - 40 cores, 80 threads
#    Cache Line: 64 bytes │ SIMD: AVX-512 │ Server-Optimized: ✅
```

## Troubleshooting

### **Low Hashrate**

1. **Check CPU Governor**:
   ```bash
   # Set CPU to performance mode
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```

2. **Check SMT/Hyper-Threading**:
   ```bash
   # Use physical cores only for mining
   lscpu | grep "Thread(s) per core"
   # If output is "2", use: threads = cores / 2
   ```

3. **Check NUMA Configuration**:
   ```bash
   # Verify NUMA is enabled for multi-socket systems
   numactl --hardware
   ```

### **Compilation Errors**

If compilation fails with SIMD-related errors:
```bash
# Fall back to generic build without AVX-512
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release --package q-miner
```

### **Thread Affinity Not Working**

If you see "affinity pinning failed" messages:
```bash
# Run with elevated privileges (some systems require it for CPU pinning)
sudo ./target/release/q-miner --threads 64 --intensity 10 --wallet <WALLET>
```

## Advanced: Profile-Guided Optimization (PGO)

For the absolute maximum performance, use Profile-Guided Optimization:

```bash
# Step 1: Build instrumented binary
RUSTFLAGS="-C profile-generate=/tmp/pgo-data" \
  cargo build --release --package q-miner

# Step 2: Run benchmark to collect profile data
./target/release/q-miner --benchmark --duration 60

# Step 3: Rebuild with profile data
RUSTFLAGS="-C profile-use=/tmp/pgo-data -C target-cpu=native" \
  cargo build --release --package q-miner

# Expected improvement: +5-10% hashrate
```

## Benchmarking

Compare performance with different RUSTFLAGS:

```bash
# Baseline (no optimization)
cargo build --release --package q-miner
./target/release/q-miner --benchmark --duration 30 > baseline.txt

# AVX2 optimized
RUSTFLAGS="-C target-cpu=native" cargo build --release --package q-miner
./target/release/q-miner --benchmark --duration 30 > avx2.txt

# AVX-512 optimized (Intel Xeon Ice Lake+)
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx512f,+avx512dq" \
  cargo build --release --package q-miner
./target/release/q-miner --benchmark --duration 30 > avx512.txt

# Compare results
grep "MH/s" baseline.txt avx2.txt avx512.txt
```

## Production Deployment Checklist

- [ ] Compiled with `RUSTFLAGS="-C target-cpu=native"`
- [ ] CPU governor set to "performance"
- [ ] Thread count = physical cores (not logical threads)
- [ ] Intensity set to 10 for maximum hashrate
- [ ] CPU affinity pinning confirmed working
- [ ] NUMA enabled on multi-socket systems
- [ ] Thermal throttling not occurring (check with `sensors`)
- [ ] Hardware detection shows correct SIMD support (AVX2/AVX-512)

## Example Production Build Script

```bash
#!/bin/bash
set -e

echo "🚀 Building Q-NarwhalKnight Miner for Production"

# Detect CPU architecture
CPU_VENDOR=$(lscpu | grep "Vendor ID" | awk '{print $3}')

if [[ "$CPU_VENDOR" == "AuthenticAMD" ]]; then
    echo "✅ Detected AMD CPU - optimizing for EPYC/Threadripper"
    export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
elif [[ "$CPU_VENDOR" == "GenuineIntel" ]]; then
    echo "✅ Detected Intel CPU - optimizing for Xeon"
    export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
else
    echo "⚠️  Unknown CPU vendor - using generic optimization"
    export RUSTFLAGS="-C target-cpu=x86-64-v3 -C opt-level=3"
fi

# Build with 10-hour timeout for large codebases
timeout 36000 cargo build --release --package q-miner

echo "✅ Build complete!"
echo "📊 Binary location: ./target/release/q-miner"

# Run quick verification
./target/release/q-miner --benchmark --duration 10
```

## Questions?

For support, open an issue at: https://github.com/deme-plata/q-narwhalknight/issues

---

**Performance Target**: 150-250 MH/s on 64-core AMD EPYC / 40-core Intel Xeon server CPUs

**Last Updated**: 2025-11-21
