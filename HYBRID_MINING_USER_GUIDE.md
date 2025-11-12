# 🚀 Hybrid CPU+GPU Mining - Simple Explanation

## **TL;DR (2 Lines)**

**CPU miners solve VDF proofs (memory-based), GPU miners solve SHA-3 hashes (compute-based).**
**Both are required for a valid block and both earn 50% of the reward - keeping CPU mining profitable!**

---

## 🎯 Why Hybrid Mining?

### The Problem:
- Pure GPU mining → CPUs become worthless
- Only rich miners with GPU farms profit
- Network becomes centralized

### The Solution:
- **CPUs mine what they're good at**: VDF proofs (memory-bound)
- **GPUs mine what they're good at**: SHA-3 hashing (compute-bound)
- **Both get 50% of block rewards**: Always profitable!

---

## 💰 Reward Split

```
Total Block Reward: 2.0 QNK

CPU Miner gets: 1.0 QNK (50%)
GPU Miner gets: 1.0 QNK (50%)
```

### Example:

```
Block #12,847:
├─ VDF Proof by AMD EPYC 9654 (CPU)  → 1.0 QNK
└─ SHA-3 Hash by RTX 5090 (GPU)      → 1.0 QNK

Both miners earn equally regardless of hashrate!
```

---

## 🔧 How It Works

### Step 1: CPU Miner Creates VDF Proof
```
CPU miner (any CPU):
└─> Solves VDF proof (takes ~30 seconds)
└─> Submits to network
└─> Waits for GPU miner to complete block
```

### Step 2: GPU Miner Solves SHA-3 Hash
```
GPU miner (any GPU):
└─> Solves SHA-3 hash (fast on GPU)
└─> Finds matching VDF proof
└─> Creates complete block
```

### Step 3: Block Complete - Both Get Paid!
```
Complete Block:
├─ VDF Proof ✅
├─ SHA-3 Hash ✅
└─> Rewards distributed 50/50
```

---

## 📊 Example Earnings

### Before (Pure GPU Mining):
```
RTX 5090:       200 MH/s  →  10 QNK/day   ✅
AMD EPYC 9654:  10 MH/s   →  0.5 QNK/day  ❌ (Not profitable!)
```

### After (Hybrid Mining):
```
RTX 5090 (GPU): 200 MH/s  →  5 QNK/day    ✅
AMD EPYC 9654 (CPU): 10 MH/s  →  5 QNK/day   ✅ (Now profitable!)
```

---

## 🎮 Mining as a CPU Miner

### Requirements:
- **Any CPU** (even low-end works!)
- **2GB RAM minimum**
- **Q-Miner software**

### Commands:
```bash
# Start CPU mining
./q-miner --cpu-only --threads 8

# Your CPU will:
# 1. Generate VDF proofs
# 2. Submit to network
# 3. Earn 50% of block rewards
```

### Expected Earnings:
```
CPU Type               | VDF Proofs/Day | QNK Earned/Day
-----------------------|----------------|----------------
Low-end (4 cores)      | ~10 blocks     | ~10 QNK
Mid-range (8 cores)    | ~20 blocks     | ~20 QNK
High-end (16 cores)    | ~40 blocks     | ~40 QNK
Server CPU (96 cores)  | ~200 blocks    | ~200 QNK
```

---

## 🎮 Mining as a GPU Miner

### Requirements:
- **Any GPU** (NVIDIA, AMD, Intel)
- **Q-Miner software with GPU support**

### Commands:
```bash
# Start GPU mining
./q-miner --gpu --gpu-ids 0,1

# Your GPU will:
# 1. Solve SHA-3 hashes
# 2. Find matching VDF proofs
# 3. Create blocks and earn 50% of rewards
```

### Expected Earnings:
```
GPU Type        | Blocks/Day | QNK Earned/Day
----------------|------------|----------------
RTX 3080        | ~50        | ~50 QNK
RTX 4090        | ~100       | ~100 QNK
RTX 5090        | ~150       | ~150 QNK
```

---

## 🤝 Mining as Both (Hybrid Mode)

### Best Setup:
```bash
# Run both CPU and GPU mining together
./q-miner --cpu --gpu --threads 8 --gpu-ids 0

# You can:
# - Submit your own VDF proofs (CPU)
# - Complete your own blocks (GPU)
# - Earn BOTH rewards! (100% of block)
```

### Example:
```
Your CPU creates VDF proof     →  1.0 QNK (50%)
Your GPU completes block       →  1.0 QNK (50%)
Total:                         →  2.0 QNK (100%) 🎉
```

---

## 🔄 Pool Mining

### CPU Pool:
```bash
# Join CPU mining pool
./q-miner --cpu-pool http://cpu-pool.quillon.xyz:3333

# Pool distributes CPU rewards (50%) to all CPU miners
```

### GPU Pool:
```bash
# Join GPU mining pool
./q-miner --gpu-pool http://gpu-pool.quillon.xyz:3334

# Pool distributes GPU rewards (50%) to all GPU miners
```

### Hybrid Pool:
```bash
# Join hybrid pool (both CPU+GPU)
./q-miner --hybrid-pool http://pool.quillon.xyz:3335

# Pool manages both VDF and PoW submissions
```

---

## 📈 Why This Is Fair

### Traditional Mining (Unfair):
```
Rich miner with 100 GPUs:  Earns 100× more than hobbyist
Hobbyist with 1 CPU:       Earns almost nothing
Result:                    Centralized, unfair
```

### Hybrid Mining (Fair):
```
Professional with GPUs:    Earns 50% from GPU work
Hobbyist with CPU:         Earns 50% from CPU work
Result:                    Both profitable, decentralized!
```

---

## 🎯 Summary

1. **CPUs remain profitable** - Always earn 50% of block rewards
2. **GPUs still efficient** - Earn other 50% with fast SHA-3 mining
3. **Network stays decentralized** - Needs both CPU and GPU miners
4. **Fair for everyone** - Hobbyists and professionals both profit
5. **Simple to use** - Just run the miner, no complicated setup

---

## 🚀 Getting Started

### For CPU Miners:
```bash
cargo build --release --package q-miner
./target/release/q-miner --cpu-only --threads 8
```

### For GPU Miners:
```bash
cargo build --release --package q-miner --features "cuda-mining"
./target/release/q-miner --gpu --gpu-ids 0
```

### For Both:
```bash
./target/release/q-miner --cpu --gpu --threads 8 --gpu-ids 0
```

---

## 💡 Pro Tips

1. **Run both CPU and GPU** - Maximize earnings by mining both components yourself
2. **Use server CPUs** - High core count = more VDF proofs = more earnings
3. **Join a pool** - More consistent payouts for small miners
4. **Monitor the TUI** - See real-time CPU and GPU earnings separately

---

**Happy Mining! ⛏️🚀**
