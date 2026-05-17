# Discord Announcement: CPU Mining Returns

---

## Channel: #announcements

---

**CPU MINERS: Your Time Has Come**

We hear you. After the difficulty adjustment, many of you with CPUs couldn't compete anymore. GPU miners set the pace, and the bar went up. 500 miners dropped to 40.

That changes now.

We're building **Dual-Lane Mining** — a fundamentally new approach where CPU and GPU miners each have their own dedicated lane with **50% of all block rewards reserved for each**.

---

**How It Works**

Two lanes. One blockchain. Equal rewards.

**GPU Lane (BLAKE3)** — Same as today. Find a hash below the difficulty target. GPUs excel here. Nothing changes for existing miners.

**CPU Lane (Genus-2 VDF)** — Brand new. Instead of guessing hashes, you compute a **Verifiable Delay Function** — a chain of 4,300 sequential mathematical operations on a hyperelliptic curve. Each step depends on the previous one. No shortcuts. No parallelization. A 48-core GPU computes this at the same speed as a single CPU core.

Your laptop. Your desktop. Your Raspberry Pi. They all run at roughly the same speed per VDF step. That's the whole point.

---

**The Math (For the Curious)**

The VDF operates on the Jacobian of a genus-2 hyperelliptic curve:

`y² = x⁵ + x² - 1` over a 256-bit prime field

Each "step" is a doubling operation in this group — inherently sequential. To prove you did the work, you submit a **Wesolowski proof** that the server verifies in 11 milliseconds (while it took you 2 seconds to compute). 

The cryptography is post-quantum resistant. No trusted setup required.

---

**What This Means For You**

- If you're a **CPU miner**: You'll earn 50% of every block reward. No GPU needed. Download the updated miner, point it at the server, done.
- If you're a **GPU miner**: You keep earning. Your lane still exists with 50% of rewards. Nothing breaks.
- If you're a **dual miner**: Run both lanes simultaneously — GPU thread does BLAKE3, CPU thread does VDF. Earn from both pools.

---

**Development Status**

The cryptographic core is **complete and tested** (27 tests passing). Server integration is done. We're now in the testing phase on our Delta Docker environment.

Activation will happen via a height-gated upgrade — we'll announce the exact block height at least 1 week before it goes live. Old miners keep working. No emergency updates needed.

---

**Performance Numbers**

| Metric | Value |
|--------|-------|
| VDF proof time | ~2 seconds (mid-range CPU) |
| Proof verification | 11 milliseconds |
| Proof size | 145 bytes |
| CPU reward share | 50% of block reward |
| GPU reward share | 50% of block reward |
| Miner binary update needed? | Only for VDF lane. BLAKE3 miners: no change. |

---

We're bringing mining back to the people. More details coming soon.

— Q-NarwhalKnight Team

---

## Channel: #dev-updates

---

**Phase C: Genus-2 VDF Dual-Lane Mining — Dev Update**

Quick technical update on where we are with dual-lane mining.

**Completed:**
- Genus-2 Jacobian arithmetic with correct Cantor's algorithm (full 2-step composition)
- Wesolowski proof generation and O(log T) verification
- Tonelli-Shanks square root, deterministic Miller-Rabin
- Server PATH A: real cryptographic VDF verification (replaces old stub)
- Challenge endpoint returns VDF lane parameters when active
- `double_fast()` optimization: 454μs per doubling (release)
- VDF evaluator with checkpoint storage for efficient proof generation
- Montgomery 256-bit field arithmetic foundation (mul/add/sub working)

**In Progress:**
- Dual-lane reward split in block producer (50/50 with grace period)
- Delta Docker integration testing
- Montgomery inverse fix (will bring doubling cost from 454μs to ~50μs)

**Architecture:**
```
Challenge → Miner chooses lane:

  CPU Lane:
    seed = blake3(challenge || nonce)
    g = hash_to_curve(seed)        → Jacobian element
    y = [2^4300]g                  → 4300 sequential doublings (~2s)
    π = wesolowski_proof(g, y)     → proof of work done
    submit(y, π)                   → server verifies in 11ms

  GPU Lane:
    seed = blake3(challenge || nonce)
    hash = blake3^100(seed)        → 100 sequential hashes
    if hash < target: submit       → standard PoW
```

**Activation:** Height-gated behind `GENUS2_VDF_MINING` upgrade (currently `u64::MAX`). Env var override `Q_GENUS2_VDF_ACTIVATION_HEIGHT` for testing on Delta Docker. Production activation announced 1 week in advance.

**Binary compatibility:** Old miners work fine — they just earn from the BLAKE3 lane (50% instead of 100%). No forced update.

Code: 3 commits on `feature/safe-batched-sync-v1.0.2`. 27 tests passing. All work peer-reviewed by DeepSeek for cryptographic correctness.

---

## Channel: #mining

---

**Heads up miners — big changes coming to how rewards work**

Short version: We're adding a second mining lane specifically for CPUs. 50% of every block reward goes to CPU miners, 50% to GPU miners.

**FAQ:**

**Q: Do I need to update my miner?**
A: Only if you want to mine the VDF (CPU) lane. Your current GPU miner keeps working exactly as-is. You'll earn 50% of rewards instead of 100% once the upgrade activates.

**Q: Can I mine both lanes at once?**
A: Yes. Run the GPU thread for BLAKE3 and a CPU thread for VDF simultaneously. Earn from both pools.

**Q: How much can a CPU earn?**
A: The VDF lane gets 50% of the total block reward. If you're the only CPU miner, you get all 50%. As more CPU miners join, rewards split proportionally by difficulty weight (same as GPU lane).

**Q: When does this activate?**
A: We'll announce an activation block height at least 1 week before it goes live. Currently in testing phase.

**Q: What hardware do I need for VDF mining?**
A: Any modern CPU. The VDF computation is sequential — more cores don't help. A single core produces one proof every ~2 seconds. Laptops, desktops, even ARM processors work.

**Q: Is this like RandomX / Monero CPU mining?**
A: Similar goal (CPU fairness) but completely different approach. RandomX uses memory-hard hashing. We use a Verifiable Delay Function on a genus-2 hyperelliptic curve — mathematically proven to be sequential, with efficient O(log T) proof verification.

**Q: Will GPU mining become unprofitable?**
A: No. GPU miners keep 50% of all rewards. The total emission doesn't change. You just now share the pie with CPU miners.

Stay tuned. We'll post updated miner binaries and setup instructions before activation.
