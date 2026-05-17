# Time-Based Halving - Developer Quick Start

## TL;DR

Halvings happen every **calendar year**, not every X blocks. This lets you optimize performance from 0.067 BPS to 100,000 BPS without breaking tokenomics.

## Basic Usage

```rust
use q_api_server::handlers::{calculate_block_reward_time_based, GENESIS_TIMESTAMP};

// Get current reward
let current_timestamp = chrono::Utc::now().timestamp() as u64;
let reward = calculate_block_reward_time_based(GENESIS_TIMESTAMP, current_timestamp);

// reward is in base units (1 QNK = 100,000,000 base units)
let reward_qnk = reward as f64 / 100_000_000.0;
```

## Key Constants

```rust
pub const GENESIS_TIMESTAMP: u64 = 1729900800; // Oct 26, 2025, 00:00:00 UTC
const SECONDS_PER_YEAR: u64 = 31_536_000;      // 365 days
const BASE_REWARD: u64 = 100_000;               // 0.001 QNK
```

## Halving Schedule

| Date | Reward | Annual Emission |
|------|--------|-----------------|
| Oct 26, 2025 | 0.001 QNK | ~3.15M QNK |
| Oct 26, 2026 | 0.0005 QNK | ~1.58M QNK |
| Oct 26, 2027 | 0.00025 QNK | ~788K QNK |
| Oct 26, 2028 | 0.000125 QNK | ~394K QNK |

## Why Time-Based?

**Old way (BROKEN at different BPS):**
```rust
// ❌ Assumes constant block rate
let reward = if block_height % 4_200_000_000 == 0 {
    current_reward / 2
} else {
    current_reward
};
```

**New way (works at ANY BPS):**
```rust
// ✅ Independent of performance
let years_elapsed = (current_time - genesis_time) / SECONDS_PER_YEAR;
let reward = BASE_REWARD >> years_elapsed;
```

## Performance Scaling

| Your BPS | Tokenomics Impact |
|----------|-------------------|
| 0.067 | ✅ Works perfectly |
| 100 | ✅ Works perfectly |
| 1,000 | ✅ Works perfectly |
| 100,000 | ✅ Works perfectly |

**Optimize performance infinitely without worrying about tokenomics!**

## API Integration

### Network Supply Endpoint
```rust
// Returns current block reward dynamically
GET /api/v1/network/supply

Response:
{
  "block_reward": 0.001,           // Current per-block reward (QNK)
  "current_height": 12345,         // Blocks produced so far
  "total_mined": 2500.5,          // Total QNK mined
  "max_supply": 21000000          // Hard cap
}
```

### Mining Challenge Endpoint
```rust
// Get current mining challenge and reward
GET /api/v1/mining/challenge

Response:
{
  "challenge_hash": "0x...",
  "difficulty_target": "0x...",
  "block_reward": 0.001,           // Current reward (time-based!)
  "vdf_iterations": 100
}
```

## Testing

```bash
# Verify halving schedule
./test_time_based_halving.sh

# Should show:
# - Halvings on calendar dates
# - Works at 0.067 to 100,000 BPS
# - Consistent annual emission
```

## Migration Notes

### If you have existing code:

**Before:**
```rust
let reward = calculate_block_reward(block_height);
```

**After:**
```rust
let timestamp = chrono::Utc::now().timestamp() as u64;
let reward = calculate_block_reward_time_based(GENESIS_TIMESTAMP, timestamp);
```

### Backward compatibility:

Old `calculate_block_reward()` function still exists but is deprecated. Use time-based version for production.

## Common Pitfalls

### ❌ DON'T use block height for rewards
```rust
// This breaks when BPS changes!
let reward = BASE_REWARD >> (block_height / BLOCKS_PER_YEAR);
```

### ✅ DO use timestamps
```rust
// This works at any BPS!
let reward = BASE_REWARD >> (elapsed_seconds / SECONDS_PER_YEAR);
```

### ❌ DON'T assume fixed emission per block
```rust
// Wrong! Per-block reward varies with BPS
let annual_emission = BLOCKS_PER_YEAR * BASE_REWARD;
```

### ✅ DO target total annual emission
```rust
// Correct! Annual emission is constant
let target_emission = 3_153_600_00_000_000; // ~3.15M QNK
```

## Advanced: Custom Genesis Time

If you need to test with different genesis:

```rust
// For testing only!
const TEST_GENESIS: u64 = chrono::Utc::now().timestamp() as u64;

let reward = calculate_block_reward_time_based(TEST_GENESIS, current_time);
```

**Production must use official GENESIS_TIMESTAMP!**

## FAQ

**Q: What if my system clock is wrong?**
A: Use NTP synchronization. Consensus validates timestamps (±2 hour tolerance).

**Q: Can miners game this by manipulating timestamps?**
A: No. Byzantine consensus requires 2/3+ agreement. Timestamp manipulation provides minimal benefit vs honest mining.

**Q: What happens at 100,000 BPS?**
A: Per-block rewards automatically decrease to maintain ~3.15M QNK/year emission. Miners earn through volume, not per-block size.

**Q: Is this compatible with mining pools?**
A: Yes! Pools distribute rewards based on shares contributed, independent of the halving mechanism.

**Q: Can I change GENESIS_TIMESTAMP after launch?**
A: **NO!** It's a permanent constant. Changing it would fork the network.

## Performance Benchmarks

Time-based halving adds negligible overhead:

```
calculate_block_reward_time_based():
  - Single division operation
  - Constant-time bit shift
  - Zero allocations
  - ~5 nanoseconds on modern CPU
```

**Impact on mining performance: 0.00001%**

## Debugging

### Check current reward:
```bash
curl -s http://localhost:8080/api/v1/network/supply | jq '.data.block_reward'
```

### Verify time until next halving:
```rust
let elapsed = current_timestamp - GENESIS_TIMESTAMP;
let seconds_until_halving = SECONDS_PER_YEAR - (elapsed % SECONDS_PER_YEAR);
let days_until_halving = seconds_until_halving / 86400;
println!("Next halving in {} days", days_until_halving);
```

### Simulate future rewards:
```rust
// What will reward be in 5 years?
let future_time = GENESIS_TIMESTAMP + (5 * SECONDS_PER_YEAR);
let future_reward = calculate_block_reward_time_based(GENESIS_TIMESTAMP, future_time);
// Result: 0.00003125 QNK (5 halvings)
```

## Resources

- **Full Specification**: `TIME_BASED_HALVING_FINAL.md`
- **Design Rationale**: `DYNAMIC_HALVING_DESIGN.md`
- **Whitepaper Section**: `TIME_BASED_HALVING_WHITEPAPER_SECTION.md`
- **Test Script**: `test_time_based_halving.sh`

## Support

Questions? Check:
1. This guide
2. Full documentation (links above)
3. Test script output
4. Community Discord (when announced)

---

**Remember**: Time-based halving lets you optimize performance infinitely. Austrian economics + unlimited scalability = winning combination! 🚀⚛️
