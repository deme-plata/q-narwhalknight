# Epsilon Docker Test Plan — Multi-Format Block Serving

**Priority:** SAFETY FIRST. This is a $1B mainnet.
**Rule:** ALL testing in Docker containers. Epsilon production node is NEVER modified.

---

## What We're Testing

The `get_qblocks_range_any_format()` function reads blocks using multiple key formats. It is:
- **READ-ONLY** — never writes to RocksDB
- **Additive** — falls back to the new reader only when the standard reader returns 0
- **Used only in the block-pack serving handler** — responds to P2P sync requests from peers

## Safety Guarantees

1. The Docker container runs on a DIFFERENT port (8085 API, 9005 P2P)
2. The Docker container has its OWN data directory (`/home/orobit/docker-test-multiformat/`)
3. The Docker container CANNOT write to Epsilon's production database (mounted read-only via `:ro`)
4. Epsilon's production node continues running undisturbed on port 8080/9001
5. If anything goes wrong, `docker stop` kills only the test — Epsilon is unaffected

## Test Plan (4 tests, then Docker sync)

### Test 1: Verify multi-format reader is READ-ONLY
- Mount Epsilon's production DB as `:ro` (read-only)
- Call `get_qblocks_range_any_format(5249, 10)` — should return blocks
- Call `get_qblocks_range_any_format(100441, 10)` — should find DAG-layer blocks
- Verify: Epsilon's production DB is unchanged (checksum before/after)

### Test 2: Verify no crashes on empty ranges
- Call `get_qblocks_range_any_format(0, 200)` — heights 0-199 have no blocks
- Call `get_qblocks_range_any_format(1000000, 200)` — sparse range
- Verify: returns empty/partial results, no crash, no panic

### Test 3: Verify block-pack handler serves sparse blocks
- Start Docker node with multi-format binary
- Send a simulated block-pack request for heights 5249-7249
- Verify: response contains actual blocks (not 0)

### Test 4: Full Docker sync test
- Start a SECOND Docker container (the syncing node)
- It connects to the FIRST Docker container (the serving node)
- The serving node has Epsilon's DB mounted read-only
- Monitor: does the syncing node receive blocks at sparse heights?

## Implementation Order
1. Write a simple diagnostic binary that calls `get_qblocks_range_any_format` and reports results
2. Run it in Docker with Epsilon's DB mounted read-only
3. If it works, start the full Docker sync test
4. Monitor both containers + Epsilon production health
