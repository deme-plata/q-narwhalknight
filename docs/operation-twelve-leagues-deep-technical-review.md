# Operation Twelve Leagues Deep — Incident Technical Review

**Project:** Q-NarwhalKnight (QUG)  
**Market Cap at Incident:** ~$920M  
**Incident Start:** 2026-04-05  
**Status as of 2026-04-07 14:00 UTC:** ONGOING — final build deploying  
**Prepared for:** Internal team + DeepSeek peer review  
**Document Version:** 1.0  

---

## 1. Executive Summary

A `kill -9` sent to the Epsilon supernode during active RocksDB writes triggered a cascading failure across 15 distinct bugs spanning storage, sync, serialization, balance accounting, VDF verification, and mining gating logic. The chain has been stuck for 24+ hours. Each fix exposed the next layer of bugs, creating a "whack-a-mole" progression through deeply interacting subsystems.

The final build incorporating all 15 fixes is deploying now.

---

## 2. Bug Interaction Map

```
kill -9 on Epsilon
    |
    v
Bug 1: cleanup only scans forward (corrupt blocks persist)
    |
    v
Bug 2: cleanup in recover() sees height 0 (fix is no-op)
    |
    v
Bug 3: checkpoint auto-repair fights pointer reset (infinite loop)
    |--- Manual intervention: remove checkpoints, delete orphans
    v
Bug 4: 0-block peer responses = "success" (sync stalls)
    |
    v
Bug 5: current_height_atomic stale (HEIGHT CLAMP stuck)
    |
    v
Bug 6: HEIGHT CLAMP caps sync at stale+10K forever
    |--- Fix 5 breaks the cycle, but sync still needs peers
    v
Bug 13: gossipsub mesh broken (no peer-height announcements)
    |
    v
Bug 9: HTTP bootstrap only registers Beta (single peer)
    |
    v
Bug 10: Q_P2P_ONLY=1 blocks HTTP entirely (zero sync paths)
    |
    v
Bug 12: Delta disk full (backup sync source unavailable)
    |
    v
Bug 11: old peer sends Vec<QBlock> not BlockPackResponse (unparseable)
    |--- Fallback chain: postcard, msgpack, raw Vec, legacy structs
    v
Bug 7+8: balance merge inflates values (economic bug, not blocking)
    |
    v
Bug 14: VDF cap 10K vs needed 134K (100% mining rejection) *** BLOCKING ***
    |
    v
Bug 15: fast-sync gate discards all solutions when >10K behind *** BLOCKING ***
    |
    v
    [NO BLOCKS PRODUCED]
```

---

## 3. All 15 Bugs

| # | Bug | File:Line | Severity | Fix |
|---|-----|-----------|----------|-----|
| 1 | cleanup_corrupt_blocks_above() only scans forward | lib.rs:2880 | Critical | New cleanup_corrupt_blocks_near_tip() |
| 2 | recover() gets height 0 from empty cache | lib.rs:3255 | Critical | Moved cleanup to after scan |
| 3 | Auto-repair cascade (checkpoints vs pointer) | lib.rs:2180 | Critical | Manual: remove checkpoints |
| 4 | 0-block responses = success in turbo sync | turbo_sync.rs:5043 | Critical | Return Err on empty response |
| 5 | current_height_atomic never updated after sync | main.rs:19462 | Critical | Read DB pointer after sync |
| 6 | HEIGHT CLAMP poisons sync target | main.rs:14540 | High | Fixed by #5 (stale atomic) |
| 7 | Balance "keep higher" preserves inflated values | main.rs:19706 | High | RocksDB authoritative |
| 8 | token_balances same bug | main.rs:19762 | High | Same fix as #7 |
| 9 | HTTP bootstrap hardcoded for Beta only | main.rs:18521 | High | URL-to-peer-ID mapping for all servers |
| 10 | Q_P2P_ONLY=1 blocks HTTP fallback | main.rs:19510 | Medium | Env var removed |
| 11 | Block-pack serialization mismatch | block_pack.rs:186 | Critical | Legacy QBlock V1/V2/V3 fallbacks |
| 12 | Delta root partition full (Docker 25GB) | Infrastructure | Medium | Docker moved to /home |
| 13 | Gossipsub mesh broken for peer-heights | main.rs:2930 | High | Mitigated by HTTP bootstrap |
| 14 | **VDF cap 10K vs 134K iterations** | **main.rs:15153** | **BLOCKING** | **Cap raised to 200K** |
| 15 | **Fast-sync gate discards solutions** | **main.rs:15504** | **BLOCKING** | **Bypass with Q_ALLOW_SOLO_MINING** |

---

## 4. Incident Timeline

| Time (CEST) | Event |
|-------------|-------|
| Apr 05 ~15:00 | `kill -9` on Epsilon during RocksDB writes at height ~13,489,443 |
| Apr 05 17:11 | v10.2.7 deployed with debug logging — no improvement |
| Apr 06 13:36 | Investigation begins — 100% mining rejection |
| Apr 06 15:50 | v10.2.8 first deploy — 180 corrupt blocks cleaned |
| Apr 06 16:08 | Bug 2: cleanup in recover() was no-op (height 0) |
| Apr 06 17:49 | Fix: cleanup moved to after scan — works correctly |
| Apr 06 19:30 | Bug 3: auto-repair cascade discovered |
| Apr 06 20:14 | fix-corrupt-tip --delete-above: 710 orphaned blocks removed |
| Apr 06 20:22 | Checkpoint files moved, pointer stabilized |
| Apr 06 21:00 | Bugs 4-6: turbo sync stall discovered and fixed |
| Apr 07 05:35 | Bug 7: balance "keep higher" inflation fixed |
| Apr 07 05:42 | Bug 11: postcard fallback (wrong format) |
| Apr 07 07:00 | Bug 11: rmp_serde fallback (wrong format) |
| Apr 07 07:07 | Hex dump reveals 0xc8 = bincode Vec length 200 |
| Apr 07 07:32 | Vec<QBlock> fallback — struct layout incompatible |
| Apr 07 09:16 | Bug 9: HTTP bootstrap multi-peer fix |
| Apr 07 09:59 | Bug 10: Q_P2P_ONLY=1 found and removed |
| Apr 07 10:03 | Bug 12: Delta disk full, Docker 25GB cleaned |
| Apr 07 11:06 | Debian 12 build tested successfully |
| Apr 07 12:25 | Bug 11: Legacy QBlock V1/V2/V3 fallbacks added |
| Apr 07 12:30 | Balances normalized |
| Apr 07 13:37 | **Bug 14: VDF cap discovered — first solution PASSES** |
| Apr 07 13:38 | **Bug 15: fast-sync gate discovered — solutions discarded** |
| Apr 07 13:40 | Fast-sync gate bypass added |
| Apr 07 ~14:30 | Final build deploying (all 15 fixes) |

---

## 5. Current Status

**Working:**
- Corrupt blocks cleaned, pointers stable
- Balance merge corrected (RocksDB authoritative)
- HTTP bootstrap discovers all peers
- Block-pack handles 8 serialization formats
- VDF cap raised (solutions PASS verification)
- Fast-sync gate bypassed for solo mining
- Delta disk space recovered

**Deploying:**
- Final v10.2.8 build with all 15 fixes + heavy debug logging

**Expected after deploy:**
- Mining solutions pass VDF -> queued to producer -> NEW BLOCK within 30 seconds
- Height advances past 13,475,449
- Explorer shows live chain

---

## 6. Recovery Checklist

- [ ] Deploy v10.2.8 final build to Epsilon
- [ ] Verify `Q_ALLOW_SOLO_MINING=true` set
- [ ] Verify `Q_P2P_ONLY` NOT set
- [ ] Check logs: `passed=N` where N > 0
- [ ] Check logs: `PHASE 4` entries (solutions reaching producer)
- [ ] Check logs: `NEW BLOCK` (block production)
- [ ] Explorer: height advancing
- [ ] Monitor 100 blocks for stability
- [ ] Deploy to Beta and Delta
- [ ] Update community download

---

## 7. Architectural Recommendations

1. **Dynamic VDF cap** — derive from height formula with 2x margin, not hardcoded
2. **Corruption integration test** — simulate kill -9, verify auto-recovery
3. **Fix checkpoint vs cleanup conflict** — verify checkpoint blocks are deserializable
4. **Eliminate "keep higher" pattern** — audit all HashMap merges
5. **Block-pack format negotiation** — version byte header instead of 8 fallbacks
6. **HTTP bootstrap return peer_id** — eliminate hardcoded URL-to-ID mapping
7. **RocksDB WAL flush on SIGTERM** — pre-stop hook in systemd
8. **Peer quality scoring** — deprioritize peers returning 0 blocks
9. **HEIGHT CLAMP recovery mode** — temporarily raise cap during large-gap sync
10. **Gossipsub mesh monitoring** — alert when peer-heights topic drops below threshold

---

**Prepared by:** Claude Code (Server Alpha)  
**Operation:** Twelve Leagues Deep  
**Classification:** Mainnet Critical
