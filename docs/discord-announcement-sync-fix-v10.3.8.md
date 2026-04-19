# Discord Announcement — v10.3.8 Sync Fix

---

**Quillon Graph v10.3.8 — Full Chain History Restored**

We just shipped one of the most important infrastructure fixes since mainnet launch.

**What changed:**
New nodes joining the network now sync from block 75,000 (near-genesis, Feb 2026) instead of block 14,000,000 (April 2026). That's 2 months of chain history that was previously invisible to new nodes — now fully accessible.

**The backstory:**
545,710 blocks from the early chain (Feb-March 2026) were stored in an older database format by the original gossipsub receiver. The sync code couldn't read them because the deserializer didn't understand the old format, and a separate cleanup process was quietly deleting blocks it couldn't parse. We caught the deletion, stopped it, decoded the old binary format from a hex dump, wrote a manual parser, and optimized the block serving with a lazy RocksDB iterator.

**Performance:**
- Sync speed through the recovered region: **2,000-12,000 blocks/sec**
- Full sync from near-genesis to tip (~15.8M blocks): **~2 hours** on a decent connection
- No impact on mining, balances, or DEX — those were never affected

**What this means for you:**
- **Miners:** Nothing changes for you. Keep mining.
- **Node operators:** Update to v10.3.8. New nodes will sync the complete chain automatically.
- **Everyone:** The block explorer now shows the full chain history from day 8 of mainnet onward.

**Download:**
```
wget https://quillon.xyz/downloads/q-api-server-linux-x86_64
chmod +x q-api-server-linux-x86_64
```

**Technical details for the curious:**
- Stopped 3 code paths that permanently deleted blocks on deserialization failure
- Added `scan_prefix_seek` to bypass RocksDB bloom filter false negatives on CF_BLOCKS
- Wrote a manual binary parser for the v7.x-v9.x block format (1 byte field width difference at offset 8)
- Implemented forward-seek lazy iterator for sparse block ranges (200 blocks in 257ms vs 3,200ms before)
- Checkpoint sync probe now searches 1000-block windows instead of single heights

Zero database modifications. All fixes are read-path only. The blocks were never lost — the code just couldn't find them.

*— Quillon Foundation*
