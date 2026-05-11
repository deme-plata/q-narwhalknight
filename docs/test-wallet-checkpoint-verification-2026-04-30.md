# Test Wallet — Checkpoint Verification 2026-04-30

**Purpose:** Verify that the v10.5.2 checkpoint system correctly credits this wallet after a fresh-node sync from Docker.  
**Date created:** 2026-04-30  
**Created by:** Server Beta (Claude Code)

---

## Wallet Credentials

**Mnemonic (12 words — BIP39 English):**
```
year detail animal stuff day buzz notice accident element law flock good
```

**Wallet Address (qnk format):**
```
qnkd0f5cd1696087120afc959b354056cd230deb2639ba60de412667298b35a906e
```

**Raw SHA3-256 private seed (hex):**
```
715a41e138f6aea6327fa2ceb052542354f21289cb9300710560ff569d135687
```

---

## Key Derivation

- Private key seed = SHA3-256(mnemonic phrase)
- Public key = Ed25519 public key from private seed
- Wallet address = `qnk` + hex(public key)

---

## Miner Details

**Mining server:** `https://quillon.xyz`  
**Mode:** solo  
**Miner name:** ClaudeTestMiner-Epsilon  
**Threads:** 4  
**PID on Epsilon:** 3535920  
**Log file:** `/home/orobit/miner-test-wallet.log` (on Epsilon, 89.149.241.126)

---

## Checkpoint Verification Test

**Docker container on Delta:** `q-fresh-v10502` (started 2026-04-30)  
**Binary version:** v10.5.2  
**Expected behavior:**
1. Fresh node detects height = 0 → triggers checkpoint loading
2. Checkpoint at height 16,538,868 loads 1,326 wallet balances
3. Node replays Coinbase + Transfer transactions from 16,538,869 to current tip
4. `apply_dex_qug_adjustments()` re-applies any post-checkpoint DEX credits (v10.5.2 fix)
5. Mining rewards sent to `qnkd0f5cd...` should appear on fresh-node after syncing

**Balance check:**
- After the fresh Docker container syncs to tip, query balance of `qnkd0f5cd...`
- Expected: matches the balance shown on `quillon.xyz` frontend for this wallet
- Any discrepancy indicates a checkpoint replay bug

---

## How to Check Balance

```bash
# From Epsilon (authoritative node):
journalctl -u q-api-server --since "1 hour ago" | grep "qnkd0f5cd"

# On Delta Docker container (after sync):
docker exec q-fresh-v10502 curl -s http://localhost:8080/api/v1/status | python3 -m json.tool
```

**Note:** Balance API endpoints require authentication. Use the mnemonic above to log into `https://quillon.xyz` to check the balance from the frontend.

---

## Security Note

This is a **test wallet** created for checkpoint verification. Do not use for production funds beyond the small mining rewards generated during this test.
