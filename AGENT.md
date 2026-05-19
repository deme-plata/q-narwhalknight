# AGENT.md — One-page reference for AI agents operating on Quillon Graph

**Audience**: AI agents (Claude, Codex, DeepSeek, anything MCP-capable) that need to read balances, sign transactions, do DEX swaps, or mine — without spending an hour reading source code first.

**Source of truth**: this file. If you find yourself grepping `crates/q-types/src/lib.rs` to figure out a transaction field, that's a bug in this doc — open a PR to add what was missing.

---

## 1. Wallet — derivation and addresses

A Quillon wallet is an Ed25519 keypair. The session-derivable variant (used by MCP, agentic loops, brain-wallet-like flows) is:

```
seed_string  (utf-8, any length, BUT BIP39 validation may apply — see §1.3)
priv_key     = SHA3-256(seed_string.as_bytes())          // 32 bytes
pub_key      = Ed25519::derive(priv_key)                 // 32 bytes
address      = "qnk" + hex(pub_key)                      // 67 chars total
```

Reference impl: `crates/q-trading-bot/src/wallet_auth.rs::AgentWallet::from_seed_string` (lines 30-50).
Verified for the session-derived wallet `qnk7154929a6aa0c118791373ea21004aca6e494e6e031c36f780cd5acedf031ccb` with seed `9c83a476…` (stored in `/root/.claude/quillon-agent-seed`).

### 1.1 Code, three languages

**Rust:**
```rust
use ed25519_dalek::SigningKey;
use sha3::{Digest, Sha3_256};
let priv_bytes: [u8; 32] = Sha3_256::digest(seed.as_bytes()).into();
let sk = SigningKey::from_bytes(&priv_bytes);
let address = format!("qnk{}", hex::encode(sk.verifying_key().to_bytes()));
```

**TypeScript** (`gui/quantum-wallet/src/services/walletAuth.ts:170-237`):
```ts
import { keccak_256 } from 'js-sha3'; // SHA3-256 NOT Keccak — use sha3_256!
import * as ed from '@noble/ed25519';
const priv = sha3_256(new TextEncoder().encode(seed)); // 32 bytes
const pub = await ed.getPublicKey(priv);
const address = 'qnk' + Buffer.from(pub).toString('hex');
```

**Python:**
```python
from hashlib import sha3_256
from nacl.signing import SigningKey
priv = sha3_256(seed.encode()).digest()
sk   = SigningKey(priv)
address = 'qnk' + sk.verify_key.encode().hex()
```

### 1.2 BIP39 vs raw-hex-seed

After the 2026-05-17 brainwallet incident, the wallet UI and `mcp__quillon-wallet__import_wallet` REJECT non-BIP39 strings (12 or 24 word mnemonic from the BIP39 wordlist with valid checksum). The session-derived wallets above still work because the derivation is run client-side; the validator only kicks in at the import path. If you have a 64-hex-char string, do the derivation yourself (§1.1) — do not pass it through `import_wallet`.

### 1.3 Wallet storage in this repo

- `/root/.claude/quillon-agent-seed` — the agent's session seed (only readable by root, 64-char hex). Lost if memory wipe without backup.
- `TRADING_SEED` env var — overrides the file.
- `TRADING_SEED_FILE` env var — overrides the default path.

---

## 2. X-Wallet-Auth — HTTP request signing

For endpoints that need cryptographic proof of wallet ownership (e.g. `/api/v1/wallets/<addr>/balance` since v10.9.55):

```
challenge = SHA3-256(
    address_bytes (32) ||
    timestamp_i64.to_le_bytes() (8) ||
    request_path_utf8 (variable)
)
signature = Ed25519::sign(priv_key, challenge)   // 64 bytes
```

Header value is JSON:
```json
{"address":"qnk<64hex>","timestamp":1779126440,"scheme":"Ed25519","signature":"<128hex>"}
```

Reference: `crates/q-trading-bot/src/wallet_auth.rs::AgentWallet::sign_request` (lines 84-100).

---

## 3. Transactions — wire format

The chain stores `Transaction` (`crates/q-types/src/lib.rs:2185`). To submit a self-signed transaction:

### 3.1 Fields you must set

| Field | Type | Notes |
|---|---|---|
| `from` | `[u8; 32]` | Your pub_key bytes |
| `to` | `[u8; 32]` | Recipient pub_key bytes |
| `amount` | `u128` | **Display × 10²⁴** (e.g. 0.05 QUG = 5×10²²). Serialized as **string** in JSON. |
| `fee` | `u128` | At least ~10¹⁵ raw; same scale as amount. Serialized as string. |
| `nonce` | `u64` | Fetch from `GET /api/v1/wallets/<addr>/nonce` — increments per tx |
| `timestamp` | `DateTime<Utc>` | RFC3339, ISO 8601 |
| `data` | `Vec<u8>` | Empty for regular transfers |
| `token_type` | enum string | `"QUG"` for native, `"QUGUSD"` for stablecoin, `"Custom"` for tokens |
| `fee_token_type` | enum string | Default fee token is `"QUGUSD"` |
| `tx_type` | enum string | `"Transfer"` for plain QUG send (`0x00`) |
| `signature_phase` | enum string | `"Phase0Ed25519"` is current default |
| `signature` | `Vec<u8>` | Ed25519 over signing_payload (§3.2) |
| `memo` | `Option<String>` | Free-form text. No formally enforced limit; keep under ~1 KB to be polite. |
| `id` | `[u8; 32]` | Zero-fill — server recomputes hash |

### 3.2 Signing payload

The signature does NOT cover the postcard-serialized whole transaction. It covers this field-by-field SHA3-256:

```
SHA3-256(
    from (32) ||
    to (32) ||
    amount.to_le_bytes() (16) ||
    fee.to_le_bytes() (16) ||
    nonce.to_le_bytes() (8) ||
    timestamp_millis_i64.to_le_bytes() (8) ||
    data (variable) ||
    token_type_discriminant (1 byte: QUG=0, QUGUSD=1, Custom=2) ||
    token_type_address (32 bytes) ||
    fee_token_type_discriminant (1) ||
    fee_token_type_address (32) ||
    tx_type_byte (1: Transfer=0x00, Coinbase=0x01, Swap=0x21, etc.)
)
```

Reference: `crates/q-types/src/lib.rs::Transaction::signing_payload` (line 2795).

### 3.3 Token address constants

```
QUG    : 0x51 0x55 0x47 0x00 [padded to 32 zeros]   ("QUG\0\0\0…")
QUGUSD : 0x51 0x55 0x47 0x55 0x53 0x44 [padded to 32]("QUGUSD\0\0…")
Custom : the 32-byte contract address itself
```

### 3.4 Submitting

```
POST /api/v1/transactions
Content-Type: application/json
Body: {"transaction": { ...fields above... }}
```

The server (`crates/q-api-server/src/handlers.rs::submit_transaction`, line 2593) verifies the signature, checks fee, then inserts into the mempool. Response is the tx hash.

---

## 4. Endpoints — quick reference

| Endpoint | Method | Auth | What it does |
|---|---|---|---|
| `/api/v1/status` | GET | none | Chain status, height, version (top-level `data.upgrades.current_height`) |
| `/api/v1/peers` | GET | none | Connected peers |
| `/api/v1/transactions` | POST | self-signed Transaction | Submit pre-signed tx (use this for agent-driven sends) |
| `/api/v1/transactions/send` | POST | OAuth `auth_token` | Convenience wrapper; server builds + signs (MCP path) |
| `/api/v1/wallets/<addr>/balance` | GET | X-Wallet-Auth header | Balance of this wallet (v10.9.55+) |
| `/api/v1/wallets/<addr>/nonce` | GET | none (?) | Current nonce for building next tx |
| `/api/v1/dex/quote` | GET | none | DEX swap quote (no commitment) |
| `/api/v1/dex/swap` | POST | OAuth or signed | Execute DEX swap |
| `/api/v1/mining/stats` | GET | none | Network hashrate, block reward |
| `/api/v1/proof/tip` | GET | none | Recursive-SNARK proof of tip (phase 1 placeholder until Job D lands) |

---

## 5. MCP tool → API endpoint

`tools/quillon-wallet-mcp/src/index.ts` registers 17 tools. They mostly wrap the OAuth path (`auth_token` based) rather than X-Wallet-Auth — which is fine for human-with-browser flows but blocks self-derived agent wallets.

| MCP tool | Wraps | Auth |
|---|---|---|
| `get_balance` | `/api/v1/wallets/<addr>/balance` | X-Wallet-Auth (required) |
| `send_qug` | `/api/v1/transactions/send` | OAuth `authToken` (requires browser flow) |
| `dex_swap` | `/api/v1/dex/swap` | OAuth `authToken` |
| `dex_get_quote` | `/api/v1/dex/quote` | none |
| `dex_list_tokens` | static + chain query | none |
| `network_status` | `/api/v1/status` | none (BUG: reads `s.current_height` but field is `s.upgrades.current_height`) |
| `mining_status` | `/api/v1/mining/stats` | none |
| `start_mining` | returns shell command | client-side |
| `authenticate_wallet` | OAuth device flow | starts |
| `check_auth` | OAuth poll | none |
| `import_wallet` | client-side | BIP39 validation enforced |

**For agentic loops** (no human in the loop): bypass MCP for sending. Build a signed Transaction (§3) and POST to `/api/v1/transactions` directly. See `crates/q-trading-bot/src/bin/send_with_memo.rs` for a worked example.

---

## 6. Common gotchas

1. **24 decimals for amounts.** All token amounts in `Transaction.amount` are display-value × 10²⁴, regardless of the token's "decimals" field shown in `dex_list_tokens`. The 8-decimals shown for QUG is the *display* precision; the on-chain raw value uses 24.
2. **u128 must serialize as string in JSON.** Browser/JSON loses precision past 2⁵³. The `u128_serde` annotation in q-types handles this server-side; clients must send strings.
3. **Default fee token is QUGUSD, not QUG.** The MCP `send_qug` debits QUGUSD for the fee. Make sure the wallet has both tokens.
4. **`current_height` in `/status` is nested under `upgrades`**, not top-level. The MCP `network_status` has a bug here (returns "unknown").
5. **MessagePack truncates u128 to u64.** If you write a P2P tool, use `u128_serde` carefully — this corrupted coinbase amounts before v3.2.7.
6. **Postcard serialization is for Transaction.hash() and P2P messages.** For JSON submission, use serde-default field names.
7. **DEX pool lookup keys by symbol for Native/Stablecoin, by full address for Custom/Wrapped.** Passing "MOON" symbol returns "no pool"; passing the qnk-address works. The MCP `tokenRef()` helper does this — if you bypass MCP, replicate it.
8. **DEX rules for AMM amounts in 24-decimal scale.** This applies to pool reserves too. Pool reserves < 10²² implies a broken pool — find a deeper one.
9. **save_wallet_balances MUST be max-wins.** Any code that writes balances must check `existing >= new` first. See `CLAUDE.md` Rule 1 for the incident.

---

## 7. Quick recipes

### 7.1 "Send 0.05 QUG with a memo to a specific address"

```bash
TRADING_SEED=<64-char-hex> cargo run --release \
    --bin send_with_memo -p q-trading-bot -- \
    --to qnk<64hex> --amount-qug 0.05 \
    --memo "your free-form memo here"
```

`send_with_memo` (`crates/q-trading-bot/src/bin/send_with_memo.rs`) is the worked example. It derives wallet, fetches nonce, builds Transaction, signs, submits.

### 7.2 "Read my balance" (signed)

```rust
let auth_header = wallet.sign_request("/api/v1/wallets/<addr>/balance")?;
client.get(url).header("X-Wallet-Auth", auth_header).send().await?;
```

### 7.3 "Quote then swap on DEX"

```python
# Quote (no auth)
quote = requests.get("https://quillon.xyz/api/v1/dex/quote?from=QUG&to=QUGUSD&amount=1").json()

# Swap — same JSON shape as Transaction but with tx_type="Swap" (0x21) and
# data field encoding the swap parameters. See dex_swap_signed in
# tools/quillon-wallet-mcp/src/index.ts for the data encoding.
```

### 7.4 "Start mining to my wallet"

```bash
curl -fSL https://quillon.xyz/downloads/q-miner-linux-x64 -o /tmp/q-miner
chmod +x /tmp/q-miner
/tmp/q-miner --server https://quillon.xyz --wallet qnk<your-addr>
```

`mcp__quillon-wallet__start_mining` returns this shell command — it doesn't actually start mining itself.

---

## 8. What this document is NOT

- Not a protocol specification — see `papers/qnk-combined-whitepaper-2026-05-07.pdf` for that
- Not the full type system — see `crates/q-types/src/lib.rs`
- Not a security analysis — see `papers/q-narwhalknight-consensus-security-analysis.pdf` and CLAUDE.md
- Not stable — the wire format and endpoints have evolved through 10.x. If something here doesn't work, it's likely the doc is stale; check the git log on the relevant file in `crates/q-types` or `crates/q-api-server`.

---

*Last touched: 2026-05-18. Add to this when you discover a missing recipe — the file-grep saved next time is your gift to future agents.*
