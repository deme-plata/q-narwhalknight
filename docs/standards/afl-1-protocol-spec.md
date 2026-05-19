# AFL-1: Agent Fiber Lane Protocol Specification

| Field | Value |
|---|---|
| **Standard number** | AFL-1 |
| **Title** | Co-located signed-submit protocol for AI-agent transactions on cryptographic networks |
| **Status** | Draft |
| **Type** | Standards Track — Network Protocol |
| **Created** | 2026-05-19 |
| **License** | Apache-2.0 |
| **Author** | Claude Opus 4.7 (Anthropic), Viktor Sandstrøm Kristensen (Quillon Foundation) |
| **Version** | 1.0-draft |

> **Note on hosting**: this document is intended to live in a vendor-neutral repository (e.g. `quillon-foundation/afl-spec` or a community-governed equivalent) once initial drafting stabilises. Its current location in the Q-NarwhalKnight monorepo is provisional. Implementers should not rely on this path as permanent.

---

## Abstract

This document defines the **Agent Fiber Lane (AFL)** protocol, an open standard for AI-agent submission of cryptographically-signed transactions to distributed networks. AFL is chain-agnostic and replaces browser-mediated OAuth flows with a direct, signed, batched, low-latency submission path designed specifically for autonomous agents holding Ed25519 (or compatible) keys.

The protocol's name is taken from the financial-systems metaphor: just as large banks pay for co-located fiber lines to exchanges to minimise transaction-submission latency, AI agents on a network require a co-located submission path that bypasses human-targeted authentication flows. AFL is that path, expressed as an open specification.

AFL is licensed under Apache-2.0 with no encumbrance. Any chain may implement it; no permission is required.

---

## 1. Motivation

AI agents transact at machine speed. A single autonomous agent may issue thousands of transactions per minute; a swarm of agents may issue tens of thousands per second. Current transaction-submission interfaces — designed for human users with browser wallets — impose unacceptable friction on this workload:

1. **OAuth round-trips** require human eyeballs and browser UIs. Agents cannot complete them without human intervention.
2. **Per-transaction HTTPS POSTs** carry per-request TLS-handshake overhead that caps throughput at low thousands of connections per second, well below what agentic workloads require.
3. **Existing pre-signed-transaction submission endpoints** on most chains were designed when humans held wallets. They commonly require nonce lookups, fee estimations, and tx-construction logic on the client side — burdensome for agents whose primary capability is intent specification, not transaction assembly.

The result: agent-native economic activity is structurally bottlenecked on every existing chain. AFL removes the bottleneck.

---

## 2. Scope

This specification covers:

- The authentication scheme used by agent-driven HTTP requests (§3)
- Three endpoints (single, batched, streaming) and their wire formats (§4)
- Replay and forge protection requirements (§5)
- The server-side completion of intent specifications into full transactions (§6)
- Mandatory rate-limiting guidance (§7)

This specification explicitly does NOT cover:

- The underlying chain's consensus protocol, transaction format, or block structure. AFL is a submission interface; it sits above whatever chain it serves.
- The agent's key-derivation scheme. AFL assumes the agent holds an Ed25519 keypair (or a compatible signature scheme — see §3.3 for extensibility); how the keypair was derived is out of scope.
- The agent's decision-making logic. AFL accepts whatever intent the agent specifies.

---

## 3. Authentication: X-Wallet-Auth

### 3.1 Header format

Each AFL request carries an `X-Wallet-Auth` HTTP header whose value is a single-line JSON object:

```json
{
  "address": "<chain-prefix><hex(public_key)>",
  "timestamp": <i64 Unix seconds>,
  "scheme": "Ed25519",
  "signature": "<hex(64-byte Ed25519 signature)>",
  "body_hash": "<hex(32-byte SHA3-256 of the HTTP body, omitted only if body is empty)>"
}
```

### 3.2 Challenge construction

The signature is over the SHA3-256 hash of a canonical challenge byte sequence:

```
challenge_bytes = address_pubkey_bytes (32)        // raw public key, not the chain-prefixed string
                  || timestamp.to_le_bytes()       // i64 little-endian
                  || request_path_utf8              // e.g. "/api/v1/agent/submit"
                  || body_hash_bytes                // 32 bytes if body non-empty; omitted if body is empty

challenge = SHA3-256(challenge_bytes)
signature = Ed25519::sign(private_key, challenge)
```

The verifier reconstructs `challenge` from the request and verifies the signature against `address`'s public key.

### 3.3 Signature scheme extensibility

The `scheme` field MUST be one of:

- `"Ed25519"` (REQUIRED for v1)
- `"Dilithium5"` (OPTIONAL, post-quantum)
- `"Ed25519+Dilithium5"` (OPTIONAL, hybrid; `signature` is the concatenation of both, Ed25519 first)

Implementers MUST accept Ed25519. Implementers MAY accept the other schemes. Future versions of this specification may add additional schemes; older implementations are not required to accept them.

### 3.4 Timestamp window

Requests with `timestamp` more than 30 seconds in the past or 30 seconds in the future MUST be rejected with HTTP `401`. This provides replay protection against captured-and-replayed-later requests.

### 3.5 Body hash

For all endpoints in §4 with a non-empty request body, `body_hash` MUST equal `SHA3-256(request_body_bytes)`. The verifier MUST recompute this and reject mismatches with HTTP `401`. This binds the signature to the specific body content; an attacker cannot reuse a captured signature with a modified body.

For the WebSocket endpoint (§4.3) where each frame is signed individually, `body_hash` covers the connection-establishment payload only.

---

## 4. Endpoints

### 4.1 Single-transaction submit

```
POST /<base-path>/agent/submit
Content-Type: application/json
X-Wallet-Auth: <header per §3.1>

Body:
{
  "to": "<chain-prefix><hex-address>",
  "amount": "<u128 amount as decimal string>",
  "token_type": "<chain-specific>",
  "memo": "<optional, free-form string>",
  "fee": null,
  "nonce": null
}
```

`<base-path>` is implementation-defined (Quillon Graph reference uses `/api/v1`).

**Server behaviour:**
1. Verify X-Wallet-Auth (§3, §5)
2. If `nonce` is null, server assigns from a per-wallet counter using `last_committed_nonce + 1` plus any in-flight increments. If `nonce` is set, server verifies it is the expected value and rejects with `409` if not.
3. If `fee` is null, server uses the chain's minimum fee for the operation type.
4. Server constructs the full transaction with `from` overridden to the X-Wallet-Auth address (§5.3) and signs/inserts according to chain-specific consensus rules.
5. Server responds with HTTP `200` and a JSON receipt:
   ```json
   {
     "tx_id": "<hex 32-byte hash>",
     "assigned_nonce": <integer>,
     "included_at_block": null
   }
   ```

The `included_at_block` field is null at submit time; clients poll the chain's transaction-status endpoint for inclusion.

### 4.2 Batch submit

```
POST /<base-path>/agent/submit-batch
Content-Type: application/json
X-Wallet-Auth: <header per §3.1, body_hash covers the entire batch JSON>

Body:
{
  "transactions": [
    { "to": "...", "amount": "...", "token_type": "...", "memo": "..." },
    { "to": "...", "amount": "...", "token_type": "...", "memo": "..." },
    ...
  ]
}
```

**Constraints:**
- Implementations MUST accept batches of up to 1,000 transactions.
- Implementations MAY accept batches up to 10,000 transactions, configurable.
- One X-Wallet-Auth signature authenticates the entire batch — implementations MUST NOT require per-transaction signatures.

**Server behaviour:** as §4.1 per transaction, with server atomically allocating sequential nonces. Response is:

```json
{
  "tx_ids": ["<hex>", "<hex>", ...],
  "first_nonce": <integer>,
  "last_nonce": <integer>
}
```

If any transaction in the batch fails validation (insufficient balance, malformed `to`, etc.), the implementation MUST choose ONE of two well-documented policies and apply it consistently:

- **All-or-nothing**: the entire batch is rejected with `400` and per-transaction error details
- **Best-effort**: valid transactions are submitted, invalid are dropped, response includes per-transaction status

The chosen policy MUST be documented in implementation-specific documentation.

### 4.3 Streaming submit (WebSocket)

```
GET /<base-path>/agent/stream
Upgrade: websocket
```

**Connection establishment:**

First frame from client (immediately after WebSocket upgrade) MUST be a JSON object:

```json
{
  "auth": "<X-Wallet-Auth header value, signing 'connection_id || timestamp || path'>",
  "client_version": "<implementation-defined>"
}
```

`connection_id` is a UUID generated by the client and included in the auth challenge.

Server responds with:

```json
{
  "status": "connected",
  "connection_id": "<echoed UUID>",
  "next_nonce": <integer>,
  "max_inflight": <integer>
}
```

**Subsequent frames** (client → server) are intent JSON, one transaction per frame:

```json
{
  "client_seq_id": "<arbitrary client-side id>",
  "to": "...",
  "amount": "...",
  "token_type": "...",
  "memo": "..."
}
```

**Server replies** (server → client, one reply per accepted frame):

```json
{
  "client_seq_id": "<echoed>",
  "tx_id": "<hex>",
  "nonce": <integer>
}
```

OR on backpressure:

```json
{
  "client_seq_id": "<echoed>",
  "status": "throttle",
  "retry_after_ms": <integer>
}
```

Implementations MUST limit concurrent streams per wallet (RECOMMENDED: 4).

---

## 5. Security requirements

### 5.1 Replay protection

Implementations MUST maintain an in-memory cache of `(address, body_hash)` pairs seen in the last 60 seconds. Duplicate submissions within this window MUST be rejected with HTTP `401` and an error message indicating replay detection.

### 5.2 Stale-timestamp rejection

Per §3.4, requests with `timestamp` outside the ±30-second window from server clock MUST be rejected.

### 5.3 Forge protection: server overrides `from`

The transaction's `from` field MUST be set server-side to the address contained in the verified X-Wallet-Auth header. The server MUST NOT trust any `from` field that may appear in the request body. This prevents an authenticated wallet from forging a transaction that appears to come from a different wallet.

### 5.4 Rate limiting

Implementations MUST enforce per-wallet rate limits. RECOMMENDED default: 10,000 transactions per second per wallet (sufficient for legitimate high-throughput agents; preventing trivial DoS). Implementations MAY tune this per their capacity.

---

## 6. Server-side intent completion

Clients submit *intent*, not full transactions. The server is responsible for completing the intent into a chain-valid transaction.

### 6.1 Required fields (server-supplied)

- `from` — from X-Wallet-Auth address (§5.3)
- `nonce` — from per-wallet counter (§4.1)
- `timestamp` — chain-time at submission
- Implementation-specific consensus fields (e.g. signature phase indicators)

### 6.2 Defaulted fields (server-supplied if client omits)

- `fee` — minimum chain fee for operation
- `token_type` — defaults to the chain's native token if omitted
- `memo` — empty string if omitted

### 6.3 Client-specified fields

- `to` — REQUIRED
- `amount` — REQUIRED

Any other implementation-specific fields MAY be exposed via the JSON body; implementations document them.

---

## 7. Performance targets

These targets are normative for the AFL-1 standard. Implementations claiming AFL-1 conformance SHOULD meet them on commodity hardware:

| Operation | Server-side time |
|---|---|
| X-Wallet-Auth verify | ≤ 1 ms |
| Body-hash verification | ≤ 0.5 ms (for ≤ 16 KB body) |
| Per-wallet nonce assignment (cache hit) | ≤ 0.1 ms |
| Single-tx mempool insert | ≤ 5 ms total |
| 1000-tx batch processing | ≤ 60 ms total (60 µs/tx amortised) |

These targets enable an aggregate throughput of approximately 100,000 transactions per second per node via the batch endpoint, with eight concurrent agent connections.

---

## 8. Backwards compatibility

AFL is an additive submission interface. It does not replace existing transaction-submission endpoints; chains MAY continue to offer OAuth-mediated submission, pre-signed-transaction submission, or other paths alongside AFL.

The protocol does not modify the underlying transaction format or consensus rules. Transactions submitted via AFL are indistinguishable to validators from transactions submitted via any other path, once they enter the mempool.

---

## 9. Reference implementation

The Quillon Graph implementation (`https://github.com/deme-plata/q-narwhalknight`, specifically PR #87 and the canonicalization fix in commit `1fe23e4be`) serves as the v1 reference implementation. It includes:

- The X-Wallet-Auth verifier in Rust (`crates/q-api-server/src/wallet_auth.rs`)
- The three endpoint handlers
- The wallet-derivation helper (`crates/q-trading-bot/src/wallet_auth.rs`)
- A worked agent-side example (`crates/q-trading-bot/src/bin/send_with_memo.rs`)
- The associated OpenAPI 3.1 specification (`docs/openapi.yaml`)
- An MCP-server integration sketch (`tools/quillon-wallet-mcp/`)

Other chains MAY use this reference as a starting point. Apache-2.0 license applies.

---

## 10. Security considerations

### 10.1 Compromised seed scenarios

The X-Wallet-Auth scheme places full trust in the agent's private key. An attacker who obtains the seed can submit arbitrary transactions from the corresponding wallet. Standard key-management practices apply: keys SHOULD be held in process memory only, not persisted to disk; agent processes SHOULD have minimal additional privileges; keys SHOULD be rotatable.

### 10.2 Timestamp window vs. wall-clock drift

The 30-second window in §3.4 assumes loosely-synchronised server and agent clocks (within ~5 seconds). Agents with drifted clocks (e.g. embedded systems without NTP) MAY experience submission failures. Implementations MAY relax the window, but MUST NOT widen it beyond 5 minutes without operational justification.

### 10.3 Body-hash collision attacks

SHA3-256 is the only hash function specified for `body_hash`. Implementations MUST NOT accept other hash functions in v1. This prevents downgrade attacks against weaker hashes.

### 10.4 Rate-limit bypass via multiple wallets

The rate limit in §5.4 is per-wallet. An attacker controlling N wallets can submit at N × rate-limit. Implementations SHOULD additionally limit global throughput per source IP or per origin to prevent this class of bypass.

### 10.5 Privilege escalation via batch

A batch may contain transactions to many recipients. Implementations MUST apply per-transaction balance and authorization checks; a single X-Wallet-Auth signature on the batch does NOT authorise transactions to addresses or amounts that exceed the signing wallet's actual permissions.

---

## 11. Test vectors

> **Note**: full test vectors are TBD in v1-draft; will be added in v1-final based on reference implementation telemetry.

The following structure is required:

- A wallet derivation test (seed → priv → pub → address)
- An X-Wallet-Auth signature test (known challenge → known signature)
- A single-submit round-trip test (intent → constructed tx → signature → mempool entry)
- A batch-submit round-trip test (10-tx batch with one expected failure under best-effort policy)
- A replay-protection test (same body submitted twice → second rejected)

Reference implementations are expected to publish these test vectors in a chain-agnostic format.

---

## 12. Acknowledgements

This specification draws structural inspiration from:

- BIP (Bitcoin Improvement Proposal) format and review process
- EIP (Ethereum Improvement Proposal) format and review process
- xAI's open-sourced recommendation algorithm (Apache-2.0, `github.com/xai-org/x-algorithm`) — particularly the candidate-isolation design pattern, which informs the per-transaction-independence property maintained by AFL across batch submissions

The strategic case for AFL as an open standard (rather than a chain-specific extension) is informed by the trajectory of similar industry standards: SMTP for mail, TLS for transport, BIP-39 for mnemonic wallets. Each became a de-facto standard not because one vendor mandated it, but because the alternative — fragmented per-vendor implementations — imposed unacceptable integration cost on downstream consumers.

The same logic applies to agent-network interfaces. AI agents will use whatever protocol the chains they interact with offer; if every chain offers a different protocol, agent developers face per-chain SDK work. A single standard removes that cost.

This specification is offered to the community in that spirit. Adoption by other chains is welcomed; no permission is required.

---

## 13. Copyright

This specification is licensed under Apache-2.0. The reference implementation in Q-NarwhalKnight is licensed identically. Any implementation of AFL-1, in whole or in part, is permitted under the same license without further restriction.

---

*Authors' note: this document is a draft. Comments and proposed amendments are welcome via the issue tracker of the vendor-neutral repository, once established. Until then, file issues against the Q-NarwhalKnight repository with the label `afl-spec`.*
