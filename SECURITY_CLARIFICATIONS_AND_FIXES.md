# Security Clarifications and Critical Fixes

**Date**: 2025-10-22
**Priority**: CRITICAL
**Status**: Immediate Action Required

---

## Executive Summary

This document addresses **critical security concerns** identified in the Developer Integration Guide, particularly around private key handling, over-simplification of cryptographic operations, and the trust model. We provide corrected examples and clarify our actual security architecture.

---

## 1. CRITICAL FIX: Private Key Handling

### ❌ **DANGEROUS PATTERN (REMOVED FROM GUIDE)**

```python
# WRONG - DO NOT DO THIS
response = requests.post(
    f"{BASE_URL}/api/v1/privacy/zk-stark/prove",
    json={
        "private_inputs": {
            "wallet_address": wallet_address,
            "private_key": private_key  # ❌ NEVER SEND PRIVATE KEYS!
        }
    }
)
```

**Why This Is Wrong**: The comment says "never sent over network" but the code shows it in a POST request. This is **dangerously misleading**.

### ✅ **CORRECT PATTERN: Client-Side Proving**

**Our Actual Architecture**: Private keys NEVER leave the user's device.

```python
from q_narwhalknight import ZKProver  # Local library, no network calls

def generate_balance_proof_SECURE(
    wallet_address: str,
    minimum_balance_wei: int,
    private_key: str  # Stays on YOUR machine
) -> dict:
    """
    Generate ZK-STARK proof CLIENT-SIDE.
    Private key never transmitted to Q-NarwhalKnight servers.
    """

    # Step 1: Initialize LOCAL prover (runs on your machine)
    prover = ZKProver.local_prover(
        circuit_type="balance_threshold",
        witness_data={
            "wallet_address": wallet_address,
            "private_key": private_key,  # Used locally only
            "balance": get_balance_from_rpc(wallet_address)  # You fetch
        }
    )

    # Step 2: Generate proof locally (CPU/GPU intensive)
    # This takes 30 seconds on RTX 3080, runs entirely on your hardware
    proof = prover.generate_proof(
        public_inputs={
            "minimum_balance_wei": minimum_balance_wei,
            "block_number": get_latest_block()
        }
    )

    # Step 3: ONLY send the proof (not private key!)
    response = requests.post(
        f"{BASE_URL}/api/v1/privacy/zk-stark/verify",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "circuit_type": "balance_threshold",
            "proof": proof.to_hex(),  # Just the ZK proof
            "public_inputs": {
                "minimum_balance_wei": minimum_balance_wei,
                "block_number": proof.block_number
            }
        }
    )

    # We verify the proof, but we NEVER see your private key or balance
    return response.json()

# Example usage
proof_result = generate_balance_proof_SECURE(
    wallet_address="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb1",
    minimum_balance_wei=10 * 10**18,  # 10 ETH
    private_key="0x..."  # Never leaves this function
)

print(f"Proof verified: {proof_result['data']['verified']}")
# Server knows: "This wallet has ≥10 ETH"
# Server does NOT know: Exact balance, private key, transaction history
```

**What Changed**:
1. ✅ Private key used in **local prover** (your machine)
2. ✅ Only the **proof** is sent over network
3. ✅ Server **cannot** learn your private key or exact balance
4. ✅ Zero-knowledge property is actually maintained

---

## 2. CRITICAL FIX: Transaction Signing

### ❌ **DANGEROUS PATTERN (REMOVED)**

```python
# WRONG - Implies we sign transactions for you
result = client.bitcoin.mix(
    from_address="bc1q...",
    to_address="bc1q...",
    amount_satoshis=10_000_000,
    private_key="L5oLk..."  # ❌ NEVER GIVE US YOUR KEYS!
)
```

### ✅ **CORRECT PATTERN: You Sign, We Coordinate**

```python
from bitcoin import SelectParams, Transaction
from bitcoin.wallet import CBitcoinSecret

def mix_bitcoin_transaction_SECURE(
    from_address: str,
    to_address: str,
    amount_satoshis: int,
    private_key_wif: str  # Stays on YOUR machine
) -> dict:
    """
    Correct mixing flow:
    1. YOU build and sign the transaction locally
    2. YOU send only the SIGNED transaction (not the key!)
    3. We coordinate mixing with other signed transactions
    4. We broadcast the final mixed result
    """

    SelectParams('mainnet')

    # Step 1: Build transaction LOCALLY (your machine)
    txin = get_utxos_for_address(from_address)  # Your RPC node
    txout = create_output(to_address, amount_satoshis)

    tx = Transaction([txin], [txout])

    # Step 2: Sign transaction LOCALLY (private key never leaves)
    seckey = CBitcoinSecret(private_key_wif)
    sighash = SignatureHash(...)
    sig = seckey.sign(sighash) + bytes([SIGHASH_ALL])

    txin.scriptSig = CScript([sig, seckey.pub])

    # Step 3: Serialize SIGNED transaction
    signed_tx_hex = tx.serialize().hex()

    # Step 4: Send ONLY the signed transaction (not the key!)
    response = requests.post(
        f"{BASE_URL}/api/v1/privacy/mix/submit",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Idempotency-Key": str(uuid.uuid4())
        },
        json={
            "chain": "bitcoin",
            "signed_transaction_hex": signed_tx_hex,  # ✅ No private key!
            "privacy_level": "maximum",
            "recipient_address": to_address,
            "options": {
                "stealth_address": True,
                "tor_relay": True,
                "timing_jitter": 120
            }
        }
    )

    return response.json()

# Example usage
result = mix_bitcoin_transaction_SECURE(
    from_address="bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq",
    to_address="bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",
    amount_satoshis=10_000_000,
    private_key_wif="L5oLkpV3aqBjhki6LmvChTCV6odsp4SXM6FfU2gpqgGx8aYLYUY1"
)

print(f"Transaction mixed: {result['data']['transaction_id']}")
```

**Key Security Properties**:
1. ✅ **You sign** locally with your private key
2. ✅ **We receive** only the signed transaction
3. ✅ **We coordinate** mixing with other pre-signed transactions
4. ✅ **We cannot** steal your funds (transaction is already signed by you)
5. ✅ **We cannot** learn your private key (we never see it)

---

## 3. Trust Model Clarification

### What We Can Do (As a Service Provider)

| **Action** | **Can We Do It?** | **Explanation** |
|------------|-------------------|-----------------|
| See your private key | ❌ NO | Never transmitted, stays on your device |
| See your exact balance | ❌ NO | ZK proofs hide witness data |
| See which transaction is yours in a mix | ⚠️ YES (Phase 0) | We coordinate the mix, so we know inputs ↔ outputs |
| Censor your transaction | ✅ YES | We're a centralized service (Phase 0) |
| Steal your funds | ❌ NO | You sign transactions, we can't modify them |
| Deanonymize your IP | ⚠️ YES (without Tor) | Use Tor relay option to prevent this |
| Comply with lawful disclosure | ✅ YES | Court orders in compliance mode |

### What Changes in Federation (Phase 1, Q3 2025)

```python
# Phase 1: User picks 3 federation members
client = PaaSClient(
    api_key="your_key",
    federation_mode=True,
    selected_nodes=[
        "qnk-us-node",        # Q-NarwhalKnight (US, compliant)
        "swiss-privacy-node",  # Independent Swiss operator
        "arweave-dao-node"     # Community-run DAO node
    ],
    threshold=2  # Any 2 of 3 can coordinate mixing
)

# Now: ALL 3 nodes must collude to deanonymize you
# If one is compromised, the other 2 still protect privacy
```

**Trust Model**:
- **Phase 0 (Now)**: Trust Q-NarwhalKnight (centralized)
- **Phase 1 (2025)**: Trust threshold of N federation members (e.g., 2-of-3)
- **Phase 2 (2026)**: Trust cryptographic proofs, not operators (DAO)

---

## 4. Corrected "Magic Abstraction" - Transparent SDK

### ❌ **Over-Simplified (Hides Security Decisions)**

```python
# Too simple - what's actually happening?
result = client.bitcoin.mix(from_address, to_address, amount)
```

### ✅ **Explicit Security Parameters**

```python
from q_narwhalknight import PaaSClient, PrivacyLevel, SecurityOptions

client = PaaSClient(
    api_key="your_key",
    security_options=SecurityOptions(
        verify_tls_certificates=True,  # Prevent MITM
        pin_certificate_fingerprint="sha256:ABC123...",  # Certificate pinning
        require_reproducible_build=True,  # Verify our binaries match open source
        audit_mode=True  # Log all requests locally for your review
    )
)

result = client.bitcoin.mix(
    from_address="bc1q...",
    to_address="bc1q...",
    amount_satoshis=10_000_000,
    privacy_options={
        # EXPLICIT privacy/performance trade-offs
        "privacy_level": PrivacyLevel.MAXIMUM,  # epsilon < 0.7
        "anonymity_set_size": 64,  # Wait for 64 participants
        "timing_jitter_seconds": 180,  # Random delay 0-180s
        "tor_relay": True,  # Route via Tor (adds 150ms latency)
        "require_stealth_address": True,  # One-time recipient address

        # EXPLICIT security requirements
        "allow_amount_rounding": True,  # Round to bucket (e.g., 1.0-5.0 BTC)
        "max_acceptable_epsilon": 0.7,  # Reject if privacy worse than this
        "require_proof_of_mixing": True,  # We provide Merkle proof

        # EXPLICIT compliance settings
        "enable_kyc_screening": False,  # Disable for max privacy (if allowed)
        "jurisdiction_routing": "privacy_friendly",  # Route via Swiss/Cayman nodes
    },
    signing_callback=local_sign_transaction  # YOU control the signing
)

# SDK shows you EXACTLY what's happening
print(f"Privacy achieved: epsilon = {result.privacy_epsilon}")
print(f"Anonymity set: {result.anonymity_set_size} participants")
print(f"Mixing proof: {result.merkle_proof}")  # Verify we didn't cheat

# Verify the Merkle proof yourself (don't trust us)
assert client.verify_mixing_proof(
    transaction_id=result.transaction_id,
    merkle_proof=result.merkle_proof,
    merkle_root=result.on_chain_commitment  # Published to Ethereum
)
```

**What Changed**:
1. ✅ **Explicit security options** (no hidden defaults)
2. ✅ **You control signing** (callback function)
3. ✅ **Provable mixing** (Merkle proof you can verify)
4. ✅ **Transparent trade-offs** (privacy vs. latency)

---

## 5. QUG Token Lock-In: Mitigation

### The Concern

"Proprietary QUG token creates vendor lock-in and accounting complexity."

### Our Response: Multi-Currency Support (Q2 2025)

```python
# Pay with multiple currencies (not just QUG)
client = PaaSClient(api_key="your_key")

# Option 1: Pay with BTC (on-chain)
result = client.bitcoin.mix(
    ...,
    payment_method="bitcoin",
    payment_address="bc1q..."  # Send BTC directly
)

# Option 2: Pay with stablecoins (USDC, USDT)
result = client.ethereum.mix(
    ...,
    payment_method="usdc",
    payment_token="0xA0b8..."  # USDC contract
)

# Option 3: Pay with fiat (Stripe)
result = client.bitcoin.mix(
    ...,
    payment_method="stripe",
    stripe_payment_method_id="pm_1abc..."
)

# Option 4: Pay with QUG (native token, 10% discount)
result = client.bitcoin.mix(
    ...,
    payment_method="qug",
    payment_source="balance"  # Use prepaid QUG balance
)
```

**Benefits**:
- ✅ No forced QUG exposure
- ✅ Pay in your preferred currency
- ✅ Enterprise accounting simplified (USDC/fiat)
- ✅ QUG remains optional (but offers discount)

---

## 6. Security Best Practices (Updated)

### Mandatory Checklist

- [ ] **Never send private keys to ANY API** (including ours)
- [ ] **Always sign transactions client-side** (use local libraries)
- [ ] **Verify Merkle proofs** (don't trust our mixing claims)
- [ ] **Enable Tor relay** (hide IP from us and blockchain observers)
- [ ] **Pin TLS certificates** (prevent MITM attacks)
- [ ] **Review audit logs** (SDK provides local logging)
- [ ] **Understand trust model** (centralized Phase 0 → federated Phase 1 → DAO Phase 2)
- [ ] **Set explicit privacy parameters** (don't rely on defaults)
- [ ] **Use idempotency keys** (prevent double-charges on retries)
- [ ] **Monitor rate limits** (check X-RateLimit headers)

### Code Template (Secure by Default)

```python
from q_narwhalknight import PaaSClient, SecurityOptions, PrivacyLevel

# Initialize with MAXIMUM security settings
client = PaaSClient(
    api_key=os.getenv("PAAS_API_KEY"),  # Never hardcode!
    security_options=SecurityOptions(
        verify_tls_certificates=True,
        pin_certificate_fingerprint=os.getenv("PAAS_CERT_FP"),
        require_reproducible_build=True,
        audit_mode=True,
        local_audit_log="/var/log/paas-audit.log"
    )
)

# Define YOUR signing function (private key stays local)
def sign_transaction_locally(unsigned_tx):
    # Load private key from hardware wallet / secure enclave
    private_key = load_from_hardware_wallet()

    # Sign locally (no network calls in this function)
    signed_tx = private_key.sign(unsigned_tx)

    # Clear private key from memory
    del private_key

    return signed_tx

# Mix transaction with EXPLICIT security parameters
result = client.bitcoin.mix(
    from_address="bc1q...",
    to_address="bc1q...",
    amount_satoshis=10_000_000,
    privacy_options={
        "privacy_level": PrivacyLevel.MAXIMUM,
        "tor_relay": True,
        "require_proof_of_mixing": True,
        "max_acceptable_epsilon": 0.7
    },
    signing_callback=sign_transaction_locally  # YOU control keys
)

# Verify we didn't cheat (don't trust, verify!)
assert client.verify_mixing_proof(
    transaction_id=result.transaction_id,
    merkle_proof=result.merkle_proof,
    on_chain_commitment=result.on_chain_commitment
)

print(f"✅ Transaction mixed securely!")
print(f"Privacy: epsilon={result.privacy_epsilon} (lower is better)")
print(f"Anonymity: {result.anonymity_set_size} participants")
```

---

## 7. Updated Trust Assumptions

### What You Must Trust (Phase 0)

1. **We won't censor your transaction** (centralized coordinator)
2. **We won't collude with adversaries** to correlate inputs/outputs
3. **Our cryptography is implemented correctly** (third-party audits help)
4. **We'll honor privacy parameters** you specify

### What You DON'T Need to Trust

1. ❌ **We can't steal your funds** (you sign transactions)
2. ❌ **We can't see your private keys** (never transmitted)
3. ❌ **We can't change transaction amounts** (already signed by you)
4. ❌ **We can't fake mixing proofs** (verifiable on-chain commitments)

### Verifiable Claims (Don't Trust, Verify!)

```python
# Verify on-chain commitment (published to Ethereum)
commitment = client.get_mixing_pool_commitment(epoch=12345)

# Check Ethereum smart contract
web3 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/..."))
contract = web3.eth.contract(
    address="0x...",  # PaaSCommitment contract
    abi=[...]
)

# Read commitment from blockchain
on_chain_hash = contract.functions.poolStateHashes(12345).call()

# Verify it matches what we claimed
assert on_chain_hash == commitment.merkle_root

# Now verify YOUR transaction was included
assert MerkleProof.verify(
    proof=result.merkle_proof,
    root=on_chain_hash,
    leaf=hash(result.transaction_id)
)

print("✅ Cryptographically verified: We didn't censor your transaction!")
```

---

## 8. Comparison: Our Model vs. Alternatives

| **Aspect** | **Q-NarwhalKnight (Phase 0)** | **Tornado Cash** | **Wasabi Wallet** |
|------------|-------------------------------|------------------|-------------------|
| Private key custody | ❌ You keep (never sent) | ❌ You keep | ❌ You keep |
| Transaction signing | ✅ Client-side | ✅ Client-side | ✅ Client-side |
| Mixing coordination | ⚠️ Centralized (us) | ✅ Smart contract | ⚠️ Central coordinator |
| Censorship resistance | ❌ We can censor | ✅ Permissionless | ❌ Coordinator can censor |
| Regulatory compliance | ✅ KYT/AML built-in | ❌ None (sanctioned) | ⚠️ Optional |
| Multi-chain | ✅ BTC/ETH/SOL/... | ❌ Ethereum only | ❌ Bitcoin only |
| Quantum-resistant | ✅ Dilithium5/Kyber | ❌ ECDSA vulnerable | ❌ ECDSA vulnerable |
| Verifiable mixing | ✅ On-chain commitments | ✅ Smart contract | ❌ Trust coordinator |
| Decentralization | ⚠️ Roadmap (2025-2026) | ✅ Already decentralized | ❌ Centralized |

**Our Honest Assessment**:
- **Privacy**: Tornado Cash > Q-NarwhalKnight Phase 0 > Wasabi
- **Compliance**: Q-NarwhalKnight ≫ Wasabi > Tornado Cash
- **Censorship Resistance**: Tornado Cash > Q-NarwhalKnight Phase 2 > Phase 0 > Wasabi
- **Usability**: Q-NarwhalKnight ≫ Wasabi > Tornado Cash
- **Legal Risk**: Tornado Cash (sanctioned) > Wasabi (scrutinized) > Q-NarwhalKnight (compliant)

---

## 9. Immediate Actions

### For Developers

1. **Update Integration Guide** (CRITICAL):
   - Remove all examples showing private keys in API requests
   - Add client-side signing examples
   - Clarify trust model in every code example
   - Add security warnings

2. **Update SDK Documentation**:
   - Make security options EXPLICIT (not hidden in defaults)
   - Add "Security Considerations" section to every function
   - Provide secure-by-default templates

3. **Add Verification Tools**:
   - Merkle proof verification library
   - On-chain commitment checker
   - Reproducible build verification script

### For Q-NarwhalKnight Engineering

1. **Open-Source Critical Components (Q2 2025)**:
   - ZK-STARK prover/verifier (already audited)
   - Client-side signing libraries
   - Merkle proof generation

2. **Publish Security Architecture Document**:
   - Detailed threat model
   - What we can/can't see
   - How to verify our claims

3. **Implement On-Chain Commitments**:
   - Deploy Ethereum smart contract (testnet already live)
   - Publish Merkle root every epoch
   - Provide verification tools

---

## 10. Conclusion

**We made critical errors in the initial Developer Guide** by showing examples that could be misinterpreted as sending private keys to our API. This is **unacceptable** for a security-focused service.

**Corrected Security Model**:
1. ✅ Private keys **NEVER** leave your device
2. ✅ You **always sign** transactions client-side
3. ✅ We **coordinate mixing** of pre-signed transactions
4. ✅ You **can verify** our claims via on-chain commitments
5. ⚠️ You **must trust** us not to censor (Phase 0) → Fixed in Phase 1 (federation)

**We acknowledge the trust trade-offs** and are working to minimize them through:
- Verifiable on-chain commitments
- Third-party audits
- Open-source roadmap
- Federation (2025) and DAO (2026) transitions

**Thank you** for identifying these critical issues. Security is our top priority, and we are committed to transparency about both our capabilities and limitations.

---

**Questions? Security Concerns?** Email: security@q-narwhalknight.io

**Bug Bounty**: \$50,000 for critical findings

**Public Security Roadmap**: https://security.q-narwhalknight.io

---

*This document supersedes conflicting information in other materials. Last updated: 2025-10-22*
