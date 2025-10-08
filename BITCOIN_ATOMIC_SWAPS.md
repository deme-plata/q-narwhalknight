# Bitcoin Atomic Swaps - Trustless BTC ↔ QNKUSD Exchange

## Overview

The Quillon Bank implements **atomic swaps** using Hash Time-Locked Contracts (HTLCs) to enable trustless exchanges between Bitcoin and QNKUSD. No bridges, no custodians, no trust required.

## Why Atomic Swaps Instead of Bridges?

### Problems with Bridges:
- **Centralized**: Require trusted third parties
- **Hackable**: Single points of failure (see: Ronin, Poly Network, Wormhole hacks)
- **Custodial**: Bridges hold your assets
- **Vulnerable**: Multisig compromises, smart contract bugs

### Atomic Swap Advantages:
- ✅ **Trustless**: No third party can steal funds
- ✅ **Non-custodial**: You control your keys
- ✅ **Atomic**: Either both parties get assets or both get refunds
- ✅ **Secure**: Uses Bitcoin's native script capabilities
- ✅ **Time-bounded**: Funds cannot be locked forever

## How It Works

### Hash Time-Locked Contracts (HTLCs)

HTLCs use two locks:
1. **Hash Lock**: Requires revealing a secret (preimage) to unlock
2. **Time Lock**: Allows refund after timeout expires

### Atomic Swap Protocol

```
User (wants QNKUSD)        Quillon Bank (wants BTC)
─────────────────────────────────────────────────────────

1. Generate secret + hash
   [secret: random 32 bytes]
   [hash: SHA256(secret)]

2. Create swap proposal ────────────────────────►
   - BTC amount: 0.5 BTC
   - QNKUSD amount: 10,000
   - Hash lock: SHA256(secret)
   - Timelock: 24 hours

3. Receive HTLC address ◄────────────────────────
   - Bitcoin P2WSH address
   - Locks BTC with hash + time

4. Send BTC to HTLC address
   [0.5 BTC locked on Bitcoin]

5. Bank detects BTC lock ────────────────────────►

6. Bank locks QNKUSD ◄────────────────────────────
   [10,000 QNKUSD locked on Q-Network]

7. Reveal secret to claim QNKUSD
   [secret revealed to Q-Network]

8. Bank sees secret, claims BTC ◄────────────────
   [Bank uses secret to claim BTC]

9. ✅ Swap complete!
   User has: 10,000 QNKUSD
   Bank has: 0.5 BTC (as collateral)
```

### Timeout Refund Path

If swap fails:
- **User timeout (24 hours)**: User can reclaim their BTC
- **Bank timeout (12 hours)**: Bank can reclaim their QNKUSD
- **Safety**: No party can lose funds permanently

## Bitcoin HTLC Script

The Bitcoin script implements conditional logic:

```bitcoin
OP_IF
  OP_SHA256 <hash_lock> OP_EQUALVERIFY
  <bank_pubkey> OP_CHECKSIG
OP_ELSE
  <timelock> OP_CHECKLOCKTIMEVERIFY OP_DROP
  <user_pubkey> OP_CHECKSIG
OP_ENDIF
```

### Two Spending Paths:

**Path 1 (Bank claims with secret):**
- Provide `<secret>` that hashes to `<hash_lock>`
- Sign with bank's private key
- Can claim immediately after user reveals secret

**Path 2 (User refund after timeout):**
- Wait for `<timelock>` block height
- Sign with user's private key
- Can only claim after timeout

## API Usage

### 1. Create Atomic Swap

```bash
curl -X POST http://localhost:8090/api/quillon-bank/atomic-swap/create \
  -H "Content-Type: application/json" \
  -d '{
    "direction": "BtcToQnkusd",
    "btc_amount": 50000000,
    "qnkusd_amount": 10000000000000000,
    "user_address": "qnk1abc123...",
    "user_btc_pubkey": "02..."
  }'
```

Response:
```json
{
  "swap_id": "swap_abc123...",
  "htlc_address": "bc1q...",
  "hash_lock": "a1b2c3...",
  "timelock_btc": 800144,
  "timelock_qnk": "2025-10-01T12:00:00Z",
  "status": "proposed"
}
```

### 2. Send BTC to HTLC Address

```bash
# From your Bitcoin wallet
bitcoin-cli sendtoaddress bc1q... 0.5
```

### 3. Monitor Swap Status

```bash
curl http://localhost:8090/api/quillon-bank/atomic-swap/status/swap_abc123
```

### 4. Claim QNKUSD (reveals secret)

```bash
curl -X POST http://localhost:8090/api/quillon-bank/atomic-swap/claim \
  -H "Content-Type: application/json" \
  -d '{
    "swap_id": "swap_abc123...",
    "secret": "0x1234..."
  }'
```

### 5. Bank Claims BTC Automatically

The bank monitors the Q-Network and automatically claims BTC once the secret is revealed.

## Security Guarantees

### Atomicity
- **Either**: Both parties get their assets
- **Or**: Both parties get refunds
- **Never**: One party gets assets while the other doesn't

### Non-Custodial
- User controls their Bitcoin private keys
- Bank controls their Q-Network private keys
- No third party ever holds funds

### Time-Bounded
- BTC refund available after 24 hours
- QNKUSD refund available after 12 hours
- No permanent fund locking

### Trustless
- Bitcoin scripting enforces rules
- Q-Network consensus enforces rules
- Mathematics, not trust

## Integration with Bitcoin Node

### Requirements
- Full Bitcoin node (bitcoind)
- RPC access enabled
- Sufficient confirmations (3+ blocks)

### Node Configuration

```ini
# bitcoin.conf
server=1
rpcuser=rpcuser
rpcpassword=rpcpassword
rpcallowip=127.0.0.1
txindex=1  # Required for HTLC monitoring
```

### Monitoring HTLCs

The atomic swap manager:
1. Connects to Bitcoin node via RPC
2. Monitors mempool for HTLC funding transactions
3. Waits for 3 confirmations
4. Locks QNKUSD after BTC confirmation
5. Claims BTC after secret revelation

## Advantages Over Traditional Collateral

### Traditional Approach (PROBLEM):
```
User: "I have 0.5 BTC"
Bank: "OK, I trust you, here's QNKUSD"
User: [Never sends BTC] 💰🏃
```

### Atomic Swap Approach (SOLUTION):
```
User: [Locks 0.5 BTC in HTLC]
Bank: [Verifies on Bitcoin blockchain]
Bank: [Locks QNKUSD in HTLC]
User: [Reveals secret, gets QNKUSD]
Bank: [Uses secret to claim BTC]
```

## Production Deployment

### Phase 1: Testnet (Current)
- Bitcoin testnet integration
- Testing atomic swap protocol
- UI/UX refinement

### Phase 2: Mainnet Launch
- Bitcoin mainnet integration
- Security audits complete
- Insurance fund established
- Multi-signature bank keys
- Hardware Security Module (HSM) integration

### Phase 3: Scale
- Support for Lightning Network swaps
- Cross-chain atomic swaps (Ethereum, Litecoin)
- Automated market making
- Liquidity pools

## Technical Implementation

### Bitcoin Bridge: `q-bitcoin-bridge/src/atomic_swap.rs`
- HTLC script generation
- Bitcoin transaction monitoring
- Timelock enforcement
- Secret revelation handling

### Quillon Bank Integration: `q-quillon-bank/src/atomic_swap_integration.rs`
- Swap proposal creation
- QNKUSD minting after BTC lock
- Oracle price verification
- Refund handling

### API Endpoints: TBD
- REST API for swap creation
- WebSocket for real-time status
- Webhook callbacks for events

## Security Considerations

### Attack Vectors Prevented:

1. **User Doesn't Send BTC**
   - ❌ Bank doesn't lock QNKUSD until BTC confirmed on-chain

2. **Bank Doesn't Lock QNKUSD**
   - ❌ User can refund BTC after timeout

3. **User Doesn't Reveal Secret**
   - ❌ Both parties refund after timeout

4. **Bank Doesn't Claim BTC**
   - ❌ Bank's own problem, user already has QNKUSD

5. **Bitcoin Reorg Attack**
   - ✅ Mitigated: Wait for 6+ confirmations for large swaps

6. **Front-Running**
   - ✅ Mitigated: Secret only revealed when user claims

### Remaining Risks:

1. **Bitcoin Node Compromise**
   - Mitigation: Use multiple trusted Bitcoin nodes

2. **Q-Network Consensus Failure**
   - Mitigation: Robust Byzantine fault tolerance

3. **Key Management**
   - Mitigation: HSM for bank keys, user controls own keys

## Comparison with Other Solutions

| Feature | Atomic Swaps | Bridges | Wrapped Tokens | Centralized Exchange |
|---------|--------------|---------|----------------|----------------------|
| Trustless | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Non-custodial | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Atomic | ✅ Yes | ❌ No | ❌ No | ✅ Yes (within CEX) |
| On-chain | ✅ Yes | Hybrid | ✅ Yes | ❌ No |
| KYC Required | ❌ No | Varies | ❌ No | ✅ Yes |
| Hack Risk | Very Low | High | Medium | High |

## Future Enhancements

### Lightning Network Integration
- Instant swaps using Lightning HTLCs
- Sub-second finality
- Microtransaction support

### Submarine Swaps
- On-chain ↔ Lightning atomic swaps
- Unified liquidity

### Cross-Chain Expansion
- Ethereum (ERC-20 ↔ QNKUSD)
- Litecoin (LTC ↔ QNKUSD)
- Monero (XMR ↔ QNKUSD via atomic swaps)

### Automated Market Making
- Liquidity pools for atomic swaps
- Automated pricing
- Yield for liquidity providers

## Conclusion

Atomic swaps represent the gold standard for trustless cross-chain exchanges. By implementing HTLCs on both Bitcoin and Q-NarwhalKnight, we achieve:

- ✅ **True decentralization**: No trusted third parties
- ✅ **Maximum security**: Cryptographic guarantees
- ✅ **User sovereignty**: Non-custodial throughout
- ✅ **Regulatory clarity**: No securities involved

This is how cross-chain exchange should be done.