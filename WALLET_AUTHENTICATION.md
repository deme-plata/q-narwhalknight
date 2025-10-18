# Wallet Authentication & Privacy

## Overview

Q-NarwhalKnight implements signature-based authentication to protect wallet privacy. **No wallet information (balance, transactions, etc.) can be accessed without cryptographic proof of ownership.**

## Security Model

### Before Authentication (INSECURE ❌)
```
GET /api/v1/wallets/qnk7f079101d01afc2f.../balance
→ Returns balance to ANYONE (MAJOR SECURITY VULNERABILITY)
```

### After Authentication (SECURE ✅)
```
GET /api/v1/wallets/qnk7f079101d01afc2f.../balance
X-Wallet-Auth: {"address":"qnk7f07...","timestamp":1234567890,"signature":"a3f2b..."}
→ Verifies signature matches wallet's private key
→ Only wallet owner can access their balance
```

## Authentication Protocol

### 1. Challenge Generation

The challenge is: `SHA3-256(address + timestamp + request_path)`

- **address**: 32-byte wallet address
- **timestamp**: Unix timestamp (prevents replay attacks, max 5 minutes old)
- **request_path**: API endpoint being accessed (e.g., `/api/v1/wallets/qnk.../balance`)

### 2. Signature Creation

The wallet owner signs the challenge with their Ed25519 private key:

```javascript
// JavaScript example
const challenge = SHA3_256(address + timestamp + request_path);
const signature = ed25519.sign(challenge, privateKey);
```

```python
# Python example
challenge = SHA3_256(address + timestamp + request_path)
signature = ed25519.sign(challenge, private_key)
```

```rust
// Rust example
let mut hasher = Sha3_256::new();
hasher.update(&address);
hasher.update(&timestamp.to_le_bytes());
hasher.update(request_path.as_bytes());
let message = hasher.finalize();

let signature = signing_key.sign(&message);
```

### 3. Authentication Header

Include the authentication in the `X-Wallet-Auth` header:

```json
{
  "address": "qnk7f079101d01afc2f...",
  "timestamp": 1234567890,
  "signature": "a3f2b1c9d8e7f6..."
}
```

## Protected Endpoints

### Wallet Balance
```bash
GET /api/v1/wallets/{address}/balance
Headers:
  X-Wallet-Auth: {"address":"...","timestamp":...,"signature":"..."}
```

**Requires**: Signature from the wallet's private key

### Wallet Information
```bash
GET /api/v1/wallets/{id}
Headers:
  X-Wallet-Auth: {"address":"...","timestamp":...,"signature":"..."}
```

**Requires**: Signature from the wallet's private key

### Transaction History
```bash
GET /api/v1/wallets/{address}/transactions
Headers:
  X-Wallet-Auth: {"address":"...","timestamp":...,"signature":"..."}
```

**Requires**: Signature from the wallet's private key

### List Wallets
```bash
GET /api/v1/wallets
Headers:
  X-Wallet-Auth: {"address":"...","timestamp":...,"signature":"..."}
```

**Requires**: Signature from ANY valid wallet (shows only that wallet's info)

## Public Endpoints (No Authentication Required)

- **Faucet**: `POST /api/v1/faucet` - Request test tokens
- **Submit Transaction**: `POST /api/v1/transactions` - Submit signed transactions
- **Node Status**: `GET /api/v1/status` - Get network status
- **Health Check**: `GET /api/v1/health` - Server health

## Error Responses

### Missing Authentication
```json
{
  "success": false,
  "error": "Missing X-Wallet-Auth header. Please sign your request.",
  "timestamp": "2025-10-12T13:00:00Z"
}
```
**HTTP Status**: 401 Unauthorized

### Invalid Signature
```json
{
  "success": false,
  "error": "Signature verification failed. You must sign with the wallet's private key.",
  "timestamp": "2025-10-12T13:00:00Z"
}
```
**HTTP Status**: 401 Unauthorized

### Expired Authentication
```json
{
  "success": false,
  "error": "Authentication expired. Timestamp must be within 5 minutes of current time.",
  "timestamp": "2025-10-12T13:00:00Z"
}
```
**HTTP Status**: 401 Unauthorized

## Implementation Example

### cURL Example
```bash
# 1. Generate timestamp
TIMESTAMP=$(date +%s)

# 2. Generate challenge
CHALLENGE=$(echo -n "${ADDRESS}${TIMESTAMP}/api/v1/wallets/${ADDRESS}/balance" | sha3sum -a 256 | cut -d' ' -f1)

# 3. Sign with private key (using your wallet software)
SIGNATURE=$(your-wallet-cli sign "$CHALLENGE")

# 4. Make authenticated request
curl -X GET "http://localhost:8200/api/v1/wallets/${ADDRESS}/balance" \
  -H "X-Wallet-Auth: {\"address\":\"${ADDRESS}\",\"timestamp\":${TIMESTAMP},\"signature\":\"${SIGNATURE}\"}"
```

### JavaScript Example
```javascript
import { Sha3_256 } from 'crypto-js';
import * as ed25519 from 'ed25519';

async function getWalletBalance(address, privateKey) {
  const timestamp = Math.floor(Date.now() / 1000);
  const path = `/api/v1/wallets/${address}/balance`;

  // Generate challenge
  const challenge = Sha3_256(address + timestamp + path);

  // Sign challenge
  const signature = ed25519.sign(Buffer.from(challenge, 'hex'), privateKey);

  // Make request
  const response = await fetch(`http://localhost:8200${path}`, {
    headers: {
      'X-Wallet-Auth': JSON.stringify({
        address,
        timestamp,
        signature: signature.toString('hex')
      })
    }
  });

  return await response.json();
}
```

## Security Features

### 1. Replay Attack Prevention
- Timestamps must be within 5 minutes of current time
- Old signatures cannot be reused

### 2. Path Binding
- Signatures are tied to specific API endpoints
- A signature for `/balance` cannot be used for `/transactions`

### 3. Address Binding
- Signatures prove ownership of the specific wallet
- Cannot access other wallets' data

### 4. Ed25519 Cryptography
- Industry-standard elliptic curve signatures
- Same keys used for transactions

## Migration Guide

### For Existing Users

1. **Update your client code** to include authentication headers
2. **Generate signatures** using your wallet's private key
3. **Test with a single request** before bulk operations

### For API Developers

1. **Import** `AuthenticatedWallet` extractor
2. **Add to handler** as first parameter
3. **Verify address** matches requested resource

Example:
```rust
pub async fn get_wallet_balance(
    auth: AuthenticatedWallet,  // ✅ Add authentication
    State(state): State<Arc<AppState>>,
    Path(address): Path<String>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    // Verify the authenticated address matches requested address
    let requested_address = parse_address(&address)?;
    if requested_address != auth.address {
        return Err(StatusCode::FORBIDDEN);
    }

    // Now safe to return balance
    // ...
}
```

## Privacy Guarantees

✅ **Your balance is private** - Only you can view it
✅ **Your transactions are private** - Only you can access your history
✅ **Your wallet list is private** - Only you see your wallets
✅ **No data leakage** - Failed authentication returns NO information
✅ **Quantum-ready** - Will support post-quantum signatures in Phase 2

## Future Enhancements

- **Session tokens**: Avoid signing every request (optional convenience)
- **Post-quantum signatures**: Dilithium5 for quantum resistance
- **Hardware wallet support**: Direct integration with Ledger/Trezor
- **Multi-signature wallets**: Require multiple approvals
- **Zero-knowledge proofs**: Prove balance ranges without revealing exact amounts
