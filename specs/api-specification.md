# Q-NarwhalKnight API Specification

## Overview
This document specifies the REST API for Q-NarwhalKnight Phase 0, providing endpoints for wallet management, transaction operations, and chain queries.

## Base URL
```
http://localhost:8080/api/v1
```

## Authentication
Phase 0 uses simple password-based authentication for wallet operations. Future phases will implement more sophisticated authentication mechanisms.

## Response Format
All API responses follow this format:
```json
{
  "success": boolean,
  "data": object | null,
  "error": string | null,
  "timestamp": string (ISO 8601)
}
```

## Endpoints

### Wallet Management

#### Create Wallet
**POST** `/wallets`

Creates a new Ed25519 wallet with optional password protection.

**Request Body:**
```json
{
  "password": "optional_password",
  "mnemonic": "optional_mnemonic_phrase"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "address": "hex_string",
    "public_key": "hex_string",
    "balance": 0,
    "nonce": 0,
    "created_at": "2025-08-31T14:15:22Z"
  }
}
```

#### Get Wallet
**GET** `/wallets/{id}`

Retrieves wallet information by UUID.

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "address": "hex_string",
    "public_key": "hex_string", 
    "balance": 1000,
    "nonce": 5,
    "created_at": "2025-08-31T14:15:22Z"
  }
}
```

#### List Wallets
**GET** `/wallets`

Returns all wallets managed by this node.

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "uuid",
      "address": "hex_string",
      "balance": 1000,
      "nonce": 5
    }
  ]
}
```

#### Sign Transaction
**POST** `/wallets/{id}/sign`

Creates and signs a transaction from the specified wallet.

**Request Body:**
```json
{
  "to": "hex_address",
  "amount": 1000,
  "fee": 10,
  "password": "wallet_password"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "transaction_hash",
    "from": "hex_address",
    "to": "hex_address", 
    "amount": 1000,
    "fee": 10,
    "nonce": 6,
    "signature": "hex_signature",
    "timestamp": "2025-08-31T14:15:22Z"
  }
}
```

### Chain Operations

#### Node Status
**GET** `/status`

Returns current node status and metrics.

**Response:**
```json
{
  "success": true,
  "data": {
    "node_id": "hex_string",
    "current_round": 1234,
    "current_height": 1200,
    "connected_peers": 3,
    "tx_pool_size": 45,
    "is_validator": true,
    "uptime": 86400
  }
}
```

#### Submit Transaction
**POST** `/transactions`

Submits a signed transaction to the network.

**Request Body:**
```json
{
  "transaction": {
    "id": "transaction_hash",
    "from": "hex_address",
    "to": "hex_address",
    "amount": 1000,
    "fee": 10,
    "nonce": 6,
    "signature": "hex_signature",
    "timestamp": "2025-08-31T14:15:22Z"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": "transaction_hash"
}
```

#### Get Transaction Status
**GET** `/transactions/{hash}`

Returns the status of a transaction.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "Confirmed",
    "block_height": 1201,
    "round": 1235
  }
}
```

Possible statuses:
- `"Pending"` - Transaction created but not submitted
- `"InMempool"` - Transaction in mempool awaiting consensus
- `"Confirmed"` - Transaction included in a block
- `"Failed"` - Transaction failed with error message

#### Get Block
**GET** `/blocks/{height}`

Returns transactions in a block at the specified height.

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "transaction_hash",
      "from": "hex_address",
      "to": "hex_address",
      "amount": 1000,
      "fee": 10,
      "nonce": 6,
      "signature": "hex_signature",
      "timestamp": "2025-08-31T14:15:22Z"
    }
  ]
}
```

### Health & Monitoring

#### Health Check
**GET** `/health`

Simple health check endpoint.

**Response:**
```json
{
  "success": true,
  "data": "OK"
}
```

#### Metrics
**GET** `/metrics`

Prometheus-formatted metrics (Phase 0 placeholder).

**Response:**
```
# Q-NarwhalKnight metrics
# Coming soon...
```

## Error Handling

### Error Response Format
```json
{
  "success": false,
  "data": null,
  "error": "Error description",
  "timestamp": "2025-08-31T14:15:22Z"
}
```

### HTTP Status Codes
- **200 OK** - Successful operation
- **400 Bad Request** - Invalid request format
- **404 Not Found** - Resource not found
- **500 Internal Server Error** - Server error

### Common Errors
- `"Wallet not found"` - Specified wallet ID doesn't exist
- `"Invalid transaction hash format"` - Transaction hash is not valid hex
- `"Failed to create wallet: {reason}"` - Wallet creation failed
- `"Failed to sign transaction: {reason}"` - Transaction signing failed
- `"Transaction not found"` - Transaction hash not found
- `"Block not found"` - Block height not found

## Usage Examples

### Create and Use Wallet
```bash
# Create wallet
curl -X POST http://localhost:8080/api/v1/wallets \
  -H "Content-Type: application/json" \
  -d '{"password": "secure123"}'

# Sign transaction
curl -X POST http://localhost:8080/api/v1/wallets/{wallet_id}/sign \
  -H "Content-Type: application/json" \
  -d '{
    "to": "0x1234...",
    "amount": 1000,
    "fee": 10,
    "password": "secure123"
  }'

# Submit transaction
curl -X POST http://localhost:8080/api/v1/transactions \
  -H "Content-Type: application/json" \
  -d '{"transaction": {...}}'
```

## Future Enhancements (Phase 1+)

### Cryptographic Agility
- Multi-codec support for different signature schemes
- Automatic algorithm negotiation
- Key rotation capabilities

### Enhanced Security
- Hardware wallet integration
- Multi-signature support
- Time-locked transactions

### Advanced Features
- Quantum randomness verification
- QKD transport status
- STARK proof verification

## Rate Limiting
Phase 0 has no rate limiting. Future phases will implement:
- Per-IP request limits
- Wallet operation throttling
- VDF-based DoS protection