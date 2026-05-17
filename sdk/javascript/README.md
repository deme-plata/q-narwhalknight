# @q-narwhalknight/paas-sdk

Official JavaScript/TypeScript SDK for Q-NarwhalKnight Privacy-as-a-Service

## Installation

```bash
npm install @q-narwhalknight/paas-sdk ethers @solana/web3.js
```

## Quick Start - Ethereum

```javascript
const { QNarwhalKnightPaaSClient, EthereumWallet, PrivacyLevel } = require('@q-narwhalknight/paas-sdk');

// Initialize client
const client = new QNarwhalKnightPaaSClient({
  apiKey: 'your_api_key',
  baseUrl: 'https://quillon.xyz'
});

// Create wallet
const wallet = new EthereumWallet({
  privateKey: 'your_private_key',
  rpcUrl: 'https://eth-mainnet.g.alchemy.com/v2/your-api-key'
});

// Mix transaction
const result = await client.mixTransaction({
  to: '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb',
  value: '1000000000000000000', // 1 ETH
  privacyLevel: PrivacyLevel.MAXIMUM
});
```

## Quick Start - Solana

```javascript
const { SolanaPaaSClient, SolanaWallet } = require('@q-narwhalknight/paas-sdk');

// Initialize client
const client = new SolanaPaaSClient({
  apiKey: 'your_api_key',
  baseUrl: 'https://quillon.xyz'
});

// Create wallet
const wallet = new SolanaWallet({
  privateKeyBase58: 'your_base58_private_key',
  rpcUrl: 'https://api.mainnet-beta.solana.com'
});

// Transfer with privacy
const result = await client.mixTransfer({
  recipient: 'recipient_pubkey',
  amount: 1_000_000_000, // 1 SOL
  privacyLevel: 'maximum'
});
```

## Features

- ✅ Production-ready transaction construction
- ✅ Automatic nonce/blockhash management
- ✅ Client-side signing (keys never leave your machine)
- ✅ Retry logic with exponential backoff
- ✅ MEV protection for Ethereum
- ✅ Priority fees for Solana
- ✅ Multi-chain support

## Documentation

Full documentation: https://quillon.xyz/docs

## License

MIT
