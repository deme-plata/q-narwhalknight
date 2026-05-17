# Q-NarwhalKnight Privacy-as-a-Service SDKs

Production-ready SDKs for integrating privacy features across multiple blockchains.

## 🚀 Quick Start

### Python (Bitcoin, Litecoin, Dogecoin)
```bash
pip install q-narwhalknight-paas python-bitcoinlib coincurve

# Example
from q_paas import QNarwhalKnightPaaSClient, BitcoinWallet

wallet = BitcoinWallet(private_key_wif)
client = QNarwhalKnightPaaSClient(api_key)
result = client.mix_bitcoin_transaction(signed_tx)
```

### JavaScript (Ethereum, Solana, Polygon, Avalanche)
```bash
npm install q-narwhalknight-paas ethers @solana/web3.js axios uuid

// Ethereum example
const { QNarwhalKnightPaaSClient, EthereumWallet } = require('./q_paas_ethereum_production');

const wallet = new EthereumWallet(privateKey, rpcUrl);
const client = new QNarwhalKnightPaaSClient(apiKey);
const result = await client.privateUniswapSwap(wallet, tokenIn, tokenOut, amount);
```

## 📦 Available SDKs

### Production SDKs (v4.0)
- ✅ **Bitcoin** - `python/q_paas_bitcoin_production.py` (500+ lines)
- ✅ **Ethereum** - `javascript/q_paas_ethereum_production.js` (450+ lines)
- ✅ **Solana** - `javascript/q_paas_solana_production.js` (420+ lines)

### Supported Chains via SDK Adaptation
- **Litecoin** - Use Bitcoin SDK with Litecoin network parameters
- **Dogecoin** - Use Bitcoin SDK with Dogecoin network parameters
- **Polygon** - Use Ethereum SDK with Polygon RPC (chainId: 137)
- **Avalanche** - Use Ethereum SDK with Avalanche C-Chain RPC (chainId: 43114)
- **SUI** - Adapt Solana SDK architecture for `@mysten/sui.js`

## 🔐 Security Model

**All SDKs follow these principles:**

1. **CLIENT-SIDE SIGNING**
   - Private keys NEVER leave your machine
   - Transactions signed LOCALLY
   - API only receives SIGNED transactions

2. **IDEMPOTENCY**
   - Automatic UUID generation
   - Safe retries without duplicate submissions

3. **RETRY LOGIC**
   - Exponential backoff (2^n seconds)
   - Automatic retry on 429, 500, 502, 503, 504

## 📚 Documentation

**Full Developer Guide**: [PAAS_DEVELOPER_INTEGRATION_GUIDE.pdf](../PAAS_DEVELOPER_INTEGRATION_GUIDE.pdf)

**Individual SDK READMEs**:
- [Python SDK](python/README.md)
- JavaScript SDKs (see individual files)

## 🎯 Features

### Bitcoin SDK
- Real UTXO selection (greedy algorithm)
- Base58 encoding/decoding
- WIF private key import
- Complete transaction construction
- Bitcoin protocol serialization
- Comprehensive error handling

### Ethereum SDK
- Nonce management
- Gas estimation (EIP-1559)
- MEV protection (Flashbots)
- ERC-20 token support
- Private Uniswap swaps
- Retry logic with axios

### Solana SDK
- Recent blockhash handling
- Priority fee calculation
- Compute budget instructions
- SPL token support
- Transaction retry with confirmation
- Base58 key decoding

## 🛠️ Development

### Running Examples

**Bitcoin**:
```bash
export QNKPAAS_API_KEY="your_key"
export BTC_PRIVATE_KEY_WIF="your_wif"
export BTC_RPC_URL="http://localhost:8332"

python python/q_paas_bitcoin_production.py
```

**Ethereum**:
```bash
export QNKPAAS_API_KEY="your_key"
export ETH_PRIVATE_KEY="0x..."
export ETH_RPC_URL="https://mainnet.infura.io/v3/YOUR_KEY"

node javascript/q_paas_ethereum_production.js
```

**Solana**:
```bash
export QNKPAAS_API_KEY="your_key"
export SOLANA_SECRET_KEY="base58_key"
export SOLANA_RPC_URL="https://api.mainnet-beta.solana.com"

node javascript/q_paas_solana_production.js
```

## 📊 Production Checklist

Before using in production:

- [ ] Test on testnet first
- [ ] Use environment variables for keys (never hardcode)
- [ ] Implement proper error handling
- [ ] Set up monitoring and alerting
- [ ] Review transaction fees and gas prices
- [ ] Test retry logic under network failures
- [ ] Verify idempotency key generation
- [ ] Audit all code paths

## 🐛 Troubleshooting

**Common Issues**:

1. **"No such file or directory"** - Make sure you're in the sdk directory
2. **"Module not found"** - Install dependencies: `pip install ...` or `npm install ...`
3. **"Connection refused"** - Check RPC URL and API endpoint
4. **"Insufficient funds"** - Ensure wallet has enough balance + fees
5. **"Invalid signature"** - Verify private key format (WIF for Bitcoin, hex for Ethereum)

## 📞 Support

- **Documentation**: https://quillon.xyz/docs
- **Email**: developers@q-narwhalknight.io
- **Discord**: https://discord.gg/q-narwhalknight
- **GitHub Issues**: github.com/q-narwhalknight/sdk/issues

## 📄 License

See main repository LICENSE file.

## 🌟 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## ⚠️ Disclaimer

**PRODUCTION CODE - USE AT YOUR OWN RISK**

These SDKs handle real cryptocurrency transactions. Always:
- Test thoroughly on testnet first
- Review all code before use
- Keep private keys secure
- Understand the risks involved
- Start with small amounts

**We are not responsible for lost funds due to misuse.**

---

**Version**: 4.0 Production Ready
**Last Updated**: October 22, 2025
**Status**: ✅ Production
