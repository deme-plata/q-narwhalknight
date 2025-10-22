# Q-NarwhalKnight PaaS Python SDK

Production-ready Python SDK for Privacy-as-a-Service integration.

## Installation

```bash
pip install q-narwhalknight-paas
```

## Quick Start

```python
from q_paas import QNarwhalKnightPaaSClient, PrivacyLevel

# Initialize client
client = QNarwhalKnightPaaSClient(
    api_key="your_api_key",
    base_url="https://api.quillon.xyz"
)

# Mix Bitcoin transaction
result = client.mix_bitcoin_transaction(
    signed_tx_hex="01000000...",
    privacy_level=PrivacyLevel.MAXIMUM
)

print(f"Transaction mixed: {result['data']['transaction_id']}")
print(f"Privacy: epsilon = {result['data']['privacy_epsilon']}")
```

## Production Examples

### Bitcoin Integration
See `q_paas_bitcoin_production.py` for complete production implementation with:
- Real UTXO selection
- Proper transaction construction
- Actual signing (requires `python-bitcoinlib`)
- Complete error handling

### Dependencies

**Required**:
```bash
pip install requests cryptography
```

**For Bitcoin**:
```bash
pip install python-bitcoinlib coincurve
```

**For Ethereum**:
```bash
pip install web3 eth-account
```

**For Solana**:
```bash
pip install solana anchorpy
```

## Security Best Practices

1. **Never hardcode API keys** - Use environment variables
2. **Always test on testnet first** - Before using mainnet
3. **Validate all inputs** - Check addresses, amounts
4. **Handle errors properly** - Implement retry logic
5. **Use idempotency keys** - Prevent duplicate submissions

## Support

- Documentation: https://quillon.xyz/docs
- Email: developers@q-narwhalknight.io
- Discord: https://discord.gg/q-narwhalknight
