# DEX Liquidity Pool Setup Guide

## Problem: "No liquidity pool found for QUG -> QUGUSD"

This error occurs when you try to perform a swap but haven't created a liquidity pool yet. Decentralized exchanges (DEXes) require liquidity providers to supply tokens to enable trading.

## Solution: Create a Liquidity Pool

### Method 1: Using the Setup Script (Easiest)

```bash
# Run the automated setup script
./setup_qug_qugusd_pool.sh qnk<your-wallet-address>

# Example with custom amounts
AMOUNT_QUG=5000000000 AMOUNT_QUGUSD=5000000000 \
  ./setup_qug_qugusd_pool.sh qnkyour-wallet-address-here
```

### Method 2: Using curl Directly

```bash
# Replace YOUR_WALLET_ADDRESS with your actual wallet address
curl -X POST http://localhost:8090/api/v1/liquidity/add \
  -H "Content-Type: application/json" \
  -d '{
    "token0": "QUG",
    "token1": "QUGUSD",
    "amount0": 1000000000,
    "amount1": 1000000000,
    "provider": "qnkYOUR_WALLET_ADDRESS"
  }'
```

### Method 3: Using the Web UI

1. Open the Quantum Wallet GUI: `http://localhost:5173`
2. Navigate to the **DEX** or **Liquidity** section
3. Select **Add Liquidity**
4. Choose tokens: **QUG** and **QUGUSD**
5. Enter amounts (e.g., 10 QUG and 10 QUGUSD)
6. Click **Add Liquidity**

## Understanding Liquidity Pools

### What is a Liquidity Pool?

A liquidity pool is a smart contract that holds reserves of two tokens (e.g., QUG and QUGUSD). When you want to swap QUG for QUGUSD, the pool:
1. Takes your QUG tokens
2. Calculates the exchange rate based on the reserves
3. Gives you QUGUSD tokens in return

### Why Add Liquidity?

As a liquidity provider, you:
- **Enable trading**: Others can swap tokens using your pool
- **Earn fees**: You earn a portion of trading fees (typically 0.3%)
- **Support the ecosystem**: Help bootstrap the Q-NarwhalKnight DEX

### Token Amounts

Amounts are in **micro-units** (8 decimal places):
- `100000000` = 1 QUG (or 1 QUGUSD)
- `1000000000` = 10 QUG (or 10 QUGUSD)
- `10000000000` = 100 QUG (or 100 QUGUSD)

## Checking Pool Status

### List All Pools

```bash
curl http://localhost:8090/api/v1/liquidity/pools | jq
```

### Check Specific Pool

```bash
curl http://localhost:8090/api/v1/liquidity/pools/pool-QUG-QUGUSD-<timestamp> | jq
```

### Get Pool Info via DEX API

```bash
curl http://localhost:8090/api/v1/dex/pools | jq
```

## Performing Swaps After Adding Liquidity

Once your pool is created, you can swap:

```bash
# Swap 1 QUG for QUGUSD
curl -X POST http://localhost:8090/api/v1/swap \
  -H "Content-Type: application/json" \
  -d '{
    "from_token": "QUG",
    "to_token": "QUGUSD",
    "amount": 100000000,
    "wallet": "qnkYOUR_WALLET_ADDRESS"
  }'
```

## Removing Liquidity

To remove liquidity from your pool:

```bash
curl -X POST http://localhost:8090/api/v1/liquidity/remove \
  -H "Content-Type: application/json" \
  -d '{
    "pool_id": "pool-QUG-QUGUSD-<timestamp>",
    "percentage": 100,
    "provider": "qnkYOUR_WALLET_ADDRESS"
  }'
```

**Note**: `percentage` can be 1-100 (remove partial or all liquidity)

## Troubleshooting

### Error: "Insufficient QUG balance"

You need QUG tokens in your wallet. Options:
1. **Mine QUG**: Use the mining feature in the GUI
2. **Receive QUG**: Get tokens from another wallet
3. **Faucet**: Use the testnet faucet (if available)

### Error: "Insufficient token1 balance"

You need QUGUSD tokens. Options:
1. **Mint QUGUSD**: Use the CDP (Collateralized Debt Position) feature
2. **Swap for QUGUSD**: Once pools exist, swap QUG for QUGUSD
3. **Deploy QUGUSD**: The stablecoin contract may need deployment

### Error: "Token symbol not found"

Make sure the token contract is deployed:

```bash
# Check deployed contracts
curl http://localhost:8090/api/v1/vm/contracts | jq
```

### API Server Not Running

Start the API server:

```bash
cd /opt/orobit/shared/q-narwhalknight
timeout 36000 cargo run --bin q-api-server --release
```

## Advanced: Custom Token Pools

You can create pools for any token pair:

```bash
curl -X POST http://localhost:8090/api/v1/liquidity/add \
  -H "Content-Type: application/json" \
  -d '{
    "token0": "MyToken",
    "token1": "QUG",
    "amount0": 1000000000,
    "amount1": 1000000000,
    "provider": "qnkYOUR_WALLET_ADDRESS"
  }'
```

## DEX Integration API

For external DEX integrations, use the `/api/v1/dex/` endpoints:

```bash
# Create pool via DEX API
curl -X POST http://localhost:8090/api/v1/dex/pools/create \
  -H "Content-Type: application/json" \
  -d '{
    "token0": "QUG",
    "token1": "QUGUSD",
    "initial_reserve0": "1000000000",
    "initial_reserve1": "1000000000"
  }'
```

## Next Steps

1. ✅ Create QUG/QUGUSD liquidity pool
2. ✅ Test swapping small amounts
3. ✅ Monitor pool performance
4. ✅ Add more liquidity as needed
5. ✅ Explore creating pools for custom tokens

## Resources

- **API Documentation**: http://localhost:8090/api-docs
- **Liquidity API**: `/api/v1/liquidity/`
- **DEX Integration API**: `/api/v1/dex/`
- **Smart Contracts Guide**: Check `api-docs` app for contract deployment

## Support

If you encounter issues:
1. Check the API server logs: `api-server.log`
2. Verify wallet balance: `curl http://localhost:8090/api/v1/wallet/balance/<address>`
3. Review deployed contracts: `curl http://localhost:8090/api/v1/vm/contracts`

---

**Happy Trading! 🚀**

The Q-NarwhalKnight DEX combines quantum-resistant security with high-performance consensus for a truly next-generation decentralized exchange.
