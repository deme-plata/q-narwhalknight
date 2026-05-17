# Q-NarwhalKnight Trading Bot

🤖 **Automated Buy/Sell Bot for Native QNK and Custom Tokens**

## Features

### ✅ Implemented
- **Multi-Wallet Support** - Trade across all wallets simultaneously
- **Native QNK Trading** - Buy and sell native QNK tokens
- **Custom Token Support** - Trade any custom tokens on Q-NarwhalKnight
- **Multiple Trading Strategies**:
  - Grid Trading
  - Market Making
  - Dollar-Cost Averaging (DCA)
  - Arbitrage
  - Momentum Trading
  - Mean Reversion
- **Risk Management**:
  - Per-trade loss limits
  - Daily loss limits
  - Stop-loss orders
  - Take-profit orders
  - Portfolio allocation limits
- **Real-time Market Data** - Live ticker and order book feeds
- **Performance Tracking** - Trade history and statistics
- **Dry Run Mode** - Test strategies without real trades

## Installation

```bash
cd /opt/orobit/shared/q-narwhalknight
cargo build --release --package q-trading-bot
```

## Quick Start

### 1. Initialize Configuration

```bash
./target/release/q-trading-bot init-config
```

This creates `trading-bot.toml` with default settings.

### 2. Configure Your Strategy

Edit `trading-bot.toml`:

```toml
[general]
update_interval_seconds = 60
max_concurrent_orders = 10
auto_rebalance = false

[[wallets]]
id = "wallet_1"
name = "Trading Wallet 1"
enabled = true
max_trade_amount = 1000

[[strategies]]
name = "Grid Trading QNK"
type = "grid"
enabled = true
wallets = ["wallet_1"]
pairs = ["QNK/USDT"]

[strategies.parameters]
grid_levels = 10
price_range_percent = 5.0

[risk_management]
max_loss_per_trade = 0.02  # 2%
max_daily_loss = 0.10       # 10%
enable_stop_loss = true
default_stop_loss_percentage = 0.05  # 5%
```

### 3. Run the Bot

```bash
# Production mode
./target/release/q-trading-bot run

# Dry run mode (no real trades)
./target/release/q-trading-bot run --dry-run

# Custom config file
./target/release/q-trading-bot --config my-config.toml run
```

## Trading Strategies

### Grid Trading
Places buy and sell orders at regular price intervals.

**Parameters:**
- `grid_levels` - Number of grid levels
- `price_range_percent` - Price range for grid (%)

**Best for:** Sideways/ranging markets

### Market Making
Provides liquidity by placing orders on both sides of the order book.

**Parameters:**
- `spread_percent` - Spread between buy/sell (%)
- `order_size` - Size of each order

**Best for:** Earning spreads on liquid tokens

### Dollar-Cost Averaging (DCA)
Buys fixed amounts at regular intervals.

**Parameters:**
- `buy_amount` - Amount to buy each interval
- `interval_hours` - Hours between purchases

**Best for:** Long-term accumulation

### Arbitrage
Exploits price differences between trading pairs.

**Parameters:**
- `min_profit_percent` - Minimum profit to execute (%)
- `max_slippage` - Maximum acceptable slippage (%)

**Best for:** Multi-exchange or multi-pair trading

### Momentum Trading
Follows strong price trends.

**Parameters:**
- `momentum_period` - Period for momentum calculation
- `entry_threshold` - Momentum threshold for entry

**Best for:** Trending markets

### Mean Reversion
Buys oversold and sells overbought conditions.

**Parameters:**
- `lookback_period` - Period for mean calculation
- `deviation_threshold` - Standard deviations for signals

**Best for:** Oscillating markets

## CLI Commands

### Show Balances
```bash
./target/release/q-trading-bot balances
```

### Show Statistics
```bash
# Last 10 trades
./target/release/q-trading-bot stats

# Last 50 trades
./target/release/q-trading-bot stats --limit 50
```

### Run with Custom API Endpoint
```bash
./target/release/q-trading-bot --api-endpoint http://mainnet.quillon.xyz:8080 run
```

## Risk Management

The bot includes comprehensive risk management:

1. **Per-Trade Limits** - Maximum loss per trade
2. **Daily Loss Limits** - Stops trading if daily loss exceeded
3. **Position Sizing** - Automatic position size calculation
4. **Stop-Loss Orders** - Automatic stop-loss placement
5. **Take-Profit Orders** - Automatic profit-taking
6. **Portfolio Limits** - Maximum allocation per token
7. **Minimum Balances** - Keeps minimum native token balance

## Architecture

```
q-trading-bot/
├── src/
│   ├── main.rs           # CLI and main entry point
│   ├── types.rs          # Core data types
│   ├── config.rs         # Configuration management
│   ├── api_client.rs     # Q-NarwhalKnight API client
│   ├── wallet_manager.rs # Multi-wallet management
│   ├── engine.rs         # Main trading engine
│   └── strategies/       # Trading strategies
│       ├── mod.rs
│       ├── grid.rs
│       ├── market_maker.rs
│       ├── dca.rs
│       ├── arbitrage.rs
│       ├── momentum.rs
│       └── mean_reversion.rs
└── Cargo.toml
```

## Example Usage

### Basic Grid Trading Bot

```bash
# 1. Create config
cat > trading-bot.toml << EOF
[general]
update_interval_seconds = 30

[[wallets]]
id = "my_wallet"
enabled = true
max_trade_amount = 1000

[[strategies]]
name = "QNK Grid"
type = "grid"
enabled = true
wallets = ["my_wallet"]
pairs = ["QNK/USDT"]

[strategies.parameters]
grid_levels = 20
price_range_percent = 10.0
EOF

# 2. Run the bot
./target/release/q-trading-bot run
```

### Market Making Bot for Custom Token

```bash
# Config for market making
cat > mm-bot.toml << EOF
[[strategies]]
name = "TOKEN1 Market Maker"
type = "market_maker"
enabled = true
pairs = ["TOKEN1/QNK"]

[strategies.parameters]
spread_percent = 0.5
order_size = 100
refresh_interval_seconds = 10
EOF

./target/release/q-trading-bot --config mm-bot.toml run
```

## Performance Monitoring

The bot tracks:
- Total trades executed
- Win rate
- Total volume
- Profit/loss
- Sharpe ratio
- Maximum drawdown
- Average trade duration

All data is stored in `trade-history.db` (sled database).

## Safety Features

- **Dry Run Mode** - Test without real money
- **Position Limits** - Prevent over-exposure
- **Emergency Stop** - Press Ctrl+C to stop gracefully
- **Transaction Validation** - All orders validated before execution
- **Balance Checks** - Ensures sufficient balance before trading

## Advanced Configuration

### Multiple Strategies

```toml
[[strategies]]
name = "Grid QNK"
type = "grid"
enabled = true
pairs = ["QNK/USDT"]

[[strategies]]
name = "DCA Bitcoin"
type = "dca"
enabled = true
pairs = ["BTC/QNK"]

[[strategies]]
name = "Market Make TOKEN1"
type = "market_maker"
enabled = true
pairs = ["TOKEN1/QNK"]
```

### Multi-Wallet Trading

```toml
[[wallets]]
id = "wallet_1"
name = "Conservative"
max_trade_amount = 100

[[wallets]]
id = "wallet_2"
name = "Aggressive"
max_trade_amount = 1000

[[strategies]]
name = "Conservative Grid"
wallets = ["wallet_1"]

[[strategies]]
name = "Aggressive Momentum"
wallets = ["wallet_2"]
```

## Troubleshooting

### Bot won't start
- Check API endpoint is reachable
- Verify configuration file syntax
- Ensure wallets are properly configured

### No trades executing
- Check strategy is enabled
- Verify sufficient balance
- Check risk management limits
- Enable debug logging: `--log-level debug`

### Orders failing
- Check network connectivity
- Verify wallet has sufficient balance
- Check token exists and is tradeable

## Security

- Never commit private keys to git
- Use environment variables for sensitive data
- Enable trade limits
- Start with small amounts and dry run mode
- Monitor the bot regularly

## Future Enhancements

- [ ] Machine learning price prediction
- [ ] Technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Telegram/Discord notifications
- [ ] Web dashboard
- [ ] Backtesting framework
- [ ] Paper trading mode
- [ ] Multi-exchange support
- [ ] Advanced order types (trailing stop, iceberg)

## License

Same as Q-NarwhalKnight project

## Support

For issues and questions:
- GitHub Issues: https://github.com/deme-plata/q-narwhalknight/issues
- Documentation: https://docs.quillon.xyz/trading-bot

---

**⚠️ Disclaimer:** Trading cryptocurrency involves substantial risk. This bot is provided as-is with no guarantees. Always test thoroughly and start with small amounts.
