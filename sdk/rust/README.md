# q-paas-sdk (Rust)

Production-ready Rust SDK for Q-NarwhalKnight Privacy-as-a-Service.

## Features

- ✅ **Bitcoin Support**: UTXO management, WIF key import, transaction signing
- ✅ **Ethereum Support**: EIP-1559 gas, MEV protection, Web3 integration
- ✅ **Type-Safe API**: Comprehensive error handling with `thiserror`
- ✅ **Async/Await**: Built on `tokio` and `reqwest`
- ✅ **Retry Logic**: Automatic exponential backoff
- ✅ **Client-Side Signing**: Private keys never leave your machine

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
q-paas-sdk = "4.0"
tokio = { version = "1", features = ["full"] }
```

Or install from local path:

```bash
cargo add --path /opt/orobit/shared/q-narwhalknight/sdk/rust
```

## Quick Start - Bitcoin

```rust
use q_paas_sdk::{PaaSClient, BitcoinWallet, PrivacyLevel};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize client
    let client = PaaSClient::new(
        "your_api_key".to_string(),
        "https://quillon.xyz".to_string()
    );

    // Create wallet
    let wallet = BitcoinWallet::from_wif("YOUR_WIF_KEY")?;

    // Mix transaction
    let result = client.mix_bitcoin_transaction(
        "signed_tx_hex",
        PrivacyLevel::Maximum,
        true // use Tor
    ).await?;

    println!("Mix ID: {}", result.mix_id);
    Ok(())
}
```

## Quick Start - Ethereum

```rust
use q_paas_sdk::{PaaSClient, EthereumWallet, PrivacyLevel};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize client
    let client = PaaSClient::new(
        "your_api_key".to_string(),
        "https://quillon.xyz".to_string()
    );

    // Create wallet
    let wallet = EthereumWallet::from_private_key("your_private_key_hex")?;

    // Mix transaction with MEV protection
    let result = client.mix_ethereum_transaction(
        "signed_tx_hex",
        PrivacyLevel::Enhanced,
        true // MEV protection
    ).await?;

    println!("Mix ID: {}", result.mix_id);
    Ok(())
}
```

## Examples

Run the examples:

```bash
# Bitcoin mixing example
export Q_PAAS_API_KEY="your_api_key"
export BITCOIN_WIF_KEY="your_wif_key"
cargo run --example bitcoin_mixing

# Ethereum mixing example
export Q_PAAS_API_KEY="your_api_key"
export ETHEREUM_PRIVATE_KEY="your_private_key"
cargo run --example ethereum_mixing
```

## API Reference

### Client Methods

- `mix_bitcoin_transaction()` - Mix a Bitcoin transaction
- `mix_ethereum_transaction()` - Mix an Ethereum transaction
- `get_mix_status()` - Get status of a mix operation
- `get_billing_info()` - Get account balance and usage
- `get_api_key_info()` - Get API key metadata

### Privacy Levels

- `PrivacyLevel::Standard` - 10 QUG/tx
- `PrivacyLevel::Enhanced` - 50 QUG/tx
- `PrivacyLevel::Maximum` - 200 QUG/tx

## Features

Enable/disable blockchain support:

```toml
[dependencies.q-paas-sdk]
version = "4.0"
default-features = false
features = ["bitcoin-support"] # or ["ethereum-support"] or ["full"]
```

## Testing

```bash
cargo test
cargo test --features full
```

## Documentation

Generate documentation:

```bash
cargo doc --open
```

## License

MIT
