# Q-NarwhalKnight macOS Installation Guide

## Quantum-Enhanced DAG-BFT Consensus Node for macOS

This package contains pre-built binaries of the Q-NarwhalKnight consensus node for macOS.

### System Requirements

**Intel Mac (x86_64):**
- macOS 10.7 or later
- 4 GB RAM minimum (8 GB recommended)
- 20 GB available disk space
- Internet connection

**Apple Silicon Mac (aarch64 - M1/M2/M3):**
- macOS 11.0 (Big Sur) or later
- 4 GB RAM minimum (8 GB recommended)
- 20 GB available disk space
- Internet connection

### Installation

#### Option 1: Quick Start (Recommended)

1. **Download** the appropriate binary for your Mac:
   - Intel Mac: `q-api-server-macos-x86_64`
   - Apple Silicon: `q-api-server-macos-aarch64`

2. **Make it executable:**
   ```bash
   chmod +x q-api-server-macos-*
   ```

3. **Move to applications:**
   ```bash
   sudo mv q-api-server-macos-* /usr/local/bin/q-api-server
   ```

4. **Run the node:**
   ```bash
   q-api-server --port 8080
   ```

#### Option 2: Homebrew Installation (Coming Soon)

```bash
brew tap deme-plata/quillon
brew install q-narwhalknight
q-api-server --port 8080
```

### First Run

When you first run the binary, macOS may block it because it's not from an identified developer.

**To allow the app:**

1. Open **System Preferences** → **Security & Privacy**
2. Click the **General** tab
3. Click the lock icon to make changes
4. Click **Allow Anyway** next to the message about q-api-server
5. Run the command again

Or use the command line:
```bash
sudo xattr -r -d com.apple.quarantine /usr/local/bin/q-api-server
```

### Configuration

Create a configuration directory:
```bash
mkdir -p ~/.q-narwhalknight
```

#### Basic Configuration

Create `~/.q-narwhalknight/config.toml`:
```toml
[node]
port = 8080
node_id = "my-mac-node"
data_path = "$HOME/.q-narwhalknight/data"

[network]
bootstrap_peers = []
max_peers = 50

[consensus]
phase = 0  # 0=Classical, 1=Post-Quantum

[api]
enable_cors = true
allowed_origins = ["http://localhost:5173"]
```

### Running as a Service (launchd)

Create `~/Library/LaunchAgents/com.quillon.q-narwhalknight.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.quillon.q-narwhalknight</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/q-api-server</string>
        <string>--port</string>
        <string>8080</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardErrorPath</key>
    <string>/tmp/q-narwhalknight.err</string>
    <key>StandardOutPath</key>
    <string>/tmp/q-narwhalknight.out</string>
    <key>WorkingDirectory</key>
    <string>/Users/YOUR_USERNAME/.q-narwhalknight</string>
</dict>
</plist>
```

Load the service:
```bash
launchctl load ~/Library/LaunchAgents/com.quillon.q-narwhalknight.plist
```

### Usage

#### Start the Node

```bash
q-api-server --port 8080
```

#### With Custom Data Directory

```bash
export Q_DB_PATH=~/.q-narwhalknight/data
q-api-server --port 8080 --node-id my-mac-node
```

#### Enable Verbose Logging

```bash
export RUST_LOG=debug
q-api-server --port 8080
```

### API Endpoints

Once running, the API is available at `http://localhost:8080`:

- **Health**: `GET http://localhost:8080/api/v1/health`
- **Node Info**: `GET http://localhost:8080/api/v1/node/info`
- **Balance**: `GET http://localhost:8080/api/v1/wallet/balance`
- **Stream Events (SSE)**: `GET http://localhost:8080/api/v1/stream`

### Wallet GUI

The Quantum Wallet GUI is available at:
```
https://wallet.quillon.xyz
```

Or run it locally by downloading the web app from the releases page.

### Troubleshooting

#### Port Already in Use

```bash
# Find process using port 8080
lsof -i :8080

# Kill the process
kill -9 <PID>

# Or use a different port
q-api-server --port 8090
```

#### Permission Denied

```bash
# Make sure the binary is executable
chmod +x /usr/local/bin/q-api-server

# Remove quarantine attribute
sudo xattr -r -d com.apple.quarantine /usr/local/bin/q-api-server
```

#### Database Errors

```bash
# Clear the database
rm -rf ~/.q-narwhalknight/data

# Restart the node
q-api-server --port 8080
```

### Upgrading

1. Stop the running node
2. Download the new version
3. Replace the old binary
4. Restart the node

If using launchd:
```bash
launchctl unload ~/Library/LaunchAgents/com.quillon.q-narwhalknight.plist
# Replace binary
launchctl load ~/Library/LaunchAgents/com.quillon.q-narwhalknight.plist
```

### Uninstallation

```bash
# Stop the service
launchctl unload ~/Library/LaunchAgents/com.quillon.q-narwhalknight.plist

# Remove binary
sudo rm /usr/local/bin/q-api-server

# Remove data (optional - this deletes your wallet!)
rm -rf ~/.q-narwhalknight

# Remove service file
rm ~/Library/LaunchAgents/com.quillon.q-narwhalknight.plist
```

### Building from Source

If you prefer to build from source:

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone repository
git clone https://github.com/deme-plata/q-narwhalknight.git
cd q-narwhalknight

# Build
cargo build --release --package q-api-server

# Binary is at:
# target/release/q-api-server
```

### Support

- **Documentation**: https://api.quillon.xyz
- **GitHub Issues**: https://github.com/deme-plata/q-narwhalknight/issues
- **Discord**: https://discord.gg/jEhaYtAhfx
- **Email**: bitknight.dipper688@passmail.net

### Security

This software uses post-quantum cryptography to protect against quantum computer attacks:

- **Phase 0**: Ed25519 signatures (classical)
- **Phase 1**: Dilithium5 + Kyber1024 (post-quantum)
- **Phase 2**: Quantum Key Distribution (QKD) integration (planned)

Always verify the SHA-256 checksum of downloaded binaries.

### License

MIT License - see [LICENSE](../../LICENSE) for details.

---

**Built with ❤️ by the Quillon Team | Powered by Post-Quantum Cryptography**

Version: 0.0.3-beta
Build Date: 2025-10-19
Architecture: Universal macOS Binary (Intel + Apple Silicon)
