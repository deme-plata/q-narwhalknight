# Q-NarwhalKnight v0.0.6-beta - Linux Release

## What's New in v0.0.6-beta
- **Fixed**: Peer discovery connection count display
- **Fixed**: Compilation issues with handler trait bounds
- **Improved**: Network stability and peer management

## Quick Start

```bash
chmod +x q-api-server
./q-api-server --port 8080
```

## System Requirements
- Linux x86_64 (Ubuntu 20.04+, Debian 11+, RHEL 8+)
- 8GB RAM minimum (32GB recommended)
- 50GB storage (500GB SSD recommended)

## Configuration

### Environment Variables
- `Q_DB_PATH`: Database directory (default: ./data)
- `RUST_LOG`: Log level (info, debug, trace)

### Command Line Options
```bash
./q-api-server --help
```

## Running as a Service (systemd)

Create `/etc/systemd/system/q-narwhalknight.service`:

```ini
[Unit]
Description=Q-NarwhalKnight Quantum Consensus Node
After=network.target

[Service]
Type=simple
User=qnk
WorkingDirectory=/opt/q-narwhalknight
Environment="Q_DB_PATH=/var/lib/q-narwhalknight"
ExecStart=/opt/q-narwhalknight/q-api-server --port 8080
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable q-narwhalknight
sudo systemctl start q-narwhalknight
```

## Support
- Documentation: https://api.quillon.xyz
- GitHub: https://github.com/deme-plata/q-narwhalknight

## License
See LICENSE file for details.
