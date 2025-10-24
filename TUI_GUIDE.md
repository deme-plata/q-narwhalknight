# Q-NarwhalKnight Terminal UI Guide

## 🎨 Beautiful Terminal UI for Node Monitoring

The Q-NarwhalKnight node now includes a beautiful, interactive terminal UI that transforms the logging experience from scrolling text to a professional dashboard.

## Features

### 📊 Dashboard View (Default)
- **Network Metrics**: Peer count, inbound/outbound connections, Tor circuits, bandwidth
- **Blockchain Metrics**: Block height, DAG size, last block time, anchors, vertices
- **Performance Metrics**: TPS, latency (P50/P99), CPU usage, RAM usage, disk usage
- **TPS Chart**: Real-time sparkline showing transactions per second over last 60 seconds
- **Recent Logs**: Color-coded streaming logs with timestamps

### 📝 Full Logs View
- Full-screen log display
- Pause/resume streaming with `P` key
- Scroll through logs with `↑↓` arrow keys
- Color-coded by severity:
  - 🔴 **ERROR** - Red
  - 🟡 **WARN** - Yellow
  - 🔵 **INFO** - Cyan
  - ⚪ **DEBUG** - Gray
  - ⚫ **TRACE** - Dark Gray

### 🌐 Network View
- Peer connection table (ID, address, type, latency, traffic)
- Network topology visualization
- Tor circuit status
- Bootstrap node connectivity

### 📋 Interactive Menu
- Arrow key navigation
- Options:
  - Node Control
  - Network Information
  - Mining Status
  - Blockchain Explorer
  - Wallet Management
  - Performance Metrics
  - Configuration
  - Export Logs
  - Exit

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Tab` | Switch between views (Dashboard → Logs → Network → Menu) |
| `L` | Jump directly to logs view |
| `M` | Open menu |
| `N` | Jump to network view |
| `P` | Pause/resume logs |
| `↑↓` | Navigate menus / Scroll logs |
| `PgUp/PgDn` | Scroll logs quickly |
| `Enter` | Select menu item |
| `Esc` | Go back / Return to dashboard |
| `Q` | Quit |

## Usage

### Building with TUI Support

```bash
# Build q-api-server with TUI feature
cargo build --release --package q-api-server --features tui

# Or build with optimizations
timeout 36000 cargo build --release --package q-api-server --features tui
```

### Running with TUI

```bash
# Enable beautiful terminal UI
./target/release/q-api-server --tui

# With other options
./target/release/q-api-server --tui --port 8080 --node-id node1

# Production mode with TUI
./target/release/q-api-server --tui --production
```

### Running without TUI (Classic Mode)

```bash
# Standard logging output
./target/release/q-api-server

# TUI binary without --tui flag will run in classic mode
./target/release/q-api-server --port 8080
```

## Architecture

### Crates
- **q-tui**: Terminal UI implementation using ratatui 0.25
- **q-api-server**: Integrates TUI as optional feature

### Components
- `lib.rs`: Terminal initialization, event loop
- `app.rs`: Application state management
- `metrics.rs`: Metrics collection and formatting
- `events.rs`: Async keyboard/mouse event handling
- `ui/dashboard.rs`: Main dashboard view
- `ui/logs.rs`: Full-screen logs view
- `ui/network.rs`: Network monitoring view
- `ui/menu.rs`: Interactive menu system

### Dependencies
- **ratatui 0.25**: Modern terminal UI framework
- **crossterm 0.27**: Cross-platform terminal manipulation
- **ringbuf 0.3**: Circular buffer for log storage
- **tokio**: Async runtime

## Performance

- **Refresh Rate**: 250ms (4 FPS)
- **Log Buffer**: 1000 entries (circular)
- **TPS History**: 60 seconds of data
- **Memory Usage**: Minimal overhead (~2MB)

## Troubleshooting

### "TUI feature not enabled" Error
```bash
# You built without --features tui flag
# Solution: Rebuild with TUI feature
cargo build --release --package q-api-server --features tui
```

### Terminal Display Issues
```bash
# Ensure terminal supports Unicode and colors
export TERM=xterm-256color

# Clear terminal before running
clear && ./target/release/q-api-server --tui
```

### Logs Not Updating
- Press `P` to unpause logs
- Check if logs view is active (press `L`)
- Verify node is producing logs

## Future Enhancements

Planned features for TUI:
- [ ] Real-time metrics integration with node stats
- [ ] Custom log filtering by level/module
- [ ] Searchable logs with `/` key
- [ ] Export view to screenshot
- [ ] Mouse support for clicking UI elements
- [ ] Custom themes and color schemes
- [ ] Resizable panels
- [ ] Transaction pool visualization
- [ ] Consensus round visualization

## Development

### Running in Development Mode
```bash
# Quick development build with TUI
cargo run --package q-api-server --features tui -- --tui

# With logging
RUST_LOG=debug cargo run --package q-api-server --features tui -- --tui
```

### Adding New Metrics
1. Update `crates/q-tui/src/metrics.rs` with new field
2. Modify dashboard rendering in `crates/q-tui/src/ui/dashboard.rs`
3. Connect to real metrics in `q-api-server/src/main.rs`

### Creating New Views
1. Create new file in `crates/q-tui/src/ui/`
2. Add view mode to `ViewMode` enum in `app.rs`
3. Implement render function
4. Add keyboard shortcut in event handler

## Credits

Built with ❤️ using:
- [Ratatui](https://ratatui.rs/) - Terminal UI framework
- [Crossterm](https://github.com/crossterm-rs/crossterm) - Terminal manipulation
- [Tokio](https://tokio.rs/) - Async runtime

---

**Quantum consensus monitoring, beautifully rendered.** ⚛️✨
