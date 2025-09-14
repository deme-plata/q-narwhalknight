# 🌊🤖 Quantum Water Robot Control CLI (qrobot)

A comprehensive command-line interface for controlling quantum-enhanced water robots integrated with the Q-NarwhalKnight consensus system.

## Features

- **🤖 Robot Control**: Connect to and control individual quantum water robots
- **🐟 Swarm Coordination**: Manage robot swarms with quantum entanglement
- **⚛️ Quantum Monitoring**: Real-time quantum state visualization and measurement
- **📊 Sensor Analytics**: Environmental monitoring and data visualization
- **🌊 Marine Conservation**: Ecosystem protection and restoration tools
- **🔐 Post-Quantum Security**: Dilithium5/Kyber1024 cryptographic integration
- **🎯 Interactive UI**: Full-featured terminal interface with real-time updates

## Installation

### Prerequisites

- Rust 1.70+
- Q-NarwhalKnight consensus node running
- Network access to quantum water robots

### Build from Source

```bash
cd crates/q-robot-cli
cargo build --release
```

### Install Binary

```bash
cargo install --path crates/q-robot-cli
```

## Quick Start

### 1. Connect to Robots

```bash
# List available robots
qrobot robot list

# Connect to a specific robot
qrobot robot connect quantum_jelly_001 --robot-type jellyfish

# Check robot status
qrobot robot status quantum_jelly_001
```

### 2. Control Robot Movement

```bash
# Move robot to coordinates
qrobot robot move quantum_jelly_001 --target 45.2 -12.8 -15.5 --speed 0.7

# Activate quantum abilities
qrobot robot ability quantum_jelly_001 bioluminescence --params 0.8
```

### 3. Create and Manage Swarms

```bash
# Create a new swarm
qrobot swarm create exploration_team --size 5 --formation spiral

# Change formation
qrobot swarm formation exploration_team school

# Deploy on mission
qrobot swarm mission exploration_team explore --area -100 -100 -50 100 100 0
```

### 4. Monitor Quantum States

```bash
# Visualize quantum superposition
qrobot quantum visualize quantum_jelly_001 --viz-type superposition

# Measure quantum observables
qrobot quantum measure quantum_jelly_001 position

# Generate quantum random numbers
qrobot quantum random --bytes 32 --format hex
```

### 5. Environmental Monitoring

```bash
# Scan marine environment
qrobot ecosystem scan --radius 100 --depth 50

# Monitor water quality
qrobot ecosystem water --watch

# Track marine life
qrobot ecosystem life --species tuna
```

### 6. Interactive Terminal UI

```bash
# Launch full-screen interactive interface
qrobot ui --fullscreen

# Launch windowed mode
qrobot ui
```

## Robot Types

The CLI supports various quantum-enhanced water robot types:

- **🪼 Quantum Jellyfish** (`jellyfish`): Bioluminescent sensors with superposition states
- **🐬 Entangled Dolphins** (`dolphin`): Quantum communication and echolocation
- **🐙 Tunneling Octopi** (`octopus`): Quantum tunneling and phase camouflage  
- **🐋 Wave-Particle Whales** (`whale`): Massive quantum duality demonstrations
- **🦄 Superposition Seahorses** (`seahorse`): Multi-position quantum states
- **🦠 Nano Quantumonas** (`nano`): Microscopic quantum swimmers
- **🐟 Schooling Robotichthys** (`school`): Coordinated swarm intelligence
- **🦈 Cyber Cetus** (`guardian`): Large ecosystem management robots

## Swarm Formations

Available swarm formations for coordinated behavior:

- **School**: Fish-like schooling with leader-follower dynamics
- **Spiral**: Helical formation around central axis
- **Sphere**: 3D spherical coverage pattern
- **Line**: Linear formation for patrol missions
- **Grid**: Systematic grid coverage
- **Quantum**: Entangled formation maintaining Bell states

## Mission Types

Supported autonomous mission types:

- **Exploration**: Unknown area mapping with various search patterns
- **Patrol**: Perimeter monitoring with waypoint navigation
- **Research**: Scientific data collection and sampling
- **Rescue**: Search and rescue operations
- **Monitor**: Environmental monitoring with alert thresholds
- **Restoration**: Coral reef and ecosystem restoration

## Configuration

Configuration file: `robot-config.toml`

```toml
[network]
listen_addresses = ["/ip4/0.0.0.0/tcp/0"]
max_connections = 100
quantum_secure = true

[quantum]
enable_qrng = true
coherence_monitor_interval = 1.0

[quantum.entanglement]
auto_establish = true
target_fidelity = 0.9
bell_state_type = "PhiPlus"

[security.post_quantum]
enable_signatures = true
signature_algorithm = "Dilithium5"
enable_key_exchange = true
key_exchange_algorithm = "Kyber1024"
hybrid_mode = true

[[robots]]
id = "quantum_jelly_001"
robot_type = "QuantumJellyfish" 
endpoint = "tcp://192.168.1.100:8080"

[robots.auth]
auth_type = "certificate"
certificate = "./certs/jelly_001.pem"
private_key = "./certs/jelly_001.key"

[robots.capabilities]
bioluminescence = true
superposition_glow = true
quantum_sensing = true
```

## Interactive UI Guide

The terminal UI provides tabs for different functionalities:

### 🤖 Robots Tab
- View connected robots and their status
- Monitor battery levels, positions, and quantum coherence
- Control individual robot abilities

### 🐟 Swarms Tab  
- Manage active swarms and formations
- Visualize swarm coordination patterns
- Monitor entanglement fidelity

### ⚛️ Quantum Tab
- Real-time quantum state visualization
- Entanglement matrix display
- Quantum measurement results

### 📊 Sensors Tab
- Environmental sensor readings
- Water quality metrics
- Real-time data charts

### 🌊 Environment Tab
- Marine life detection and tracking
- Water quality assessment
- Conservation status updates

### 📝 Logs Tab
- System logs and events
- Error reporting and debugging
- Real-time message stream

### Keyboard Shortcuts

- `Tab` / `Shift+Tab`: Switch between tabs
- `↑` / `↓`: Navigate lists
- `Enter`: Select/activate items
- `r`: Refresh data
- `h` / `F1`: Toggle help screen
- `q`: Quit application

## Integration with Q-NarwhalKnight

The CLI integrates seamlessly with the Q-NarwhalKnight consensus system:

### Consensus Integration
```bash
# Connect to consensus network
qrobot consensus connect

# Submit robot data to consensus
qrobot consensus submit sensor "temperature:22.4,ph:8.1"

# Query consensus for coordination
qrobot consensus query robots

# Monitor consensus participation
qrobot consensus monitor
```

### Security Features

- **Post-Quantum Signatures**: Dilithium5 for data integrity
- **Quantum Key Exchange**: Kyber1024 for secure communication
- **Hybrid Cryptography**: Classical+post-quantum for transition period
- **Consensus Integration**: Secure data submission to distributed ledger

## Examples

### Basic Robot Control

```bash
# Connect to multiple robots
qrobot robot connect quantum_jelly_001 --robot-type jellyfish
qrobot robot connect dolphin_alpha_002 --robot-type dolphin
qrobot robot connect octopus_stealth_003 --robot-type octopus

# Create exploration swarm
qrobot swarm create deep_exploration --size 3 --formation triangle

# Deploy exploration mission
qrobot swarm mission deep_exploration explore \
  --area -200 -200 -100 200 200 0

# Monitor quantum entanglement
qrobot swarm entanglement deep_exploration
```

### Environmental Monitoring

```bash
# Comprehensive environmental scan
qrobot ecosystem scan --radius 500 --depth 100

# Set up continuous water quality monitoring
qrobot ecosystem water --watch

# Track endangered species
qrobot ecosystem life --species "north atlantic right whale"

# Execute coral restoration
qrobot ecosystem conserve coral-restore --location 34.2 -45.1 -12.0
```

### Quantum Operations

```bash
# Visualize multi-robot quantum states
qrobot quantum visualize swarm_alpha --viz-type entanglement

# Perform quantum measurements
qrobot quantum measure quantum_jelly_001 spin
qrobot quantum measure dolphin_alpha_002 momentum

# Generate cryptographic randomness
qrobot quantum random --bytes 256 --format base64

# Monitor quantum coherence over time
qrobot quantum coherence quantum_jelly_001 --duration 30.0
```

### Advanced Scripting

```bash
#!/bin/bash
# Automated patrol mission setup

# Create patrol swarm
qrobot swarm create harbor_patrol --size 4 --formation line

# Set up patrol route
qrobot swarm mission harbor_patrol patrol \
  --area -50 -50 -20 50 50 0

# Monitor for 1 hour
sleep 3600

# Return to base formation
qrobot swarm formation harbor_patrol sphere

echo "Patrol mission completed"
```

## API Integration

For programmatic access, the CLI can be integrated into larger systems:

```rust
use q_robot_cli::{RobotManager, SwarmController, QuantumStateMonitor};

#[tokio::main]
async fn main() -> Result<()> {
    let config = RobotConfig::load("robot-config.toml").await?;
    let mut robot_manager = RobotManager::new(config).await?;
    
    // Connect to robot
    let robot_id = RobotId::new("quantum_jelly_001");
    robot_manager.connect_robot(robot_id, Some("jellyfish".to_string())).await?;
    
    // Move robot
    robot_manager.move_robot("quantum_jelly_001", vec![10.0, 20.0, -5.0], 0.5).await?;
    
    // Get status
    let status = robot_manager.get_robot_status("quantum_jelly_001").await?;
    println!("Robot battery: {:.1}%", status.battery_level);
    
    Ok(())
}
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Verify robot endpoints are accessible
   - Check network connectivity and firewall settings
   - Ensure robots are powered on and responding

2. **Authentication Failed**
   - Verify certificate files exist and are readable
   - Check certificate validity and expiration
   - Ensure proper permissions on private key files

3. **Quantum State Errors**
   - Check quantum coherence times haven't expired
   - Verify entanglement network connectivity
   - Monitor environmental decoherence sources

4. **Consensus Connection Issues**
   - Verify Q-NarwhalKnight node is running
   - Check consensus endpoint configuration
   - Ensure post-quantum keys are properly generated

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
qrobot --debug robot connect quantum_jelly_001
```

### Log Files

Check log files for detailed error information:

```bash
# View recent logs
tail -f /var/log/qrobot/robot-control.log

# Search for errors
grep ERROR /var/log/qrobot/robot-control.log
```

## Contributing

Contributions are welcome! Please see the main Q-NarwhalKnight project for contribution guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/q-narwhalknight.git
cd q-narwhalknight

# Build CLI
cd crates/q-robot-cli
cargo build

# Run tests
cargo test

# Run with debug logging
RUST_LOG=debug cargo run -- robot list
```

## License

Licensed under the MIT License. See LICENSE file for details.

## Support

For support and questions:

- GitHub Issues: https://github.com/your-org/q-narwhalknight/issues
- Documentation: https://docs.q-narwhalknight.org
- Discord: https://discord.gg/q-narwhalknight

---

**🌊🤖 Quantum consensus awaits your command! 🤖🌊**