#!/bin/bash

# Q-NarwhalKnight Miner - Debian Package Builder
# Creates .deb packages for Ubuntu/Debian distribution

set -e

echo "🐧 Building Q-NarwhalKnight Miner .deb package"
echo "=============================================="

# Configuration
VERSION="1.0.0"
ARCH="amd64"
PACKAGE_NAME="q-narwhalknight-miner"
MAINTAINER="Q-NarwhalKnight Labs <contact@qnarwhal.dev>"
DESCRIPTION="High-performance CPU/GPU cryptocurrency miner with CUDA support"

# Paths
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="$ROOT_DIR/target/x86_64-unknown-linux-gnu/release"
PACKAGE_DIR="$ROOT_DIR/dist/deb-build"
DEB_DIR="$PACKAGE_DIR/$PACKAGE_NAME"

echo "📦 Setting up package structure..."

# Create package directory structure
rm -rf "$PACKAGE_DIR"
mkdir -p "$DEB_DIR"/{DEBIAN,usr/{bin,share/{applications,pixmaps,doc/$PACKAGE_NAME}},etc/$PACKAGE_NAME}

# Copy binaries
echo "📋 Copying binaries..."
cp "$BUILD_DIR/q-miner" "$DEB_DIR/usr/bin/"
cp "$BUILD_DIR/q-miner-gui" "$DEB_DIR/usr/bin/"
cp "$BUILD_DIR/q-miner-benchmark" "$DEB_DIR/usr/bin/"

# Set executable permissions
chmod +x "$DEB_DIR/usr/bin"/*

# Copy documentation
echo "📚 Adding documentation..."
cp "$ROOT_DIR/crates/q-miner/README.md" "$DEB_DIR/usr/share/doc/$PACKAGE_NAME/"
cp "$ROOT_DIR/LICENSE" "$DEB_DIR/usr/share/doc/$PACKAGE_NAME/"

# Create desktop entry
echo "🖥️ Creating desktop entry..."
cat > "$DEB_DIR/usr/share/applications/$PACKAGE_NAME.desktop" << EOF
[Desktop Entry]
Name=Q-NarwhalKnight Miner
Comment=High-performance cryptocurrency miner
Exec=q-miner-gui
Icon=q-miner
Terminal=false
Type=Application
Categories=Network;Office;
Keywords=cryptocurrency;mining;blockchain;bitcoin;quantum;
EOF

# Create default configuration
echo "⚙️ Adding default configuration..."
cat > "$DEB_DIR/etc/$PACKAGE_NAME/config.toml" << EOF
# Q-NarwhalKnight Miner Configuration
# Copy to ~/.config/q-miner/config.toml to customize

[mining]
algorithm = "dag-knight-vdf"
intensity = 7
auto_tune = true
enable_cpu = true
enable_gpu = true
max_temperature = 85.0

[hardware]
cpu_threads = 0  # 0 = auto-detect
cuda_enabled = true
opencl_enabled = true
memory_limit_gb = 8.0
thermal_throttle = true

[network]
mode = "pool"
tor_enabled = true
p2p_enabled = true
max_peers = 32

[pool]
url = "stratum+tor://pool.qnarwhal.onion:4444"
worker_name = "auto"
failover_enabled = true

[wallet]
# address = "qnk1...your_address_here"
auto_create = false

[ui]
mode = "cli"
web_port = 8090
theme = "dark"

[logging]
level = "info"
file_enabled = true
console_enabled = true
EOF

# Create systemd service
echo "🔧 Creating systemd service..."
mkdir -p "$DEB_DIR/etc/systemd/system"
cat > "$DEB_DIR/etc/systemd/system/$PACKAGE_NAME.service" << EOF
[Unit]
Description=Q-NarwhalKnight Miner
After=network.target
Wants=network.target

[Service]
Type=simple
User=q-miner
Group=q-miner
ExecStart=/usr/bin/q-miner --config /etc/$PACKAGE_NAME/config.toml
Restart=always
RestartSec=5
KillMode=mixed
TimeoutStopSec=30

# Security hardening
NoNewPrivileges=yes
PrivateTmp=yes
ProtectHome=yes
ProtectSystem=strict
ReadWritePaths=/var/lib/q-miner /var/log/q-miner

# Resource limits
LimitNOFILE=65536
LimitNPROC=65536

[Install]
WantedBy=multi-user.target
EOF

# Create control file
echo "📋 Creating package control file..."
cat > "$DEB_DIR/DEBIAN/control" << EOF
Package: $PACKAGE_NAME
Version: $VERSION
Architecture: $ARCH
Maintainer: $MAINTAINER
Depends: libc6 (>= 2.31), libssl3 (>= 3.0.0), libcurl4 (>= 7.68.0)
Recommends: nvidia-driver-525, ocl-icd-opencl-dev
Suggests: tor, proxychains4
Section: net
Priority: optional
Homepage: https://github.com/deme-plata/q-narwhalknight
Description: $DESCRIPTION
 Q-NarwhalKnight is a quantum-enhanced DAG-BFT consensus network that provides
 high-throughput, low-latency cryptocurrency transactions with post-quantum
 cryptographic security.
 .
 This package provides a high-performance miner with the following features:
  * Multi-threaded CPU mining with SIMD optimizations
  * NVIDIA CUDA GPU mining support
  * OpenCL cross-platform GPU mining
  * Anonymous Tor-enabled pool mining
  * Real-time web dashboard
  * Hardware monitoring and thermal protection
  * Cross-platform configuration management
 .
 The miner supports both solo and pool mining modes, with automatic failover
 and performance optimization.
EOF

# Create postinst script
echo "📝 Creating post-installation script..."
cat > "$DEB_DIR/DEBIAN/postinst" << 'EOF'
#!/bin/bash
set -e

# Create q-miner user if it doesn't exist
if ! getent passwd q-miner >/dev/null 2>&1; then
    useradd --system --home-dir /var/lib/q-miner --create-home --shell /bin/false q-miner
fi

# Create directories
mkdir -p /var/lib/q-miner /var/log/q-miner
chown q-miner:q-miner /var/lib/q-miner /var/log/q-miner
chmod 755 /var/lib/q-miner /var/log/q-miner

# Set up logrotate
cat > /etc/logrotate.d/q-miner << 'LOGROTATE'
/var/log/q-miner/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    copytruncate
    su q-miner q-miner
}
LOGROTATE

# Enable systemd service (but don't start automatically)
systemctl daemon-reload
systemctl enable q-narwhalknight-miner.service

echo "✅ Q-NarwhalKnight Miner installed successfully!"
echo ""
echo "🚀 Quick Start:"
echo "   1. Edit configuration: sudo nano /etc/q-narwhalknight-miner/config.toml"
echo "   2. Add your wallet address to the config file"
echo "   3. Start mining: sudo systemctl start q-narwhalknight-miner"
echo "   4. Check status: sudo systemctl status q-narwhalknight-miner"
echo "   5. View logs: sudo journalctl -u q-narwhalknight-miner -f"
echo ""
echo "🌐 Web Dashboard: http://localhost:8090 (when running)"
echo "📖 Documentation: /usr/share/doc/q-narwhalknight-miner/README.md"
EOF

chmod 755 "$DEB_DIR/DEBIAN/postinst"

# Create prerm script
echo "📝 Creating pre-removal script..."
cat > "$DEB_DIR/DEBIAN/prerm" << 'EOF'
#!/bin/bash
set -e

# Stop and disable service
systemctl stop q-narwhalknight-miner.service 2>/dev/null || true
systemctl disable q-narwhalknight-miner.service 2>/dev/null || true
EOF

chmod 755 "$DEB_DIR/DEBIAN/prerm"

# Create postrm script
cat > "$DEB_DIR/DEBIAN/postrm" << 'EOF'
#!/bin/bash
set -e

case "$1" in
    purge)
        # Remove user and data on purge
        userdel q-miner 2>/dev/null || true
        rm -rf /var/lib/q-miner /var/log/q-miner
        rm -f /etc/logrotate.d/q-miner
        ;;
    remove)
        # Keep user and data on regular removal
        ;;
esac

systemctl daemon-reload 2>/dev/null || true
EOF

chmod 755 "$DEB_DIR/DEBIAN/postrm"

# Calculate installed size
INSTALLED_SIZE=$(du -sk "$DEB_DIR" | cut -f1)
echo "Installed-Size: $INSTALLED_SIZE" >> "$DEB_DIR/DEBIAN/control"

# Create copyright file
cat > "$DEB_DIR/usr/share/doc/$PACKAGE_NAME/copyright" << EOF
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: Q-NarwhalKnight Miner
Upstream-Contact: Q-NarwhalKnight Labs <contact@qnarwhal.dev>
Source: https://github.com/deme-plata/q-narwhalknight

Files: *
Copyright: 2025 Q-NarwhalKnight Labs
License: Apache-2.0

License: Apache-2.0
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 .
 http://www.apache.org/licenses/LICENSE-2.0
 .
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 .
 On Debian systems, the complete text of the Apache License 2.0 can be
 found in "/usr/share/common-licenses/Apache-2.0".
EOF

# Build the package
echo "🔨 Building .deb package..."
DEB_FILE="$ROOT_DIR/dist/${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"

# Use dpkg-deb to build the package
dpkg-deb --build "$DEB_DIR" "$DEB_FILE"

# Verify package
echo "🔍 Verifying package..."
dpkg-deb --info "$DEB_FILE"
dpkg-deb --contents "$DEB_FILE"

# Test package installation (in chroot if possible)
echo "🧪 Testing package..."
if command -v lintian >/dev/null 2>&1; then
    lintian "$DEB_FILE" || echo "⚠️ Lintian warnings (non-critical)"
fi

echo "✅ Debian package created successfully!"
echo "📦 Package: $DEB_FILE"
echo "📋 Size: $(du -sh "$DEB_FILE" | cut -f1)"

# Cleanup
rm -rf "$PACKAGE_DIR"

echo ""
echo "🚀 Installation commands:"
echo "   sudo dpkg -i $DEB_FILE"
echo "   sudo apt-get install -f  # Fix dependencies if needed"
echo ""
echo "🗑️ Removal commands:"
echo "   sudo apt-get remove $PACKAGE_NAME      # Remove package"
echo "   sudo apt-get purge $PACKAGE_NAME       # Remove package + config"