import { motion } from 'framer-motion';
import { Download, Server, Shield, Zap, Terminal, CheckCircle, Code, BookOpen } from 'lucide-react';

export default function DownloadNodeScreen() {
  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Hero Section */}
      <motion.div
        className="text-center space-y-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink bg-clip-text text-transparent">
          Download Q-NarwhalKnight Node
        </h1>
        <p className="text-xl text-gray-300 max-w-3xl mx-auto">
          Join the quantum consensus network. Run your own validator node with Phase 1 post-quantum cryptography.
        </p>
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-quantum-green/20 border border-quantum-green/50 rounded-full">
          <span className="w-2 h-2 bg-quantum-green rounded-full animate-pulse"></span>
          <span className="text-sm font-bold text-quantum-green">v0.0.9-beta Released - Peer Discovery Fixed!</span>
        </div>
      </motion.div>

      {/* Feature Cards */}
      <div className="grid md:grid-cols-4 gap-4">
        {[
          { icon: Shield, title: 'Post-Quantum', desc: 'Dilithium5 + Kyber1024' },
          { icon: Zap, title: '1.2M+ TPS', desc: 'Sub-50ms finality' },
          { icon: Server, title: 'Validator Ready', desc: 'Full node support' },
          { icon: Terminal, title: 'CLI + API', desc: 'REST & WebSocket' },
        ].map((feature, i) => (
          <motion.div
            key={feature.title}
            className="p-4 bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-purple/30 rounded-xl"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
          >
            <feature.icon className="w-8 h-8 text-quantum-cyan mb-2" />
            <h3 className="font-bold text-white">{feature.title}</h3>
            <p className="text-sm text-gray-400">{feature.desc}</p>
          </motion.div>
        ))}
      </div>

      {/* Download Cards */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Linux Download */}
        <motion.div
          className="p-8 bg-gradient-to-br from-quantum-indigo/30 to-quantum-purple/20 backdrop-blur-xl border border-quantum-cyan/30 rounded-2xl"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex items-center gap-4 mb-6">
            <div className="w-16 h-16 bg-quantum-cyan/20 rounded-xl flex items-center justify-center">
              <Terminal className="w-8 h-8 text-quantum-cyan" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">Linux x86_64</h2>
              <p className="text-gray-400">Ubuntu 20.04+ / Debian 11+ / RHEL 8+</p>
            </div>
          </div>

          <div className="space-y-4 mb-6">
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">Complete Linux Package</p>
                <p className="text-sm text-gray-400">Tarball with binary + comprehensive documentation</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">Production Ready</p>
                <p className="text-sm text-gray-400">Optimized release build with systemd service template</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">No Dependencies</p>
                <p className="text-sm text-gray-400">Static linking - only requires standard Linux libs</p>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <a
              href="/downloads/q-narwhalknight-linux-v0.0.9-beta.tar.gz"
              className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-xl font-bold text-white hover:shadow-lg hover:shadow-quantum-cyan/50 transition-all"
            >
              <Download className="w-5 h-5" />
              Download Linux Package (Latest)
            </a>
            <p className="text-center text-sm text-gray-400">
              Size: 36 MB (tar.gz) | Version: 0.0.9-beta
            </p>
          </div>

          {/* Installation Instructions */}
          <div className="mt-6 p-4 bg-quantum-dark/50 rounded-xl border border-quantum-purple/20">
            <p className="text-sm font-mono text-gray-300 mb-2">Quick Start:</p>
            <pre className="text-xs text-quantum-cyan overflow-x-auto">
{`tar -xzf q-narwhalknight-linux-v0.0.9-beta.tar.gz
cd q-narwhalknight-v0.0.9-beta/bin
chmod +x q-api-server
./q-api-server --port 8080`}
            </pre>
            <p className="text-xs text-quantum-green mt-2">
              ✅ Includes: q-api-server binary (98MB) + README + RELEASE_NOTES + Fixed peer discovery!
            </p>
          </div>
        </motion.div>

        {/* Windows Download */}
        <motion.div
          className="p-8 bg-gradient-to-br from-quantum-purple/30 to-quantum-pink/20 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="flex items-center gap-4 mb-6">
            <div className="w-16 h-16 bg-quantum-purple/20 rounded-xl flex items-center justify-center">
              <Code className="w-8 h-8 text-quantum-purple" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">Windows x64</h2>
              <p className="text-gray-400">Windows 10+ / Windows Server 2019+</p>
            </div>
          </div>

          <div className="space-y-4 mb-6">
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">Complete Windows Package</p>
                <p className="text-sm text-gray-400">Includes all required DLL dependencies</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">Full Consensus Support</p>
                <p className="text-sm text-gray-400">Validator & mining capabilities</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">No Installation Required</p>
                <p className="text-sm text-gray-400">Extract and run - all dependencies bundled</p>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <a
              href="/downloads/q-narwhalknight-windows-v0.0.2-beta.zip"
              className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-gradient-to-r from-quantum-purple to-quantum-pink rounded-xl font-bold text-white hover:shadow-lg hover:shadow-quantum-purple/50 transition-all"
            >
              <Download className="w-5 h-5" />
              Download Windows Package (Latest)
            </a>
            <p className="text-center text-sm text-gray-400">
              Size: 29 MB (zip) | Version: 0.0.2-beta | Includes all DLLs
            </p>
          </div>

          {/* Installation Instructions */}
          <div className="mt-6 p-4 bg-quantum-dark/50 rounded-xl border border-quantum-purple/20">
            <p className="text-sm font-mono text-gray-300 mb-2">Quick Start:</p>
            <pre className="text-xs text-quantum-purple overflow-x-auto">
{`# 1. Extract the zip file
# 2. Keep all DLL files with the .exe
# 3. Run:
q-api-server.exe --port 8080`}
            </pre>
            <p className="text-xs text-quantum-green mt-2">
              ✅ Includes: q-api-server.exe + 4 runtime DLLs + README + LICENSE
            </p>
          </div>
        </motion.div>

        {/* macOS Build from Source */}
        <motion.div
          className="p-8 bg-gradient-to-br from-quantum-green/20 to-quantum-cyan/10 backdrop-blur-xl border border-quantum-green/30 rounded-2xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <div className="flex items-center gap-4 mb-6">
            <div className="w-16 h-16 bg-quantum-green/20 rounded-xl flex items-center justify-center">
              <Terminal className="w-8 h-8 text-quantum-green" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">macOS (Build from Source)</h2>
              <p className="text-gray-400">Intel & Apple Silicon - Native Compilation</p>
            </div>
          </div>

          <div className="space-y-4 mb-6">
            <div className="p-4 bg-quantum-green/10 border border-quantum-green/30 rounded-xl">
              <p className="text-quantum-green font-bold mb-2">✅ Native macOS Performance</p>
              <p className="text-gray-300 text-sm">
                Build natively on your Mac for optimal performance and full feature support. The build process is automated and takes ~10-15 minutes.
              </p>
            </div>

            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-white font-medium">Full Feature Set</p>
                  <p className="text-sm text-gray-400">All quantum consensus features enabled</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-white font-medium">Optimized for Your Hardware</p>
                  <p className="text-sm text-gray-400">Native compilation for M1/M2/M3/M4 or Intel</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-white font-medium">Latest Code</p>
                  <p className="text-sm text-gray-400">Always get the newest features and fixes</p>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Start Guide */}
          <div className="space-y-4">
            <div className="p-4 bg-quantum-dark/50 rounded-xl border border-quantum-green/20">
              <p className="text-sm font-bold text-white mb-3">📋 Prerequisites (one-time setup):</p>
              <div className="space-y-2">
                <div className="p-3 bg-quantum-dark/50 rounded-lg">
                  <p className="text-quantum-green text-xs font-bold mb-1">Step 1: Install Homebrew (if not installed)</p>
                  <pre className="text-xs text-gray-300 overflow-x-auto">
{`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`}
                  </pre>
                </div>

                <div className="p-3 bg-quantum-dark/50 rounded-lg">
                  <p className="text-quantum-green text-xs font-bold mb-1">Step 2: Install Rust</p>
                  <pre className="text-xs text-gray-300 overflow-x-auto">
{`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env`}
                  </pre>
                </div>

                <div className="p-3 bg-quantum-dark/50 rounded-lg">
                  <p className="text-quantum-green text-xs font-bold mb-1">Step 3: Install Build Tools</p>
                  <pre className="text-xs text-gray-300 overflow-x-auto">
{`brew install cmake pkg-config openssl`}
                  </pre>
                </div>
              </div>
            </div>

            <div className="p-4 bg-quantum-dark/50 rounded-xl border border-quantum-cyan/20">
              <p className="text-sm font-bold text-white mb-3">🚀 Build & Run Q-NarwhalKnight:</p>
              <pre className="text-xs text-quantum-cyan overflow-x-auto">
{`# Clone the repository
git clone https://code.quillon.xyz/repo.git q-narwhalknight
cd q-narwhalknight

# Build the API server (takes ~10-15 minutes)
cargo build --release --package q-api-server

# Run the server
./target/release/q-api-server --port 8080`}
              </pre>
              <p className="text-xs text-quantum-green mt-3">
                ✅ Server will start on http://localhost:8080 with full API and consensus features
              </p>
            </div>

            <div className="p-4 bg-quantum-purple/10 border border-quantum-purple/30 rounded-xl">
              <p className="text-sm font-bold text-quantum-purple mb-2">💡 Pro Tips:</p>
              <ul className="text-gray-300 text-xs space-y-1">
                <li>• First build will download dependencies (~5 min), subsequent builds are faster</li>
                <li>• On Apple Silicon (M1/M2/M3/M4), the binary will be optimized for ARM64</li>
                <li>• Use <code className="text-quantum-cyan">--release</code> for production performance (10x faster)</li>
                <li>• The binary will be at <code className="text-quantum-cyan">./target/release/q-api-server</code></li>
              </ul>
            </div>

            <div className="p-4 bg-quantum-indigo/10 border border-quantum-indigo/30 rounded-xl">
              <p className="text-sm font-bold text-white mb-2">🔄 Alternative: Pre-built macOS Miner</p>
              <p className="text-gray-400 text-xs">
                For mining only, download the pre-compiled <a href="/mining" className="text-quantum-cyan hover:underline">macOS miner binaries</a> (Intel & Apple Silicon available)
              </p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* System Requirements */}
      <motion.div
        className="p-8 bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-cyan/30 rounded-2xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
          <Server className="w-7 h-7 text-quantum-cyan" />
          System Requirements
        </h2>

        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h3 className="text-lg font-bold text-quantum-cyan mb-4">Minimum</h3>
            <ul className="space-y-2 text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-quantum-green">•</span>
                <span><strong>CPU:</strong> 4 cores @ 2.5GHz</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-quantum-green">•</span>
                <span><strong>RAM:</strong> 8 GB</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-quantum-green">•</span>
                <span><strong>Storage:</strong> 50 GB SSD</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-quantum-green">•</span>
                <span><strong>Network:</strong> 10 Mbps</span>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-lg font-bold text-quantum-purple mb-4">Recommended</h3>
            <ul className="space-y-2 text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-quantum-purple">•</span>
                <span><strong>CPU:</strong> 8+ cores @ 3.0GHz+</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-quantum-purple">•</span>
                <span><strong>RAM:</strong> 32 GB+</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-quantum-purple">•</span>
                <span><strong>Storage:</strong> 500 GB NVMe SSD</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-quantum-purple">•</span>
                <span><strong>Network:</strong> 100 Mbps+</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-quantum-purple">•</span>
                <span><strong>GPU:</strong> NVIDIA RTX 3060+ (for ZK-STARK acceleration)</span>
              </li>
            </ul>
          </div>
        </div>
      </motion.div>

      {/* Documentation & Resources */}
      <motion.div
        className="p-8 bg-gradient-to-br from-quantum-purple/20 to-quantum-pink/10 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
          <BookOpen className="w-7 h-7 text-quantum-purple" />
          Documentation & Resources
        </h2>

        <div className="grid md:grid-cols-3 gap-4">
          <a
            href="https://code.quillon.xyz/"
            target="_blank"
            rel="noopener noreferrer"
            className="p-4 bg-quantum-dark/50 rounded-xl border border-quantum-cyan/20 hover:border-quantum-cyan/50 transition-all"
          >
            <Code className="w-6 h-6 text-quantum-cyan mb-2" />
            <h3 className="font-bold text-white mb-1">GitHub Repository</h3>
            <p className="text-sm text-gray-400">View source code & contribute</p>
          </a>

          <a
            href="https://api.quillon.xyz"
            target="_blank"
            rel="noopener noreferrer"
            className="p-4 bg-quantum-dark/50 rounded-xl border border-quantum-purple/20 hover:border-quantum-purple/50 transition-all"
          >
            <Terminal className="w-6 h-6 text-quantum-purple mb-2" />
            <h3 className="font-bold text-white mb-1">API Documentation</h3>
            <p className="text-sm text-gray-400">REST & WebSocket endpoints</p>
          </a>

          <div className="p-4 bg-quantum-dark/50 rounded-xl border border-quantum-green/20">
            <Shield className="w-6 h-6 text-quantum-green mb-2" />
            <h3 className="font-bold text-white mb-1">Security Audit</h3>
            <p className="text-sm text-gray-400">Post-quantum cryptography details</p>
          </div>
        </div>
      </motion.div>

      {/* CLI Configuration Guide */}
      <motion.div
        className="p-8 bg-quantum-dark/50 backdrop-blur-xl border border-quantum-cyan/20 rounded-2xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
      >
        <h2 className="text-2xl font-bold text-white mb-4">Configuration Options</h2>
        <p className="text-gray-400 mb-6">
          Customize your node with command-line flags or environment variables:
        </p>

        <div className="space-y-4">
          <div className="p-4 bg-quantum-indigo/10 rounded-xl border border-quantum-cyan/10">
            <code className="text-quantum-cyan text-sm">--port &lt;PORT&gt;</code>
            <p className="text-gray-400 text-sm mt-2">Set API server port (default: 8080)</p>
          </div>

          <div className="p-4 bg-quantum-indigo/10 rounded-xl border border-quantum-cyan/10">
            <code className="text-quantum-cyan text-sm">Q_DB_PATH=&lt;PATH&gt;</code>
            <p className="text-gray-400 text-sm mt-2">Database storage directory (default: ./data)</p>
          </div>

          <div className="p-4 bg-quantum-indigo/10 rounded-xl border border-quantum-cyan/10">
            <code className="text-quantum-cyan text-sm">--validator</code>
            <p className="text-gray-400 text-sm mt-2">Enable validator mode for consensus participation</p>
          </div>

          <div className="p-4 bg-quantum-indigo/10 rounded-xl border border-quantum-cyan/10">
            <code className="text-quantum-cyan text-sm">--enable-mining</code>
            <p className="text-gray-400 text-sm mt-2">Enable GPU mining (requires CUDA or OpenCL)</p>
          </div>
        </div>

        <div className="mt-6 p-4 bg-quantum-purple/10 rounded-xl border border-quantum-purple/20">
          <p className="text-sm font-bold text-quantum-purple mb-2">Example: Full Validator Setup</p>
          <pre className="text-xs text-gray-300 overflow-x-auto">
{`Q_DB_PATH=./validator-data ./q-api-server-linux-x86_64 \\
  --port 8080 \\
  --validator \\
  --enable-mining \\
  --log-level info`}
          </pre>
        </div>
      </motion.div>
    </div>
  );
}
