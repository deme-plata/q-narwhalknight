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
          <span className="text-sm font-bold text-quantum-green">v0.0.29-beta Released - High-Performance Mining (930% Faster!)</span>
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
                <p className="text-white font-medium">Latest Standalone Binary</p>
                <p className="text-sm text-gray-400">Single executable - ready to run immediately</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">High-Performance Mining Queue</p>
                <p className="text-sm text-gray-400">v0.0.29-beta with 930% throughput increase + real peer count</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">Auto-Connect to Network</p>
                <p className="text-sm text-gray-400">Connects to bootstrap masternode automatically</p>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <a
              href="/downloads/q-api-server-v0.0.29-beta"
              download="q-api-server"
              className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-xl font-bold text-white hover:shadow-lg hover:shadow-quantum-cyan/50 transition-all"
            >
              <Download className="w-5 h-5" />
              Download Linux Binary (v0.0.29-beta)
            </a>
            <p className="text-center text-sm text-gray-400">
              Size: 104 MB | Latest version | Single executable
            </p>
          </div>

          {/* Installation Instructions */}
          <div className="mt-6 p-4 bg-quantum-dark/50 rounded-xl border border-quantum-purple/20">
            <p className="text-sm font-mono text-gray-300 mb-2">Quick Start (Standalone Binary):</p>
            <pre className="text-xs text-quantum-cyan overflow-x-auto">
{`chmod +x q-api-server
./q-api-server --port 8080`}
            </pre>
            <p className="text-xs text-quantum-green mt-2">
              ✅ Auto-connects to bootstrap masternode at 185.182.185.227:8081
            </p>
            <p className="text-xs text-quantum-purple mt-2">
              🌐 Network discovery: mDNS (local) + Kademlia DHT (global) + Bootstrap node
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

      {/* Validator Node Setup Guide */}
      <motion.div
        className="p-8 bg-gradient-to-br from-quantum-purple/20 to-quantum-indigo/20 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
      >
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
          <Shield className="w-7 h-7 text-quantum-purple" />
          Validator Node Setup Guide (v0.0.29-beta)
        </h2>
        <p className="text-gray-400 mb-6">
          Run a validator node to participate in consensus and earn block production rewards. Each validator needs a unique configuration.
        </p>

        <div className="space-y-6">
          {/* Single Node / Regular User */}
          <div className="p-6 bg-quantum-dark/50 rounded-xl border border-quantum-cyan/30">
            <h3 className="text-lg font-bold text-quantum-cyan mb-4">🚀 Option 1: Single Node (Regular Users)</h3>
            <p className="text-gray-300 text-sm mb-4">
              Perfect for mining, testing, or running a single validator. No special configuration needed.
            </p>
            <pre className="text-xs text-quantum-cyan overflow-x-auto p-4 bg-quantum-dark/50 rounded-lg">
{`# Download and make executable
chmod +x q-api-server-v0.0.29-beta

# Run the node (automatic block production every 15 seconds)
./q-api-server-v0.0.29-beta --port 8080

# With custom data directory
Q_DB_PATH=./my-node-data ./q-api-server-v0.0.29-beta --port 8080`}
            </pre>
            <div className="mt-3 p-3 bg-quantum-green/10 border border-quantum-green/30 rounded-lg">
              <p className="text-quantum-green text-xs">
                ✅ Blocks will be produced automatically every 15 seconds<br/>
                ✅ Mining rewards will be accepted and distributed<br/>
                ✅ Connects to network via bootstrap nodes automatically
              </p>
            </div>
          </div>

          {/* Multi-Validator Setup */}
          <div className="p-6 bg-quantum-dark/50 rounded-xl border border-quantum-purple/30">
            <h3 className="text-lg font-bold text-quantum-purple mb-4">⚡ Option 2: Multi-Validator Setup (Advanced)</h3>
            <p className="text-gray-300 text-sm mb-4">
              Run multiple validators on different servers or ports. Each validator needs a unique index to prevent competing block production.
            </p>

            <div className="space-y-4">
              <div className="p-4 bg-quantum-indigo/10 rounded-xl border border-quantum-purple/20">
                <p className="text-quantum-purple font-bold text-sm mb-3">Required Environment Variables:</p>
                <div className="space-y-2 text-xs">
                  <div className="p-2 bg-quantum-dark/50 rounded">
                    <code className="text-quantum-cyan">Q_VALIDATOR_INDEX</code>
                    <span className="text-gray-400"> - Unique validator ID (0, 1, 2, ...)</span>
                  </div>
                  <div className="p-2 bg-quantum-dark/50 rounded">
                    <code className="text-quantum-cyan">Q_TOTAL_VALIDATORS</code>
                    <span className="text-gray-400"> - Total number of validators in network</span>
                  </div>
                  <div className="p-2 bg-quantum-dark/50 rounded">
                    <code className="text-quantum-cyan">Q_DB_PATH</code>
                    <span className="text-gray-400"> - Unique database path per validator</span>
                  </div>
                  <div className="p-2 bg-quantum-dark/50 rounded">
                    <code className="text-quantum-cyan">--node-id</code>
                    <span className="text-gray-400"> - Unique node identifier</span>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-quantum-dark/50 rounded-lg">
                <p className="text-sm font-bold text-white mb-3">Example: 3-Validator Network</p>

                <div className="space-y-4">
                  <div>
                    <p className="text-quantum-green text-xs font-bold mb-2">Validator 0 (Primary - Produces empty blocks):</p>
                    <pre className="text-xs text-gray-300 overflow-x-auto p-3 bg-quantum-dark/80 rounded">
{`Q_VALIDATOR_INDEX=0 \\
Q_TOTAL_VALIDATORS=3 \\
Q_DB_PATH=./data-validator-0 \\
./q-api-server-v0.0.29-beta --port 8080 --node-id validator-0`}
                    </pre>
                  </div>

                  <div>
                    <p className="text-quantum-purple text-xs font-bold mb-2">Validator 1 (Secondary):</p>
                    <pre className="text-xs text-gray-300 overflow-x-auto p-3 bg-quantum-dark/80 rounded">
{`Q_VALIDATOR_INDEX=1 \\
Q_TOTAL_VALIDATORS=3 \\
Q_DB_PATH=./data-validator-1 \\
./q-api-server-v0.0.29-beta --port 8081 --node-id validator-1`}
                    </pre>
                  </div>

                  <div>
                    <p className="text-quantum-cyan text-xs font-bold mb-2">Validator 2 (Tertiary):</p>
                    <pre className="text-xs text-gray-300 overflow-x-auto p-3 bg-quantum-dark/80 rounded">
{`Q_VALIDATOR_INDEX=2 \\
Q_TOTAL_VALIDATORS=3 \\
Q_DB_PATH=./data-validator-2 \\
./q-api-server-v0.0.29-beta --port 8082 --node-id validator-2`}
                    </pre>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-quantum-yellow/10 border border-quantum-yellow/30 rounded-lg">
                <p className="text-quantum-yellow font-bold text-sm mb-2">⚠️ Important Notes:</p>
                <ul className="text-gray-300 text-xs space-y-1">
                  <li>• Only Validator 0 produces empty blocks to maintain DAG continuity</li>
                  <li>• All validators accept mining solutions and produce blocks with transactions</li>
                  <li>• Each validator MUST have a unique database path (Q_DB_PATH)</li>
                  <li>• Each validator MUST have a unique port number</li>
                  <li>• Each validator MUST have a unique node-id</li>
                  <li>• Validator indices must be sequential: 0, 1, 2, 3, ... (no gaps)</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Advanced Configuration */}
          <div className="p-6 bg-quantum-dark/50 rounded-xl border border-quantum-cyan/30">
            <h3 className="text-lg font-bold text-quantum-cyan mb-4">⚙️ Advanced Configuration Options</h3>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="p-3 bg-quantum-indigo/10 rounded-lg border border-quantum-cyan/10">
                <code className="text-quantum-cyan text-xs">Q_BLOCK_INTERVAL_SECS</code>
                <p className="text-gray-400 text-xs mt-2">Block production interval in seconds (default: 15, range: 5-300)</p>
                <pre className="text-xs text-quantum-green mt-2">
{`Q_BLOCK_INTERVAL_SECS=30`}
                </pre>
              </div>

              <div className="p-3 bg-quantum-indigo/10 rounded-lg border border-quantum-cyan/10">
                <code className="text-quantum-cyan text-xs">Q_MIN_SOLUTIONS_PER_BLOCK</code>
                <p className="text-gray-400 text-xs mt-2">Minimum mining solutions before block production (default: 1)</p>
                <pre className="text-xs text-quantum-green mt-2">
{`Q_MIN_SOLUTIONS_PER_BLOCK=10`}
                </pre>
              </div>

              <div className="p-3 bg-quantum-indigo/10 rounded-lg border border-quantum-cyan/10">
                <code className="text-quantum-cyan text-xs">Q_MAX_SOLUTIONS_PER_BLOCK</code>
                <p className="text-gray-400 text-xs mt-2">Maximum solutions per block (default: 100, max: 1000)</p>
                <pre className="text-xs text-quantum-green mt-2">
{`Q_MAX_SOLUTIONS_PER_BLOCK=500`}
                </pre>
              </div>

              <div className="p-3 bg-quantum-indigo/10 rounded-lg border border-quantum-cyan/10">
                <code className="text-quantum-cyan text-xs">Q_ALLOW_MANUAL_TRIGGER</code>
                <p className="text-gray-400 text-xs mt-2">Enable manual block triggering (default: false, security)</p>
                <pre className="text-xs text-quantum-green mt-2">
{`Q_ALLOW_MANUAL_TRIGGER=true`}
                </pre>
              </div>
            </div>
          </div>

          {/* Verification Steps */}
          <div className="p-6 bg-quantum-dark/50 rounded-xl border border-quantum-green/30">
            <h3 className="text-lg font-bold text-quantum-green mb-4">✅ Verify Your Validator Setup</h3>

            <div className="space-y-3">
              <div className="p-3 bg-quantum-dark/50 rounded-lg">
                <p className="text-white text-sm font-bold mb-2">1. Check Node Status:</p>
                <pre className="text-xs text-quantum-cyan overflow-x-auto">
{`curl http://localhost:8080/api/v1/status | jq`}
                </pre>
                <p className="text-gray-400 text-xs mt-2">Should show increasing block height, peer count, and consensus status</p>
              </div>

              <div className="p-3 bg-quantum-dark/50 rounded-lg">
                <p className="text-white text-sm font-bold mb-2">2. Monitor Block Production:</p>
                <pre className="text-xs text-quantum-cyan overflow-x-auto">
{`# Watch logs for automatic block production
tail -f /path/to/logs | grep "BLOCK PRODUCED"`}
                </pre>
                <p className="text-gray-400 text-xs mt-2">You should see blocks every 15 seconds (default interval)</p>
              </div>

              <div className="p-3 bg-quantum-dark/50 rounded-lg">
                <p className="text-white text-sm font-bold mb-2">3. Test Mining Integration:</p>
                <pre className="text-xs text-quantum-cyan overflow-x-auto">
{`# Submit a test mining solution
curl -X POST http://localhost:8080/api/v1/submit-solution \\
  -H "Content-Type: application/json" \\
  -d '{"nonce": 12345, "hash": "test..."}'`}
                </pre>
                <p className="text-gray-400 text-xs mt-2">Solution should be accepted and included in next block</p>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
