import { motion } from 'framer-motion';
import { Download, Server, Shield, Zap, Terminal, CheckCircle, Code, BookOpen, Rocket } from 'lucide-react';

export default function DownloadNodeScreen() {
  return (
    <div className="max-w-6xl mx-auto space-y-8">
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
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-quantum-cyan/20 border border-quantum-cyan/50 rounded-full">
          <span className="w-2 h-2 bg-quantum-cyan rounded-full animate-pulse"></span>
          <span className="text-sm font-bold text-quantum-cyan">v3.5.14-beta • WarpSync + ZK Privacy + Real Cryptography</span>
        </div>
      </motion.div>

      {/* WarpSync Highlight Banner */}
      <motion.div
        className="relative overflow-hidden p-6 bg-gradient-to-r from-quantum-green/30 via-quantum-cyan/20 to-quantum-purple/30 backdrop-blur-xl border border-quantum-green/50 rounded-2xl"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
      >
        <div className="absolute top-0 right-0 w-64 h-64 bg-quantum-green/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2"></div>
        <div className="relative flex items-center gap-6">
          <div className="flex-shrink-0 w-16 h-16 bg-quantum-green/30 rounded-2xl flex items-center justify-center">
            <Rocket className="w-8 h-8 text-quantum-green" />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <h2 className="text-2xl font-bold text-white">WarpSync Technology</h2>
              <span className="px-3 py-1 bg-quantum-green/30 text-quantum-green text-xs font-bold rounded-full uppercase">New in v2.3</span>
            </div>
            <p className="text-gray-300 mb-3">
              Sync 900,000+ blocks in under 5 minutes. New nodes join the network instantly with parallel block downloads,
              adaptive timeouts, and scan-forward gap closure.
            </p>
            <div className="flex flex-wrap gap-4 text-sm">
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-quantum-cyan" />
                <span className="text-quantum-cyan font-medium">10x Faster Sync</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-quantum-green" />
                <span className="text-quantum-green font-medium">Instant Endgame</span>
              </div>
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4 text-quantum-purple" />
                <span className="text-quantum-purple font-medium">Zero Data Loss</span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Feature Cards */}
      <div className="grid md:grid-cols-4 gap-4">
        {[
          { icon: Rocket, title: 'WarpSync', desc: '900K blocks in 5 min', highlight: true },
          { icon: Shield, title: 'Post-Quantum', desc: 'Dilithium5 + Kyber1024' },
          { icon: Zap, title: '1.2M+ TPS', desc: 'Sub-50ms finality' },
          { icon: Server, title: 'Validator Ready', desc: 'Full node support' },
        ].map((feature, i) => (
          <motion.div
            key={feature.title}
            className={`p-4 backdrop-blur-xl rounded-xl ${
              feature.highlight
                ? 'bg-gradient-to-br from-quantum-green/30 to-quantum-cyan/20 border-2 border-quantum-green/50'
                : 'bg-quantum-indigo/20 border border-quantum-purple/30'
            }`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
          >
            <feature.icon className={`w-8 h-8 mb-2 ${feature.highlight ? 'text-quantum-green' : 'text-quantum-cyan'}`} />
            <h3 className="font-bold text-white">{feature.title}</h3>
            <p className="text-sm text-gray-400">{feature.desc}</p>
            {feature.highlight && (
              <span className="inline-block mt-2 px-2 py-0.5 bg-quantum-green/20 text-quantum-green text-xs font-bold rounded">NEW</span>
            )}
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

          <div className="space-y-3 mb-6">
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">Ring-LWE VRF Mining</p>
                <p className="text-sm text-gray-400">Post-quantum secure mining leader election with lattice-based VRF</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-cyan flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">Genus-2 VDF Consensus</p>
                <p className="text-sm text-gray-400">Hyperelliptic curve VDF for quantum-resistant time proofs</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-purple flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">DAG-Knight + Slashing</p>
                <p className="text-sm text-gray-400">Byzantine fault tolerant consensus with economic penalties</p>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <a
              href="/downloads/q-api-server-v4.1.2-beta"
              download="q-api-server"
              className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-xl font-bold text-white hover:shadow-lg hover:shadow-quantum-cyan/50 transition-all"
            >
              <Download className="w-5 h-5" />
              Download Linux Binary (v4.1.2-beta)
            </a>
            <p className="text-center text-sm text-gray-400">
              Size: 162 MB | WarpSync + ZK Privacy + Ring-LWE VRF + Genus-2 VDF
            </p>
          </div>

          {/* Quick Start */}
          <div className="mt-6 p-4 bg-quantum-dark/50 rounded-xl border border-quantum-purple/20">
            <p className="text-sm font-mono text-gray-300 mb-2">Quick Start:</p>
            <pre className="text-xs text-quantum-cyan overflow-x-auto">
{`wget https://quillon.xyz/downloads/q-api-server-v4.1.2-beta
chmod +x q-api-server-v4.1.2-beta
./q-api-server-v4.1.2-beta --port 8080`}
            </pre>
            <p className="text-xs text-quantum-green mt-2">
              WarpSync auto-discovers peers & syncs 900K+ blocks in minutes
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

          <div className="space-y-3 mb-6">
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">Complete Windows Package</p>
                <p className="text-sm text-gray-400">All-in-one ZIP with node binary + required DLLs</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">Full Node + P2P Sync</p>
                <p className="text-sm text-gray-400">Block validation, wallet, DEX, mining - all included</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-cyan flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">Post-Quantum Cryptography</p>
                <p className="text-sm text-gray-400">Dilithium5 + Kyber1024 signatures built-in</p>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <a
              href="/downloads/q-narwhalknight-windows-x64.zip"
              download="q-narwhalknight-windows-x64.zip"
              className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-gradient-to-r from-quantum-purple to-quantum-pink rounded-xl font-bold text-white hover:shadow-lg hover:shadow-quantum-purple/50 transition-all"
            >
              <Download className="w-5 h-5" />
              Download Windows Package (ZIP)
            </a>
            <p className="text-center text-sm text-gray-400">
              Size: 77 MB (compressed) | Node + DLLs included
            </p>
          </div>

          {/* Quick Start */}
          <div className="mt-6 p-4 bg-quantum-dark/50 rounded-xl border border-quantum-purple/20">
            <p className="text-sm font-mono text-gray-300 mb-2">Quick Start:</p>
            <pre className="text-xs text-quantum-purple overflow-x-auto">
{`# 1. Extract the ZIP to a folder
# 2. Open PowerShell in that folder
# 3. Run:
.\\q-api-server-windows-x64.exe --port 9090 --p2p-port 9001`}
            </pre>
            <p className="text-xs text-gray-500 mt-2">
              Tip: Use port 9090 to avoid permission issues, or run as Administrator for port 8080
            </p>
          </div>
        </motion.div>
      </div>

      {/* macOS Build from Source */}
      <motion.div
        className="p-6 bg-gradient-to-br from-quantum-green/20 to-quantum-cyan/10 backdrop-blur-xl border border-quantum-green/30 rounded-2xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-quantum-green/20 rounded-xl flex items-center justify-center">
            <Terminal className="w-6 h-6 text-quantum-green" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">macOS (Build from Source)</h2>
            <p className="text-gray-400 text-sm">Intel & Apple Silicon</p>
          </div>
        </div>

        <div className="p-4 bg-quantum-dark/50 rounded-xl border border-quantum-green/20">
          <pre className="text-xs text-quantum-green overflow-x-auto">
{`# Install Rust if needed: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
git clone https://code.quillon.xyz/repo.git q-narwhalknight && cd q-narwhalknight
cargo build --release --package q-api-server
./target/release/q-api-server --port 8080`}
          </pre>
        </div>
      </motion.div>

      {/* System Requirements */}
      <motion.div
        className="p-6 bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-cyan/30 rounded-2xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-3">
          <Server className="w-6 h-6 text-quantum-cyan" />
          System Requirements
        </h2>

        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-sm font-bold text-quantum-cyan mb-3">Minimum</h3>
            <ul className="space-y-1 text-gray-300 text-sm">
              <li>CPU: 4 cores @ 2.5GHz</li>
              <li>RAM: 8 GB</li>
              <li>Storage: 50 GB SSD</li>
              <li>Network: 10 Mbps</li>
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-bold text-quantum-purple mb-3">Recommended</h3>
            <ul className="space-y-1 text-gray-300 text-sm">
              <li>CPU: 8+ cores @ 3.0GHz+</li>
              <li>RAM: 32 GB+</li>
              <li>Storage: 500 GB NVMe SSD</li>
              <li>Network: 100 Mbps+</li>
            </ul>
          </div>
        </div>
      </motion.div>

      {/* Resources */}
      <motion.div
        className="grid md:grid-cols-3 gap-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
      >
        <a
          href="https://code.quillon.xyz/"
          target="_blank"
          rel="noopener noreferrer"
          className="p-4 bg-quantum-dark/50 rounded-xl border border-quantum-cyan/20 hover:border-quantum-cyan/50 transition-all"
        >
          <Code className="w-6 h-6 text-quantum-cyan mb-2" />
          <h3 className="font-bold text-white mb-1">Source Code</h3>
          <p className="text-sm text-gray-400">View & contribute</p>
        </a>

        <a
          href="https://api.quillon.xyz"
          target="_blank"
          rel="noopener noreferrer"
          className="p-4 bg-quantum-dark/50 rounded-xl border border-quantum-purple/20 hover:border-quantum-purple/50 transition-all"
        >
          <BookOpen className="w-6 h-6 text-quantum-purple mb-2" />
          <h3 className="font-bold text-white mb-1">API Docs</h3>
          <p className="text-sm text-gray-400">REST & WebSocket</p>
        </a>

        <div className="p-4 bg-quantum-dark/50 rounded-xl border border-quantum-green/20">
          <Shield className="w-6 h-6 text-quantum-green mb-2" />
          <h3 className="font-bold text-white mb-1">PQC Security</h3>
          <p className="text-sm text-gray-400">Dilithium5 + Kyber1024</p>
        </div>
      </motion.div>
    </div>
  );
}
