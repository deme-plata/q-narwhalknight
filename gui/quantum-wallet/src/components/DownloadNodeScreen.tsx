import { motion } from 'framer-motion';
import { Download, Server, Shield, Zap, Terminal, CheckCircle, Code, BookOpen, Rocket, Cpu, Pickaxe, Wallet, Monitor } from 'lucide-react';

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
        <p className="text-xl text-gray-100 max-w-3xl mx-auto">
          Join the quantum consensus network. Run your own validator node with Phase 1 post-quantum cryptography.
        </p>
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-quantum-cyan/20 border border-quantum-cyan/50 rounded-full">
          <span className="w-2 h-2 bg-quantum-cyan rounded-full animate-pulse"></span>
          <span className="text-sm font-bold text-quantum-cyan">v10.4.12 • Post-Quantum + DAG-Knight + GPU Mining + DeFi Stack</span>
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
              <span className="px-3 py-1 bg-quantum-green/30 text-quantum-green text-xs font-bold rounded-full uppercase">Mainnet 2026.1</span>
            </div>
            <p className="text-gray-100 mb-3">
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
            <p className="text-sm text-gray-200">{feature.desc}</p>
            {feature.highlight && (
              <span className="inline-block mt-2 px-2 py-0.5 bg-quantum-green/20 text-quantum-green text-xs font-bold rounded">NEW</span>
            )}
          </motion.div>
        ))}
      </div>

      {/* Download Cards */}
      <div className="grid md:grid-cols-3 gap-6">
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
              <p className="text-gray-200">Ubuntu 20.04+ / Debian 11+ / RHEL 8+</p>
            </div>
          </div>

          <div className="space-y-3 mb-6">
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">Ring-LWE VRF Mining</p>
                <p className="text-sm text-gray-200">Post-quantum secure mining leader election with lattice-based VRF</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-cyan flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">Genus-2 VDF Consensus</p>
                <p className="text-sm text-gray-200">Hyperelliptic curve VDF for quantum-resistant time proofs</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-purple flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">DAG-Knight + Slashing</p>
                <p className="text-sm text-gray-200">Byzantine fault tolerant consensus with economic penalties</p>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <a
              href="https://quillon.xyz/downloads/q-api-server-v10.4.12"
              download="q-api-server"
              className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-xl font-bold text-white hover:shadow-lg hover:shadow-quantum-cyan/50 transition-all"
            >
              <Download className="w-5 h-5" />
              Download Linux Binary (v10.4.12)
            </a>
            <p className="text-center text-sm text-gray-200">
              Size: ~90 MB | Bitcoin-Style 21M Emission + P2P Gossipsub + DeFi Stack
            </p>
          </div>

          {/* Quick Start */}
          <div className="mt-6 p-4 bg-quantum-dark/50 rounded-xl border border-quantum-purple/20">
            <p className="text-sm font-mono text-gray-100 mb-2">Quick Start:</p>
            <pre className="text-xs text-quantum-cyan overflow-x-auto">
{`wget https://quillon.xyz/downloads/q-api-server-v10.4.12
chmod +x q-api-server-v10.4.12
./q-api-server-v10.4.12 --port 8080 --tui --admin-wallet YOUR_WALLET_ADDRESS`}
            </pre>
            <p className="text-xs text-quantum-green mt-2">
              WarpSync auto-discovers peers & syncs 900K+ blocks in minutes
            </p>
            <div className="mt-3 p-3 bg-emerald-900/30 border border-emerald-700/40 rounded-lg">
              <p className="text-xs text-emerald-300 font-semibold mb-1">Node Admin Panel & Fee Earnings</p>
              <p className="text-xs text-gray-300">
                Use <span className="text-quantum-cyan font-mono">--admin-wallet</span> with your wallet address to enable the admin panel.
                Open <span className="text-quantum-cyan font-mono">http://localhost:8080</span> in your browser to access the Node Settings
                gear icon, view sync status, manage OAuth2 clients, and track your fee earnings.
              </p>
            </div>
          </div>
        </motion.div>

        {/* Linux ARM64 Download */}
        <motion.div
          className="p-8 bg-gradient-to-br from-emerald-500/20 to-quantum-green/20 backdrop-blur-xl border border-emerald-500/30 rounded-2xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
        >
          <div className="flex items-center gap-4 mb-6">
            <div className="w-16 h-16 bg-emerald-500/20 rounded-xl flex items-center justify-center">
              <Cpu className="w-8 h-8 text-emerald-400" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">Linux ARM64</h2>
              <p className="text-gray-200">Raspberry Pi 4+ / AWS Graviton / Apple M-series</p>
            </div>
          </div>

          <div className="space-y-3 mb-6">
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">Native ARM64 Binary</p>
                <p className="text-sm text-gray-200">No emulation — compiled directly for aarch64 architecture</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">Low-Power Validator</p>
                <p className="text-sm text-gray-200">Run a full node on Raspberry Pi, Oracle Cloud, or ARM VPS</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-cyan flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">Full Feature Parity</p>
                <p className="text-sm text-gray-200">Same TUI, P2P sync, mining, and DeFi stack as x86_64</p>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <a
              href="https://quillon.xyz/downloads/q-api-server-v10.4.12-linux-arm64"
              download="q-api-server-arm64"
              className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-gradient-to-r from-emerald-500 to-quantum-green rounded-xl font-bold text-white hover:shadow-lg hover:shadow-emerald-500/50 transition-all"
            >
              <Download className="w-5 h-5" />
              Download ARM64 Binary (v10.4.12)
            </a>
            <p className="text-center text-sm text-gray-200">
              Size: ~75 MB | aarch64-unknown-linux-gnu
            </p>
          </div>

          <div className="mt-6 p-4 bg-quantum-dark/50 rounded-xl border border-emerald-500/20">
            <p className="text-sm font-mono text-gray-100 mb-2">Quick Start (ARM64):</p>
            <pre className="text-xs text-emerald-400 overflow-x-auto">
{`wget https://quillon.xyz/downloads/q-api-server-v10.4.12-linux-arm64
chmod +x q-api-server-v10.4.12-linux-arm64
./q-api-server-v10.4.12-linux-arm64 --port 8080 --tui --admin-wallet YOUR_WALLET_ADDRESS`}
            </pre>
            <p className="text-xs text-gray-300 mt-2">
              Tested on: Raspberry Pi 4/5, AWS Graviton, Oracle Ampere, Apple M1/M2 (Linux VM)
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
              <p className="text-gray-200">Windows 10+ / Windows Server 2019+</p>
            </div>
          </div>

          <div className="space-y-3 mb-6">
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">Single EXE — No Install Needed</p>
                <p className="text-sm text-gray-200">Download and run. No ZIP extraction, no DLLs required.</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">Full Node + TUI + P2P Sync</p>
                <p className="text-sm text-gray-200">Block validation, wallet, DEX, mining, and TUI dashboard built-in</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-quantum-cyan flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-white font-medium">Post-Quantum Cryptography</p>
                <p className="text-sm text-gray-200">Dilithium5 + Kyber1024 signatures built-in</p>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <a
              href="https://quillon.xyz/downloads/q-api-server-v10.4.12-windows-x64.exe"
              download="q-api-server-v10.4.12-windows-x64.exe"
              className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-gradient-to-r from-quantum-purple to-quantum-pink rounded-xl font-bold text-white hover:shadow-lg hover:shadow-quantum-purple/50 transition-all"
            >
              <Download className="w-5 h-5" />
              Download Windows Node (v10.4.12)
            </a>
            <p className="text-center text-sm text-gray-200">
              Size: ~70 MB | Single EXE — Full Node + TUI + DeFi Stack
            </p>
          </div>

          {/* Quick Start */}
          <div className="mt-6 p-4 bg-quantum-dark/50 rounded-xl border border-quantum-purple/20">
            <p className="text-sm font-mono text-gray-100 mb-2">Quick Start (PowerShell):</p>
            <pre className="text-xs text-quantum-purple overflow-x-auto">
{`# Download from browser or PowerShell:
Invoke-WebRequest -Uri https://quillon.xyz/downloads/q-api-server-v10.4.12-windows-x64.exe -OutFile q-api-server.exe
.\\q-api-server.exe --port 8080 --tui --admin-wallet YOUR_WALLET_ADDRESS`}
            </pre>
            <p className="text-xs text-gray-100 mt-2">
              Tip: Use port 9090 to avoid permission issues, or run as Administrator for port 8080
            </p>
          </div>
        </motion.div>
      </div>

      {/* Miner Download Section */}
      <motion.div
        className="p-8 rounded-2xl border border-amber-500/60 relative overflow-hidden"
        style={{ backgroundColor: '#1a1225' }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35 }}
      >
        <div className="relative">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-16 h-16 rounded-xl flex items-center justify-center" style={{ backgroundColor: '#3d2200' }}>
              <Pickaxe className="w-8 h-8 text-orange-400" />
            </div>
            <div>
              <div className="flex items-center gap-3">
                <h2 className="text-2xl font-bold text-orange-300">Q-Miner</h2>
                <span className="px-3 py-1 text-xs font-bold rounded-full uppercase animate-pulse text-yellow-200" style={{ backgroundColor: '#7c3a00' }}>Solo Mining</span>
              </div>
              <p className="text-orange-100">Standalone mining client for Q-NarwhalKnight network</p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <Monitor className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-orange-300 font-semibold">GPU + CPU Mining (OpenCL)</p>
                  <p className="text-sm text-orange-100">Multi-GPU OpenCL mining alongside multi-threaded CPU — auto-detects NVIDIA, AMD, Intel GPUs</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <Cpu className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-cyan-300 font-semibold">Multi-threaded CPU Mining</p>
                  <p className="text-sm text-orange-100">Scales across all available cores with adaptive difficulty</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <Zap className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-yellow-300 font-semibold">Adaptive Work Sizing</p>
                  <p className="text-sm text-orange-100">GPU dispatches auto-tune to keep 100-400ms per kernel for maximum throughput</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <Wallet className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-purple-300 font-semibold">OAuth2 Zero-Config Login</p>
                  <p className="text-sm text-orange-100">Double-click to mine — approve in browser, no mnemonic paste needed</p>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              {/* Linux GPU Miner — Native */}
              <div className="p-3 rounded-xl border border-yellow-600" style={{ backgroundColor: '#2a1a00' }}>
                <p className="text-xs text-yellow-300 font-bold mb-2 uppercase tracking-wider">Linux GPU + CPU Miner — Native (AVX2)</p>
                <a
                  href="https://quillon.xyz/downloads/q-miner-gpu-linux-x86_64"
                  download="q-miner-gpu-linux-x86_64"
                  className="w-full flex items-center justify-center gap-3 px-6 py-3 bg-gradient-to-r from-yellow-600 to-orange-600 rounded-xl font-bold text-yellow-100 hover:shadow-lg hover:shadow-yellow-500/50 transition-all"
                >
                  <Download className="w-5 h-5" />
                  Linux x64 — GPU + CPU Miner (Native)
                </a>
                <p className="text-xs text-yellow-200 mt-1 text-center">OpenCL GPU mining + native CPU optimizations (AVX2, LTO) — up to 30% faster hashing</p>
              </div>

              {/* Linux GPU Miner — Portable */}
              <div className="p-3 rounded-xl border border-gray-600" style={{ backgroundColor: '#15101e' }}>
                <p className="text-xs text-orange-200 font-bold mb-2 uppercase tracking-wider">Linux GPU + CPU Miner — Portable (Older CPUs)</p>
                <a
                  href="https://quillon.xyz/downloads/q-miner-gpu-linux-x64"
                  download="q-miner-gpu-linux-x64"
                  className="w-full flex items-center justify-center gap-3 px-5 py-3 bg-gradient-to-r from-orange-700 to-rose-700 rounded-xl font-bold text-orange-100 hover:shadow-lg hover:shadow-orange-500/30 transition-all text-sm"
                >
                  <Download className="w-5 h-5" />
                  Linux x64 — GPU + CPU Miner (Portable)
                </a>
                <p className="text-xs text-orange-200 mt-1 text-center">Compatible with all x86-64 CPUs (2009+). Use this if the native version crashes.</p>
              </div>

              {/* Windows GPU Miner */}
              <a
                href="https://quillon.xyz/downloads/q-miner-gpu-windows-x64.exe"
                download="q-miner-gpu-windows-x64.exe"
                className="w-full flex items-center justify-center gap-3 px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl font-bold text-purple-100 hover:shadow-lg hover:shadow-purple-500/50 transition-all"
              >
                <Download className="w-5 h-5" />
                Download GPU + CPU Miner — Windows x64 (v10.3.8)
              </a>
              <p className="text-center text-xs text-orange-200">v10.3.8 — OpenCL GPU mining + AVX2 SIMD + CPU core pinning + P2P peer compute</p>

              {/* Universal wget download */}
              <div className="p-4 rounded-xl border border-amber-600" style={{ backgroundColor: '#0d0a00' }}>
                <p className="text-sm font-mono text-amber-200 font-bold mb-2">Quick Download (wget):</p>
                <pre className="text-xs text-amber-300 overflow-x-auto whitespace-pre-wrap select-all cursor-pointer p-2 rounded" style={{ backgroundColor: '#000000' }}>
{`wget https://quillon.xyz/downloads/q-miner-gpu-linux-x86_64 && chmod +x q-miner-gpu-linux-x86_64`}
                </pre>
              </div>

              <div className="p-4 rounded-xl border border-orange-600" style={{ backgroundColor: '#0d0a00' }}>
                <p className="text-sm font-mono text-orange-200 font-bold mb-2">Linux Quick Start (GPU + CPU):</p>
                <pre className="text-xs text-amber-300 overflow-x-auto whitespace-pre-wrap" style={{ backgroundColor: '#000000', padding: '8px', borderRadius: '6px' }}>
{`wget https://quillon.xyz/downloads/q-miner-gpu-linux-x86_64
chmod +x q-miner-gpu-linux-x86_64
./q-miner-gpu-linux-x86_64 \\
  --mode solo \\
  --wallet YOUR_WALLET_ADDRESS \\
  --threads 4 \\
  --server https://quillon.xyz`}
                </pre>
              </div>
              <div className="p-4 rounded-xl border border-purple-600" style={{ backgroundColor: '#0a0515' }}>
                <p className="text-sm font-mono text-purple-200 font-bold mb-2">Windows Quick Start (PowerShell):</p>
                <pre className="text-xs text-purple-300 overflow-x-auto whitespace-pre-wrap" style={{ backgroundColor: '#000000', padding: '8px', borderRadius: '6px' }}>
{`# Download from browser or:
Invoke-WebRequest -Uri https://quillon.xyz/downloads/q-miner-gpu-windows-x64.exe -OutFile q-miner-gpu.exe
.\\q-miner-gpu.exe --mode solo --wallet YOUR_WALLET_ADDRESS --threads 4 --server https://quillon.xyz`}
                </pre>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Slint Native Wallet */}
      <motion.div
        className="p-8 bg-gradient-to-br from-emerald-500/30 via-teal-500/20 to-quantum-cyan/20 backdrop-blur-xl border border-emerald-500/40 rounded-2xl relative overflow-hidden"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.37 }}
      >
        <div className="absolute top-0 right-0 w-48 h-48 bg-emerald-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2"></div>
        <div className="absolute bottom-0 left-0 w-32 h-32 bg-teal-500/10 rounded-full blur-2xl translate-y-1/2 -translate-x-1/2"></div>

        <div className="relative">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-16 h-16 bg-gradient-to-br from-emerald-400/40 to-teal-500/40 rounded-xl flex items-center justify-center">
              <Wallet className="w-8 h-8 text-emerald-300" />
            </div>
            <div>
              <div className="flex items-center gap-3">
                <h2 className="text-2xl font-bold text-white">Slint Native Wallet</h2>
                <span className="px-3 py-1 bg-emerald-500/40 text-white text-xs font-bold rounded-full uppercase animate-pulse">GPU MINING</span>
              </div>
              <p className="text-white">Lightweight native desktop wallet with built-in CPU + GPU mining</p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-emerald-200 font-semibold">Native Performance</p>
                  <p className="text-sm text-white">Built with Slint UI framework in pure Rust — instant startup, minimal resources</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <Monitor className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-orange-200 font-semibold">GPU Mining (OpenCL)</p>
                  <p className="text-sm text-white">Mine with your GPU directly from the wallet — auto-detects NVIDIA, AMD, and Intel GPUs via OpenCL. Runs alongside CPU mining for maximum hashrate.</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-teal-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-teal-200 font-semibold">Send, Receive, PoS & Multi-Token</p>
                  <p className="text-sm text-white">Full wallet with Point of Sale mode, QR payment requests, transaction history, token selector, and built-in miner</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-cyan-200 font-semibold">Connect to Any Node</p>
                  <p className="text-sm text-white">Points to quillon.xyz by default or your own local node</p>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <a
                href="https://quillon.xyz/downloads/slint-wallet-linux-x86_64"
                download="slint-wallet-linux-x86_64"
                className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-xl font-bold text-white hover:shadow-lg hover:shadow-emerald-500/50 transition-all"
              >
                <Download className="w-5 h-5" />
                Download Wallet — Linux x64 (GPU + CPU Mining)
              </a>
              <a
                href="https://quillon.xyz/downloads/slint-wallet-gpu-windows-x64.exe"
                download="slint-wallet-gpu-windows-x64.exe"
                className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-gradient-to-r from-teal-500 to-cyan-500 rounded-xl font-bold text-white hover:shadow-lg hover:shadow-teal-500/50 transition-all"
              >
                <Download className="w-5 h-5" />
                Download Wallet — Windows x64 (GPU + CPU Mining)
              </a>
              <p className="text-center text-xs text-gray-300">v10.3.8 — Linux: ~22 MB | Windows: ~16 MB — GPU Mining, Point of Sale, OpenGL auto-fallback</p>

              <div className="p-3 bg-orange-500/10 rounded-xl border border-orange-400/30">
                <p className="text-xs text-orange-200 font-bold mb-1">GPU Mining Requirements</p>
                <p className="text-xs text-gray-300">Any GPU with OpenCL support (NVIDIA, AMD, Intel). GPU drivers install OpenCL automatically. No extra setup needed — the wallet detects GPUs on startup.</p>
              </div>

              <div className="p-4 bg-black/30 rounded-xl border border-emerald-400/30">
                <p className="text-sm font-mono text-white mb-2">Linux Quick Start:</p>
                <pre className="text-xs text-white overflow-x-auto whitespace-pre-wrap">
{`wget https://quillon.xyz/downloads/slint-wallet-linux-x86_64
chmod +x slint-wallet-linux-x86_64
./slint-wallet-linux-x86_64`}
                </pre>
              </div>
              <div className="p-4 bg-black/30 rounded-xl border border-teal-400/30">
                <p className="text-sm font-mono text-white mb-2">Windows:</p>
                <pre className="text-xs text-white overflow-x-auto whitespace-pre-wrap">
{`# Download slint-wallet-gpu-windows-x64.exe
# Double-click to launch — no install needed
# GPU mining auto-detects your graphics card`}
                </pre>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* macOS Build from Source */}
      <motion.div
        className="p-6 bg-gradient-to-br from-quantum-green/30 to-quantum-cyan/20 backdrop-blur-xl border border-quantum-green/40 rounded-2xl"
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
            <p className="text-gray-200 text-sm">Intel & Apple Silicon</p>
          </div>
        </div>

        <div className="p-4 bg-white/10 rounded-xl border border-quantum-green/20">
          <pre className="text-xs text-quantum-green overflow-x-auto">
{`# Install Rust if needed: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
git clone https://code.quillon.xyz/repo.git q-narwhalknight && cd q-narwhalknight
cargo build --release --package q-api-server
./target/release/q-api-server --port 8080 --tui`}
          </pre>
        </div>
      </motion.div>

      {/* HiveOS One-Line Install */}
      <motion.div
        className="p-6 bg-gradient-to-br from-yellow-500/30 to-amber-600/20 backdrop-blur-xl border border-yellow-500/40 rounded-2xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.45 }}
      >
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-yellow-500/20 rounded-xl flex items-center justify-center">
            <Cpu className="w-6 h-6 text-yellow-400" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">HiveOS</h2>
            <p className="text-gray-200 text-sm">One-line install for mining rigs</p>
          </div>
        </div>

        <p className="text-gray-200 text-sm mb-4">
          Install and start mining on HiveOS rigs. Choose either the one-line installer or set up directly via Flight Sheet.
        </p>

        <div className="p-4 bg-white/10 rounded-xl border border-yellow-500/20 mb-4">
          <p className="text-sm font-bold text-yellow-300 mb-2">Option 1: One-Line Install (SSH)</p>
          <p className="text-xs text-gray-100 mb-2">SSH into your rig and run:</p>
          <pre className="text-xs text-yellow-300 overflow-x-auto whitespace-pre-wrap select-all cursor-pointer bg-black/30 p-2 rounded">{`curl -sL https://quillon.xyz/downloads/install-hiveos.sh | bash`}</pre>
        </div>

        <div className="p-4 bg-white/10 rounded-xl border border-yellow-500/20">
          <p className="text-sm font-bold text-yellow-300 mb-3">Option 2: Flight Sheet Setup</p>

          <div className="mb-3 p-3 bg-red-500/20 border border-red-400/40 rounded-lg">
            <p className="text-xs text-red-200 font-bold">IMPORTANT: Use the .tar.gz URL below, NOT the raw binary URL. HiveOS requires a .tar.gz archive.</p>
          </div>

          <div className="mb-3 p-3 bg-black/40 border border-yellow-400/50 rounded-lg">
            <p className="text-xs text-white mb-1 font-semibold">Installation URL (copy this exactly):</p>
            <pre className="text-sm text-yellow-300 font-mono select-all cursor-pointer break-all">https://quillon.xyz/downloads/q-miner-hiveos.tar.gz</pre>
          </div>

          <ol className="space-y-2 text-sm text-gray-200">
            <li><span className="text-yellow-400 font-mono mr-2">1.</span>Go to HiveOS Dashboard &rarr; <span className="text-white font-medium">Flight Sheets</span> &rarr; Add New</li>
            <li><span className="text-yellow-400 font-mono mr-2">2.</span>Coin: <span className="text-white font-medium">Custom</span></li>
            <li><span className="text-yellow-400 font-mono mr-2">3.</span>Wallet: <span className="text-white font-medium">Your QNK wallet address</span> (qnk...)</li>
            <li><span className="text-yellow-400 font-mono mr-2">4.</span>Pool URL: <span className="font-mono text-yellow-300">https://quillon.xyz</span></li>
            <li><span className="text-yellow-400 font-mono mr-2">5.</span>Miner: <span className="text-white font-medium">Custom</span>
              <ul className="ml-6 mt-1 space-y-1 text-gray-100 text-xs">
                <li>Installation URL: <span className="font-mono text-yellow-300 font-bold select-all">https://quillon.xyz/downloads/q-miner-hiveos.tar.gz</span></li>
                <li>Hash algorithm: <span className="font-mono text-yellow-300">qnk-dagknight</span></li>
                <li>Wallet template: <span className="font-mono text-yellow-300">%WAL%</span></li>
                <li>Pool URL: <span className="font-mono text-yellow-300">https://quillon.xyz</span></li>
              </ul>
            </li>
            <li><span className="text-yellow-400 font-mono mr-2">6.</span>Apply flight sheet to your rig</li>
          </ol>
          <p className="text-xs text-gray-100 mt-3">Optional extra config JSON: <code className="text-yellow-300 bg-black/30 px-2 py-0.5 rounded">{`{"threads": 8}`}</code></p>
        </div>

        <p className="text-gray-100 text-xs mt-3">
          Supports HiveOS 0.6+. Auto-detects CPU threads and configures optimal mining settings. Package includes h-run.sh and h-stats.sh for HiveOS dashboard integration.
        </p>
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
            <ul className="space-y-1 text-gray-100 text-sm">
              <li>CPU: 4 cores @ 2.5GHz</li>
              <li>RAM: 8 GB</li>
              <li>Storage: 50 GB SSD</li>
              <li>Network: 10 Mbps</li>
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-bold text-quantum-purple mb-3">Recommended</h3>
            <ul className="space-y-1 text-gray-100 text-sm">
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
          <p className="text-sm text-gray-200">View & contribute</p>
        </a>

        <a
          href="https://api.quillon.xyz"
          target="_blank"
          rel="noopener noreferrer"
          className="p-4 bg-quantum-dark/50 rounded-xl border border-quantum-purple/20 hover:border-quantum-purple/50 transition-all"
        >
          <BookOpen className="w-6 h-6 text-quantum-purple mb-2" />
          <h3 className="font-bold text-white mb-1">API Docs</h3>
          <p className="text-sm text-gray-200">REST & WebSocket</p>
        </a>

        <div className="p-4 bg-white/10 rounded-xl border border-quantum-green/20">
          <Shield className="w-6 h-6 text-quantum-green mb-2" />
          <h3 className="font-bold text-white mb-1">PQC Security</h3>
          <p className="text-sm text-gray-200">Dilithium5 + Kyber1024</p>
        </div>
      </motion.div>
    </div>
  );
}
