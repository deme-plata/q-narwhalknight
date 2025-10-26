import { motion } from 'framer-motion';
import { Pickaxe, Download, Cpu, Zap, Award, TrendingUp, AlertCircle, ExternalLink, Terminal } from 'lucide-react';
import MiningDashboard from './MiningDashboard';

export default function MiningScreen() {
  const walletAddress = localStorage.getItem('walletAddress') || '';

  const handleDownloadMiner = (platform: 'linux' | 'windows' | 'linux-optimized' | 'windows-optimized') => {
    // Link to download the miner binary
    if (platform === 'windows-optimized') {
      window.open('/downloads/q-narwhalknight-miner-v1.1.0-windows-x64.zip', '_blank');
    } else if (platform === 'windows') {
      window.open('/downloads/q-miner-windows-x64.exe', '_blank');
    } else if (platform === 'linux-optimized') {
      window.open('/downloads/q-narwhalknight-miner-v1.1.0-linux-x86_64.tar.gz', '_blank');
    } else {
      window.open('/downloads/q-miner-linux-x64', '_blank');
    }
  };

  const copyCommand = (command: string) => {
    navigator.clipboard.writeText(command);
  };

  const miningCommand = `./q-miner --mode solo --wallet ${walletAddress} --threads 4 --intensity 7 --server http://185.182.185.227:8080`;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <div className="p-3 rainbow-box rounded-xl">
          <Pickaxe className="w-8 h-8 text-white" />
        </div>
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-quantum-cyan to-quantum-purple bg-clip-text text-transparent">
            Quantum Mining
          </h1>
          <p className="text-gray-400">
            Mine QUG with Austrian Economics & DAG-Knight VDF
          </p>
        </div>
      </div>

      {/* Mining Dashboard with Real-Time SSE Updates */}
      {walletAddress && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <MiningDashboard />
        </motion.div>
      )}

      {/* Austrian Economics Notice */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-quantum-yellow/10 to-quantum-orange/10 border border-quantum-yellow/30 rounded-xl p-6"
      >
        <div className="flex items-start gap-4">
          <AlertCircle className="w-6 h-6 text-quantum-yellow flex-shrink-0 mt-1" />
          <div>
            <h3 className="text-lg font-bold text-quantum-yellow mb-2">Austrian Economics Enabled</h3>
            <div className="space-y-2 text-gray-300 text-sm">
              <p>
                <strong>Fixed Supply:</strong> 21,000,000 QUG total (hard cap, immutable)
              </p>
              <p>
                <strong>Block Reward:</strong> 0.5 QUG initially, halves every 210,000 blocks (~4 years)
              </p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Download Miner Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6"
      >
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Download className="w-6 h-6 text-quantum-cyan" />
          Download Q-NarwhalKnight Miner
        </h2>

        <div className="space-y-4">
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-cyan/20">
              <div className="flex items-center gap-3 mb-3">
                <Cpu className="w-5 h-5 text-quantum-green" />
                <span className="font-bold text-white">CPU Mining</span>
              </div>
              <p className="text-gray-400 text-sm mb-3">
                Optimized for multi-core CPUs with AVX2/AVX-512 acceleration
              </p>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>✓ Multi-threaded support</li>
                <li>✓ Blake3 + VDF algorithm</li>
                <li>✓ Real-time hash rate monitoring</li>
              </ul>
            </div>

            <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-purple/20 opacity-60">
              <div className="flex items-center gap-3 mb-3">
                <Zap className="w-5 h-5 text-quantum-purple" />
                <span className="font-bold text-white">GPU Mining</span>
                <span className="text-xs bg-quantum-purple/20 text-quantum-purple px-2 py-1 rounded">Coming Soon</span>
              </div>
              <p className="text-gray-400 text-sm mb-3">
                CUDA, OpenCL, and Vulkan support (in development)
              </p>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>⏳ NVIDIA GPU support</li>
                <li>⏳ AMD GPU support</li>
                <li>⏳ Parallel VDF computation</li>
              </ul>
            </div>
          </div>

          {/* NEW OPTIMIZED LINUX MINER - Highlighted */}
          <div className="mb-6 p-6 bg-gradient-to-r from-quantum-green/20 to-quantum-cyan/20 border-2 border-quantum-green/50 rounded-xl">
            <div className="flex items-center gap-3 mb-4">
              <Zap className="w-6 h-6 text-quantum-green" />
              <div>
                <h3 className="text-xl font-bold text-white">
                  🔥 NEW: Optimized Linux Miner v1.1.0
                </h3>
                <p className="text-sm text-quantum-green">+15% Hash Rate • 99% CPU Utilization • Zero-Allocation VDF</p>
              </div>
            </div>

            <div className="grid md:grid-cols-3 gap-4 mb-4">
              <div className="bg-quantum-dark/50 rounded-lg p-3">
                <p className="text-quantum-green font-bold text-sm">⚡ 166 KH/s</p>
                <p className="text-gray-400 text-xs">vs 144 KH/s (v1.0)</p>
              </div>
              <div className="bg-quantum-dark/50 rounded-lg p-3">
                <p className="text-quantum-green font-bold text-sm">99% CPU</p>
                <p className="text-gray-400 text-xs">Maximum performance</p>
              </div>
              <div className="bg-quantum-dark/50 rounded-lg p-3">
                <p className="text-quantum-green font-bold text-sm">10x Batches</p>
                <p className="text-gray-400 text-xs">Fewer context switches</p>
              </div>
            </div>

            <motion.button
              onClick={() => handleDownloadMiner('linux-optimized')}
              className="w-full bg-gradient-to-r from-quantum-green to-quantum-cyan hover:from-quantum-green/80 hover:to-quantum-cyan/80 text-white font-bold py-4 px-6 rounded-xl transition-all flex items-center justify-center gap-3 shadow-lg shadow-quantum-green/30"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Download className="w-5 h-5" />
              <span>Download Optimized Linux Package (v1.1.0)</span>
            </motion.button>
            <p className="text-center text-sm text-gray-400 mt-2">
              Size: 3.6 MB tar.gz • Complete package with README & start script
            </p>
          </div>

          {/* NEW OPTIMIZED WINDOWS MINER - Highlighted */}
          <div className="mb-6 p-6 bg-gradient-to-r from-quantum-purple/20 to-quantum-pink/20 border-2 border-quantum-purple/50 rounded-xl">
            <div className="flex items-center gap-3 mb-4">
              <Zap className="w-6 h-6 text-quantum-purple" />
              <div>
                <h3 className="text-xl font-bold text-white">
                  🔥 NEW: Optimized Windows Miner v1.1.0
                </h3>
                <p className="text-sm text-quantum-purple">+15% Hash Rate • 99% CPU Utilization • Easy PowerShell Script</p>
              </div>
            </div>

            <div className="grid md:grid-cols-3 gap-4 mb-4">
              <div className="bg-quantum-dark/50 rounded-lg p-3">
                <p className="text-quantum-purple font-bold text-sm">⚡ 166 KH/s</p>
                <p className="text-gray-400 text-xs">vs 144 KH/s (v1.0)</p>
              </div>
              <div className="bg-quantum-dark/50 rounded-lg p-3">
                <p className="text-quantum-purple font-bold text-sm">2.8 MB ZIP</p>
                <p className="text-gray-400 text-xs">Complete package</p>
              </div>
              <div className="bg-quantum-dark/50 rounded-lg p-3">
                <p className="text-quantum-purple font-bold text-sm">Auto-detect</p>
                <p className="text-gray-400 text-xs">CPU cores & features</p>
              </div>
            </div>

            <motion.button
              onClick={() => handleDownloadMiner('windows-optimized')}
              className="w-full bg-gradient-to-r from-quantum-purple to-quantum-pink hover:from-quantum-purple/80 hover:to-quantum-pink/80 text-white font-bold py-4 px-6 rounded-xl transition-all flex items-center justify-center gap-3 shadow-lg shadow-quantum-purple/30"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Download className="w-5 h-5" />
              <span>Download Optimized Windows Package (v1.1.0)</span>
            </motion.button>
            <p className="text-center text-sm text-gray-400 mt-2">
              Size: 2.8 MB zip • Includes q-miner.exe + PowerShell script + README
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <motion.button
              onClick={() => handleDownloadMiner('linux')}
              className="bg-gradient-to-r from-quantum-cyan/70 to-quantum-blue/70 hover:from-quantum-cyan/60 hover:to-quantum-blue/60 text-white font-bold py-4 px-6 rounded-xl transition-all flex flex-col items-center justify-center gap-2 opacity-75"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="flex items-center gap-3">
                <Download className="w-5 h-5" />
                <span>Linux x86_64 (Legacy)</span>
              </div>
              <span className="text-xs text-quantum-cyan/70">13 MB • CPU Mining • v1.0</span>
            </motion.button>

            <motion.button
              onClick={() => handleDownloadMiner('windows')}
              className="bg-gradient-to-r from-quantum-purple/70 to-quantum-pink/70 hover:from-quantum-purple/60 hover:to-quantum-pink/60 text-white font-bold py-4 px-6 rounded-xl transition-all flex flex-col items-center justify-center gap-2 opacity-75"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="flex items-center gap-3">
                <Download className="w-5 h-5" />
                <span>Windows x64 (Legacy)</span>
              </div>
              <span className="text-xs text-quantum-purple/70">22 MB • CPU Mining • v1.0</span>
            </motion.button>

            <motion.button
              onClick={() => window.open('/downloads/q-miner-macos-x64', '_blank')}
              className="bg-gradient-to-r from-quantum-green to-quantum-cyan hover:from-quantum-green/80 hover:to-quantum-cyan/80 text-white font-bold py-4 px-6 rounded-xl transition-all flex flex-col items-center justify-center gap-2"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="flex items-center gap-3">
                <Download className="w-5 h-5" />
                <span>macOS Intel (x64)</span>
              </div>
              <span className="text-xs text-quantum-green/70">8.7 MB • CPU Mining</span>
            </motion.button>

            <motion.button
              onClick={() => window.open('/downloads/q-miner-macos-arm64', '_blank')}
              className="bg-gradient-to-r from-quantum-orange to-quantum-yellow hover:from-quantum-orange/80 hover:to-quantum-yellow/80 text-white font-bold py-4 px-6 rounded-xl transition-all flex flex-col items-center justify-center gap-2"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="flex items-center gap-3">
                <Download className="w-5 h-5" />
                <span>macOS Apple Silicon (ARM64)</span>
              </div>
              <span className="text-xs text-quantum-orange/70">8.3 MB • CPU Mining</span>
            </motion.button>
          </div>
        </div>
      </motion.div>

      {/* Quick Start Guide */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6"
      >
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Terminal className="w-6 h-6 text-quantum-green" />
          Quick Start Guide
        </h2>

        <div className="space-y-4">
          <div>
            <p className="text-gray-300 mb-2">1. Download the miner for your platform:</p>
            <div className="space-y-2">
              <div className="bg-gradient-to-r from-quantum-green/20 to-quantum-cyan/10 border border-quantum-green/30 rounded-lg p-3">
                <p className="text-quantum-green text-sm font-bold mb-1">🚀 Linux (Optimized v1.1.0 - RECOMMENDED):</p>
                <code className="font-mono text-xs text-gray-300 block mb-1">tar -xzf q-narwhalknight-miner-v1.1.0-linux-x86_64.tar.gz</code>
                <code className="font-mono text-xs text-gray-300 block mb-1">cd linux</code>
                <code className="font-mono text-xs text-gray-300">./start-mining.sh {walletAddress || 'YOUR_WALLET_ADDRESS'}</code>
                <p className="text-xs text-quantum-green mt-2">✨ Includes easy start script with auto-configuration!</p>
              </div>
              <div className="bg-gradient-to-r from-quantum-purple/20 to-quantum-pink/10 border border-quantum-purple/30 rounded-lg p-3">
                <p className="text-quantum-purple text-sm font-bold mb-1">🚀 Windows (Optimized v1.1.0 - RECOMMENDED):</p>
                <code className="font-mono text-xs text-gray-300 block mb-1">Expand-Archive q-narwhalknight-miner-v1.1.0-windows-x64.zip</code>
                <code className="font-mono text-xs text-gray-300 block mb-1">cd windows</code>
                <code className="font-mono text-xs text-gray-300">.\start-mining.ps1 {walletAddress || 'YOUR_WALLET_ADDRESS'}</code>
                <p className="text-xs text-quantum-purple mt-2">✨ PowerShell script with auto-detection!</p>
              </div>
              <div className="bg-quantum-dark/50 rounded-lg p-3">
                <p className="text-quantum-cyan text-sm mb-1">🐧 Linux (Legacy):</p>
                <code className="font-mono text-xs text-gray-300">chmod +x q-miner-linux-x64 && mv q-miner-linux-x64 q-miner</code>
              </div>
              <div className="bg-quantum-dark/50 rounded-lg p-3">
                <p className="text-gray-400 text-sm mb-1">🪟 Windows (Legacy):</p>
                <code className="font-mono text-xs text-gray-300">Rename q-miner-windows-x64.exe to q-miner.exe</code>
              </div>
              <div className="bg-quantum-dark/50 rounded-lg p-3">
                <p className="text-quantum-green text-sm mb-1">🍎 macOS Intel:</p>
                <code className="font-mono text-xs text-gray-300">chmod +x q-miner-macos-x64 && mv q-miner-macos-x64 q-miner</code>
              </div>
              <div className="bg-quantum-dark/50 rounded-lg p-3">
                <p className="text-quantum-orange text-sm mb-1">🍎 macOS Apple Silicon:</p>
                <code className="font-mono text-xs text-gray-300">chmod +x q-miner-macos-arm64 && mv q-miner-macos-arm64 q-miner</code>
              </div>
            </div>
          </div>

          <div>
            <p className="text-gray-300 mb-2">2. Run the miner with your wallet address:</p>
            <div className="bg-quantum-dark/50 rounded-lg p-3 font-mono text-sm text-quantum-green border border-quantum-green/20 relative">
              <code className="block overflow-x-auto">{miningCommand}</code>
              <button
                onClick={() => copyCommand(miningCommand)}
                className="absolute top-2 right-2 bg-quantum-green/20 hover:bg-quantum-green/30 text-quantum-green px-2 py-1 rounded text-xs transition-colors"
              >
                Copy
              </button>
            </div>
          </div>

          <div>
            <p className="text-gray-300 mb-2">3. Optional parameters:</p>
            <div className="bg-quantum-dark/50 rounded-lg p-3 text-sm text-gray-300 space-y-1">
              <p><code className="text-quantum-cyan">--threads 4</code> - Number of CPU threads to use (0 = all cores)</p>
              <p><code className="text-quantum-cyan">--intensity 7</code> - Mining intensity (1-10)</p>
              <p><code className="text-quantum-cyan">--server http://185.182.185.227:8080</code> - Server URL (defaults to localhost:8080)</p>
            </div>
          </div>

          <div className="bg-quantum-purple/10 border border-quantum-purple/30 rounded-lg p-4">
            <p className="text-quantum-purple font-bold mb-2">💡 Pro Tips:</p>
            <ul className="text-gray-300 text-sm space-y-1">
              <li>• <strong className="text-quantum-green">v1.1.0 optimized miner</strong> delivers 15% higher hash rates with zero-allocation VDF</li>
              <li>• Use <code className="text-quantum-cyan">--intensity 10</code> for maximum 99% CPU utilization</li>
              <li>• The easy start script auto-detects your CPU cores and capabilities (AVX2, etc.)</li>
              <li>• Monitor hash rate in real-time - miner shows statistics every 5 seconds</li>
              <li>• Mining rewards appear instantly in your wallet balance (SSE real-time updates)</li>
              <li>• Close other applications when mining at intensity 10 for best performance</li>
            </ul>
          </div>
        </div>
      </motion.div>

      {/* Mining Statistics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="grid md:grid-cols-3 gap-4"
      >
        <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-green/30 rounded-xl p-6">
          <div className="flex items-center justify-between mb-3">
            <Award className="w-6 h-6 text-quantum-green" />
            <span className="text-2xl font-bold text-quantum-green">0.5 QUG</span>
          </div>
          <p className="text-gray-300 text-sm">Current Block Reward</p>
          <p className="text-gray-500 text-xs mt-1">Halves every 210,000 blocks</p>
        </div>

        <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-cyan/30 rounded-xl p-6">
          <div className="flex items-center justify-between mb-3">
            <TrendingUp className="w-6 h-6 text-quantum-cyan" />
            <span className="text-2xl font-bold text-quantum-cyan">21M</span>
          </div>
          <p className="text-gray-300 text-sm">Total Supply Cap</p>
          <p className="text-gray-500 text-xs mt-1">Fixed, immutable hard cap</p>
        </div>

        <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6">
          <div className="flex items-center justify-between mb-3">
            <Zap className="w-6 h-6 text-quantum-purple" />
            <span className="text-2xl font-bold text-quantum-purple">1s</span>
          </div>
          <p className="text-gray-300 text-sm">Target Block Time</p>
          <p className="text-gray-500 text-xs mt-1">After bootstrap phase</p>
        </div>
      </motion.div>

      {/* Learn More */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6"
      >
        <h3 className="text-lg font-bold text-white mb-3">Learn More About Q-NarwhalKnight Mining</h3>
        <div className="space-y-2">
          <a
            href="https://code.quillon.xyz/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-quantum-cyan hover:text-quantum-cyan/80 transition-colors"
          >
            <ExternalLink className="w-4 h-4" />
            GitHub Repository
          </a>
          <a
            href="/quantum-physics-whitepaper-full.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-quantum-cyan hover:text-quantum-cyan/80 transition-colors"
          >
            <ExternalLink className="w-4 h-4" />
            Quantum Consensus Whitepaper
          </a>
        </div>
      </motion.div>
    </div>
  );
}