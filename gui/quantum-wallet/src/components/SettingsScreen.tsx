import { useState } from 'react';
import { motion } from 'framer-motion';
import { Shield, Palette, Activity, Globe, Lock, Eye, Zap, LogOut } from 'lucide-react';

interface SettingsScreenProps {
  onLogout?: () => void;
}

export default function SettingsScreen({ onLogout }: SettingsScreenProps) {
  const [activeTab, setActiveTab] = useState('crypto');
  const [cryptoSuite, setCryptoSuite] = useState('Q1');
  const [visualEffects, setVisualEffects] = useState({
    entanglementMoire: true,
    photonWaterfall: true,
    rainbowBoxes: true,
    fractalOverlay: true,
  });

  const tabs = [
    { id: 'crypto', label: 'Crypto Agility', icon: Shield },
    { id: 'visuals', label: 'Quantum Visuals', icon: Palette },
    { id: 'performance', label: 'Performance', icon: Activity },
    { id: 'network', label: 'Network', icon: Globe },
  ];

  const cryptoSuites = [
    { 
      id: 'Q0', 
      name: 'Q0 Classical', 
      description: 'Ed25519 + ECDH + QUIC',
      status: 'legacy',
      color: 'text-gray-400'
    },
    { 
      id: 'Q1', 
      name: 'Q1 Post-Quantum', 
      description: 'Dilithium5 + Kyber1024 + PQ-TLS',
      status: 'active',
      color: 'text-quantum-green'
    },
    { 
      id: 'Q2', 
      name: 'Q2 Full Quantum', 
      description: 'QKD + Quantum Signatures',
      status: 'future',
      color: 'text-quantum-cyan'
    },
  ];

  const performanceMetrics = [
    { label: 'Phase', value: 'Q1 Post-Quantum' },
    { label: 'Throughput', value: '48,234 TPS' },
    { label: 'Latency', value: '2.3s finality' },
    { label: 'Memory Usage', value: '234 MB' },
    { label: 'Network Bandwidth', value: '1.2 MB/s' },
  ];

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl lg:text-4xl font-bold text-white mb-2">Settings</h1>
        <p className="text-gray-400">Configure your quantum wallet experience</p>
      </div>

      {/* Tab Navigation */}
      <div className="bg-quantum-indigo/30 backdrop-blur-xl rounded-2xl p-2">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
          {tabs.map((tab) => (
            <motion.button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center justify-center gap-3 p-4 rounded-xl font-medium transition-all relative ${
                activeTab === tab.id
                  ? 'bg-gradient-to-r from-quantum-purple/30 to-quantum-cyan/30 text-white border border-quantum-cyan/30'
                  : 'text-gray-400 hover:text-white hover:bg-quantum-purple/20'
              }`}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              {activeTab === tab.id && (
                <motion.div
                  className="absolute inset-0 rainbow-box opacity-10 rounded-xl"
                  layoutId="tab-bg"
                />
              )}
              <tab.icon className="w-5 h-5" />
              <span className="hidden sm:block">{tab.label}</span>
            </motion.button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {/* Crypto Agility Tab */}
        {activeTab === 'crypto' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
              <h3 className="text-xl font-semibold mb-6 flex items-center gap-3">
                <Shield className="w-6 h-6 text-quantum-cyan" />
                Cryptographic Suite
              </h3>

              <div className="space-y-4">
                {cryptoSuites.map((suite) => (
                  <motion.label
                    key={suite.id}
                    className={`block p-4 rounded-xl border-2 cursor-pointer transition-all ${
                      cryptoSuite === suite.id
                        ? 'border-quantum-cyan bg-quantum-cyan/10'
                        : 'border-quantum-purple/20 hover:border-quantum-purple/40'
                    }`}
                    whileHover={{ scale: 1.02 }}
                  >
                    <input
                      type="radio"
                      name="cryptoSuite"
                      value={suite.id}
                      checked={cryptoSuite === suite.id}
                      onChange={(e) => setCryptoSuite(e.target.value)}
                      className="sr-only"
                    />
                    <div className="flex items-center justify-between">
                      <div>
                        <div className={`font-semibold ${suite.color}`}>{suite.name}</div>
                        <div className="text-sm text-gray-400">{suite.description}</div>
                      </div>
                      <div className={`text-xs px-2 py-1 rounded-full ${
                        suite.status === 'active' ? 'bg-quantum-green/20 text-quantum-green' :
                        suite.status === 'future' ? 'bg-quantum-cyan/20 text-quantum-cyan' :
                        'bg-gray-500/20 text-gray-400'
                      }`}>
                        {suite.status}
                      </div>
                    </div>
                  </motion.label>
                ))}
              </div>
            </div>

            <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
              <h3 className="text-xl font-semibold mb-6 flex items-center gap-3">
                <Lock className="w-6 h-6 text-quantum-green" />
                Security Status
              </h3>

              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-quantum-dark/30 rounded-xl">
                  <span>Quantum Resistance</span>
                  <span className="text-quantum-green font-semibold">Active</span>
                </div>
                <div className="flex items-center justify-between p-4 bg-quantum-dark/30 rounded-xl">
                  <span>Forward Secrecy</span>
                  <span className="text-quantum-green font-semibold">Enabled</span>
                </div>
                <div className="flex items-center justify-between p-4 bg-quantum-dark/30 rounded-xl">
                  <span>Key Rotation</span>
                  <span className="text-quantum-yellow font-semibold">Every 24h</span>
                </div>
                <div className="flex items-center justify-between p-4 bg-quantum-dark/30 rounded-xl">
                  <span>Threat Level</span>
                  <span className="text-quantum-green font-semibold">Minimal</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Quantum Visuals Tab */}
        {activeTab === 'visuals' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
              <h3 className="text-xl font-semibold mb-6 flex items-center gap-3">
                <Eye className="w-6 h-6 text-quantum-pink" />
                Visual Effects
              </h3>

              <div className="space-y-6">
                {Object.entries(visualEffects).map(([key, enabled]) => (
                  <div key={key} className="flex items-center justify-between">
                    <div>
                      <div className="font-medium capitalize">
                        {key.replace(/([A-Z])/g, ' $1').trim()}
                      </div>
                      <div className="text-sm text-gray-400">
                        {key === 'entanglementMoire' && 'Quantum entanglement visualization patterns'}
                        {key === 'photonWaterfall' && 'Animated photon detection streams'}
                        {key === 'rainbowBoxes' && 'Rainbow-colored quantum state indicators'}
                        {key === 'fractalOverlay' && 'Background fractal interference patterns'}
                      </div>
                    </div>
                    <motion.button
                      onClick={() => setVisualEffects(prev => ({ ...prev, [key]: !prev[key as keyof typeof prev] }))}
                      className={`w-12 h-6 rounded-full relative transition-all ${
                        enabled ? 'bg-quantum-cyan' : 'bg-gray-600'
                      }`}
                      whileTap={{ scale: 0.95 }}
                    >
                      <motion.div
                        className="w-5 h-5 bg-white rounded-full absolute top-0.5"
                        animate={{ x: enabled ? 26 : 2 }}
                        transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                      />
                    </motion.button>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
              <h3 className="text-xl font-semibold mb-6 flex items-center gap-3">
                <Palette className="w-6 h-6 text-quantum-yellow" />
                Preview
              </h3>

              <div className="relative h-48 bg-quantum-dark/50 rounded-xl overflow-hidden">
                {/* Live preview of effects */}
                <div className="absolute inset-0">
                  {visualEffects.fractalOverlay && <div className="fractal-overlay opacity-30" />}
                  
                  {visualEffects.rainbowBoxes && (
                    <div className="absolute top-4 left-4">
                      <div className="w-8 h-8 rainbow-box rounded-lg" />
                    </div>
                  )}

                  {visualEffects.photonWaterfall && (
                    <motion.div
                      className="absolute w-1 h-16 bg-gradient-to-b from-transparent via-quantum-cyan to-transparent"
                      animate={{ y: [0, 200] }}
                      transition={{ duration: 2, repeat: Infinity }}
                      style={{ left: '60%' }}
                    />
                  )}

                  {visualEffects.entanglementMoire && (
                    <svg className="absolute inset-0 w-full h-full">
                      <motion.circle
                        cx="50%"
                        cy="50%"
                        r="30"
                        fill="none"
                        stroke="rgba(107, 70, 193, 0.5)"
                        strokeWidth="2"
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: 1 }}
                        transition={{ duration: 2, repeat: Infinity }}
                      />
                    </svg>
                  )}
                </div>

                <div className="absolute bottom-4 left-4 text-sm text-gray-400">
                  Live Preview
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Performance Tab */}
        {activeTab === 'performance' && (
          <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
            <h3 className="text-xl font-semibold mb-6 flex items-center gap-3">
              <Zap className="w-6 h-6 text-quantum-yellow" />
              System Performance Metrics
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {performanceMetrics.map((metric, index) => (
                <motion.div
                  key={metric.label}
                  className="p-6 bg-quantum-dark/30 rounded-xl"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <div className="text-sm text-gray-400 mb-2">{metric.label}</div>
                  <div className="text-2xl font-bold text-white">{metric.value}</div>
                </motion.div>
              ))}
            </div>

            <div className="mt-8 p-6 bg-quantum-green/10 border border-quantum-green/20 rounded-xl">
              <div className="flex items-center gap-3 mb-2">
                <Activity className="w-5 h-5 text-quantum-green" />
                <span className="font-semibold text-quantum-green">Optimal Performance</span>
              </div>
              <p className="text-sm text-gray-400">
                System is operating within quantum consensus parameters. All metrics are within expected ranges for Phase 1 post-quantum deployment.
              </p>
            </div>
          </div>
        )}

        {/* Network Tab */}
        {activeTab === 'network' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
              <h3 className="text-xl font-semibold mb-6 flex items-center gap-3">
                <Globe className="w-6 h-6 text-quantum-cyan" />
                Network Connection
              </h3>

              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-quantum-dark/30 rounded-xl">
                  <span>Node Status</span>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-quantum-green rounded-full animate-pulse" />
                    <span className="text-quantum-green font-semibold">Connected</span>
                  </div>
                </div>
                <div className="flex items-center justify-between p-4 bg-quantum-dark/30 rounded-xl">
                  <span>Consensus Participation</span>
                  <span className="text-quantum-green font-semibold">Active</span>
                </div>
                <div className="flex items-center justify-between p-4 bg-quantum-dark/30 rounded-xl">
                  <span>Peer Count</span>
                  <span className="text-white font-semibold">127 peers</span>
                </div>
                <div className="flex items-center justify-between p-4 bg-quantum-dark/30 rounded-xl">
                  <span>Sync Status</span>
                  <span className="text-quantum-green font-semibold">Synchronized</span>
                </div>
              </div>
            </div>

            <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
              <h3 className="text-xl font-semibold mb-6">Node Endpoints</h3>

              <div className="space-y-3">
                <div className="p-3 bg-quantum-dark/30 rounded-lg font-mono text-sm">
                  wss://node1.qnk.network:8545
                </div>
                <div className="p-3 bg-quantum-dark/30 rounded-lg font-mono text-sm">
                  wss://node2.qnk.network:8545
                </div>
                <div className="p-3 bg-quantum-dark/30 rounded-lg font-mono text-sm">
                  wss://quantum.bitcoinoro.xyz:8545
                </div>
              </div>

              <button className="w-full mt-4 py-2 px-4 border border-quantum-purple/30 rounded-lg text-white hover:border-quantum-cyan/60 transition-colors">
                Add Custom Node
              </button>
            </div>
          </div>
        )}
      </motion.div>

      {/* Philosophical Quote */}
      <div className="text-center">
        <blockquote className="text-lg italic text-gray-400 max-w-2xl mx-auto">
          "Beauty is truth, truth beauty" - Where quantum consensus meets computational sublime
        </blockquote>
      </div>

      {/* Logout Button */}
      {onLogout && (
        <motion.div 
          className="mt-8 flex justify-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <motion.button
            onClick={onLogout}
            className="px-8 py-4 bg-gradient-to-r from-red-600/20 to-red-500/20 border border-red-500/30 rounded-xl text-red-400 font-semibold flex items-center gap-3 hover:border-red-400/60 hover:text-red-300 transition-all"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <LogOut className="w-5 h-5" />
            <span>Logout from Quantum Wallet</span>
          </motion.button>
        </motion.div>
      )}
    </div>
  );
}