import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Book, Code2, Zap, Wallet, Network, Lock, Menu, X, Rocket, ArrowRight, CheckCircle2, Cpu, Shield } from 'lucide-react';
import APIEndpoints from './components/APIEndpoints';
import WalletExamples from './components/WalletExamples';
import DEXExamples from './components/DEXExamples';
import WebSocketGuide from './components/WebSocketGuide';
import SmartContractGuide from './components/SmartContractGuide';
import OAuth2Integration from './components/OAuth2Integration';
import PrivacyAsAService from './components/PrivacyAsAService';

type Tab = 'overview' | 'endpoints' | 'wallet' | 'dex' | 'contracts' | 'websocket' | 'oauth2' | 'paas';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('overview');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const tabs = [
    { id: 'overview' as Tab, label: 'Overview', icon: Book },
    { id: 'endpoints' as Tab, label: 'API Endpoints', icon: Code2 },
    { id: 'wallet' as Tab, label: 'Wallet Integration', icon: Wallet },
    { id: 'dex' as Tab, label: 'DEX Building', icon: Network },
    { id: 'contracts' as Tab, label: 'Smart Contracts', icon: Cpu },
    { id: 'websocket' as Tab, label: 'WebSocket Streams', icon: Zap },
    { id: 'oauth2' as Tab, label: 'OAuth2', icon: Lock },
    { id: 'paas' as Tab, label: 'Privacy-as-a-Service', icon: Shield },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-quantum-dark via-quantum-indigo/20 to-quantum-dark">
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-quantum-dark/80 border-b border-quantum-purple/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-quantum-cyan via-quantum-purple to-quantum-pink rounded-lg flex items-center justify-center">
                <Lock className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink bg-clip-text text-transparent">
                  Quillon API
                </h1>
                <p className="text-xs text-gray-400">Quantum-Enhanced Blockchain</p>
              </div>
            </div>
            <nav className="hidden md:flex items-center gap-1">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={"relative px-4 py-2 rounded-lg text-sm font-medium transition-all " + (isActive ? 'text-white' : 'text-gray-400 hover:text-white')}
                  >
                    {isActive && (
                      <motion.div
                        layoutId="activeTab"
                        className="absolute inset-0 bg-gradient-to-r from-quantum-purple/20 to-quantum-cyan/20 rounded-lg border border-quantum-purple/30"
                        transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
                      />
                    )}
                    <span className="relative flex items-center gap-2">
                      <Icon className="w-4 h-4" />
                      {tab.label}
                    </span>
                  </button>
                );
              })}
            </nav>
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 rounded-lg text-gray-400 hover:text-white hover:bg-quantum-purple/10"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden border-t border-quantum-purple/30"
            >
              <div className="px-4 py-2 space-y-1">
                {tabs.map((tab) => {
                  const Icon = tab.icon;
                  const isActive = activeTab === tab.id;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => {
                        setActiveTab(tab.id);
                        setMobileMenuOpen(false);
                      }}
                      className={"w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all " + (isActive ? 'bg-gradient-to-r from-quantum-purple/20 to-quantum-cyan/20 text-white border border-quantum-purple/30' : 'text-gray-400 hover:text-white hover:bg-quantum-purple/10')}
                    >
                      <Icon className="w-5 h-5" />
                      {tab.label}
                    </button>
                  );
                })}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </header>
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <AnimatePresence mode="wait">
          {activeTab === 'overview' && (
            <motion.div key="overview" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }}>
              <OverviewSection setActiveTab={setActiveTab} />
            </motion.div>
          )}
          {activeTab === 'endpoints' && (
            <motion.div key="endpoints" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }}>
              <APIEndpoints />
            </motion.div>
          )}
          {activeTab === 'wallet' && (
            <motion.div key="wallet" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }}>
              <WalletExamples />
            </motion.div>
          )}
          {activeTab === 'dex' && (
            <motion.div key="dex" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }}>
              <DEXExamples />
            </motion.div>
          )}
          {activeTab === 'contracts' && (
            <motion.div key="contracts" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }}>
              <SmartContractGuide />
            </motion.div>
          )}
          {activeTab === 'websocket' && (
            <motion.div key="websocket" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }}>
              <WebSocketGuide />
            </motion.div>
          )}
          {activeTab === 'oauth2' && (
            <motion.div key="oauth2" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }}>
              <OAuth2Integration />
            </motion.div>
          )}
          {activeTab === 'paas' && (
            <motion.div key="paas" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }}>
              <PrivacyAsAService />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

function OverviewSection({ setActiveTab }: { setActiveTab: (tab: Tab) => void }) {
  return (
    <div className="space-y-8">
      <motion.div className="text-center space-y-4" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink bg-clip-text text-transparent">
          Quillon API
        </h1>
        <p className="text-xl text-gray-300 max-w-3xl mx-auto">
          Build wallets, DEXes, and smart contracts effortlessly with our quantum-ready blockchain API.
          Post-quantum cryptography, <span className="text-quantum-cyan font-bold">1M+ TPS</span>, and <span className="text-quantum-cyan font-bold">&lt;50ms finality</span>.
        </p>
        <div className="flex items-center justify-center gap-4 mt-4">
          <a
            href="https://code.quillon.xyz"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-gray-400 hover:text-quantum-cyan transition-colors flex items-center gap-2"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
            View Source Code
          </a>
          <span className="text-gray-600">|</span>
          <a
            href="https://quillon.xyz"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-gray-400 hover:text-quantum-cyan transition-colors"
          >
            Download Wallet
          </a>
        </div>
        <div className="flex items-center justify-center gap-4 pt-4">
          <button
            onClick={() => setActiveTab('endpoints')}
            className="px-6 py-3 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-xl font-bold text-white hover:shadow-lg hover:shadow-quantum-cyan/50 transition-all flex items-center gap-2"
          >
            <Rocket className="w-5 h-5" />
            Explore API
          </button>
          <button
            onClick={() => setActiveTab('contracts')}
            className="px-6 py-3 bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl font-bold text-white hover:border-quantum-cyan/50 transition-all flex items-center gap-2"
          >
            <Cpu className="w-5 h-5" />
            Smart Contracts
          </button>
        </div>
      </motion.div>
      <div className="grid md:grid-cols-4 gap-6 mt-12">
        {[
          { icon: Zap, title: 'Lightning Fast', description: '1M+ TPS with <50ms finality', color: 'from-quantum-cyan to-quantum-purple' },
          { icon: Lock, title: 'Post-Quantum', description: 'Dilithium5 + Kyber1024', color: 'from-quantum-purple to-quantum-pink' },
          { icon: Cpu, title: 'Rust Smart Contracts', description: 'WASM VM with type safety', color: 'from-quantum-pink to-quantum-cyan' },
          { icon: Code2, title: 'Simple API', description: 'REST & WebSocket', color: 'from-quantum-cyan to-quantum-purple' },
        ].map((feature, i) => (
          <motion.div
            key={feature.title}
            className="p-6 bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
          >
            <div className={"w-12 h-12 bg-gradient-to-br " + feature.color + " rounded-xl flex items-center justify-center mb-4"}>
              <feature.icon className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">{feature.title}</h3>
            <p className="text-gray-400">{feature.description}</p>
          </motion.div>
        ))}
      </div>
      <motion.div
        className="mt-12 p-8 bg-gradient-to-br from-quantum-indigo/30 to-quantum-purple/20 backdrop-blur-xl border border-quantum-cyan/30 rounded-2xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
          <Rocket className="w-7 h-7 text-quantum-cyan" />
          Quick Start: Create a Wallet in 30 Seconds
        </h2>
        <div className="space-y-4">
          {[
            { step: '1', text: 'Send a POST request to /api/v1/wallets/create', code: 'POST https://quillon.xyz/api/v1/wallets/create' },
            { step: '2', text: 'Receive wallet address and private key', code: '{ "address": "qnk...", "private_key": "..." }' },
            { step: '3', text: 'Check balance', code: 'GET https://quillon.xyz/api/v1/wallets/qnk.../balance' },
          ].map((item) => (
            <div key={item.step} className="flex items-start gap-4">
              <div className="w-8 h-8 bg-quantum-cyan/20 rounded-full flex items-center justify-center flex-shrink-0 border border-quantum-cyan/30">
                <span className="text-quantum-cyan font-bold">{item.step}</span>
              </div>
              <div className="flex-1">
                <p className="text-white font-medium mb-1">{item.text}</p>
                <code className="block px-3 py-2 bg-quantum-dark/50 rounded-lg text-sm text-quantum-cyan border border-quantum-purple/20">
                  {item.code}
                </code>
              </div>
            </div>
          ))}
        </div>
        <button
          onClick={() => setActiveTab('wallet')}
          className="mt-6 px-6 py-3 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-xl font-bold text-white hover:shadow-lg hover:shadow-quantum-cyan/50 transition-all flex items-center gap-2"
        >
          See Full Examples
          <ArrowRight className="w-5 h-5" />
        </button>
      </motion.div>
      <motion.div
        className="mt-12 p-8 bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <h2 className="text-2xl font-bold text-white mb-4">Why Build on Quillon?</h2>
        <p className="text-gray-400 mb-6">
          Quillon combines cutting-edge quantum-resistant cryptography with developer-friendly APIs.
          Built on a high-performance Rust architecture with <a href="https://code.quillon.xyz" target="_blank" rel="noopener noreferrer" className="text-quantum-cyan hover:underline">DAG-Knight consensus</a>,
          it delivers enterprise-grade performance without sacrificing security or ease of use.
        </p>
        <div className="grid md:grid-cols-2 gap-4">
          {[
            { text: 'RESTful API - No blockchain complexity', link: 'https://code.quillon.xyz' },
            { text: 'WebSocket Streams - Real-time updates', link: 'https://code.quillon.xyz' },
            { text: 'Rust Smart Contracts - Type-safe WASM VM', link: 'https://code.quillon.xyz' },
            { text: 'Instant Finality - <50ms', link: 'https://code.quillon.xyz' },
            { text: 'No Gas Fees (Dev) - Unlimited faucet', link: 'https://code.quillon.xyz' },
            { text: 'Post-Quantum Secure - Future-proof', link: 'https://code.quillon.xyz' },
          ].map((benefit) => (
            <a
              key={benefit.text}
              href={benefit.link}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-3 p-3 rounded-lg hover:bg-quantum-purple/10 transition-colors"
            >
              <CheckCircle2 className="w-5 h-5 text-quantum-green flex-shrink-0" />
              <span className="text-gray-300 hover:text-white transition-colors">{benefit.text}</span>
            </a>
          ))}
        </div>
      </motion.div>
    </div>
  );
}

export default App;
