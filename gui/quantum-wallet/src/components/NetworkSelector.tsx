import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, Clock } from 'lucide-react';

type NetworkType = 'testnet' | 'mainnet';

export default function NetworkSelector() {
  const [selectedNetwork, setSelectedNetwork] = useState<NetworkType>(
    (localStorage.getItem('selectedNetwork') as NetworkType) || 'testnet'
  );
  const [showNetworkDropdown, setShowNetworkDropdown] = useState(false);
  const [mainnetCountdown, setMainnetCountdown] = useState('');

  // Mainnet launch date: December 15, 2025 00:00 UTC
  const MAINNET_LAUNCH = new Date('2025-12-15T00:00:00Z');

  // Update countdown every second
  useEffect(() => {
    const updateCountdown = () => {
      const now = new Date();
      const diff = MAINNET_LAUNCH.getTime() - now.getTime();
      if (diff <= 0) {
        setMainnetCountdown('LIVE NOW!');
        return;
      }
      const days = Math.floor(diff / (1000 * 60 * 60 * 24));
      const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
      const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
      const seconds = Math.floor((diff % (1000 * 60)) / 1000);
      setMainnetCountdown(`${days}d ${hours}h ${minutes}m ${seconds}s`);
    };
    updateCountdown();
    const interval = setInterval(updateCountdown, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleNetworkSwitch = (network: NetworkType) => {
    setSelectedNetwork(network);
    localStorage.setItem('selectedNetwork', network);
    setShowNetworkDropdown(false);
    const newPort = network === 'testnet' ? '8080' : '8081';
    const newBaseURL = `http://localhost:${newPort}`;
    localStorage.setItem('apiBaseURL', newBaseURL);
    window.location.reload();
  };

  return (
    <div className="relative network-selector-container" style={{ border: '2px solid red' }}>
      <motion.button
        onClick={() => setShowNetworkDropdown(!showNetworkDropdown)}
        whileHover={{ scale: 1.05 }}
        className={`flex items-center gap-2 rounded-lg px-3 py-1.5 border transition-all ${
          selectedNetwork === 'testnet'
            ? 'bg-gradient-to-r from-purple-600/20 to-pink-600/20 border-purple-500/40'
            : 'bg-gradient-to-r from-green-600/20 to-emerald-600/20 border-green-500/40'
        }`}
      >
        <span className={`text-sm font-bold ${
          selectedNetwork === 'testnet' ? 'text-purple-300' : 'text-green-300'
        }`}>
          {selectedNetwork.toUpperCase()}
        </span>
        <ChevronDown className={`w-4 h-4 transition-transform ${
          showNetworkDropdown ? 'rotate-180' : ''
        } ${selectedNetwork === 'testnet' ? 'text-purple-300' : 'text-green-300'}`} />
      </motion.button>

      {/* Network Dropdown */}
      <AnimatePresence>
        {showNetworkDropdown && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute top-full right-0 mt-2 w-72 bg-quantum-indigo/95 backdrop-blur-xl border border-quantum-purple/30 rounded-xl shadow-2xl overflow-hidden z-50"
          >
            {/* Testnet Option */}
            <motion.div
              whileHover={{ backgroundColor: 'rgba(147, 51, 234, 0.1)' }}
              onClick={() => handleNetworkSwitch('testnet')}
              className="p-4 cursor-pointer border-b border-quantum-purple/20"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-300 font-bold">TESTNET</span>
                {selectedNetwork === 'testnet' && (
                  <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" />
                )}
              </div>
              <p className="text-gray-400 text-xs">
                Development & testing network (Port 8080)
              </p>
            </motion.div>

            {/* Mainnet Option */}
            <motion.div
              whileHover={{ backgroundColor: 'rgba(34, 197, 94, 0.1)' }}
              onClick={() => handleNetworkSwitch('mainnet')}
              className="p-4 cursor-pointer"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-green-300 font-bold">MAINNET</span>
                {selectedNetwork === 'mainnet' && (
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                )}
              </div>
              <p className="text-gray-400 text-xs mb-2">
                Production network (Port 8081)
              </p>
              {/* Countdown Timer */}
              <div className="flex items-center gap-2 bg-quantum-yellow/10 border border-quantum-yellow/30 rounded-lg px-2 py-1">
                <Clock className="w-3 h-3 text-quantum-yellow" />
                <span className="text-quantum-yellow text-xs font-mono">
                  {mainnetCountdown}
                </span>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
