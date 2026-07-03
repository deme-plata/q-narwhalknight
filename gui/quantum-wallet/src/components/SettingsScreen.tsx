import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, Palette, Activity, Globe, Lock, Eye, Zap, LogOut, Clock, Info, Key, Download, EyeOff, Cloud, Code, Trash2, RefreshCw, AlertCircle, Server, ArrowDownToLine } from 'lucide-react';
import { qnkAPI } from '../services/api';
import { walletSession } from '../services/walletAuth';
import QuantumChamberCanvas from './QuantumChamberCanvas';

interface SettingsScreenProps {
  onLogout?: () => void;
}

export default function SettingsScreen({ onLogout }: SettingsScreenProps) {
  const [activeTab, setActiveTab] = useState<string>('crypto');
  const [cryptoSuite, setCryptoSuite] = useState('Q1');
  const [visualEffects, setVisualEffects] = useState({
    entanglementMoire: true,
    photonWaterfall: true,
    rainbowBoxes: true,
    fractalOverlay: true,
  });
  const [showChamberInfo, setShowChamberInfo] = useState(false);

  // v2.4.0: Performance mode - disables heavy effects for better frame rates (DEFAULT: ON)
  const [performanceMode, setPerformanceMode] = useState(() => {
    const saved = localStorage.getItem('performanceMode');
    // Default to true if not set
    return saved === null ? true : saved === 'true';
  });

  // Apply performance mode to document
  useEffect(() => {
    if (performanceMode) {
      document.documentElement.classList.add('performance-mode');
    } else {
      document.documentElement.classList.remove('performance-mode');
    }
    localStorage.setItem('performanceMode', performanceMode.toString());
    // Notify App.tsx about the change
    window.dispatchEvent(new CustomEvent('performance-mode-changed'));
  }, [performanceMode]);

  // Blockchain Benchmark State
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);
  const [benchmarkResult, setBenchmarkResult] = useState<any>(null);
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null);
  const [benchmarkCooldown, setBenchmarkCooldown] = useState<number>(0);

  // Load session timeout setting from localStorage
  const [sessionTimeout, setSessionTimeout] = useState(() => {
    const saved = localStorage.getItem('walletSessionTimeout');
    return saved || 'never'; // Default: never expire (user convenience)
  });

  // Save session timeout setting to localStorage when changed
  useEffect(() => {
    localStorage.setItem('walletSessionTimeout', sessionTimeout);
  }, [sessionTimeout]);

  // Password modal states
  const [showPasswordModal, setShowPasswordModal] = useState(false);
  const [passwordModalAction, setPasswordModalAction] = useState<'private-key' | 'mnemonic' | 'download'>('private-key');
  const [passwordInput, setPasswordInput] = useState('');
  const [showPrivateKey, setShowPrivateKey] = useState(false);
  const [showMnemonic, setShowMnemonic] = useState(false);
  const [privateKeyValue, setPrivateKeyValue] = useState('');
  const [mnemonicValue, setMnemonicValue] = useState('');
  const [passwordError, setPasswordError] = useState('');

  const tabs = [
    { id: 'crypto', label: 'Crypto Agility', icon: Shield },
    { id: 'security', label: 'Security', icon: Lock },
    { id: 'node', label: 'Node Admin', icon: Server },
    { id: 'paas', label: 'Privacy-as-a-Service', icon: Cloud },
    { id: 'oauth2', label: 'OAuth2 Settings', icon: Code },
    { id: 'visuals', label: 'Quantum Visuals', icon: Palette },
    { id: 'performance', label: 'Performance', icon: Activity },
    { id: 'network', label: 'Network', icon: Globe },
    { id: 'about', label: 'About', icon: Info },
  ];

  // Handle password verification and action
  const handlePasswordSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setPasswordError('');

    try {
      const walletAddress = localStorage.getItem('walletAddress');
      if (!walletAddress) {
        setPasswordError('No wallet found. Please create or import a wallet first.');
        return;
      }

      // ALWAYS require password verification for sensitive operations
      const { loadWallet, recoverMnemonic } = await import('../services/walletAuth');

      // Get encrypted mnemonic to verify password
      const encryptedMnemonic = localStorage.getItem('walletEncryptedMnemonic');
      if (!encryptedMnemonic) {
        setPasswordError('No encrypted wallet data found. This may happen if you logged in with an older wallet version. Please log out and log in again to re-encrypt your wallet.');
        return;
      }

      // CRITICAL SECURITY: ALWAYS verify password by attempting to decrypt wallet
      try {
        const walletData = await loadWallet(passwordInput);

        // Convert private key bytes to hex for display
        const privateKeyHex = Array.from(walletData.privateKey)
          .map(b => b.toString(16).padStart(2, '0'))
          .join('');

        if (passwordModalAction === 'private-key') {
          setPrivateKeyValue(privateKeyHex);
          setShowPrivateKey(true);
        } else if (passwordModalAction === 'mnemonic') {
          // Recover mnemonic from encrypted storage
          const mnemonic = await recoverMnemonic(passwordInput);
          setMnemonicValue(mnemonic);
          setShowMnemonic(true);
        } else if (passwordModalAction === 'download') {
          // Recover mnemonic for download
          const mnemonic = await recoverMnemonic(passwordInput);

          // Download wallet key file
          const keyFileContent = JSON.stringify({
            version: '1.0',
            address: walletData.address,
            private_key: privateKeyHex,
            mnemonic: mnemonic,
            created_at: new Date().toISOString(),
            quantum_suite: 'Q1-Dilithium5-Kyber1024',
          }, null, 2);

          const blob = new Blob([keyFileContent], { type: 'application/json' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `quantum-wallet-${walletData.address.slice(0, 8)}.json`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
        }

        setShowPasswordModal(false);
        setPasswordInput('');
      } catch (error) {
        console.error('Password verification error:', error);
        setPasswordError('Incorrect password');
      }
    } catch (error) {
      console.error('Password verification error:', error);
      setPasswordError('Error verifying password');
    }
  };

  const openPasswordModal = (action: 'private-key' | 'mnemonic' | 'download') => {
    const isMetaMask = !!localStorage.getItem('metamaskLinked');

    if (isMetaMask) {
      // MetaMask users have no user-set password — use in-memory session data
      const session = walletSession.getSession();
      if (!session) {
        setPasswordError('Session expired. Please log out and log back in with MetaMask.');
        return;
      }

      const privateKeyHex = Array.from(session.privateKey)
        .map(b => b.toString(16).padStart(2, '0'))
        .join('');

      if (action === 'private-key') {
        setPrivateKeyValue(privateKeyHex);
        setShowPrivateKey(true);
      } else if (action === 'mnemonic') {
        if (session.mnemonic) {
          setMnemonicValue(session.mnemonic);
          setShowMnemonic(true);
        } else {
          setPasswordError('Mnemonic not available. Please log out and re-login with MetaMask.');
        }
      } else if (action === 'download') {
        const mnemonic = session.mnemonic || '';
        const keyFileContent = JSON.stringify({
          version: '1.0',
          address: session.address,
          private_key: privateKeyHex,
          mnemonic: mnemonic,
          created_at: new Date().toISOString(),
          quantum_suite: 'Q1-Dilithium5-Kyber1024',
        }, null, 2);

        const blob = new Blob([keyFileContent], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `quantum-wallet-${session.address.slice(0, 8)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
      return; // Skip password modal
    }

    // Non-MetaMask: normal password flow
    setPasswordModalAction(action);
    setShowPasswordModal(true);
    setPasswordError('');
    setPasswordInput('');
  };

  // Handle blockchain benchmark
  const handleBenchmark = async () => {
    setBenchmarkRunning(true);
    setBenchmarkError(null);
    setBenchmarkResult(null);

    try {
      const apiUrl = localStorage.getItem('apiUrl') || 'http://localhost:8080';
      const response = await fetch(`${apiUrl}/api/v1/benchmark`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        if (response.status === 429) {
          // Rate limited - extract cooldown time
          const cooldownMinutes = data.cooldown_minutes || 1440; // Default to 24 hours
          setBenchmarkCooldown(cooldownMinutes);
          setBenchmarkError(data.error || 'Benchmark rate limit reached. Please try again later.');
        } else {
          setBenchmarkError(data.error || 'Failed to run benchmark');
        }
      } else {
        setBenchmarkResult(data.result);
      }
    } catch (error) {
      console.error('Benchmark error:', error);
      setBenchmarkError('Failed to connect to API server. Please ensure the node is running.');
    } finally {
      setBenchmarkRunning(false);
    }
  };

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
    { label: 'Throughput', value: '1.2M+ TPS' },
    { label: 'Latency', value: 'Sub-50ms finality' },
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
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
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

        {/* Security Tab - Session Timeout Settings */}
        {activeTab === 'security' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
              <h3 className="text-xl font-semibold mb-6 flex items-center gap-3">
                <Clock className="w-6 h-6 text-quantum-cyan" />
                Session Timeout
              </h3>

              <p className="text-gray-400 mb-6">
                Choose how often you want to enter your password when accessing wallet features. Longer timeouts are more convenient but less secure.
              </p>

              <div className="space-y-3">
                {[
                  { value: '5', label: '5 minutes', description: 'Maximum security', color: 'quantum-green' },
                  { value: '15', label: '15 minutes', description: 'Recommended balance', color: 'quantum-cyan' },
                  { value: '30', label: '30 minutes', description: 'Moderate convenience', color: 'quantum-yellow' },
                  { value: '60', label: '1 hour', description: 'High convenience', color: 'quantum-yellow' },
                  { value: '240', label: '4 hours', description: 'Maximum convenience', color: 'quantum-pink' },
                  { value: 'never', label: 'Never expire', description: 'No auto-logout (not recommended)', color: 'red-400' },
                ].map((option) => (
                  <motion.label
                    key={option.value}
                    className={`block p-4 rounded-xl border-2 cursor-pointer transition-all ${
                      sessionTimeout === option.value
                        ? 'border-quantum-cyan bg-quantum-cyan/10'
                        : 'border-quantum-purple/20 hover:border-quantum-purple/40'
                    }`}
                    whileHover={{ scale: 1.01 }}
                  >
                    <input
                      type="radio"
                      name="sessionTimeout"
                      value={option.value}
                      checked={sessionTimeout === option.value}
                      onChange={(e) => setSessionTimeout(e.target.value)}
                      className="sr-only"
                    />
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-semibold text-white">{option.label}</div>
                        <div className={`text-sm text-${option.color}`}>{option.description}</div>
                      </div>
                      {sessionTimeout === option.value && (
                        <div className="w-3 h-3 bg-quantum-cyan rounded-full" />
                      )}
                    </div>
                  </motion.label>
                ))}
              </div>

              {sessionTimeout === 'never' && (
                <div className="mt-4 p-4 bg-red-500/10 border border-red-500/30 rounded-xl">
                  <div className="flex items-center gap-2 text-red-400 font-semibold mb-2">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                      <line x1="12" y1="9" x2="12" y2="13"></line>
                      <line x1="12" y1="17" x2="12.01" y2="17"></line>
                    </svg>
                    Security Warning
                  </div>
                  <p className="text-sm text-red-300">
                    With "Never expire" enabled, your wallet will remain unlocked indefinitely. Anyone with access to your device can access your funds. Use this option only on trusted, secure devices.
                  </p>
                </div>
              )}
            </div>

            <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
              <h3 className="text-xl font-semibold mb-6 flex items-center gap-3">
                <Shield className="w-6 h-6 text-quantum-green" />
                Wallet Security
              </h3>

              <div className="space-y-4">
                <div className="p-4 bg-quantum-dark/30 rounded-xl">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">Private Key Encryption</span>
                    <span className="text-quantum-green font-semibold">Active</span>
                  </div>
                  <p className="text-sm text-gray-400">
                    Your private keys are encrypted with AES-256-GCM and stored securely on your device.
                  </p>
                </div>

                <div className="p-4 bg-quantum-dark/30 rounded-xl">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">Password Protection</span>
                    <span className="text-quantum-green font-semibold">Enabled</span>
                  </div>
                  <p className="text-sm text-gray-400">
                    Your wallet requires password authentication for all sensitive operations.
                  </p>
                </div>

                <div className="p-4 bg-quantum-dark/30 rounded-xl">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">Current Session Timeout</span>
                    <span className="text-quantum-cyan font-semibold">
                      {sessionTimeout === 'never' ? 'Never' : `${sessionTimeout} min`}
                    </span>
                  </div>
                  <p className="text-sm text-gray-400">
                    {sessionTimeout === 'never'
                      ? 'Your session will not expire automatically.'
                      : `Your session will expire after ${sessionTimeout} minutes of activity.`
                    }
                  </p>
                </div>

                <div className="p-4 bg-quantum-green/10 border border-quantum-green/30 rounded-xl">
                  <div className="flex items-center gap-2 mb-2">
                    <Lock className="w-4 h-4 text-quantum-green" />
                    <span className="font-semibold text-quantum-green">Best Practices</span>
                  </div>
                  <ul className="text-sm text-gray-400 space-y-1">
                    <li>• Use a strong, unique password</li>
                    <li>• Enable shorter timeout on shared devices</li>
                    <li>• Keep your mnemonic phrase secure</li>
                    <li>• Log out when not in use</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Node Admin Tab */}
        {activeTab === 'node' && (
          <NodeAdminTab />
        )}

        {/* Privacy-as-a-Service Tab */}
        {activeTab === 'paas' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
              <h3 className="text-xl font-semibold mb-6 flex items-center gap-3">
                <Cloud className="w-6 h-6 text-quantum-cyan" />
                PaaS API Configuration
              </h3>

              <p className="text-gray-400 mb-6">
                Configure your Privacy-as-a-Service API keys for Bitcoin, Ethereum, and Solana privacy features.
              </p>

              <div className="space-y-4">
                <div className="p-4 bg-quantum-dark/30 rounded-xl">
                  <label className="block text-sm font-medium text-gray-300 mb-2">API Key</label>
                  <div className="flex gap-2">
                    <input
                      type="password"
                      placeholder="paas_1a2b3c4d5e6f7g8h9i0j_..."
                      className="flex-1 px-4 py-2 bg-quantum-dark/50 border border-quantum-purple/30 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-quantum-cyan/60"
                    />
                    <motion.button
                      className="px-4 py-2 bg-gradient-to-r from-quantum-purple to-quantum-cyan rounded-lg text-white font-medium hover:shadow-lg hover:shadow-quantum-purple/50 transition-all whitespace-nowrap"
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={async () => {
                        try {
                          // Get wallet address from local storage
                          const walletAddress = localStorage.getItem('currentWallet');

                          console.log('[PaaS] Generating API key...', {
                            hasWallet: !!walletAddress,
                            wallet: walletAddress
                          });

                          if (!walletAddress) {
                            alert('Please create or select a wallet first before generating an API key.');
                            return;
                          }

                          // Call real API to generate PaaS API key
                          console.log('[PaaS] Calling API endpoint...');
                          const response = await fetch('http://localhost:8080/api/v1/privacy/paas/api-keys/generate', {
                            method: 'POST',
                            headers: {
                              'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                              wallet_address: walletAddress,
                              tier: 'free',
                              expires_days: 90
                            })
                          });

                          console.log('[PaaS] API response status:', response.status, response.statusText);

                          if (!response.ok) {
                            const errorText = await response.text();
                            console.error('[PaaS] API error response:', errorText);
                            throw new Error(`API request failed: ${response.status} ${response.statusText}`);
                          }

                          const data = await response.json();
                          console.log('[PaaS] API response data:', data);

                          if (data.success && data.data && data.data.api_key) {
                            const input = document.querySelector('input[type="password"][placeholder*="paas_"]') as HTMLInputElement;
                            if (input) {
                              input.type = 'text';
                              input.value = data.data.api_key;
                              console.log('[PaaS] API key generated successfully:', data.data.key_id);

                              // Show success message
                              const successDiv = document.createElement('div');
                              successDiv.className = 'text-green-400 text-sm mt-2';
                              successDiv.textContent = '✓ API key generated successfully! (Visible for 5 seconds)';
                              input.parentElement?.appendChild(successDiv);

                              // Show key for 5 seconds then hide it
                              setTimeout(() => {
                                input.type = 'password';
                                successDiv.remove();
                              }, 5000);
                            } else {
                              console.error('[PaaS] Could not find password input element');
                              alert('API key generated but could not display it. Check console.');
                            }
                          } else {
                            console.error('[PaaS] API returned unsuccessful response:', data);
                            alert('Failed to generate API key: ' + (data.error || 'Unknown error'));
                          }
                        } catch (error: any) {
                          console.error('[PaaS] Error generating PaaS API key:', error);
                          console.error('[PaaS] Error details:', {
                            message: error?.message,
                            stack: error?.stack,
                            type: error?.constructor?.name
                          });
                          alert(`Error generating API key: ${error?.message || 'Please try again.'}\n\nCheck browser console (F12) for details.`);
                        }
                      }}
                    >
                      Generate Key
                    </motion.button>
                  </div>
                  <p className="text-xs text-gray-500 mt-2">
                    Generate a local API key or get one at <a href="https://quillon.xyz/console" target="_blank" rel="noopener noreferrer" className="text-quantum-cyan hover:underline">quillon.xyz/console</a>
                  </p>
                </div>

                <div className="p-4 bg-quantum-dark/30 rounded-xl">
                  <label className="block text-sm font-medium text-gray-300 mb-2">Subscription Tier</label>
                  <select className="w-full px-4 py-2 bg-quantum-dark/50 border border-quantum-purple/30 rounded-lg text-white focus:outline-none focus:border-quantum-cyan/60">
                    <option value="free">Free (10,000 calls/day)</option>
                    <option value="professional">Professional ($499/mo)</option>
                    <option value="enterprise">Enterprise ($1,999/mo)</option>
                  </select>
                </div>

                <div className="p-4 bg-quantum-dark/30 rounded-xl">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">Default Privacy Level</span>
                  </div>
                  <div className="space-y-2">
                    {[
                      { value: 'standard', label: 'Standard (ε ≈ 2.3)', description: 'Fast mixing, moderate privacy' },
                      { value: 'maximum', label: 'Maximum (ε < 0.7)', description: 'Slower, maximum privacy' },
                    ].map((option) => (
                      <label
                        key={option.value}
                        className="block p-3 rounded-lg border border-quantum-purple/20 hover:border-quantum-purple/40 cursor-pointer"
                      >
                        <input
                          type="radio"
                          name="privacyLevel"
                          value={option.value}
                          defaultChecked={option.value === 'standard'}
                          className="mr-2"
                        />
                        <span className="font-medium text-white">{option.label}</span>
                        <p className="text-xs text-gray-400 ml-6">{option.description}</p>
                      </label>
                    ))}
                  </div>
                </div>
              </div>

              <motion.button
                className="w-full mt-6 py-3 px-4 bg-gradient-to-r from-quantum-cyan/20 to-quantum-purple/20 border border-quantum-cyan/30 rounded-xl text-white font-semibold hover:border-quantum-cyan/60 transition-all"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Save API Configuration
              </motion.button>
            </div>

            <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
              <h3 className="text-xl font-semibold mb-6 flex items-center gap-3">
                <Shield className="w-6 h-6 text-quantum-green" />
                Privacy Features
              </h3>

              <div className="space-y-4">
                <div className="p-4 bg-quantum-dark/30 rounded-xl">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">Tor Relay</span>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-quantum-green rounded-full" />
                      <span className="text-quantum-green text-sm">Active</span>
                    </div>
                  </div>
                  <p className="text-sm text-gray-400">
                    Route transactions through Tor network to hide your IP address
                  </p>
                </div>

                <div className="p-4 bg-quantum-dark/30 rounded-xl">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">Transaction Mixing</span>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-quantum-green rounded-full" />
                      <span className="text-quantum-green text-sm">Enabled</span>
                    </div>
                  </div>
                  <p className="text-sm text-gray-400">
                    Mix your transactions with others for enhanced privacy
                  </p>
                </div>

                <div className="p-4 bg-quantum-dark/30 rounded-xl">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">MEV Protection</span>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-quantum-green rounded-full" />
                      <span className="text-quantum-green text-sm">Enabled</span>
                    </div>
                  </div>
                  <p className="text-sm text-gray-400">
                    Protect Ethereum transactions from front-running
                  </p>
                </div>

                <div className="p-4 bg-quantum-dark/30 rounded-xl">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">Stealth Addresses</span>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-quantum-green rounded-full" />
                      <span className="text-quantum-green text-sm">Enabled</span>
                    </div>
                  </div>
                  <p className="text-sm text-gray-400">
                    Generate one-time addresses for unlinkable transactions
                  </p>
                </div>

                <div className="p-4 bg-quantum-cyan/10 border border-quantum-cyan/30 rounded-xl">
                  <div className="flex items-center gap-2 mb-2">
                    <Shield className="w-4 h-4 text-quantum-cyan" />
                    <span className="font-semibold text-quantum-cyan">Security Model</span>
                  </div>
                  <p className="text-sm text-gray-400">
                    Your private keys NEVER leave your device. You sign transactions client-side, then submit signed transactions to the privacy service.
                  </p>
                </div>

                <div className="p-4 bg-quantum-purple/10 border border-quantum-purple/30 rounded-xl">
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-medium text-white">API Usage</span>
                    <span className="text-quantum-cyan font-semibold">4,231 / 10,000</span>
                  </div>
                  <div className="w-full bg-quantum-dark/50 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-quantum-cyan to-quantum-purple h-2 rounded-full"
                      style={{ width: '42%' }}
                    />
                  </div>
                  <p className="text-xs text-gray-400 mt-2">Daily quota resets in 6h 24m</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* OAuth2 Settings Tab */}
        {activeTab === 'oauth2' && (
          <OAuth2SettingsTab />
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

            <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8 border border-quantum-purple/30">
              <h3 className="text-xl font-semibold mb-6 flex items-center gap-3">
                <Palette className="w-6 h-6 text-quantum-yellow" />
                <span className="bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink bg-clip-text text-transparent">
                  Quantum Visualization Chamber
                </span>
                <button
                  onClick={() => setShowChamberInfo(p => !p)}
                  className="ml-auto p-1.5 rounded-lg hover:bg-white/10 transition-colors"
                  title="What am I looking at?"
                >
                  <Info className="w-5 h-5 text-quantum-cyan/70 hover:text-quantum-cyan" />
                </button>
              </h3>

              <AnimatePresence>
                {showChamberInfo && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.25 }}
                    className="overflow-hidden mb-4"
                  >
                    <div className="p-5 rounded-xl bg-quantum-dark/60 border border-quantum-cyan/20 text-sm leading-relaxed space-y-4">
                      <p className="text-white/90 font-medium text-base">What you're seeing is a real-time simulation of quantum phenomena that power this blockchain.</p>

                      <div className="space-y-3 text-gray-300">
                        <div>
                          <span className="text-quantum-cyan font-semibold">Interference Web</span>
                          <span className="text-gray-500 ml-1">(Fractal Overlay)</span>
                          <p className="mt-0.5">Curved lines weaving across the screen — these represent <em>quantum interference patterns</em>. In physics, when two quantum waves overlap they can reinforce or cancel each other out, producing these beautiful fringes. In our network, this visualizes how multiple transaction paths through the DAG interfere constructively to reach consensus faster.</p>
                        </div>

                        <div>
                          <span className="text-green-400 font-semibold">Photon Rain</span>
                          <span className="text-gray-500 ml-1">(Photon Waterfall)</span>
                          <p className="mt-0.5">Colored streaks falling like rain with glowing tips — each one is a <em>photon</em>, the smallest packet of light energy. Think of each streak as a single transaction being broadcast across the network. The trailing glow shows its propagation history, and the sparkles represent confirmations arriving at different nodes.</p>
                        </div>

                        <div>
                          <span className="text-purple-400 font-semibold">Entangled Pairs</span>
                          <span className="text-gray-500 ml-1">(Entanglement Moire)</span>
                          <p className="mt-0.5">Glowing orbs connected by pulsing lines — this demonstrates <em>quantum entanglement</em>, where two particles share a state instantly no matter how far apart they are. Einstein called it "spooky action at a distance." Here it represents how validator nodes stay perfectly in sync: when one confirms a block, its entangled partner knows immediately. The expanding ripples show this consensus propagating outward.</p>
                        </div>

                        <div>
                          <span className="text-amber-400 font-semibold">Rainbow Gemstones</span>
                          <span className="text-gray-500 ml-1">(Rainbow Boxes)</span>
                          <p className="mt-0.5">Colorful shapes that morph between hexagons and circles — these are <em>quantum state superpositions</em>. In quantum mechanics, a particle can be in multiple states simultaneously until it is measured. The morphing shape shows this superposition collapsing (circle = measured, hexagon = superposed). Each color represents a different qubit state in our post-quantum cryptographic signatures.</p>
                        </div>

                        <div>
                          <span className="text-white font-semibold">Central Orb</span>
                          <p className="mt-0.5">The pulsing white-purple orb at the center is the <em>network heartbeat</em> — a visual representation of the DAG-Knight consensus engine producing blocks in real time.</p>
                        </div>
                      </div>

                      <p className="text-gray-500 text-xs border-t border-white/10 pt-3">Toggle each effect on/off with the switches on the left panel. These are cosmetic visualizations inspired by real quantum physics — they don't affect wallet performance or security.</p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="relative h-80 bg-gradient-to-br from-quantum-dark via-quantum-indigo/30 to-quantum-dark rounded-xl overflow-hidden border border-quantum-cyan/20 shadow-2xl shadow-quantum-purple/20">
                {/* Animated border glow */}
                <motion.div
                  className="absolute inset-0 rounded-xl pointer-events-none"
                  animate={{
                    boxShadow: [
                      'inset 0 0 30px rgba(0, 212, 255, 0.1), 0 0 20px rgba(107, 70, 193, 0.2)',
                      'inset 0 0 50px rgba(107, 70, 193, 0.2), 0 0 40px rgba(0, 212, 255, 0.3)',
                      'inset 0 0 30px rgba(0, 212, 255, 0.1), 0 0 20px rgba(107, 70, 193, 0.2)',
                    ]
                  }}
                  transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                />

                {/* Canvas-based live preview of effects */}
                <QuantumChamberCanvas
                  fractalOverlay={visualEffects.fractalOverlay}
                  photonWaterfall={visualEffects.photonWaterfall}
                  entanglementMoire={visualEffects.entanglementMoire}
                  rainbowBoxes={visualEffects.rainbowBoxes}
                />

                <div className="absolute bottom-4 left-4 text-sm text-quantum-cyan backdrop-blur-sm bg-black/50 px-4 py-2 rounded-lg border border-quantum-cyan/30">
                  <span className="animate-pulse mr-2">●</span>
                  Live Preview • Quantum Visualization Engine
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Performance Tab */}
        {activeTab === 'performance' && (
          <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
            {/* v2.4.0: Performance Mode Toggle */}
            <div className="mb-8 p-6 bg-quantum-dark/30 rounded-xl border border-amber-500/30">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-amber-500/20 rounded-xl flex items-center justify-center">
                    <Zap className="w-6 h-6 text-amber-400" />
                  </div>
                  <div>
                    <h4 className="text-lg font-semibold text-white">Performance Mode</h4>
                    <p className="text-sm text-gray-400">
                      Disable animations for better frame rates on slower devices
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setPerformanceMode(!performanceMode)}
                  className={`relative w-14 h-7 rounded-full transition-colors duration-200 ${
                    performanceMode ? 'bg-amber-500' : 'bg-gray-600'
                  }`}
                >
                  <div
                    className={`absolute top-1 w-5 h-5 bg-white rounded-full transition-transform duration-200 ${
                      performanceMode ? 'translate-x-8' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
              {performanceMode && (
                <div className="mt-4 p-3 bg-amber-500/10 rounded-lg border border-amber-500/20">
                  <p className="text-sm text-amber-300">
                    Performance mode is enabled. Animations and visual effects are disabled for smoother operation.
                  </p>
                </div>
              )}
            </div>

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

            {/* Blockchain Benchmark Section */}
            <div className="mt-8 p-6 bg-quantum-indigo/30 rounded-xl border border-quantum-purple/30">
              <h4 className="text-lg font-semibold mb-4 flex items-center gap-3">
                <Zap className="w-5 h-5 text-quantum-yellow" />
                Blockchain Benchmark
              </h4>
              <p className="text-sm text-gray-400 mb-4">
                Test the network's current performance. Limited to once per 24 hours per IP address.
              </p>

              {benchmarkCooldown > 0 ? (
                <div className="p-4 bg-quantum-yellow/10 border border-quantum-yellow/20 rounded-xl">
                  <p className="text-sm text-quantum-yellow">
                    Benchmark available in: {Math.floor(benchmarkCooldown / 60)} hours {benchmarkCooldown % 60} minutes
                  </p>
                </div>
              ) : benchmarkRunning ? (
                <div className="flex items-center gap-3 p-4 bg-quantum-cyan/10 border border-quantum-cyan/20 rounded-xl">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-quantum-cyan"></div>
                  <span className="text-quantum-cyan">Running benchmark...</span>
                </div>
              ) : benchmarkResult ? (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-3 bg-quantum-dark/30 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">TPS</div>
                      <div className="text-lg font-bold text-white">{benchmarkResult.tps?.toLocaleString() || 'N/A'}</div>
                    </div>
                    <div className="p-3 bg-quantum-dark/30 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Latency</div>
                      <div className="text-lg font-bold text-white">{benchmarkResult.latency || 'N/A'}ms</div>
                    </div>
                    <div className="p-3 bg-quantum-dark/30 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Block Time</div>
                      <div className="text-lg font-bold text-white">{benchmarkResult.blockTime || 'N/A'}ms</div>
                    </div>
                    <div className="p-3 bg-quantum-dark/30 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Consensus</div>
                      <div className="text-lg font-bold text-white">{benchmarkResult.consensusTime || 'N/A'}ms</div>
                    </div>
                  </div>
                  <motion.button
                    onClick={() => setBenchmarkResult(null)}
                    className="w-full py-2 px-4 border border-quantum-purple/30 rounded-lg text-white hover:border-quantum-cyan/60 transition-colors"
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                  >
                    Clear Results
                  </motion.button>
                </div>
              ) : (
                <motion.button
                  onClick={handleBenchmark}
                  className="w-full py-3 px-4 bg-gradient-to-r from-quantum-cyan/20 to-quantum-purple/20 border border-quantum-cyan/30 rounded-xl text-white font-semibold hover:border-quantum-cyan/60 hover:shadow-lg hover:shadow-quantum-cyan/20 transition-all flex items-center justify-center gap-3"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <Zap className="w-5 h-5" />
                  Run Blockchain Benchmark
                </motion.button>
              )}

              {benchmarkError && (
                <div className="mt-4 p-4 bg-red-500/10 border border-red-500/30 rounded-xl">
                  <p className="text-sm text-red-400">{benchmarkError}</p>
                </div>
              )}
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

        {/* About Tab */}
        {activeTab === 'about' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* About Card */}
            <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
              <h3 className="text-xl font-semibold mb-6 flex items-center gap-3">
                <Info className="w-6 h-6 text-quantum-cyan" />
                About Quantum Wallet
              </h3>

              <div className="space-y-4">
                <div className="p-4 bg-quantum-dark/30 rounded-xl">
                  <div className="text-sm text-gray-400 mb-1">Version</div>
                  <div className="text-white font-semibold">v0.0.2-beta</div>
                </div>

                <div className="p-4 bg-quantum-dark/30 rounded-xl">
                  <div className="text-sm text-gray-400 mb-1">Consensus Engine</div>
                  <div className="text-white font-semibold">Q-NarwhalKnight</div>
                </div>

                <div className="p-4 bg-quantum-dark/30 rounded-xl">
                  <div className="text-sm text-gray-400 mb-1">Cryptographic Suite</div>
                  <div className="text-white font-semibold">Q1 Post-Quantum (Dilithium5 + Kyber1024)</div>
                </div>

                <div className="p-4 bg-quantum-dark/30 rounded-xl">
                  <div className="text-sm text-gray-400 mb-1">Support Email</div>
                  <a
                    href="mailto:bitknight.dipper688@passmail.net"
                    className="text-quantum-cyan font-semibold hover:text-quantum-pink transition-colors"
                  >
                    bitknight.dipper688@passmail.net
                  </a>
                </div>

                <div className="p-4 bg-quantum-cyan/10 border border-quantum-cyan/20 rounded-xl">
                  <div className="flex items-center gap-2 mb-2">
                    <Shield className="w-4 h-4 text-quantum-cyan" />
                    <span className="font-semibold text-quantum-cyan">Post-Quantum Security</span>
                  </div>
                  <p className="text-sm text-gray-400">
                    This wallet uses NIST-approved post-quantum cryptography to protect your assets against quantum computer attacks.
                  </p>
                </div>
              </div>
            </div>

            {/* Wallet Backup & Export Card */}
            <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
              <h3 className="text-xl font-semibold mb-6 flex items-center gap-3">
                <Key className="w-6 h-6 text-quantum-green" />
                Wallet Backup & Export
              </h3>

              <div className="space-y-4">
                {/* Show Private Key */}
                <div className="p-4 bg-quantum-dark/30 rounded-xl">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">Private Key</span>
                    <motion.button
                      onClick={() => openPasswordModal('private-key')}
                      className="px-3 py-1 bg-quantum-purple/20 border border-quantum-purple/30 rounded-lg text-quantum-purple text-sm hover:border-quantum-purple/60 transition-all flex items-center gap-2"
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <Eye className="w-4 h-4" />
                      Show
                    </motion.button>
                  </div>
                  <p className="text-xs text-gray-400">
                    View your post-quantum private key (requires password)
                  </p>
                  {showPrivateKey && (
                    <div className="mt-3 p-3 bg-quantum-dark/50 rounded-lg border border-quantum-cyan/20">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs text-quantum-cyan font-semibold">Private Key</span>
                        <button
                          onClick={() => setShowPrivateKey(false)}
                          className="text-gray-400 hover:text-white"
                        >
                          <EyeOff className="w-4 h-4" />
                        </button>
                      </div>
                      <div className="font-mono text-xs text-white break-all">
                        {privateKeyValue}
                      </div>
                    </div>
                  )}
                </div>

                {/* Show Mnemonic */}
                <div className="p-4 bg-quantum-dark/30 rounded-xl">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">Recovery Phrase</span>
                    <motion.button
                      onClick={() => openPasswordModal('mnemonic')}
                      className="px-3 py-1 bg-quantum-cyan/20 border border-quantum-cyan/30 rounded-lg text-quantum-cyan text-sm hover:border-quantum-cyan/60 transition-all flex items-center gap-2"
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <Eye className="w-4 h-4" />
                      Show
                    </motion.button>
                  </div>
                  <p className="text-xs text-gray-400">
                    View your 24-word mnemonic phrase (requires password)
                  </p>
                  {showMnemonic && (
                    <div className="mt-3 p-3 bg-quantum-dark/50 rounded-lg border border-quantum-cyan/20">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs text-quantum-cyan font-semibold">Mnemonic Phrase</span>
                        <button
                          onClick={() => setShowMnemonic(false)}
                          className="text-gray-400 hover:text-white"
                        >
                          <EyeOff className="w-4 h-4" />
                        </button>
                      </div>
                      <div className="font-mono text-xs text-white break-all">
                        {mnemonicValue}
                      </div>
                    </div>
                  )}
                </div>

                {/* Download Wallet File */}
                <div className="p-4 bg-quantum-dark/30 rounded-xl">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">Wallet Key File</span>
                    <motion.button
                      onClick={() => openPasswordModal('download')}
                      className="px-3 py-1 bg-quantum-green/20 border border-quantum-green/30 rounded-lg text-quantum-green text-sm hover:border-quantum-green/60 transition-all flex items-center gap-2"
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <Download className="w-4 h-4" />
                      Download
                    </motion.button>
                  </div>
                  <p className="text-xs text-gray-400">
                    Download JSON backup of your wallet (requires password)
                  </p>
                </div>

                {/* Security Warning */}
                <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl">
                  <div className="flex items-center gap-2 text-red-400 font-semibold mb-2">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                      <line x1="12" y1="9" x2="12" y2="13"></line>
                      <line x1="12" y1="17" x2="12.01" y2="17"></line>
                    </svg>
                    Security Warning
                  </div>
                  <p className="text-xs text-red-300">
                    Never share your private key or mnemonic phrase with anyone. Store backups securely offline. Anyone with access to these can steal your funds.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </motion.div>

      {/* Password Modal */}
      {showPasswordModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-[9999] p-4">
          <motion.div
            className="bg-quantum-indigo/90 backdrop-blur-xl rounded-2xl p-8 max-w-md w-full border border-quantum-cyan/30 shadow-2xl"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
          >
            <h3 className="text-2xl font-bold text-white mb-2">Enter Password</h3>
            <p className="text-gray-400 mb-6">
              {passwordModalAction === 'private-key' && 'Enter your password to view your private key'}
              {passwordModalAction === 'mnemonic' && 'Enter your password to view your recovery phrase'}
              {passwordModalAction === 'download' && 'Enter your password to download your wallet backup'}
            </p>

            <form onSubmit={handlePasswordSubmit}>
              <input
                type="password"
                value={passwordInput}
                onChange={(e) => setPasswordInput(e.target.value)}
                placeholder="Wallet password"
                className="w-full px-4 py-3 bg-quantum-dark/50 border border-quantum-purple/30 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-quantum-cyan/60 mb-4"
                autoFocus
              />

              {passwordError && (
                <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-sm">
                  {passwordError}
                </div>
              )}

              <div className="flex gap-3">
                <motion.button
                  type="button"
                  onClick={() => {
                    setShowPasswordModal(false);
                    setPasswordInput('');
                    setPasswordError('');
                  }}
                  className="flex-1 px-4 py-3 bg-gray-600/20 border border-gray-600/30 rounded-xl text-gray-300 font-semibold hover:border-gray-500/60 transition-all"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Cancel
                </motion.button>
                <motion.button
                  type="submit"
                  className="flex-1 px-4 py-3 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-xl text-white font-semibold hover:shadow-lg hover:shadow-quantum-cyan/50 transition-all"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Confirm
                </motion.button>
              </div>
            </form>
          </motion.div>
        </div>
      )}

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

// ── OAuth2 Settings Tab (real API data, no mock) ──────────────────────────────

interface ConsentEntry {
  client_id: string;
  scopes: string[];
  granted_at: string;
}

function OAuth2SettingsTab() {
  const [consents, setConsents] = useState<ConsentEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [revoking, setRevoking] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const walletAddr = localStorage.getItem('walletAddress') || '';

  const headers: Record<string, string> = {
    'X-Wallet-Auth': walletAddr,
    'Authorization': `Bearer ${walletAddr}`,
    'Content-Type': 'application/json',
  };

  const fetchConsents = useCallback(async () => {
    setError(null);
    try {
      const res = await fetch('/api/v1/oauth2/my-consents', { headers });
      if (res.status === 401) {
        setConsents([]);
        return;
      }
      if (res.ok) {
        const data = await res.json();
        setConsents(Array.isArray(data) ? data : []);
      }
    } catch {
      setError('Failed to load OAuth2 consents');
    } finally {
      setLoading(false);
    }
  }, [walletAddr]);

  useEffect(() => {
    fetchConsents();
  }, [fetchConsents]);

  const handleRevoke = async (clientId: string) => {
    setRevoking(clientId);
    try {
      const res = await fetch('/api/v1/oauth2/my-consents/revoke', {
        method: 'POST',
        headers,
        body: JSON.stringify({ client_id: clientId }),
      });
      if (res.ok) {
        setConsents(prev => prev.filter(c => c.client_id !== clientId));
      }
    } catch { /* ignore */ }
    setRevoking(null);
  };

  const scopeLabel = (scope: string) => {
    const labels: Record<string, string> = {
      'read:balance': 'Read Balance',
      'read:transactions': 'View Transactions',
      'send:transaction': 'Send Transactions',
      'read:profile': 'View Profile',
    };
    return labels[scope] || scope;
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      {/* Left: Connected Applications */}
      <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-semibold flex items-center gap-3">
            <Code className="w-6 h-6 text-quantum-purple" />
            Connected Applications
          </h3>
          <motion.button
            onClick={() => { setLoading(true); fetchConsents(); }}
            className="p-2 rounded-lg hover:bg-white/5 transition-colors"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            title="Refresh"
          >
            <RefreshCw className={`w-4 h-4 text-gray-400 ${loading ? 'animate-spin' : ''}`} />
          </motion.button>
        </div>

        <p className="text-gray-400 mb-6 text-sm">
          Third-party applications that have been granted access to your wallet via OAuth2.
        </p>

        {error && (
          <div className="p-3 mb-4 bg-red-500/10 border border-red-500/20 rounded-xl flex items-center gap-2 text-sm text-red-400">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            {error}
          </div>
        )}

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-5 h-5 text-quantum-purple animate-spin" />
            <span className="ml-2 text-gray-400 text-sm">Loading consents...</span>
          </div>
        ) : consents.length === 0 ? (
          <div className="text-center py-12">
            <div className="w-16 h-16 mx-auto mb-4 bg-quantum-dark/40 rounded-2xl border border-quantum-purple/20 flex items-center justify-center">
              <Shield className="w-8 h-8 text-gray-600" />
            </div>
            <p className="text-gray-400 font-medium">No connected applications</p>
            <p className="text-gray-500 text-sm mt-1">
              When you authorize third-party apps via OAuth2, they will appear here.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            <AnimatePresence>
              {consents.map((consent, i) => (
                <motion.div
                  key={consent.client_id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -20, height: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="p-4 bg-quantum-dark/30 rounded-xl border border-quantum-purple/20"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-gradient-to-br from-quantum-cyan/30 to-quantum-purple/30 rounded-lg flex items-center justify-center border border-quantum-purple/20">
                        <Globe className="w-5 h-5 text-quantum-cyan" />
                      </div>
                      <div>
                        <div className="font-semibold text-white text-sm">{consent.client_id}</div>
                        <div className="text-xs text-gray-500">
                          Granted: {new Date(consent.granted_at).toLocaleDateString()}
                        </div>
                      </div>
                    </div>
                    <motion.button
                      onClick={() => handleRevoke(consent.client_id)}
                      disabled={revoking === consent.client_id}
                      className="flex items-center gap-1.5 px-3 py-1.5 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-xs font-medium hover:bg-red-500/20 disabled:opacity-50 transition-all"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      {revoking === consent.client_id ? (
                        <RefreshCw className="w-3 h-3 animate-spin" />
                      ) : (
                        <Trash2 className="w-3 h-3" />
                      )}
                      Revoke
                    </motion.button>
                  </div>
                  <div className="flex flex-wrap gap-1.5 mt-2">
                    {consent.scopes.map(scope => (
                      <span
                        key={scope}
                        className="px-2 py-0.5 bg-quantum-purple/15 border border-quantum-purple/20 rounded text-xs text-quantum-purple"
                      >
                        {scopeLabel(scope)}
                      </span>
                    ))}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Right: Security & Permissions info */}
      <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
        <h3 className="text-xl font-semibold mb-6 flex items-center gap-3">
          <Shield className="w-6 h-6 text-quantum-cyan" />
          Security & Permissions
        </h3>

        <div className="space-y-4">
          <div className="p-4 bg-quantum-dark/30 rounded-xl">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium">OAuth2 Flow</span>
              <span className="text-quantum-green font-semibold">PKCE</span>
            </div>
            <p className="text-sm text-gray-400">
              Authorization Code + PKCE for maximum security
            </p>
          </div>

          <div className="p-4 bg-quantum-dark/30 rounded-xl">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium">Token Lifetime</span>
              <span className="text-quantum-cyan font-semibold">1 hour</span>
            </div>
            <p className="text-sm text-gray-400">
              Access tokens expire after 1 hour for security
            </p>
          </div>

          <div className="p-4 bg-quantum-dark/30 rounded-xl">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium">Refresh Tokens</span>
              <span className="text-quantum-green font-semibold">30 days</span>
            </div>
            <p className="text-sm text-gray-400">
              Refresh tokens auto-rotate on each use
            </p>
          </div>

          <div className="p-4 bg-quantum-dark/30 rounded-xl">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium">Available Scopes</span>
            </div>
            <div className="flex flex-wrap gap-2 mt-2">
              <span className="px-2 py-1 bg-quantum-cyan/20 border border-quantum-cyan/30 rounded text-xs text-quantum-cyan">read:balance</span>
              <span className="px-2 py-1 bg-quantum-purple/20 border border-quantum-purple/30 rounded text-xs text-quantum-purple">read:transactions</span>
              <span className="px-2 py-1 bg-red-500/20 border border-red-500/30 rounded text-xs text-red-400">send:transaction</span>
              <span className="px-2 py-1 bg-quantum-green/20 border border-quantum-green/30 rounded text-xs text-quantum-green">read:profile</span>
            </div>
          </div>

          <div className="p-4 bg-amber-500/10 border border-amber-500/30 rounded-xl">
            <div className="flex items-center gap-2 mb-2">
              <AlertCircle className="w-4 h-4 text-amber-400" />
              <span className="font-semibold text-amber-400 text-sm">Security Tips</span>
            </div>
            <ul className="text-sm text-gray-400 space-y-1">
              <li>• Review connected applications regularly</li>
              <li>• Revoke access for apps you no longer use</li>
              <li>• Never share OAuth2 tokens or auth codes</li>
              <li>• Verify redirect URIs before authorizing</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Node Admin Tab ──────────────────────────────────────────────────────────

function NodeAdminTab() {
  const [isAdmin, setIsAdmin] = useState(false);
  const [loading, setLoading] = useState(true);
  const [nodeInfo, setNodeInfo] = useState<{
    version: string;
    uptime_secs: number;
    height: number;
    network_height: number;
    peers: number;
    network_id: string;
    mining_healthy: boolean;
  } | null>(null);
  const [updateInfo, setUpdateInfo] = useState<{
    current_version: string;
    latest_version: string | null;
    update_available: boolean;
    download_url: string | null;
  } | null>(null);
  const [checkingUpdate, setCheckingUpdate] = useState(false);
  const [operatorFees, setOperatorFees] = useState<{
    node_operator_fee_promille: number;
    node_operator_fee_percent: string;
    dex_protocol_fee_bps: number;
    dex_protocol_fee_percent: string;
    admin_wallet: string;
    admin_wallet_balance_qug: number;
    founder_wallet_balance_qug: number;
  } | null>(null);
  const [isMaster, setIsMaster] = useState(false);
  const [savingFees, setSavingFees] = useState(false);
  const [feePromille, setFeePromille] = useState(0);
  const [feeBps, setFeeBps] = useState(5);

  useEffect(() => {
    const load = async () => {
      try {
        const adminCheck = await qnkAPI.isAdmin();
        setIsAdmin(adminCheck.is_admin);
        if (!adminCheck.is_admin) {
          setLoading(false);
          return;
        }
        // Fetch node info
        const info = await qnkAPI.getNodeInfo();
        if (info.data) setNodeInfo(info.data);
        // Try fetching operator fees (master wallet only — 403 = not master)
        try {
          const fees = await qnkAPI.getOperatorFees();
          if (fees.data) {
            setOperatorFees(fees.data);
            setIsMaster(true);
            setFeePromille(fees.data.node_operator_fee_promille);
            setFeeBps(fees.data.dex_protocol_fee_bps);
          }
        } catch {
          setIsMaster(false);
        }
      } catch {
        // Not admin or network error
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const handleCheckUpdate = async () => {
    setCheckingUpdate(true);
    try {
      const res = await qnkAPI.checkNodeUpdate();
      if (res.data) setUpdateInfo(res.data);
    } catch { /* ignore */ }
    setCheckingUpdate(false);
  };

  const handleSaveFees = async () => {
    setSavingFees(true);
    try {
      const res = await qnkAPI.updateOperatorFees({
        node_operator_fee_promille: feePromille,
        dex_protocol_fee_bps: feeBps,
      });
      if (res.data) {
        setOperatorFees(res.data);
      }
    } catch { /* ignore */ }
    setSavingFees(false);
  };

  const formatUptime = (secs: number) => {
    const d = Math.floor(secs / 86400);
    const h = Math.floor((secs % 86400) / 3600);
    const m = Math.floor((secs % 3600) / 60);
    if (d > 0) return `${d}d ${h}h ${m}m`;
    if (h > 0) return `${h}h ${m}m`;
    return `${m}m`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <RefreshCw className="w-6 h-6 text-quantum-purple animate-spin" />
        <span className="ml-3 text-gray-400">Loading node info...</span>
      </div>
    );
  }

  if (!isAdmin) {
    return (
      <div className="grid grid-cols-1 gap-8">
        <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8 text-center">
          <Server className="w-12 h-12 text-gray-600 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-white mb-2">Node Admin</h3>
          <p className="text-gray-400 max-w-md mx-auto">
            Node administration settings are only available to the node operator.
            Set Q_ADMIN_WALLET in your node configuration to enable this panel.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      {/* Left: Node Status */}
      <div className="space-y-6">
        <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
          <h3 className="text-xl font-semibold mb-6 flex items-center gap-3">
            <Server className="w-6 h-6 text-quantum-cyan" />
            Node Status
          </h3>

          {nodeInfo ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-quantum-dark/30 rounded-xl">
                <span className="text-gray-400">Version</span>
                <span className="text-white font-mono font-semibold">v{nodeInfo.version}</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-quantum-dark/30 rounded-xl">
                <span className="text-gray-400">Uptime</span>
                <span className="text-white font-semibold">{formatUptime(nodeInfo.uptime_secs)}</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-quantum-dark/30 rounded-xl">
                <span className="text-gray-400">Block Height</span>
                <span className="text-white font-mono">{nodeInfo.height.toLocaleString()}</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-quantum-dark/30 rounded-xl">
                <span className="text-gray-400">Network Height</span>
                <span className="text-white font-mono">{nodeInfo.network_height.toLocaleString()}</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-quantum-dark/30 rounded-xl">
                <span className="text-gray-400">Peers</span>
                <span className="text-white font-semibold">{nodeInfo.peers}</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-quantum-dark/30 rounded-xl">
                <span className="text-gray-400">Network</span>
                <span className="text-quantum-cyan font-semibold">{nodeInfo.network_id}</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-quantum-dark/30 rounded-xl">
                <span className="text-gray-400">Mining</span>
                <span className={nodeInfo.mining_healthy ? 'text-quantum-green font-semibold' : 'text-red-400 font-semibold'}>
                  {nodeInfo.mining_healthy ? 'Healthy' : 'Inactive'}
                </span>
              </div>
              {nodeInfo.height < nodeInfo.network_height - 5 && (
                <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-xl text-sm text-amber-400">
                  Syncing... {((nodeInfo.height / nodeInfo.network_height) * 100).toFixed(1)}% ({(nodeInfo.network_height - nodeInfo.height).toLocaleString()} blocks behind)
                </div>
              )}
            </div>
          ) : (
            <p className="text-gray-400">Unable to fetch node status</p>
          )}
        </div>

        {/* Update Check */}
        <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-3">
            <ArrowDownToLine className="w-6 h-6 text-quantum-green" />
            Software Updates
          </h3>

          <motion.button
            onClick={handleCheckUpdate}
            disabled={checkingUpdate}
            className="w-full py-3 px-4 bg-quantum-dark/30 border border-quantum-purple/30 rounded-xl text-white font-medium hover:border-quantum-cyan/60 transition-all flex items-center justify-center gap-2 disabled:opacity-50"
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
          >
            {checkingUpdate ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <RefreshCw className="w-4 h-4" />
            )}
            Check for Updates
          </motion.button>

          {updateInfo && (
            <div className="mt-4 space-y-3">
              <div className="flex items-center justify-between p-3 bg-quantum-dark/30 rounded-xl">
                <span className="text-gray-400">Current</span>
                <span className="text-white font-mono">v{updateInfo.current_version}</span>
              </div>
              {updateInfo.latest_version && (
                <div className="flex items-center justify-between p-3 bg-quantum-dark/30 rounded-xl">
                  <span className="text-gray-400">Latest</span>
                  <span className="text-quantum-cyan font-mono">v{updateInfo.latest_version}</span>
                </div>
              )}
              {updateInfo.update_available ? (
                <div className="p-4 bg-quantum-green/10 border border-quantum-green/30 rounded-xl">
                  <div className="flex items-center gap-2 mb-2">
                    <Zap className="w-4 h-4 text-quantum-green" />
                    <span className="font-semibold text-quantum-green">Update Available</span>
                  </div>
                  <p className="text-sm text-gray-400 mb-3">
                    Version v{updateInfo.latest_version} is available. Download and replace your node binary.
                  </p>
                  {updateInfo.download_url && (
                    <a
                      href={updateInfo.download_url}
                      className="inline-flex items-center gap-2 px-4 py-2 bg-quantum-green/20 border border-quantum-green/40 rounded-lg text-quantum-green text-sm font-medium hover:bg-quantum-green/30 transition-all"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <Download className="w-4 h-4" />
                      Download v{updateInfo.latest_version}
                    </a>
                  )}
                </div>
              ) : (
                <div className="p-3 bg-quantum-dark/30 rounded-xl text-sm text-quantum-green flex items-center gap-2">
                  <Shield className="w-4 h-4" />
                  You are running the latest version
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Right: Operator Fee Settings (master wallet only) */}
      <div className="space-y-6">
        {isMaster && operatorFees ? (
          <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
            <h3 className="text-xl font-semibold mb-6 flex items-center gap-3">
              <Zap className="w-6 h-6 text-quantum-yellow" />
              Fee Configuration
              <span className="text-xs px-2 py-0.5 bg-quantum-yellow/20 text-quantum-yellow rounded-full">Master</span>
            </h3>

            <div className="space-y-5">
              {/* Node Operator Fee */}
              <div className="p-4 bg-quantum-dark/30 rounded-xl">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <span className="font-medium text-white">Node Operator Fee</span>
                    <p className="text-xs text-gray-400 mt-1">Share of collected fees routed to admin wallet</p>
                  </div>
                  <span className="text-quantum-cyan font-mono font-semibold">{(feePromille / 10).toFixed(1)}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={500}
                  step={10}
                  value={feePromille}
                  onChange={(e) => setFeePromille(Number(e.target.value))}
                  className="w-full accent-quantum-cyan"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>0%</span>
                  <span>{feePromille} promille</span>
                  <span>50%</span>
                </div>
              </div>

              {/* DEX Protocol Fee */}
              <div className="p-4 bg-quantum-dark/30 rounded-xl">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <span className="font-medium text-white">DEX Protocol Fee</span>
                    <p className="text-xs text-gray-400 mt-1">Fee extracted from each swap (in basis points)</p>
                  </div>
                  <span className="text-quantum-cyan font-mono font-semibold">{(feeBps / 100).toFixed(2)}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={1}
                  value={feeBps}
                  onChange={(e) => setFeeBps(Number(e.target.value))}
                  className="w-full accent-quantum-cyan"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>0 bps</span>
                  <span>{feeBps} bps</span>
                  <span>10 bps (0.1%)</span>
                </div>
              </div>

              {/* Wallet Balances */}
              <div className="p-4 bg-quantum-dark/30 rounded-xl space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-400 text-sm">Admin Wallet</span>
                  <span className="text-white font-mono text-sm">{operatorFees.admin_wallet}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400 text-sm">Admin Balance</span>
                  <span className="text-quantum-green font-mono text-sm">{operatorFees.admin_wallet_balance_qug.toFixed(4)} QUG</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400 text-sm">Founder Balance</span>
                  <span className="text-quantum-cyan font-mono text-sm">{operatorFees.founder_wallet_balance_qug.toFixed(4)} QUG</span>
                </div>
              </div>

              {/* Save Button */}
              <motion.button
                onClick={handleSaveFees}
                disabled={savingFees}
                className="w-full py-3 px-4 bg-gradient-to-r from-quantum-purple to-quantum-cyan rounded-xl text-white font-semibold hover:shadow-lg hover:shadow-quantum-purple/50 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
              >
                {savingFees ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                Save Fee Settings
              </motion.button>
            </div>
          </div>
        ) : (
          <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-3">
              <Shield className="w-6 h-6 text-quantum-cyan" />
              Admin Wallet
            </h3>
            <p className="text-gray-400 text-sm">
              You are connected as the node admin. Fee configuration is only available to the master (founder) wallet.
            </p>
          </div>
        )}

        {/* Admin Info Card */}
        <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-3">
            <Info className="w-6 h-6 text-quantum-cyan" />
            Configuration
          </h3>
          <div className="space-y-3 text-sm">
            <div className="p-3 bg-quantum-dark/30 rounded-xl">
              <span className="text-gray-400">Admin wallet is set via </span>
              <code className="text-quantum-cyan">Q_ADMIN_WALLET</code>
              <span className="text-gray-400"> env var at node startup.</span>
            </div>
            <div className="p-3 bg-quantum-dark/30 rounded-xl">
              <span className="text-gray-400">Operator fee is set via </span>
              <code className="text-quantum-cyan">Q_NODE_OPERATOR_FEE_PROMILLE</code>
              <span className="text-gray-400"> env var (0-500).</span>
            </div>
            <div className="p-3 bg-quantum-dark/30 rounded-xl">
              <span className="text-gray-400">DEX protocol fee is set via </span>
              <code className="text-quantum-cyan">Q_DEX_PROTOCOL_FEE_BPS</code>
              <span className="text-gray-400"> env var (0-10).</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}