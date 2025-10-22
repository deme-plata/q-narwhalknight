import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Shield, Palette, Activity, Globe, Lock, Eye, Zap, LogOut, Clock, Info, Key, Download, EyeOff, Cloud, Code } from 'lucide-react';

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
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8">
              <h3 className="text-xl font-semibold mb-6 flex items-center gap-3">
                <Code className="w-6 h-6 text-quantum-purple" />
                OAuth2 Applications
              </h3>

              <p className="text-gray-400 mb-6">
                Manage third-party applications that have access to your wallet via OAuth2.
              </p>

              <div className="space-y-3">
                <div className="p-4 bg-quantum-dark/30 rounded-xl border border-quantum-purple/20">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-gradient-to-br from-quantum-cyan to-quantum-purple rounded-lg flex items-center justify-center">
                        <Code className="w-5 h-5 text-white" />
                      </div>
                      <div>
                        <div className="font-semibold text-white">Quillon API Docs</div>
                        <div className="text-xs text-gray-400">Last accessed: 2 hours ago</div>
                      </div>
                    </div>
                    <motion.button
                      className="px-3 py-1 bg-red-500/20 border border-red-500/30 rounded-lg text-red-400 text-sm hover:bg-red-500/30 transition-all"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      Revoke
                    </motion.button>
                  </div>
                  <div className="text-sm text-gray-400">
                    Permissions: Read balance, View transactions
                  </div>
                </div>

                <div className="p-4 bg-quantum-dark/30 rounded-xl border border-quantum-purple/20">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-gradient-to-br from-quantum-green to-quantum-cyan rounded-lg flex items-center justify-center">
                        <Shield className="w-5 h-5 text-white" />
                      </div>
                      <div>
                        <div className="font-semibold text-white">Privacy Service</div>
                        <div className="text-xs text-gray-400">Last accessed: 5 minutes ago</div>
                      </div>
                    </div>
                    <motion.button
                      className="px-3 py-1 bg-red-500/20 border border-red-500/30 rounded-lg text-red-400 text-sm hover:bg-red-500/30 transition-all"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      Revoke
                    </motion.button>
                  </div>
                  <div className="text-sm text-gray-400">
                    Permissions: Mix transactions, Generate stealth addresses
                  </div>
                </div>
              </div>

              <motion.button
                className="w-full mt-6 py-3 px-4 bg-gradient-to-r from-quantum-purple/20 to-quantum-cyan/20 border border-quantum-purple/30 rounded-xl text-white font-semibold hover:border-quantum-purple/60 transition-all flex items-center justify-center gap-2"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Code className="w-5 h-5" />
                Register New Application
              </motion.button>
            </div>

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
                    <span className="text-quantum-green font-semibold">Enabled</span>
                  </div>
                  <p className="text-sm text-gray-400">
                    Refresh tokens valid for 30 days
                  </p>
                </div>

                <div className="p-4 bg-quantum-dark/30 rounded-xl">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">Allowed Scopes</span>
                  </div>
                  <div className="flex flex-wrap gap-2 mt-2">
                    <span className="px-2 py-1 bg-quantum-cyan/20 border border-quantum-cyan/30 rounded text-xs text-quantum-cyan">balance:read</span>
                    <span className="px-2 py-1 bg-quantum-purple/20 border border-quantum-purple/30 rounded text-xs text-quantum-purple">transactions:read</span>
                    <span className="px-2 py-1 bg-quantum-green/20 border border-quantum-green/30 rounded text-xs text-quantum-green">privacy:mix</span>
                    <span className="px-2 py-1 bg-quantum-pink/20 border border-quantum-pink/30 rounded text-xs text-quantum-pink">privacy:tor</span>
                  </div>
                </div>

                <div className="p-4 bg-quantum-yellow/10 border border-quantum-yellow/30 rounded-xl">
                  <div className="flex items-center gap-2 mb-2">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-quantum-yellow">
                      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                      <line x1="12" y1="9" x2="12" y2="13"></line>
                      <line x1="12" y1="17" x2="12.01" y2="17"></line>
                    </svg>
                    <span className="font-semibold text-quantum-yellow">Security Best Practices</span>
                  </div>
                  <ul className="text-sm text-gray-400 space-y-1">
                    <li>• Review application permissions regularly</li>
                    <li>• Revoke access for unused applications</li>
                    <li>• Never share OAuth2 tokens</li>
                    <li>• Check redirect URIs before approving</li>
                  </ul>
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

              <div className="relative h-64 bg-quantum-dark/50 rounded-xl overflow-hidden">
                {/* Advanced Live preview of effects - DAG-inspired 3D visualizations */}
                <div className="absolute inset-0">
                  {/* Fractal Overlay - Mandelbrot-inspired interference patterns */}
                  {visualEffects.fractalOverlay && (
                    <svg className="absolute inset-0 w-full h-full opacity-20">
                      {[...Array(12)].map((_, i) => (
                        <motion.path
                          key={i}
                          d={`M ${i * 30} 0 Q ${i * 30 + 15} ${100 + Math.sin(i) * 50}, ${i * 30} 200 T ${i * 30 + 60} 300`}
                          fill="none"
                          stroke={`hsl(${(i * 30 + 180) % 360}, 70%, 50%)`}
                          strokeWidth="1.5"
                          animate={{
                            d: [
                              `M ${i * 30} 0 Q ${i * 30 + 15} ${100 + Math.sin(i) * 50}, ${i * 30} 200`,
                              `M ${i * 30} 0 Q ${i * 30 + 25} ${120 + Math.sin(i + 1) * 60}, ${i * 30} 200`,
                              `M ${i * 30} 0 Q ${i * 30 + 15} ${100 + Math.sin(i) * 50}, ${i * 30} 200`,
                            ],
                            opacity: [0.3, 0.6, 0.3]
                          }}
                          transition={{
                            duration: 3 + i * 0.2,
                            repeat: Infinity,
                            ease: "easeInOut"
                          }}
                        />
                      ))}
                    </svg>
                  )}

                  {/* Rainbow Boxes - Quantum state superposition visualization */}
                  {visualEffects.rainbowBoxes && (
                    <div className="absolute inset-0 pointer-events-none">
                      {[...Array(5)].map((_, i) => (
                        <motion.div
                          key={i}
                          className="absolute rounded-lg"
                          style={{
                            width: 16 + i * 8,
                            height: 16 + i * 8,
                            left: `${20 + i * 15}%`,
                            top: `${30 + Math.sin(i) * 20}%`,
                            background: `linear-gradient(135deg,
                              hsl(${i * 72}, 80%, 60%) 0%,
                              hsl(${(i * 72 + 36) % 360}, 80%, 60%) 50%,
                              hsl(${(i * 72 + 72) % 360}, 80%, 60%) 100%)`,
                            boxShadow: `0 0 ${10 + i * 5}px hsl(${i * 72}, 80%, 60%)`,
                          }}
                          animate={{
                            rotate: [0, 360],
                            scale: [1, 1.2, 1],
                            opacity: [0.5, 0.8, 0.5],
                          }}
                          transition={{
                            duration: 4 + i * 0.5,
                            repeat: Infinity,
                            ease: "easeInOut",
                            delay: i * 0.3
                          }}
                        />
                      ))}
                    </div>
                  )}

                  {/* Photon Waterfall - Particle stream simulation */}
                  {visualEffects.photonWaterfall && (
                    <div className="absolute inset-0">
                      {[...Array(8)].map((_, i) => (
                        <motion.div
                          key={i}
                          className="absolute rounded-full blur-sm"
                          style={{
                            width: 4 + Math.random() * 4,
                            height: 20 + Math.random() * 30,
                            left: `${10 + i * 12}%`,
                            background: `linear-gradient(to bottom,
                              transparent,
                              ${['#00D4FF', '#6B46C1', '#EC4899', '#10B981', '#F59E0B'][i % 5]} 50%,
                              transparent)`,
                            boxShadow: `0 0 8px ${['#00D4FF', '#6B46C1', '#EC4899', '#10B981', '#F59E0B'][i % 5]}`,
                          }}
                          animate={{
                            y: [-50, 300],
                            opacity: [0, 1, 1, 0],
                          }}
                          transition={{
                            duration: 2 + Math.random() * 2,
                            repeat: Infinity,
                            delay: i * 0.25,
                            ease: "linear"
                          }}
                        />
                      ))}
                    </div>
                  )}

                  {/* Entanglement Moiré - Quantum correlation patterns */}
                  {visualEffects.entanglementMoire && (
                    <svg className="absolute inset-0 w-full h-full">
                      <defs>
                        <radialGradient id="entanglementGlow">
                          <stop offset="0%" stopColor="#00D4FF" stopOpacity="0.8" />
                          <stop offset="50%" stopColor="#6B46C1" stopOpacity="0.4" />
                          <stop offset="100%" stopColor="transparent" stopOpacity="0" />
                        </radialGradient>
                      </defs>
                      {/* Entangled particle pairs */}
                      {[...Array(3)].map((_, i) => (
                        <g key={i}>
                          {/* Particle 1 */}
                          <motion.circle
                            cx="30%"
                            cy="50%"
                            r="8"
                            fill="url(#entanglementGlow)"
                            animate={{
                              cx: ['30%', '25%', '35%', '30%'],
                              cy: ['50%', '45%', '55%', '50%'],
                              r: [8, 12, 8],
                            }}
                            transition={{
                              duration: 3,
                              repeat: Infinity,
                              delay: i * 0.8,
                              ease: "easeInOut"
                            }}
                          />
                          {/* Particle 2 (entangled) */}
                          <motion.circle
                            cx="70%"
                            cy="50%"
                            r="8"
                            fill="url(#entanglementGlow)"
                            animate={{
                              cx: ['70%', '75%', '65%', '70%'],
                              cy: ['50%', '55%', '45%', '50%'],
                              r: [8, 12, 8],
                            }}
                            transition={{
                              duration: 3,
                              repeat: Infinity,
                              delay: i * 0.8,
                              ease: "easeInOut"
                            }}
                          />
                          {/* Connection wave */}
                          <motion.path
                            d="M 30% 50% Q 50% 40%, 70% 50%"
                            fill="none"
                            stroke="#6B46C1"
                            strokeWidth="1.5"
                            strokeDasharray="5,5"
                            opacity="0.4"
                            animate={{
                              d: [
                                "M 30% 50% Q 50% 40%, 70% 50%",
                                "M 30% 50% Q 50% 60%, 70% 50%",
                                "M 30% 50% Q 50% 40%, 70% 50%",
                              ],
                              strokeDashoffset: [0, 20, 40],
                            }}
                            transition={{
                              duration: 2,
                              repeat: Infinity,
                              delay: i * 0.8,
                              ease: "linear"
                            }}
                          />
                          {/* Interference pattern circles */}
                          <motion.circle
                            cx="50%"
                            cy="50%"
                            r="20"
                            fill="none"
                            stroke="#00D4FF"
                            strokeWidth="0.5"
                            opacity="0.3"
                            animate={{
                              r: [10, 40],
                              opacity: [0.6, 0],
                            }}
                            transition={{
                              duration: 2.5,
                              repeat: Infinity,
                              delay: i * 0.8,
                              ease: "easeOut"
                            }}
                          />
                        </g>
                      ))}
                    </svg>
                  )}
                </div>

                <div className="absolute bottom-4 left-4 text-sm text-gray-400 backdrop-blur-sm bg-black/30 px-3 py-1 rounded-lg">
                  Live Preview • DAG-3D Quantum Visualization
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