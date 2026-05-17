import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Wallet, X, Key, AlertCircle, CheckCircle, Loader } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import {
  generateKeyPair,
  keypairFromMnemonic,
  generateAuthMessage,
  signMessage,
  storeWalletSession,
  type WalletKeyPair,
} from '../services/wallet';
import axios from 'axios';

interface WalletConnectProps {
  onClose: () => void;
}

export function WalletConnect({ onClose }: WalletConnectProps) {
  const { login } = useAuth();
  const [mode, setMode] = useState<'existing' | 'new'>('existing');
  const [mnemonic, setMnemonic] = useState('');
  const [generatedMnemonic, setGeneratedMnemonic] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [walletAddress, setWalletAddress] = useState('');

  // Generate new wallet
  const handleGenerateWallet = async () => {
    try {
      setLoading(true);
      setError('');

      // Generate random mnemonic (12 words)
      const words = [
        'quantum', 'consensus', 'narwhal', 'knight', 'blockchain', 'crypto',
        'consensus', 'validator', 'network', 'protocol', 'secure', 'defi',
      ];
      const shuffled = words.sort(() => Math.random() - 0.5);
      const newMnemonic = shuffled.slice(0, 12).join(' ');

      setGeneratedMnemonic(newMnemonic);
      setMode('new');
    } catch (err: any) {
      setError(err.message || 'Failed to generate wallet');
    } finally {
      setLoading(false);
    }
  };

  // Connect with existing wallet
  const handleConnectExisting = async () => {
    if (!mnemonic.trim()) {
      setError('Please enter your mnemonic phrase');
      return;
    }

    try {
      setLoading(true);
      setError('');

      // Derive keypair from mnemonic
      const keyPair = await keypairFromMnemonic(mnemonic);
      await authenticateWallet(keyPair);
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Failed to connect wallet');
    } finally {
      setLoading(false);
    }
  };

  // Connect with new wallet
  const handleConnectNew = async () => {
    if (!generatedMnemonic) {
      setError('No wallet generated');
      return;
    }

    try {
      setLoading(true);
      setError('');

      // Derive keypair from generated mnemonic
      const keyPair = await keypairFromMnemonic(generatedMnemonic);
      await authenticateWallet(keyPair);
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Failed to connect wallet');
    } finally {
      setLoading(false);
    }
  };

  // Authenticate wallet with backend
  const authenticateWallet = async (keyPair: WalletKeyPair) => {
    // Generate authentication message
    const message = generateAuthMessage(keyPair.address);

    // Sign the message
    const signature = await signMessage(message, keyPair.privateKey);

    // Convert public key and signature to hex for backend
    const publicKeyHex = Array.from(keyPair.publicKey)
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');

    // Send to backend for authentication
    const response = await axios.post('http://localhost:9070/wallet/connect', {
      address: keyPair.address,
      public_key: publicKeyHex,
      message,
      signature,
    });

    const { token, user_id } = response.data;

    // Store wallet session
    storeWalletSession(keyPair);

    // Update auth context
    login(user_id, keyPair.address, token);

    // Show success
    setWalletAddress(keyPair.address);
    setSuccess(true);

    // Close modal after 2 seconds
    setTimeout(() => {
      onClose();
    }, 2000);
  };

  // Copy mnemonic to clipboard
  const handleCopyMnemonic = () => {
    navigator.clipboard.writeText(generatedMnemonic);
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="bg-gradient-to-br from-gray-900 to-black border-2 border-cyan-500/50 rounded-xl max-w-md w-full shadow-2xl overflow-hidden"
      >
        {/* Header */}
        <div className="bg-gradient-to-r from-cyan-600/20 to-magenta-600/20 border-b border-cyan-500/30 p-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-magenta-500 rounded-lg flex items-center justify-center">
              <Wallet className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-cyan-400 font-mono">
                Connect Wallet
              </h2>
              <p className="text-gray-400 text-sm font-mono">
                Sign in to Bounty Campaign
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-cyan-400 transition-colors"
          >
            <X size={24} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {success ? (
            // Success State
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center py-8"
            >
              <div className="w-16 h-16 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                <CheckCircle className="w-10 h-10 text-green-400" />
              </div>
              <h3 className="text-xl font-bold text-green-400 font-mono mb-2">
                Wallet Connected!
              </h3>
              <p className="text-gray-400 font-mono text-sm mb-4">
                Address: {walletAddress.substring(0, 20)}...
              </p>
              <p className="text-gray-500 text-xs">
                Redirecting to dashboard...
              </p>
            </motion.div>
          ) : (
            <>
              {/* Mode Selection */}
              {!generatedMnemonic && (
                <div className="flex gap-3">
                  <button
                    onClick={() => setMode('existing')}
                    className={`flex-1 py-3 px-4 rounded-lg font-mono text-sm transition-all ${
                      mode === 'existing'
                        ? 'bg-cyan-500/20 text-cyan-400 border-2 border-cyan-500/50'
                        : 'bg-gray-800/50 text-gray-400 border border-gray-700 hover:border-cyan-500/30'
                    }`}
                  >
                    <Key className="w-4 h-4 inline mr-2" />
                    Existing Wallet
                  </button>
                  <button
                    onClick={handleGenerateWallet}
                    disabled={loading}
                    className={`flex-1 py-3 px-4 rounded-lg font-mono text-sm transition-all ${
                      mode === 'new'
                        ? 'bg-magenta-500/20 text-magenta-400 border-2 border-magenta-500/50'
                        : 'bg-gray-800/50 text-gray-400 border border-gray-700 hover:border-magenta-500/30'
                    }`}
                  >
                    <Wallet className="w-4 h-4 inline mr-2" />
                    New Wallet
                  </button>
                </div>
              )}

              {/* Existing Wallet Input */}
              {mode === 'existing' && !generatedMnemonic && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-cyan-400 font-mono text-sm mb-2">
                      Mnemonic Phrase (12 words)
                    </label>
                    <textarea
                      value={mnemonic}
                      onChange={(e) => setMnemonic(e.target.value)}
                      placeholder="Enter your 12-word mnemonic phrase..."
                      rows={3}
                      className="w-full px-4 py-3 bg-gray-900/50 border border-cyan-500/30 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 transition-all font-mono text-sm"
                    />
                  </div>

                  <button
                    onClick={handleConnectExisting}
                    disabled={loading || !mnemonic.trim()}
                    className="w-full py-3 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white font-bold rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {loading ? (
                      <>
                        <Loader className="w-5 h-5 animate-spin" />
                        Connecting...
                      </>
                    ) : (
                      <>
                        <Wallet className="w-5 h-5" />
                        Connect Wallet
                      </>
                    )}
                  </button>
                </div>
              )}

              {/* New Wallet Generated */}
              {generatedMnemonic && (
                <div className="space-y-4">
                  <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
                    <div className="flex items-start gap-3">
                      <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                      <div>
                        <p className="text-yellow-400 font-mono text-sm font-bold mb-1">
                          Important: Save Your Mnemonic
                        </p>
                        <p className="text-yellow-300/80 text-xs">
                          This is your wallet's recovery phrase. Store it securely and never share it with anyone.
                        </p>
                      </div>
                    </div>
                  </div>

                  <div>
                    <label className="block text-cyan-400 font-mono text-sm mb-2">
                      Your Mnemonic Phrase
                    </label>
                    <div className="relative">
                      <div className="px-4 py-3 bg-gray-900/80 border-2 border-cyan-500/50 rounded-lg text-cyan-300 font-mono text-sm select-all">
                        {generatedMnemonic}
                      </div>
                      <button
                        onClick={handleCopyMnemonic}
                        className="absolute top-2 right-2 px-3 py-1 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-400 rounded text-xs font-mono transition-colors"
                      >
                        Copy
                      </button>
                    </div>
                  </div>

                  <button
                    onClick={handleConnectNew}
                    disabled={loading}
                    className="w-full py-3 bg-gradient-to-r from-magenta-500 to-purple-500 hover:from-magenta-600 hover:to-purple-600 text-white font-bold rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {loading ? (
                      <>
                        <Loader className="w-5 h-5 animate-spin" />
                        Creating Wallet...
                      </>
                    ) : (
                      <>
                        <Wallet className="w-5 h-5" />
                        Create & Connect Wallet
                      </>
                    )}
                  </button>
                </div>
              )}

              {/* Error Display */}
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 flex items-start gap-3"
                >
                  <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-red-400 font-mono text-sm font-bold mb-1">
                      Connection Failed
                    </p>
                    <p className="text-red-300/80 text-xs">{error}</p>
                  </div>
                </motion.div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        {!success && (
          <div className="bg-gray-900/50 border-t border-gray-800 p-4 text-center">
            <p className="text-gray-500 font-mono text-xs">
              Powered by Q-NarwhalKnight Quantum Consensus
            </p>
          </div>
        )}
      </motion.div>
    </div>
  );
}
