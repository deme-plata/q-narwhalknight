import { useState } from 'react';
import { motion } from 'framer-motion';
import { Shield, Sparkles, Key, AlertCircle } from 'lucide-react';
import { qnkAPI } from '../services/api';

interface LoginScreenProps {
  onAuthenticate: () => void;
}

export default function LoginScreen({ onAuthenticate }: LoginScreenProps) {
  const [seedPhrase, setSeedPhrase] = useState('');
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [showQuantumGenerator, setShowQuantumGenerator] = useState(false);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleAuthenticate = async () => {
    setIsAuthenticating(true);
    
    // Store the seed phrase securely (in production, use proper encryption)
    if (seedPhrase) {
      localStorage.setItem('walletSeed', seedPhrase);
    }
    
    // Simulate authentication with quantum animations
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    onAuthenticate();
  };

  const generateQuantumSeed = async () => {
    setIsGenerating(true);
    setGenerationError(null);
    setShowQuantumGenerator(true);
    
    try {
      // Show quantum generation animation
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // Call the actual Q-NarwhalKnight API to generate BIP39 mnemonic
      const response = await qnkAPI.generateMnemonic();
      
      if (response.success && response.data) {
        // Set the actual BIP39 mnemonic from the quantum consensus node
        setSeedPhrase(response.data.mnemonic);
        
        // Show additional visual feedback for successful generation
        await new Promise(resolve => setTimeout(resolve, 700));
      } else {
        throw new Error(response.error || 'Failed to generate mnemonic');
      }
    } catch (error) {
      console.error('Quantum seed generation failed:', error);
      setGenerationError(error instanceof Error ? error.message : 'Unknown error');
      
      // Fallback to demo seed for development
      setSeedPhrase('abandon ability able about above absent absorb abstract absurd abuse access accident');
    } finally {
      setShowQuantumGenerator(false);
      setIsGenerating(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen px-4">
      <motion.div 
        className="w-full max-w-md"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        {/* Logo and Title */}
        <div className="text-center mb-12">
          <motion.div 
            className="inline-block mb-6"
            animate={{ rotate: 360 }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          >
            <div className="w-32 h-32 mx-auto relative">
              <div className="absolute inset-0 rainbow-box rounded-full blur-2xl opacity-50" />
              <div className="absolute inset-0 bg-gradient-to-br from-quantum-purple to-quantum-cyan rounded-full flex items-center justify-center">
                <Shield className="w-16 h-16 text-white" />
              </div>
            </div>
          </motion.div>
          
          <h1 className="text-4xl lg:text-5xl font-bold bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink bg-clip-text text-transparent">
            Q-NarwhalKnight
          </h1>
          <p className="text-gray-400 mt-2">Quantum Consensus Wallet</p>
        </div>

        {/* Login Form */}
        <div className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8 quantum-glow">
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                BIP39 Quantum Seed Phrase
              </label>
              <textarea
                value={seedPhrase}
                onChange={(e) => setSeedPhrase(e.target.value)}
                className="w-full h-24 px-4 py-3 bg-quantum-dark/50 border border-quantum-purple/30 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-quantum-cyan transition-colors resize-none"
                placeholder="Enter your 12-word seed phrase..."
              />
            </div>

            {/* Quantum Generator Button */}
            <motion.button
              onClick={generateQuantumSeed}
              disabled={isGenerating}
              className="w-full py-4 px-6 bg-gradient-to-r from-quantum-purple/20 to-quantum-cyan/20 border border-quantum-cyan/30 rounded-xl text-white font-medium flex items-center justify-center gap-3 hover:border-quantum-cyan/60 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              whileHover={{ scale: isGenerating ? 1 : 1.02 }}
              whileTap={{ scale: isGenerating ? 1 : 0.98 }}
            >
              {isGenerating ? (
                <>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  >
                    <Sparkles className="w-5 h-5" />
                  </motion.div>
                  <span>Generating BIP39 Mnemonic...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  <span>Generate Quantum Entropy</span>
                </>
              )}
            </motion.button>

            {/* Error Display */}
            {generationError && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-4 bg-red-500/20 border border-red-500/30 rounded-xl text-red-300 text-sm flex items-center gap-2"
              >
                <AlertCircle className="w-4 h-4" />
                <span>Generation failed: {generationError}</span>
              </motion.div>
            )}

            {/* Quantum Generator Animation */}
            {showQuantumGenerator && (
              <motion.div 
                className="h-32 rounded-xl overflow-hidden relative"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <div className="absolute inset-0 rainbow-box animate-rainbow-shift" />
                <div className="absolute inset-0 bg-quantum-dark/80 flex items-center justify-center">
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  >
                    <Sparkles className="w-12 h-12 text-white" />
                  </motion.div>
                </div>
              </motion.div>
            )}

            {/* Authenticate Button */}
            <motion.button
              onClick={handleAuthenticate}
              disabled={!seedPhrase || isAuthenticating}
              className="w-full py-5 px-6 bg-gradient-to-r from-quantum-purple to-quantum-cyan rounded-xl text-white font-bold text-lg flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-2xl hover:shadow-quantum-cyan/25 transition-all"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              {isAuthenticating ? (
                <>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  >
                    <Shield className="w-6 h-6" />
                  </motion.div>
                  <span>Quantum Authenticating...</span>
                </>
              ) : (
                <>
                  <Key className="w-6 h-6" />
                  <span>Authenticate</span>
                </>
              )}
            </motion.button>
          </div>

          {/* Photon Waterfall Effect */}
          {isAuthenticating && (
            <motion.div 
              className="mt-6 h-2 rounded-full overflow-hidden"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <motion.div
                className="h-full bg-gradient-to-r from-quantum-green via-quantum-cyan to-quantum-purple"
                initial={{ x: '-100%' }}
                animate={{ x: '100%' }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              />
            </motion.div>
          )}
        </div>

        {/* Security Note */}
        <p className="text-center text-gray-500 text-sm mt-6">
          🔐 Post-Quantum Cryptography • 🌊 DAG-BFT Consensus • 🧅 Tor Integration
        </p>
      </motion.div>
    </div>
  );
}