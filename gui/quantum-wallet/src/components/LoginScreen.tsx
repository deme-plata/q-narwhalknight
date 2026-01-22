import { useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Key, AlertCircle } from 'lucide-react';
import { qnkAPI } from '../services/api';
import { storeWallet, walletSession, verifyPasswordHash, hasPasswordHash } from '../services/walletAuth';

interface LoginScreenProps {
  onAuthenticate: () => void;
}

export default function LoginScreen({ onAuthenticate }: LoginScreenProps) {
  const [seedPhrase, setSeedPhrase] = useState('');
  const [password, setPassword] = useState('');
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [showQuantumGenerator, setShowQuantumGenerator] = useState(false);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleAuthenticate = async () => {
    setIsAuthenticating(true);
    setGenerationError(null);

    try {
      // Password is REQUIRED
      if (!password) {
        throw new Error('Password is required for wallet encryption');
      }

      // CRITICAL SECURITY FIX v1.0.68-beta: MANDATORY password verification for existing wallets
      // Bug: Previously, correct mnemonic + wrong password would bypass verification
      // because the private key is derived from mnemonic alone (password only encrypts storage)

      const encryptedMnemonic = localStorage.getItem('walletEncryptedMnemonic');
      const encryptedKey = localStorage.getItem('walletEncryptedKey');
      const storedAddress = localStorage.getItem('walletAddress');

      // Derive address from provided mnemonic to check if it matches stored wallet
      const { keypairFromMnemonic, recoverMnemonic, loadWallet } = await import('../services/walletAuth');
      const providedKeyPair = await keypairFromMnemonic(seedPhrase);
      const providedWalletAddress = providedKeyPair.address;

      // Check if ANY encrypted data exists for this wallet address
      const hasExistingEncryptedWallet = storedAddress && (encryptedMnemonic || encryptedKey);

      if (hasExistingEncryptedWallet && providedWalletAddress === storedAddress) {
        // SAME WALLET - password verification is ABSOLUTELY REQUIRED
        // Without this, attacker with mnemonic can bypass password protection!
        console.log('🔐 Existing wallet found - MANDATORY password verification...');
        console.log('🔐 Same wallet detected (addresses match) - password verification is MANDATORY');

        try {
          // Try to decrypt using the most reliable method available
          if (encryptedMnemonic) {
            // Best case: We have encrypted mnemonic, verify it
            const storedMnemonic = await recoverMnemonic(password);
            if (storedMnemonic.trim() !== seedPhrase.trim()) {
              console.error('❌ CRITICAL: Address matched but mnemonic different!');
              throw new Error('Wallet data corruption detected. Please contact support.');
            }
            console.log('✅ Password verified via mnemonic decryption!');
          } else if (encryptedKey) {
            // Fallback: We have encrypted key but no mnemonic (legacy wallet)
            // Try to load wallet - this will fail if password is wrong
            await loadWallet(password);
            console.log('✅ Password verified via key decryption (legacy wallet)!');
          }
        } catch (decryptError) {
          // Decryption failed = WRONG PASSWORD - MUST STOP HERE!
          console.error('❌ WRONG PASSWORD - Authentication BLOCKED');
          console.error('   Error:', decryptError);
          console.error('   This is a SECURITY feature - password protects wallet access');
          setIsAuthenticating(false);
          setGenerationError('Incorrect password. Please enter the correct password for your existing wallet.');
          return; // CRITICAL: Stop execution - do NOT proceed to createWallet!
        }
      } else if (storedAddress && providedWalletAddress !== storedAddress) {
        // DIFFERENT MNEMONIC - warn user and clear old data
        console.warn('⚠️ Different wallet detected (address mismatch)');
        console.log('   Stored address: ' + storedAddress);
        console.log('   Provided address: ' + providedWalletAddress);
        console.log('🗑️ Clearing old wallet data to create new wallet');
        localStorage.removeItem('walletEncryptedMnemonic');
        localStorage.removeItem('walletEncryptedKey');
        localStorage.removeItem('walletAddress');
        localStorage.removeItem('walletPublicKey');
        localStorage.removeItem('walletEncryptedAegisKey');
        localStorage.removeItem('walletAegisPublicKey');
        localStorage.removeItem('walletPasswordHash');
      } else if (!hasExistingEncryptedWallet && hasPasswordHash()) {
        // v2.3.8-beta: CRITICAL SECURITY FIX
        // No encrypted data but password hash exists - verify password using hash
        // This prevents login with wrong password when encrypted data is lost
        console.log('🔐 No encrypted data but password hash exists - verifying password...');

        const isPasswordValid = await verifyPasswordHash(password);
        if (!isPasswordValid) {
          console.error('❌ WRONG PASSWORD - Password hash verification failed');
          setIsAuthenticating(false);
          setGenerationError('Incorrect password. Please enter the correct password for your wallet.');
          return;
        }
        console.log('✅ Password verified via hash!');
      }
      // Note: If no stored address AND no password hash exists, this is a brand new wallet - no verification needed

      // Call the import wallet API with mnemonic and password
      const response = await qnkAPI.createWallet(seedPhrase, password);

      if (response.success && response.data) {
        // Store wallet address and ID
        localStorage.setItem('walletAddress', response.data.address_formatted || '');
        localStorage.setItem('walletId', response.data.id);

        // CRITICAL: Clear cached balance from previous wallet
        localStorage.removeItem('cachedBalance');
        console.log('🗑️ Cleared cached balance from previous wallet');

        try {
          // IMPORTANT: Enable AEGIS-QL post-quantum keys by default
          const wallet = await storeWallet(seedPhrase, password, true);
          // Automatically start session so user doesn't need to enter password again
          // Pass mnemonic to session for "Never expire" convenience (stored only if timeout is "never")
          walletSession.setSession(wallet.privateKey, wallet.address, seedPhrase);
          console.log('✅ Wallet encrypted with password-protected AES-256-GCM');
          console.log('✅ AEGIS-QL post-quantum keys generated and encrypted');
          console.log('✅ Session started with mnemonic for "Never expire" convenience');
        } catch (error) {
          console.error('Failed to encrypt wallet:', error);
          throw new Error(`Wallet encryption failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }

        // Show success animation
        await new Promise(resolve => setTimeout(resolve, 1000));

        onAuthenticate();
      } else {
        throw new Error(response.error || 'Failed to import wallet');
      }
    } catch (error) {
      console.error('Wallet authentication failed:', error);
      setGenerationError(error instanceof Error ? error.message : 'Authentication failed');
      setIsAuthenticating(false);
    }
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
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
          >
            <div className="w-40 h-40 mx-auto relative">
              {/* Cosmic glow effect */}
              <div className="absolute inset-0 bg-gradient-to-b from-amber-500/20 via-orange-500/20 to-yellow-500/20 rounded-full blur-3xl animate-pulse" />
              {/* Gold border ring */}
              <div className="absolute inset-0 rounded-full" style={{
                background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 25%, #FFA500 50%, #FFD700 75%, #D4AF37 100%)',
                padding: '3px'
              }}>
                <div className="w-full h-full bg-gradient-to-b from-slate-900 via-blue-950 to-slate-900 rounded-full flex items-center justify-center p-4">
                  <img
                    src="/quillon-logo.png"
                    alt="Quillon Graph Logo"
                    className="w-full h-full object-contain"
                    style={{ filter: 'invert(1)' }}
                  />
                </div>
              </div>
            </div>
          </motion.div>

          <h1 className="text-4xl lg:text-5xl font-bold bg-gradient-to-r from-amber-400 via-yellow-500 to-amber-600 bg-clip-text text-transparent drop-shadow-[0_0_15px_rgba(251,191,36,0.5)]">
            Quillon Graph
          </h1>
          <p className="text-amber-200/70 mt-2 font-medium">Quantum Consensus Wallet</p>
        </div>

        {/* Login Form */}
        <div className="relative rounded-3xl p-8 backdrop-blur-xl" style={{
          background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%)',
          border: '2px solid',
          borderImage: 'linear-gradient(135deg, #D4AF37, #FFD700, #FFA500, #FFD700, #D4AF37) 1',
          boxShadow: '0 0 30px rgba(212, 175, 55, 0.2), inset 0 0 20px rgba(212, 175, 55, 0.1)'
        }}>
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-amber-200 mb-2">
                BIP39 Quantum Seed Phrase
              </label>
              <textarea
                value={seedPhrase}
                onChange={(e) => setSeedPhrase(e.target.value)}
                className="w-full h-24 px-4 py-3 bg-slate-900/70 border-2 border-amber-500/30 rounded-xl text-amber-50 placeholder-slate-400 focus:outline-none focus:border-amber-400 focus:shadow-[0_0_15px_rgba(251,191,36,0.3)] transition-all resize-none"
                placeholder="Enter your 12-word seed phrase..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-amber-200 mb-2">
                Password (Required for wallet encryption)
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3 bg-slate-900/70 border-2 border-amber-500/30 rounded-xl text-amber-50 placeholder-slate-400 focus:outline-none focus:border-amber-400 focus:shadow-[0_0_15px_rgba(251,191,36,0.3)] transition-all"
                placeholder="Enter password for wallet encryption..."
                required
              />
            </div>

            {/* Quantum Generator Button */}
            <motion.button
              onClick={generateQuantumSeed}
              disabled={isGenerating}
              className="w-full py-4 px-6 bg-gradient-to-r from-amber-900/40 to-yellow-900/40 border-2 border-amber-500/40 rounded-xl text-amber-100 font-medium flex items-center justify-center gap-3 hover:border-amber-400 hover:shadow-[0_0_20px_rgba(251,191,36,0.3)] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              whileHover={{ scale: isGenerating ? 1 : 1.02 }}
              whileTap={{ scale: isGenerating ? 1 : 0.98 }}
            >
              {isGenerating ? (
                <>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  >
                    <Sparkles className="w-5 h-5 text-amber-400" />
                  </motion.div>
                  <span>Generating BIP39 Mnemonic...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5 text-amber-400" />
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
                className="h-32 rounded-xl overflow-hidden relative border-2 border-amber-500/30"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <div className="absolute inset-0 bg-gradient-to-r from-amber-500/20 via-yellow-500/20 to-orange-500/20 animate-pulse" />
                <div className="absolute inset-0 bg-slate-900/80 flex items-center justify-center">
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  >
                    <Sparkles className="w-12 h-12 text-amber-400" />
                  </motion.div>
                </div>
              </motion.div>
            )}

            {/* Authenticate Button */}
            <motion.button
              onClick={handleAuthenticate}
              disabled={!seedPhrase || !password || isAuthenticating}
              className="w-full py-5 px-6 bg-gradient-to-r from-amber-600 to-yellow-600 rounded-xl text-slate-900 font-bold text-lg flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-[0_0_30px_rgba(251,191,36,0.5)] hover:from-amber-500 hover:to-yellow-500 transition-all"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              {isAuthenticating ? (
                <>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  >
                    <Key className="w-6 h-6" />
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
              className="mt-6 h-2 rounded-full overflow-hidden border border-amber-500/30"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <motion.div
                className="h-full bg-gradient-to-r from-amber-600 via-yellow-500 to-amber-600"
                initial={{ x: '-100%' }}
                animate={{ x: '100%' }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              />
            </motion.div>
          )}
        </div>

        {/* Security Note */}
        <p className="text-center text-amber-300/60 text-sm mt-6 font-medium">
          🔐 Post-Quantum Cryptography • 🌊 DAG-BFT Consensus • 🧅 Tor Integration
        </p>
      </motion.div>
    </div>
  );
}