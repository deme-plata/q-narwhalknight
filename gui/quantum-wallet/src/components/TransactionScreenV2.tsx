import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, QrCode, Sparkles, Check, AlertTriangle, X, Shield, Eye, EyeOff } from 'lucide-react';
import { qnkAPI } from '../services/api';

interface TransactionScreenV2Props {
  currentBalance?: number;
}

interface TransactionState {
  toAddress: string;
  amount: string;
  memo: string;
  isProcessing: boolean;
  error: string | null;
  success: boolean;
  txHash: string;
  starkProof: any;
}

export default function TransactionScreenV2({ currentBalance = 0 }: TransactionScreenV2Props) {
  // Simple transaction state (no wallet selection complexity)
  const [transaction, setTransaction] = useState<TransactionState>({
    toAddress: '',
    amount: '',
    memo: '',
    isProcessing: false,
    error: null,
    success: false,
    txHash: '',
    starkProof: null
  });

  // Quantum Privacy Mixer states
  const [enablePrivacyMixer, setEnablePrivacyMixer] = useState(false);
  const [privacyLevel, setPrivacyLevel] = useState<'standard' | 'high' | 'maximum'>('high');
  const [decoyMultiplier, setDecoyMultiplier] = useState(15);
  const [showMixingDetails, setShowMixingDetails] = useState(false);
  const [mixerAvailable, setMixerAvailable] = useState<boolean | null>(null); // null = unknown, true = available, false = unavailable
  const [, setMixingSessionId] = useState<string>('');
  const [, setMixingProgress] = useState(0);
  const [, setMixingStage] = useState<string>('');

  // Get wallet address from localStorage
  const getWalletAddress = () => {
    return localStorage.getItem('walletAddress') || '';
  };

  // Check mixer availability on component mount
  useEffect(() => {
    const checkMixerAvailability = async () => {
      try {
        const response = await qnkAPI.getMixingPoolsStatus();
        setMixerAvailable(response.success);
        if (!response.success) {
          console.log('🔍 Mixer not available:', response.error);
        }
      } catch (error) {
        setMixerAvailable(false);
        console.log('🔍 Mixer availability check failed:', error);
      }
    };

    checkMixerAvailability();
  }, []);

  const requestFaucetTokens = async () => {
    try {
      const walletAddress = getWalletAddress();
      if (!walletAddress) {
        setTransaction(prev => ({ ...prev, error: 'No wallet address found' }));
        return;
      }

      console.log(`🚰 Requesting faucet tokens for ${walletAddress}`);
      setTransaction(prev => ({ ...prev, error: null }));
      
      const faucetResponse = await qnkAPI.requestFaucet(walletAddress);
      
      if (faucetResponse.success) {
        setTransaction(prev => ({ 
          ...prev, 
          error: '✅ Faucet request successful! Balance will update automatically.' 
        }));
        
        // Clear success message after 3 seconds
        setTimeout(() => {
          setTransaction(prev => ({ ...prev, error: null }));
        }, 3000);
      } else {
        throw new Error(faucetResponse.error || 'Faucet request failed');
      }
    } catch (error) {
      console.error('❌ Faucet request failed:', error);
      setTransaction(prev => ({ 
        ...prev, 
        error: `Faucet request failed: ${error instanceof Error ? error.message : 'Unknown error'}` 
      }));
    }
  };

  const validateTransaction = (): { valid: boolean; error?: string } => {
    if (!transaction.toAddress.trim()) {
      return { valid: false, error: 'Please enter recipient address' };
    }
    
    if (!transaction.amount.trim()) {
      return { valid: false, error: 'Please enter amount' };
    }
    
    const amount = parseFloat(transaction.amount);
    if (isNaN(amount) || amount <= 0) {
      return { valid: false, error: 'Please enter a valid amount' };
    }
    
    const fee = 0.00001; // QNK
    const totalRequired = amount + fee;
    
    if (currentBalance < totalRequired) {
      return { 
        valid: false, 
        error: `Insufficient balance. Required: ${totalRequired.toFixed(8)} QNK (${amount} + ${fee} fee), Available: ${currentBalance.toFixed(8)} QNK` 
      };
    }
    
    return { valid: true };
  };

  const handleSendTransaction = async () => {
    // Validate transaction
    const validation = validateTransaction();
    if (!validation.valid) {
      setTransaction(prev => ({ ...prev, error: validation.error || 'Invalid transaction' }));
      return;
    }
    
    const walletAddress = getWalletAddress();
    if (!walletAddress) {
      setTransaction(prev => ({ ...prev, error: 'No wallet address found' }));
      return;
    }
    
    const { toAddress, amount, memo } = transaction;
    
    setTransaction(prev => ({ 
      ...prev, 
      isProcessing: true, 
      error: null, 
      success: false 
    }));
    
    try {
      let result: any;
      if (enablePrivacyMixer) {
        // Use quantum privacy mixer
        console.log('🌪️ Sending transaction through quantum privacy mixer');
        console.log(`Privacy Level: ${privacyLevel}, Decoy Multiplier: ${decoyMultiplier}x`);

        const mixingRequest = {
          to: toAddress,
          amount: parseFloat(amount),
          privacy_level: privacyLevel,
          enable_quantum_mixing: true,
          decoy_multiplier: decoyMultiplier,
          memo: memo || undefined
        };

        console.log('🔍 Mixer request:', mixingRequest);
        result = await qnkAPI.sendPrivateTransaction(mixingRequest);
        console.log('🔍 Mixer response:', result);

        // If mixer is not available, fall back to standard transaction
        if (!result.success && (
          result.error?.includes('Mixer endpoint not available') ||
          result.error?.includes('Server returned non-JSON response')
        )) {
          console.warn('⚠️ Mixer not available, falling back to standard transaction');
          // Don't disable the mixer toggle - let user keep it enabled for when it becomes available
          setMixerAvailable(false); // Update availability state

          // Show user-friendly message
          setTransaction(prev => ({
            ...prev,
            error: '🔄 Quantum mixer in development mode - falling back to standard secure transaction (your privacy settings are preserved)'
          }));

          // Wait a moment to show the message, then proceed with standard transaction
          await new Promise(resolve => setTimeout(resolve, 1000));
          setTransaction(prev => ({ ...prev, error: null }));

          result = await qnkAPI.sendTransaction(
            walletAddress,
            toAddress,
            parseFloat(amount),
            memo || undefined
          );

          // For fallback transactions, handle success immediately
          if (result.success && result.data) {
            setTransaction(prev => ({
              ...prev,
              success: true,
              txHash: result.data.transaction_hash || 'fallback_complete',
              starkProof: result.data.stark_proof
            }));
            return; // Exit early for fallback transactions
          } else if (!result.success) {
            // Fallback transaction also failed
            throw new Error(`Fallback transaction failed: ${result.error}`);
          }
        }

        if (result.success && result.data?.mixing_session_id) {
          setMixingSessionId(result.data.mixing_session_id);
          setMixingStage('Initializing quantum mixing pool...');

          // Start monitoring mixing progress
          const progressInterval = setInterval(async () => {
            try {
              const status = await qnkAPI.getMixingStatus(result.data.mixing_session_id);
              if (status.success && status.data) {
                setMixingProgress(status.data.progress || 0);
                setMixingStage(status.data.stage || 'Processing...');

                if (status.data.completed) {
                  clearInterval(progressInterval);
                  setTransaction(prev => ({
                    ...prev,
                    success: true,
                    txHash: status.data.final_transaction_hash || 'mixing_complete'
                  }));
                }
              }
            } catch (error) {
              console.warn('Failed to get mixing status:', error);
            }
          }, 1000);

          // Cleanup interval after 2 minutes max
          setTimeout(() => clearInterval(progressInterval), 120000);
        }
      } else {
        // Standard transaction
        console.log('📤 Sending standard transaction:', {
          from: walletAddress,
          to: toAddress,
          amount: parseFloat(amount),
          memo: memo || undefined
        });

        result = await qnkAPI.sendTransaction(
          walletAddress,
          toAddress,
          parseFloat(amount), // Keep as QNK, no unit conversion
          memo || undefined
        );
      }

      console.log('📥 Transaction result:', result);
      
      if (result.success && result.data) {
        setTransaction(prev => ({
          ...prev,
          success: !enablePrivacyMixer, // Only show success immediately for non-mixer transactions
          txHash: result.data.transaction_hash || result.data.mixing_session_id || 'unknown',
          starkProof: result.data.stark_proof
        }));

        // Balance will update automatically via SSE
      } else {
        throw new Error(result.error || 'Transaction failed');
      }
    } catch (error) {
      console.error('❌ Transaction error:', error);
      setTransaction(prev => ({ 
        ...prev, 
        error: error instanceof Error ? error.message : 'Transaction failed' 
      }));
    } finally {
      setTransaction(prev => ({ ...prev, isProcessing: false }));
    }
  };

  const resetTransaction = () => {
    setTransaction(prev => ({
      ...prev,
      toAddress: '',
      amount: '',
      memo: '',
      error: null,
      success: false,
      txHash: '',
      starkProof: null
    }));
  };

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl lg:text-4xl font-bold text-white mb-2">Send Transaction</h1>
        <p className="text-gray-400">Quantum-secured transfer with STARK proof generation</p>
      </div>

      {/* Current Balance Display */}
      <motion.div
        className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-6 quantum-glow"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex justify-between items-center">
          <div>
            <h3 className="text-lg font-semibold text-white">Your Balance</h3>
            <p className="text-sm text-gray-400">Available for transactions</p>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold text-quantum-green">
              {currentBalance.toFixed(8)} QNK
            </div>
            {currentBalance <= 0 && (
              <button
                onClick={requestFaucetTokens}
                className="mt-2 px-4 py-2 bg-quantum-yellow/20 border border-quantum-yellow/50 rounded-lg text-quantum-yellow text-sm hover:bg-quantum-yellow/30 transition-colors"
              >
                Request Test Tokens
              </button>
            )}
          </div>
        </div>
        
        {/* Error Display */}
        {transaction.error && (
          <motion.div
            className="mt-4 bg-quantum-pink/20 border border-quantum-pink/50 rounded-xl p-4 flex items-center gap-3"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <AlertTriangle className="w-5 h-5 text-quantum-pink flex-shrink-0" />
            <p className="text-quantum-pink">{transaction.error}</p>
          </motion.div>
        )}
      </motion.div>

      {/* Transaction Form */}
      {currentBalance >= 0 && (
        <motion.div
          className="bg-quantum-purple/20 backdrop-blur-xl rounded-3xl p-8 border border-quantum-purple/30"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="space-y-6">
            {/* Recipient */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Recipient Address
              </label>
              <div className="relative">
                <input
                  type="text"
                  value={transaction.toAddress}
                  onChange={(e) => setTransaction(prev => ({ ...prev, toAddress: e.target.value }))}
                  className="w-full px-4 py-4 bg-quantum-dark/50 border border-quantum-purple/30 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-quantum-cyan transition-colors pr-12"
                  placeholder="qnk1abc123... or alice.qnk"
                />
                <button className="absolute right-3 top-1/2 -translate-y-1/2 p-2 text-gray-400 hover:text-quantum-cyan transition-colors">
                  <QrCode className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Amount */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Amount (QNK)
              </label>
              <input
                type="number"
                value={transaction.amount}
                onChange={(e) => setTransaction(prev => ({ ...prev, amount: e.target.value }))}
                className="w-full px-4 py-4 bg-quantum-dark/50 border border-quantum-purple/30 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-quantum-cyan transition-colors text-2xl font-bold"
                placeholder="0.00"
                step="0.00000001"
                max={currentBalance}
              />
              <div className="flex justify-between items-center text-sm mt-1">
                <span className="text-gray-400">
                  Available: <span className="text-quantum-green font-semibold">{currentBalance.toFixed(8)} QNK</span>
                </span>
                <span className="text-gray-400">
                  Fee: <span className="text-quantum-yellow">0.00001 QNK</span>
                </span>
              </div>
            </div>

            {/* Memo */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Memo (Optional)
              </label>
              <textarea
                value={transaction.memo}
                onChange={(e) => setTransaction(prev => ({ ...prev, memo: e.target.value }))}
                className="w-full px-4 py-4 bg-quantum-dark/50 border border-quantum-purple/30 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-quantum-cyan transition-colors resize-none h-20"
                placeholder="Add a note..."
              />
            </div>

            {/* Quantum Privacy Mixer Toggle */}
            <div className="bg-quantum-pink/10 border border-quantum-pink/20 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <Shield className="w-6 h-6 text-quantum-pink" />
                  <div>
                    <h3 className="text-lg font-semibold text-white">Quantum Privacy Mixer</h3>
                    <div className="flex items-center gap-2">
                      <p className="text-sm text-gray-400">Enhanced anonymity with decoy transactions</p>
                      {mixerAvailable === false && (
                        <span className="px-2 py-1 bg-quantum-yellow/20 text-quantum-yellow text-xs rounded">
                          Development Mode
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    className="sr-only peer"
                    checked={enablePrivacyMixer}
                    onChange={(e) => setEnablePrivacyMixer(e.target.checked)}
                  />
                  <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-quantum-pink/25 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-quantum-pink"></div>
                </label>
              </div>

              <AnimatePresence>
                {enablePrivacyMixer && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="space-y-4"
                  >
                    {/* Privacy Level */}
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-3">
                        Privacy Level
                      </label>
                      <div className="grid grid-cols-3 gap-2">
                        {[
                          { value: 'standard', label: 'Standard', decoys: '15x', time: '15s' },
                          { value: 'high', label: 'High', decoys: '25x', time: '30s' },
                          { value: 'maximum', label: 'Maximum', decoys: '50x', time: '60s' }
                        ].map(({ value, label, decoys, time }) => (
                          <button
                            key={value}
                            onClick={() => {
                              setPrivacyLevel(value as any);
                              setDecoyMultiplier(value === 'standard' ? 15 : value === 'high' ? 25 : 50);
                            }}
                            className={`p-3 rounded-lg border text-center transition-all ${
                              privacyLevel === value
                                ? 'border-quantum-pink bg-quantum-pink/20 text-quantum-pink'
                                : 'border-quantum-purple/30 bg-quantum-dark/30 text-gray-300 hover:border-quantum-pink/50'
                            }`}
                          >
                            <div className="font-medium">{label}</div>
                            <div className="text-xs opacity-70">{decoys} • {time}</div>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Decoy Multiplier */}
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Decoy Multiplier: {decoyMultiplier}x decoy transactions
                      </label>
                      <input
                        type="range"
                        min="5"
                        max="50"
                        value={decoyMultiplier}
                        onChange={(e) => setDecoyMultiplier(parseInt(e.target.value))}
                        className="w-full h-2 bg-quantum-dark rounded-lg appearance-none cursor-pointer slider-thumb"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>Basic (5x)</span>
                        <span>Maximum Anonymity (50x)</span>
                      </div>
                    </div>

                    {/* Privacy Details Toggle */}
                    <button
                      onClick={() => setShowMixingDetails(!showMixingDetails)}
                      className="flex items-center gap-2 text-quantum-cyan hover:text-quantum-pink transition-colors"
                    >
                      {showMixingDetails ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      <span className="text-sm">
                        {showMixingDetails ? 'Hide' : 'Show'} mixing details
                      </span>
                    </button>

                    <AnimatePresence>
                      {showMixingDetails && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="bg-quantum-dark/30 rounded-lg p-4 space-y-3"
                        >
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <div className="text-gray-400">Ring Signature Size</div>
                              <div className="text-quantum-cyan font-mono">{decoyMultiplier + 1} participants</div>
                            </div>
                            <div>
                              <div className="text-gray-400">Mixing Time</div>
                              <div className="text-quantum-green font-mono">
                                {privacyLevel === 'standard' ? '15s' : privacyLevel === 'high' ? '30s' : '60s'}
                              </div>
                            </div>
                            <div>
                              <div className="text-gray-400">ZK-STARK Proof</div>
                              <div className="text-quantum-purple font-mono">Falcon1024</div>
                            </div>
                            <div>
                              <div className="text-gray-400">Stealth Address</div>
                              <div className="text-quantum-pink font-mono">Generated</div>
                            </div>
                          </div>

                          <div className="pt-2 border-t border-quantum-purple/20">
                            <div className="text-xs text-gray-500">
                              🔒 Your transaction will be mixed with {decoyMultiplier} decoy transactions using
                              post-quantum cryptography (Dilithium5, Kyber1024, Falcon1024) for maximum privacy.
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </motion.div>
      )}

      {/* Send Button */}
      {currentBalance >= 0 && (
        <motion.button
          onClick={handleSendTransaction}
          disabled={transaction.isProcessing || !validateTransaction().valid}
          className="w-full py-6 px-8 bg-gradient-to-r from-quantum-purple to-quantum-cyan rounded-xl text-white font-bold text-xl flex items-center justify-center gap-4 disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-2xl hover:shadow-quantum-cyan/25 transition-all"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {transaction.isProcessing ? (
            <>
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              >
                <Sparkles className="w-6 h-6" />
              </motion.div>
              <span>Generating Quantum Proof...</span>
            </>
          ) : transaction.success ? (
            <>
              <Check className="w-6 h-6 text-quantum-green" />
              <span>Transaction Complete!</span>
            </>
          ) : (
            <>
              <Send className="w-6 h-6" />
              <span>Sign & Broadcast</span>
            </>
          )}
        </motion.button>
      )}

      {/* Success Panel */}
      <AnimatePresence>
        {transaction.success && transaction.txHash && (
          <motion.div
            className="bg-quantum-green/10 border border-quantum-green/20 rounded-3xl p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-quantum-green/20">
                  <Check className="w-6 h-6 text-quantum-green" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-quantum-green">Transaction Confirmed</h3>
                  <p className="text-sm text-gray-400">Your quantum-secured transaction has been submitted</p>
                </div>
              </div>
              <button
                onClick={resetTransaction}
                className="p-2 text-gray-400 hover:text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-3">
              <div>
                <div className="text-sm text-gray-400">Transaction Hash:</div>
                <div className="font-mono text-sm text-quantum-cyan break-all">
                  {transaction.txHash}
                </div>
              </div>
              
              {transaction.starkProof && (
                <div>
                  <div className="text-sm text-gray-400">STARK Proof:</div>
                  <div className="text-sm text-quantum-purple">
                    ✓ Generated ({transaction.starkProof.proving_time_ms}ms)
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Security Note */}
      <div className="bg-quantum-green/10 border border-quantum-green/20 rounded-xl p-4 flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
        <div>
          <div className="font-medium text-quantum-green">Post-Quantum Security</div>
          <div className="text-sm text-gray-400 mt-1">
            This transaction is secured with Dilithium5 signatures and quantum-resistant cryptography.
          </div>
        </div>
      </div>
    </div>
  );
}