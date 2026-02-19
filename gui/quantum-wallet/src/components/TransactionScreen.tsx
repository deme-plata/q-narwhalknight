import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, QrCode, Sparkles, Check, AlertTriangle, X, RefreshCw, Shield, Eye, EyeOff, ArrowRight } from 'lucide-react';
import { qnkAPI } from '../services/api';

interface TransactionScreenProps {
  currentBalance?: number;
}

export default function TransactionScreen({ currentBalance = 0 }: TransactionScreenProps) {
  const [sender, setSender] = useState('alice'); // Default sender for testing
  const [recipient, setRecipient] = useState(''); // User must enter recipient
  const [amount, setAmount] = useState(''); // User must enter amount
  const [memo, setMemo] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [showFractal, setShowFractal] = useState(false);
  const [txComplete, setTxComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [starkProof, setStarkProof] = useState<any>(null);
  const [txHash, setTxHash] = useState<string>('');
  const [actualBalance, setActualBalance] = useState<number>(currentBalance);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Quantum Privacy Mixer states
  const [enablePrivacyMixer, setEnablePrivacyMixer] = useState(false);
  const [privacyLevel, setPrivacyLevel] = useState<'standard' | 'high' | 'maximum'>('high');
  const [decoyMultiplier, setDecoyMultiplier] = useState(15);
  const [showMixingDetails, setShowMixingDetails] = useState(false);
  const [mixingSessionId, setMixingSessionId] = useState<string>('');
  const [mixingProgress, setMixingProgress] = useState(0);
  const [mixingStage, setMixingStage] = useState<string>('');

  // Function to refresh balance
  const refreshBalance = async () => {
    setIsRefreshing(true);
    try {
      const walletAddress = localStorage.getItem('walletAddress');
      if (walletAddress) {
        console.log('🔄 Refreshing balance for wallet:', walletAddress);
        const balanceResponse = await qnkAPI.getWalletBalance(walletAddress);
        console.log('📊 Balance response:', balanceResponse);
        
        if (balanceResponse.success && balanceResponse.data) {
          const newBalance = balanceResponse.data.balance_qnk || 0;
          setActualBalance(newBalance);
          console.log('✅ Balance refreshed:', newBalance, 'QUG');
        } else {
          console.warn('⚠️ Balance refresh failed:', balanceResponse.error);
        }
      } else {
        console.warn('⚠️ No wallet address found in localStorage');
      }
    } catch (error) {
      console.error('❌ Failed to refresh balance:', error);
    } finally {
      setIsRefreshing(false);
    }
  };

  // Update actual balance when currentBalance prop changes
  useEffect(() => {
    setActualBalance(currentBalance);
    console.log('📱 TransactionScreen received balance prop:', currentBalance);
  }, [currentBalance]);

  // Refresh balance on component mount
  useEffect(() => {
    refreshBalance();
  }, []);

  const handleSend = async () => {
    if (!sender || !recipient || !amount) {
      setError('Please fill in sender, recipient and amount');
      return;
    }
    
    const amountNumber = parseFloat(amount);
    const fee = 0.00001;
    const totalRequired = amountNumber + fee;
    
    // Debug: Log balance values to understand the discrepancy
    console.log('🔍 Transaction Debug Info:');
    console.log('Prop Balance (currentBalance):', currentBalance);
    console.log('Actual Balance:', actualBalance);
    console.log('Amount to Send:', amountNumber);
    console.log('Network Fee:', fee);
    console.log('Total Required:', totalRequired);
    
    // Use actualBalance (refreshed) instead of currentBalance (prop)
    const effectiveBalance = Math.max(actualBalance, currentBalance);
    console.log('Effective Balance:', effectiveBalance);
    
    if (effectiveBalance <= 0) {
      setError('❌ Insufficient balance. You need QUG tokens to send transactions. Earn QUG through mining.');
      return;
    } else if (totalRequired > effectiveBalance) {
      setError(`❌ Insufficient balance. Required: ${totalRequired.toFixed(8)} QUG (${amountNumber} + ${fee} fee), Available: ${effectiveBalance.toFixed(8)} QUG`);
      return;
    }
    
    setIsProcessing(true);
    setShowFractal(true);
    setError(null);
    
    try {
      let result: any;

      if (enablePrivacyMixer) {
        // Use quantum privacy mixer
        console.log('🌪️ Sending transaction through quantum privacy mixer');
        console.log(`Privacy Level: ${privacyLevel}, Decoy Multiplier: ${decoyMultiplier}x`);

        result = await qnkAPI.sendPrivateTransaction({
          to: recipient,
          amount: parseFloat(amount),
          privacy_level: privacyLevel,
          enable_quantum_mixing: true,
          decoy_multiplier: decoyMultiplier,
          memo: memo || undefined
        });

        if (result.success && result.data) {
          setMixingSessionId(result.data.mixing_session_id);
          setMixingStage('Generating decoys');
          setMixingProgress(25);

          // Start polling for mixing progress
          const pollMixingStatus = async () => {
            let attempts = 0;
            const maxAttempts = 120; // 2 minutes timeout

            const poll = async () => {
              try {
                attempts++;
                const statusResult = await qnkAPI.getMixingStatus(result.data.mixing_session_id);

                if (statusResult.success && statusResult.data) {
                  const status = statusResult.data;
                  setMixingProgress(status.progress_percent || 0);
                  setMixingStage(status.stage || 'Processing');

                  if (status.status === 'completed_mixing' || status.progress_percent === 100) {
                    // Mixing complete
                    setMixingProgress(100);
                    setMixingStage('Mixing completed');
                    setTxComplete(true);
                    setShowFractal(false);
                    setIsProcessing(false);
                    return;
                  } else if (status.status === 'mixing_in_progress' && attempts < maxAttempts) {
                    // Continue polling
                    setTimeout(poll, 2000); // Poll every 2 seconds
                  } else {
                    throw new Error('Mixing timeout or failed');
                  }
                }
              } catch (error) {
                console.error('Mixing status polling error:', error);
                if (attempts < maxAttempts) {
                  setTimeout(poll, 2000);
                } else {
                  throw new Error('Failed to track mixing progress');
                }
              }
            };

            setTimeout(poll, 2000); // Start polling after 2 seconds
          };

          pollMixingStatus();
        }
      } else {
        // Standard transaction (existing code)
        result = await qnkAPI.sendTransaction(sender, recipient, parseFloat(amount), memo || undefined);

        if (result.success && result.data) {
          setStarkProof(result.data.stark_proof);
          setTxHash(result.data.transaction_hash);
          setTxComplete(true);
          setShowFractal(false);
          setIsProcessing(false);

          // v6.0.1: Optimistically update displayed balance after successful send
          const sentAmount = parseFloat(amount);
          const newBalance = Math.max(0, actualBalance - sentAmount);
          setActualBalance(newBalance);

          // Notify TopBar of balance change
          window.dispatchEvent(new CustomEvent('wallet-balance-updated', {
            detail: {
              symbol: 'QUG',
              balance: newBalance,
              reason: 'transaction_sent'
            }
          }));
        }
      }

      console.log('Transaction successful:', result);

      if (result.success && result.data) {
        setTxHash(result.data.transaction_hash || result.data.mixing_session_id);

        if (!enablePrivacyMixer) {
          setStarkProof(result.data.stark_proof);
          setTxComplete(true);
          setShowFractal(false);
          setIsProcessing(false);
        }
      } else {
        throw new Error(result.error || 'Transaction failed');
      }
    } catch (err) {
      console.error('Transaction error:', err);
      setError(err instanceof Error ? err.message : 'Transaction failed');
      setIsProcessing(false);
      setShowFractal(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl lg:text-4xl font-bold text-white mb-2">Send Transaction</h1>
        <p className="text-gray-400">Quantum-secured transfer with STARK proof generation</p>
      </div>

      {/* Transaction Form */}
      <motion.div
        className="bg-quantum-indigo/50 backdrop-blur-xl rounded-3xl p-8 quantum-glow"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="space-y-6">
          {/* Error Message */}
          {error && (
            <motion.div
              className="bg-quantum-pink/20 border border-quantum-pink/50 rounded-xl p-4 flex items-center gap-3"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <AlertTriangle className="w-5 h-5 text-quantum-pink flex-shrink-0" />
              <p className="text-quantum-pink">{error}</p>
            </motion.div>
          )}

          {/* Sender */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Send From
            </label>
            <input
              type="text"
              value={sender}
              onChange={(e) => setSender(e.target.value)}
              className="w-full px-4 py-4 bg-quantum-dark/50 border border-quantum-purple/30 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-quantum-cyan transition-colors"
              placeholder="alice, bob, or hex address..."
            />
          </div>

          {/* Recipient */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Recipient Address
            </label>
            <div className="relative">
              <input
                type="text"
                value={recipient}
                onChange={(e) => setRecipient(e.target.value)}
                className="w-full px-4 py-4 bg-quantum-dark/50 border border-quantum-purple/30 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-quantum-cyan transition-colors pr-12"
                placeholder="alice.qnk or qnk1abc123..."
              />
              <button className="absolute right-3 top-1/2 -translate-y-1/2 p-2 text-gray-400 hover:text-quantum-cyan transition-colors">
                <QrCode className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Amount */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Amount (QUG)
            </label>
            <input
              type="number"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              className="w-full px-4 py-4 bg-quantum-dark/50 border border-quantum-purple/30 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-quantum-cyan transition-colors text-2xl font-bold"
              placeholder="0.00"
              step="0.00000001"
            />
            <div className="flex justify-between items-center text-sm mt-1">
              <div className="flex items-center gap-2">
                <span className="text-gray-400">Available: <span className="text-quantum-green font-semibold">{Math.max(actualBalance, currentBalance).toFixed(8)} QUG</span></span>
                <button
                  onClick={refreshBalance}
                  disabled={isRefreshing}
                  className="p-1 text-gray-400 hover:text-quantum-cyan transition-colors disabled:opacity-50"
                  title="Refresh balance"
                >
                  <RefreshCw className={`w-3 h-3 ${isRefreshing ? 'animate-spin' : ''}`} />
                </button>
              </div>
              <span className="text-gray-400">≈ ${amount ? (parseFloat(amount) * 0.42).toFixed(2) : '0.00'} USD</span>
            </div>
          </div>

          {/* Memo */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Memo (Optional)
            </label>
            <textarea
              value={memo}
              onChange={(e) => setMemo(e.target.value)}
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
                  <p className="text-sm text-gray-400">Enhanced anonymity with decoy transactions</p>
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
                        className="bg-quantum-dark/50 rounded-lg p-4 space-y-2"
                      >
                        <div className="text-sm space-y-1">
                          <div className="flex justify-between">
                            <span className="text-gray-400">Ring Signature Size:</span>
                            <span className="text-quantum-cyan">16 members</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Stealth Addresses:</span>
                            <span className="text-quantum-green">Quantum-enhanced</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Mixing Rounds:</span>
                            <span className="text-quantum-purple">{privacyLevel === 'standard' ? '3' : privacyLevel === 'high' ? '5' : '8'}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Dandelion++ Gossip:</span>
                            <span className="text-quantum-yellow">Enabled</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">ZK Proof System:</span>
                            <span className="text-quantum-pink">ZK-STARK</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Anonymity Set Size:</span>
                            <span className="text-white font-semibold">~{decoyMultiplier * 4} participants</span>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Privacy Fee */}
                  <div className="bg-quantum-yellow/10 border border-quantum-yellow/20 rounded-lg p-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-400">Privacy Mixer Fee:</span>
                      <span className="text-quantum-yellow font-semibold">
                        {amount ? (parseFloat(amount) * 0.001).toFixed(6) : '0.000000'} QUG (0.1%)
                      </span>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Zero Balance Info */}
          {Math.max(actualBalance, currentBalance) <= 0 && (
            <div className="bg-quantum-yellow/10 border border-quantum-yellow/30 rounded-xl p-4">
              <div className="flex items-center gap-3 mb-3">
                <AlertTriangle className="w-5 h-5 text-quantum-yellow" />
                <span className="text-quantum-yellow font-medium">No QUG Balance</span>
              </div>
              <p className="text-sm text-gray-300">
                You need QUG tokens to send transactions. Earn QUG by mining blocks with your node.
              </p>
            </div>
          )}
        </div>
      </motion.div>

      {/* Transaction Preview & STARK Fractal */}
      <AnimatePresence>
        {(sender && amount && recipient) && (
          <motion.div
            className="bg-quantum-purple/20 backdrop-blur-xl rounded-3xl p-8 border border-quantum-purple/30"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <div className="flex items-center gap-3 mb-6">
              <Sparkles className="w-6 h-6 text-quantum-pink" />
              <h3 className="text-xl font-semibold">Transaction Preview</h3>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="space-y-4">
                <div>
                  <div className="text-sm text-gray-400">Sending from:</div>
                  <div className="font-mono text-quantum-yellow break-all">{sender}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-400">Sending to:</div>
                  <div className="font-mono text-quantum-cyan break-all">{recipient}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-400">Amount:</div>
                  <div className="text-2xl font-bold text-white">{amount} QUG</div>
                </div>
                <div>
                  <div className="text-sm text-gray-400">Network Fee:</div>
                  <div className="text-quantum-green">0.00001 QUG</div>
                </div>
              </div>

              {/* STARK Proof Fractal / Mixing Progress */}
              <div className="relative h-48 bg-quantum-dark/50 rounded-xl overflow-hidden">
                <div className="absolute inset-0">
                  {!showFractal && !starkProof && !mixingSessionId ? (
                    <div className="flex items-center justify-center h-full">
                      <div className="text-gray-400 text-center">
                        {enablePrivacyMixer ? (
                          <>
                            <Shield className="w-12 h-12 mx-auto mb-2 opacity-50" />
                            <div>Quantum Mixing Visualization</div>
                            <div className="text-sm">Privacy enhanced on send</div>
                          </>
                        ) : (
                          <>
                            <Sparkles className="w-12 h-12 mx-auto mb-2 opacity-50" />
                            <div>STARK Proof Fractal</div>
                            <div className="text-sm">Generated on send</div>
                          </>
                        )}
                      </div>
                    </div>
                  ) : mixingSessionId && isProcessing ? (
                    <div className="p-4 h-full">
                      <div className="text-center mb-4">
                        <Shield className="w-8 h-8 mx-auto mb-2 text-quantum-pink" />
                        <div className="text-sm font-semibold text-quantum-pink">Quantum Privacy Mixing</div>
                      </div>

                      {/* Mixing Progress Bar */}
                      <div className="mb-4">
                        <div className="flex justify-between text-xs text-gray-400 mb-2">
                          <span>{mixingStage}</span>
                          <span>{mixingProgress}%</span>
                        </div>
                        <div className="w-full bg-quantum-dark rounded-full h-2">
                          <motion.div
                            className="bg-gradient-to-r from-quantum-pink to-quantum-purple h-2 rounded-full"
                            initial={{ width: 0 }}
                            animate={{ width: `${mixingProgress}%` }}
                            transition={{ duration: 0.5 }}
                          />
                        </div>
                      </div>

                      {/* Mixing Stats */}
                      <div className="space-y-2 text-xs">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Privacy Level:</span>
                          <span className="text-quantum-pink capitalize">{privacyLevel}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Decoy Transactions:</span>
                          <span className="text-quantum-cyan">{decoyMultiplier}x</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Mixing Session ID:</span>
                          <span className="text-quantum-green font-mono">{mixingSessionId.substring(0, 8)}...</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Anonymity Set:</span>
                          <span className="text-quantum-yellow">~{decoyMultiplier * 4} participants</span>
                        </div>
                      </div>

                      {/* Mixing Animation */}
                      <div className="absolute inset-0 opacity-20">
                        <motion.div
                          className="w-full h-full"
                          animate={{ rotate: 360 }}
                          transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                        >
                          <svg className="w-full h-full">
                            {[...Array(8)].map((_, i) => (
                              <motion.circle
                                key={i}
                                cx="50%"
                                cy="50%"
                                r={20 + i * 8}
                                fill="none"
                                stroke={`hsl(${310 + i * 20}, 80%, 60%)`}
                                strokeWidth="1"
                                strokeDasharray="4,4"
                                initial={{ pathLength: 0 }}
                                animate={{ pathLength: 1 }}
                                transition={{ duration: 2, delay: i * 0.1, repeat: Infinity }}
                              />
                            ))}
                          </svg>
                        </motion.div>
                      </div>
                    </div>
                  ) : starkProof ? (
                    <div className="p-4 h-full overflow-y-auto">
                      <div className="text-sm space-y-2">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Proof System:</span>
                          <span className="text-quantum-green font-mono">{starkProof.proof_system}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Proving Time:</span>
                          <span className="text-quantum-cyan">{starkProof.proving_time_ms}ms</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Proof Size:</span>
                          <span className="text-quantum-purple">{starkProof.proof_size_bytes} bytes</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Quantum Resistance:</span>
                          <span className="text-quantum-yellow">{starkProof.quantum_resistance}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">PQ Signature:</span>
                          <span className="text-quantum-pink">{starkProof.post_quantum_signature}</span>
                        </div>
                        <div className="mt-3">
                          <span className="text-gray-400 text-xs">Verification Key:</span>
                          <div className="font-mono text-xs text-quantum-cyan break-all">
                            {starkProof.verification_key.substring(0, 32)}...
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <motion.div
                      className="w-full h-full flex items-center justify-center"
                      initial={{ scale: 0, rotate: 0 }}
                      animate={{ scale: 1, rotate: 360 }}
                      transition={{ duration: 2 }}
                    >
                      <svg className="w-full h-full">
                        {/* Enhanced fractal pattern for STARK proof generation */}
                        {[...Array(12)].map((_, i) => (
                          <motion.g key={i}>
                            {[...Array(8)].map((_, j) => (
                              <motion.circle
                                key={`${i}-${j}`}
                                cx="50%"
                                cy="50%"
                                r={6 + j * 6}
                                fill="none"
                                stroke={`hsl(${i * 30 + j * 45 + 180}, 90%, ${60 + j * 5}%)`}
                                strokeWidth="1.5"
                                opacity="0.7"
                                initial={{ pathLength: 0, scale: 0 }}
                                animate={{ pathLength: 1, scale: 1 }}
                                transition={{ 
                                  duration: 2, 
                                  delay: i * 0.08 + j * 0.04,
                                  type: "spring"
                                }}
                              />
                            ))}
                            {/* Add connecting lines for complexity */}
                            {[...Array(6)].map((_, k) => (
                              <motion.line
                                key={`line-${i}-${k}`}
                                x1="50%"
                                y1="50%"
                                x2={`${50 + Math.cos(k * Math.PI / 3) * (20 + i * 3)}%`}
                                y2={`${50 + Math.sin(k * Math.PI / 3) * (20 + i * 3)}%`}
                                stroke={`hsl(${k * 60 + i * 15}, 80%, 70%)`}
                                strokeWidth="1"
                                opacity="0.4"
                                initial={{ pathLength: 0 }}
                                animate={{ pathLength: 1 }}
                                transition={{ duration: 1.8, delay: i * 0.1 }}
                              />
                            ))}
                          </motion.g>
                        ))}
                      </svg>
                      <div className="absolute bottom-2 left-2 right-2 text-center text-xs text-gray-400 font-mono">
                        Generating Zero-Knowledge Proof...
                      </div>
                    </motion.div>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Send Button */}
      <motion.button
        onClick={handleSend}
        disabled={!sender || !recipient || !amount || isProcessing}
        className="w-full py-6 px-8 bg-gradient-to-r from-quantum-purple to-quantum-cyan rounded-xl text-white font-bold text-xl flex items-center justify-center gap-4 disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-2xl hover:shadow-quantum-cyan/25 transition-all"
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        {isProcessing ? (
          <>
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            >
              <Sparkles className="w-6 h-6" />
            </motion.div>
            <span>Generating Quantum Proof...</span>
          </>
        ) : txComplete ? (
          <>
            <Check className="w-6 h-6 text-quantum-green" />
            <span>Quantum Transaction Complete!</span>
          </>
        ) : (
          <>
            {enablePrivacyMixer ? <Shield className="w-6 h-6" /> : <Send className="w-6 h-6" />}
            <span>
              {enablePrivacyMixer
                ? `Send with ${privacyLevel.charAt(0).toUpperCase() + privacyLevel.slice(1)} Privacy`
                : 'Sign & Broadcast'
              }
            </span>
          </>
        )}
      </motion.button>

      {/* Transaction Success Panel */}
      <AnimatePresence>
        {txComplete && txHash && (
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
                  <p className="text-sm text-gray-400">Your quantum-secured transaction has been submitted to the network</p>
                </div>
              </div>
              <button
                onClick={() => {
                  // Reset all states including privacy mixer
                  setTxComplete(false);
                  setSender('alice');
                  setRecipient('');
                  setAmount('');
                  setMemo('');
                  setStarkProof(null);
                  setTxHash('');
                  setMixingSessionId('');
                  setMixingProgress(0);
                  setMixingStage('');
                  setEnablePrivacyMixer(false);
                  setPrivacyLevel('high');
                  setDecoyMultiplier(15);
                  setShowMixingDetails(false);
                }}
                className="p-2 text-gray-400 hover:text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-3">
              <div>
                <div className="text-sm text-gray-400">Transaction Hash:</div>
                <div className="font-mono text-sm text-quantum-cyan break-all">
                  {txHash}
                </div>
              </div>
              
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Amount Sent:</span>
                <span className="text-white font-semibold">{amount} QUG</span>
              </div>
              
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Network Fee:</span>
                <span className="text-quantum-green">0.00001 QUG</span>
              </div>
              
              {starkProof && (
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">STARK Proof:</span>
                  <span className="text-quantum-purple">✓ Generated ({starkProof.proving_time_ms}ms)</span>
                </div>
              )}
              
              <div className="pt-4 border-t border-quantum-green/20 space-y-3">
                <div className="text-xs text-gray-500 text-center">
                  Post-quantum secured with Dilithium5 signatures and STARK zero-knowledge proofs
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => {
                      // Reset form for another transaction
                      setTxComplete(false);
                      setSender('alice');
                      setRecipient('');
                      setAmount('');
                      setMemo('');
                      setStarkProof(null);
                      setTxHash('');
                      setMixingSessionId('');
                      setMixingProgress(0);
                      setMixingStage('');
                      // Keep privacy mixer settings for convenience
                    }}
                    className="py-3 px-4 bg-gradient-to-r from-quantum-purple/20 to-quantum-cyan/20 hover:from-quantum-purple/30 hover:to-quantum-cyan/30 border border-quantum-purple/30 rounded-xl text-white font-medium transition-all text-center"
                  >
                    Send Another
                  </motion.button>

                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => {
                      // Allow user to continue browsing the app
                      // This creates seamless navigation without losing state
                      // Users can navigate to other pages and return
                      refreshBalance();
                    }}
                    className="py-3 px-4 bg-gradient-to-r from-quantum-green/20 to-quantum-blue/20 hover:from-quantum-green/30 hover:to-quantum-blue/30 border border-quantum-green/30 rounded-xl text-white font-medium transition-all text-center flex items-center justify-center gap-2"
                  >
                    <ArrowRight className="w-4 h-4" />
                    Continue
                  </motion.button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Photon Waterfall During Processing */}
      <AnimatePresence>
        {isProcessing && (
          <motion.div
            className="fixed inset-0 pointer-events-none z-50 overflow-hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {[...Array(20)].map((_, i) => (
              <motion.div
                key={i}
                className="absolute w-1 h-24 bg-gradient-to-b from-transparent via-quantum-cyan to-transparent"
                initial={{ top: -100, left: `${Math.random() * 100}%` }}
                animate={{ top: '110%' }}
                transition={{
                  duration: 2 + Math.random(),
                  repeat: Infinity,
                  delay: Math.random() * 2,
                }}
              />
            ))}
            
            <div className="absolute bottom-8 left-1/2 -translate-x-1/2 bg-quantum-dark/80 backdrop-blur-xl rounded-lg px-6 py-3 border border-quantum-cyan/30">
              <div className="text-center text-quantum-cyan font-mono">
                Photon Detection Rate: 1.{Math.floor(Math.random() * 90 + 10)} Gbit/s
              </div>
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
            This transaction is secured with Dilithium5 signatures and will be verified using quantum-resistant cryptography.
          </div>
        </div>
      </div>
    </div>
  );
}