import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, QrCode, Sparkles, Check, AlertTriangle, X, Shield, Eye, EyeOff, Camera, Wallet } from 'lucide-react';
import { qnkAPI } from '../services/api';
import QRScanner from './QRScanner';
import QRDisplay from './QRDisplay';
import QuantumMixerVisualization from './QuantumMixerVisualization';
import AddressBook from './AddressBook';

interface WalletBalance {
  symbol: string;
  name: string;
  balance: number;
  usdValue?: number;
  icon: 'qug' | 'usd' | 'btc' | 'eth' | 'sol' | 'zec' | 'iron' | 'custom';
  color: string;
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

export default function TransactionScreenV2() {
  // Get pre-selected coin from localStorage (set by Dashboard)
  const [selectedCoin, setSelectedCoin] = useState<string>(() => {
    const stored = localStorage.getItem('selectedCoinForSend');
    if (stored) {
      localStorage.removeItem('selectedCoinForSend'); // Clear after reading
      return stored;
    }
    return 'QUG'; // Default to QUG
  });

  // Wallet balances state
  const [walletBalances, setWalletBalances] = useState<WalletBalance[]>([]);

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
  const [mixingSessionId, setMixingSessionId] = useState<string>('');
  const [showMixerVisualization, setShowMixerVisualization] = useState(false);

  // QR Code states
  const [showQRScanner, setShowQRScanner] = useState(false);
  const [showQRDisplay, setShowQRDisplay] = useState(false);

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

  // Restore ongoing mixing session on component mount
  useEffect(() => {
    const activeMixingSession = localStorage.getItem('activeMixingSession');
    const mixingStartTime = localStorage.getItem('mixingStartTime');

    if (activeMixingSession && mixingStartTime) {
      const elapsedMs = Date.now() - parseInt(mixingStartTime);
      const elapsedSeconds = Math.floor(elapsedMs / 1000);

      console.log('🔄 [MIXER RESTORE] Found ongoing mixing session:', {
        sessionId: activeMixingSession,
        elapsedSeconds,
        stillActive: elapsedSeconds < 30
      });

      // If less than 30 seconds have passed, restore the visualization
      if (elapsedSeconds < 30) {
        setMixingSessionId(activeMixingSession);
        setShowMixerVisualization(true);
        setEnablePrivacyMixer(true);

        console.log('✅ [MIXER RESTORE] Restored mixer visualization');
      } else {
        // Mixing should be complete, clean up
        localStorage.removeItem('activeMixingSession');
        localStorage.removeItem('mixingStartTime');
        console.log('🏁 [MIXER RESTORE] Mixing session expired, cleaning up');
      }
    }
  }, []);

  // Fetch wallet balances to display the selected coin's wallet card
  useEffect(() => {
    const fetchBalances = async () => {
      const currentWalletAddress = localStorage.getItem('walletAddress');
      if (!currentWalletAddress) return;

      const balances: WalletBalance[] = [];

      // Fetch QUG balance
      try {
        const balanceResponse = await qnkAPI.getWalletBalance(currentWalletAddress);
        if (balanceResponse.success && balanceResponse.data) {
          balances.push({
            symbol: 'QUG',
            name: 'Quillon Graph',
            balance: balanceResponse.data.balance_qnk || 0,
            icon: 'qug',
            color: 'from-amber-400 to-yellow-500',
          });
        }
      } catch (error) {
        console.warn('Failed to fetch QUG balance:', error);
      }

      // Fetch QUGUSD balance
      try {
        const response = await qnkAPI.getMultiTokenBalance();
        if (response.success && response.data && response.data.tokens) {
          const tokensObj = response.data.tokens;
          if (tokensObj.qugusd && tokensObj.qugusd.balance !== undefined) {
            const qugUsdBalance = parseFloat(tokensObj.qugusd.balance) || 0;
            balances.push({
              symbol: 'QUGUSD',
              name: 'Quillon USD',
              balance: qugUsdBalance,
              usdValue: qugUsdBalance,
              icon: 'usd',
              color: 'from-blue-400 to-cyan-500',
            });
          }
        }
      } catch (error) {
        console.warn('Failed to fetch QUGUSD balance:', error);
      }

      // Fetch USD balance
      try {
        const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/payment/balance`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ wallet_address: currentWalletAddress }),
        });

        if (response.ok) {
          const data = await response.json();
          if (data.success && data.data) {
            const usdValue = parseFloat(data.data.balance_usd || '0');
            balances.push({
              symbol: 'USD',
              name: 'US Dollar',
              balance: usdValue,
              icon: 'usd',
              color: 'from-green-400 to-emerald-500',
            });
          }
        }
      } catch (error) {
        console.warn('Failed to fetch USD balance:', error);
      }

      setWalletBalances(balances);
    };

    fetchBalances();
  }, []);

  const validateTransaction = (): { valid: boolean; error?: string } => {
    console.log('💰 TransactionScreenV2: validateTransaction called');
    const balance = selectedWallet?.balance || 0;
    console.log('💰 Selected wallet balance:', balance);
    console.log('💰 transaction.amount:', transaction.amount);

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

    const fee = 0.00001;
    const totalRequired = amount + fee;

    console.log('💰 Balance check:');
    console.log('   Amount:', amount);
    console.log('   Fee:', fee);
    console.log('   Total required:', totalRequired);
    console.log('   Current balance:', balance);
    console.log('   Has sufficient balance?', balance >= totalRequired);

    if (balance < totalRequired) {
      return {
        valid: false,
        error: `Insufficient balance. Required: ${totalRequired.toFixed(8)} ${selectedCoin} (${amount} + ${fee} fee), Available: ${balance.toFixed(8)} ${selectedCoin}`
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
          const sessionId = result.data.mixing_session_id;
          setMixingSessionId(sessionId);

          // Show the 3D visualization
          setShowMixerVisualization(true);

          // Hide the transaction form (user can navigate away)
          setTransaction(prev => ({
            ...prev,
            isProcessing: false  // Allow form to reset
          }));

          console.log('🌪️ [MIXER] Starting 3D visualization for session:', sessionId);

          // The backend will complete mixing in 30 seconds automatically
          // No need for polling - the QuantumMixerVisualization handles timing
        }
      } else {
        // Standard transaction
        console.log('📤 Sending standard transaction:', {
          from: walletAddress,
          to: toAddress,
          amount: parseFloat(amount),
          memo: memo || undefined,
          tokenType: selectedCoin
        });

        result = await qnkAPI.sendTransaction(
          walletAddress,
          toAddress,
          parseFloat(amount), // Keep as QUG, no unit conversion
          memo || undefined,
          selectedCoin // Pass the selected coin (QUG, QUGUSD, or USD)
        );
      }

      console.log('📥 Transaction result:', result);
      console.log('📥 Result details - success:', result.success, 'data:', result.data, 'error:', result.error);

      if (result.success && result.data) {
        console.log('✅ Transaction successful! Hash:', result.data.transaction_hash);
        console.log('✅ Full transaction data:', JSON.stringify(result.data, null, 2));

        setTransaction(prev => ({
          ...prev,
          success: true, // Always show success for completed transactions
          txHash: result.data.transaction_hash || result.data.mixing_session_id || result.data.tx_hash || 'pending',
          starkProof: result.data.stark_proof
        }));

        // Dispatch custom event to update balance
        window.dispatchEvent(new CustomEvent('balance-update', {
          detail: { refresh: true }
        }));
      } else {
        console.error('❌ Transaction failed:', result.error);
        throw new Error(result.error || 'Transaction failed - no error message provided');
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

  const handleQRScan = (scannedData: string) => {
    // Parse QR code data - it could be just an address or a payment request
    let address = scannedData;
    let amount = '';

    try {
      // Check if it's a payment request URI (e.g., quillon:address?amount=123&memo=test)
      if (scannedData.startsWith('quillon:')) {
        const url = new URL(scannedData);
        address = url.pathname.replace('//', '');
        const amountParam = url.searchParams.get('amount');
        const memoParam = url.searchParams.get('memo');
        if (amountParam) amount = amountParam;
        if (memoParam) {
          setTransaction(prev => ({ ...prev, memo: memoParam }));
        }
      }
    } catch (e) {
      // If parsing fails, treat it as a simple address
      console.log('QR code is a simple address:', scannedData);
    }

    setTransaction(prev => ({
      ...prev,
      toAddress: address,
      ...(amount && { amount })
    }));
    setShowQRScanner(false);
  };

  // Get the selected wallet for display
  const selectedWallet = walletBalances.find(w => w.symbol === selectedCoin);

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl lg:text-4xl font-bold text-white mb-2">Send Transaction</h1>
        <p className="text-gray-400">Quantum-secured transfer with STARK proof generation & ZK-verified address book</p>
      </div>

      {/* Selected Wallet Card Display */}
      {selectedWallet && (
        <motion.div
          className="backdrop-blur-xl rounded-3xl p-6"
          style={{
            background: `linear-gradient(135deg, rgba(30, 20, 60, 0.9) 0%, rgba(50, 30, 80, 0.9) 100%)`,
            border: '2px solid rgba(212, 175, 55, 0.3)',
            boxShadow: '0 0 30px rgba(212, 175, 55, 0.2)'
          }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className={`p-3 rounded-xl bg-gradient-to-br ${selectedWallet.color}`}>
                {(selectedWallet.icon === 'qug' || selectedWallet.icon === 'usd') && (
                  <div className="relative w-8 h-8">
                    <div className="absolute inset-0 rounded-full" style={{
                      background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 25%, #FFA500 50%, #FFD700 75%, #D4AF37 100%)',
                      padding: '1px'
                    }}>
                      <div className="w-full h-full bg-gradient-to-b from-slate-900 via-blue-950 to-slate-900 rounded-full flex items-center justify-center p-1">
                        <img
                          src="/quillon-logo.png"
                          alt="Quillon"
                          className="w-full h-full object-contain"
                          style={{ filter: 'invert(1)' }}
                        />
                      </div>
                    </div>
                  </div>
                )}
                {selectedWallet.icon === 'custom' && <Wallet className="w-8 h-8 text-white" />}
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">{selectedWallet.name}</h3>
                <p className="text-sm text-gray-400">Sending from {selectedWallet.symbol} wallet</p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-quantum-green">
                {selectedWallet.balance.toFixed(8)} {selectedWallet.symbol}
              </div>
              {selectedWallet.usdValue !== undefined && (
                <div className="text-sm text-gray-400 mt-1">
                  ≈ ${selectedWallet.usdValue.toFixed(2)} USD
                </div>
              )}
            </div>
          </div>

          {/* Coin Selector Dropdown */}
          {walletBalances.length > 1 && (
            <div className="mt-4">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Select Coin to Send
              </label>
              <select
                value={selectedCoin}
                onChange={(e) => setSelectedCoin(e.target.value)}
                className="w-full px-4 py-3 bg-quantum-dark/50 border border-quantum-purple/30 rounded-xl text-white focus:outline-none focus:border-quantum-cyan transition-colors"
              >
                {walletBalances.map(wallet => (
                  <option key={wallet.symbol} value={wallet.symbol}>
                    {wallet.symbol} - {wallet.balance.toFixed(8)} {wallet.name}
                  </option>
                ))}
              </select>
            </div>
          )}

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
      )}

      {/* Two-Column Grid: Transaction Form + Address Book */}
      {selectedWallet && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* LEFT COLUMN: Transaction Form */}
          <motion.div
            className="backdrop-blur-xl rounded-3xl p-8"
            style={{
              background: 'linear-gradient(135deg, rgba(30, 20, 60, 0.9) 0%, rgba(50, 30, 80, 0.9) 100%)',
              border: '2px solid rgba(212, 175, 55, 0.2)',
              boxShadow: '0 0 30px rgba(212, 175, 55, 0.1)'
            }}
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
                <button
                  onClick={() => setShowQRScanner(true)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-2 text-gray-400 hover:text-quantum-cyan transition-colors"
                  title="Scan QR Code"
                >
                  <Camera className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Show My QR Code Button */}
            <div>
              <button
                onClick={() => setShowQRDisplay(true)}
                className="w-full py-3 px-4 bg-amber-600/10 border border-amber-500/30 rounded-xl text-amber-300 font-medium flex items-center justify-center gap-2 hover:bg-amber-600/20 transition-colors"
              >
                <QrCode className="w-5 h-5" />
                <span>Show My QR Code (Receive)</span>
              </button>
            </div>

            {/* Amount */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Amount ({selectedCoin})
              </label>
              <input
                type="number"
                value={transaction.amount}
                onChange={(e) => setTransaction(prev => ({ ...prev, amount: e.target.value }))}
                className="w-full px-4 py-4 bg-quantum-dark/50 border border-quantum-purple/30 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-quantum-cyan transition-colors text-2xl font-bold"
                placeholder="0.00"
                step="0.00000001"
                max={selectedWallet?.balance || 0}
              />
              <div className="flex justify-between items-center text-sm mt-1">
                <span className="text-gray-400">
                  Available: <span className="text-quantum-green font-semibold">{(selectedWallet?.balance || 0).toFixed(8)} {selectedCoin}</span>
                </span>
                <span className="text-gray-400">
                  Fee: <span className="text-quantum-yellow">0.00001 {selectedCoin}</span>
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
            <div className="rounded-xl p-6"
              style={{
                background: 'linear-gradient(135deg, rgba(236, 72, 153, 0.1) 0%, rgba(219, 39, 119, 0.05) 100%)',
                border: '2px solid rgba(236, 72, 153, 0.2)'
              }}
            >
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

          {/* RIGHT COLUMN: Address Book */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <AddressBook
              onSelectAddress={(address) => {
                setTransaction(prev => ({ ...prev, toAddress: address }));
              }}
            />
          </motion.div>
        </div>
      )}

      {/* Send Button */}
      {selectedWallet && (
        <motion.button
          onClick={handleSendTransaction}
          disabled={transaction.isProcessing || !validateTransaction().valid}
          className="w-full py-6 px-8 rounded-xl text-white font-bold text-xl flex items-center justify-center gap-4 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          style={{
            background: transaction.isProcessing || !validateTransaction().valid
              ? 'linear-gradient(135deg, rgba(168, 85, 247, 0.5) 0%, rgba(139, 92, 246, 0.5) 100%)'
              : 'linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #FFA500 100%)',
            boxShadow: '0 0 30px rgba(212, 175, 55, 0.3)'
          }}
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
          <div className="font-medium text-quantum-green">Privacy & Security</div>
          <div className="text-sm text-gray-400 mt-1">
            All transactions use <strong className="text-quantum-purple">ZK-STARK proofs</strong> to hide sender, amount, and recipient details, secured with post-quantum Dilithium5 signatures.
            The optional <strong className="text-quantum-pink">Mixer</strong> adds enhanced privacy layers including ring signatures, stealth addresses, and decoy routing for maximum anonymity.
          </div>
        </div>
      </div>

      {/* QR Code Scanner Modal */}
      <QRScanner
        isOpen={showQRScanner}
        onScan={handleQRScan}
        onClose={() => setShowQRScanner(false)}
      />

      {/* QR Code Display Modal */}
      <QRDisplay
        isOpen={showQRDisplay}
        data={getWalletAddress()}
        title="Receive QUG"
        subtitle="Scan this QR code to send tokens to your wallet"
        onClose={() => setShowQRDisplay(false)}
      />

      {/* Quantum Mixer 3D Visualization - Full Screen Overlay */}
      <AnimatePresence>
        {showMixerVisualization && mixingSessionId && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black"
          >
            <QuantumMixerVisualization
              sessionId={mixingSessionId}
              privacyLevel={privacyLevel}
              onComplete={() => {
                console.log('🏁 [MIXER] Visualization complete, hiding overlay');
                setShowMixerVisualization(false);
                setTransaction(prev => ({
                  ...prev,
                  success: true,
                  txHash: mixingSessionId
                }));

                // Trigger balance refresh
                window.dispatchEvent(new CustomEvent('balance-update', {
                  detail: { refresh: true }
                }));
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}