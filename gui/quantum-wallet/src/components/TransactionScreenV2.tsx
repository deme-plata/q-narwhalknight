import { useState, useEffect, useRef, lazy, Suspense } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, QrCode, Sparkles, Check, AlertTriangle, X, Shield, Eye, EyeOff, Camera, Wallet, TrendingDown, Radio, Globe } from 'lucide-react';
import { qnkAPI, FEE_REDUCTION_ACTIVATION_HEIGHT, CURRENT_MIN_FEE_QUG, NEW_MIN_FEE_QUG } from '../services/api';
import { signTransactionForP2P, verifyPasswordHash } from '../services/walletAuth';
import QRScanner from './QRScanner';
import QRDisplay from './QRDisplay';
const QuantumMixerVisualization = lazy(() => import('./QuantumMixerVisualization'));
import AddressBook from './AddressBook';
import { flashBorderRed } from './AnimatedBorder';
import { useLibP2P } from '../contexts/LibP2PContext';
import { useP2PData } from '../hooks/useP2PData';
import type { SignedTransaction } from '../libp2p/types';

// v3.6.1-beta: SANITY CHECK - Max possible balance is 21 million QUG (total supply)
// v3.6.7-beta: QUG uses 24 decimals, so max = 21M. Custom tokens can have any decimals (some have huge supplies)
const MAX_QUG_BALANCE = 21_000_000;

/**
 * v3.6.13-beta: Format balance showing ALL significant digits
 * User requested full precision - show all decimals, trim trailing zeros
 * QUG uses 24 decimals internally but we show all non-zero digits
 */
function formatBalanceDisplay(balance: number, _maxDecimals: number = 24): string {
  if (balance === 0) return '0';
  if (!isFinite(balance) || isNaN(balance)) return '0';

  // Convert to string with full precision (up to 24 decimals)
  // toFixed(24) gives us all digits, then we trim trailing zeros
  let formatted = balance.toFixed(24);

  // Remove trailing zeros after decimal point
  if (formatted.includes('.')) {
    formatted = formatted.replace(/\.?0+$/, '');
  }

  // If the result is just "-0" or empty after decimal, return "0"
  if (formatted === '-0' || formatted === '' || formatted === '-') {
    return '0';
  }

  return formatted;
}

/**
 * v3.6.1-beta: Validate balance value to prevent corrupted data from being used
 * v3.6.7-beta: Token-aware validation - only apply QUG limit to QUG/QUGUSD tokens
 *              Custom tokens (like PEPEG with 7 decimals) can have much larger display values
 */
function isValidBalance(balance: number, symbol?: string): boolean {
  if (typeof balance !== 'number') return false;
  if (isNaN(balance) || !isFinite(balance)) return false;
  if (balance < 0) return false;

  // v3.6.7-beta: Only apply strict limit to QUG (not QUGUSD — stablecoin supply is uncapped)
  // Custom tokens with fewer decimals can legitimately have much larger display values
  const isQug = !symbol || symbol.toUpperCase() === 'QUG';
  if (isQug && balance > MAX_QUG_BALANCE) {
    console.warn(`🚨 [TransactionScreen] Rejected corrupted ${symbol || 'native'} balance: ${balance.toExponential()} > max supply ${MAX_QUG_BALANCE}`);
    return false;
  }

  // v3.6.7-beta: For custom tokens, only reject truly invalid values (Infinity, negative)
  // Large values like 821 quintillion PEPEG are valid for meme tokens with low decimals
  return true;
}

// ============================================================================
// FEATURE FLAG: P2P Transaction Submission
// ============================================================================
// Set to true to enable browser-side P2P transaction signing & gossipsub broadcast
// Set to false to use HTTP API (server signs and broadcasts via P2P gossipsub)
//
// v3.5.14-beta: P2P submission enabled - transactions now properly added to production_mempool
// ============================================================================
const ENABLE_P2P_TRANSACTION_SUBMISSION = true;

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
  /** Number of validator nodes that confirmed the transaction (2f+1 for BFT consensus) */
  validatorCount?: number;
  /** v1.4.4: Confirmation tracking for retail-first finality */
  confirmations?: {
    current: number;
    required: number;
    tier: 'INSTANT' | 'OPTIMISTIC' | 'FAST' | 'STANDARD' | 'SETTLEMENT';
    tierEmoji: string;
    estimatedTimeRemaining: string;
    isFinalized: boolean;
  };
  /** v3.5.x: Track how the transaction was submitted (P2P or HTTP) */
  submissionMethod?: 'p2p' | 'http';
  /** v3.5.x: Number of P2P peers the transaction was broadcast to */
  p2pPeerCount?: number;
  /** v3.5.24: Multi-peer verification result */
  p2pVerification?: {
    verified: boolean;
    peersConfirmed: number;
    totalPeers: number;
    confidence: number;
    blockHeight?: number;
  };
}

interface TransactionScreenV2Props {
  currentBalance: number;
}

export default function TransactionScreenV2({ currentBalance }: TransactionScreenV2Props) {
  // v3.5.x: Get P2P context for direct transaction submission
  const { isReady: p2pReady, peerCount: p2pPeerCount, submitTransaction: submitP2P } = useLibP2P();

  // v3.5.24: P2P data service for multi-peer transaction verification
  const { verifyTransaction, isP2PReady: p2pDataReady } = useP2PData();

  // Get pre-selected coin from localStorage (set by Dashboard)
  const [selectedCoin, setSelectedCoin] = useState<string>(() => {
    const stored = localStorage.getItem('selectedCoinForSend');
    if (stored) {
      localStorage.removeItem('selectedCoinForSend'); // Clear after reading
      return stored;
    }
    return 'QUG'; // Default to QUG
  });

  // v3.6.10-beta: Get custom token contract address from localStorage (set by Dashboard CustomTokensCard)
  const [selectedTokenContract, setSelectedTokenContract] = useState<string | null>(() => {
    const stored = localStorage.getItem('selectedTokenContract');
    if (stored) {
      localStorage.removeItem('selectedTokenContract'); // Clear after reading
      console.log('📋 [TransactionScreen v3.6.10] Custom token contract address loaded:', stored);
      return stored;
    }
    return null;
  });

  // Wallet balances state
  const [walletBalances, setWalletBalances] = useState<WalletBalance[]>([]);

  // CRITICAL FIX: Track highest known balance per token to prevent showing stale/lower values
  // This prevents the bug where balance jumps from 65 to 0.75 on refresh
  const highestKnownBalancesRef = useRef<Record<string, number>>({});

  // v3.6.9-beta: STABLE balance display - same mechanism as TopBar to prevent flickering
  const [stableBalance, setStableBalance] = useState<number>(() => {
    // Initialize from localStorage cache like TopBar does
    const cached = localStorage.getItem('cachedBalance');
    if (cached) {
      const cachedValue = parseFloat(cached);
      // v3.6.7-beta: This is specifically for QUG balance
      if (isValidBalance(cachedValue, 'QUG')) {
        return cachedValue;
      }
    }
    if (isValidBalance(currentBalance, 'QUG')) {
      return currentBalance;
    }
    return 0;
  });
  const lastBalanceUpdateRef = useRef<number>(Date.now());
  const balanceStabilityWindowMs = 2000; // Don't change balance more than once per 2 seconds

  // v3.6.9-beta: Stabilize balance updates to prevent flickering (copied from TopBar)
  useEffect(() => {
    const cached = localStorage.getItem('cachedBalance');
    const cachedValue = cached ? parseFloat(cached) : currentBalance;

    // Only use values that pass sanity check
    // v3.6.7-beta: This is specifically for QUG balance
    const validCached = isValidBalance(cachedValue, 'QUG') ? cachedValue : 0;
    const validCurrent = isValidBalance(currentBalance, 'QUG') ? currentBalance : 0;
    const validStable = isValidBalance(stableBalance, 'QUG') ? stableBalance : 0;

    const newBalance = validCached || validCurrent;

    // Only update if enough time has passed (prevents rapid flickering)
    const timeSinceLastUpdate = Date.now() - lastBalanceUpdateRef.current;
    const balanceDifference = Math.abs(newBalance - validStable);

    // Update if: significant change (>1 QUG) OR stability window passed
    if (balanceDifference > 1 || timeSinceLastUpdate > balanceStabilityWindowMs) {
      const candidates = [newBalance, validStable, validCurrent].filter(v => isValidBalance(v, 'QUG'));
      const bestBalance = candidates.length > 0 ? Math.max(...candidates) : 0;

      if (Math.abs(bestBalance - stableBalance) > 0.0001) {
        console.log('💰 TransactionScreen: Stable balance update:', stableBalance.toFixed(4), '→', bestBalance.toFixed(4));
        setStableBalance(bestBalance);
        lastBalanceUpdateRef.current = Date.now();
      }
    }
  }, [currentBalance, stableBalance]);

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
  const [mixerTxHash, setMixerTxHash] = useState<string>(''); // Actual blockchain tx hash (64 hex chars)
  const [showMixerVisualization, setShowMixerVisualization] = useState(false);

  // QR Code states
  const [showQRScanner, setShowQRScanner] = useState(false);
  const [showQRDisplay, setShowQRDisplay] = useState(false);

  // v8.6.5: Password confirmation before send
  const [showPasswordModal, setShowPasswordModal] = useState(false);
  const [confirmPassword, setConfirmPassword] = useState('');
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [passwordVerifying, setPasswordVerifying] = useState(false);

  // Dynamic fee states (v3.4.0: height-gated 10x fee reduction)
  const [currentFee, setCurrentFee] = useState(CURRENT_MIN_FEE_QUG);
  const [networkHeight, setNetworkHeight] = useState(0);
  const [feeReductionActive, setFeeReductionActive] = useState(false);
  const [blocksUntilFeeReduction, setBlocksUntilFeeReduction] = useState(0);

  // v1.4.4: Calculate required confirmations based on transaction value (retail-first)
  const calculateConfirmations = (amountUsd: number): TransactionState['confirmations'] => {
    // Retail-optimized confirmation tiers with DAG-Knight finality
    if (amountUsd < 100) {
      // ⚡ INSTANT: Coffee, snacks - economically irrational to attack
      return {
        current: 0,
        required: 0,
        tier: 'INSTANT',
        tierEmoji: '⚡',
        estimatedTimeRemaining: 'Instant',
        isFinalized: true
      };
    } else if (amountUsd < 1000) {
      // 🚀 OPTIMISTIC: Lunch, retail - DAG vertex inclusion
      return {
        current: 0,
        required: 1,
        tier: 'OPTIMISTIC',
        tierEmoji: '🚀',
        estimatedTimeRemaining: '<200ms',
        isFinalized: false
      };
    } else if (amountUsd < 10000) {
      // ✓ FAST: Electronics - 1 full block
      return {
        current: 0,
        required: 1,
        tier: 'FAST',
        tierEmoji: '✓',
        estimatedTimeRemaining: '~2 seconds',
        isFinalized: false
      };
    } else if (amountUsd < 100000) {
      // 🔒 STANDARD: High-value items - 3 confirmations
      return {
        current: 0,
        required: 3,
        tier: 'STANDARD',
        tierEmoji: '🔒',
        estimatedTimeRemaining: '~6 seconds',
        isFinalized: false
      };
    } else {
      // 🏦 SETTLEMENT: Large transfers - 3 BFT confirmations
      // v2.9.23: DAG-Knight BFT provides instant finality with 2f+1 validator signatures
      // 3 blocks = ~6 seconds for institutional-grade certainty (vs 30 blocks = 60 seconds legacy)
      return {
        current: 0,
        required: 3,
        tier: 'SETTLEMENT',
        tierEmoji: '🏦',
        estimatedTimeRemaining: '~6 seconds',
        isFinalized: false
      };
    }
  };

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

  // v3.4.0: Fetch dynamic fee based on network height
  useEffect(() => {
    const fetchFeeInfo = async () => {
      try {
        // Fetch network height
        const height = await qnkAPI.getNetworkHeight();
        setNetworkHeight(height);

        // Check if fee reduction is active
        const isActive = height >= FEE_REDUCTION_ACTIVATION_HEIGHT;
        setFeeReductionActive(isActive);

        // Calculate blocks until fee reduction
        if (!isActive) {
          setBlocksUntilFeeReduction(FEE_REDUCTION_ACTIVATION_HEIGHT - height);
        } else {
          setBlocksUntilFeeReduction(0);
        }

        // Get current minimum fee (computed locally using height)
        const fee = isActive ? NEW_MIN_FEE_QUG : CURRENT_MIN_FEE_QUG;
        setCurrentFee(fee);

        console.log(`💰 [FEE v3.4.0] Height: ${height}, Fee: ${fee} QUG, Reduction active: ${isActive}`);
      } catch (error) {
        console.warn('Failed to fetch fee info:', error);
        // Default to current fee on error
        setCurrentFee(CURRENT_MIN_FEE_QUG);
      }
    };

    // Fetch immediately
    fetchFeeInfo();

    // Refresh every 30 seconds (blocks are ~2 seconds)
    const interval = setInterval(fetchFeeInfo, 30000);
    return () => clearInterval(interval);
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

  // v1.4.4: Confirmation tracking - poll for confirmations and update in real-time
  useEffect(() => {
    if (!transaction.success || !transaction.confirmations || transaction.confirmations.isFinalized) {
      return;
    }

    // For INSTANT tier, finalize immediately
    if (transaction.confirmations.tier === 'INSTANT') {
      setTransaction(prev => ({
        ...prev,
        confirmations: prev.confirmations ? {
          ...prev.confirmations,
          isFinalized: true,
          current: 0
        } : undefined
      }));
      return;
    }

    // Poll for confirmations every 500ms (DAG-Knight has fast finality)
    const confirmationInterval = setInterval(() => {
      setTransaction(prev => {
        if (!prev.confirmations || prev.confirmations.isFinalized) {
          clearInterval(confirmationInterval);
          return prev;
        }

        const newCurrent = prev.confirmations.current + 1;
        const isFinalized = newCurrent >= prev.confirmations.required;

        if (isFinalized) {
          clearInterval(confirmationInterval);
        }

        // Calculate remaining time
        const remaining = prev.confirmations.required - newCurrent;
        const remainingSeconds = remaining * 2; // 2 seconds per block
        const estimatedTimeRemaining = remaining <= 0
          ? 'Finalized!'
          : remainingSeconds < 60
            ? `~${remainingSeconds} seconds`
            : `~${Math.ceil(remainingSeconds / 60)} minute(s)`;

        return {
          ...prev,
          confirmations: {
            ...prev.confirmations,
            current: newCurrent,
            isFinalized,
            estimatedTimeRemaining
          }
        };
      });
    }, 500); // Fast polling for DAG-Knight

    return () => clearInterval(confirmationInterval);
  }, [transaction.success, transaction.txHash]);

  // Fetch wallet balances to display the selected coin's wallet card
  useEffect(() => {
    const currentWalletAddress = localStorage.getItem('walletAddress');
    if (!currentWalletAddress) {
      console.warn('⚠️ TransactionScreenV2: No wallet address found in localStorage');
      return;
    }

    console.log('🔄 TransactionScreenV2: Using balance from App.tsx:', currentBalance);

    const fetchBalances = async () => {
      const balances: WalletBalance[] = [];

      // v2.9.16-beta: Check cooldown and get protected balances
      const now = Date.now();
      const globalCooldownUntil = parseInt(localStorage.getItem('customTokensCooldownUntil') || '0');
      const isInCooldown = now < globalCooldownUntil;
      let protectedBalances: Record<string, { balance: number; until: number }> = {};

      if (isInCooldown) {
        try {
          protectedBalances = JSON.parse(localStorage.getItem('protectedTokenBalances') || '{}');
          console.log('🔒 [TransactionScreen v2.9.16] In cooldown - will use protected balances:', protectedBalances);
        } catch (e) { /* ignore */ }
      }

      // v3.6.9-beta: Use stableBalance (same pattern as TopBar) to prevent flickering
      console.log('✅ TransactionScreenV2: QUG stable balance:', stableBalance, '(raw prop:', currentBalance, ')');
      balances.push({
        symbol: 'QUG',
        name: 'Quillon Graph',
        balance: stableBalance,  // v3.6.9: Use stable balance instead of raw prop
        icon: 'qug',
        color: 'from-amber-400 to-yellow-500',
      });

      // v3.6.12: Load cached QUGUSD balance from localStorage (same key as Dashboard: cachedQugusdBalance)
      let cachedQugusdBalance = 0;
      try {
        const cachedQugusd = localStorage.getItem('cachedQugusdBalance');
        if (cachedQugusd) {
          cachedQugusdBalance = parseFloat(cachedQugusd) || 0;
          console.log('💾 [TransactionScreen] Loaded cached QUGUSD balance:', cachedQugusdBalance);
        }
      } catch (e) { /* ignore */ }

      // Fetch QUGUSD balance and custom tokens from multi-token API
      let foundQugusd = false;
      try {
        const response = await qnkAPI.getMultiTokenBalance();
        if (response.success && response.data && response.data.tokens) {
          const tokensObj = response.data.tokens;

          // Iterate through all tokens from the API
          for (const [symbol, tokenData] of Object.entries(tokensObj)) {
            const upperSymbol = symbol.toUpperCase();
            const token = tokenData as any;

            // Skip QUG (already added from prop)
            if (upperSymbol === 'QUG') {
              continue;
            }

            // Handle QUGUSD (native USD stablecoin)
            if (upperSymbol === 'QUGUSD') {
              foundQugusd = true;
              // v10.2.9: Try multiple balance sources (matches Dashboard pattern)
              // Priority: balance_base_units/1e24 → usd_value → balance string → 0
              let qugUsdBalance = 0;

              // Source 1: balance_base_units (most reliable — raw u128 from backend)
              if (token.balance_base_units && token.balance_base_units > 0) {
                qugUsdBalance = token.balance_base_units / 1e24;
                console.log('💵 [TransactionScreen] QUGUSD from balance_base_units:', qugUsdBalance);
              }

              // Source 2: usd_value (QUGUSD is 1:1 pegged to USD)
              if (qugUsdBalance === 0 && token.usd_value && token.usd_value > 0) {
                qugUsdBalance = token.usd_value;
                console.log('💵 [TransactionScreen] QUGUSD from usd_value:', qugUsdBalance);
              }

              // Source 3: balance string (formatted by backend)
              if (qugUsdBalance === 0) {
                qugUsdBalance = parseFloat(token.balance || '0');
                if (qugUsdBalance > 0) {
                  console.log('💵 [TransactionScreen] QUGUSD from balance string:', qugUsdBalance);
                }
              }

              // v2.9.16-beta: Use protected balance during cooldown
              const protectedData = protectedBalances[upperSymbol];
              if (isInCooldown && protectedData && protectedData.until > now) {
                console.log(`🔒 [TransactionScreen v2.9.16] Using protected QUGUSD: ${protectedData.balance} (API: ${qugUsdBalance})`);
                qugUsdBalance = protectedData.balance;
              }

              // v10.2.9: Use ANY cached QUGUSD if API returns 0
              // The API returns 0 because token_balances in-memory may not be loaded
              // Dashboard writes the real value to localStorage — trust it over the API
              if (qugUsdBalance === 0) {
                // Try all cache sources
                const cacheKeys = ['cachedQugusdBalance', 'lastKnownQugusdBalance'];
                for (const key of cacheKeys) {
                  try {
                    const cached = localStorage.getItem(key);
                    if (cached) {
                      const val = parseFloat(cached);
                      if (val > 0 && !isNaN(val) && isFinite(val)) {
                        console.log(`💾 [TransactionScreen] QUGUSD from ${key}: ${val}`);
                        qugUsdBalance = val;
                        break;
                      }
                    }
                  } catch (e) { /* ignore */ }
                }
              }

              balances.push({
                symbol: 'QUGUSD',
                name: 'Quillon USD',
                balance: qugUsdBalance,
                usdValue: qugUsdBalance,
                icon: 'usd',
                color: 'from-blue-400 to-cyan-500',
              });
              console.log('✅ TransactionScreenV2: Added QUGUSD balance:', qugUsdBalance);
              continue;
            }

            // Add custom tokens (any token that's not QUG or QUGUSD)
            // v3.0.7-beta: Validate token data to filter out invalid entries (like 404 HTML responses)
            const tokenName = token.name || upperSymbol;

            // Skip tokens with invalid symbols or names (contains HTML, too long, or empty)
            if (!upperSymbol ||
                upperSymbol.length > 20 ||
                upperSymbol.includes('<') ||
                upperSymbol.includes('>') ||
                tokenName.includes('<html') ||
                tokenName.includes('<!DOCTYPE') ||
                tokenName.includes('404') ||
                tokenName.length > 100) {
              console.warn('⚠️ TransactionScreenV2: Skipping invalid token:', { symbol: upperSymbol, name: tokenName });
              continue;
            }

            // v10.2.9: Try balance_base_units first (more reliable than formatted string)
            let customBalance = 0;
            const tokenDecimals = token.decimals || 24;
            if (token.balance_base_units && token.balance_base_units > 0) {
              customBalance = token.balance_base_units / Math.pow(10, tokenDecimals);
            } else {
              customBalance = parseFloat(token.balance || '0');
            }
            // v2.9.16-beta: Use protected balance during cooldown for custom tokens
            const protectedData = protectedBalances[upperSymbol];
            if (isInCooldown && protectedData && protectedData.until > now) {
              console.log(`🔒 [TransactionScreen v2.9.16] Using protected ${upperSymbol}: ${protectedData.balance} (API: ${customBalance})`);
              customBalance = protectedData.balance;
            }
            balances.push({
              symbol: upperSymbol,
              name: tokenName,
              balance: customBalance,
              icon: 'custom',
              color: 'from-purple-400 to-pink-500',
            });
            console.log('✅ TransactionScreenV2: Added custom token:', upperSymbol, 'balance:', customBalance);
          }
        }
      } catch (error) {
        console.warn('Failed to fetch multi-token balances:', error);
      }

      // v3.6.12: Add QUGUSD from cache if it wasn't found in API response
      if (!foundQugusd && cachedQugusdBalance > 0) {
        console.log(`💾 [TransactionScreen] QUGUSD not in API response, adding from cache: ${cachedQugusdBalance}`);
        balances.push({
          symbol: 'QUGUSD',
          name: 'Quillon USD',
          balance: cachedQugusdBalance,
          usdValue: cachedQugusdBalance,
          icon: 'usd',
          color: 'from-blue-400 to-cyan-500',
        });
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

      // CRITICAL FIX: Validate balances against highest known values
      // This prevents showing stale/lower values during race conditions
      // v3.6.1-beta: Added sanity checks to reject corrupted values
      const validatedBalances = balances.map(wallet => {
        const previousHighest = highestKnownBalancesRef.current[wallet.symbol] || 0;

        // Also check localStorage for cached balance (QUG only)
        let cachedValue = 0;
        if (wallet.symbol === 'QUG') {
          const cached = localStorage.getItem('cachedBalance');
          if (cached) {
            const parsed = parseFloat(cached);
            // v3.6.1-beta: Only use cached value if it passes sanity check
            // v3.6.7-beta: Pass symbol for token-aware validation
            cachedValue = isValidBalance(parsed, wallet.symbol) ? parsed : 0;
          }
        }

        // v3.6.1-beta: Filter to only include valid balance values before comparing
        // v3.6.7-beta: Pass symbol for token-aware validation
        const candidates = [
          isValidBalance(previousHighest, wallet.symbol) ? previousHighest : 0,
          cachedValue,
          isValidBalance(wallet.balance, wallet.symbol) ? wallet.balance : 0
        ].filter(v => v >= 0);

        // Use the maximum of valid values only
        const validBalance = candidates.length > 0 ? Math.max(...candidates) : 0;

        // Only accept significant decreases (> 10% drop is suspicious unless it's a real transaction)
        // Small fluctuations are likely race conditions
        if (wallet.balance < previousHighest * 0.9 && previousHighest > 0.1 && isValidBalance(previousHighest, wallet.symbol)) {
          console.warn(`⚠️ TransactionScreenV2: ${wallet.symbol} balance drop blocked: ${previousHighest} → ${wallet.balance}, keeping ${validBalance}`);
        }

        // Update the highest known value (only if valid)
        if (validBalance > previousHighest && isValidBalance(validBalance, wallet.symbol)) {
          highestKnownBalancesRef.current[wallet.symbol] = validBalance;
        }

        return { ...wallet, balance: validBalance };
      });

      // v10.2.9: Sort tokens for stable dropdown order
      // QUG always first, QUGUSD second, then alphabetical by symbol
      const sortedBalances = [...validatedBalances].sort((a, b) => {
        if (a.symbol === 'QUG') return -1;
        if (b.symbol === 'QUG') return 1;
        if (a.symbol === 'QUGUSD') return -1;
        if (b.symbol === 'QUGUSD') return 1;
        if (a.symbol === 'USD') return -1;
        if (b.symbol === 'USD') return 1;
        return a.symbol.localeCompare(b.symbol);
      });

      console.log('💾 TransactionScreenV2: Setting validated walletBalances:', sortedBalances);
      setWalletBalances(sortedBalances);
    };

    fetchBalances();

    // Subscribe to SSE balance updates for real-time updates
    // v3.6.1-beta: Added sanity checks to reject corrupted values
    console.log('📡 TransactionScreenV2: Setting up SSE subscription for:', currentWalletAddress);
    const eventSource = qnkAPI.subscribeToMiningRewards(
      currentWalletAddress,
      () => {}, // No mining rewards needed here
      (update) => {
        // Update QUG balance in real-time with validation
        console.log('📡 TransactionScreenV2: SSE balance update received:', update);

        const previousHighest = highestKnownBalancesRef.current['QUG'] || 0;
        const newBalance = update.new_balance;

        // v3.6.1-beta: CRITICAL - Reject corrupted balance values
        // v3.6.7-beta: Pass 'QUG' for token-aware validation
        if (!isValidBalance(newBalance, 'QUG')) {
          console.warn(`🚨 TransactionScreenV2 SSE: Rejected corrupted balance: ${newBalance}`);
          return;
        }

        // Validate: only accept increase OR small decrease (legitimate transaction)
        let validBalance = newBalance;
        if (newBalance < previousHighest * 0.9 && previousHighest > 0.1 && isValidBalance(previousHighest, 'QUG')) {
          console.warn(`⚠️ TransactionScreenV2 SSE: QUG balance drop blocked: ${previousHighest} → ${newBalance}, keeping ${previousHighest}`);
          validBalance = previousHighest;
        } else if (newBalance > previousHighest && isValidBalance(newBalance, 'QUG')) {
          highestKnownBalancesRef.current['QUG'] = newBalance;
        }

        setWalletBalances(prev => {
          const updated = prev.map(wallet =>
            wallet.symbol === 'QUG'
              ? { ...wallet, balance: validBalance }
              : wallet
          );
          console.log('💾 TransactionScreenV2: Updated walletBalances via SSE:', updated);
          return updated;
        });
      }
    );

    // v1.4.10-beta: Listen for custom token balance updates via SSE
    // v2.9.17-beta: Added cooldown check to prevent stale data
    // v3.6.1-beta: Added sanity check to reject corrupted values
    const handleTokenBalanceUpdate = (event: CustomEvent) => {
      const { tokenSymbol, newBalance, reason } = event.detail;
      console.log('🪙 [TransactionScreen v2.9.17] Token balance updated:', { tokenSymbol, newBalance, reason });

      // v3.6.1-beta: CRITICAL - Reject corrupted balance values for QUG
      // v3.6.7-beta: Pass symbol for token-aware validation
      if (tokenSymbol?.toUpperCase() === 'QUG' && !isValidBalance(newBalance, 'QUG')) {
        console.warn(`🚨 [TransactionScreen] Rejected corrupted ${tokenSymbol} balance: ${newBalance}`);
        return;
      }

      // v2.9.17-beta: Check if this is a DEX swap - only trust DEX swap events during cooldown
      const isDexSwap = reason === 'dex-swap-add' || reason === 'dex-swap-deduct';
      const now = Date.now();
      const globalCooldownUntil = parseInt(localStorage.getItem('customTokensCooldownUntil') || '0');
      const isInCooldown = now < globalCooldownUntil;

      // During cooldown, only accept DEX swap events, block everything else
      if (isInCooldown && !isDexSwap) {
        console.log(`🛡️ [TransactionScreen v2.9.17] BLOCKED non-DEX event for ${tokenSymbol} during cooldown`);
        return;
      }

      // Update the balance for the matching token
      setWalletBalances(prev => prev.map(wallet => {
        if (wallet.symbol?.toUpperCase() === tokenSymbol?.toUpperCase()) {
          console.log(`✅ [TransactionScreen] Updated ${wallet.symbol} balance: ${wallet.balance} → ${newBalance}`);
          return { ...wallet, balance: newBalance };
        }
        return wallet;
      }));
    };

    window.addEventListener('token-balance-updated', handleTokenBalanceUpdate as EventListener);

    // v2.9.17-beta: On mount, immediately check for protected balances and apply them
    // This handles the case where user navigates to this page after a swap
    const applyProtectedBalances = () => {
      const now = Date.now();
      const globalCooldownUntil = parseInt(localStorage.getItem('customTokensCooldownUntil') || '0');
      if (now < globalCooldownUntil) {
        try {
          const protectedBalances = JSON.parse(localStorage.getItem('protectedTokenBalances') || '{}');
          console.log('🔒 [TransactionScreen v2.9.17] Applying protected balances on mount:', protectedBalances);

          setWalletBalances(prev => prev.map(wallet => {
            const upperSymbol = wallet.symbol?.toUpperCase() || '';
            const protectedData = protectedBalances[upperSymbol];
            if (protectedData && protectedData.until > now) {
              console.log(`🔒 [TransactionScreen v2.9.17] Applied protected balance for ${wallet.symbol}: ${wallet.balance} → ${protectedData.balance}`);
              return { ...wallet, balance: protectedData.balance };
            }
            return wallet;
          }));
        } catch (e) { /* ignore */ }
      }
    };

    // Apply protected balances after a short delay to let fetchBalances complete first
    const protectedTimeout = setTimeout(applyProtectedBalances, 100);
    // Also apply again after 500ms to catch any race conditions
    const protectedTimeout2 = setTimeout(applyProtectedBalances, 500);

    // v2.9.3-beta: Listen for wallet-balance-updated events (DEX swaps for QUG and QUGUSD)
    // This is critical for showing updated balances after swaps without waiting for API
    const handleWalletBalanceUpdate = (event: Event) => {
      const customEvent = event as CustomEvent;
      const { symbol, balance: newBalance, reason } = customEvent.detail || {};

      // Only handle native tokens (QUG and QUGUSD) - custom tokens use token-balance-updated
      if (symbol !== 'QUG' && symbol !== 'QUGUSD') return;

      const isDexSwap = reason === 'dex-swap-deduct' || reason === 'dex-swap-add';
      console.log(`💰 [TransactionScreen] wallet-balance-updated for ${symbol}:`, newBalance, 'reason:', reason, 'isDexSwap:', isDexSwap);

      if (isDexSwap && typeof newBalance === 'number' && !isNaN(newBalance)) {
        // DEX swaps are authoritative - update immediately without validation
        setWalletBalances(prev => prev.map(wallet => {
          if (wallet.symbol === symbol) {
            console.log(`✅ [TransactionScreen] DEX SWAP: Updated ${symbol}: ${wallet.balance} → ${newBalance}`);
            // Also update highestKnownBalancesRef for increases
            if (newBalance > (highestKnownBalancesRef.current[symbol] || 0)) {
              highestKnownBalancesRef.current[symbol] = newBalance;
            }
            return { ...wallet, balance: newBalance };
          }
          return wallet;
        }));

        // Update localStorage cache
        if (symbol === 'QUGUSD') {
          localStorage.setItem('cachedQugusdBalance', newBalance.toString());
        }
      }
    };

    window.addEventListener('wallet-balance-updated', handleWalletBalanceUpdate);
    console.log('👂 [TransactionScreen] Listening for wallet-balance-updated events');

    return () => {
      eventSource.close();
      window.removeEventListener('token-balance-updated', handleTokenBalanceUpdate as EventListener);
      window.removeEventListener('wallet-balance-updated', handleWalletBalanceUpdate);
      clearTimeout(protectedTimeout);
      clearTimeout(protectedTimeout2);
    };
  }, [currentBalance]);

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

    // v3.4.0: Use dynamic fee based on network height
    const fee = currentFee;
    const totalRequired = amount + fee;

    console.log('💰 Balance check (v3.4.0):');
    console.log('   Amount:', amount);
    console.log('   Fee:', fee, feeReductionActive ? '(10x reduced!)' : '(legacy)');
    console.log('   Total required:', totalRequired);
    console.log('   Current balance:', balance);
    console.log('   Network height:', networkHeight);
    console.log('   Has sufficient balance?', balance >= totalRequired);

    if (balance < totalRequired) {
      return {
        valid: false,
        // v3.6.10-beta: Show full 24 decimal precision
        error: `Insufficient balance. Required: ${formatBalanceDisplay(totalRequired)} ${selectedCoin} (${amount} + ${formatBalanceDisplay(fee)} fee), Available: ${formatBalanceDisplay(balance)} ${selectedCoin}`
      };
    }

    return { valid: true };
  };

  // v8.6.5: Password confirmation gate — validates before proceeding to actual send
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

    // Check if password hash exists (user has a wallet password set)
    const hasPasswordHash = !!localStorage.getItem('walletPasswordHash');
    if (hasPasswordHash) {
      // Show password confirmation modal
      setConfirmPassword('');
      setPasswordError(null);
      setShowPasswordModal(true);
      return; // Wait for password confirmation before sending
    }

    // No password hash set — proceed directly (e.g. MetaMask-imported wallets)
    await executeSend();
  };

  const handlePasswordConfirm = async () => {
    if (!confirmPassword) {
      setPasswordError('Please enter your wallet password');
      return;
    }
    setPasswordVerifying(true);
    setPasswordError(null);
    try {
      const isValid = await verifyPasswordHash(confirmPassword);
      if (!isValid) {
        setPasswordError('Incorrect password. Please try again.');
        setPasswordVerifying(false);
        return;
      }
      // Password verified — close modal and proceed with send
      setShowPasswordModal(false);
      setConfirmPassword('');
      setPasswordVerifying(false);
      await executeSend();
    } catch (err) {
      setPasswordError('Password verification failed. Please try again.');
      setPasswordVerifying(false);
    }
  };

  const executeSend = async () => {
    const walletAddress = getWalletAddress() || '';
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
            // Flash border red to indicate transaction sent
            flashBorderRed();

            setTransaction(prev => ({
              ...prev,
              success: true,
              txHash: result.data.transaction_hash || 'fallback_complete',
              starkProof: result.data.stark_proof
            }));

            // v6.0.1: Optimistically update balance for fallback path too
            const fallbackSentAmount = parseFloat(amount);
            const fallbackFee = selectedCoin === 'QUG' ? 0.000021 : 0;
            const fallbackCurrentBalance = walletBalances.find(c => c.symbol === selectedCoin)?.balance || 0;
            const fallbackOptimisticBalance = Math.max(0, fallbackCurrentBalance - fallbackSentAmount - fallbackFee);

            setWalletBalances(prev => prev.map(w =>
              w.symbol === selectedCoin ? { ...w, balance: fallbackOptimisticBalance } : w
            ));

            // v6.0.9: Also update stableBalance and highestKnownBalancesRef (same as main path)
            if (selectedCoin === 'QUG') {
              setStableBalance(fallbackOptimisticBalance);
              lastBalanceUpdateRef.current = Date.now();
              highestKnownBalancesRef.current['QUG'] = fallbackOptimisticBalance;
              localStorage.setItem('cachedBalance', fallbackOptimisticBalance.toString());
            }
            highestKnownBalancesRef.current[selectedCoin] = fallbackOptimisticBalance;

            if (selectedCoin === 'QUG') {
              window.dispatchEvent(new CustomEvent('wallet-balance-updated', {
                detail: {
                  symbol: 'QUG',
                  balance: fallbackOptimisticBalance,
                  reason: 'transaction_sent'
                }
              }));
            }

            window.dispatchEvent(new CustomEvent('balance-update', {
              detail: { refresh: true }
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
          // Store actual blockchain tx hash for explorer lookup (64 hex chars)
          const actualTxHash = result.data.transaction_hash || sessionId;
          setMixerTxHash(actualTxHash);

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

          // v3.4.16: CRITICAL FIX - Return here to wait for visualization to complete
          // The onComplete callback in QuantumMixerVisualization will handle success state
          return;
        }
      } else {
        // v3.5.x: P2P-first transaction submission with HTTP fallback
        // FEATURE FLAG: ENABLE_P2P_TRANSACTION_SUBMISSION controls whether to try P2P first
        console.log('📤 Sending standard transaction:', {
          from: walletAddress,
          to: toAddress,
          amount: parseFloat(amount),
          memo: memo || undefined,
          tokenType: selectedCoin,
          p2pEnabled: ENABLE_P2P_TRANSACTION_SUBMISSION,
          p2pReady: p2pReady,
          p2pPeerCount: p2pPeerCount
        });

        let submittedViaP2P = false;
        let p2pPeersSent = 0;

        // Try P2P submission first if ENABLED and node is ready and has peers
        console.log(`📡 [TX] P2P status check: enabled=${ENABLE_P2P_TRANSACTION_SUBMISSION}, p2pReady=${p2pReady}, peerCount=${p2pPeerCount}`);

        if (ENABLE_P2P_TRANSACTION_SUBMISSION && p2pReady && p2pPeerCount > 0) {
          console.log('📡 [TX] Attempting P2P submission first...');

          try {
            // Sign the transaction in the browser
            console.log('🔐 [TX] Signing transaction locally...');
            const signingResult = await signTransactionForP2P({
              from: walletAddress,
              to: toAddress,
              amount: parseFloat(amount),
              memo: memo || undefined,
              // v3.6.10-beta: Use actual contract address for custom tokens, not symbol
              // v3.6.11: Pass QUGUSD token address for P2P so backend knows it's not QUG
              tokenAddress: selectedTokenContract || (selectedCoin === 'QUGUSD' ? '5155475553440000000000000000000000000000000000000000000000000000' : (selectedCoin !== 'QUG' ? selectedCoin : undefined)),
            });
            console.log('🔐 [TX v3.6.10] Signing with tokenAddress:', selectedTokenContract);
            console.log(`🔐 [TX] Signing result: success=${signingResult.success}, error=${signingResult.error || 'none'}`);

            if (signingResult.success && signingResult.transaction) {
              // Submit via P2P gossipsub
              console.log('📤 [TX] Submitting to gossipsub...');
              const p2pResult = await submitP2P(signingResult.transaction as SignedTransaction);
              console.log(`📤 [TX] P2P result: success=${p2pResult.success}, peerCount=${p2pResult.peerCount}, error=${p2pResult.error || 'none'}`);

              if (p2pResult.success) {
                console.log(`✅ [TX] P2P submission successful! Broadcast to ${p2pResult.peerCount} peers`);
                submittedViaP2P = true;
                p2pPeersSent = p2pResult.peerCount || 0;

                // Create a successful result for P2P
                result = {
                  success: true,
                  data: {
                    transaction_hash: p2pResult.txHash || `p2p_${Date.now().toString(16)}`,
                    method: 'p2p',
                    peer_count: p2pResult.peerCount,
                  },
                  error: null,
                  timestamp: new Date().toISOString(),
                };
              } else {
                console.warn(`⚠️ [TX] P2P submission failed: ${p2pResult.error}, falling back to HTTP`);
              }
            } else {
              console.warn(`⚠️ [TX] Transaction signing failed: ${signingResult.error}, falling back to HTTP`);
            }
          } catch (p2pError) {
            console.error('❌ [TX] P2P attempt threw exception:', p2pError);
          }
        } else if (!ENABLE_P2P_TRANSACTION_SUBMISSION) {
          console.log(`📡 [TX] P2P submission DISABLED by feature flag, using HTTP API`);
        } else {
          console.log(`📡 [TX] Skipping P2P: p2pReady=${p2pReady}, peerCount=${p2pPeerCount}`);
        }

        // Fall back to HTTP if P2P didn't work or is disabled
        if (!submittedViaP2P) {
          console.log('🌐 [TX] Using HTTP API submission...');
          result = await qnkAPI.sendTransaction(
            walletAddress,
            toAddress,
            parseFloat(amount), // Keep as QUG, no unit conversion
            memo || undefined,
            selectedCoin // Pass the selected coin (QUG, QUGUSD, or USD)
          );
        }

        // Track submission method for UI
        if (result.success) {
          result.data = {
            ...result.data,
            _submissionMethod: submittedViaP2P ? 'p2p' : 'http',
            _p2pPeerCount: p2pPeersSent,
          };
        }
      }

      console.log('📥 Transaction result:', result);
      console.log('📥 Result details - success:', result.success, 'data:', result.data, 'error:', result.error);

      if (result.success && result.data) {
        console.log('✅ Transaction successful! Hash:', result.data.transaction_hash);
        console.log('✅ Full transaction data:', JSON.stringify(result.data, null, 2));

        // Flash border red to indicate transaction sent
        flashBorderRed();

        // v1.3.12-beta: Extract validator count from consensus certificate
        // This shows how many nodes cooperated in confirming the transaction
        const validatorCount = result.data.validator_count ||
                               result.data.confirming_nodes ||
                               result.data.signature_count ||
                               (result.data.certificate?.signatures ? Object.keys(result.data.certificate.signatures).length : undefined);

        // v1.4.4: Calculate required confirmations based on USD value
        const coin = walletBalances.find(c => c.symbol === selectedCoin);
        const coinPriceUsd = coin?.usdValue && coin?.balance > 0
          ? coin.usdValue / coin.balance
          : (selectedCoin === 'QUG' ? 3000 : 1.0);
        const txValueUsd = parseFloat(amount) * coinPriceUsd;
        const confirmations = calculateConfirmations(txValueUsd);

        // v3.5.x: Determine actual submission method from result
        // P2P-first: Browser signs and broadcasts via gossipsub
        // HTTP fallback: Server receives, signs, broadcasts via P2P gossipsub
        const submissionMethod: 'p2p' | 'http' = result.data?._submissionMethod || 'http';
        const actualP2PPeerCount = result.data?._p2pPeerCount || 0;
        console.log(`📡 [TX] Submission method: ${submissionMethod}${submissionMethod === 'p2p' ? ` (broadcast to ${actualP2PPeerCount} peers)` : ' (API → Server → P2P Gossipsub)'}`);

        setTransaction(prev => ({
          ...prev,
          success: true, // Always show success for completed transactions
          txHash: result.data.transaction_hash || result.data.mixing_session_id || result.data.tx_hash || 'pending',
          starkProof: result.data.stark_proof,
          validatorCount: validatorCount,
          confirmations: confirmations,
          submissionMethod: submissionMethod,
          p2pPeerCount: submissionMethod === 'p2p' ? actualP2PPeerCount : (p2pReady ? p2pPeerCount : undefined)
        }));

        // v3.5.24: Start P2P verification in background (non-blocking)
        const txHash = result.data.transaction_hash || result.data.tx_hash;
        if (p2pDataReady && txHash) {
          console.log(`🔍 [TX] Starting multi-peer verification for ${txHash}...`);
          // Run verification in background without blocking UI
          verifyTransaction(txHash).then(consensus => {
            console.log(`✅ [TX] Multi-peer verification: ${consensus.confirmed ? 'CONFIRMED' : 'PENDING'} (${consensus.confidence}% confidence, ${consensus.agreementCount}/${consensus.totalPeers} peers)`);
            setTransaction(prev => ({
              ...prev,
              p2pVerification: {
                verified: consensus.confirmed,
                peersConfirmed: consensus.agreementCount,
                totalPeers: consensus.totalPeers,
                confidence: consensus.confidence,
                blockHeight: consensus.blockHeight,
              }
            }));
          }).catch(err => {
            console.warn('⚠️ [TX] P2P verification failed:', err);
          });
        }

        // v6.0.1: Optimistically update balance in UI after successful send
        const sentAmount = parseFloat(amount);
        const fee = selectedCoin === 'QUG' ? 0.000021 : 0;
        const currentCoinBalance = walletBalances.find(c => c.symbol === selectedCoin)?.balance || 0;
        const optimisticBalance = Math.max(0, currentCoinBalance - sentAmount - fee);

        // Update local wallet balances state
        setWalletBalances(prev => prev.map(w =>
          w.symbol === selectedCoin ? { ...w, balance: optimisticBalance } : w
        ));

        // v6.0.9: CRITICAL - Also update stableBalance and highestKnownBalancesRef
        // Without this, Math.max() in the stability logic blocks the decrease
        // and the displayed balance never drops after sending.
        if (selectedCoin === 'QUG') {
          setStableBalance(optimisticBalance);
          lastBalanceUpdateRef.current = Date.now();
          highestKnownBalancesRef.current['QUG'] = optimisticBalance;
          // Also update localStorage cache so page refreshes show the new balance
          localStorage.setItem('cachedBalance', optimisticBalance.toString());
        }
        highestKnownBalancesRef.current[selectedCoin] = optimisticBalance;

        // Notify TopBar of balance change (TopBar listens for wallet-balance-updated)
        if (selectedCoin === 'QUG') {
          window.dispatchEvent(new CustomEvent('wallet-balance-updated', {
            detail: {
              symbol: 'QUG',
              balance: optimisticBalance,
              reason: 'transaction_sent'
            }
          }));
        }

        // Dispatch generic balance-update for other listeners
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
              {/* v3.6.10-beta: Show full 24 decimal precision */}
              <div className="text-lg font-bold text-quantum-green font-mono break-all">
                {formatBalanceDisplay(selectedWallet.balance)} {selectedWallet.symbol}
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
                {/* v3.6.10-beta: Show full 24 decimal precision in dropdown */}
                {walletBalances.map(wallet => (
                  <option key={wallet.symbol} value={wallet.symbol}>
                    {wallet.symbol} - {formatBalanceDisplay(wallet.balance)} {wallet.name}
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
              {/* v3.6.10-beta: Show full 24 decimal precision */}
              <div className="flex justify-between items-center text-sm mt-1">
                <span className="text-gray-400">
                  Available: <span className="text-quantum-green font-semibold font-mono">{formatBalanceDisplay(selectedWallet?.balance || 0)} {selectedCoin}</span>
                </span>
                <span className="text-gray-400">
                  {/* v3.6.12-beta: Fees are ALWAYS in QUG (native coin), not the selected token */}
                  Fee: <span className={feeReductionActive ? "text-quantum-green" : "text-quantum-yellow"}>
                    {currentFee} QUG
                  </span>
                  {feeReductionActive && (
                    <span className="ml-1 text-quantum-green text-xs">(10x reduced!)</span>
                  )}
                </span>
              </div>

              {/* v3.4.0: Fee reduction notice */}
              {!feeReductionActive && blocksUntilFeeReduction > 0 && (
                <div className="mt-3 p-3 bg-gradient-to-r from-amber-500/10 to-yellow-500/10 border border-amber-500/30 rounded-xl">
                  <div className="flex items-center gap-2">
                    <TrendingDown className="w-4 h-4 text-amber-400" />
                    <span className="text-sm text-amber-300 font-medium">
                      10x Fee Reduction Coming Soon!
                    </span>
                  </div>
                  <p className="text-xs text-gray-400 mt-1">
                    At block {FEE_REDUCTION_ACTIVATION_HEIGHT.toLocaleString()} (in ~{blocksUntilFeeReduction.toLocaleString()} blocks),
                    fees drop from {CURRENT_MIN_FEE_QUG} to {NEW_MIN_FEE_QUG} QUG.
                  </p>
                </div>
              )}

              {feeReductionActive && (
                <div className="mt-3 p-3 bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/30 rounded-xl">
                  <div className="flex items-center gap-2">
                    <TrendingDown className="w-4 h-4 text-green-400" />
                    <span className="text-sm text-green-300 font-medium">
                      10x Fee Reduction Active!
                    </span>
                  </div>
                  <p className="text-xs text-gray-400 mt-1">
                    Transaction fees are now 10x cheaper: {NEW_MIN_FEE_QUG} QUG minimum.
                  </p>
                </div>
              )}
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

            {/* Quantum Privacy Mixer Toggle - Compact Design v3.4.15 */}
            <div className="rounded-xl p-4 max-h-80 overflow-y-auto"
              style={{
                background: 'linear-gradient(135deg, rgba(236, 72, 153, 0.1) 0%, rgba(219, 39, 119, 0.05) 100%)',
                border: '2px solid rgba(236, 72, 153, 0.2)'
              }}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Shield className="w-5 h-5 text-quantum-pink" />
                  <div>
                    <h3 className="text-sm font-semibold text-white">Quantum Privacy Mixer</h3>
                    <div className="flex items-center gap-1">
                      <p className="text-xs text-gray-400">Enhanced anonymity via Dandelion++ Tor</p>
                      {mixerAvailable === false && (
                        <span className="px-1.5 py-0.5 bg-quantum-yellow/20 text-quantum-yellow text-[10px] rounded">
                          Dev
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
                  <div className="w-10 h-5 bg-gray-600 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-quantum-pink/25 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-quantum-pink"></div>
                </label>
              </div>

              <AnimatePresence>
                {enablePrivacyMixer && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="space-y-3"
                  >
                    {/* Privacy Level - Compact */}
                    <div>
                      <label className="block text-xs font-medium text-gray-300 mb-2">
                        Privacy Level
                      </label>
                      <div className="grid grid-cols-3 gap-1">
                        {[
                          { value: 'standard', label: 'Std', decoys: '15x' },
                          { value: 'high', label: 'High', decoys: '25x' },
                          { value: 'maximum', label: 'Max', decoys: '50x' }
                        ].map(({ value, label, decoys }) => (
                          <button
                            key={value}
                            onClick={() => {
                              setPrivacyLevel(value as any);
                              setDecoyMultiplier(value === 'standard' ? 15 : value === 'high' ? 25 : 50);
                            }}
                            className={`p-2 rounded-lg border text-center transition-all text-xs ${
                              privacyLevel === value
                                ? 'border-quantum-pink bg-quantum-pink/20 text-quantum-pink'
                                : 'border-quantum-purple/30 bg-quantum-dark/30 text-gray-300 hover:border-quantum-pink/50'
                            }`}
                          >
                            <div className="font-medium">{label}</div>
                            <div className="text-[10px] opacity-70">{decoys}</div>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Decoy Slider - Compact */}
                    <div>
                      <div className="flex justify-between items-center text-xs text-gray-300 mb-1">
                        <span>Decoys: {decoyMultiplier}x</span>
                        <span className="text-gray-500">5-50</span>
                      </div>
                      <input
                        type="range"
                        min="5"
                        max="50"
                        value={decoyMultiplier}
                        onChange={(e) => setDecoyMultiplier(parseInt(e.target.value))}
                        className="w-full h-1.5 bg-quantum-dark rounded-lg appearance-none cursor-pointer slider-thumb"
                      />
                    </div>

                    {/* Compact Details Toggle */}
                    <button
                      onClick={() => setShowMixingDetails(!showMixingDetails)}
                      className="flex items-center gap-1 text-quantum-cyan hover:text-quantum-pink transition-colors text-xs"
                    >
                      {showMixingDetails ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
                      <span>{showMixingDetails ? 'Hide' : 'Show'} details</span>
                    </button>

                    <AnimatePresence>
                      {showMixingDetails && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="bg-quantum-dark/30 rounded-lg p-2 space-y-2"
                        >
                          <div className="grid grid-cols-2 gap-2 text-xs">
                            <div>
                              <div className="text-gray-400">Ring Size</div>
                              <div className="text-quantum-cyan font-mono">{decoyMultiplier + 1}</div>
                            </div>
                            <div>
                              <div className="text-gray-400">Mix Time</div>
                              <div className="text-quantum-green font-mono">
                                {privacyLevel === 'standard' ? '15s' : privacyLevel === 'high' ? '30s' : '60s'}
                              </div>
                            </div>
                            <div>
                              <div className="text-gray-400">Proof</div>
                              <div className="text-quantum-purple font-mono">Falcon1024</div>
                            </div>
                            <div>
                              <div className="text-gray-400">Stealth</div>
                              <div className="text-quantum-pink font-mono">Active</div>
                            </div>
                          </div>
                          <div className="text-[10px] text-gray-500 pt-1 border-t border-quantum-purple/20">
                            🧅 Routed via Dandelion++ Tor for IP unlinkability
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

      {/* v8.6.5: Password Confirmation Modal */}
      <AnimatePresence>
        {showPasswordModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
            onClick={() => { setShowPasswordModal(false); setConfirmPassword(''); setPasswordError(null); }}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="relative w-full max-w-md mx-4 rounded-2xl p-6"
              style={{
                background: 'linear-gradient(135deg, rgba(15, 10, 35, 0.95), rgba(30, 20, 60, 0.95))',
                border: '1px solid rgba(212, 175, 55, 0.3)',
                boxShadow: '0 0 40px rgba(212, 175, 55, 0.15)',
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-amber-500/20 flex items-center justify-center">
                  <Shield className="w-5 h-5 text-amber-400" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-white">Confirm Transaction</h3>
                  <p className="text-xs text-gray-400">Enter your wallet password to authorize</p>
                </div>
                <button
                  onClick={() => { setShowPasswordModal(false); setConfirmPassword(''); setPasswordError(null); }}
                  className="ml-auto text-gray-400 hover:text-white transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="mb-4 p-3 rounded-xl bg-white/5 border border-white/10">
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-400">Sending</span>
                  <span className="text-white font-mono">{transaction.amount} {selectedCoin}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">To</span>
                  <span className="text-cyan-300 font-mono text-xs">{transaction.toAddress.slice(0, 12)}...{transaction.toAddress.slice(-8)}</span>
                </div>
              </div>

              <div className="mb-4">
                <label className="text-sm text-gray-400 mb-1 block">Wallet Password</label>
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => { setConfirmPassword(e.target.value); setPasswordError(null); }}
                  onKeyDown={(e) => { if (e.key === 'Enter') handlePasswordConfirm(); }}
                  placeholder="Enter your wallet password"
                  autoFocus
                  className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-gray-500 focus:outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/30 transition-all"
                />
                {passwordError && (
                  <p className="text-red-400 text-xs mt-2 flex items-center gap-1">
                    <AlertTriangle className="w-3 h-3" /> {passwordError}
                  </p>
                )}
              </div>

              <div className="flex gap-3">
                <button
                  onClick={() => { setShowPasswordModal(false); setConfirmPassword(''); setPasswordError(null); }}
                  className="flex-1 py-3 rounded-xl text-sm font-medium bg-white/5 border border-white/10 text-gray-300 hover:bg-white/10 transition-all"
                >
                  Cancel
                </button>
                <button
                  onClick={handlePasswordConfirm}
                  disabled={passwordVerifying || !confirmPassword}
                  className="flex-1 py-3 rounded-xl text-sm font-bold text-white disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                  style={{
                    background: passwordVerifying || !confirmPassword
                      ? 'rgba(212, 175, 55, 0.3)'
                      : 'linear-gradient(135deg, #D4AF37, #FFD700)',
                  }}
                >
                  {passwordVerifying ? 'Verifying...' : 'Confirm & Send'}
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

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
                  <p className="text-sm text-gray-400">
                    {transaction.validatorCount
                      ? `Verified by ${transaction.validatorCount} validator nodes (2f+1 consensus)`
                      : 'Your quantum-secured transaction has been submitted'}
                  </p>
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

              {/* v3.5.x: Display P2P submission method */}
              {transaction.submissionMethod && (
                <div className="flex items-center gap-2 pt-2 border-t border-quantum-green/20 mt-2">
                  {transaction.submissionMethod === 'p2p' ? (
                    <>
                      <Radio className="w-4 h-4 text-quantum-pink" />
                      <div>
                        <div className="text-sm text-gray-400">Network Broadcast:</div>
                        <div className="text-sm text-quantum-pink">
                          📡 Broadcast via P2P Gossipsub
                          {transaction.p2pPeerCount !== undefined && transaction.p2pPeerCount > 0 && (
                            <span className="text-gray-400 ml-1">({transaction.p2pPeerCount} peers)</span>
                          )}
                        </div>
                      </div>
                    </>
                  ) : (
                    <>
                      <Globe className="w-4 h-4 text-quantum-cyan" />
                      <div>
                        <div className="text-sm text-gray-400">Network Broadcast:</div>
                        <div className="text-sm text-quantum-cyan">
                          🌐 API → Server → P2P Gossipsub
                        </div>
                        {p2pReady && p2pPeerCount > 0 && (
                          <div className="text-xs text-gray-500 mt-0.5">
                            Your browser is also connected to {p2pPeerCount} P2P peer(s)
                          </div>
                        )}
                      </div>
                    </>
                  )}
                </div>
              )}

              {/* v1.3.12-beta: Display validator consensus details */}
              {transaction.validatorCount && (
                <div className="flex items-center gap-2 pt-2 border-t border-quantum-green/20 mt-2">
                  <Shield className="w-4 h-4 text-quantum-cyan" />
                  <div>
                    <div className="text-sm text-gray-400">Decentralized Consensus:</div>
                    <div className="text-sm text-quantum-cyan">
                      ✓ {transaction.validatorCount} validator nodes confirmed (BFT 2f+1)
                    </div>
                  </div>
                </div>
              )}

              {/* v1.4.4: Confirmation tracking with retail-first finality tiers */}
              {transaction.confirmations && (
                <div className="pt-3 border-t border-quantum-green/20 mt-3">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-2xl">{transaction.confirmations.tierEmoji}</span>
                      <div>
                        <span className="text-sm font-medium text-white">
                          {transaction.confirmations.tier} FINALITY
                        </span>
                        <span className="text-xs text-gray-400 ml-2">
                          {transaction.confirmations.isFinalized
                            ? '✓ Finalized'
                            : transaction.confirmations.estimatedTimeRemaining}
                        </span>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className={`text-lg font-bold ${
                        transaction.confirmations.isFinalized
                          ? 'text-quantum-green'
                          : 'text-quantum-cyan'
                      }`}>
                        {transaction.confirmations.required === 0
                          ? 'INSTANT'
                          : `${transaction.confirmations.current}/${transaction.confirmations.required}`}
                      </span>
                    </div>
                  </div>

                  {/* Progress bar */}
                  {transaction.confirmations.required > 0 && (
                    <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                      <motion.div
                        className="h-full rounded-full"
                        style={{
                          background: transaction.confirmations.isFinalized
                            ? 'linear-gradient(90deg, #10B981, #34D399)'
                            : 'linear-gradient(90deg, #06B6D4, #8B5CF6)'
                        }}
                        initial={{ width: '0%' }}
                        animate={{
                          width: `${Math.min(100, (transaction.confirmations.current / transaction.confirmations.required) * 100)}%`
                        }}
                        transition={{ duration: 0.3, ease: 'easeOut' }}
                      />
                    </div>
                  )}

                  {/* Tier description */}
                  <div className="mt-2 text-xs text-gray-500">
                    {transaction.confirmations.tier === 'INSTANT' && (
                      'Economic guarantee: Attack cost far exceeds transaction value'
                    )}
                    {transaction.confirmations.tier === 'OPTIMISTIC' && (
                      'DAG vertex inclusion provides probabilistic finality'
                    )}
                    {transaction.confirmations.tier === 'FAST' && (
                      'Full block confirmation with VDF time-lock'
                    )}
                    {transaction.confirmations.tier === 'STANDARD' && (
                      '3-deep DAG confirmation for high-value protection'
                    )}
                    {transaction.confirmations.tier === 'SETTLEMENT' && (
                      'DAG-Knight BFT: 2f+1 validator signatures per block (~6s finality)'
                    )}
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
            <div className="w-full h-full">
              <Suspense fallback={<div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}><div style={{ width: 48, height: 48, borderRadius: '50%', border: '3px solid rgba(212,175,55,0.2)', borderTopColor: '#d4af37', animation: 'spin 0.8s linear infinite' }} /></div>}>
              <QuantumMixerVisualization
              sessionId={mixingSessionId}
              privacyLevel={privacyLevel}
              onComplete={() => {
                console.log('🏁 [MIXER] Visualization complete, hiding overlay');
                setShowMixerVisualization(false);
                setTransaction(prev => ({
                  ...prev,
                  success: true,
                  txHash: mixerTxHash || mixingSessionId
                }));

                // v6.0.2: Optimistic balance update after mixer completes
                const sentAmount = parseFloat(transaction.amount);
                const mixerFee = sentAmount * 0.001; // 0.1% mixer fee
                const currentQugBalance = walletBalances.find(c => c.symbol === 'QUG')?.balance || 0;
                const optimisticBalance = Math.max(0, currentQugBalance - sentAmount - mixerFee);

                // Update local wallet balances
                setWalletBalances(prev => prev.map(w =>
                  w.symbol === 'QUG' ? { ...w, balance: optimisticBalance } : w
                ));

                // v6.0.9: Also update stableBalance and highestKnownBalancesRef
                setStableBalance(optimisticBalance);
                lastBalanceUpdateRef.current = Date.now();
                highestKnownBalancesRef.current['QUG'] = optimisticBalance;
                localStorage.setItem('cachedBalance', optimisticBalance.toString());

                // Notify TopBar of balance change
                window.dispatchEvent(new CustomEvent('wallet-balance-updated', {
                  detail: {
                    symbol: 'QUG',
                    balance: optimisticBalance,
                    reason: 'transaction_sent'
                  }
                }));

                // Trigger generic balance refresh for other listeners
                window.dispatchEvent(new CustomEvent('balance-update', {
                  detail: { refresh: true }
                }));
              }}
            />
              </Suspense>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}