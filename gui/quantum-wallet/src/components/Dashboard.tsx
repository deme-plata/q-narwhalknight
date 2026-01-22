import { useState, useEffect, memo, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Zap, AlertCircle, Copy, Check, Wallet, Coins, ChevronLeft, ChevronRight, Calendar, DollarSign, TrendingUp, TrendingDown, QrCode, Info, Plus, Send, BarChart3 } from 'lucide-react';
import { qnkAPI, type NodeStatus } from '../services/api'; // debounce not needed - SSE in App.tsx
import TransactionDetailsModal from './TransactionDetailsModal';
import QRCodeModal from './QRCodeModal';
import StripeCheckout from './StripeCheckout';
import DAGKnightVisualization from './DAGKnightVisualization';
import QNOOracleVisualization from './QNOOracleVisualization';
import LoanApplicationModal from './LoanApplicationModal';
import LoanApprovalModal from './LoanApprovalModal';
import LoanPaybackModal from './LoanPaybackModal';
import ActiveLoansCard from './ActiveLoansCard';
import WalletCardWithGraph from './WalletCardWithGraph';
import PhaseTransitionModal from './PhaseTransitionModal';
import StakingModal from './StakingModal';
import CustomTokensCard from './CustomTokensCard';
import FinanceModal from './FinanceModal';
import { TICKER_SYMBOL } from '../constants/ticker';

interface Transaction {
  id: string;
  type: 'receive' | 'send' | 'mining';
  amount: number;
  from?: string;
  to?: string;
  timestamp: string;
  txHash: string;
}

interface BalanceHistoryPoint {
  timestamp: number;
  balance: number;
}

interface WalletBalance {
  symbol: string;
  name: string;
  balance: number;
  usdValue?: number;
  icon: 'qug' | 'usd' | 'btc' | 'eth' | 'sol' | 'zec' | 'iron' | 'custom';
  color: string;
  comingSoon?: boolean;
  shieldedOnly?: boolean; // For Privacy coins like Zcash
  history?: BalanceHistoryPoint[]; // Balance history for mini-graph
}

interface DashboardProps {
  onNavigateToSend?: (coinSymbol: string) => void;
}

const Dashboard = memo(function Dashboard({ onNavigateToSend }: DashboardProps) {
  const [nodeStatus, setNodeStatus] = useState<NodeStatus | null>(null);
  const [recentTransactions, setRecentTransactions] = useState<Transaction[]>(() => {
    // Load faucet transactions from localStorage on mount
    try {
      const stored = localStorage.getItem('faucetTransactions');
      const loaded = stored ? JSON.parse(stored) : [];
      console.log('💾 Loaded faucet transactions from localStorage:', loaded.length);
      return loaded;
    } catch (err) {
      console.error('❌ Failed to load faucet transactions:', err);
      return [];
    }
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [walletAddress, setWalletAddress] = useState('');
  const [copiedAddress, setCopiedAddress] = useState(false);
  const [faucetLoading, setFaucetLoading] = useState(false);
  const [faucetMessage, setFaucetMessage] = useState('');
  const [sseConnected, setSseConnected] = useState(false);
  const [showLoanModal, setShowLoanModal] = useState(false);
  const [showLoanApprovalModal, setShowLoanApprovalModal] = useState(false);
  const [approvedLoanDetails, setApprovedLoanDetails] = useState<any>(null);
  const [showLoanPaybackModal, setShowLoanPaybackModal] = useState(false);
  const [selectedLoanId, setSelectedLoanId] = useState<string | null>(null);
  const [transactionError, setTransactionError] = useState<string | null>(null);
  const [showFinanceModal, setShowFinanceModal] = useState(false);

  // Multi-wallet state - 🚨 v2.3.7-beta: Initialize from cache to prevent zero balance on refresh
  const [walletBalances, setWalletBalances] = useState<WalletBalance[]>(() => {
    // Try to load cached balances immediately to avoid showing 0 on refresh
    const cachedQugBalance = localStorage.getItem('cachedBalance');
    const cachedQugValue = cachedQugBalance ? parseFloat(cachedQugBalance) : 0;
    const validQugBalance = !isNaN(cachedQugValue) && isFinite(cachedQugValue) ? cachedQugValue : 0;

    // Also load QUGUSD cached balance
    const cachedQugusdBalance = localStorage.getItem('cachedQugusdBalance');
    const cachedQugusdValue = cachedQugusdBalance ? parseFloat(cachedQugusdBalance) : 0;
    const validQugusdBalance = !isNaN(cachedQugusdValue) && isFinite(cachedQugusdValue) ? cachedQugusdValue : 0;

    // Also load balance history for the graph
    let qugHistory: { timestamp: number; balance: number }[] = [];
    let qugusdHistory: { timestamp: number; balance: number }[] = [];
    try {
      const storedHistory = localStorage.getItem('walletBalanceHistory');
      if (storedHistory) {
        const parsed = JSON.parse(storedHistory);
        qugHistory = parsed['QUG'] || [];
        qugusdHistory = parsed['QUGUSD'] || [];
      }
    } catch (e) {
      console.warn('Failed to load balance history from cache:', e);
    }

    const initialBalances: WalletBalance[] = [];

    // Add QUG if we have a cached balance
    if (validQugBalance > 0) {
      console.log('🚀 [INIT] Initializing with cached QUG balance:', validQugBalance);
      initialBalances.push({
        symbol: 'QUG',
        name: 'Quillon Gold',
        balance: validQugBalance,
        icon: 'qug' as const,
        color: 'from-amber-400 to-yellow-600',
        history: qugHistory.length >= 2 ? qugHistory : undefined,
      });
    }

    // Add QUGUSD if we have a cached balance
    if (validQugusdBalance > 0) {
      console.log('🚀 [INIT] Initializing with cached QUGUSD balance:', validQugusdBalance);
      initialBalances.push({
        symbol: 'QUGUSD',
        name: 'Quillon USD',
        balance: validQugusdBalance,
        usdValue: validQugusdBalance, // 1:1 peg to USD
        icon: 'usd' as const,
        color: 'from-blue-400 to-cyan-500',
        history: qugusdHistory.length >= 2 ? qugusdHistory : undefined,
      });
    }

    return initialBalances;
  });
  const [usdBalance, setUsdBalance] = useState<number>(0);

  // Animation state for balance updates
  const [balanceAnimations, setBalanceAnimations] = useState<Record<string, boolean>>({});

  // CRITICAL FIX: Track highest known balance per token to prevent showing stale/lower values
  // This prevents the bug where balance jumps from 66 to 0.71 on refresh
  // 🚨 v2.3.7-beta: Initialize from cache immediately (IIFE pattern since useRef doesn't accept functions)
  const highestKnownBalancesRef = useRef<Record<string, number>>((() => {
    const result: Record<string, number> = {};

    // Load QUG from cache
    const cachedQugBalance = localStorage.getItem('cachedBalance');
    const cachedQugValue = cachedQugBalance ? parseFloat(cachedQugBalance) : 0;
    if (!isNaN(cachedQugValue) && isFinite(cachedQugValue) && cachedQugValue > 0) {
      result['QUG'] = cachedQugValue;
    }

    // Load QUGUSD from cache
    const cachedQugusdBalance = localStorage.getItem('cachedQugusdBalance');
    const cachedQugusdValue = cachedQugusdBalance ? parseFloat(cachedQugusdBalance) : 0;
    if (!isNaN(cachedQugusdValue) && isFinite(cachedQugusdValue) && cachedQugusdValue > 0) {
      result['QUGUSD'] = cachedQugusdValue;
    }

    return result;
  })());

  // Initialize highestKnownBalancesRef from localStorage on mount
  // 🚨 v2.3.7-beta: Dispatch cached balance to App.tsx IMMEDIATELY on mount
  // This ensures TopBar shows correct balance before API call completes
  useEffect(() => {
    const cachedBalance = localStorage.getItem('cachedBalance');
    if (cachedBalance) {
      const value = parseFloat(cachedBalance);
      if (!isNaN(value) && isFinite(value) && value > 0) {
        // Update local tracking
        if (value > (highestKnownBalancesRef.current['QUG'] || 0)) {
          highestKnownBalancesRef.current['QUG'] = value;
        }
        // IMMEDIATELY dispatch to App.tsx - don't wait for API
        console.log('🚀 [INSTANT SYNC] Dispatching cached balance to App.tsx on mount:', value);
        window.dispatchEvent(new CustomEvent('balance-update', {
          detail: { balance: value, source: 'Dashboard.mount.instant' }
        }));
      }
    }
  }, []); // Run once on mount

  // Balance history tracking (keep last 20 data points per wallet) - load from localStorage
  const [_balanceHistory, setBalanceHistory] = useState<Record<string, BalanceHistoryPoint[]>>(() => {
    try {
      const saved = localStorage.getItem('walletBalanceHistory');
      return saved ? JSON.parse(saved) : {};
    } catch {
      return {};
    }
  });

  // Transaction details modal state
  const [selectedTransaction, setSelectedTransaction] = useState<Transaction | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // QR code modal state
  const [isQRModalOpen, setIsQRModalOpen] = useState(false);

  // Node info modal state
  const [isNodeInfoModalOpen, setIsNodeInfoModalOpen] = useState(false);

  // AI Report modal state
  const [isAIReportModalOpen, setIsAIReportModalOpen] = useState(false);
  const [aiReportLoading, setAiReportLoading] = useState(false);
  const [aiReport, setAiReport] = useState<string>('');

  // USD wallet modal states
  const [isAddUSDModalOpen, setIsAddUSDModalOpen] = useState(false);
  const [isSendUSDModalOpen, setIsSendUSDModalOpen] = useState(false);
  const [showStripeCheckout, setShowStripeCheckout] = useState(false);
  const [usdAmount, setUsdAmount] = useState('');
  const [usdRecipient, setUsdRecipient] = useState('');
  const [stripeLoading, setStripeLoading] = useState(false);
  const [stripeError, setStripeError] = useState<string | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  // Enhanced filtering and pagination state
  const [filterType, setFilterType] = useState<'all' | 'receive' | 'send' | 'mining'>('all');
  const [sortBy, setSortBy] = useState<'date' | 'amount'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 20;

  // Phase transition modal state
  const [showPhaseModal, setShowPhaseModal] = useState(false); // Disabled - phase transition modal no longer needed
  const [showStakingModal, setShowStakingModal] = useState(false);

  // Generate AI Report
  const generateAIReport = async () => {
    setAiReportLoading(true);
    setIsAIReportModalOpen(true);
    setAiReport('');

    try {
      // Prepare context for AI
      const networkStats = nodeStatus ? {
        tpsCurrent: nodeStatus.tps_current || 0,
        tpsAverage: nodeStatus.tps_average || 0,
        connectedPeers: nodeStatus.connected_peers || 0,
        isValidator: nodeStatus.is_validator,
        currentHeight: nodeStatus.current_height || 0
      } : null;

      const walletContext = {
        balance: nodeStatus?.balance || 0,
        walletAddress: walletAddress,
        recentTransactions: recentTransactions.slice(0, 10),
        networkStats
      };

      const prompt = `Analyze this Q-NarwhalKnight wallet and network performance:

Wallet Balance: ${walletContext.balance.toFixed(4)} QUG
${networkStats ? `Network TPS: ${networkStats.tpsCurrent} current, ${networkStats.tpsAverage} average
Connected Peers: ${networkStats.connectedPeers}
Block Height: ${networkStats.currentHeight}
Validator Status: ${networkStats.isValidator ? 'Active' : 'Not active'}` : 'Network: Offline'}

Recent Transactions: ${walletContext.recentTransactions.length} transactions

Provide a brief analysis (under 250 tokens) covering:
1. Balance health & recommendations
2. Network participation insights
3. Key optimizations for earning rewards`;

      // Stream AI response with reduced token limit for faster generation
      const eventSource = new EventSource(`/api/chat/stream?content=${encodeURIComponent(prompt)}&max_tokens=250`);
      let fullReport = '';

      eventSource.addEventListener('token', (event) => {
        try {
          const data = JSON.parse(event.data);
          fullReport = data.cumulative || '';
          setAiReport(fullReport);
        } catch (e) {
          console.error('Failed to parse AI token:', e);
        }
      });

      eventSource.addEventListener('complete', () => {
        setAiReportLoading(false);
        eventSource.close();
      });

      eventSource.addEventListener('error', (error) => {
        console.error('AI Report generation error:', error);
        setAiReportLoading(false);
        setAiReport('Failed to generate AI report. Please try again.');
        eventSource.close();
      });

    } catch (error) {
      console.error('Failed to generate AI report:', error);
      setAiReportLoading(false);
      setAiReport('Failed to generate AI report. Please try again.');
    }
  };

  // Save faucet transactions to localStorage whenever they change
  useEffect(() => {
    try {
      // Only save faucet transactions (manually added)
      const faucetTxs = recentTransactions.filter(tx => tx.id.startsWith('faucet-'));
      console.log('💾 Saving faucet transactions to localStorage:', faucetTxs.length);
      localStorage.setItem('faucetTransactions', JSON.stringify(faucetTxs));
    } catch (err) {
      console.error('❌ Failed to save faucet transactions:', err);
    }
  }, [recentTransactions]);

  // Detect balance changes and trigger animations
  // Use ref to track previous balances to avoid re-render loops
  const previousBalancesRef = useRef<Record<string, number>>({});
  const animationTimeoutsRef = useRef<Record<string, NodeJS.Timeout>>({});

  useEffect(() => {
    const newAnimations: Record<string, boolean> = {};

    walletBalances.forEach(wallet => {
      const key = wallet.symbol;
      const prevBalance = previousBalancesRef.current[key];
      const currentBalance = wallet.balance;

      // Update ref with current balance
      previousBalancesRef.current[key] = currentBalance;

      // CRITICAL FIX: Only trigger animation if balance changed by meaningful amount (> 0.0001)
      // This prevents flickering from floating-point rounding or micro-variations
      const balanceDiff = Math.abs((prevBalance ?? currentBalance) - currentBalance);
      const isSignificantChange = prevBalance !== undefined && balanceDiff > 0.0001;

      if (isSignificantChange) {
        newAnimations[key] = true;
        console.log(`🎨 Balance animation triggered for ${key}: ${prevBalance?.toFixed(4)} → ${currentBalance.toFixed(4)} (diff: ${balanceDiff.toFixed(6)})`);

        // Clear any existing timeout for this wallet
        if (animationTimeoutsRef.current[key]) {
          clearTimeout(animationTimeoutsRef.current[key]);
        }

        // Auto-disable animation after 3 seconds
        animationTimeoutsRef.current[key] = setTimeout(() => {
          setBalanceAnimations(prev => ({ ...prev, [key]: false }));
        }, 3000);
      } else {
        // Only set to false if no animation was just triggered
        newAnimations[key] = balanceAnimations[key] || false;
      }
    });

    // Only update state if animations actually changed
    setBalanceAnimations(prev => {
      const hasChanges = Object.keys(newAnimations).some(key => prev[key] !== newAnimations[key]);
      return hasChanges ? newAnimations : prev;
    });

    // Cleanup function
    return () => {
      Object.values(animationTimeoutsRef.current).forEach(timeout => clearTimeout(timeout));
    };
  }, [walletBalances]);

  // Fetch real data from Q-NarwhalKnight API
  useEffect(() => {
    let mounted = true;
    // let eventSource: EventSource | null = null; // Disabled - App.tsx handles SSE

    // Create debounced versions of fetch functions to prevent request storms
    // These will delay execution by 1.5s after the last call
    const fetchNodeStatusCore = async () => {
      console.log('Fetching node status...');
      if (!mounted) return;

      // v2.3.31-beta: Check BOTH local ref AND global localStorage cooldown
      const nodeGlobalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
      const nodeGlobalCooldownActive = Date.now() < nodeGlobalCooldownUntil;
      if (dexSwapCooldownRef.current || nodeGlobalCooldownActive) {
        console.log('🚫 [fetchNodeStatusCore] SKIPPING during DEX cooldown (global:', nodeGlobalCooldownActive, ')');
        return;
      }

      try {
        const response = await qnkAPI.getNodeStatus();
        console.log('Node status response:', response);
        if (!mounted) return;

        if (response.success && response.data) {
          const currentWalletAddress = localStorage.getItem('walletAddress');
          let walletBalance = 0;

          if (currentWalletAddress) {
            const previousHighest = highestKnownBalancesRef.current['QUG'] || 0;
            const cachedBalance = localStorage.getItem('cachedBalance');
            const cachedValue = cachedBalance ? parseFloat(cachedBalance) : 0;

            try {
              const balanceResponse = await qnkAPI.getWalletBalance(currentWalletAddress);
              if (!mounted) return;

              if (balanceResponse.success && balanceResponse.data) {
                const fetchedBalance = balanceResponse.data.balance_qnk || 0;
                console.log('✅ Balance fetched:', fetchedBalance, '(highest known:', previousHighest, ', cached:', cachedValue, ')');

                // CRITICAL FIX: Validate balance before accepting
                // Allow decreases up to 10% or 1 QUG for legitimate transactions
                const referenceBalance = Math.max(previousHighest, cachedValue);
                const minAcceptable = Math.max(0, referenceBalance * 0.9 - 1);

                if (fetchedBalance >= minAcceptable || referenceBalance === 0) {
                  walletBalance = fetchedBalance;
                  // Update tracking
                  if (fetchedBalance > previousHighest) {
                    highestKnownBalancesRef.current['QUG'] = fetchedBalance;
                  }
                  // v2.3.31-beta: Check global cooldown for localStorage write
                  const lsGlobalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
                  const lsGlobalCooldownActive = Date.now() < lsGlobalCooldownUntil;
                  if (!dexSwapCooldownRef.current && !lsGlobalCooldownActive) {
                    localStorage.setItem('cachedBalance', fetchedBalance.toString());
                  } else {
                    console.log('🚫 [fetchNodeStatusCore] SKIPPING localStorage write during DEX cooldown (global:', lsGlobalCooldownActive, ')');
                  }
                } else {
                  // Suspiciously low - use reference balance
                  console.warn(`⚠️ Rejecting suspiciously low balance: ${fetchedBalance} (expected ~${referenceBalance})`);
                  walletBalance = referenceBalance;
                }
              } else {
                // Authentication failed - use highest known or cached balance
                console.warn('⚠️ Balance query failed:', balanceResponse.error);
                walletBalance = Math.max(previousHighest, cachedValue);
                console.log('💰 Using best known balance:', walletBalance);
              }
            } catch (balanceErr) {
              console.warn('❌ Failed to fetch wallet balance:', balanceErr);
              // Fallback: use highest known or cached balance
              walletBalance = Math.max(previousHighest, cachedValue);
              console.log('💰 Using best known balance (error fallback):', walletBalance);
            }
          }

          if (mounted) {
            setNodeStatus({
              ...response.data,
              balance: walletBalance
            });
            setError(null);
          }
        } else {
          throw new Error(response.error || 'Failed to fetch node status');
        }
      } catch (err) {
        console.error('Error fetching node status:', err);
        if (mounted) {
          // Even if node status fails, try to load best known balance
          const cachedBalance = localStorage.getItem('cachedBalance');
          const cachedValue = cachedBalance ? parseFloat(cachedBalance) : 0;
          const previousHighest = highestKnownBalancesRef.current['QUG'] || 0;
          const bestBalance = Math.max(previousHighest, cachedValue);

          if (bestBalance > 0) {
            console.log('💰 Using best known balance after node status error:', bestBalance);
            setNodeStatus({
              balance: bestBalance,
              network_health: 'unknown',
              consensus_status: 'unknown',
              is_validator: false,
              current_round: 0,
              current_height: 0,
              tx_pool_size: 0,
              tps_current: 0,
              tps_average: 0,
              uptime_formatted: '0h 0m 0s',
            } as NodeStatus);
          } else {
            setError('Failed to connect to Q-NarwhalKnight node');
          }
        }
      }
    };

    const fetchWalletBalances = async () => {
      console.log('💰 Fetching wallet balances...');
      const currentWalletAddress = localStorage.getItem('walletAddress');

      if (!currentWalletAddress) {
        console.warn('⚠️ No wallet address found');
        return;
      }

      // Fetch fresh QUG balance from API (includes mining rewards)
      // 🚨 v2.3.7-beta CRITICAL FIX: Always have a fallback to localStorage cache
      // This prevents showing 0 balance on refresh when API fails
      const cachedBalanceStr = localStorage.getItem('cachedBalance');
      const cachedBalanceValue = cachedBalanceStr ? parseFloat(cachedBalanceStr) : 0;
      const validCachedBalance = !isNaN(cachedBalanceValue) && isFinite(cachedBalanceValue) ? cachedBalanceValue : 0;

      let qugBalance = validCachedBalance; // Start with cached value, not 0
      const previousHighest = highestKnownBalancesRef.current['QUG'] || validCachedBalance;

      console.log('🔍 [fetchWalletBalances] Starting with:', {
        cachedBalance: validCachedBalance,
        previousHighest: previousHighest,
        refValue: highestKnownBalancesRef.current['QUG']
      });

      try {
        const balanceResponse = await qnkAPI.getWalletBalance(currentWalletAddress);
        if (balanceResponse.success && balanceResponse.data) {
          const fetchedBalance = balanceResponse.data.balance_qnk || 0;
          console.log('💰 Fresh QUG balance fetched:', fetchedBalance, '(previous highest:', previousHighest, ', cached:', validCachedBalance, ')');

          // CRITICAL FIX: Only accept new balance if it's higher than or close to previous
          // Allow small decreases (up to 10% or 1 QUG) for legitimate transactions
          const referenceBalance = Math.max(previousHighest, validCachedBalance);
          const minAcceptable = Math.max(0, referenceBalance * 0.9 - 1);
          if (fetchedBalance >= minAcceptable || referenceBalance === 0) {
            qugBalance = fetchedBalance;
            // Update highest known if this is higher
            if (fetchedBalance > (highestKnownBalancesRef.current['QUG'] || 0)) {
              highestKnownBalancesRef.current['QUG'] = fetchedBalance;
            }
            // v2.3.31-beta: Check global cooldown for localStorage write
            const wbGlobalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
            const wbGlobalCooldownActive = Date.now() < wbGlobalCooldownUntil;
            if (!dexSwapCooldownRef.current && !wbGlobalCooldownActive) {
              localStorage.setItem('cachedBalance', fetchedBalance.toString());
            } else {
              console.log('🚫 [fetchWalletBalances] SKIPPING localStorage write during DEX cooldown (global:', wbGlobalCooldownActive, ')');
            }
          } else {
            // Fetched balance is suspiciously low - use cached/highest
            console.warn(`⚠️ Ignoring suspiciously low balance: ${fetchedBalance} (expected ~${referenceBalance})`);
            qugBalance = referenceBalance;
          }
        } else {
          // Fall back to highest known balance if API fails
          qugBalance = Math.max(previousHighest, validCachedBalance, nodeStatus?.balance || 0);
          console.warn('⚠️ Balance query failed, using best known:', qugBalance);
        }
      } catch (error) {
        // Fall back to highest known balance on error
        qugBalance = Math.max(previousHighest, validCachedBalance, nodeStatus?.balance || 0);
        console.error('❌ Failed to fetch QUG balance, using best known:', qugBalance, error);
      }

      // 🚨 NEVER allow 0 balance if we have cached value
      if (qugBalance === 0 && validCachedBalance > 0) {
        console.warn('⚠️ [fetchWalletBalances] qugBalance was 0 but cache has value, using cache:', validCachedBalance);
        qugBalance = validCachedBalance;
      }

      const now = Date.now();

      // Load saved history from localStorage
      let savedHistory: Record<string, BalanceHistoryPoint[]> = {};
      try {
        const saved = localStorage.getItem('walletBalanceHistory');
        savedHistory = saved ? JSON.parse(saved) : {};
      } catch {
        savedHistory = {};
      }

      // Merge saved history with new data point
      const qugSavedHistory = savedHistory['QUG'] || [];
      let qugHistory: BalanceHistoryPoint[] = [
        ...qugSavedHistory,
        { timestamp: now, balance: qugBalance }
      ].slice(-20); // Keep last 20 points

      // Ensure at least 2 points for graph rendering
      if (qugHistory.length < 2) {
        qugHistory = [
          { timestamp: now - 60000, balance: qugBalance }, // 1 minute ago
          { timestamp: now, balance: qugBalance }
        ];
      }

      const balances: WalletBalance[] = [
        {
          symbol: 'QUG',
          name: 'Quillon Graph',
          balance: qugBalance,
          icon: 'qug',
          color: 'from-amber-400 to-yellow-500',
          history: qugHistory
        }
      ];

      console.log('📊 [fetchWalletBalances] Initialized QUG with history:', qugHistory.length, 'points (', qugSavedHistory.length, 'from localStorage)');

      // 🚨 v2.3.7-beta: Fetch QUGUSD balance with cache fallback (same pattern as QUG)
      const cachedQugusdStr = localStorage.getItem('cachedQugusdBalance');
      const cachedQugusdValue = cachedQugusdStr ? parseFloat(cachedQugusdStr) : 0;
      const validCachedQugusd = !isNaN(cachedQugusdValue) && isFinite(cachedQugusdValue) ? cachedQugusdValue : 0;

      let qugUsdBalance = validCachedQugusd; // Start with cached value, not 0
      const previousHighestQugusd = highestKnownBalancesRef.current['QUGUSD'] || validCachedQugusd;

      console.log('🔍 [fetchWalletBalances] QUGUSD starting with:', {
        cachedBalance: validCachedQugusd,
        previousHighest: previousHighestQugusd,
        refValue: highestKnownBalancesRef.current['QUGUSD']
      });

      try {
        const response = await qnkAPI.getMultiTokenBalance();
        console.log('🔍 [Dashboard] Multi-token balance response:', JSON.stringify(response, null, 2));
        if (response.success && response.data && response.data.tokens) {
          // API returns tokens as object with lowercase keys: { qug: {...}, qugusd: {...} }
          const tokensObj = response.data.tokens;
          console.log('🔍 [Dashboard] Tokens object:', JSON.stringify(tokensObj, null, 2));

          let fetchedQugusd = 0;
          if (tokensObj.qugusd && tokensObj.qugusd.balance !== undefined) {
            fetchedQugusd = parseFloat(tokensObj.qugusd.balance) || 0;
            console.log('💵 [Dashboard] QUGUSD balance fetched:', fetchedQugusd);
          } else if (tokensObj.QUGUSD && tokensObj.QUGUSD.balance !== undefined) {
            // Try uppercase key as fallback
            fetchedQugusd = parseFloat(tokensObj.QUGUSD.balance) || 0;
            console.log('💵 [Dashboard] QUGUSD balance fetched (uppercase):', fetchedQugusd);
          } else {
            console.warn('⚠️ [Dashboard] QUGUSD not found in tokens object');
          }

          // Validate fetched balance (same logic as QUG)
          const referenceQugusd = Math.max(previousHighestQugusd, validCachedQugusd);
          const minAcceptableQugusd = Math.max(0, referenceQugusd * 0.9 - 1);
          if (fetchedQugusd >= minAcceptableQugusd || referenceQugusd === 0) {
            qugUsdBalance = fetchedQugusd;
            if (fetchedQugusd > (highestKnownBalancesRef.current['QUGUSD'] || 0)) {
              highestKnownBalancesRef.current['QUGUSD'] = fetchedQugusd;
            }
            // Cache the new value
            localStorage.setItem('cachedQugusdBalance', fetchedQugusd.toString());
          } else {
            console.warn(`⚠️ Ignoring suspiciously low QUGUSD balance: ${fetchedQugusd} (expected ~${referenceQugusd})`);
            qugUsdBalance = referenceQugusd;
          }
        } else {
          console.warn('⚠️ [Dashboard] Response not successful or missing data, using cached QUGUSD:', qugUsdBalance);
        }
      } catch (error) {
        console.warn('⚠️ Failed to fetch QUGUSD balance, using cached:', qugUsdBalance, error);
      }

      // 🚨 NEVER allow 0 QUGUSD balance if we have cached value
      if (qugUsdBalance === 0 && validCachedQugusd > 0) {
        console.warn('⚠️ [fetchWalletBalances] QUGUSD was 0 but cache has value, using cache:', validCachedQugusd);
        qugUsdBalance = validCachedQugusd;
      }

      // Add QUGUSD to balances
      const qugusdSavedHistory = savedHistory['QUGUSD'] || [];
      const qugusdHistory: BalanceHistoryPoint[] = [
        ...qugusdSavedHistory,
        { timestamp: now, balance: qugUsdBalance }
      ].slice(-20);

      balances.push({
        symbol: 'QUGUSD',
        name: 'Quillon USD',
        balance: qugUsdBalance,
        usdValue: qugUsdBalance, // 1:1 peg to USD
        icon: 'usd',
        color: 'from-blue-400 to-cyan-500',
        history: qugusdHistory
      });

      console.log('📊 [fetchWalletBalances] Initialized QUGUSD with history:', qugusdHistory.length, 'points (', qugusdSavedHistory.length, 'from localStorage), balance:', qugUsdBalance);

      // Fetch USD balance from payment API - ALWAYS show USD wallet
      let usdValue = 0;
      try {
        const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/payment/balance`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ wallet_address: currentWalletAddress }),
        });

        if (response.ok) {
          const data = await response.json();
          if (data.success && data.data) {
            usdValue = parseFloat(data.data.balance_usd || '0');
            setUsdBalance(usdValue);
            console.log('💵 USD balance fetched:', usdValue);
          }
        } else {
          console.warn('⚠️ USD balance fetch failed, showing 0 balance');
        }
      } catch (error) {
        // Silently fail if payment API is not available
        console.warn('⚠️ Payment API not available - USD wallet features disabled');
      }

      // Always add USD wallet (even with 0 balance) so user can see Add/Send controls
      const usdSavedHistory = savedHistory['USD'] || [];
      const usdHistory: BalanceHistoryPoint[] = [
        ...usdSavedHistory,
        { timestamp: now, balance: usdValue }
      ].slice(-20);

      balances.push({
        symbol: 'USD',
        name: 'US Dollar',
        balance: usdValue,
        icon: 'usd',
        color: 'from-green-400 to-emerald-500',
        history: usdHistory
      });

      console.log('📊 [fetchWalletBalances] Initialized USD with history:', usdHistory.length, 'points (', usdSavedHistory.length, 'from localStorage)');

      // Note: Custom tokens would be fetched here if the API supported them
      // Currently, only QUG and QUGUSD are supported in the multi-token balance endpoint

      // Add placeholder for future cryptos
      balances.push(
        {
          symbol: 'ZEC',
          name: 'Zcash (Shielded)',
          balance: 0,
          icon: 'zec',
          color: 'from-yellow-400 to-amber-600',
          comingSoon: true,
          shieldedOnly: true,
        },
        {
          symbol: 'IRON',
          name: 'Iron Fish',
          balance: 0,
          icon: 'iron',
          color: 'from-slate-400 to-zinc-600',
          comingSoon: true,
          shieldedOnly: true,
        },
        {
          symbol: 'BTC',
          name: 'Bitcoin',
          balance: 0,
          icon: 'btc',
          color: 'from-orange-400 to-amber-500',
          comingSoon: true,
        },
        {
          symbol: 'ETH',
          name: 'Ethereum',
          balance: 0,
          icon: 'eth',
          color: 'from-blue-400 to-indigo-500',
          comingSoon: true,
        },
        {
          symbol: 'SOL',
          name: 'Solana',
          balance: 0,
          icon: 'sol',
          color: 'from-violet-400 to-purple-500',
          comingSoon: true,
        }
      );

      // v2.3.31-beta: Check BOTH local ref AND global localStorage cooldown
      const globalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
      const globalCooldownActive = Date.now() < globalCooldownUntil;
      if (dexSwapCooldownRef.current || globalCooldownActive) {
        console.log('🚫 [fetchWalletBalances] SKIPPING setWalletBalances during DEX cooldown (global:', globalCooldownActive, ')');
        return; // Don't overwrite the correct DEX-updated balance with stale API data
      }

      setWalletBalances(balances);

      // 🚨 v2.3.7-beta: SYNC FIX - Notify App.tsx of the QUG balance update
      // This ensures TopBar "Total Balance" stays in sync with Dashboard "My Wallets"
      const qugWallet = balances.find(b => b.symbol === 'QUG');
      if (qugWallet) {
        console.log('🔄 [SYNC] Dispatching balance-update event to App.tsx:', qugWallet.balance);
        window.dispatchEvent(new CustomEvent('balance-update', {
          detail: { balance: qugWallet.balance, source: 'Dashboard.fetchWalletBalances' }
        }));
      }
    };

    const fetchRecentTransactionsCore = async () => {
      console.log('📋 [fetchRecentTransactions] START - Fetching recent transactions...');
      console.log('📋 [fetchRecentTransactions] Mounted status:', mounted);
      console.log('📋 [fetchRecentTransactions] Current wallet:', localStorage.getItem('walletAddress'));
      if (!mounted) {
        console.log('📋 [fetchRecentTransactions] ABORT - Component not mounted');
        return;
      }

      try {
        const response = await qnkAPI.getRecentTransactions(100);
        console.log('📋 Transactions API response:', response);
        if (!mounted) return;

        // Always merge with existing faucet transactions, even if API fails
        setRecentTransactions(prev => {
          console.log('📋 Current transactions before merge:', prev.length);

          // Preserve both faucet and mining transactions (client-side added)
          const preservedTxs = prev.filter(tx =>
            tx.id.startsWith('faucet-') || tx.id.startsWith('mining-')
          );
          console.log('📋 Transactions to preserve (faucet + mining):', preservedTxs.length);

          // If API call failed or returned no data, keep preserved transactions
          if (!response.success || !response.data) {
            console.log('📋 API failed or no data, keeping preserved transactions');
            console.log('📋 API error details:', response.error);

            // Set visible error for user
            setTransactionError(response.error || 'Failed to load transactions');

            return preservedTxs;
          }

          // Clear error on successful load
          setTransactionError(null);

          // Get current wallet address to determine send/receive
          const currentWalletAddress = localStorage.getItem('walletAddress') || '';
          console.log('📋 Current wallet address for comparison:', currentWalletAddress);

          // Transform API data to match frontend Transaction interface
          const burnAddress = '0000000000000000000000000000000000000000000000000000000000000000';
          const transformedTransactions: Transaction[] = response.data
            .filter((tx: any) => {
              // Filter out invalid transactions
              if (!tx.from || !tx.to) return false;
              // Allow burn transactions (to burn address) but not from burn address
              if (tx.from === burnAddress) return false;
              return true;
            })
            .map((tx: any) => {
              // Determine transaction type based on current wallet address
              // Compare both with and without "qnk" prefix for compatibility
              const walletHex = currentWalletAddress?.startsWith('qnk')
                ? currentWalletAddress.substring(3)
                : (currentWalletAddress || '');
              const toHex = tx.to?.startsWith('qnk') ? tx.to.substring(3) : (tx.to || '');
              const fromHex = tx.from?.startsWith('qnk') ? tx.from.substring(3) : (tx.from || '');

              const type: 'receive' | 'send' = toHex === walletHex ? 'receive' : 'send';

              console.log('📋 Transaction type detection:', {
                txTo: tx.to,
                txFrom: tx.from,
                currentWallet: currentWalletAddress,
                toHex,
                fromHex,
                walletHex,
                detectedType: type
              });

              // Convert Unix timestamp to ISO string
              const timestamp = typeof tx.timestamp === 'number'
                ? new Date(tx.timestamp * 1000).toISOString()
                : tx.timestamp;

              // Convert amount from smallest units to QNK (divide by 1,000,000,000 - 9 decimals)
              const amount = typeof tx.amount === 'number'
                ? tx.amount / 1000000000
                : tx.amount;

              // Label burn address as "Nitro Points Purchase"
              const displayTo = toHex === burnAddress ? 'Nitro Points Purchase ⚡' : tx.to;
              const displayFrom = fromHex === burnAddress ? 'Burn Address' : tx.from;

              return {
                id: tx.id || tx.hash,
                type,
                amount,
                from: displayFrom,
                to: displayTo,
                timestamp,
                txHash: tx.hash || tx.id,
              };
            });

          console.log('📋 Transformed API transactions:', transformedTransactions.length);

          // Merge and deduplicate by id
          const allTxs = [...preservedTxs, ...transformedTransactions];
          const uniqueTxs = allTxs.filter((tx, index, self) =>
            index === self.findIndex(t => t.id === tx.id)
          );

          console.log('📋 Final merged transactions:', uniqueTxs.length);

          // Sort by timestamp descending
          return uniqueTxs.sort((a, b) =>
            new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
          );
        });
      } catch (err) {
        console.error('❌ Error fetching transactions:', err);
        // On error, preserve faucet and mining transactions
        setRecentTransactions(prev => prev.filter(tx =>
          tx.id.startsWith('faucet-') || tx.id.startsWith('mining-')
        ));
      }
    };

    const generateWalletAddress = async () => {
      console.log('Generating wallet address...');
      const storedAddress = localStorage.getItem('walletAddress');
      const storedMnemonic = localStorage.getItem('walletSeed');

      // If address exists, always use it (regardless of mnemonic)
      if (storedAddress) {
        console.log('Using stored wallet address:', storedAddress);
        setWalletAddress(storedAddress);
        return;
      }

      if (storedMnemonic && !storedAddress) {
        try {
          const encoder = new TextEncoder();
          const data = encoder.encode(storedMnemonic);
          const hashBuffer = await crypto.subtle.digest('SHA-256', data);
          const hashArray = Array.from(new Uint8Array(hashBuffer));
          const address = 'qnk' + hashArray.map(b => b.toString(16).padStart(2, '0')).join('').substring(0, 40);

          localStorage.setItem('walletAddress', address);
          setWalletAddress(address);
          console.log('Generated wallet from mnemonic');
          return;
        } catch (error) {
          console.error('Failed to derive address from mnemonic:', error);
        }
      }

      try {
        const response = await qnkAPI.generateMnemonic();
        if (response.success && response.data?.mnemonic) {
          const encoder = new TextEncoder();
          const data = encoder.encode(response.data.mnemonic);
          const hashBuffer = await crypto.subtle.digest('SHA-256', data);
          const hashArray = Array.from(new Uint8Array(hashBuffer));
          const address = 'qnk' + hashArray.map(b => b.toString(16).padStart(2, '0')).join('').substring(0, 40);

          localStorage.setItem('walletAddress', address);
          // SECURITY: Do NOT store plaintext mnemonic
          // User must import wallet via LoginScreen with password encryption

          setWalletAddress(address);
          console.log('Generated new wallet address (mnemonic NOT stored - user must import with password)');
        } else {
          const prefix = 'qnk';
          const randomBytes = new Uint8Array(20);
          crypto.getRandomValues(randomBytes);
          const address = prefix + Array.from(randomBytes).map(b => b.toString(16).padStart(2, '0')).join('');

          // CRITICAL FIX: Save fallback wallet to localStorage
          localStorage.setItem('walletAddress', address);
          setWalletAddress(address);
          console.log('Generated fallback wallet and saved to localStorage');
        }
      } catch (error) {
        console.error('Failed to generate wallet address:', error);
        const prefix = 'qnk';
        const randomBytes = new Uint8Array(20);
        crypto.getRandomValues(randomBytes);
        const address = prefix + Array.from(randomBytes).map(b => b.toString(16).padStart(2, '0')).join('');

        // CRITICAL FIX: Save error fallback wallet to localStorage
        localStorage.setItem('walletAddress', address);
        setWalletAddress(address);
        console.log('Generated error fallback wallet and saved to localStorage');
      }
    };

    // ============================================
    // DEBOUNCED WRAPPERS - DISABLED (SSE in App.tsx)
    // ============================================
    // SSE connection moved to App.tsx, so Dashboard no longer needs these
    // const fetchNodeStatus = debounce(fetchNodeStatusCore, 200);
    // const fetchRecentTransactions = debounce(fetchRecentTransactionsCore, 200);
    console.log('ℹ️  [SSE DISABLED] Dashboard no longer uses local SSE - App.tsx handles it');
    // ============================================

    const loadData = async () => {
      console.log('🚀 [loadData] START - Loading dashboard data...');
      setLoading(true);
      try {
        console.log('🚀 [loadData] Step 1: Generating wallet address...');
        await generateWalletAddress();
        console.log('🚀 [loadData] Step 2: Calling fetchNodeStatus and fetchRecentTransactions (DIRECT - bypass debounce)...');
        // For initial load, call core functions directly to bypass debounce
        await Promise.all([fetchNodeStatusCore(), fetchRecentTransactionsCore()]);
        console.log('🚀 [loadData] Step 3: Both API calls completed');
        console.log('🚀 [loadData] Step 4: Fetching wallet balances...');
        await fetchWalletBalances();
        console.log('🚀 [loadData] Step 5: Wallet balances fetched');
      } catch (error) {
        console.error('❌ [loadData] Error loading data:', error);
      } finally {
        setLoading(false);
        console.log('✅ [loadData] COMPLETE - Dashboard data loaded');
      }
    };

    console.log('🎬 [Dashboard useEffect] Calling loadData()...');
    loadData();

    // Set up SSE for real-time balance updates
    // CRITICAL: Pass wallet_address parameter for privacy-filtered SSE
    // const currentWalletForSSE = localStorage.getItem('walletAddress') || ''; // Disabled - App.tsx handles SSE
    // SSE URL construction disabled - App.tsx handles SSE
    // const sseUrl = import.meta.env.VITE_API_URL ?
    //   `${import.meta.env.VITE_API_URL}/v1/events?wallet_address=${encodeURIComponent(currentWalletForSSE)}` :
    //   `/api/v1/events?wallet_address=${encodeURIComponent(currentWalletForSSE)}`;

    // CRITICAL FIX: Dashboard's SSE connection is DISABLED
    // App.tsx already has an SSE connection that handles balance updates
    // Having two SSE connections causes duplicate events and flickering
    // Dashboard will receive balance updates via App.tsx's SSE connection
    console.log('ℹ️  Dashboard SSE disabled - using App.tsx SSE connection instead');

    // Mark as "connected" immediately since App.tsx handles SSE
    if (mounted) {
      setSseConnected(true);
    }

    // Early return - skip all SSE setup since App.tsx handles it
    // The cleanup function below will still run on unmount
    /*
    console.log('📡 Attempting SSE connection to:', sseUrl);
    console.log('📡 SSE wallet filter:', currentWalletForSSE);

    try {
      eventSource = new EventSource(sseUrl);

      eventSource.onopen = () => {
        console.log('✅ SSE connection established');
        console.log('🔍 Current wallet address:', localStorage.getItem('walletAddress'));
        if (mounted) {
          setSseConnected(true);
        }
      };

      // Listen for all possible SSE event types that backend might send
      const handleSpecificEvent = (eventType: string) => (event: MessageEvent) => {
        console.log(`🎯 SSE SPECIFIC EVENT [${eventType}]:`, event);
        console.log(`📨 Event data [${eventType}]:`, event.data);
        if (!mounted) return;

        try {
          const data = JSON.parse(event.data);
          console.log(`📦 Parsed data [${eventType}]:`, data);

          // Handle transaction events - refresh recent activity immediately
          if (eventType === 'transaction-submitted' || eventType === 'transaction-confirmed' || eventType === 'transaction-status') {
            console.log(`🔄 Transaction event received [${eventType}] - refreshing recent activity`);
            fetchRecentTransactions();
            fetchNodeStatus(); // Also refresh balance
            return; // Exit early
          }

          // Handle mining reward events - add to recent activity
          if (eventType === 'mining_reward') {
            console.log(`⛏️ Mining reward event received - adding to recent activity`);
            console.log(`⛏️ Mining reward RAW event:`, event);
            console.log(`⛏️ Mining reward RAW data string:`, event.data);
            console.log(`⛏️ Mining reward PARSED data:`, data);
            console.log(`⛏️ Mining reward data keys:`, Object.keys(data));
            console.log(`⛏️ Mining reward data structure:`, JSON.stringify(data, null, 2));

            // The data is nested: data.data contains the actual mining reward fields
            const miningData = data.data || data;
            console.log(`⛏️ Mining reward miningData:`, miningData);
            console.log(`⛏️ Mining reward miningData.reward_qnk:`, miningData.reward_qnk);
            console.log(`⛏️ Mining reward miningData.nonce:`, miningData.nonce);
            console.log(`⛏️ Mining reward miningData.block_height:`, miningData.block_height);
            console.log(`⛏️ Mining reward miningData.miner_address:`, miningData.miner_address);
            console.log(`⛏️ Mining reward miningData.timestamp:`, miningData.timestamp);
            try {
              const miningTransaction: Transaction = {
                id: `mining-${Date.now()}-${miningData.nonce}`,
                type: 'mining',
                amount: miningData.reward_qnk, // Already in QUG units from backend
                from: 'Mining Reward',
                to: miningData.miner_address,
                timestamp: miningData.timestamp,
                txHash: `mining-${miningData.block_height}-${miningData.nonce}`,
              };
              setRecentTransactions(prev => [miningTransaction, ...prev.slice(0, 49)]); // Keep last 50
              console.log(`⛏️ Mining reward added to recent activity: ${miningData.reward_qnk} QUG`);
            } catch (error) {
              console.error('Failed to process mining reward:', error);
            }
            fetchNodeStatus(); // Refresh balance
            return;
          }

          // Handle balance-updated event
          if (eventType === 'balance-updated') {
            const currentWalletAddress = localStorage.getItem('walletAddress');
            console.log('🔍 WALLET COMPARISON DEBUG v2:', {
              raw_current: currentWalletAddress,
              has_prefix: currentWalletAddress?.startsWith('qnk'),
              raw_event: data.wallet_address
            });

            const currentHex = (currentWalletAddress?.startsWith('qnk')
              ? currentWalletAddress.substring(3)
              : currentWalletAddress)?.toLowerCase();
            const eventHex = data.data?.wallet_address?.toLowerCase() || data.wallet_address?.toLowerCase();

            console.log('🔍 AFTER PROCESSING v2:', {
              currentHex,
              eventHex,
              areEqual: eventHex === currentHex,
              currentLength: currentHex?.length,
              eventLength: eventHex?.length
            });

            console.log('💰 BALANCE UPDATE EVENT v2:', {
              eventType,
              eventData: data,
              eventWallet: eventHex,
              currentWallet: currentHex,
              match: eventHex === currentHex,
              newBalance: data.new_balance || data.data?.new_balance,
              oldBalance: data.old_balance || data.data?.old_balance,
              reason: data.change_reason || data.data?.change_reason
            });

            // CRITICAL FIX: Only apply balance update if wallet addresses EXACTLY match
            // Do NOT accept if currentHex is empty - that would apply ALL balance updates
            if (currentHex && eventHex === currentHex) {
              const newBalance = data.data?.new_balance || data.new_balance;

              console.log('✅ APPLYING BALANCE UPDATE v2:', newBalance);
              setNodeStatus(prev => {
                console.log('🔄 Updating nodeStatus from', prev?.balance, 'to', newBalance);
                return prev ? { ...prev, balance: newBalance } : prev;
              });

              // NOTE: No need to dispatch to App.tsx - it has its own SSE connection

              // Refresh recent transactions to show new activity
              console.log('🔄 Refreshing recent transactions after balance update');
              fetchRecentTransactions();
            } else {
              console.log('❌ Balance update IGNORED (different wallet)', {
                reason: !currentHex ? 'no current wallet hex' : 'wallet address mismatch',
                currentHex,
                eventHex
              });
            }
          } else if (eventType === 'faucet-dispensed') {
            console.log('🚰 FAUCET EVENT:', data);
            console.log('🔄 Refreshing balance via fetchNodeStatus...');
            fetchNodeStatus();
            fetchRecentTransactions();
          } else if (eventType === 'mining_reward') {
            console.log('💎 MINING REWARD EVENT:', data);
            const currentWalletAddress = localStorage.getItem('walletAddress');

            // Normalize wallet addresses for comparison (strip "qnk" prefix and compare hex)
            const currentHex = (currentWalletAddress?.startsWith('qnk')
              ? currentWalletAddress.substring(3)
              : currentWalletAddress)?.toLowerCase();
            const minerHex = (data.miner_address?.startsWith('qnk')
              ? data.miner_address.substring(3)
              : data.miner_address)?.toLowerCase();

            console.log('💎 Mining reward wallet comparison:', {
              currentWallet: currentWalletAddress,
              currentHex,
              minerAddress: data.miner_address,
              minerHex,
              match: currentHex && minerHex === currentHex
            });

            // Check if this mining reward is for the current wallet
            if (currentHex && minerHex === currentHex) {
              console.log('✅ Mining reward for current wallet:', {
                reward: data.reward_qnk,
                nonce: data.nonce,
                blockHeight: data.block_height
              });

              // Refresh balance immediately
              fetchNodeStatus();

              // NOTE: No need to dispatch to App.tsx - it has its own SSE connection

              // Add mining transaction to recent activity
              const miningTx: Transaction = {
                id: `mining-${data.nonce}-${data.block_height}`,
                type: 'receive',
                amount: data.reward_qnk,
                from: `Mining Reward (Block ${data.block_height})`,
                to: data.miner_address,
                timestamp: data.timestamp,
                txHash: `mining-${data.nonce}`,
              };

              setRecentTransactions(prev => [miningTx, ...prev]);

              console.log('💎 Mining reward transaction added to recent activity');
            } else {
              console.log('ℹ️ Mining reward for different wallet, ignoring', {
                reason: !currentHex ? 'no current wallet hex' : 'wallet address mismatch'
              });
            }
          } else if (eventType === 'mining_stats') {
            console.log('📊 MINING STATS EVENT:', data);
            const currentWalletAddress = localStorage.getItem('walletAddress');

            // Normalize wallet addresses for comparison (strip "qnk" prefix and compare hex)
            const currentHex = (currentWalletAddress?.startsWith('qnk')
              ? currentWalletAddress.substring(3)
              : currentWalletAddress)?.toLowerCase();
            const minerHex = (data.miner_address?.startsWith('qnk')
              ? data.miner_address.substring(3)
              : data.miner_address)?.toLowerCase();

            console.log('📊 Mining stats wallet comparison:', {
              currentWallet: currentWalletAddress,
              currentHex,
              minerAddress: data.miner_address,
              minerHex,
              match: currentHex && minerHex === currentHex
            });

            // Check if these stats are for the current wallet
            if (currentHex && minerHex === currentHex) {
              console.log('✅ Mining stats for current wallet:', {
                totalRewards: data.total_rewards,
                totalBlocks: data.total_blocks_found,
                currentBalance: data.current_balance
              });

              // Update balance if provided
              if (data.current_balance !== undefined) {
                setNodeStatus(prev => prev ? { ...prev, balance: data.current_balance } : prev);

                // NOTE: No need to dispatch to App.tsx - it has its own SSE connection
              }
            } else {
              console.log('ℹ️ Mining stats for different wallet, ignoring', {
                reason: !currentHex ? 'no current wallet hex' : 'wallet address mismatch'
              });
            }
          }
        } catch (error) {
          console.error(`❌ Error parsing ${eventType} event:`, error);
        }
      };

      // Add listeners for specific event types
      eventSource.addEventListener('balance-updated', handleSpecificEvent('balance-updated'));
      eventSource.addEventListener('faucet-dispensed', handleSpecificEvent('faucet-dispensed'));
      eventSource.addEventListener('transaction-confirmed', handleSpecificEvent('transaction-confirmed'));
      eventSource.addEventListener('transaction-submitted', handleSpecificEvent('transaction-submitted'));
      eventSource.addEventListener('transaction-status', handleSpecificEvent('transaction-status'));
      eventSource.addEventListener('mining_reward', handleSpecificEvent('mining_reward'));
      eventSource.addEventListener('mining_stats', handleSpecificEvent('mining_stats'));

      console.log('✅ SSE event listeners registered for: balance-updated, faucet-dispensed, transaction-confirmed, transaction-submitted, transaction-status, mining_reward, mining_stats');

      eventSource.onmessage = (event) => {
        console.log('📨 SSE DEFAULT MESSAGE (onmessage):', event);
        console.log('📨 Event type:', event.type);
        console.log('📨 Event data:', event.data);
        console.log('📨 Event lastEventId:', event.lastEventId);
        if (!mounted) return;

        try {
          const data = JSON.parse(event.data);
          console.log('📦 SSE data parsed (onmessage):', data);
          console.log('📦 Data type field:', data.type);
          console.log('📦 Data keys:', Object.keys(data));

          // Handle different SSE event types from the data.type field
          if (data.type === 'balance-updated' && data.data?.new_balance !== undefined) {
            const currentWalletAddress = localStorage.getItem('walletAddress');
            const currentHex = (currentWalletAddress?.startsWith('qnk')
              ? currentWalletAddress.substring(3)
              : currentWalletAddress)?.toLowerCase();
            const eventHex = data.data.wallet_address?.toLowerCase();

            console.log('💰 Dashboard: Balance update SSE event (onmessage) v2:', {
              eventWallet: eventHex,
              currentWallet: currentHex,
              match: eventHex === currentHex,
              newBalance: data.data.new_balance,
              reason: data.data.change_reason
            });

            // CRITICAL FIX: Only apply balance update if wallet addresses EXACTLY match
            // Do NOT accept if currentHex is empty - that would apply ALL balance updates
            if (currentHex && eventHex === currentHex) {
              const changeReason = data.data.change_reason || '';
              const isP2PMiningReward = changeReason === 'p2p_mining_reward' || changeReason === 'pending_mining_reward';

              if (isP2PMiningReward) {
                // P2P mining rewards: ACCUMULATE instead of replace
                // The new_balance from bootstrap is STALE (it doesn't have accumulated balance)
                // Calculate reward amount and ADD to current balance
                const rewardAmount = (data.data.new_balance || 0) - (data.data.old_balance || 0);
                console.log('✅ Dashboard: P2P mining reward - ACCUMULATING:', {
                  rewardAmount,
                  oldBalance: data.data.old_balance,
                  newBalance: data.data.new_balance,
                  reason: changeReason
                });
                setNodeStatus(prev => {
                  if (!prev) return prev;
                  const newBalance = (prev.balance || 0) + rewardAmount;
                  console.log('💰 Dashboard: Balance accumulated:', prev.balance, '+', rewardAmount, '=', newBalance);
                  return { ...prev, balance: newBalance };
                });
              } else {
                // Local mining rewards: use new_balance directly (local RocksDB has correct value)
                console.log('✅ Dashboard: Balance update applied (onmessage):', data.data.new_balance);
                setNodeStatus(prev => prev ? { ...prev, balance: data.data.new_balance } : prev);
              }

              // NOTE: No need to dispatch to App.tsx - it has its own SSE connection

              // Refresh recent transactions to show new activity
              console.log('🔄 Refreshing recent transactions after balance update (onmessage)');
              fetchRecentTransactions();
            } else {
              console.log('❌ Dashboard: Balance update ignored (not for current wallet)', {
                reason: !currentHex ? 'no current wallet hex' : 'wallet address mismatch',
                currentHex,
                eventHex
              });
            }
          } else if (data.type === 'transaction-confirmed' || data.type === 'transaction-submitted') {
            console.log('🔄 Transaction event - refreshing data');
            fetchNodeStatus();
            fetchRecentTransactions();
          } else if (data.type === 'faucet-dispensed') {
            console.log('🚰 Faucet dispensed event (onmessage) - refreshing balance');
            fetchNodeStatus();
            fetchRecentTransactions();
          } else if (data.type === 'Custom' && data.data?.event_type === 'mining_reward') {
            // Mining reward event - data is nested in data.data
            console.log('💎 Mining reward event received:', data);
            const rewardData = data.data.data; // Extract nested reward data
            const currentWalletAddress = localStorage.getItem('walletAddress');

            // Normalize wallet addresses for comparison (strip "qnk" prefix and compare hex)
            const currentHex = (currentWalletAddress?.startsWith('qnk')
              ? currentWalletAddress.substring(3)
              : currentWalletAddress)?.toLowerCase();
            const minerHex = (rewardData.miner_address?.startsWith('qnk')
              ? rewardData.miner_address.substring(3)
              : rewardData.miner_address)?.toLowerCase();

            console.log('💎 Mining reward (Custom) wallet comparison:', {
              currentWallet: currentWalletAddress,
              currentHex,
              minerAddress: rewardData.miner_address,
              minerHex,
              match: currentHex && minerHex === currentHex
            });

            // Check if this mining reward is for the current wallet
            if (currentHex && minerHex === currentHex) {
              console.log('✅ Mining reward for current wallet:', {
                reward: rewardData.reward_qnk,
                newBalance: rewardData.new_balance_qnk,
                nonce: rewardData.nonce
              });

              // Update balance immediately
              setNodeStatus(prev => prev ? { ...prev, balance: rewardData.new_balance_qnk } : prev);

              // NOTE: No need to dispatch to App.tsx - it has its own SSE connection

              // Add mining transaction to recent activity
              const miningTx: Transaction = {
                id: `mining-${rewardData.tx_hash}`,
                type: 'receive',
                amount: rewardData.reward_qnk,
                from: 'Mining Reward (VDF)',
                to: rewardData.miner_address,
                timestamp: rewardData.timestamp,
                txHash: rewardData.tx_hash,
              };

              setRecentTransactions(prev => [miningTx, ...prev]);

              console.log('💎 Mining reward transaction added to recent activity');
            } else {
              console.log('ℹ️ Mining reward (Custom) for different wallet, ignoring', {
                reason: !currentHex ? 'no current wallet hex' : 'wallet address mismatch'
              });
            }
          } else if (data.type === 'new-transaction' && data.transaction) {
            console.log('📬 New transaction via SSE:', data.transaction);
            setRecentTransactions(prev => [data.transaction, ...prev].slice(0, 5));
          } else if (data.type === 'block-confirmed' || data.type === 'consensus-round') {
            console.log('⛓️ Block confirmed - updating status');
            fetchNodeStatus();
          } else {
            console.log('❓ Unknown SSE event type:', data.type);
          }
        } catch (error) {
          console.error('❌ Error processing SSE event:', error);
          console.error('❌ Raw event data:', event.data);
        }
      };

      eventSource.onerror = (error) => {
        console.error('❌ SSE connection error:', error);
        console.log('SSE readyState:', eventSource?.readyState);
        console.log('SSE url:', eventSource?.url);
        if (mounted) {
          setSseConnected(false);
        }
        eventSource?.close();
      };
    } catch (error) {
      console.error('❌ Failed to create SSE connection:', error);
    }

    // Listen for CDP mint events from MintQUGUSDModal
    const handleCDPMint = (event: Event) => {
      if (!mounted) return;

      const customEvent = event as CustomEvent;
      const data = customEvent.detail;

      console.log('💵 CDP Mint Event:', data);

      // Create transaction for Recent Activity
      const cdpTransaction: Transaction = {
        id: `cdp-mint-${data.transaction_id}`,
        type: 'send', // Sending QUG to CDP vault
        amount: data.collateral_amount,
        from: data.wallet_address,
        to: 'CDP Vault (QUGUSD Minting)',
        timestamp: data.timestamp,
        txHash: data.transaction_id,
      };

      // Add to recent transactions
      setRecentTransactions(prev => [cdpTransaction, ...prev]);

      console.log('💵 CDP transaction added to recent activity');
    };

    window.addEventListener('cdp-mint', handleCDPMint);

    return () => {
      mounted = false;
      // eventSource cleanup not needed - SSE disabled, App.tsx handles it
      window.removeEventListener('cdp-mint', handleCDPMint);
    };
    */
  }, []);

  // v2.3.26-beta: Track DEX swap cooldown AND lock the balance value
  const dexSwapCooldownRef = useRef(false);
  const lockedQugBalanceRef = useRef<number | null>(null);

  // v2.3.26-beta: Listen for qug-balance-changed event (from DEX swap) - highest priority
  useEffect(() => {
    const handleQugBalanceChanged = (event: Event) => {
      const customEvent = event as CustomEvent;
      const newBalance = customEvent.detail?.balance;
      if (typeof newBalance === 'number') {
        console.log('🔥 Dashboard: qug-balance-changed - LOCKING QUG balance to:', newBalance);

        // LOCK this balance - it cannot be overwritten for 10 seconds
        lockedQugBalanceRef.current = newBalance;
        dexSwapCooldownRef.current = true;
        setTimeout(() => {
          dexSwapCooldownRef.current = false;
          lockedQugBalanceRef.current = null;
          console.log('🔓 Dashboard: DEX swap cooldown ended, balance unlocked');
        }, 10000);

        // Update tracking and localStorage
        highestKnownBalancesRef.current['QUG'] = newBalance;
        localStorage.setItem('cachedBalance', newBalance.toString());

        // Update wallet balances state
        setWalletBalances(wallets => {
          return wallets.map(wallet => {
            if (wallet.symbol === 'QUG') {
              console.log(`🔥 Dashboard: Updating QUG balance: ${wallet.balance} -> ${newBalance}`);
              return {
                ...wallet,
                balance: newBalance,
                history: [...(wallet.history || []), { timestamp: Date.now(), balance: newBalance }].slice(-20)
              };
            }
            return wallet;
          });
        });
      }
    };

    window.addEventListener('qug-balance-changed', handleQugBalanceChanged);
    return () => window.removeEventListener('qug-balance-changed', handleQugBalanceChanged);
  }, []);

  // v2.3.33-beta: Listen for dex-cooldown-expired to sync walletBalances state from cached values
  // This is CRITICAL: After cooldown expires, walletBalances state needs to be updated with correct values
  useEffect(() => {
    const handleDexCooldownExpired = (event: Event) => {
      const customEvent = event as CustomEvent;
      const { qugBalance, qugusdBalance, source } = customEvent.detail;

      console.log('🔄 Dashboard: Received dex-cooldown-expired event from', source, {
        qugBalance,
        qugusdBalance
      });

      // Update walletBalances state with the cached correct values
      setWalletBalances(wallets => {
        return wallets.map(wallet => {
          if (wallet.symbol === 'QUG' && qugBalance !== null && !isNaN(qugBalance)) {
            console.log(`🔄 Dashboard: Syncing QUG state after cooldown: ${wallet.balance} -> ${qugBalance}`);
            return {
              ...wallet,
              balance: qugBalance,
              history: [...(wallet.history || []), { timestamp: Date.now(), balance: qugBalance }].slice(-20)
            };
          }
          if (wallet.symbol === 'QUGUSD' && qugusdBalance !== null && !isNaN(qugusdBalance)) {
            console.log(`🔄 Dashboard: Syncing QUGUSD state after cooldown: ${wallet.balance} -> ${qugusdBalance}`);
            return {
              ...wallet,
              balance: qugusdBalance,
              history: [...(wallet.history || []), { timestamp: Date.now(), balance: qugusdBalance }].slice(-20)
            };
          }
          return wallet;
        });
      });

      // Also clear local cooldown ref
      dexSwapCooldownRef.current = false;
      lockedQugBalanceRef.current = null;
      console.log('🔓 Dashboard: Cooldown refs cleared after dex-cooldown-expired event');
    };

    window.addEventListener('dex-cooldown-expired', handleDexCooldownExpired);
    return () => window.removeEventListener('dex-cooldown-expired', handleDexCooldownExpired);
  }, []);

  // v2.3.26-beta: Force locked balance on every render during cooldown
  useEffect(() => {
    if (dexSwapCooldownRef.current && lockedQugBalanceRef.current !== null) {
      const lockedBalance = lockedQugBalanceRef.current;
      setWalletBalances(wallets => {
        const qugWallet = wallets.find(w => w.symbol === 'QUG');
        if (qugWallet && qugWallet.balance !== lockedBalance) {
          console.log('🔒 Dashboard: FORCING locked QUG balance:', lockedBalance, '(tried to show:', qugWallet.balance, ')');
          return wallets.map(wallet =>
            wallet.symbol === 'QUG' ? { ...wallet, balance: lockedBalance } : wallet
          );
        }
        return wallets;
      });
    }
  });

  // Listen for real-time balance updates from SSE (via App.tsx custom event)
  useEffect(() => {
    const handleWalletBalanceUpdate = (event: Event) => {
      const customEvent = event as CustomEvent;
      const { symbol, balance: incomingBalance, reason } = customEvent.detail;

      // v2.3.13-beta: DEX swaps are ALWAYS trusted - simplified logic
      const isDexSwap = reason === 'dex-swap-deduct' || reason === 'dex-swap-add';

      // v2.3.31-beta: Check BOTH our ref AND localStorage cooldown (set by DexScreen BEFORE API call)
      const globalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
      const globalCooldownActive = Date.now() < globalCooldownUntil;
      const cooldownActive = dexSwapCooldownRef.current || globalCooldownActive;

      // Block non-DEX updates during cooldown, but ALWAYS allow DEX updates
      if (cooldownActive && !isDexSwap) {
        console.log('🚫 Dashboard: Ignoring non-DEX wallet-balance-updated during cooldown:', symbol, incomingBalance, '(global:', globalCooldownActive, ')');
        return;
      }

      console.log(`💰 Dashboard: Received wallet-balance-updated event for ${symbol}:`, incomingBalance, 'Reason:', reason, 'isDexSwap:', isDexSwap);

      // v2.3.13-beta: For DEX swaps, IMMEDIATELY update everything without validation
      // This is the ONLY way to ensure the correct balance is displayed
      if (isDexSwap) {
        console.log(`🔥 Dashboard: DEX SWAP - Force updating ${symbol} to ${incomingBalance}`);

        // Immediately update all tracking refs and storage
        highestKnownBalancesRef.current[symbol] = incomingBalance;
        if (symbol === 'QUG') {
          localStorage.setItem('cachedBalance', incomingBalance.toString());
        } else if (symbol === 'QUGUSD') {
          localStorage.setItem('cachedQugusdBalance', incomingBalance.toString());
        }

        // Immediately update wallet balances state
        setWalletBalances(wallets => {
          return wallets.map(wallet => {
            if (wallet.symbol === symbol) {
              console.log(`🔥 Dashboard: DEX SWAP updating ${symbol} balance: ${wallet.balance} -> ${incomingBalance}`);
              return {
                ...wallet,
                balance: incomingBalance,
                history: [...(wallet.history || []), { timestamp: Date.now(), balance: incomingBalance }].slice(-20)
              };
            }
            return wallet;
          });
        });

        // Also update balance history state
        setBalanceHistory(prev => {
          const history = prev[symbol] || [];
          const newPoint: BalanceHistoryPoint = { timestamp: Date.now(), balance: incomingBalance };
          const updatedHistory = [...history, newPoint].slice(-20);
          try {
            localStorage.setItem('walletBalanceHistory', JSON.stringify({ ...prev, [symbol]: updatedHistory }));
          } catch {}
          return { ...prev, [symbol]: updatedHistory };
        });

        return; // Skip all other logic for DEX swaps
      }

      // For non-DEX updates, apply anti-fraud validation
      const previousHighest = highestKnownBalancesRef.current[symbol] || 0;
      const cachedBalance = symbol === 'QUG'
        ? parseFloat(localStorage.getItem('cachedBalance') || '0')
        : symbol === 'QUGUSD'
          ? parseFloat(localStorage.getItem('cachedQugusdBalance') || '0')
          : 0;
      const referenceBalance = Math.max(previousHighest, cachedBalance);
      const minAcceptable = Math.max(0, referenceBalance * 0.9 - 1);
      let validatedBalance = incomingBalance;

      if (incomingBalance < minAcceptable && referenceBalance > 0) {
        console.warn(`⚠️ Dashboard: Rejecting suspicious balance update for ${symbol}: ${incomingBalance} (expected ~${referenceBalance})`);
        validatedBalance = referenceBalance;
      } else if (incomingBalance > previousHighest) {
        highestKnownBalancesRef.current[symbol] = incomingBalance;
        // v2.3.27-beta: Don't write to localStorage during DEX cooldown
        if (symbol === 'QUG' && !dexSwapCooldownRef.current) {
          localStorage.setItem('cachedBalance', incomingBalance.toString());
        }
      }

      console.log(`💰 Dashboard: Non-DEX update for ${symbol}:`, incomingBalance, '-> validated:', validatedBalance);

      // Update balance history and wallet balance atomically
      setBalanceHistory(prev => {
        const history = prev[symbol] || [];
        const newPoint: BalanceHistoryPoint = {
          timestamp: Date.now(),
          balance: validatedBalance
        };
        const updatedHistory = [...history, newPoint].slice(-20); // Keep last 20 points
        const newHistoryState = { ...prev, [symbol]: updatedHistory };

        // Save to localStorage
        try {
          localStorage.setItem('walletBalanceHistory', JSON.stringify(newHistoryState));
        } catch (error) {
          console.warn('Failed to save balance history to localStorage:', error);
        }

        console.log(`📊 Dashboard: Updated history for ${symbol}:`, updatedHistory.length, 'points');

        // Update walletBalances with the new history
        setWalletBalances(wallets => {
          return wallets.map(wallet => {
            if (wallet.symbol === symbol) {
              console.log(`✅ Dashboard: Updating ${symbol} balance from ${wallet.balance} to ${validatedBalance} with ${updatedHistory.length} history points`);
              return {
                ...wallet,
                balance: validatedBalance,
                history: updatedHistory
              };
            }
            return wallet;
          });
        });

        return newHistoryState;
      });
    };

    window.addEventListener('wallet-balance-updated', handleWalletBalanceUpdate);
    console.log('👂 Dashboard: Listening for wallet-balance-updated events');

    return () => {
      window.removeEventListener('wallet-balance-updated', handleWalletBalanceUpdate);
      console.log('🔇 Dashboard: Stopped listening for wallet-balance-updated events');
    };
  }, []);

  // Listen for loan approval events from SSE (via App.tsx custom event)
  useEffect(() => {
    const handleLoanApproval = (event: Event) => {
      const customEvent = event as CustomEvent;
      const loanData = customEvent.detail;

      console.log('🏦 Dashboard: Received loan-approved event:', loanData);

      // Store loan details and show approval modal
      setApprovedLoanDetails(loanData);
      setShowLoanApprovalModal(true);
    };

    window.addEventListener('loan-approved', handleLoanApproval);
    console.log('👂 Dashboard: Listening for loan-approved events');

    return () => {
      window.removeEventListener('loan-approved', handleLoanApproval);
      console.log('🔇 Dashboard: Stopped listening for loan-approved events');
    };
  }, []);

  // Refresh wallet balances when refreshTrigger changes
  useEffect(() => {
    if (refreshTrigger > 0) {
      console.log('🔄 Refreshing wallet balances...');
      const refresh = async () => {
        const currentWalletAddress = localStorage.getItem('walletAddress');
        if (!currentWalletAddress) return;

        // CRITICAL FIX: Use best known balance, not just nodeStatus
        const cachedBalance = localStorage.getItem('cachedBalance');
        const cachedValue = cachedBalance ? parseFloat(cachedBalance) : 0;
        const previousHighest = highestKnownBalancesRef.current['QUG'] || 0;
        const nodeBalance = nodeStatus?.balance || 0;
        // Use the maximum of all known sources
        const qugBalance = Math.max(previousHighest, cachedValue, nodeBalance);
        console.log('🔄 Refresh using best balance:', qugBalance, '(highest:', previousHighest, ', cached:', cachedValue, ', node:', nodeBalance, ')');

        const now = Date.now();
        const qugHistory: BalanceHistoryPoint[] = [
          { timestamp: now - 60000, balance: qugBalance }, // 1 minute ago
          { timestamp: now, balance: qugBalance }
        ];

        const balances: WalletBalance[] = [
          {
            symbol: 'QUG',
            name: 'Quillon Graph',
            balance: qugBalance,
            icon: 'qug',
            color: 'from-amber-400 to-yellow-500',
            history: qugHistory  // Add history directly
          }
        ];

        console.log('📊 Initialized QUG with balance history:', qugHistory.length, 'points');

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
              setUsdBalance(usdValue);

              const usdHistory: BalanceHistoryPoint[] = [
                { timestamp: now - 60000, balance: usdValue },
                { timestamp: now, balance: usdValue }
              ];

              balances.push({
                symbol: 'USD',
                name: 'US Dollar',
                balance: usdValue,
                icon: 'usd',
                color: 'from-green-400 to-emerald-500',
                history: usdHistory  // Add history directly
              });

              console.log('📊 Initialized USD with balance history:', usdHistory.length, 'points');
            }
          }
        } catch (error) {
          // Silently fail if payment API is not available
          console.warn('⚠️ Payment API not available - USD wallet features disabled');
        }

        // 🚨 v2.3.7-beta: Fetch QUGUSD balance with cache fallback
        const cachedQugusd = localStorage.getItem('cachedQugusdBalance');
        const cachedQugusdVal = cachedQugusd ? parseFloat(cachedQugusd) : 0;
        const previousHighestQugusd = highestKnownBalancesRef.current['QUGUSD'] || 0;
        let qugusdBalance = Math.max(previousHighestQugusd, cachedQugusdVal);

        try {
          const response = await qnkAPI.getMultiTokenBalance();
          if (response.success && response.data && response.data.tokens) {
            // API returns tokens as object with uppercase keys: { QUG: {...}, QUGUSD: {...} }
            const tokensObj = response.data.tokens;

            // Try to get QUGUSD balance
            let fetchedQugusd = 0;
            if (tokensObj.QUGUSD && tokensObj.QUGUSD.balance_base_units > 0) {
              fetchedQugusd = tokensObj.QUGUSD.balance_base_units / 1e24;
            } else if (tokensObj.qugusd && tokensObj.qugusd.balance !== undefined) {
              fetchedQugusd = parseFloat(tokensObj.qugusd.balance) || 0;
            }

            // Only use fetched if it's valid
            const referenceQugusd = Math.max(previousHighestQugusd, cachedQugusdVal);
            const minAcceptable = Math.max(0, referenceQugusd * 0.9 - 1);
            if (fetchedQugusd >= minAcceptable || referenceQugusd === 0) {
              qugusdBalance = fetchedQugusd;
              if (fetchedQugusd > 0) {
                localStorage.setItem('cachedQugusdBalance', fetchedQugusd.toString());
              }
            }
          }
        } catch (error) {
          console.warn('⚠️ Failed to fetch QUGUSD in refresh, using cached:', qugusdBalance);
        }

        // Always add QUGUSD if we have a balance (cached or fetched)
        if (qugusdBalance > 0) {
          const qugusdHistory: BalanceHistoryPoint[] = [
            { timestamp: now - 60000, balance: qugusdBalance },
            { timestamp: now, balance: qugusdBalance }
          ];

          balances.push({
            symbol: 'QUGUSD',
            name: 'Quillon USD',
            balance: qugusdBalance,
            usdValue: qugusdBalance,
            icon: 'usd' as const,
            color: 'from-blue-400 to-cyan-500',
            history: qugusdHistory
          });
          console.log('📊 Initialized QUGUSD with balance:', qugusdBalance, 'history:', qugusdHistory.length, 'points');
        }

        // Add placeholders
        balances.push(
          {
            symbol: 'ZEC',
            name: 'Zcash (Shielded)',
            balance: 0,
            icon: 'zec',
            color: 'from-yellow-400 to-amber-600',
            comingSoon: true,
            shieldedOnly: true,
          },
          {
            symbol: 'IRON',
            name: 'Iron Fish',
            balance: 0,
            icon: 'iron',
            color: 'from-slate-400 to-zinc-600',
            comingSoon: true,
            shieldedOnly: true,
          },
          {
            symbol: 'BTC',
            name: 'Bitcoin',
            balance: 0,
            icon: 'btc',
            color: 'from-orange-400 to-amber-500',
            comingSoon: true,
          },
          {
            symbol: 'ETH',
            name: 'Ethereum',
            balance: 0,
            icon: 'eth',
            color: 'from-blue-400 to-indigo-500',
            comingSoon: true,
          },
          {
            symbol: 'SOL',
            name: 'Solana',
            balance: 0,
            icon: 'sol',
            color: 'from-violet-400 to-purple-500',
            comingSoon: true,
          }
        );

        // v2.3.31-beta: Check BOTH local ref AND global localStorage cooldown
        const refreshGlobalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
        const refreshGlobalCooldownActive = Date.now() < refreshGlobalCooldownUntil;
        if (dexSwapCooldownRef.current || refreshGlobalCooldownActive) {
          console.log('🚫 [refreshTrigger] SKIPPING setWalletBalances during DEX cooldown (global:', refreshGlobalCooldownActive, ')');
          return; // Don't overwrite the correct DEX-updated balance
        }

        setWalletBalances(balances);

        // 🚨 v2.3.7-beta: SYNC FIX - Notify App.tsx of the QUG balance update
        const qugWallet = balances.find(b => b.symbol === 'QUG');
        if (qugWallet) {
          console.log('🔄 [SYNC] Dispatching balance-update from refresh:', qugWallet.balance);
          window.dispatchEvent(new CustomEvent('balance-update', {
            detail: { balance: qugWallet.balance, source: 'Dashboard.refreshTrigger' }
          }));
        }
      };

      refresh();
    }
  }, [refreshTrigger, nodeStatus?.balance]);

  // v3.0.6-beta: Updated to show more decimals for small amounts
  const formatBalance = (amount: number, hidden = false) => {
    if (hidden) return '••••••••';
    // For very small amounts, show more decimal places
    const maxDecimals = amount > 0 && amount < 0.00000001 ? 16 : 8;
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: maxDecimals,
    }).format(amount);
  };

  // Smart filtering and sorting logic
  const filteredAndSortedTransactions = (() => {
    let filtered = [...recentTransactions];

    // Apply type filter
    if (filterType !== 'all') {
      filtered = filtered.filter(tx => tx.type === filterType);
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let comparison = 0;

      if (sortBy === 'date') {
        comparison = new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
      } else if (sortBy === 'amount') {
        comparison = a.amount - b.amount;
      }

      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return filtered;
  })();

  // Pagination
  const totalPages = Math.ceil(filteredAndSortedTransactions.length / itemsPerPage);
  const paginatedTransactions = filteredAndSortedTransactions.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  // Reset to first page when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [filterType, sortBy, sortOrder]);

  const copyWalletAddress = () => {
    navigator.clipboard.writeText(walletAddress);
    setCopiedAddress(true);
    setTimeout(() => setCopiedAddress(false), 2000);
  };

  const requestFaucetTokens = async () => {
    setFaucetLoading(true);
    setFaucetMessage('');

    try {
      const currentWalletAddress = localStorage.getItem('walletAddress');
      if (!currentWalletAddress) {
        setFaucetMessage('Error: No wallet address found');
        setFaucetLoading(false);
        return;
      }

      const result = await qnkAPI.requestFaucet(currentWalletAddress);

      if (result.success) {
        const receivedAmount = result.data?.amount_qnk || result.data?.new_balance_qnk || 10;

        if (result.data?.new_balance_qnk) {
          setNodeStatus(prev => prev ? {...prev, balance: result.data.new_balance_qnk} : prev);
        }

        // NOTE: No need to dispatch to App.tsx - it has its own SSE connection

        // Add faucet transaction to recent activity
        const faucetTransaction: Transaction = {
          id: `faucet-${Date.now()}`,
          type: 'receive',
          amount: receivedAmount,
          from: 'Faucet',
          to: currentWalletAddress,
          timestamp: new Date().toISOString(),
          txHash: result.data?.tx_hash || `faucet-${Date.now()}`
        };
        setRecentTransactions(prev => [faucetTransaction, ...prev]);
      } else {
        setFaucetMessage(result.error || 'Faucet request failed');
      }
    } catch (error) {
      setFaucetMessage('Network error: Could not connect to faucet');
      console.error('Faucet error:', error);
    } finally {
      setFaucetLoading(false);
    }
  };

  // Handle Loan Payback
  const handleLoanPayback = (loanId: string) => {
    console.log('💰 Opening loan payback modal for loan:', loanId);
    setSelectedLoanId(loanId);
    setShowLoanPaybackModal(true);
  };

  // Handle Add USD - show Stripe checkout
  const handleAddUSD = async () => {
    if (!usdAmount || parseFloat(usdAmount) <= 0) {
      setStripeError('Please enter a valid amount');
      return;
    }

    const currentWalletAddress = localStorage.getItem('walletAddress');
    if (!currentWalletAddress) {
      setStripeError('No wallet address found');
      return;
    }

    // Show the Stripe checkout component
    setShowStripeCheckout(true);
  };

  // Handle Send USD
  const handleSendUSD = async () => {
    if (!usdAmount || parseFloat(usdAmount) <= 0) {
      setStripeError('Please enter a valid amount');
      return;
    }

    if (!usdRecipient) {
      setStripeError('Please enter a recipient wallet address');
      return;
    }

    const currentWalletAddress = localStorage.getItem('walletAddress');
    if (!currentWalletAddress) {
      setStripeError('No wallet address found');
      return;
    }

    setStripeLoading(true);
    setStripeError(null);

    try {
      // This would call a USD transfer API endpoint (to be implemented)
      const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/payment/transfer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          from_wallet: currentWalletAddress,
          to_wallet: usdRecipient,
          amount_usd: usdAmount,
        }),
      });

      const data = await response.json();

      if (data.success) {
        console.log('✅ USD sent successfully');
        alert(`Successfully sent $${usdAmount} USD to ${usdRecipient}`);

        // Close modal and refresh balance
        setIsSendUSDModalOpen(false);
        setUsdAmount('');
        setUsdRecipient('');
        setRefreshTrigger(prev => prev + 1);
      } else {
        setStripeError(data.error || 'Failed to send USD');
      }
    } catch (error) {
      console.error('❌ Failed to send USD:', error);
      setStripeError('Network error: Could not send USD');
    } finally {
      setStripeLoading(false);
    }
  };

  console.log('Dashboard rendering, loading:', loading, 'error:', error, 'nodeStatus:', nodeStatus);

  if (loading) {
    return (
      <div className="space-y-8 animate-pulse">
        <div className="h-8 bg-quantum-indigo/30 rounded w-64"></div>
        <div className="h-64 bg-quantum-indigo/30 rounded-3xl"></div>
        <div className="grid grid-cols-1 gap-8">
          <div className="h-64 bg-quantum-indigo/30 rounded-3xl"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-8">
        <div className="bg-quantum-pink/20 border border-quantum-pink/50 rounded-3xl p-8 text-center">
          <AlertCircle className="w-12 h-12 text-quantum-pink mx-auto mb-4" />
          <h2 className="text-xl font-bold text-quantum-pink mb-2">Node Connection Error</h2>
          <p className="text-gray-400 mb-4">{error}</p>
          <p className="text-sm text-gray-500">
            Please ensure the Q-NarwhalKnight node is running and accessible at the configured API endpoint.
          </p>
        </div>
      </div>
    );
  }

  if (!nodeStatus) {
    return (
      <div className="space-y-8">
        <div className="bg-quantum-yellow/20 border border-quantum-yellow/50 rounded-3xl p-8 text-center">
          <AlertCircle className="w-12 h-12 text-quantum-yellow mx-auto mb-4" />
          <h2 className="text-xl font-bold text-quantum-yellow mb-2">No Node Data</h2>
          <p className="text-gray-400">Unable to retrieve node status</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Phase 8 Transition Modal */}
      {showPhaseModal && (
        <PhaseTransitionModal
          onClose={() => {
            setShowPhaseModal(false);
            localStorage.setItem('v0978betaModalSeen', 'true');
          }}
        />
      )}

      {/* QNO Staking Modal */}
      <StakingModal
        isOpen={showStakingModal}
        onClose={() => setShowStakingModal(false)}
        availableBalance={walletBalances.find(w => w.symbol === 'QUG')?.balance || 0}
        walletAddress={walletAddress}
        onStakeSuccess={() => {
          setRefreshTrigger(prev => prev + 1);
        }}
      />

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl lg:text-4xl font-bold text-white">Dashboard</h1>
          <p className="text-gray-400 mt-1 flex items-center gap-2">
            Quantum Consensus Wallet
            {sseConnected && (
              <span className="inline-flex items-center gap-1 text-xs text-quantum-green">
                <span className="w-2 h-2 rounded-full bg-quantum-green animate-pulse"></span>
                Live Updates
              </span>
            )}
          </p>
        </div>
        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-quantum-purple to-quantum-cyan flex items-center justify-center">
          <Activity className={`w-6 h-6 text-white ${nodeStatus.network_health === 'healthy' ? 'animate-pulse' : 'opacity-50'}`} />
        </div>
      </div>

      {/* Multi-Wallet Card */}
      <motion.div
        className="backdrop-blur-xl rounded-3xl p-6 relative overflow-hidden"
        style={{
          background: 'linear-gradient(135deg, rgba(30, 20, 60, 0.9) 0%, rgba(50, 30, 80, 0.9) 100%)',
          border: '2px solid rgba(212, 175, 55, 0.3)',
          boxShadow: '0 0 30px rgba(212, 175, 55, 0.2), inset 0 0 20px rgba(212, 175, 55, 0.1)'
        }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        {/* Animated shimmer effect */}
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-transparent via-amber-500/10 to-transparent"
          initial={{ x: '-100%' }}
          animate={{ x: '100%' }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        />

        <div className="relative space-y-6">
          {/* Wallet Address Section */}
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div
                className="p-3 rounded-xl"
                style={{
                  background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(255, 215, 0, 0.15))',
                  border: '2px solid rgba(212, 175, 55, 0.3)'
                }}
              >
                <Wallet className="w-6 h-6 text-amber-400" />
              </div>
              <div>
                <h3 className="text-lg font-bold bg-gradient-to-r from-amber-400 via-yellow-500 to-amber-600 bg-clip-text text-transparent mb-2">
                  Wallet Address
                </h3>
                <div className="font-mono text-xs text-amber-100 break-all max-w-md">
                  {walletAddress || 'Generating...'}
                </div>
              </div>
            </div>

            <div className="flex gap-2 flex-shrink-0">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setIsQRModalOpen(true)}
                disabled={!walletAddress}
                className="p-3 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                style={{
                  background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(255, 215, 0, 0.15))',
                  border: '2px solid rgba(212, 175, 55, 0.3)'
                }}
                title="Show QR Code"
              >
                <QrCode className="w-5 h-5 text-amber-400" />
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setIsNodeInfoModalOpen(true)}
                className="p-3 rounded-xl transition-colors"
                style={{
                  background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(37, 99, 235, 0.15))',
                  border: '2px solid rgba(59, 130, 246, 0.3)'
                }}
                title="Node Information"
              >
                <Info className="w-5 h-5 text-blue-400" />
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={generateAIReport}
                className="p-3 rounded-xl transition-colors group relative"
                style={{
                  background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(147, 51, 234, 0.15))',
                  border: '2px solid rgba(168, 85, 247, 0.3)'
                }}
                title="AI Wallet Analysis"
              >
                <img
                  src="/quantum-ai-logo.png"
                  alt="AI Report"
                  className="w-5 h-5 object-contain"
                  style={{
                    filter: 'drop-shadow(0 0 8px rgba(168, 85, 247, 0.5))'
                  }}
                />
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setShowFinanceModal(true)}
                className="p-3 rounded-xl transition-colors group relative"
                style={{
                  background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(20, 184, 166, 0.15))',
                  border: '2px solid rgba(6, 182, 212, 0.3)'
                }}
                title="K-Law Financial Intelligence"
              >
                <BarChart3 className="w-5 h-5 text-cyan-400" style={{ filter: 'drop-shadow(0 0 8px rgba(6, 182, 212, 0.5))' }} />
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={copyWalletAddress}
                disabled={!walletAddress}
                className="p-3 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                style={{
                  background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(255, 215, 0, 0.15))',
                  border: '2px solid rgba(212, 175, 55, 0.3)'
                }}
                title="Copy Address"
              >
                {copiedAddress ? (
                  <Check className="w-5 h-5 text-green-400" />
                ) : (
                  <Copy className="w-5 h-5 text-amber-400" />
                )}
              </motion.button>
            </div>
          </div>

          {/* Multi-Wallet Balances */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-bold text-white">My Wallets</h3>
              <div className="flex items-center gap-2">
                {/* Apply for Loan Button */}
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => {
                    console.log('🏦 Apply for Loan button clicked - opening modal');
                    setShowLoanModal(true);
                  }}
                  className="px-4 py-2 rounded-xl transition-colors text-sm font-medium flex items-center gap-2"
                  style={{
                    background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(147, 51, 234, 0.15))',
                    border: '2px solid rgba(168, 85, 247, 0.3)',
                    color: 'rgb(192, 132, 252)'
                  }}
                >
                  <DollarSign className="w-4 h-4" />
                  Apply for Loan
                </motion.button>

                {nodeStatus && nodeStatus.balance === 0 && (
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={requestFaucetTokens}
                    disabled={faucetLoading}
                    className="px-4 py-2 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium flex items-center gap-2"
                    style={{
                      background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(22, 163, 74, 0.15))',
                      border: '2px solid rgba(34, 197, 94, 0.3)',
                      color: 'rgb(74, 222, 128)'
                    }}
                  >
                    {faucetLoading ? (
                      <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: "linear" }}>
                        <Coins className="w-4 h-4" />
                      </motion.div>
                    ) : (
                      <Coins className="w-4 h-4" />
                    )}
                    Get Test Tokens
                  </motion.button>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {walletBalances.map((wallet, index) => {
                // v2.3.29-beta: For QUG, check if we have a locked balance from DEX
                let displayWallet = wallet;
                if (wallet.symbol === 'QUG') {
                  const lockedBalance = localStorage.getItem('dexLockedBalance');
                  const cooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
                  if (lockedBalance && Date.now() < cooldownUntil) {
                    const locked = parseFloat(lockedBalance);
                    if (!isNaN(locked) && isFinite(locked)) {
                      console.log('🔒 Dashboard: Using LOCKED QUG balance:', locked);
                      displayWallet = { ...wallet, balance: locked };
                    }
                  }
                }
                return (
                  <WalletCardWithGraph
                    key={wallet.symbol}
                    wallet={displayWallet}
                    index={index}
                    isAnimating={balanceAnimations[wallet.symbol] || false}
                    onCardClick={!wallet.comingSoon && wallet.symbol !== 'USD' && onNavigateToSend ? () => onNavigateToSend(wallet.symbol) : undefined}
                  >
                    {/* USD Action Buttons */}
                    {!wallet.comingSoon && wallet.symbol === 'USD' && (
                      <div className="flex gap-2">
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            setIsAddUSDModalOpen(true);
                            setStripeError(null);
                          }}
                          className="flex-1 py-2 px-3 rounded-lg text-xs font-medium bg-green-500/20 border border-green-500/30 text-green-300 flex items-center justify-center gap-1"
                          title="Add USD"
                        >
                          <Plus className="w-3 h-3" />
                          Add
                        </motion.button>
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            if (onNavigateToSend) {
                              onNavigateToSend('USD');
                            }
                          }}
                          className="flex-1 py-2 px-3 rounded-lg text-xs font-medium bg-blue-500/20 border border-blue-500/30 text-blue-300 flex items-center justify-center gap-1"
                          title="Send USD"
                        >
                          <Send className="w-3 h-3" />
                          Send
                        </motion.button>
                      </div>
                    )}

                    {/* Send and Stake Buttons for QUG wallet */}
                    {!wallet.comingSoon && wallet.symbol === 'QUG' && (
                      <div className="flex gap-2">
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            if (onNavigateToSend) {
                              onNavigateToSend(wallet.symbol);
                            }
                          }}
                          className="flex-1 py-2 px-3 rounded-lg text-xs font-medium bg-blue-500/20 border border-blue-500/30 text-blue-300 flex items-center justify-center gap-1"
                          title="Send"
                        >
                          <Send className="w-3 h-3" />
                          Send
                        </motion.button>
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            setShowStakingModal(true);
                          }}
                          className="flex-1 py-2 px-3 rounded-lg text-xs font-medium bg-purple-500/20 border border-purple-500/30 text-purple-300 flex items-center justify-center gap-1"
                          title="Stake for QNO Predictions"
                        >
                          <Zap className="w-3 h-3" />
                          Stake
                        </motion.button>
                      </div>
                    )}

                    {/* Send Button for QUGUSD wallet */}
                    {!wallet.comingSoon && wallet.symbol === 'QUGUSD' && (
                      <div className="flex gap-2">
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            if (onNavigateToSend) {
                              onNavigateToSend(wallet.symbol);
                            }
                          }}
                          className="flex-1 py-2 px-3 rounded-lg text-xs font-medium bg-blue-500/20 border border-blue-500/30 text-blue-300 flex items-center justify-center gap-1"
                          title="Send"
                        >
                          <Send className="w-3 h-3" />
                          Send
                        </motion.button>
                      </div>
                    )}
                  </WalletCardWithGraph>
                );
              })}
            </div>
          </div>

          {/* Messages */}
          {faucetMessage && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`p-3 rounded-xl text-sm ${
                faucetMessage.startsWith('Success')
                  ? 'bg-quantum-green/20 text-quantum-green border border-quantum-green/30'
                  : 'bg-quantum-pink/20 text-quantum-pink border border-quantum-pink/30'
              }`}
            >
              {faucetMessage}
            </motion.div>
          )}

          {transactionError && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-3 rounded-xl text-sm bg-red-500/20 text-red-400 border border-red-500/30"
            >
              <div className="font-semibold mb-1">⚠️ Failed to load transaction history</div>
              <div className="text-xs">{transactionError}</div>
              <div className="text-xs mt-2 opacity-80">
                💡 Tip: This usually means you need to log in with your wallet password. Go to Settings → Login/Import Wallet.
              </div>
            </motion.div>
          )}
        </div>
      </motion.div>

      {/* Active Loans Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35 }}
      >
        <ActiveLoansCard onPayback={handleLoanPayback} />
      </motion.div>

      {/* Custom Tokens Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.37 }}
      >
        <CustomTokensCard onSendToken={(symbol, contractAddress) => {
          // When user clicks send on a custom token, navigate to send screen
          // You can enhance this to pass the contract address as well
          if (onNavigateToSend) {
            // Store the contract address in localStorage for the send screen to use
            localStorage.setItem('selectedTokenContract', contractAddress);
            onNavigateToSend(symbol);
          }
        }} />
      </motion.div>

      {/* DAG-Knight Consensus Visualization */}
      <motion.div
        className="mb-8"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <DAGKnightVisualization currentHeight={nodeStatus?.current_height || 0} />
      </motion.div>

      {/* QNO Oracle Resolution Visualization */}
      <motion.div
        className="mb-8"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <div className="backdrop-blur-xl rounded-3xl overflow-hidden"
          style={{
            background: 'linear-gradient(135deg, rgba(30, 20, 60, 0.9) 0%, rgba(50, 30, 80, 0.9) 100%)',
            border: '2px solid rgba(139, 92, 246, 0.3)',
            boxShadow: '0 0 30px rgba(139, 92, 246, 0.1)'
          }}
        >
          <div className="p-4 border-b border-purple-500/20">
            <h3 className="text-lg font-semibold text-purple-100 flex items-center gap-2">
              <span className="text-xl">🔮</span>
              QNO Oracle Resolution Monitor
            </h3>
            <p className="text-sm text-purple-300/60 mt-1">
              Real-time prediction staking outcomes, oracle feeds, and resolution events
            </p>
          </div>
          <QNOOracleVisualization />
        </div>
      </motion.div>

      <div className="grid grid-cols-1 gap-8">
        {/* Recent Activity */}
        <motion.div
          className="backdrop-blur-xl rounded-3xl p-8"
          style={{
            background: 'linear-gradient(135deg, rgba(30, 20, 60, 0.9) 0%, rgba(50, 30, 80, 0.9) 100%)',
            border: '2px solid rgba(212, 175, 55, 0.2)',
            boxShadow: '0 0 30px rgba(212, 175, 55, 0.1)'
          }}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          {/* Header with Filters */}
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 mb-6">
            <div className="flex items-center gap-3">
              <Zap className="w-6 h-6 text-amber-400" />
              <h3 className="text-xl font-semibold text-amber-100">Recent Activity</h3>
              <span className="text-sm text-amber-300/60">
                {filteredAndSortedTransactions.length} transaction{filteredAndSortedTransactions.length !== 1 ? 's' : ''}
              </span>
            </div>

            {/* Filter Controls */}
            <div className="flex flex-wrap items-center gap-3">
              {/* Type Filter */}
              <div className="flex items-center gap-2 p-1 rounded-lg"
                style={{
                  background: 'rgba(30, 20, 60, 0.7)',
                  border: '1px solid rgba(212, 175, 55, 0.2)'
                }}
              >
                {(['all', 'receive', 'send', 'mining'] as const).map((type) => (
                  <motion.button
                    key={type}
                    onClick={() => setFilterType(type)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      filterType === type ? 'text-slate-900' : 'text-amber-300/60 hover:text-amber-200'
                    }`}
                    style={filterType === type ? {
                      background: 'linear-gradient(135deg, #D4AF37, #FFD700)',
                      boxShadow: '0 0 15px rgba(212, 175, 55, 0.3)'
                    } : {}}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    {type === 'all' ? 'All' : type === 'receive' ? '↓ Received' : type === 'send' ? '↑ Sent' : '⛏️ Mining'}
                  </motion.button>
                ))}
              </div>

              {/* Sort Controls */}
              <div className="flex items-center gap-2">
                <motion.button
                  onClick={() => setSortBy(sortBy === 'date' ? 'amount' : 'date')}
                  className="px-4 py-2 rounded-lg text-sm font-medium text-amber-100 flex items-center gap-2"
                  style={{
                    background: 'rgba(212, 175, 55, 0.15)',
                    border: '1px solid rgba(212, 175, 55, 0.3)'
                  }}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {sortBy === 'date' ? <Calendar className="w-4 h-4" /> : <DollarSign className="w-4 h-4" />}
                  {sortBy === 'date' ? 'Date' : 'Amount'}
                </motion.button>

                <motion.button
                  onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                  className="px-3 py-2 rounded-lg text-amber-100"
                  style={{
                    background: 'rgba(212, 175, 55, 0.15)',
                    border: '1px solid rgba(212, 175, 55, 0.3)'
                  }}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {sortOrder === 'asc' ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                </motion.button>
              </div>
            </div>
          </div>

          {/* Transaction List */}
          <div className="space-y-3">
            <AnimatePresence initial={false}>
              {paginatedTransactions.length > 0 ? paginatedTransactions.map((tx) => (
                <motion.div
                  key={tx.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: 0.15 }}
                  className="flex items-center justify-between p-4 rounded-xl cursor-pointer group"
                  style={{
                    background: 'rgba(15, 23, 42, 0.5)',
                    border: '1px solid rgba(212, 175, 55, 0.1)'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'rgba(212, 175, 55, 0.1)';
                    e.currentTarget.style.borderColor = 'rgba(212, 175, 55, 0.3)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'rgba(15, 23, 42, 0.5)';
                    e.currentTarget.style.borderColor = 'rgba(212, 175, 55, 0.1)';
                  }}
                  onClick={() => {
                    setSelectedTransaction(tx);
                    setIsModalOpen(true);
                  }}
                >
                  <div className="flex items-center gap-4">
                    <div
                      className={`w-10 h-10 rounded-lg flex items-center justify-center`}
                      style={{
                        background: tx.type === 'receive'
                          ? 'linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(22, 163, 74, 0.15))'
                          : tx.type === 'mining'
                          ? 'linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(245, 158, 11, 0.15))'
                          : 'linear-gradient(135deg, rgba(244, 63, 94, 0.2), rgba(225, 29, 72, 0.15))',
                        border: `1px solid ${tx.type === 'receive' ? 'rgba(34, 197, 94, 0.3)' : tx.type === 'mining' ? 'rgba(251, 191, 36, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`
                      }}
                    >
                      {tx.type === 'receive' ? '↓' : tx.type === 'mining' ? '⛏️' : '↑'}
                    </div>
                    <div>
                      <div className="font-semibold text-amber-100">
                        {tx.type === 'receive' ? 'Received from' : tx.type === 'mining' ? 'Mining Reward' : 'Sent to'} {' '}
                        <span className="text-amber-300/70">
                          {tx.type === 'receive' ? (tx.from || 'Unknown') : tx.type === 'mining' ? '' : (tx.to || 'Unknown')}
                        </span>
                      </div>
                      <div className="text-sm text-amber-300/50 flex items-center gap-2">
                        {new Date(tx.timestamp).toLocaleString()} •
                        <span className="font-mono">{tx.txHash?.slice(0, 8) || 'N/A'}...</span>
                      </div>
                    </div>
                  </div>
                  <div
                    className={`font-bold text-lg ${
                      tx.type === 'receive' ? 'text-green-400' : tx.type === 'mining' ? 'text-amber-400' : 'text-rose-400'
                    }`}
                  >
                    {tx.type === 'receive' ? '+' : tx.type === 'mining' ? '+' : '-'}{formatBalance(tx.amount)} {TICKER_SYMBOL}
                  </div>
                </motion.div>
              )) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-center py-12 text-amber-300/60"
                >
                  <Activity className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p className="font-medium">No transactions found</p>
                  <p className="text-sm mt-1">
                    Activity will appear here once the node processes transactions
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-6 pt-6"
              style={{ borderTop: '1px solid rgba(212, 175, 55, 0.2)' }}
            >
              <div className="text-sm text-amber-300/60">
                Page {currentPage} of {totalPages} • Showing {paginatedTransactions.length} of {filteredAndSortedTransactions.length}
              </div>

              <div className="flex items-center gap-2">
                <motion.button
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="p-2 rounded-lg disabled:opacity-30 disabled:cursor-not-allowed"
                  style={{
                    background: 'rgba(212, 175, 55, 0.15)',
                    border: '1px solid rgba(212, 175, 55, 0.3)'
                  }}
                  whileHover={{ scale: currentPage === 1 ? 1 : 1.1 }}
                  whileTap={{ scale: currentPage === 1 ? 1 : 0.9 }}
                >
                  <ChevronLeft className="w-4 h-4 text-amber-300" />
                </motion.button>

                {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                  let pageNum;
                  if (totalPages <= 5) {
                    pageNum = i + 1;
                  } else if (currentPage <= 3) {
                    pageNum = i + 1;
                  } else if (currentPage >= totalPages - 2) {
                    pageNum = totalPages - 4 + i;
                  } else {
                    pageNum = currentPage - 2 + i;
                  }

                  return (
                    <motion.button
                      key={pageNum}
                      onClick={() => setCurrentPage(pageNum)}
                      className={`w-8 h-8 rounded-lg text-sm font-medium ${
                        currentPage === pageNum ? 'text-slate-900' : 'text-amber-300'
                      }`}
                      style={currentPage === pageNum ? {
                        background: 'linear-gradient(135deg, #D4AF37, #FFD700)',
                        boxShadow: '0 0 15px rgba(212, 175, 55, 0.3)'
                      } : {
                        background: 'rgba(212, 175, 55, 0.1)',
                        border: '1px solid rgba(212, 175, 55, 0.2)'
                      }}
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                    >
                      {pageNum}
                    </motion.button>
                  );
                })}

                <motion.button
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                  className="p-2 rounded-lg disabled:opacity-30 disabled:cursor-not-allowed"
                  style={{
                    background: 'rgba(212, 175, 55, 0.15)',
                    border: '1px solid rgba(212, 175, 55, 0.3)'
                  }}
                  whileHover={{ scale: currentPage === totalPages ? 1 : 1.1 }}
                  whileTap={{ scale: currentPage === totalPages ? 1 : 0.9 }}
                >
                  <ChevronRight className="w-4 h-4 text-amber-300" />
                </motion.button>
              </div>
            </div>
          )}
        </motion.div>
      </div>

      {/* Transaction Details Modal */}
      <TransactionDetailsModal
        transaction={selectedTransaction}
        isOpen={isModalOpen}
        onClose={() => {
          setIsModalOpen(false);
          setSelectedTransaction(null);
        }}
      />

      {/* QR Code Modal */}
      <QRCodeModal
        isOpen={isQRModalOpen}
        onClose={() => setIsQRModalOpen(false)}
        walletAddress={walletAddress}
        balance={nodeStatus?.balance || 0}
      />

      {/* AI Report Modal */}
      <AnimatePresence>
        {isAIReportModalOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4"
            onClick={() => setIsAIReportModalOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="rounded-2xl p-6 max-w-3xl w-full max-h-[80vh] overflow-y-auto shadow-2xl"
              style={{
                background: 'linear-gradient(135deg, rgba(20, 20, 30, 0.98), rgba(30, 30, 45, 0.98))',
                border: '2px solid rgba(168, 85, 247, 0.3)',
                boxShadow: '0 0 40px rgba(168, 85, 247, 0.3)'
              }}
            >
              {/* Modal Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-3 rounded-xl">
                    <img
                      src="/quantum-ai-logo.png"
                      alt="Quantum AI"
                      className="w-8 h-8 object-contain"
                      style={{
                        filter: 'drop-shadow(0 0 10px rgba(168, 85, 247, 0.6))'
                      }}
                    />
                  </div>
                  <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-400">
                    AI Wallet Analysis
                  </h2>
                </div>
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => setIsAIReportModalOpen(false)}
                  className="p-2 rounded-lg hover:bg-white/10 transition-colors"
                >
                  <svg className="w-6 h-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </motion.button>
              </div>

              {/* AI Report Content */}
              <div className="space-y-4">
                {aiReportLoading && !aiReport && (
                  <div className="flex items-center justify-center py-12">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      className="w-16 h-16"
                    >
                      <img
                        src="/quantum-ai-logo.png"
                        alt="Loading"
                        className="w-full h-full object-contain"
                        style={{
                          filter: 'drop-shadow(0 0 20px rgba(168, 85, 247, 0.8))'
                        }}
                      />
                    </motion.div>
                  </div>
                )}

                {aiReport && (
                  <div className="p-6 rounded-xl bg-gradient-to-br from-purple-500/10 to-pink-500/10 border-2 border-purple-500/20">
                    <div className="prose prose-invert max-w-none">
                      <div className="text-gray-200 whitespace-pre-wrap leading-relaxed">
                        {aiReport}
                      </div>
                    </div>

                    {aiReportLoading && (
                      <motion.div
                        animate={{ opacity: [0.5, 1, 0.5] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                        className="mt-4 text-purple-400 text-sm flex items-center gap-2"
                      >
                        <div className="w-2 h-2 rounded-full bg-purple-400"></div>
                        Generating analysis...
                      </motion.div>
                    )}
                  </div>
                )}

                {!aiReportLoading && !aiReport && (
                  <div className="text-center py-8 text-gray-400">
                    Click "Generate Report" to analyze your wallet and mining performance.
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="mt-6 flex gap-3 justify-end">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setIsAIReportModalOpen(false)}
                  className="px-6 py-3 rounded-xl transition-colors"
                  style={{
                    background: 'linear-gradient(135deg, rgba(100, 100, 120, 0.2), rgba(80, 80, 100, 0.15))',
                    border: '2px solid rgba(100, 100, 120, 0.3)'
                  }}
                >
                  Close
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={generateAIReport}
                  disabled={aiReportLoading}
                  className="px-6 py-3 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{
                    background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.3), rgba(147, 51, 234, 0.2))',
                    border: '2px solid rgba(168, 85, 247, 0.4)'
                  }}
                >
                  {aiReportLoading ? 'Generating...' : 'Regenerate Report'}
                </motion.button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Node Info Modal */}
      <AnimatePresence>
        {isNodeInfoModalOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4"
            onClick={() => setIsNodeInfoModalOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="rounded-2xl p-6 max-w-2xl w-full shadow-2xl"
              style={{
                background: 'linear-gradient(135deg, rgba(20, 20, 30, 0.98), rgba(30, 30, 45, 0.98))',
                border: '2px solid rgba(59, 130, 246, 0.3)',
                boxShadow: '0 0 40px rgba(59, 130, 246, 0.2)'
              }}
            >
              {/* Modal Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-3 rounded-xl bg-gradient-to-br from-blue-500/20 to-cyan-500/20 border-2 border-blue-500/30">
                    <Info className="w-6 h-6 text-blue-400" />
                  </div>
                  <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-400">
                    Node Information
                  </h2>
                </div>
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => setIsNodeInfoModalOpen(false)}
                  className="p-2 rounded-lg hover:bg-white/10 transition-colors"
                >
                  <svg className="w-6 h-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </motion.button>
              </div>

              {/* Performance Metrics */}
              <div className="space-y-4">
                <div className="p-4 rounded-xl bg-gradient-to-br from-purple-500/10 to-pink-500/10 border-2 border-purple-500/20">
                  <h3 className="text-lg font-semibold text-purple-300 mb-3 flex items-center gap-2">
                    <Zap className="w-5 h-5" />
                    Performance Metrics
                  </h3>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-black/20 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Current TPS</div>
                      <div className="text-xl font-bold text-purple-300">
                        {nodeStatus?.tps_current?.toLocaleString() || '0'}
                      </div>
                    </div>
                    <div className="bg-black/20 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Average TPS</div>
                      <div className="text-xl font-bold text-purple-300">
                        {nodeStatus?.tps_average?.toLocaleString() || '0'}
                      </div>
                    </div>
                    <div className="bg-black/20 p-3 rounded-lg col-span-2">
                      <div className="text-xs text-gray-400 mb-1">Max Theoretical TPS</div>
                      <div className="text-xl font-bold text-purple-300">
                        {nodeStatus?.performance?.max_theoretical_tps?.toLocaleString() || '0'}
                      </div>
                    </div>
                    <div className="bg-black/20 p-3 rounded-lg col-span-2">
                      <div className="text-xs text-gray-400 mb-1">Optimization Level</div>
                      <div className="text-sm font-semibold text-purple-300">
                        {nodeStatus?.performance?.optimization_level || 'Unknown'}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Consensus Information */}
                <div className="p-4 rounded-xl bg-gradient-to-br from-green-500/10 to-emerald-500/10 border-2 border-green-500/20">
                  <h3 className="text-lg font-semibold text-green-300 mb-3 flex items-center gap-2">
                    <Activity className="w-5 h-5" />
                    Consensus Status
                  </h3>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-black/20 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Status</div>
                      <div className="text-sm font-bold text-green-300 capitalize">
                        {nodeStatus?.consensus_status || 'unknown'}
                      </div>
                    </div>
                    <div className="bg-black/20 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Validator</div>
                      <div className="text-sm font-bold text-green-300">
                        {nodeStatus?.is_validator ? 'Yes' : 'No'}
                      </div>
                    </div>
                    <div className="bg-black/20 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Round</div>
                      <div className="text-sm font-bold text-green-300">
                        {nodeStatus?.current_round || 0}
                      </div>
                    </div>
                    <div className="bg-black/20 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Height</div>
                      <div className="text-sm font-bold text-green-300">
                        {nodeStatus?.current_height || 0}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Network Health */}
                <div className="p-4 rounded-xl bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border-2 border-blue-500/20">
                  <h3 className="text-lg font-semibold text-blue-300 mb-3 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5" />
                    Network Health
                  </h3>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-black/20 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Uptime</div>
                      <div className="text-sm font-bold text-blue-300">
                        {nodeStatus?.uptime_formatted || '0h 0m 0s'}
                      </div>
                    </div>
                    <div className="bg-black/20 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">TX Pool Size</div>
                      <div className="text-sm font-bold text-blue-300">
                        {nodeStatus?.tx_pool_size || 0}
                      </div>
                    </div>
                    <div className="bg-black/20 p-3 rounded-lg col-span-2">
                      <div className="text-xs text-gray-400 mb-1">Features Enabled</div>
                      <div className="text-xs font-semibold text-blue-300 flex gap-2 flex-wrap mt-1">
                        {nodeStatus?.performance?.simd_crypto_enabled && (
                          <span className="px-2 py-1 bg-green-500/20 border border-green-500/30 rounded">
                            SIMD Crypto
                          </span>
                        )}
                        {nodeStatus?.performance?.kernel_io_enabled && (
                          <span className="px-2 py-1 bg-green-500/20 border border-green-500/30 rounded">
                            Kernel I/O
                          </span>
                        )}
                        {!nodeStatus?.performance?.simd_crypto_enabled && !nodeStatus?.performance?.kernel_io_enabled && (
                          <span className="px-2 py-1 bg-gray-500/20 border border-gray-500/30 rounded">
                            None
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Close Button */}
              <div className="mt-6 flex justify-end">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setIsNodeInfoModalOpen(false)}
                  className="px-6 py-3 rounded-xl font-semibold transition-all"
                  style={{
                    background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(37, 99, 235, 0.15))',
                    border: '2px solid rgba(59, 130, 246, 0.3)',
                    color: 'rgb(96, 165, 250)'
                  }}
                >
                  Close
                </motion.button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Add USD Modal */}
      <AnimatePresence>
        {isAddUSDModalOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4"
            onClick={() => {
              setIsAddUSDModalOpen(false);
              setShowStripeCheckout(false);
              setStripeError(null);
              setUsdAmount('');
            }}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="rounded-2xl p-6 max-w-md w-full shadow-2xl"
              style={{
                background: 'linear-gradient(135deg, rgba(20, 30, 20, 0.98), rgba(30, 45, 30, 0.98))',
                border: '2px solid rgba(34, 197, 94, 0.3)',
                boxShadow: '0 0 40px rgba(34, 197, 94, 0.2)'
              }}
            >
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-3 rounded-xl bg-gradient-to-br from-green-500/20 to-emerald-500/20 border-2 border-green-500/30">
                    <Plus className="w-6 h-6 text-green-400" />
                  </div>
                  <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-green-400 to-emerald-400">
                    Add USD
                  </h2>
                </div>
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => {
                    setIsAddUSDModalOpen(false);
                    setShowStripeCheckout(false);
                    setStripeError(null);
                    setUsdAmount('');
                  }}
                  className="p-2 rounded-lg hover:bg-white/10 transition-colors"
                >
                  <svg className="w-6 h-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </motion.button>
              </div>

              {!showStripeCheckout ? (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Amount (USD)</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0.50"
                      placeholder="0.00"
                      value={usdAmount}
                      onChange={(e) => setUsdAmount(e.target.value)}
                      className="w-full px-4 py-3 rounded-lg text-white placeholder-gray-500"
                      style={{
                        background: 'rgba(0, 0, 0, 0.3)',
                        border: '2px solid rgba(34, 197, 94, 0.2)'
                      }}
                    />
                    <p className="text-xs text-gray-400 mt-1">Minimum: $0.50</p>
                  </div>

                  {stripeError && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="p-3 rounded-lg text-sm bg-red-500/20 text-red-400 border border-red-500/30"
                    >
                      {stripeError}
                    </motion.div>
                  )}

                  <div className="flex gap-3 pt-4">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => {
                        setIsAddUSDModalOpen(false);
                        setShowStripeCheckout(false);
                        setStripeError(null);
                        setUsdAmount('');
                      }}
                      className="flex-1 px-6 py-3 rounded-xl font-semibold transition-all"
                      style={{
                        background: 'rgba(107, 114, 128, 0.2)',
                        border: '2px solid rgba(107, 114, 128, 0.3)',
                        color: 'rgb(156, 163, 175)'
                      }}
                    >
                      Cancel
                    </motion.button>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={handleAddUSD}
                      disabled={stripeLoading || !usdAmount || parseFloat(usdAmount) <= 0}
                      className="flex-1 px-6 py-3 rounded-xl font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                      style={{
                        background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.3), rgba(22, 163, 74, 0.2))',
                        border: '2px solid rgba(34, 197, 94, 0.5)',
                        color: 'rgb(74, 222, 128)'
                      }}
                    >
                      {stripeLoading ? 'Processing...' : 'Continue to Payment'}
                    </motion.button>
                  </div>

                  <p className="text-xs text-gray-500 text-center mt-4">
                    You will be redirected to Stripe to complete the payment securely.
                  </p>
                </div>
              ) : (
                <StripeCheckout
                  amount={usdAmount}
                  walletAddress={localStorage.getItem('walletAddress') || ''}
                  onSuccess={() => {
                    setShowStripeCheckout(false);
                    setIsAddUSDModalOpen(false);
                    setUsdAmount('');
                    setStripeError(null);
                    setRefreshTrigger(prev => prev + 1);
                  }}
                  onCancel={() => {
                    setShowStripeCheckout(false);
                  }}
                />
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Send USD Modal */}
      <AnimatePresence>
        {isSendUSDModalOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4"
            onClick={() => {
              setIsSendUSDModalOpen(false);
              setStripeError(null);
              setUsdAmount('');
              setUsdRecipient('');
            }}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="rounded-2xl p-6 max-w-md w-full shadow-2xl"
              style={{
                background: 'linear-gradient(135deg, rgba(20, 20, 40, 0.98), rgba(30, 30, 60, 0.98))',
                border: '2px solid rgba(59, 130, 246, 0.3)',
                boxShadow: '0 0 40px rgba(59, 130, 246, 0.2)'
              }}
            >
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-3 rounded-xl bg-gradient-to-br from-blue-500/20 to-cyan-500/20 border-2 border-blue-500/30">
                    <Send className="w-6 h-6 text-blue-400" />
                  </div>
                  <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-400">
                    Send USD
                  </h2>
                </div>
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => {
                    setIsSendUSDModalOpen(false);
                    setStripeError(null);
                    setUsdAmount('');
                    setUsdRecipient('');
                  }}
                  className="p-2 rounded-lg hover:bg-white/10 transition-colors"
                >
                  <svg className="w-6 h-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </motion.button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Recipient Wallet Address</label>
                  <input
                    type="text"
                    placeholder="qnk..."
                    value={usdRecipient}
                    onChange={(e) => setUsdRecipient(e.target.value)}
                    className="w-full px-4 py-3 rounded-lg text-white placeholder-gray-500 font-mono text-sm"
                    style={{
                      background: 'rgba(0, 0, 0, 0.3)',
                      border: '2px solid rgba(59, 130, 246, 0.2)'
                    }}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Amount (USD)</label>
                  <input
                    type="number"
                    step="0.01"
                    min="0.01"
                    placeholder="0.00"
                    value={usdAmount}
                    onChange={(e) => setUsdAmount(e.target.value)}
                    className="w-full px-4 py-3 rounded-lg text-white placeholder-gray-500"
                    style={{
                      background: 'rgba(0, 0, 0, 0.3)',
                      border: '2px solid rgba(59, 130, 246, 0.2)'
                    }}
                  />
                  <p className="text-xs text-gray-400 mt-1">Available: ${formatBalance(usdBalance)} USD</p>
                </div>

                {stripeError && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-3 rounded-lg text-sm bg-red-500/20 text-red-400 border border-red-500/30"
                  >
                    {stripeError}
                  </motion.div>
                )}

                <div className="flex gap-3 pt-4">
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => {
                      setIsSendUSDModalOpen(false);
                      setStripeError(null);
                      setUsdAmount('');
                      setUsdRecipient('');
                    }}
                    className="flex-1 px-6 py-3 rounded-xl font-semibold transition-all"
                    style={{
                      background: 'rgba(107, 114, 128, 0.2)',
                      border: '2px solid rgba(107, 114, 128, 0.3)',
                      color: 'rgb(156, 163, 175)'
                    }}
                  >
                    Cancel
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={handleSendUSD}
                    disabled={stripeLoading || !usdAmount || parseFloat(usdAmount) <= 0 || !usdRecipient}
                    className="flex-1 px-6 py-3 rounded-xl font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    style={{
                      background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(37, 99, 235, 0.2))',
                      border: '2px solid rgba(59, 130, 246, 0.5)',
                      color: 'rgb(96, 165, 250)'
                    }}
                  >
                    {stripeLoading ? 'Sending...' : 'Send USD'}
                  </motion.button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loan Application Modal */}
      {showLoanModal && (
        <LoanApplicationModal
          onClose={() => setShowLoanModal(false)}
          walletBalances={walletBalances}
          walletAddress={walletAddress}
        />
      )}

      {/* Loan Approval Modal - Triggered by SSE loan-approved event */}
      {showLoanApprovalModal && approvedLoanDetails && (
        <LoanApprovalModal
          onClose={() => {
            setShowLoanApprovalModal(false);
            setApprovedLoanDetails(null);
          }}
          loanDetails={{
            amount: approvedLoanDetails.amount,
            interestRate: approvedLoanDetails.interestRate,
            termMonths: approvedLoanDetails.termMonths,
            monthlyPayment: approvedLoanDetails.monthlyPayment,
            collateralAmount: approvedLoanDetails.collateralAmount,
            collateralType: approvedLoanDetails.collateralType,
          }}
        />
      )}

      {/* Loan Payback Modal */}
      {showLoanPaybackModal && selectedLoanId && (
        <LoanPaybackModal
          loanId={selectedLoanId}
          onClose={() => {
            setShowLoanPaybackModal(false);
            setSelectedLoanId(null);
          }}
        />
      )}

      {/* K-Law Financial Intelligence Modal */}
      <FinanceModal
        isOpen={showFinanceModal}
        onClose={() => setShowFinanceModal(false)}
      />

    </div>
  );
});

export default Dashboard;
