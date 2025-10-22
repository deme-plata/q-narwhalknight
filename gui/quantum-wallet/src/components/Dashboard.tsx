import { useState, useEffect, memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Zap, AlertCircle, Copy, Check, Wallet, Coins, ChevronLeft, ChevronRight, Calendar, DollarSign, TrendingUp, TrendingDown, QrCode, Info, Plus, Send, ArrowUpRight } from 'lucide-react';
import { qnkAPI, type NodeStatus } from '../services/api';
import TransactionDetailsModal from './TransactionDetailsModal';
import QRCodeModal from './QRCodeModal';
import StripeCheckout from './StripeCheckout';
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

interface WalletBalance {
  symbol: string;
  name: string;
  balance: number;
  usdValue?: number;
  icon: 'qug' | 'usd' | 'btc' | 'eth' | 'sol' | 'custom';
  color: string;
  comingSoon?: boolean;
}

interface DashboardProps {
  // Remove mock props - will fetch from API
}

// Animated Balance Component with wicked awesome effects
const AnimatedBalance = memo(function AnimatedBalance({
  value,
  isAnimating,
  symbol
}: {
  value: number;
  isAnimating: boolean;
  symbol: string;
}) {
  const [displayValue, setDisplayValue] = useState(value);
  const [particles, setParticles] = useState<Array<{ id: number; x: number; y: number }>>([]);

  // Smooth counting animation
  useEffect(() => {
    if (value === displayValue) return;

    const difference = value - displayValue;
    const duration = 300; // ms - faster animation for frequent mining rewards
    const steps = 20; // fewer steps for snappier animation
    const increment = difference / steps;
    const stepTime = duration / steps;

    let currentStep = 0;
    const interval = setInterval(() => {
      currentStep++;
      if (currentStep >= steps) {
        setDisplayValue(value);
        clearInterval(interval);
      } else {
        setDisplayValue(prev => prev + increment);
      }
    }, stepTime);

    return () => clearInterval(interval);
  }, [value]);

  // Particle burst effect when balance increases
  useEffect(() => {
    if (isAnimating && value > displayValue) {
      const newParticles = Array.from({ length: 8 }, (_, i) => ({
        id: Date.now() + i,
        x: Math.random() * 100 - 50,
        y: Math.random() * 100 - 50,
      }));
      setParticles(newParticles);
      setTimeout(() => setParticles([]), 600); // faster particle cleanup
    }
  }, [isAnimating]);

  const formatBalance = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 8,
    }).format(amount);
  };

  return (
    <div className="relative inline-block">
      {/* Particle effects */}
      <AnimatePresence>
        {particles.map(particle => (
          <motion.div
            key={particle.id}
            initial={{
              opacity: 1,
              scale: 0,
              x: 0,
              y: 0,
            }}
            animate={{
              opacity: 0,
              scale: 1.5,
              x: particle.x,
              y: particle.y,
            }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="absolute top-1/2 left-1/2 w-2 h-2 rounded-full pointer-events-none"
            style={{
              background: symbol === 'QUG'
                ? 'linear-gradient(135deg, #FFD700, #FFA500)'
                : 'linear-gradient(135deg, #10b981, #34d399)',
              boxShadow: '0 0 8px currentColor',
            }}
          />
        ))}
      </AnimatePresence>

      {/* Quantum glow pulse */}
      {isAnimating && (
        <motion.div
          className="absolute inset-0 rounded-lg pointer-events-none"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{
            opacity: [0, 0.6, 0],
            scale: [0.8, 1.1, 0.8],
          }}
          transition={{
            duration: 0.5,
            repeat: 1,
            ease: "easeInOut"
          }}
          style={{
            background: symbol === 'QUG'
              ? 'radial-gradient(circle, rgba(255, 215, 0, 0.4), transparent 70%)'
              : 'radial-gradient(circle, rgba(16, 185, 129, 0.4), transparent 70%)',
            filter: 'blur(10px)',
          }}
        />
      )}

      {/* Balance number with rainbow shimmer */}
      <motion.div
        className="relative z-10 text-2xl font-bold text-white"
        animate={isAnimating ? {
          textShadow: [
            '0 0 10px rgba(255, 215, 0, 0.8)',
            '0 0 20px rgba(255, 107, 0, 0.8)',
            '0 0 20px rgba(16, 185, 129, 0.8)',
            '0 0 20px rgba(59, 130, 246, 0.8)',
            '0 0 10px rgba(168, 85, 247, 0.8)',
            '0 0 10px rgba(255, 215, 0, 0.8)',
          ],
        } : {}}
        transition={{
          duration: 2,
          repeat: isAnimating ? 1 : 0,
          ease: "easeInOut"
        }}
      >
        {formatBalance(displayValue)}
      </motion.div>

      {/* Sparkle effect overlay */}
      {isAnimating && (
        <motion.div
          className="absolute inset-0 pointer-events-none overflow-hidden rounded-lg"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          {[...Array(6)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 bg-white rounded-full"
              style={{
                top: `${Math.random() * 100}%`,
                left: `${Math.random() * 100}%`,
              }}
              animate={{
                scale: [0, 1.5, 0],
                opacity: [0, 1, 0],
              }}
              transition={{
                duration: 1,
                delay: i * 0.15,
                repeat: 1,
              }}
            />
          ))}
        </motion.div>
      )}
    </div>
  );
});

const Dashboard = memo(function Dashboard({}: DashboardProps) {
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
  const [transactionError, setTransactionError] = useState<string | null>(null);

  // Multi-wallet state
  const [walletBalances, setWalletBalances] = useState<WalletBalance[]>([]);
  const [usdBalance, setUsdBalance] = useState<number>(0);

  // Animation state for balance updates
  const [balanceAnimations, setBalanceAnimations] = useState<Record<string, boolean>>({});
  const [previousBalances, setPreviousBalances] = useState<Record<string, number>>({});

  // Transaction details modal state
  const [selectedTransaction, setSelectedTransaction] = useState<Transaction | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // QR code modal state
  const [isQRModalOpen, setIsQRModalOpen] = useState(false);

  // Node info modal state
  const [isNodeInfoModalOpen, setIsNodeInfoModalOpen] = useState(false);

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
  useEffect(() => {
    const newAnimations: Record<string, boolean> = {};
    const newPreviousBalances: Record<string, number> = {};

    walletBalances.forEach(wallet => {
      const key = wallet.symbol;
      const prevBalance = previousBalances[key] ?? wallet.balance;
      newPreviousBalances[key] = wallet.balance;

      // Trigger animation if balance changed
      if (prevBalance !== wallet.balance && prevBalance !== undefined) {
        newAnimations[key] = true;
        console.log(`🎨 Balance animation triggered for ${key}: ${prevBalance} → ${wallet.balance}`);

        // Auto-disable animation after 3 seconds
        setTimeout(() => {
          setBalanceAnimations(prev => ({ ...prev, [key]: false }));
        }, 3000);
      } else {
        newAnimations[key] = false;
      }
    });

    setBalanceAnimations(newAnimations);
    setPreviousBalances(newPreviousBalances);
  }, [walletBalances]);

  // Fetch real data from Q-NarwhalKnight API
  useEffect(() => {
    let mounted = true;
    let eventSource: EventSource | null = null;

    const fetchNodeStatus = async () => {
      console.log('Fetching node status...');
      if (!mounted) return;

      try {
        const response = await qnkAPI.getNodeStatus();
        console.log('Node status response:', response);
        if (!mounted) return;

        if (response.success && response.data) {
          const currentWalletAddress = localStorage.getItem('walletAddress');
          let walletBalance = 0;

          if (currentWalletAddress) {
            try {
              const balanceResponse = await qnkAPI.getWalletBalance(currentWalletAddress);
              if (!mounted) return;

              if (balanceResponse.success && balanceResponse.data) {
                walletBalance = balanceResponse.data.balance_qnk || 0;
                console.log('✅ Balance fetched successfully:', walletBalance);
                // Store balance in localStorage for fallback on refresh
                localStorage.setItem('cachedBalance', walletBalance.toString());
              } else {
                // Authentication failed - use cached balance from localStorage
                console.warn('⚠️ Balance query failed (authentication required):', balanceResponse.error);
                const cachedBalance = localStorage.getItem('cachedBalance');
                if (cachedBalance) {
                  walletBalance = parseFloat(cachedBalance);
                  console.log('💰 Using cached balance from localStorage:', walletBalance);
                }
              }
            } catch (balanceErr) {
              console.warn('❌ Failed to fetch wallet balance:', balanceErr);
              // Fallback: use cached balance from localStorage
              const cachedBalance = localStorage.getItem('cachedBalance');
              if (cachedBalance) {
                walletBalance = parseFloat(cachedBalance);
                console.log('💰 Using cached balance from localStorage (error fallback):', walletBalance);
              }
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
          // Even if node status fails, try to load cached balance
          const cachedBalance = localStorage.getItem('cachedBalance');
          if (cachedBalance) {
            const balanceValue = parseFloat(cachedBalance);
            console.log('💰 Using cached balance after node status error:', balanceValue);
            setNodeStatus({
              balance: balanceValue,
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
      let qugBalance = 0;
      try {
        const balanceResponse = await qnkAPI.getWalletBalance(currentWalletAddress);
        if (balanceResponse.success && balanceResponse.data) {
          qugBalance = balanceResponse.data.balance_qnk || 0;
          console.log('💰 Fresh QUG balance fetched:', qugBalance);
        } else {
          // Fall back to nodeStatus if API fails
          qugBalance = nodeStatus?.balance || 0;
          console.warn('⚠️ Using nodeStatus balance as fallback:', qugBalance);
        }
      } catch (error) {
        // Fall back to nodeStatus on error
        qugBalance = nodeStatus?.balance || 0;
        console.error('❌ Failed to fetch QUG balance, using nodeStatus:', error);
      }

      const balances: WalletBalance[] = [
        {
          symbol: 'QUG',
          name: 'Quillon Graph',
          balance: qugBalance,
          icon: 'qug',
          color: 'from-amber-400 to-yellow-500',
        }
      ];

      // Fetch QUGUSD balance (Quillon USD stablecoin)
      let qugUsdBalance = 0;
      try {
        const response = await qnkAPI.getMultiTokenBalance();
        console.log('🔍 [Dashboard] Multi-token balance response:', JSON.stringify(response, null, 2));
        if (response.success && response.data && response.data.tokens) {
          // API returns tokens as object with lowercase keys: { qug: {...}, qugusd: {...} }
          const tokensObj = response.data.tokens;
          console.log('🔍 [Dashboard] Tokens object:', JSON.stringify(tokensObj, null, 2));
          if (tokensObj.qugusd && tokensObj.qugusd.balance !== undefined) {
            qugUsdBalance = parseFloat(tokensObj.qugusd.balance) || 0;
            console.log('💵 [Dashboard] QUGUSD balance fetched:', qugUsdBalance);
          } else if (tokensObj.QUGUSD && tokensObj.QUGUSD.balance !== undefined) {
            // Try uppercase key as fallback
            qugUsdBalance = parseFloat(tokensObj.QUGUSD.balance) || 0;
            console.log('💵 [Dashboard] QUGUSD balance fetched (uppercase):', qugUsdBalance);
          } else {
            console.warn('⚠️ [Dashboard] QUGUSD not found in tokens object');
          }
        } else {
          console.warn('⚠️ [Dashboard] Response not successful or missing data');
        }
      } catch (error) {
        console.warn('⚠️ Failed to fetch QUGUSD balance:', error);
      }

      // Add QUGUSD to balances (always show, even with 0 balance)
      balances.push({
        symbol: 'QUGUSD',
        name: 'Quillon USD',
        balance: qugUsdBalance,
        usdValue: qugUsdBalance, // 1:1 peg to USD
        icon: 'usd',
        color: 'from-blue-400 to-cyan-500',
      });

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
      balances.push({
        symbol: 'USD',
        name: 'US Dollar',
        balance: usdValue,
        icon: 'usd',
        color: 'from-green-400 to-emerald-500',
      });

      // Note: Custom tokens would be fetched here if the API supported them
      // Currently, only QUG and QUGUSD are supported in the multi-token balance endpoint

      // Add placeholder for future cryptos
      balances.push(
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

      setWalletBalances(balances);
    };

    const fetchRecentTransactions = async () => {
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
              const walletHex = currentWalletAddress.startsWith('qnk')
                ? currentWalletAddress.substring(3)
                : currentWalletAddress;
              const toHex = tx.to.startsWith('qnk') ? tx.to.substring(3) : tx.to;
              const fromHex = tx.from.startsWith('qnk') ? tx.from.substring(3) : tx.from;

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

              // Convert amount from smallest units to QNK (divide by 100,000,000)
              const amount = typeof tx.amount === 'number'
                ? tx.amount / 100000000
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

    const loadData = async () => {
      console.log('🚀 [loadData] START - Loading dashboard data...');
      setLoading(true);
      try {
        console.log('🚀 [loadData] Step 1: Generating wallet address...');
        await generateWalletAddress();
        console.log('🚀 [loadData] Step 2: Calling fetchNodeStatus and fetchRecentTransactions in parallel...');
        await Promise.all([fetchNodeStatus(), fetchRecentTransactions()]);
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
    const currentWalletForSSE = localStorage.getItem('walletAddress') || '';
    const sseUrl = import.meta.env.VITE_API_URL ?
      `${import.meta.env.VITE_API_URL}/v1/events?wallet_address=${encodeURIComponent(currentWalletForSSE)}` :
      `/api/v1/events?wallet_address=${encodeURIComponent(currentWalletForSSE)}`;

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

              // Dispatch custom event to trigger App.tsx balance refresh
              window.dispatchEvent(new CustomEvent('balance-update', {
                detail: { balance: newBalance }
              }));

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

              // Dispatch custom event to trigger App.tsx balance refresh
              window.dispatchEvent(new CustomEvent('balance-update-refresh'));

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

                // Dispatch custom event to trigger App.tsx balance refresh
                window.dispatchEvent(new CustomEvent('balance-update', {
                  detail: { balance: data.current_balance }
                }));
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
              console.log('✅ Dashboard: Balance update applied (onmessage):', data.data.new_balance);
              setNodeStatus(prev => prev ? { ...prev, balance: data.data.new_balance } : prev);

              // Dispatch custom event to trigger App.tsx balance refresh
              window.dispatchEvent(new CustomEvent('balance-update', {
                detail: { balance: data.data.new_balance }
              }));

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

              // Dispatch custom event to trigger App.tsx balance refresh
              window.dispatchEvent(new CustomEvent('balance-update', {
                detail: { balance: rewardData.new_balance_qnk }
              }));

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
      if (eventSource) {
        eventSource.close();
      }
      window.removeEventListener('cdp-mint', handleCDPMint);
    };
  }, []);

  // Refresh wallet balances when refreshTrigger changes
  useEffect(() => {
    if (refreshTrigger > 0) {
      console.log('🔄 Refreshing wallet balances...');
      const refresh = async () => {
        const currentWalletAddress = localStorage.getItem('walletAddress');
        if (!currentWalletAddress) return;

        const balances: WalletBalance[] = [
          {
            symbol: 'QUG',
            name: 'Quillon Graph',
            balance: nodeStatus?.balance || 0,
            icon: 'qug',
            color: 'from-amber-400 to-yellow-500',
          }
        ];

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
          // Silently fail if payment API is not available
          console.warn('⚠️ Payment API not available - USD wallet features disabled');
        }

        // Fetch QUGUSD balance from multi-token API
        try {
          const response = await qnkAPI.getMultiTokenBalance();
          if (response.success && response.data && response.data.tokens) {
            // API returns tokens as object with uppercase keys: { QUG: {...}, QUGUSD: {...} }
            const tokensObj = response.data.tokens;

            // Add QUGUSD if it exists and has balance
            if (tokensObj.QUGUSD && tokensObj.QUGUSD.balance_base_units > 0) {
              balances.push({
                symbol: 'QUGUSD',
                name: 'Quillon USD',
                balance: tokensObj.QUGUSD.balance_base_units / 1e8,
                icon: 'usd' as const,
                color: 'from-green-400 to-emerald-500',
              });
              console.log('💵 Added QUGUSD balance:', tokensObj.QUGUSD.balance_base_units / 1e8);
            }
          }
        } catch (error) {
          console.error('❌ Failed to fetch QUGUSD balance:', error);
        }

        // Add placeholders
        balances.push(
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

        setWalletBalances(balances);
      };

      refresh();
    }
  }, [refreshTrigger, nodeStatus?.balance]);

  const formatBalance = (amount: number, hidden = false) => {
    if (hidden) return '••••••••';
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 8,
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

        // Dispatch custom event to trigger App.tsx balance refresh
        window.dispatchEvent(new CustomEvent('balance-update', {
          detail: { balance: result.data?.new_balance_qnk || receivedAmount }
        }));

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

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {walletBalances.map((wallet, index) => (
                <motion.div
                  key={wallet.symbol}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 * index }}
                  className="p-4 rounded-xl cursor-pointer group relative overflow-hidden"
                  style={{
                    background: `linear-gradient(135deg, rgba(30, 20, 60, 0.8), rgba(50, 30, 80, 0.8))`,
                    border: `2px solid rgba(212, 175, 55, ${wallet.comingSoon ? '0.1' : '0.3'})`,
                  }}
                  whileHover={{ scale: wallet.comingSoon ? 1 : 1.02 }}
                >
                  {wallet.comingSoon && (
                    <div className="absolute top-2 right-2 px-2 py-1 rounded-lg text-xs font-bold bg-gradient-to-r from-purple-500/30 to-pink-500/30 border border-purple-400/30 text-purple-300">
                      Coming Soon
                    </div>
                  )}

                  <div className="flex items-start justify-between mb-3">
                    <div className={`p-2 rounded-lg bg-gradient-to-br ${wallet.color}`}>
                      {(wallet.icon === 'qug' || wallet.icon === 'usd') && (
                        <div className="relative w-5 h-5">
                          <div className="absolute inset-0 rounded-full" style={{
                            background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 25%, #FFA500 50%, #FFD700 75%, #D4AF37 100%)',
                            padding: '1px'
                          }}>
                            <div className="w-full h-full bg-gradient-to-b from-slate-900 via-blue-950 to-slate-900 rounded-full flex items-center justify-center p-0.5">
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
                      {wallet.icon === 'btc' && (
                        <div className="relative w-5 h-5">
                          {/* Bitcoin logo with gradient styling */}
                          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <defs>
                              <linearGradient id="btcGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stopColor="#F7931A" />
                                <stop offset="100%" stopColor="#FFA500" />
                              </linearGradient>
                            </defs>
                            <circle cx="12" cy="12" r="10" fill="url(#btcGradient)"/>
                            <path d="M13.5 9.5c0-.5-.3-.9-.8-1.1.3-.2.5-.6.5-1 0-.8-.6-1.4-1.4-1.4h-.6V5h-1v1h-.7V5h-1v1H7v1h.5c.3 0 .5.2.5.5v7c0 .3-.2.5-.5.5H7v1h1.5v1h1v-1h.7v1h1v-1h.6c1.5 0 2.7-1.2 2.7-2.7 0-.9-.4-1.6-1-2.1.6-.4 1-1.1 1-1.7zm-3.3-.5h.6c.6 0 1 .4 1 1s-.4 1-1 1h-.6V9zm.8 5h-.8v-2.5h.8c.8 0 1.5.7 1.5 1.5s-.7 1-1.5 1z" fill="white"/>
                          </svg>
                        </div>
                      )}
                      {wallet.icon === 'eth' && (
                        <div className="relative w-5 h-5">
                          {/* Ethereum logo with proper styling */}
                          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M11.944 17.97L4.58 13.62 11.943 24l7.37-10.38-7.372 4.35h.003zM12.056 0L4.69 12.223l7.365 4.354 7.365-4.35L12.056 0z" fill="currentColor" className="text-indigo-400"/>
                          </svg>
                        </div>
                      )}
                      {wallet.icon === 'sol' && (
                        <div className="relative w-5 h-5">
                          {/* Solana logo with gradient styling */}
                          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <defs>
                              <linearGradient id="solGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stopColor="#9945FF" />
                                <stop offset="100%" stopColor="#14F195" />
                              </linearGradient>
                            </defs>
                            <path d="M4.5 17.5l2.5-2.5h14l-2.5 2.5H4.5zM4.5 11.5L7 9h14l-2.5 2.5H4.5zM7 6.5L4.5 9h14L21 6.5H7z" fill="url(#solGradient)"/>
                          </svg>
                        </div>
                      )}
                      {wallet.icon === 'custom' && <Wallet className="w-5 h-5 text-white" />}
                    </div>
                    <div className="text-right">
                      <div className="text-xs text-gray-400 mb-1">{wallet.name}</div>
                      <div className="text-lg font-bold text-white">{wallet.symbol}</div>
                    </div>
                  </div>

                  <div className={`mb-2 ${wallet.comingSoon ? 'text-gray-500' : ''}`}>
                    {wallet.comingSoon ? (
                      <div className="text-2xl font-bold text-gray-500">0.00</div>
                    ) : (
                      <AnimatedBalance
                        value={wallet.balance}
                        isAnimating={balanceAnimations[wallet.symbol] || false}
                        symbol={wallet.symbol}
                      />
                    )}
                  </div>

                  {!wallet.comingSoon && wallet.symbol === 'USD' && (
                    <div className="flex gap-2 mt-3">
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
                          setIsSendUSDModalOpen(true);
                          setStripeError(null);
                        }}
                        className="flex-1 py-2 px-3 rounded-lg text-xs font-medium bg-blue-500/20 border border-blue-500/30 text-blue-300 flex items-center justify-center gap-1"
                        title="Send USD"
                      >
                        <Send className="w-3 h-3" />
                        Send
                      </motion.button>
                    </div>
                  )}

                  {!wallet.comingSoon && wallet.symbol !== 'USD' && wallet.balance > 0 && (
                    <div className="flex gap-2 mt-3">
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="flex-1 py-2 px-3 rounded-lg text-xs font-medium bg-amber-500/20 border border-amber-500/30 text-amber-300 flex items-center justify-center gap-1"
                        title="Swap"
                      >
                        <ArrowUpRight className="w-3 h-3" />
                        Swap
                      </motion.button>
                    </div>
                  )}
                </motion.div>
              ))}
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

    </div>
  );
});

export default Dashboard;
