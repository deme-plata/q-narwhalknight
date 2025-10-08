import { useState, useEffect, memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Zap, AlertCircle, Copy, Check, Wallet, Coins, ChevronLeft, ChevronRight, Calendar, DollarSign, TrendingUp, TrendingDown, QrCode, Info } from 'lucide-react';
import { qnkAPI, type NodeStatus } from '../services/api';
import TransactionDetailsModal from './TransactionDetailsModal';
import QRCodeModal from './QRCodeModal';
import { TICKER_SYMBOL } from '../constants/ticker';

interface Transaction {
  id: string;
  type: 'receive' | 'send';
  amount: number;
  from?: string;
  to?: string;
  timestamp: string;
  txHash: string;
}

interface DashboardProps {
  // Remove mock props - will fetch from API
}

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

  // Transaction details modal state
  const [selectedTransaction, setSelectedTransaction] = useState<Transaction | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // QR code modal state
  const [isQRModalOpen, setIsQRModalOpen] = useState(false);

  // Node info modal state
  const [isNodeInfoModalOpen, setIsNodeInfoModalOpen] = useState(false);

  // Enhanced filtering and pagination state
  const [filterType, setFilterType] = useState<'all' | 'receive' | 'send'>('all');
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
              }
            } catch (balanceErr) {
              console.warn('Failed to fetch wallet balance:', balanceErr);
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
          setError('Failed to connect to Q-NarwhalKnight node');
        }
      }
    };

    const fetchRecentTransactions = async () => {
      console.log('📋 Fetching recent transactions...');
      if (!mounted) return;

      try {
        const response = await qnkAPI.getRecentTransactions(100);
        console.log('📋 Transactions API response:', response);
        if (!mounted) return;

        // Always merge with existing faucet transactions, even if API fails
        setRecentTransactions(prev => {
          console.log('📋 Current transactions before merge:', prev.length);

          // Get all faucet transactions (manually added)
          const faucetTxs = prev.filter(tx => tx.id.startsWith('faucet-'));
          console.log('📋 Faucet transactions to preserve:', faucetTxs.length);

          // If API call failed or returned no data, just keep faucet transactions
          if (!response.success || !response.data) {
            console.log('📋 API failed or no data, keeping only faucet transactions');
            return faucetTxs;
          }

          // Get current wallet address to determine send/receive
          const currentWalletAddress = localStorage.getItem('walletAddress') || '';
          console.log('📋 Current wallet address for comparison:', currentWalletAddress);

          // Transform API data to match frontend Transaction interface
          const transformedTransactions: Transaction[] = response.data
            .filter((tx: any) => {
              // Filter out invalid transactions
              if (!tx.from || !tx.to) return false;
              if (tx.from === '0000000000000000000000000000000000000000000000000000000000000000') return false;
              if (tx.to === '0000000000000000000000000000000000000000000000000000000000000000') return false;
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

              return {
                id: tx.id || tx.hash,
                type,
                amount,
                from: tx.from,
                to: tx.to,
                timestamp,
                txHash: tx.hash || tx.id,
              };
            });

          console.log('📋 Transformed API transactions:', transformedTransactions.length);

          // Merge and deduplicate by id
          const allTxs = [...faucetTxs, ...transformedTransactions];
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
        // On error, preserve existing faucet transactions
        setRecentTransactions(prev => prev.filter(tx => tx.id.startsWith('faucet-')));
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
          localStorage.setItem('walletSeed', response.data.mnemonic);

          setWalletAddress(address);
          console.log('Generated new wallet');
        } else {
          const prefix = 'qnk';
          const randomBytes = new Uint8Array(20);
          crypto.getRandomValues(randomBytes);
          const address = prefix + Array.from(randomBytes).map(b => b.toString(16).padStart(2, '0')).join('');
          setWalletAddress(address);
          console.log('Generated fallback wallet');
        }
      } catch (error) {
        console.error('Failed to generate wallet address:', error);
        const prefix = 'qnk';
        const randomBytes = new Uint8Array(20);
        crypto.getRandomValues(randomBytes);
        const address = prefix + Array.from(randomBytes).map(b => b.toString(16).padStart(2, '0')).join('');
        setWalletAddress(address);
        console.log('Generated error fallback wallet');
      }
    };

    const loadData = async () => {
      console.log('Loading dashboard data...');
      setLoading(true);
      try {
        await generateWalletAddress();
        await Promise.all([fetchNodeStatus(), fetchRecentTransactions()]);
      } catch (error) {
        console.error('Error loading data:', error);
      } finally {
        setLoading(false);
        console.log('Dashboard data loaded');
      }
    };

    loadData();

    // Set up SSE for real-time balance updates
    const sseUrl = import.meta.env.VITE_API_URL ?
      `${import.meta.env.VITE_API_URL}/v1/events` :
      '/api/v1/events';

    console.log('📡 Attempting SSE connection to:', sseUrl);

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

            if (!currentHex || eventHex === currentHex) {
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
              console.log('❌ Balance update IGNORED (different wallet)');
            }
          } else if (eventType === 'faucet-dispensed') {
            console.log('🚰 FAUCET EVENT:', data);
            console.log('🔄 Refreshing balance via fetchNodeStatus...');
            fetchNodeStatus();
            fetchRecentTransactions();
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

      console.log('✅ SSE event listeners registered for: balance-updated, faucet-dispensed, transaction-confirmed, transaction-submitted');

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

            if (!currentHex || eventHex === currentHex) {
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
              console.log('❌ Dashboard: Balance update ignored (not for current wallet)');
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

            // Check if this mining reward is for the current wallet
            if (currentWalletAddress === rewardData.miner_address) {
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

    return () => {
      mounted = false;
      if (eventSource) {
        eventSource.close();
      }
    };
  }, []);

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

      {/* Wallet Address Card */}
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
        {/* Animated gold shimmer effect */}
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-transparent via-amber-500/10 to-transparent"
          initial={{ x: '-100%' }}
          animate={{ x: '100%' }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        />

        <div className="relative">
          <div className="flex items-start justify-between mb-4">
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

              {/* Show faucet button if balance is 0 */}
              {nodeStatus && nodeStatus.balance === 0 && (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={requestFaucetTokens}
                  disabled={faucetLoading}
                  className="p-3 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{
                    background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(22, 163, 74, 0.15))',
                    border: '2px solid rgba(34, 197, 94, 0.3)'
                  }}
                  title="Get test tokens"
                >
                  {faucetLoading ? (
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    >
                      <Coins className="w-5 h-5 text-green-400" />
                    </motion.div>
                  ) : (
                    <Coins className="w-5 h-5 text-green-400" />
                  )}
                </motion.button>
              )}
            </div>
          </div>
        </div>

        {/* Faucet message */}
        {faucetMessage && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`mt-3 p-3 rounded-xl text-sm ${
              faucetMessage.startsWith('Success')
                ? 'bg-quantum-green/20 text-quantum-green border border-quantum-green/30'
                : 'bg-quantum-pink/20 text-quantum-pink border border-quantum-pink/30'
            }`}
          >
            {faucetMessage}
          </motion.div>
        )}
      </motion.div>

      <div className="grid grid-cols-1 gap-8">
        {/* Recent Activity - Enhanced */}
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
                {(['all', 'receive', 'send'] as const).map((type) => (
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
                    {type === 'all' ? 'All' : type === 'receive' ? '↓ Received' : '↑ Sent'}
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
            <AnimatePresence mode="popLayout">
              {paginatedTransactions.length > 0 ? paginatedTransactions.map((tx, index) => (
                <motion.div
                  key={tx.id}
                  layout
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ delay: index * 0.05 }}
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
                    <motion.div
                      className={`w-10 h-10 rounded-lg flex items-center justify-center`}
                      style={{
                        background: tx.type === 'receive'
                          ? 'linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(22, 163, 74, 0.15))'
                          : 'linear-gradient(135deg, rgba(244, 63, 94, 0.2), rgba(225, 29, 72, 0.15))',
                        border: `1px solid ${tx.type === 'receive' ? 'rgba(34, 197, 94, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`
                      }}
                      whileHover={{ scale: 1.1, rotate: 360 }}
                      transition={{ duration: 0.3 }}
                    >
                      {tx.type === 'receive' ? '↓' : '↑'}
                    </motion.div>
                    <div>
                      <div className="font-semibold text-amber-100">
                        {tx.type === 'receive' ? 'Received from' : 'Sent to'} {' '}
                        <span className="text-amber-300/70">
                          {tx.type === 'receive' ? (tx.from || 'Unknown') : (tx.to || 'Unknown')}
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
                      tx.type === 'receive' ? 'text-green-400' : 'text-rose-400'
                    }`}
                  >
                    {tx.type === 'receive' ? '+' : '-'}{formatBalance(tx.amount)} {TICKER_SYMBOL}
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

    </div>
  );
});

export default Dashboard;
