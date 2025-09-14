import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Activity, Zap, AlertCircle, Copy, Check, Wallet, Coins } from 'lucide-react';
import { qnkAPI } from '../services/api';

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

interface NodeStatus {
  balance: number;
  qci: number;
  nodeId: string;
  blockHeight: number;
  peers: number;
  isConnected: boolean;
}

export default function Dashboard({}: DashboardProps) {
  const [nodeStatus, setNodeStatus] = useState<NodeStatus | null>(null);
  const [recentTransactions, setRecentTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [walletAddress, setWalletAddress] = useState('');
  const [copiedAddress, setCopiedAddress] = useState(false);
  const [faucetLoading, setFaucetLoading] = useState(false);
  const [faucetMessage, setFaucetMessage] = useState('');
  const [realWalletAddress, setRealWalletAddress] = useState('');

  // Fetch real data from Q-NarwhalKnight API
  useEffect(() => {
    const fetchNodeStatus = async () => {
      try {
        const response = await qnkAPI.getNodeStatus();
        if (response.success && response.data) {
          // Get wallet balance separately if we have a wallet address
          let walletBalance = 0;
          if (realWalletAddress) {
            try {
              console.log('Fetching balance for wallet address:', realWalletAddress);
              const balanceResponse = await qnkAPI.getWalletBalance(realWalletAddress);
              console.log('Balance API response:', balanceResponse);
              if (balanceResponse.success && balanceResponse.data) {
                walletBalance = balanceResponse.data.balance_qnk || 0;
                console.log('Parsed wallet balance:', walletBalance);
              }
            } catch (balanceErr) {
              console.warn('Failed to fetch wallet balance:', balanceErr);
            }
          } else {
            console.log('No realWalletAddress available yet for balance fetch');
          }

          setNodeStatus({
            balance: walletBalance, // Use wallet balance instead of node balance
            qci: 42, // Quantum Consensus Index
            nodeId: response.data.node_id,
            blockHeight: response.data.current_height,
            peers: response.data.connected_peers,
            isConnected: response.data.network_health === 'healthy'
          });
          setError(null);
        } else {
          throw new Error(response.error || 'Failed to fetch node status');
        }
      } catch (err) {
        console.error('Error fetching node status:', err);
        setError('Failed to connect to Q-NarwhalKnight node');
      }
    };

    const fetchRecentTransactions = async () => {
      try {
        const response = await fetch('/api/v1/transactions/recent?limit=5');
        if (!response.ok) throw new Error('Failed to fetch transactions');
        const data = await response.json();
        setRecentTransactions(data);
      } catch (err) {
        console.error('Error fetching transactions:', err);
      }
    };

    const generateWalletAddress = async () => {
      // Check if we already have a wallet address stored
      const storedAddress = localStorage.getItem('walletAddress');
      const storedMnemonic = localStorage.getItem('walletSeed'); // Use the same key as App.tsx
      
      if (storedAddress && storedMnemonic) {
        setWalletAddress(storedAddress.substring(0, 42)); // Display truncated
        setRealWalletAddress(storedAddress); // Store full address
        return;
      }

      // If we have a mnemonic but no address, derive the address
      if (storedMnemonic && !storedAddress) {
        try {
          // Use the stored mnemonic to generate consistent wallet address
          const encoder = new TextEncoder();
          const data = encoder.encode(storedMnemonic);
          const hashBuffer = await crypto.subtle.digest('SHA-256', data);
          const hashArray = Array.from(new Uint8Array(hashBuffer));
          const address = 'qnk' + hashArray.map(b => b.toString(16).padStart(2, '0')).join('').substring(0, 40);
          
          localStorage.setItem('walletAddress', address);
          setWalletAddress(address.substring(0, 42)); // Display truncated
          setRealWalletAddress(address); // Store full address
          return;
        } catch (error) {
          console.error('Failed to derive address from mnemonic:', error);
        }
      }

      try {
        // Generate a new wallet using the API
        const response = await qnkAPI.generateMnemonic();
        if (response.success && response.data?.mnemonic) {
          // Derive address from the mnemonic consistently
          const encoder = new TextEncoder();
          const data = encoder.encode(response.data.mnemonic);
          const hashBuffer = await crypto.subtle.digest('SHA-256', data);
          const hashArray = Array.from(new Uint8Array(hashBuffer));
          const address = 'qnk' + hashArray.map(b => b.toString(16).padStart(2, '0')).join('').substring(0, 40);
          
          localStorage.setItem('walletAddress', address);
          localStorage.setItem('walletSeed', response.data.mnemonic); // Match App.tsx key
          
          setWalletAddress(address.substring(0, 42)); // Display truncated
          setRealWalletAddress(address); // Store full address
        } else {
          // Fallback to random address if API fails
          const prefix = 'qnk';
          const randomBytes = new Uint8Array(20);
          crypto.getRandomValues(randomBytes);
          const address = prefix + Array.from(randomBytes).map(b => b.toString(16).padStart(2, '0')).join('');
          setWalletAddress(address.substring(0, 42));
          setRealWalletAddress(address);
        }
      } catch (error) {
        console.error('Failed to generate wallet address:', error);
        // Fallback to random address
        const prefix = 'qnk';
        const randomBytes = new Uint8Array(20);
        crypto.getRandomValues(randomBytes);
        const address = prefix + Array.from(randomBytes).map(b => b.toString(16).padStart(2, '0')).join('');
        setWalletAddress(address.substring(0, 42));
        setRealWalletAddress(address);
      }
    };

    const loadData = async () => {
      setLoading(true);
      await Promise.all([fetchNodeStatus(), fetchRecentTransactions(), generateWalletAddress()]);
      setLoading(false);
    };

    loadData();

    // Set up SSE for real-time balance updates instead of polling
    const sseUrl = import.meta.env.VITE_API_URL ? 
      `${import.meta.env.VITE_API_URL}/v1/events` : 
      '/api/v1/events';
    const eventSource = new EventSource(sseUrl);
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Update balance when faucet or transaction events occur
        if (data.type === 'faucet-dispensed' || data.type === 'transaction-submitted') {
          console.log('SSE event received:', data.type, 'for address:', realWalletAddress);
          // Refresh node status to get updated balance - inline version
          (async () => {
            try {
              const response = await qnkAPI.getNodeStatus();
              if (response.success && response.data) {
                let walletBalance = 0;
                // Get current realWalletAddress from state
                const currentWalletAddress = realWalletAddress;
                if (currentWalletAddress) {
                  try {
                    console.log('SSE refreshing balance for:', currentWalletAddress);
                    const balanceResponse = await qnkAPI.getWalletBalance(currentWalletAddress);
                    console.log('SSE balance response:', balanceResponse);
                    if (balanceResponse.success && balanceResponse.data) {
                      walletBalance = balanceResponse.data.balance_qnk || 0;
                      console.log('SSE updated balance to:', walletBalance);
                    }
                  } catch (balanceErr) {
                    console.warn('Failed to fetch wallet balance:', balanceErr);
                  }
                } else {
                  console.log('SSE: No wallet address available for balance update');
                }

                setNodeStatus({
                  balance: walletBalance,
                  qci: 42,
                  nodeId: response.data.node_id,
                  blockHeight: response.data.current_height,
                  peers: response.data.connected_peers,
                  isConnected: response.data.network_health === 'healthy'
                });
              }
            } catch (err) {
              console.error('Error refreshing node status:', err);
            }
          })();
        }
      } catch (error) {
        console.error('Error processing SSE event:', error);
      }
    };
    
    eventSource.onerror = (error) => {
      console.error('SSE connection error:', error);
    };

    return () => {
      eventSource.close();
    };
  }, [realWalletAddress]); // Re-run when realWalletAddress changes

  const formatBalance = (amount: number, hidden = false) => {
    if (hidden) return '••••••••';
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 8,
    }).format(amount);
  };

  const copyWalletAddress = () => {
    navigator.clipboard.writeText(walletAddress);
    setCopiedAddress(true);
    setTimeout(() => setCopiedAddress(false), 2000);
  };

  const requestFaucetTokens = async () => {
    setFaucetLoading(true);
    setFaucetMessage('');
    
    try {
      // Use the real wallet address for the faucet request
      const result = await qnkAPI.requestFaucet(realWalletAddress);
      
      if (result.success) {
        const receivedAmount = result.data?.amount_qnk || result.data?.new_balance_qnk || 10;
        setFaucetMessage(`Success! Received ${receivedAmount} QNK test tokens`);
        
        // Update local balance display immediately from faucet response
        if (result.data?.new_balance_qnk) {
          setNodeStatus(prev => prev ? {...prev, balance: result.data.new_balance_qnk} : prev);
        }
        
        // SSE will handle real-time balance updates in App.tsx
        
        // Also refresh after a delay to get the latest from server
        setTimeout(async () => {
          try {
            const response = await qnkAPI.getNodeStatus();
            if (response.success && response.data) {
              let walletBalance = 0;
              if (realWalletAddress) {
                try {
                  const balanceResponse = await qnkAPI.getWalletBalance(realWalletAddress);
                  if (balanceResponse.success && balanceResponse.data) {
                    walletBalance = balanceResponse.data.balance_qnk || 0;
                  }
                } catch (balanceErr) {
                  console.warn('Failed to fetch wallet balance:', balanceErr);
                }
              }

              setNodeStatus({
                balance: walletBalance,
                qci: 42,
                nodeId: response.data.node_id,
                blockHeight: response.data.current_height,
                peers: response.data.connected_peers,
                isConnected: response.data.network_health === 'healthy'
              });
              
              // SSE will handle real-time balance updates in App.tsx
            }
          } catch (err) {
            console.error('Error refreshing node status:', err);
          }
        }, 1000);
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
          <p className="text-gray-400 mt-1">
            Quantum Consensus Wallet
          </p>
        </div>
        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-quantum-purple to-quantum-cyan flex items-center justify-center">
          <Activity className={`w-6 h-6 text-white ${nodeStatus.isConnected ? 'animate-pulse' : 'opacity-50'}`} />
        </div>
      </div>

      {/* Wallet Address Card */}
      <motion.div 
        className="bg-gradient-to-br from-quantum-indigo/50 to-quantum-purple/30 backdrop-blur-xl rounded-3xl p-6 quantum-glow relative overflow-hidden"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-xl bg-quantum-cyan/20">
              <Wallet className="w-6 h-6 text-quantum-cyan" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white mb-1">Wallet Address</h3>
              <div className="font-mono text-sm text-gray-300 break-all">
                {walletAddress || 'Generating...'}
              </div>
            </div>
          </div>
          <div className="flex gap-2">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={copyWalletAddress}
              disabled={!walletAddress}
              className="p-3 rounded-xl bg-quantum-purple/20 hover:bg-quantum-purple/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {copiedAddress ? (
                <Check className="w-5 h-5 text-quantum-green" />
              ) : (
                <Copy className="w-5 h-5 text-white" />
              )}
            </motion.button>
            
            {/* Show faucet button if balance is 0 */}
            {nodeStatus && nodeStatus.balance === 0 && (
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={requestFaucetTokens}
                disabled={faucetLoading}
                className="p-3 rounded-xl bg-quantum-green/20 hover:bg-quantum-green/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                title="Get test tokens"
              >
                {faucetLoading ? (
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  >
                    <Coins className="w-5 h-5 text-quantum-green" />
                  </motion.div>
                ) : (
                  <Coins className="w-5 h-5 text-quantum-green" />
                )}
              </motion.button>
            )}
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
        {/* Recent Activity */}
        <motion.div
          className="bg-quantum-indigo/30 backdrop-blur-xl rounded-3xl p-8"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="flex items-center gap-3 mb-6">
            <Zap className="w-6 h-6 text-quantum-yellow" />
            <h3 className="text-xl font-semibold">Recent Activity</h3>
          </div>

          <div className="space-y-4">
            {recentTransactions.length > 0 ? (
              recentTransactions.map((tx, index) => (
                <motion.div
                  key={tx.id}
                  className="flex items-center justify-between p-4 bg-quantum-dark/30 rounded-xl hover:bg-quantum-dark/50 cursor-pointer transition-colors"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 + index * 0.1 }}
                  onClick={() => window.open(`/api/transactions/${tx.txHash}`, '_blank')}
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-3 h-3 rounded-full ${
                      tx.type === 'receive' ? 'bg-quantum-green' : 'bg-quantum-pink'
                    }`} />
                    <div>
                      <div className="font-medium">
                        {tx.type === 'receive' ? 'Received from' : 'Sent to'} {tx.type === 'receive' ? tx.from : tx.to}
                      </div>
                      <div className="text-sm text-gray-400">
                        {new Date(tx.timestamp).toLocaleString()} • {tx.txHash.slice(0, 8)}...
                      </div>
                    </div>
                  </div>
                  <div className={`font-bold ${
                    tx.type === 'receive' ? 'text-quantum-green' : 'text-quantum-pink'
                  }`}>
                    {tx.type === 'receive' ? '+' : '-'}{formatBalance(tx.amount)} QNK
                  </div>
                </motion.div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-400">
                <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>No recent transactions</p>
                <p className="text-sm mt-1">Activity will appear here once the node processes transactions</p>
              </div>
            )}
          </div>
        </motion.div>
      </div>

    </div>
  );
}