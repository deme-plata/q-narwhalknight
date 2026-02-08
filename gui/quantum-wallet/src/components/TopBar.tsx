import { useState, useEffect, useRef, memo, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Copy, Check, ExternalLink, Hash, User, Blocks, Shield, X, Clock, ArrowRight, CheckCircle, XCircle, Key, FileCode, Wifi, Zap, Globe, MessageCircle, Bell, Send, UserCircle, CreditCard, LogOut, BookOpen } from 'lucide-react';
import { TICKER_SYMBOL } from '../constants/ticker';
import { qnkAPI } from '../services/api';
import type { MiningStatsEvent } from '../services/api';
import SmartContractModal from './SmartContractModal';
import NetworkMapModal from './NetworkMapModal';

// v3.6.1-beta: SANITY CHECK - Max possible balance is 21 million QUG (total supply)
// Any balance exceeding this is corrupted data and must be rejected
const MAX_SANE_BALANCE = 21_000_000; // 21 million QUG

/**
 * v3.6.1-beta: Validate balance value to prevent corrupted data from displaying
 * Returns true if the balance is sane, false if it's corrupted
 */
function isValidBalance(balance: number): boolean {
  if (typeof balance !== 'number') return false;
  if (isNaN(balance) || !isFinite(balance)) return false;
  if (balance < 0) return false;
  if (balance > MAX_SANE_BALANCE) {
    console.warn(`🚨 [TopBar] Rejected corrupted balance: ${balance.toExponential()} > max supply ${MAX_SANE_BALANCE}`);
    return false;
  }
  return true;
}

/**
 * v3.6.1-beta: Clear corrupted localStorage balance values
 */
function clearCorruptedBalanceCache(): void {
  const cached = localStorage.getItem('cachedBalance');
  if (cached) {
    const value = parseFloat(cached);
    if (!isValidBalance(value)) {
      console.warn(`🚨 [TopBar] Clearing corrupted localStorage cachedBalance: ${value}`);
      localStorage.removeItem('cachedBalance');
    }
  }
  const locked = localStorage.getItem('dexLockedBalance');
  if (locked) {
    const value = parseFloat(locked);
    if (!isValidBalance(value)) {
      console.warn(`🚨 [TopBar] Clearing corrupted localStorage dexLockedBalance: ${value}`);
      localStorage.removeItem('dexLockedBalance');
    }
  }
}

interface SearchResult {
  type: 'transaction' | 'block' | 'address' | 'node' | 'contract' | 'error';
  id: string;
  title: string;
  subtitle?: string;
  hash?: string;
  data?: any; // Full data for detail view
}

// v3.9.1-beta: Bank Messaging System Types
interface BankMessage {
  id: string;
  from: 'user' | 'bank';
  content: string;
  timestamp: number;
  read: boolean;
  subject?: string;
  loanId?: string;
}

interface LoanDetails {
  id: string;
  amount: number;
  collateral: number;
  interestRate: number;
  status: 'pending' | 'approved' | 'rejected' | 'active' | 'paid' | 'liquidated';
  createdAt: number;
  dueDate?: number;
  remainingBalance?: number;
}

interface TopBarProps {
  currentBalance: number;
  nodeId: string;
  blockHeight: number;
  peers: number;
  isOnline: boolean;
  qci: number; // Quantum Coherence Index
  onNavigate?: (screen: 'dashboard' | 'transactions' | 'explorer' | 'dex' | 'mining' | 'vm' | 'download' | 'aichat' | 'settings') => void;
}

// v2.4.0: Memoized for performance
const TopBar = memo(function TopBar({ currentBalance, nodeId, blockHeight, peers, isOnline, qci, onNavigate }: TopBarProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<SearchResult | null>(null); // v3.4.2: Detail modal
  const [selectedContract, setSelectedContract] = useState<any | null>(null); // v3.4.20: Enhanced contract modal
  const [showNetworkMap, setShowNetworkMap] = useState(false); // v3.4.21: Network map modal

  // v3.9.1-beta: Profile & Banking Messaging System
  const [showProfileModal, setShowProfileModal] = useState(false);
  const [hasActiveLoan, setHasActiveLoan] = useState(false);
  const [unreadMessages, setUnreadMessages] = useState(0);
  const [bankMessages, setBankMessages] = useState<BankMessage[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [isSendingMessage, setIsSendingMessage] = useState(false);
  const [loanDetails, setLoanDetails] = useState<LoanDetails | null>(null);

  // v3.9.2-beta: Universal profile dropdown
  const [copiedWallet, setCopiedWallet] = useState(false);
  const [recentInboxItems, setRecentInboxItems] = useState<any[]>([]);
  const walletAddr = useMemo(() => localStorage.getItem('walletAddress') || '', []);

  // v3.4.16-beta: SSE-updated live metrics
  const [liveBlockHeight, setLiveBlockHeight] = useState(blockHeight);
  const [livePeers, setLivePeers] = useState(peers);
  const [personalHashrate, setPersonalHashrate] = useState<number>(0);
  const [isTorConnected, setIsTorConnected] = useState(false);
  const sseRef = useRef<EventSource | null>(null);

  // v3.6.1-beta: Clear corrupted balance caches on mount
  useEffect(() => {
    clearCorruptedBalanceCache();
  }, []);

  // v2.9.0-beta: STABLE balance display - prevent bouncing between multiple sources
  // v3.6.1-beta: Add sanity check to reject corrupted values
  const [stableBalance, setStableBalance] = useState<number>(() => {
    // Initialize from localStorage cache to prevent flash
    const cached = localStorage.getItem('cachedBalance');
    const cachedValue = cached ? parseFloat(cached) : 0;
    // v3.6.1-beta: Validate cached value before using
    if (isValidBalance(cachedValue)) {
      return cachedValue;
    }
    // If cached is corrupted, try currentBalance
    if (isValidBalance(currentBalance)) {
      return currentBalance;
    }
    // All sources corrupted - start at 0
    return 0;
  });
  const lastBalanceUpdateRef = useRef<number>(Date.now());
  const balanceStabilityWindowMs = 2000; // Don't change balance more than once per 2 seconds

  // v2.9.0-beta: Stabilized balance getter - prevents rapid flickering
  // v3.6.1-beta: Added sanity checks to reject corrupted values
  const getDisplayBalance = (): number => {
    // Check if we have a locked balance from DEX (SOURCE OF TRUTH during cooldown)
    const lockedBalance = localStorage.getItem('dexLockedBalance');
    const cooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');

    if (lockedBalance && Date.now() < cooldownUntil) {
      const locked = parseFloat(lockedBalance);
      // v3.6.1-beta: Validate before returning
      if (isValidBalance(locked)) {
        return locked;
      } else {
        console.warn(`🚨 [TopBar] getDisplayBalance: Rejected corrupted dexLockedBalance: ${locked}`);
        // Clear corrupted value
        localStorage.removeItem('dexLockedBalance');
      }
    }

    // Return the stable balance if valid, otherwise 0
    if (isValidBalance(stableBalance)) {
      return stableBalance;
    }

    console.warn(`🚨 [TopBar] getDisplayBalance: stableBalance is corrupted: ${stableBalance}`);
    return 0;
  };

  // v2.9.0-beta: Update stable balance with debouncing to prevent flickering
  // v3.6.1-beta: Added sanity checks to reject corrupted values
  useEffect(() => {
    const cached = localStorage.getItem('cachedBalance');
    const cachedValue = cached ? parseFloat(cached) : currentBalance;

    // v3.6.1-beta: Only use values that pass sanity check
    const validCached = isValidBalance(cachedValue) ? cachedValue : 0;
    const validCurrent = isValidBalance(currentBalance) ? currentBalance : 0;
    const validStable = isValidBalance(stableBalance) ? stableBalance : 0;

    const newBalance = validCached || validCurrent;

    // Only update if enough time has passed (prevents rapid flickering)
    const timeSinceLastUpdate = Date.now() - lastBalanceUpdateRef.current;
    const balanceDifference = Math.abs(newBalance - validStable);

    // Update if: significant change (>1 QUG) OR stability window passed
    if (balanceDifference > 1 || timeSinceLastUpdate > balanceStabilityWindowMs) {
      // v3.6.1-beta: Use Math.max ONLY on validated values to prevent corrupted values from persisting
      // Filter out any corrupted values before comparing
      const candidates = [newBalance, validStable, validCurrent].filter(v => isValidBalance(v));
      const bestBalance = candidates.length > 0 ? Math.max(...candidates) : 0;

      if (Math.abs(bestBalance - stableBalance) > 0.0001) {
        console.log('💰 TopBar: Stable balance update:', stableBalance.toFixed(4), '→', bestBalance.toFixed(4));
        setStableBalance(bestBalance);
        lastBalanceUpdateRef.current = Date.now();
      }
    }
  }, [currentBalance, stableBalance]);

  // v2.9.0-beta: Listen for balance change events and update stable balance
  // v3.6.1-beta: Added sanity checks to reject corrupted values
  useEffect(() => {
    const handleBalanceChanged = (event: Event) => {
      const customEvent = event as CustomEvent;
      const newBalance = customEvent.detail?.balance;

      // v3.6.1-beta: CRITICAL - Validate balance before using
      if (!isValidBalance(newBalance)) {
        console.warn(`🚨 [TopBar] Rejecting invalid qug-balance-changed: ${newBalance}`);
        return;
      }

      console.log('🔥 TopBar: qug-balance-changed received:', newBalance);
      // v2.9.3-beta: DEX swaps are AUTHORITATIVE - allow both increases AND decreases
      // The qug-balance-changed event is ONLY dispatched by DexScreen after successful swaps
      // so we MUST trust the value even if it's lower (e.g., QUG -> QUGUSD swap)
      setStableBalance(newBalance);
      lastBalanceUpdateRef.current = Date.now();
    };

    const handleDexCooldownExpired = (event: Event) => {
      const customEvent = event as CustomEvent;
      const { qugBalance } = customEvent.detail || {};
      console.log('🔄 TopBar: DEX cooldown expired, balance:', qugBalance);

      // v3.6.1-beta: CRITICAL - Validate balance before using
      if (!isValidBalance(qugBalance)) {
        console.warn(`🚨 [TopBar] Rejecting invalid dex-cooldown-expired balance: ${qugBalance}`);
        return;
      }

      setStableBalance(qugBalance);
      lastBalanceUpdateRef.current = Date.now();
    };

    const handleWalletBalanceUpdated = (event: Event) => {
      const customEvent = event as CustomEvent;
      const { symbol, balance, reason } = customEvent.detail || {};
      if (symbol !== 'QUG') return;

      // v3.6.1-beta: CRITICAL - Validate balance before using
      if (!isValidBalance(balance)) {
        console.warn(`🚨 [TopBar] Rejecting invalid wallet-balance-updated: ${balance} (reason: ${reason})`);
        return;
      }

      console.log('💰 TopBar: wallet-balance-updated:', balance, 'reason:', reason);

      // v2.9.1-beta: Mining updates are authoritative - use directly
      // DEX updates use Math.max() to prevent race condition drops
      const isMiningUpdate = reason && (
        reason === 'mining_reward' ||
        reason === 'p2p_mining_reward' ||
        reason === 'mining_stats_update' ||
        reason.startsWith('mining_reward')
      );

      // v2.9.3-beta: Check if this is a DEX swap deduction
      const isDexSwapDeduct = reason && (
        reason === 'dex-swap-deduct' ||
        reason === 'DexScreen.swap.deduct'
      );

      if (isMiningUpdate || isDexSwapDeduct) {
        // Mining updates and DEX swap deductions: use value directly
        // DEX swaps are authoritative - the user just spent QUG, balance MUST decrease
        console.log(isDexSwapDeduct ? '💸 TopBar: DEX deduct - setting balance to:' : '⛏️ TopBar: Mining update - setting balance to:', balance);
        setStableBalance(balance);
      } else {
        // v3.6.1-beta: Use Math.max only if BOTH values are valid
        setStableBalance(prev => {
          if (!isValidBalance(prev)) return balance;
          return Math.max(prev, balance);
        });
      }
      lastBalanceUpdateRef.current = Date.now();
    };

    window.addEventListener('qug-balance-changed', handleBalanceChanged);
    window.addEventListener('dex-cooldown-expired', handleDexCooldownExpired);
    window.addEventListener('wallet-balance-updated', handleWalletBalanceUpdated);

    return () => {
      window.removeEventListener('qug-balance-changed', handleBalanceChanged);
      window.removeEventListener('dex-cooldown-expired', handleDexCooldownExpired);
      window.removeEventListener('wallet-balance-updated', handleWalletBalanceUpdated);
    };
  }, []);

  // Fetch total personal hashrate from API (sums all workers for this wallet)
  useEffect(() => {
    const walletAddress = localStorage.getItem('walletAddress') || '';
    if (!walletAddress) return;

    const fetchMiningHashrate = async () => {
      try {
        const res = await qnkAPI.getMiningStats(walletAddress);
        if (res.success && res.data) {
          // Only show hashrate if miner is actually active
          if (res.data.is_active && res.data.hash_rate > 0) {
            setPersonalHashrate(res.data.hash_rate);
          } else {
            setPersonalHashrate(0);
          }
        }
      } catch {
        // Mining stats endpoint may not be available
      }
    };

    fetchMiningHashrate();
    const interval = setInterval(fetchMiningHashrate, 15000);
    return () => clearInterval(interval);
  }, []);

  // v3.4.16-beta: Detect Tor connection (.onion domain)
  useEffect(() => {
    if (typeof window !== 'undefined' && window.location) {
      const isTor = window.location.hostname.endsWith('.onion');
      setIsTorConnected(isTor);
      if (isTor) {
        console.log('🧅 [TopBar] Tor hidden service detected');
      }
    }
  }, []);

  // v3.4.16-beta: SSE subscription for live block height, peers, and hashrate
  useEffect(() => {
    const walletAddress = localStorage.getItem('walletAddress') || '';
    if (!walletAddress) return;

    // Create SSE connection for live updates
    const baseUrl = window.location.hostname.endsWith('.onion')
      ? '' // Relative URL for .onion
      : (localStorage.getItem('nodeUrl') || '');
    const sseUrl = `${baseUrl}/api/v1/events?wallet_address=${encodeURIComponent(walletAddress)}`;

    console.log('📡 [TopBar] Connecting to SSE for live metrics...');
    const eventSource = new EventSource(sseUrl);
    sseRef.current = eventSource;

    // Listen for node status updates (block height, peers)
    eventSource.addEventListener('node-status', (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data);
        if (data.current_height) {
          setLiveBlockHeight(data.current_height);
        }
        if (data.connected_peers !== undefined) {
          setLivePeers(data.connected_peers);
        }
      } catch (err) {
        console.error('❌ [TopBar] Failed to parse node-status:', err);
      }
    });

    // Listen for mining stats updates (personal hashrate)
    // The periodic API fetch is authoritative for total; SSE just provides instant updates.
    eventSource.addEventListener('miner-stats', (e: MessageEvent) => {
      try {
        const data: MiningStatsEvent = JSON.parse(e.data);
        const normalizedWallet = walletAddress.replace(/^qnk/, '').toLowerCase();
        const normalizedMiner = (data.miner_address || '').replace(/^qnk/, '').toLowerCase();
        if (normalizedMiner === normalizedWallet && data.avg_hash_rate) {
          setPersonalHashrate(data.avg_hash_rate);
        }
      } catch (err) {
        console.error('❌ [TopBar] Failed to parse miner-stats:', err);
      }
    });

    // Also listen for block height from mining rewards
    eventSource.addEventListener('mining_reward', (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data);
        if (data.block_height) {
          setLiveBlockHeight(data.block_height);
        }
        // Update hashrate from mining reward if available
        if (data.hash_rate) {
          const normalizedWallet = walletAddress.replace(/^qnk/, '').toLowerCase();
          const normalizedMiner = (data.miner_address || '').replace(/^qnk/, '').toLowerCase();
          if (normalizedMiner === normalizedWallet) {
            setPersonalHashrate(data.hash_rate);
          }
        }
      } catch (err) {
        // Ignore parse errors for mining_reward
      }
    });

    eventSource.onerror = () => {
      console.warn('⚠️ [TopBar] SSE connection error, will retry...');
    };

    return () => {
      console.log('📡 [TopBar] Closing SSE connection');
      eventSource.close();
      sseRef.current = null;
    };
  }, []);

  // v3.4.16-beta: Update from props when SSE hasn't provided updates yet
  useEffect(() => {
    if (blockHeight > liveBlockHeight) {
      setLiveBlockHeight(blockHeight);
    }
    if (peers !== livePeers && livePeers === 0) {
      setLivePeers(peers);
    }
  }, [blockHeight, peers]);

  // v3.9.1-beta: Fetch loan status and bank messages
  useEffect(() => {
    const walletAddress = localStorage.getItem('walletAddress') || '';
    if (!walletAddress) return;

    const fetchBankingData = async () => {
      try {
        const baseUrl = localStorage.getItem('nodeUrl') || '';

        // Fetch loan applications for this wallet
        const loanRes = await fetch(`${baseUrl}/api/v1/quillon-bank/lending/applications`);
        if (loanRes.ok) {
          const loans = await loanRes.json();
          // Find active loan for this wallet
          const normalizedWallet = walletAddress.replace(/^qnk/, '').toLowerCase();
          const activeLoan = loans.find((loan: any) => {
            const loanWallet = (loan.borrower_address || loan.borrower || '').replace(/^qnk/, '').toLowerCase();
            return loanWallet === normalizedWallet &&
              (loan.status === 'approved' || loan.status === 'active');
          });

          if (activeLoan) {
            setHasActiveLoan(true);
            setLoanDetails({
              id: activeLoan.id || activeLoan.loan_id,
              amount: activeLoan.amount || activeLoan.loan_amount,
              collateral: activeLoan.collateral || activeLoan.collateral_amount,
              interestRate: activeLoan.interest_rate || 5,
              status: activeLoan.status,
              createdAt: activeLoan.created_at || activeLoan.timestamp || Date.now(),
              dueDate: activeLoan.due_date,
              remainingBalance: activeLoan.remaining_balance || activeLoan.amount,
            });
          }
        }

        // Fetch messages for this wallet
        const msgRes = await fetch(`${baseUrl}/api/v1/quillon-bank/messages/${encodeURIComponent(walletAddress)}`);
        if (msgRes.ok) {
          const messages = await msgRes.json();
          setBankMessages(messages);
          const unread = messages.filter((m: BankMessage) => !m.read && m.from === 'bank').length;
          setUnreadMessages(unread);
        }
      } catch (err) {
        console.log('📬 [TopBar] Banking data fetch (endpoints may not exist yet):', err);
      }
    };

    // Initial fetch
    fetchBankingData();

    // Refresh every 30 seconds
    const interval = setInterval(fetchBankingData, 30000);
    return () => clearInterval(interval);
  }, []);

  // v3.9.1-beta: Send message to bank
  const sendMessageToBank = async () => {
    if (!newMessage.trim() || isSendingMessage) return;

    const walletAddress = localStorage.getItem('walletAddress') || '';
    if (!walletAddress) return;

    setIsSendingMessage(true);
    try {
      const baseUrl = localStorage.getItem('nodeUrl') || '';
      const res = await fetch(`${baseUrl}/api/v1/quillon-bank/messages/send`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wallet_address: walletAddress,
          content: newMessage.trim(),
          subject: loanDetails?.id ? `Re: Loan #${loanDetails.id}` : 'General Inquiry',
        }),
      });

      if (res.ok) {
        const newMsg: BankMessage = {
          id: `msg_${Date.now()}`,
          from: 'user',
          content: newMessage.trim(),
          timestamp: Date.now(),
          read: true,
          loanId: loanDetails?.id,
        };
        setBankMessages(prev => [...prev, newMsg]);
        setNewMessage('');
      }
    } catch (err) {
      console.error('❌ [TopBar] Failed to send message:', err);
    } finally {
      setIsSendingMessage(false);
    }
  };

  // v3.9.2-beta: Fetch recent transactions for inbox
  useEffect(() => {
    if (!walletAddr) return;
    const fetchInbox = async () => {
      try {
        const res = await qnkAPI.getWalletHistory(walletAddr, 10);
        if (res.success && res.data) {
          setRecentInboxItems(res.data.slice(0, 5));
        }
      } catch (err) {
        console.log('📬 [TopBar] Inbox fetch:', err);
      }
    };
    fetchInbox();
    const interval = setInterval(fetchInbox, 60000);
    return () => clearInterval(interval);
  }, [walletAddr]);

  // v3.9.2-beta: Copy wallet address to clipboard
  const copyWalletAddress = () => {
    if (walletAddr) {
      navigator.clipboard.writeText(walletAddr);
      setCopiedWallet(true);
      setTimeout(() => setCopiedWallet(false), 2000);
    }
  };

  // Get the display balance (always from localStorage)
  const displayBalance = getDisplayBalance();

  // v3.4.16-beta: Format hashrate for display
  const formatHashrate = (hashrate: number): string => {
    if (hashrate === 0) return '0 H/s';
    if (hashrate >= 1e12) return `${(hashrate / 1e12).toFixed(2)} TH/s`;
    if (hashrate >= 1e9) return `${(hashrate / 1e9).toFixed(2)} GH/s`;
    if (hashrate >= 1e6) return `${(hashrate / 1e6).toFixed(2)} MH/s`;
    if (hashrate >= 1e3) return `${(hashrate / 1e3).toFixed(2)} KH/s`;
    return `${hashrate.toFixed(0)} H/s`;
  };

  // v3.4.2: Real API search function (same as ExplorerScreen)
  const performSearch = async (query: string) => {
    if (query.length < 2) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    const results: SearchResult[] = [];

    try {
      // Determine search type based on query format
      let searchType = '';
      if (query.match(/^tx_[a-f0-9]+/i) || query.match(/^[a-f0-9]{64}$/i)) searchType = 'transaction';
      else if (query.match(/^vtx_[a-f0-9]+/i)) searchType = 'vertex';
      else if (query.match(/^0x[a-f0-9]{40}$/i)) searchType = 'contract';
      else if (query.match(/^qnk[a-z0-9]{39}$/i)) searchType = 'address';
      else if (query.match(/^\d+$/)) searchType = 'block';

      console.log(`🔍 TopBar searching for: ${query} (type: ${searchType})`);

      if (searchType === 'block') {
        const blockNum = parseInt(query);
        const blockResponse = await qnkAPI.getBlock(blockNum);
        if (blockResponse.success && blockResponse.data) {
          results.push({
            type: 'block',
            id: blockNum.toString(),
            title: `Block #${blockNum}`,
            subtitle: `${Array.isArray(blockResponse.data) ? blockResponse.data.length : 0} transactions`,
            data: {
              height: blockNum,
              tx_count: Array.isArray(blockResponse.data) ? blockResponse.data.length : 0,
              hash: blockResponse.data[0]?.hash || 'N/A',
              transactions: blockResponse.data
            }
          });
        } else {
          results.push({
            type: 'error',
            id: 'not-found',
            title: 'Block Not Found',
            subtitle: `Block #${blockNum} does not exist yet`
          });
        }
      } else if (searchType === 'transaction') {
        const txResponse = await qnkAPI.getTransactionByHash(query);
        if (txResponse.success && txResponse.data) {
          const txData = txResponse.data;
          // v3.4.3: Store raw amount/fee - modal will convert using DECIMALS (1e6)
          results.push({
            type: 'transaction',
            id: query,
            title: 'Transaction Found',
            subtitle: txData.status || 'confirmed',
            hash: txData.hash || query,
            data: {
              hash: txData.hash || query,
              amount: txData.amount || 0,  // Raw value - modal converts
              status: txData.status || 'confirmed',
              timestamp: txData.timestamp ? new Date(txData.timestamp * 1000).toLocaleString() : 'N/A',
              from: txData.from || 'N/A',
              to: txData.to || 'N/A',
              block_height: txData.block_height,
              confirmations: txData.confirmations,
              fee: txData.fee || 0,  // Raw value - modal converts
              token_type: txData.token_type
            }
          });
        } else {
          results.push({
            type: 'error',
            id: 'not-found',
            title: 'Transaction Not Found',
            subtitle: `${query.substring(0, 16)}...${query.substring(48)}`
          });
        }
      } else if (searchType === 'address') {
        // v3.4.3: Try contract lookup first, then wallet lookup
        // Both contracts and wallets use qnk... format
        const contractResponse = await qnkAPI.getContractInfo(query);
        if (contractResponse.success && contractResponse.data && contractResponse.data.name) {
          // Found a contract at this address
          results.push({
            type: 'contract',
            id: query,
            title: contractResponse.data.name || 'Smart Contract',
            subtitle: contractResponse.data.symbol ? `${contractResponse.data.symbol} Token` : 'Contract Address',
            hash: query,
            data: {
              address: query,
              name: contractResponse.data.name,
              symbol: contractResponse.data.symbol,
              contract_type: contractResponse.data.contract_type || 'Unknown',
              total_supply: contractResponse.data.total_supply,
              decimals: contractResponse.data.decimals || 18,
              deployer: contractResponse.data.deployer,
              deployment_height: contractResponse.data.deployment_height,
              verified: contractResponse.data.verified || false
            }
          });
        } else {
          // Not a contract, try as wallet address
          const balanceResponse = await qnkAPI.getWalletBalance(query);
          if (balanceResponse.success && balanceResponse.data) {
            results.push({
              type: 'address',
              id: query,
              title: 'Wallet Address',
              subtitle: `Balance: ${(balanceResponse.data.balance_qnk || 0).toFixed(4)} ${TICKER_SYMBOL}`,
              hash: query,
              data: {
                address: query,
                balance: balanceResponse.data.balance_qnk || 0,
                nonce: balanceResponse.data.nonce || 0
              }
            });
          } else {
            results.push({
              type: 'address',
              id: query,
              title: 'New Wallet Address',
              subtitle: 'Balance: 0 ' + TICKER_SYMBOL,
              hash: query,
              data: { address: query, balance: 0, nonce: 0 }
            });
          }
        }
      } else if (searchType === 'contract') {
        const contractResponse = await qnkAPI.getContractInfo(query);
        if (contractResponse.success && contractResponse.data) {
          results.push({
            type: 'contract',
            id: query,
            title: contractResponse.data.name || 'Smart Contract',
            subtitle: contractResponse.data.symbol ? `${contractResponse.data.symbol} Token` : 'Contract Address',
            hash: query,
            data: {
              address: query,
              name: contractResponse.data.name,
              symbol: contractResponse.data.symbol,
              contract_type: contractResponse.data.contract_type || 'Unknown',
              total_supply: contractResponse.data.total_supply,
              decimals: contractResponse.data.decimals || 18,
              deployer: contractResponse.data.deployer,
              deployment_height: contractResponse.data.deployment_height,
              verified: contractResponse.data.verified || false
            }
          });
        } else {
          // Contract address format but not found - might be newly deployed
          results.push({
            type: 'contract',
            id: query,
            title: 'Unknown Contract',
            subtitle: 'Contract not found or not indexed',
            hash: query,
            data: { address: query }
          });
        }
      } else if (query.length >= 3) {
        // Generic search - show helpful hints
        results.push({
          type: 'block',
          id: 'hint-block',
          title: 'Search by block number',
          subtitle: `Enter a number like "${blockHeight}" to find a block`
        });
        results.push({
          type: 'transaction',
          id: 'hint-tx',
          title: 'Search by transaction hash',
          subtitle: 'Enter a 64-character hex hash'
        });
        results.push({
          type: 'address',
          id: 'hint-address',
          title: 'Search by address',
          subtitle: 'Enter wallet or contract address starting with "qnk"'
        });
      }
    } catch (error) {
      console.error('Search failed:', error);
      results.push({
        type: 'error',
        id: 'error',
        title: 'Search Failed',
        subtitle: 'Unable to connect to the network'
      });
    }

    setSearchResults(results);
    setIsSearching(false);
  };

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (searchQuery.trim()) {
        performSearch(searchQuery);
      } else {
        setSearchResults([]);
      }
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [searchQuery]);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const getQCIStatus = (qci: number) => {
    if (qci >= 0.9) return 'Sublime';
    if (qci >= 0.8) return 'Coherent';
    return 'Stabilizing';
  };

  const getResultIcon = (type: SearchResult['type']) => {
    switch (type) {
      case 'transaction': return <Hash className="w-4 h-4" />;
      case 'block': return <Blocks className="w-4 h-4" />;
      case 'address': return <User className="w-4 h-4" />;
      case 'node': return <User className="w-4 h-4" />;
      case 'contract': return <FileCode className="w-4 h-4" />;
    }
  };

  return (
    <>
    <div
      className="backdrop-blur-xl border-b px-6 py-4 relative z-50"
      style={{
        background: 'linear-gradient(135deg, rgba(30, 20, 60, 0.95) 0%, rgba(50, 30, 80, 0.95) 100%)',
        borderColor: 'rgba(212, 175, 55, 0.2)',
        boxShadow: '0 4px 20px rgba(212, 175, 55, 0.15)'
      }}
    >
      <div className="flex items-center justify-between">
        {/* Left: Search */}
        <div className="flex-1 max-w-md relative">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-amber-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setShowResults(true);
              }}
              onBlur={() => setTimeout(() => setShowResults(false), 200)}
              onFocus={() => setShowResults(true)}
              placeholder="Search transactions, blocks, addresses..."
              className="w-full pl-10 pr-4 py-2 bg-slate-900/70 border-2 border-amber-500/30 rounded-lg text-amber-50 placeholder-amber-300/40 focus:outline-none focus:border-amber-400 focus:shadow-[0_0_15px_rgba(251,191,36,0.3)] transition-all"
            />
            {isSearching && (
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                className="absolute right-3 top-1/2 transform -translate-y-1/2"
              >
                <Search className="w-4 h-4 text-amber-400" />
              </motion.div>
            )}
          </div>

          {/* Search Results Dropdown */}
          <AnimatePresence>
            {showResults && searchResults.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute top-full mt-2 w-full backdrop-blur-xl rounded-lg shadow-2xl max-h-96 overflow-y-auto z-50"
                style={{
                  background: 'linear-gradient(135deg, rgba(30, 20, 60, 0.98) 0%, rgba(50, 30, 80, 0.98) 100%)',
                  border: '2px solid rgba(212, 175, 55, 0.3)',
                  boxShadow: '0 10px 40px rgba(212, 175, 55, 0.2)'
                }}
              >
                {searchResults.map((result, index) => (
                  <motion.div
                    key={`${result.type}-${result.id}-${index}`}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="flex items-center justify-between p-4 border-b last:border-b-0 cursor-pointer group"
                    style={{
                      borderColor: 'rgba(212, 175, 55, 0.1)'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = 'linear-gradient(135deg, rgba(212, 175, 55, 0.1) 0%, rgba(255, 215, 0, 0.05) 100%)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent';
                    }}
                    onClick={() => {
                      console.log('View detail:', result);
                      if (result.data) {
                        // v3.4.20: Use enhanced SmartContractModal for contracts
                        if (result.type === 'contract') {
                          setSelectedContract(result.data);
                        } else {
                          setSelectedDetail(result);
                        }
                      }
                      setShowResults(false);
                      setSearchQuery('');
                    }}
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className="p-2 rounded-lg"
                        style={{
                          background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(255, 215, 0, 0.15))',
                          border: '1px solid rgba(212, 175, 55, 0.3)'
                        }}
                      >
                        <div className="text-amber-400">{getResultIcon(result.type)}</div>
                      </div>
                      <div>
                        <div className="text-amber-100 font-semibold">{result.title}</div>
                        {result.subtitle && (
                          <div className="text-amber-300/60 text-sm">{result.subtitle}</div>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      {result.hash && (
                        <motion.button
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.9 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            copyToClipboard(result.hash!, `${result.type}-${result.id}`);
                          }}
                          className="p-1 text-amber-400/60 hover:text-amber-400 transition-colors"
                        >
                          {copiedId === `${result.type}-${result.id}` ? (
                            <Check className="w-4 h-4 text-green-400" />
                          ) : (
                            <Copy className="w-4 h-4" />
                          )}
                        </motion.button>
                      )}
                      <ExternalLink className="w-4 h-4 text-amber-400/60" />
                    </div>
                  </motion.div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Center: Network Status */}
        <div className="flex items-center gap-6">
          <div className="text-center relative group">
            <motion.div
              className="font-bold text-lg bg-gradient-to-r from-amber-400 via-yellow-500 to-amber-600 bg-clip-text text-transparent cursor-help font-mono"
              key="stable-balance-display"
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              title={`Full 24-decimal precision balance`}
            >
              {/* v3.6.10-beta: Always show full 24 decimal precision */}
              {displayBalance.toFixed(24)} {TICKER_SYMBOL}
            </motion.div>
            <div className="text-amber-300/60 text-sm font-medium flex items-center gap-2 justify-center">
              <span>Total Balance</span>
              <motion.div
                className="w-1.5 h-1.5 rounded-full bg-green-400"
                animate={{ scale: [1, 1.3, 1], opacity: [0.7, 1, 0.7] }}
                transition={{ duration: 2, repeat: Infinity }}
                title="Live updates enabled"
              />
            </div>
          </div>

          <div className="h-8 w-px bg-gradient-to-b from-transparent via-amber-500/30 to-transparent" />

          <div className="flex items-center gap-3">
            {/* v3.4.16-beta: Tor privacy indicator */}
            {isTorConnected && (
              <motion.div
                className="flex items-center gap-1.5 px-2 py-1 bg-purple-500/20 border border-purple-500/40 rounded-lg"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                title="Connected via Tor Hidden Service - Enhanced Privacy"
              >
                <span className="text-lg">🧅</span>
                <span className="text-purple-300 text-xs font-medium">Tor</span>
              </motion.div>
            )}

            <motion.div
              className={`w-2 h-2 rounded-full ${isOnline ? 'bg-green-400' : 'bg-red-500'}`}
              animate={isOnline ? { scale: [1, 1.2, 1], opacity: [0.7, 1, 0.7] } : {}}
              transition={{ duration: 2, repeat: Infinity }}
            />
            <div className="text-amber-100 text-sm font-medium">
              Block #{liveBlockHeight.toLocaleString()} •{' '}
              <motion.button
                className="inline-flex items-center gap-1 px-2 py-0.5 bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 rounded-lg transition-colors cursor-pointer"
                onClick={() => setShowNetworkMap(true)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                title="Click to view P2P network map with Tor visualization"
              >
                <Globe className="w-3 h-3 text-amber-400" />
                <span>{livePeers} peers</span>
              </motion.button>
            </div>

            {/* v3.4.16-beta: Personal hashrate display */}
            {personalHashrate > 0 && (
              <motion.div
                className="flex items-center gap-1.5 px-2 py-1 bg-cyan-500/20 border border-cyan-500/40 rounded-lg"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                title="Your Personal Mining Hashrate"
              >
                <Zap className="w-3.5 h-3.5 text-cyan-400" />
                <span className="text-cyan-300 text-xs font-medium">{formatHashrate(personalHashrate)}</span>
              </motion.div>
            )}
          </div>
        </div>

        {/* Right: Coherence Index */}
        <div className="flex items-center gap-3">
          <motion.div
            animate={{
              scale: qci >= 0.9 ? [1, 1.1, 1] : [1, 1.05, 1],
              rotate: [0, 360],
            }}
            transition={{
              scale: { duration: 2, repeat: Infinity },
              rotate: { duration: qci >= 0.9 ? 20 : 40, repeat: Infinity, ease: "linear" }
            }}
          >
            <Shield className="w-5 h-5 text-amber-400" />
          </motion.div>
          <div className="text-right">
            <div className="flex items-center gap-2">
              <motion.span
                className="text-amber-100 text-sm font-bold"
                key={qci}
                initial={{ scale: 1.2, opacity: 0.5 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.3 }}
              >
                {(qci * 100).toFixed(0)}%
              </motion.span>
              <motion.span
                className="text-xs font-semibold bg-gradient-to-r from-amber-400 to-yellow-500 bg-clip-text text-transparent"
                key={getQCIStatus(qci)}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4 }}
              >
                {getQCIStatus(qci)}
              </motion.span>
            </div>
            <div className="text-amber-300/60 text-xs font-medium">Quantum Coherence</div>
          </div>
          <div
            className="relative w-12 h-2 rounded-full overflow-hidden"
            style={{
              background: 'rgba(15, 23, 42, 0.7)',
              border: '1px solid rgba(212, 175, 55, 0.3)',
              boxShadow: qci >= 0.9
                ? '0 0 15px rgba(212, 175, 55, 0.6), 0 0 30px rgba(255, 215, 0, 0.4)'
                : qci >= 0.8
                ? '0 0 10px rgba(212, 175, 55, 0.4)'
                : '0 0 5px rgba(212, 175, 55, 0.2)'
            }}
          >
            <motion.div
              className="h-full relative"
              style={{
                background: qci >= 0.9
                  ? 'linear-gradient(90deg, #FFD700, #FFA500, #FF8C00, #FFD700)'
                  : qci >= 0.8
                  ? 'linear-gradient(90deg, #D4AF37, #FFD700, #FFA500)'
                  : 'linear-gradient(90deg, #8B7355, #D4AF37, #FFD700)',
                backgroundSize: '200% 100%'
              }}
              initial={{ width: 0 }}
              animate={{
                width: `${qci * 100}%`,
                backgroundPosition: qci >= 0.8 ? ['0% 0%', '200% 0%'] : '0% 0%'
              }}
              transition={{
                width: { duration: 1, ease: "easeOut" },
                backgroundPosition: {
                  duration: 2,
                  repeat: Infinity,
                  ease: "linear"
                }
              }}
            >
              {/* Shimmer effect for high coherence */}
              {qci >= 0.8 && (
                <motion.div
                  className="absolute inset-0"
                  style={{
                    background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent)',
                  }}
                  animate={{
                    x: ['-100%', '200%']
                  }}
                  transition={{
                    duration: 1.5,
                    repeat: Infinity,
                    ease: "linear"
                  }}
                />
              )}
            </motion.div>
          </div>

          {/* v3.9.2-beta: Profile Icon - Always visible for all users */}
          <div className="w-px h-8 bg-amber-500/30 mx-2" />
          <div className="relative">
            <motion.button
              onClick={() => setShowProfileModal(!showProfileModal)}
              className="relative p-2 rounded-lg bg-gradient-to-br from-amber-500/20 to-yellow-500/20 border border-amber-500/40 hover:border-amber-400/60 transition-all"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              title="Profile"
            >
              <UserCircle className="w-5 h-5 text-amber-400" />
              {unreadMessages > 0 && (
                <motion.div
                  className="absolute -top-1 -right-1 bg-red-500 text-white text-xs font-bold rounded-full min-w-[18px] h-[18px] flex items-center justify-center px-1"
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: 'spring', stiffness: 500 }}
                >
                  {unreadMessages > 9 ? '9+' : unreadMessages}
                </motion.div>
              )}
            </motion.button>
          </div>
        </div>
      </div>
    </div>

    {/* v3.9.2-beta: Universal Profile Dropdown Panel - Portal to escape AnimatedBorder z-index */}
    {createPortal(
    <AnimatePresence>
      {showProfileModal && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-[10000]"
            onClick={() => setShowProfileModal(false)}
          />
          {/* Dropdown Panel */}
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="fixed top-16 right-4 w-80 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-2 border-amber-500/30 rounded-2xl shadow-2xl z-[10001] overflow-hidden"
            style={{ boxShadow: '0 0 40px rgba(212, 175, 55, 0.15)' }}
          >
            {/* Wallet Section */}
            <div className="p-4 border-b border-amber-500/20">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2 rounded-lg bg-gradient-to-br from-amber-500/20 to-yellow-500/20">
                  <UserCircle className="w-6 h-6 text-amber-400" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-base font-bold text-amber-100">My Wallet</h3>
                  <div className="flex items-center gap-2">
                    <span className="text-amber-300/60 text-xs font-mono truncate">
                      {walletAddr ? `${walletAddr.slice(0, 12)}...${walletAddr.slice(-6)}` : 'Not connected'}
                    </span>
                    {walletAddr && (
                      <button onClick={copyWalletAddress} className="text-amber-400 hover:text-amber-300 transition-colors flex-shrink-0">
                        {copiedWallet ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                      </button>
                    )}
                  </div>
                </div>
              </div>
              <div className="flex items-center justify-between bg-slate-800/60 rounded-lg p-3">
                <span className="text-slate-400 text-sm">Balance</span>
                <span className="text-amber-100 font-bold font-mono">
                  {displayBalance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })} {TICKER_SYMBOL}
                </span>
              </div>
            </div>

            {/* Network Status */}
            <div className="px-4 py-3 border-b border-amber-500/20">
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="flex items-center gap-2">
                  <Blocks className="w-3.5 h-3.5 text-amber-400" />
                  <span className="text-slate-400 text-xs">Block:</span>
                  <span className="text-amber-100 text-xs font-mono">{liveBlockHeight.toLocaleString()}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Globe className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-slate-400 text-xs">Peers:</span>
                  <span className="text-emerald-100 text-xs font-mono">{livePeers}</span>
                </div>
                {personalHashrate > 0 && (
                  <div className="flex items-center gap-2 col-span-2">
                    <Zap className="w-3.5 h-3.5 text-yellow-400" />
                    <span className="text-slate-400 text-xs">Mining:</span>
                    <span className="text-yellow-100 text-xs font-mono">{formatHashrate(personalHashrate)}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Inbox / Recent Transactions */}
            <div className="px-4 py-3 border-b border-amber-500/20">
              <div className="flex items-center gap-2 mb-2">
                <Bell className="w-3.5 h-3.5 text-amber-400" />
                <span className="text-amber-300 font-medium text-sm">Recent Activity</span>
              </div>
              {recentInboxItems.length === 0 ? (
                <div className="text-center text-slate-500 py-3 text-xs">
                  No recent transactions
                </div>
              ) : (
                <div className="space-y-1.5 max-h-[140px] overflow-y-auto">
                  {recentInboxItems.map((item: any, i: number) => (
                    <div key={item.id || i} className="flex items-center gap-2 p-2 rounded-lg bg-slate-800/30 text-xs">
                      <div className={`w-2 h-2 rounded-full flex-shrink-0 ${
                        item.memo && item.direction === 'received' ? 'bg-cyan-400' :
                        item.direction === 'received' ? 'bg-emerald-400' :
                        item.tx_type === 'swap' ? 'bg-blue-400' :
                        item.tx_type === 'mining_reward' ? 'bg-yellow-400' :
                        'bg-amber-400'
                      }`} />
                      <div className="flex-1 min-w-0">
                        <div className="text-slate-200 truncate">
                          {item.direction === 'received' ? 'Received' :
                           item.tx_type === 'swap' ? 'Swap' :
                           item.tx_type === 'mining_reward' ? 'Mining Reward' :
                           'Sent'}{' '}
                          <span className="font-mono text-amber-300">
                            {parseFloat(item.amount || '0').toLocaleString(undefined, { maximumFractionDigits: 4 })}
                          </span>{' '}
                          {item.token_symbol || TICKER_SYMBOL}
                        </div>
                        {item.memo && (
                          <div className="text-cyan-300/80 truncate flex items-center gap-1 mt-0.5">
                            <MessageCircle className="w-2.5 h-2.5 flex-shrink-0" />
                            <span className="truncate">{item.memo}</span>
                          </div>
                        )}
                        <div className="text-slate-500">Block #{item.block_height?.toLocaleString()}</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Loan Section - Only shows if user has active loan */}
            {hasActiveLoan && loanDetails && (
              <div className="px-4 py-3 border-b border-amber-500/20">
                <div className="flex items-center gap-2 mb-2">
                  <CreditCard className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-emerald-300 font-medium text-sm">Active Loan</span>
                  <span className={`ml-auto px-2 py-0.5 rounded-full text-[10px] font-bold ${
                    loanDetails.status === 'active' ? 'bg-emerald-500/20 text-emerald-400' :
                    loanDetails.status === 'approved' ? 'bg-blue-500/20 text-blue-400' :
                    'bg-amber-500/20 text-amber-400'
                  }`}>
                    {loanDetails.status.toUpperCase()}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-slate-400">Amount:</span>
                    <span className="text-white ml-1 font-mono">{loanDetails.amount.toLocaleString()} {TICKER_SYMBOL}</span>
                  </div>
                  <div>
                    <span className="text-slate-400">Remaining:</span>
                    <span className="text-amber-400 ml-1 font-mono">{(loanDetails.remainingBalance || loanDetails.amount).toLocaleString()}</span>
                  </div>
                </div>
                {bankMessages.length > 0 && (
                  <div className="mt-2 flex items-center gap-2 text-xs text-emerald-400 cursor-pointer hover:text-emerald-300">
                    <MessageCircle className="w-3 h-3" />
                    <span>{bankMessages.length} bank message{bankMessages.length !== 1 ? 's' : ''}</span>
                    {unreadMessages > 0 && <span className="text-red-400">({unreadMessages} new)</span>}
                  </div>
                )}
              </div>
            )}

            {/* Quick Links */}
            <div className="p-2">
              <button
                onClick={() => { onNavigate?.('transactions'); setShowProfileModal(false); }}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-slate-700/50 text-slate-300 hover:text-amber-100 transition-colors text-sm"
              >
                <Clock className="w-4 h-4 text-amber-400/70" />
                <span>Transaction History</span>
              </button>
              <button
                onClick={() => { onNavigate?.('dex'); setShowProfileModal(false); }}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-slate-700/50 text-slate-300 hover:text-amber-100 transition-colors text-sm"
              >
                <ArrowRight className="w-4 h-4 text-amber-400/70" />
                <span>DEX Trading</span>
              </button>
              <button
                onClick={() => { onNavigate?.('mining'); setShowProfileModal(false); }}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-slate-700/50 text-slate-300 hover:text-amber-100 transition-colors text-sm"
              >
                <Zap className="w-4 h-4 text-amber-400/70" />
                <span>Mining</span>
              </button>
              <button
                onClick={() => { onNavigate?.('settings'); setShowProfileModal(false); }}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-slate-700/50 text-slate-300 hover:text-amber-100 transition-colors text-sm"
              >
                <Shield className="w-4 h-4 text-amber-400/70" />
                <span>Settings</span>
              </button>
              <div className="border-t border-slate-700/50 mt-1 pt-1">
                <button
                  onClick={() => {
                    localStorage.removeItem('walletAddress');
                    localStorage.removeItem('cachedBalance');
                    localStorage.removeItem('authToken');
                    window.location.reload();
                  }}
                  className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-red-500/10 text-slate-400 hover:text-red-400 transition-colors text-sm"
                >
                  <LogOut className="w-4 h-4" />
                  <span>Disconnect Wallet</span>
                </button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>,
    document.body
    )}

    {/* v3.4.2: Detail Modal for search results - OUTSIDE TopBar div for proper z-index */}
    <AnimatePresence>
      {selectedDetail && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-[9999] p-4 overflow-y-auto"
          onClick={() => setSelectedDetail(null)}
        >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-2 border-amber-500/30 rounded-2xl p-6 max-w-lg w-full shadow-2xl max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
              style={{ boxShadow: '0 0 40px rgba(212, 175, 55, 0.2)' }}
            >
              {/* Modal Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${selectedDetail.type === 'contract' ? 'bg-purple-500/20' : 'bg-amber-500/20'}`}>
                    {selectedDetail.type === 'transaction' && <Hash className="w-6 h-6 text-amber-400" />}
                    {selectedDetail.type === 'block' && <Blocks className="w-6 h-6 text-amber-400" />}
                    {selectedDetail.type === 'address' && <User className="w-6 h-6 text-amber-400" />}
                    {selectedDetail.type === 'contract' && <FileCode className="w-6 h-6 text-purple-400" />}
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-amber-100">{selectedDetail.title}</h3>
                    <p className="text-amber-300/60 text-sm">{selectedDetail.type.toUpperCase()}</p>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedDetail(null)}
                  className="p-2 hover:bg-amber-500/20 rounded-lg transition-colors"
                >
                  <X className="w-5 h-5 text-amber-400" />
                </button>
              </div>

              {/* Transaction Details - Show based on what data is actually available */}
              {selectedDetail.type === 'transaction' && selectedDetail.data && (
                <div className="space-y-4">
                  <div className="flex items-center gap-2 p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span className="text-green-400 font-medium capitalize">{selectedDetail.data.status}</span>
                    {selectedDetail.data.confirmations && (
                      <span className="text-green-300/60 text-sm">({selectedDetail.data.confirmations} confirmations)</span>
                    )}
                  </div>

                  <div className="grid gap-3">
                    {/* Check if we have full access (from/to are populated) */}
                    {(() => {
                      // v3.4.3: Fix hasFullAccess check - 'N/A' is truthy but means no access
                      const hasFullAccess = selectedDetail.data.from && selectedDetail.data.to &&
                                           selectedDetail.data.from !== 'N/A' && selectedDetail.data.to !== 'N/A';
                      // v3.4.3: Convert raw amounts to human-readable (1 QUG = 10^24 raw units - 24 decimal precision)
                      const DECIMALS = 1e24;
                      const humanAmount = selectedDetail.data.amount ? selectedDetail.data.amount / DECIMALS : 0;
                      const humanFee = selectedDetail.data.fee ? selectedDetail.data.fee / DECIMALS : 0;
                      return (
                        <>
                          {/* Amount - only show if we have full access AND amount exists */}
                          {hasFullAccess && humanAmount > 0 && (
                            <div className="p-3 bg-slate-800/50 rounded-lg">
                              <div className="text-amber-300/60 text-xs mb-1">Amount</div>
                              <div className="text-xl font-bold text-amber-100">{humanAmount.toFixed(6)} {TICKER_SYMBOL}</div>
                            </div>
                          )}

                          <div className="p-3 bg-slate-800/50 rounded-lg">
                            <div className="text-amber-300/60 text-xs mb-1">Transaction Hash</div>
                            <div className="flex items-center gap-2">
                              <code className="text-amber-100 text-xs font-mono break-all">{selectedDetail.data.hash}</code>
                              <button
                                onClick={() => copyToClipboard(selectedDetail.data.hash, 'modal-hash')}
                                className="p-1 hover:bg-amber-500/20 rounded transition-colors"
                              >
                                {copiedId === 'modal-hash' ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4 text-amber-400" />}
                              </button>
                            </div>
                          </div>

                          {/* From/To - only show if we have full access */}
                          {hasFullAccess ? (
                            <div className="grid grid-cols-2 gap-3">
                              <div className="p-3 bg-slate-800/50 rounded-lg">
                                <div className="text-amber-300/60 text-xs mb-1">From</div>
                                <code className="text-amber-100 text-xs font-mono break-all">
                                  {selectedDetail.data.from || 'Coinbase (Mining)'}
                                </code>
                              </div>
                              <div className="p-3 bg-slate-800/50 rounded-lg">
                                <div className="text-amber-300/60 text-xs mb-1">To</div>
                                <code className="text-amber-100 text-xs font-mono break-all">
                                  {selectedDetail.data.to}
                                </code>
                              </div>
                            </div>
                          ) : null}

                          <div className="grid grid-cols-2 gap-3">
                            <div className="p-3 bg-slate-800/50 rounded-lg">
                              <div className="text-amber-300/60 text-xs mb-1">Block</div>
                              <div className="text-amber-100 font-medium">#{selectedDetail.data.block_height || 'Pending'}</div>
                            </div>
                            {selectedDetail.data.timestamp && (
                              <div className="p-3 bg-slate-800/50 rounded-lg">
                                <div className="text-amber-300/60 text-xs mb-1">Time</div>
                                <div className="text-amber-100 font-medium text-xs">{selectedDetail.data.timestamp}</div>
                              </div>
                            )}
                          </div>

                          {/* Fee - only show if we have full access AND fee exists */}
                          {hasFullAccess && humanFee > 0 && (
                            <div className="p-3 bg-slate-800/50 rounded-lg">
                              <div className="text-amber-300/60 text-xs mb-1">Fee</div>
                              <div className="text-amber-100 font-medium">{humanFee.toFixed(6)} {TICKER_SYMBOL}</div>
                            </div>
                          )}

                          {/* Show appropriate notice based on access level */}
                          {hasFullAccess ? (
                            <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                              <div className="flex items-center gap-2 text-amber-300 text-sm">
                                <Key className="w-4 h-4" />
                                <span>Authenticated - full transaction details visible</span>
                              </div>
                            </div>
                          ) : (
                            <div className="p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                              <div className="flex items-center gap-2 text-purple-300 text-sm">
                                <Shield className="w-4 h-4" />
                                <span>ZK-STARK Privacy: Only sender/receiver can view full details</span>
                              </div>
                            </div>
                          )}
                        </>
                      );
                    })()}
                  </div>
                </div>
              )}

              {/* Block Details */}
              {selectedDetail.type === 'block' && selectedDetail.data && (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Block Height</div>
                      <div className="text-xl font-bold text-amber-100">#{selectedDetail.data.height}</div>
                    </div>
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Transactions</div>
                      <div className="text-xl font-bold text-amber-100">{selectedDetail.data.tx_count}</div>
                    </div>
                  </div>

                  {selectedDetail.data.hash && selectedDetail.data.hash !== 'N/A' && (
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Block Hash</div>
                      <div className="flex items-center gap-2">
                        <code className="text-amber-100 text-xs font-mono break-all">{selectedDetail.data.hash}</code>
                        <button
                          onClick={() => copyToClipboard(selectedDetail.data.hash, 'modal-block-hash')}
                          className="p-1 hover:bg-amber-500/20 rounded transition-colors"
                        >
                          {copiedId === 'modal-block-hash' ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4 text-amber-400" />}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Address Details */}
              {selectedDetail.type === 'address' && selectedDetail.data && (
                <div className="space-y-4">
                  {/* Wallet Badge */}
                  <div className="flex items-center gap-2 p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                    <User className="w-5 h-5 text-amber-400" />
                    <span className="text-amber-400 font-medium">Wallet Address</span>
                    {selectedDetail.data.balance > 0 && (
                      <span className="ml-auto px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded-full">Active</span>
                    )}
                  </div>

                  {/* Address */}
                  <div className="p-3 bg-slate-800/50 rounded-lg">
                    <div className="text-amber-300/60 text-xs mb-1">Address</div>
                    <div className="flex items-center gap-2">
                      <code className="text-amber-100 text-xs font-mono break-all">{selectedDetail.data.address}</code>
                      <button
                        onClick={() => copyToClipboard(selectedDetail.data.address, 'modal-address')}
                        className="p-1 hover:bg-amber-500/20 rounded transition-colors"
                      >
                        {copiedId === 'modal-address' ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4 text-amber-400" />}
                      </button>
                    </div>
                  </div>

                  {/* Balance & Nonce */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Balance</div>
                      <div className="text-xl font-bold text-amber-100">{selectedDetail.data.balance?.toFixed(6)} {TICKER_SYMBOL}</div>
                    </div>
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Nonce</div>
                      <div className="text-xl font-bold text-amber-100">{selectedDetail.data.nonce}</div>
                    </div>
                  </div>

                  {/* Wallet Info Notice */}
                  <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                    <div className="flex items-center gap-2 text-amber-300 text-sm">
                      <User className="w-4 h-4" />
                      <span>External wallet on Q-NarwhalKnight Network</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Smart Contract Details */}
              {selectedDetail.type === 'contract' && selectedDetail.data && (
                <div className="space-y-4">
                  {/* Contract Badge */}
                  <div className="flex items-center gap-2 p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                    <FileCode className="w-5 h-5 text-purple-400" />
                    <span className="text-purple-400 font-medium">{selectedDetail.data.contract_type || 'Smart Contract'}</span>
                    {selectedDetail.data.verified && (
                      <span className="ml-auto px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded-full">Verified</span>
                    )}
                  </div>

                  {/* Contract Address */}
                  <div className="p-3 bg-slate-800/50 rounded-lg">
                    <div className="text-amber-300/60 text-xs mb-1">Contract Address</div>
                    <div className="flex items-center gap-2">
                      <code className="text-amber-100 text-xs font-mono break-all">{selectedDetail.data.address}</code>
                      <button
                        onClick={() => copyToClipboard(selectedDetail.data.address, 'modal-contract')}
                        className="p-1 hover:bg-amber-500/20 rounded transition-colors"
                      >
                        {copiedId === 'modal-contract' ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4 text-amber-400" />}
                      </button>
                    </div>
                  </div>

                  {/* Token Info (if token contract) */}
                  {selectedDetail.data.symbol && (
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 bg-slate-800/50 rounded-lg">
                        <div className="text-amber-300/60 text-xs mb-1">Token Name</div>
                        <div className="text-lg font-bold text-amber-100">{selectedDetail.data.name}</div>
                      </div>
                      <div className="p-3 bg-slate-800/50 rounded-lg">
                        <div className="text-amber-300/60 text-xs mb-1">Symbol</div>
                        <div className="text-lg font-bold text-amber-100">{selectedDetail.data.symbol}</div>
                      </div>
                    </div>
                  )}

                  {/* Supply & Decimals */}
                  {selectedDetail.data.total_supply !== undefined && (
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 bg-slate-800/50 rounded-lg">
                        <div className="text-amber-300/60 text-xs mb-1">Total Supply</div>
                        <div className="text-amber-100 font-medium">
                          {Number(selectedDetail.data.total_supply).toLocaleString()}
                        </div>
                      </div>
                      <div className="p-3 bg-slate-800/50 rounded-lg">
                        <div className="text-amber-300/60 text-xs mb-1">Decimals</div>
                        <div className="text-amber-100 font-medium">{selectedDetail.data.decimals}</div>
                      </div>
                    </div>
                  )}

                  {/* Deployer & Block */}
                  {selectedDetail.data.deployer && (
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Deployed By</div>
                      <code className="text-amber-100 text-xs font-mono break-all">{selectedDetail.data.deployer}</code>
                    </div>
                  )}

                  {selectedDetail.data.deployment_height && (
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Deployment Block</div>
                      <div className="text-amber-100 font-medium">#{selectedDetail.data.deployment_height.toLocaleString()}</div>
                    </div>
                  )}

                  {/* Contract Info Notice */}
                  <div className="p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                    <div className="flex items-center gap-2 text-purple-300 text-sm">
                      <FileCode className="w-4 h-4" />
                      <span>Smart Contract on Q-NarwhalKnight Network</span>
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* v3.4.20: Enhanced Smart Contract Modal (Polygonscan-inspired) */}
      <SmartContractModal
        isOpen={!!selectedContract}
        onClose={() => setSelectedContract(null)}
        contractData={selectedContract || {}}
      />

      {/* v3.4.21: P2P Network Map Modal with Tor visualization */}
      <NetworkMapModal
        isOpen={showNetworkMap}
        onClose={() => setShowNetworkMap(false)}
        peers={livePeers}
        blockHeight={liveBlockHeight}
      />
    </>
  );
});

export default TopBar;