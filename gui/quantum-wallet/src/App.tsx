import { useState, useEffect, useRef } from 'react';
import LoginScreen from './components/LoginScreen';
import Dashboard from './components/Dashboard';
import TransactionScreenV2 from './components/TransactionScreenV2';
import ExplorerScreen from './components/ExplorerScreen';
import DexScreen from './components/DexScreen';
import MiningScreen from './components/MiningScreen';
import VittuaVMScreen from './components/VittuaVMScreen';
import DownloadNodeScreen from './components/DownloadNodeScreen';
import AIChatScreen from './components/AIChatScreen';
import SettingsScreen from './components/SettingsScreen';
import Navigation from './components/Navigation';
import TopBar from './components/TopBar';
import TokenBar from './components/TokenBar';
import QuantumBackground from './components/QuantumBackground';
import AIWorkerDemo from './components/AIWorkerDemo';
import AnimatedBorder from './components/AnimatedBorder';
import './App.css';

// v3.6.1-beta: SANITY CHECK - Max possible balance is 21 million QUG (total supply)
// Any balance exceeding this is corrupted data and must be rejected
const MAX_SANE_BALANCE = 21_000_000; // 21 million QUG

/**
 * v3.6.1-beta: Validate balance value to prevent corrupted data from being stored
 * Returns true if the balance is sane, false if it's corrupted
 */
function isValidBalance(balance: number): boolean {
  if (typeof balance !== 'number') return false;
  if (isNaN(balance) || !isFinite(balance)) return false;
  if (balance < 0) return false;
  if (balance > MAX_SANE_BALANCE) {
    console.warn(`🚨 [App] Rejected corrupted balance: ${balance.toExponential()} > max supply ${MAX_SANE_BALANCE}`);
    return false;
  }
  return true;
}

/**
 * v3.6.1-beta: Safe localStorage set for cachedBalance - validates before storing
 */
function safeCacheBalance(balance: number): void {
  if (isValidBalance(balance)) {
    localStorage.setItem('cachedBalance', balance.toString());
  } else {
    console.warn(`🚨 [App] safeCacheBalance: Refusing to cache invalid balance: ${balance}`);
  }
}

type Screen = 'dashboard' | 'transactions' | 'explorer' | 'dex' | 'mining' | 'vm' | 'download' | 'aichat' | 'settings';

function App() {
  console.log('🚀 App function executing - TOP OF FUNCTION');

  // v2.4.0: Performance mode state - disables heavy effects (DEFAULT: ON for better UX)
  const [performanceMode, setPerformanceMode] = useState(() => {
    const saved = localStorage.getItem('performanceMode');
    // Default to true if not set
    return saved === null ? true : saved === 'true';
  });

  // v2.4.0: Apply performance mode on initial load and listen for changes
  useEffect(() => {
    if (performanceMode) {
      document.documentElement.classList.add('performance-mode');
    } else {
      document.documentElement.classList.remove('performance-mode');
    }
  }, [performanceMode]);

  // Listen for performance mode changes from Settings
  useEffect(() => {
    const handleStorageChange = () => {
      const newMode = localStorage.getItem('performanceMode') === 'true';
      setPerformanceMode(newMode);
    };
    window.addEventListener('storage', handleStorageChange);
    // Also listen for custom event from same tab
    const handlePerfChange = () => handleStorageChange();
    window.addEventListener('performance-mode-changed', handlePerfChange);
    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('performance-mode-changed', handlePerfChange);
    };
  }, []);

  // Load authentication state from localStorage on mount
  const [authenticated, setAuthenticated] = useState(() => {
    const saved = localStorage.getItem('authenticated');
    console.log('🔐 Initializing authenticated state:', saved);
    return saved === 'true';
  });
  const [currentScreen, setCurrentScreen] = useState<Screen>(() => {
    console.log('🎬 Initializing currentScreen to dashboard');
    return 'dashboard';
  });
  // CRITICAL FIX v0.9.44-beta: Initialize balance from cached value for instant display
  // This prevents balance showing as zero while waiting for API/SSE
  // 🚨 v2.3.7-beta: Handle NaN from parseFloat and ensure valid number
  // v3.6.1-beta: Add sanity check for max balance to prevent corrupted values
  const [nodeData, setNodeData] = useState(() => {
    const cachedBalance = localStorage.getItem('cachedBalance');
    let initialBalance = cachedBalance ? parseFloat(cachedBalance) : 0;
    // Guard against NaN and corrupted values - parseFloat returns NaN for invalid strings
    if (!isValidBalance(initialBalance)) {
      console.warn(`🚨 [App] Rejecting corrupted cached balance: ${initialBalance}`);
      // Clear corrupted cache
      if (cachedBalance) localStorage.removeItem('cachedBalance');
      initialBalance = 0;
    }
    console.log('⚡ App.tsx: Initializing balance from cache:', {
      raw: cachedBalance,
      parsed: initialBalance,
      type: typeof initialBalance
    });

    return {
      balance: initialBalance,
      nodeId: '',
      blockHeight: 0,
      peers: 0,
      isOnline: false,
      qci: 0.10, // Quantum Coherence Index - starts low, calculated dynamically
    };
  });

  // Debounce balance updates to prevent flickering
  const [pendingBalanceUpdate, setPendingBalanceUpdate] = useState<number | null>(null);

  // v2.3.11-beta: Track when DEX swap just happened to ignore stale SSE updates
  // SSE balance updates from server can be stale and overwrite correct DEX swap balance
  const dexSwapInProgressRef = useRef(false);

  // Debug: Log whenever currentScreen changes
  useEffect(() => {
    console.log('📺 Current screen changed to:', currentScreen);
  }, [currentScreen]);

  // Log when App mounts
  useEffect(() => {
    console.log('🏗️ App component MOUNTED');
    return () => {
      console.log('💥 App component UNMOUNTING');
    };
  }, []);


  // Save authentication state to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('authenticated', String(authenticated));
  }, [authenticated]);

  // v2.9.24-beta: FAST balance updates for better UX when receiving coins
  // Balance INCREASES: Apply immediately (instant feedback when receiving)
  // Balance DECREASES: Small 50ms debounce to prevent flickering
  useEffect(() => {
    if (pendingBalanceUpdate === null) return;

    // v2.3.31-beta: Check BOTH local ref AND global localStorage cooldown
    const globalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
    const globalCooldownActive = Date.now() < globalCooldownUntil;
    if (dexSwapInProgressRef.current || globalCooldownActive) {
      console.log('🚫 [BALANCE DEBUG] Ignoring debounced update during DEX cooldown (global:', globalCooldownActive, ')');
      setPendingBalanceUpdate(null);
      return;
    }

    const isBalanceIncrease = pendingBalanceUpdate > nodeData.balance;

    // v2.9.24-beta: INSTANT updates for receiving coins (balance increases)
    if (isBalanceIncrease) {
      console.log('⚡ [BALANCE DEBUG] INSTANT balance increase (receiving coins):', {
        oldBalance: nodeData.balance,
        newBalance: pendingBalanceUpdate,
        increase: pendingBalanceUpdate - nodeData.balance
      });
      setNodeData(prev => ({ ...prev, balance: pendingBalanceUpdate }));
      safeCacheBalance(pendingBalanceUpdate);
      setPendingBalanceUpdate(null);
      return;
    }

    // v2.9.24-beta: Fast 50ms debounce for balance decreases (sending coins)
    console.log('⏱️ [BALANCE DEBUG] Pending balance decrease queued:', {
      pendingValue: pendingBalanceUpdate,
      currentValue: nodeData.balance,
      willUpdateIn: '50ms'
    });

    const timer = setTimeout(() => {
      // v2.3.31-beta: Double-check cooldown before applying (both local and global)
      const timerGlobalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
      const timerGlobalCooldownActive = Date.now() < timerGlobalCooldownUntil;
      if (dexSwapInProgressRef.current || timerGlobalCooldownActive) {
        console.log('🚫 [BALANCE DEBUG] Skipping debounced update - DEX cooldown active (global:', timerGlobalCooldownActive, ')');
        setPendingBalanceUpdate(null);
        return;
      }
      console.log('✅ [BALANCE DEBUG] Applying debounced balance update:', {
        oldBalance: nodeData.balance,
        newBalance: pendingBalanceUpdate,
        source: 'debounced-50ms'
      });
      setNodeData(prev => ({ ...prev, balance: pendingBalanceUpdate }));
      safeCacheBalance(pendingBalanceUpdate);
      setPendingBalanceUpdate(null);
    }, 50);  // v2.9.24-beta: Reduced from 300ms to 50ms for faster UX

    return () => clearTimeout(timer);
  }, [pendingBalanceUpdate, nodeData.balance]);

  // Fetch initial node data and set up SSE for real-time updates
  useEffect(() => {
    if (!authenticated) return;

    console.log('🎬 App.tsx: Setting up authenticated SSE for real-time balance updates');

    let mounted = true;

    const fetchNodeStatus = async () => {
      // v2.3.31-beta: Check BOTH local ref AND global cooldown
      const fetchGlobalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
      const fetchGlobalCooldownActive = Date.now() < fetchGlobalCooldownUntil;
      if (dexSwapInProgressRef.current || fetchGlobalCooldownActive) {
        console.log('🚫 App.tsx fetchNodeStatus: SKIPPING during DEX cooldown (global:', fetchGlobalCooldownActive, ')');
        return;
      }

      try {
        const response = await fetch('/api/v1/node/status');
        if (!response.ok) throw new Error('Failed to fetch node status');

        const data = await response.json();
        if (!mounted) return;

        // v2.3.31-beta: Double-check cooldown after async call (both local and global)
        const postFetchGlobalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
        const postFetchGlobalCooldownActive = Date.now() < postFetchGlobalCooldownUntil;
        if (dexSwapInProgressRef.current || postFetchGlobalCooldownActive) {
          console.log('🚫 App.tsx fetchNodeStatus: SKIPPING after fetch - DEX cooldown (global:', postFetchGlobalCooldownActive, ')');
          return;
        }

        if (data.success && data.data) {
          // 🚨 v2.3.7-beta FIX: Use cached balance for instant display
          // Dashboard handles authenticated API calls and dispatches balance-update events
          // App.tsx cannot call balance API directly - requires auth session which may not be ready
          const cachedBalance = localStorage.getItem('cachedBalance');
          let walletBalance = cachedBalance ? parseFloat(cachedBalance) : 0;
          if (isNaN(walletBalance) || !isFinite(walletBalance)) walletBalance = 0;
          console.log('💰 App.tsx: Using cached balance:', walletBalance, '(Dashboard will update via events)');

          if (mounted) {
            // Calculate dynamic Quantum Coherence Index (QCI)
            const peers = data.data.connected_peers || 0;
            const blockHeight = data.data.current_height || 0;
            const isHealthy = data.data.network_health === 'healthy';

            // QCI Components:
            // 1. Peer connectivity (30%): More peers = better network resilience
            const peerScore = Math.min(peers / 10, 1.0) * 0.30; // Max score at 10+ peers

            // 2. Block production (30%): Higher block height = stable production
            const blockScore = (blockHeight > 0 ? 0.30 : 0.0); // Active if producing blocks

            // 3. Network health (40%): Healthy state = optimal coherence
            const healthScore = isHealthy ? 0.40 : 0.10; // Big penalty if unhealthy

            // Calculate total QCI (0.0 to 1.0)
            const calculatedQCI = peerScore + blockScore + healthScore;

            // CRITICAL FIX v0.9.46-beta: Update balance from API fetch above
            console.log('🔵 [BALANCE DEBUG] Setting initial node data:', {
              balance: walletBalance,
              source: 'fetchNodeStatus',
              blockHeight: blockHeight
            });
            setNodeData(prev => ({
              ...prev,
              balance: walletBalance,
              nodeId: data.data.node_id || '',
              blockHeight: blockHeight,
              peers: peers,
              isOnline: isHealthy,
              qci: calculatedQCI
            }));
          }
        }
      } catch (err) {
        console.error('Error fetching node status:', err);
      }
    };

    // Initial fetch
    fetchNodeStatus();

    // Listen for custom balance update events from Dashboard (e.g., after transactions)
    const handleBalanceUpdate = (event: Event) => {
      const customEvent = event as CustomEvent;
      console.log('💰 App.tsx: Received custom balance-update event:', customEvent.detail);

      if (customEvent.detail?.balance !== undefined) {
        const newBalance = customEvent.detail.balance;
        const source = customEvent.detail?.source || '';

        // v3.6.1-beta: Validate balance before processing
        if (!isValidBalance(newBalance)) {
          console.warn(`🚨 [App] Rejecting invalid balance-update event: ${newBalance} (source: ${source})`);
          return;
        }

        // v2.3.13-beta: Track DEX swaps to block SSE from overwriting
        const isDexSwap = source.includes('DexScreen.swap');
        if (isDexSwap) {
          dexSwapInProgressRef.current = true;
          console.log('🔒 App.tsx: DEX swap detected, blocking SSE balance updates for 10 seconds');

          // v2.3.13-beta: For DEX swaps, update IMMEDIATELY without debounce
          // v3.6.1-beta: Validate balance before accepting
          if (!isValidBalance(newBalance)) {
            console.warn(`🚨 [App] DEX swap balance rejected - invalid value: ${newBalance}`);
            return;
          }
          console.log('🔥 App.tsx: DEX SWAP - Force updating TopBar balance to:', newBalance);
          setNodeData(prev => ({ ...prev, balance: newBalance }));
          safeCacheBalance(newBalance);

          // Clear the flag after 10 seconds
          setTimeout(() => {
            dexSwapInProgressRef.current = false;
            console.log('🔓 App.tsx: DEX swap cooldown ended, SSE balance updates re-enabled');
          }, 10000);

          return; // Skip debounce for DEX swaps
        }

        // v2.3.31-beta: Check BOTH local ref AND global cooldown
        const nonDexGlobalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
        const nonDexGlobalCooldownActive = Date.now() < nonDexGlobalCooldownUntil;
        if (dexSwapInProgressRef.current || nonDexGlobalCooldownActive) {
          console.log('🚫 App.tsx: IGNORING non-DEX balance-update during cooldown (global:', nonDexGlobalCooldownActive, '):', {
            staleBalance: newBalance,
            source: source
          });
          return;
        }

        console.log('🟡 [BALANCE DEBUG] Custom balance-update event:', {
          newBalance: newBalance,
          currentBalance: nodeData.balance,
          source: source,
          isDexSwap: isDexSwap
        });

        // Use debounced update for non-DEX updates to prevent flickering
        setPendingBalanceUpdate(newBalance);
        console.log('⏱️ [BALANCE DEBUG] Balance update queued from custom event (debounced):', newBalance);
      } else {
        // v2.3.31-beta: Block API refresh during cooldown (both local and global)
        const apiGlobalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
        const apiGlobalCooldownActive = Date.now() < apiGlobalCooldownUntil;
        if (dexSwapInProgressRef.current || apiGlobalCooldownActive) {
          console.log('🚫 App.tsx: IGNORING API balance refresh during cooldown (global:', apiGlobalCooldownActive, ')');
          return;
        }

        // If no balance in event, refresh from API (will fetch and cache fresh balance)
        console.log('🔄 App.tsx: No balance in event, fetching from API');

        // Fetch fresh balance from API after transaction
        (async () => {
          try {
            const { walletSession } = await import('./services/walletAuth');
            const session = walletSession.getSession();

            if (!session) {
              console.warn('⚠️ App.tsx: No session for balance refresh after transaction');
              return;
            }

            const { qnkAPI } = await import('./services/api');
            const walletAddress = localStorage.getItem('walletAddress');
            if (!walletAddress) {
              console.warn('⚠️ App.tsx: No wallet address for balance refresh');
              return;
            }

            const balanceResponse = await qnkAPI.getWalletBalance(walletAddress);
            if (balanceResponse.success && balanceResponse.data) {
              const freshBalance = balanceResponse.data.balance_qnk || 0;
              console.log('💰 App.tsx: Fresh balance after transaction:', freshBalance);

              // Use debounced update to prevent flickering
              setPendingBalanceUpdate(freshBalance);
              console.log('⏱️ App.tsx: Balance update queued from API fetch (debounced):', freshBalance);
            }
          } catch (err) {
            console.error('❌ App.tsx: Failed to fetch balance after transaction:', err);
          }
        })();
      }
    };

    window.addEventListener('balance-update', handleBalanceUpdate);

    // v2.3.33-beta: Listen for dex-cooldown-expired to clear refs and update balance
    const handleDexCooldownExpired = (event: Event) => {
      const customEvent = event as CustomEvent;
      const { qugBalance, source } = customEvent.detail;
      console.log('🔄 App.tsx: Received dex-cooldown-expired from', source, 'balance:', qugBalance);

      // Clear the DEX swap in progress flag
      dexSwapInProgressRef.current = false;

      // Update nodeData with correct balance from cache
      if (typeof qugBalance === 'number' && !isNaN(qugBalance)) {
        console.log('🔄 App.tsx: Syncing nodeData balance after cooldown:', qugBalance);
        setNodeData(prev => ({ ...prev, balance: qugBalance }));
      }
    };

    window.addEventListener('dex-cooldown-expired', handleDexCooldownExpired);

    // Set up authenticated SSE for real-time updates with privacy filtering
    // Using custom fetch-based SSE to support X-Wallet-Auth authentication header
    const currentWalletAddress = localStorage.getItem('walletAddress') || '';
    const sseUrl = `/api/v1/events?wallet_address=${encodeURIComponent(currentWalletAddress)}`;
    console.log('📡 App.tsx: Attempting authenticated SSE connection to:', sseUrl);
    console.log('🔐 App.tsx: SSE connection with wallet filter:', currentWalletAddress);

    // Generate authentication header for SSE connection
    const setupAuthenticatedSSE = async () => {
      try {
        // Import wallet auth dynamically to generate X-Wallet-Auth header
        const { generateAuthHeader, walletSession } = await import('./services/walletAuth');

        // Get private key from session
        const session = walletSession.getSession();
        if (!session || !session.privateKey) {
          console.error('❌ App.tsx: No wallet session found for SSE authentication');
          return;
        }

        // Generate authentication header
        const authHeaderJson = await generateAuthHeader(
          session.privateKey,
          currentWalletAddress,
          '/api/v1/events'
        );

        console.log('🔐 App.tsx: Generated X-Wallet-Auth header for SSE');

        // Set up custom SSE using fetch with authentication
        const response = await fetch(sseUrl, {
          method: 'GET',
          headers: {
            'Accept': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'X-Wallet-Auth': authHeaderJson,
          },
        });

        if (!response.ok) {
          throw new Error(`SSE connection failed: ${response.status} ${response.statusText}`);
        }

        if (!response.body) {
          throw new Error('SSE response has no body');
        }

        console.log('✅ App.tsx: Authenticated SSE connection established');

        // Process SSE stream using ReadableStream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        // SSE event parser state
        let eventType = '';
        let eventData = '';

        const processLine = (line: string) => {
          if (line.startsWith('event:')) {
            eventType = line.substring(6).trim();
          } else if (line.startsWith('data:')) {
            eventData = line.substring(5).trim();
          } else if (line === '') {
            // Empty line signals end of event
            if (eventType && eventData) {
              handleSSEEvent(eventType, eventData);
              eventType = '';
              eventData = '';
            }
          }
        };

        const handleSSEEvent = (type: string, data: string) => {
          if (!mounted) return;

          console.log(`📨 App.tsx: SSE event received - type: ${type}`, data);

          try {
            const parsedData = JSON.parse(data);

            if (type === 'balance-updated') {
              // CRITICAL FIX: Backend wraps data in {type: "BalanceUpdated", data: {...}}
              const balanceData = parsedData.data || parsedData;

              const currentWalletAddress = localStorage.getItem('walletAddress');
              // Strip "qnk" prefix for comparison since backend sends hex without prefix
              const currentHex = currentWalletAddress?.startsWith('qnk')
                ? currentWalletAddress.substring(3)
                : currentWalletAddress;

              // Handle both formats: with or without "qnk" prefix in the event
              let eventHex = balanceData.wallet_address;
              if (eventHex?.startsWith('qnk')) {
                eventHex = eventHex.substring(3);
              }

              console.log('💰 App.tsx: Balance update SSE event received!', {
                eventWallet: eventHex,
                currentWallet: currentHex,
                match: eventHex === currentHex,
                oldBalance: balanceData.old_balance,
                newBalance: balanceData.new_balance,
                reason: balanceData.change_reason,
                timestamp: balanceData.timestamp
              });

              // Only update if this balance event is for the current wallet
              if (currentHex && eventHex === currentHex) {
                const changeReason = balanceData.change_reason || '';
                const isP2PMiningReward = changeReason === 'p2p_mining_reward' || changeReason === 'pending_mining_reward';

                // v2.3.31-beta: Check BOTH local ref AND global cooldown for SSE updates
                const sseGlobalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
                const sseGlobalCooldownActive = Date.now() < sseGlobalCooldownUntil;
                if (dexSwapInProgressRef.current || sseGlobalCooldownActive) {
                  console.log('🚫 App.tsx: IGNORING SSE balance update - DEX cooldown (global:', sseGlobalCooldownActive, ')', {
                    staleBalance: balanceData.new_balance,
                    reason: changeReason
                  });
                  return; // Skip this SSE update entirely
                }

                console.log('🟢 [BALANCE DEBUG] SSE balance-updated event:', {
                  oldBalance: balanceData.old_balance,
                  newBalance: balanceData.new_balance,
                  currentNodeDataBalance: nodeData.balance,
                  reason: balanceData.change_reason,
                  isP2PMiningReward,
                  source: 'SSE'
                });

                if (isP2PMiningReward) {
                  // v3.2.9-beta: Backend now correctly tracks accumulated balance!
                  // The in-memory HashMap sync fix means new_balance is ACCURATE.
                  // Just use new_balance directly - no more manual accumulation needed.
                  const rewardAmount = (balanceData.new_balance || 0) - (balanceData.old_balance || 0);
                  console.log('💰 App.tsx: P2P mining reward - using backend balance:', {
                    rewardAmount,
                    backendNewBalance: balanceData.new_balance,
                    backendOldBalance: balanceData.old_balance
                  });
                  setPendingBalanceUpdate(balanceData.new_balance);
                  console.log('⏱️ [BALANCE DEBUG] P2P balance from backend:', balanceData.new_balance);

                  // Dispatch with backend's accumulated balance
                  window.dispatchEvent(new CustomEvent('wallet-balance-updated', {
                    detail: {
                      symbol: 'QUG',
                      balance: balanceData.new_balance,
                      reason: balanceData.change_reason,
                      rewardAmount
                    }
                  }));
                } else {
                  // Local mining rewards: use new_balance directly
                  setPendingBalanceUpdate(balanceData.new_balance);
                  console.log('⏱️ [BALANCE DEBUG] Balance update queued (debounced):', balanceData.new_balance);

                  // Dispatch custom event for Dashboard to update wallet balances
                  window.dispatchEvent(new CustomEvent('wallet-balance-updated', {
                    detail: {
                      symbol: 'QUG',
                      balance: balanceData.new_balance,
                      reason: balanceData.change_reason
                    }
                  }));
                }
                console.log('📢 App.tsx: Dispatched wallet-balance-updated event for Dashboard');
              } else {
                console.log('❌ App.tsx: Balance update IGNORED (not for current wallet)', {
                  eventWallet: eventHex,
                  currentWallet: currentHex
                });
              }
            } else if (type === 'token-balance-updated') {
              // v1.4.10-beta: Handle custom token balance updates for instant DEX updates
              // v2.9.16-beta: Check cooldown BEFORE dispatching - this is the root cause fix!
              const tokenData = parsedData.data || parsedData;
              const tokenSymbol = tokenData.token_symbol || '';
              const tokenUpper = tokenSymbol.toUpperCase();

              // v2.9.16-beta: Check if we're in cooldown - if so, DON'T dispatch stale SSE events
              const now = Date.now();
              const globalCooldownUntil = parseInt(localStorage.getItem('customTokensCooldownUntil') || '0');
              if (now < globalCooldownUntil) {
                console.log(`🛡️ [App.tsx v2.9.16] BLOCKED SSE token-balance-updated for ${tokenUpper} - cooldown active for ${Math.round((globalCooldownUntil - now) / 1000)}s more`);
                return; // Don't dispatch during cooldown - this prevents stale data from reaching ANY component
              }

              const currentWalletAddress = localStorage.getItem('walletAddress');
              const currentHex = currentWalletAddress?.startsWith('qnk')
                ? currentWalletAddress.substring(3)
                : currentWalletAddress;

              let eventHex = tokenData.wallet_address;
              if (eventHex?.startsWith('qnk')) {
                eventHex = eventHex.substring(3);
              }

              console.log('🪙 App.tsx: Token balance update SSE event received!', {
                token: tokenData.token_symbol,
                tokenAddress: tokenData.token_address,
                eventWallet: eventHex,
                currentWallet: currentHex,
                match: eventHex === currentHex,
                oldBalance: tokenData.old_balance,
                newBalance: tokenData.new_balance,
                reason: tokenData.change_reason
              });

              // Only dispatch if this event is for the current wallet
              if (currentHex && eventHex === currentHex) {
                // Dispatch custom event for DEX and Dashboard to update custom token balances
                window.dispatchEvent(new CustomEvent('token-balance-updated', {
                  detail: {
                    tokenAddress: tokenData.token_address,
                    tokenSymbol: tokenData.token_symbol,
                    oldBalance: tokenData.old_balance,
                    newBalance: tokenData.new_balance,
                    reason: tokenData.change_reason,
                    blockHeight: tokenData.block_height,
                    confirmationStatus: tokenData.confirmation_status,
                    source: 'backend-sse' // v2.9.16: Mark source for debugging
                  }
                }));
                console.log('📢 App.tsx: Dispatched token-balance-updated event for DEX');
              }
            } else if (type === 'token_price_update') {
              // v2.9.25-beta: Forward token_price_update to DexScreen via CustomEvent
              // This ensures price updates are received even when DexScreen's own EventSource disconnects
              const priceData = parsedData.data || parsedData;
              console.log('📈 App.tsx: Token price update SSE received!', {
                symbol: priceData.token_symbol,
                address: priceData.token_address,
                price: priceData.price,
                change1h: priceData.change_1h,
                change24h: priceData.change_24h,
                change7d: priceData.change_7d,
                volume24h: priceData.volume_24h
              });

              // Dispatch to window for DexScreen to catch
              window.dispatchEvent(new CustomEvent('token-price-updated', {
                detail: {
                  token_symbol: priceData.token_symbol,
                  token_address: priceData.token_address,
                  price: priceData.price,
                  change_1h: priceData.change_1h,
                  change_24h: priceData.change_24h,
                  change_7d: priceData.change_7d,
                  volume_24h: priceData.volume_24h,
                  source: 'app-sse-forward'
                }
              }));
              console.log('📢 App.tsx: Dispatched token-price-updated event for DexScreen');
            } else if (type === 'faucet-dispensed') {
              console.log('🚰 App.tsx: Faucet dispensed - refreshing balance');
              fetchNodeStatus();
            } else if (type === 'loan-approved') {
              // Handle loan approval from Quillon Bank CLI
              const loanData = parsedData.data || parsedData;

              console.log('🏦 App.tsx: Loan approved via CLI:', loanData);

              // Dispatch custom event for Dashboard to show approval modal
              window.dispatchEvent(new CustomEvent('loan-approved', {
                detail: {
                  loanId: loanData.loan_id,
                  amount: loanData.amount,
                  interestRate: loanData.interest_rate,
                  termMonths: loanData.term_months,
                  monthlyPayment: loanData.monthly_payment,
                  collateralAmount: loanData.collateral_amount,
                  collateralType: loanData.collateral_type || 'QUG',
                }
              }));
              console.log('📢 App.tsx: Dispatched loan-approved event for Dashboard');
            } else if (type === 'pending_mining_reward') {
              // v2.3.31-beta: Check BOTH local ref AND global cooldown
              const miningGlobalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
              const miningGlobalCooldownActive = Date.now() < miningGlobalCooldownUntil;
              if (dexSwapInProgressRef.current || miningGlobalCooldownActive) {
                console.log('🚫 App.tsx: IGNORING pending_mining_reward SSE - DEX cooldown (global:', miningGlobalCooldownActive, ')');
                return;
              }

              // v2.7.5-beta: Handle P2P pending mining rewards for instant balance updates
              // When mining to a peer node, bootstrap receives mining stats via P2P gossipsub
              // This provides instant feedback even before the block is confirmed
              const rewardData = parsedData.data || parsedData;

              const currentWalletAddress = localStorage.getItem('walletAddress');
              const currentHex = currentWalletAddress?.startsWith('qnk')
                ? currentWalletAddress.substring(3)
                : currentWalletAddress;

              // Handle address matching (with or without "qnk" prefix)
              let eventHex = rewardData.miner_address;
              if (eventHex?.startsWith('qnk')) {
                eventHex = eventHex.substring(3);
              }

              console.log('💎 App.tsx: Pending mining reward SSE event!', {
                minerAddress: eventHex,
                currentWallet: currentHex,
                match: eventHex === currentHex,
                pendingReward: rewardData.pending_reward_qnk,
                fromNode: rewardData.from_node_id
              });

              // Only update if this reward is for the current wallet
              if (currentHex && eventHex === currentHex) {
                // ADD the pending reward to the current balance (don't replace!)
                const currentBalance = nodeData.balance;
                const rewardQnk = rewardData.pending_reward_qnk || 0;
                const newBalance = currentBalance + rewardQnk;

                console.log('🟢 [PENDING REWARD] Adding to balance:', {
                  currentBalance,
                  pendingReward: rewardQnk,
                  newBalance,
                  source: 'P2P_SSE'
                });

                // Update balance with the new total (current + pending reward)
                setPendingBalanceUpdate(newBalance);

                // Also dispatch event for Dashboard to update wallet balances
                window.dispatchEvent(new CustomEvent('wallet-balance-updated', {
                  detail: {
                    symbol: 'QUG',
                    balance: newBalance,
                    reason: 'pending_mining_reward'
                  }
                }));
                console.log('📢 App.tsx: Dispatched wallet-balance-updated for pending mining reward');
              }
            } else if (type === 'mining_stats') {
              // v2.7.5-beta: Handle P2P mining stats (hash rate, solutions count)
              // These are informational - no balance update needed
              const statsData = parsedData.data || parsedData;
              console.log('📊 App.tsx: Mining stats SSE event:', {
                minerAddress: statsData.miner_address,
                hashRate: statsData.hash_rate_khs,
                solutionsCount: statsData.solutions_count
              });
              // Dispatch for MiningScreen to update stats display
              window.dispatchEvent(new CustomEvent('mining-stats-updated', {
                detail: statsData
              }));
            } else {
              console.log(`📨 App.tsx: SSE event type '${type}' received:`, parsedData);
            }
          } catch (error) {
            console.error('❌ App.tsx: Error processing SSE event:', error);
          }
        };

        // Read stream continuously
        const readStream = async () => {
          try {
            while (mounted) {
              const { done, value } = await reader.read();

              if (done) {
                console.log('🔄 App.tsx: SSE stream ended, will reconnect...');
                break;
              }

              // Decode chunk and add to buffer
              buffer += decoder.decode(value, { stream: true });

              // Process complete lines
              const lines = buffer.split('\n');
              buffer = lines.pop() || ''; // Keep incomplete line in buffer

              for (const line of lines) {
                processLine(line);
              }
            }
          } catch (error) {
            console.error('❌ App.tsx: SSE stream read error:', error);
          } finally {
            reader.releaseLock();

            // 🚨 v1.0.41-beta: CRITICAL FIX - Auto-reconnect SSE when stream ends
            // BUG: SSE connection was one-shot - when it ended (network hiccup, server restart),
            // balance updates would stop permanently until page refresh
            // FIX: Automatically reconnect after 3 seconds with exponential backoff
            if (mounted) {
              const reconnectDelay = 3000; // 3 seconds
              console.log(`🔄 App.tsx: SSE disconnected, reconnecting in ${reconnectDelay/1000}s...`);
              setTimeout(() => {
                if (mounted) {
                  console.log('🔄 App.tsx: Attempting SSE reconnection...');
                  setupAuthenticatedSSE();
                }
              }, reconnectDelay);
            }
          }
        };

        // Start reading the stream
        readStream();

      } catch (error) {
        console.error('❌ App.tsx: Failed to establish authenticated SSE connection:', error);

        // 🚨 v1.0.41-beta: Also reconnect on connection failure (not just stream end)
        if (mounted) {
          const reconnectDelay = 5000; // 5 seconds on error
          console.log(`🔄 App.tsx: SSE connection failed, retrying in ${reconnectDelay/1000}s...`);
          setTimeout(() => {
            if (mounted) {
              console.log('🔄 App.tsx: Retrying SSE connection...');
              setupAuthenticatedSSE();
            }
          }, reconnectDelay);
        }
      }
    };

    // Initialize authenticated SSE connection
    setupAuthenticatedSSE();

    return () => {
      console.log('🎬 App.tsx: useEffect cleanup - closing SSE');
      mounted = false;
      window.removeEventListener('balance-update', handleBalanceUpdate);
      window.removeEventListener('dex-cooldown-expired', handleDexCooldownExpired);
      // SSE stream will automatically stop when mounted = false
    };
  }, [authenticated]);

  // Logout handler
  const handleLogout = () => {
    setAuthenticated(false);
    // Clear all wallet-related data from localStorage
    localStorage.removeItem('authenticated');
    localStorage.removeItem('walletSeed');
    localStorage.removeItem('walletAddress');
    localStorage.removeItem('walletData');
    localStorage.removeItem('faucetTransactions');
    // v3.9.2-beta: Clear ALL balance/token caches to prevent stale data on new login
    localStorage.removeItem('cachedBalance');
    localStorage.removeItem('dexLockedBalance');
    localStorage.removeItem('dexCooldownUntil');
    localStorage.removeItem('protectedTokenBalances');
    localStorage.removeItem('customTokensCooldownUntil');
    localStorage.removeItem('customTokensCache');
    localStorage.removeItem('authToken');
    // Reset the current screen to dashboard
    setCurrentScreen('dashboard');
    // Reset node data
    setNodeData({
      balance: 0,
      nodeId: '',
      blockHeight: 0,
      peers: 0,
      isOnline: false,
      qci: 0.42,
    });
  };

  if (!authenticated) {
    console.log('🔓 Rendering LoginScreen');
    // v3.4.2-beta: Login page gets full quality - no frame, always show QuantumBackground
    return (
      <div className="min-h-screen relative overflow-hidden bg-gradient-to-br from-slate-950 via-blue-950 to-slate-900">
        {/* Always show QuantumBackground on login for best visual quality */}
        <QuantumBackground />
        <LoginScreen onAuthenticate={() => setAuthenticated(true)} />
      </div>
    );
  }

  console.log('✅ Authenticated - Rendering main app');

  // Handle token click - navigate to DEX screen with token selected
  const handleTokenClick = (token: any) => {
    console.log('Token clicked:', token);
    setCurrentScreen('dex');
    // Store selected token in localStorage for DEX to pick up
    localStorage.setItem('selectedToken', JSON.stringify(token));
  };

  // Handle coin send click - navigate to transaction screen with pre-selected coin
  const handleCoinSendClick = (coinSymbol: string) => {
    console.log('Coin send clicked:', coinSymbol);
    setCurrentScreen('transactions');
    // Store selected coin in localStorage for TransactionV2 to pick up
    localStorage.setItem('selectedCoinForSend', coinSymbol);
  };

  return (
    <AnimatedBorder>
      {/* Background and content are INSIDE the border so they don't cover the ornate frame */}
      <div className="min-h-full relative overflow-hidden" style={{ background: 'transparent' }}>
        {/* v2.4.0: Skip QuantumBackground in performance mode for better frame rates */}
        {!performanceMode && <QuantumBackground />}
        <div className="relative z-10 flex flex-col min-h-full">
          {/* Global TopBar */}
          <TopBar
            currentBalance={nodeData.balance}
            nodeId={nodeData.nodeId}
            blockHeight={nodeData.blockHeight}
            peers={nodeData.peers}
            isOnline={nodeData.isOnline}
            qci={nodeData.qci}
            onNavigate={setCurrentScreen}
          />

          {/* Token Bar - Below TopBar */}
          <TokenBar onTokenClick={handleTokenClick} />

          <div className="flex flex-1 lg:flex-row">
            <Navigation
              currentScreen={currentScreen}
              onNavigate={setCurrentScreen}
              className="lg:w-20 xl:w-64"
            />

            <main className="flex-1 p-4 lg:p-8 pb-20 lg:pb-8">
              {/* v2.3.12-beta: Keep Dashboard mounted to receive wallet-balance-updated events while on DEX */}
              {/* Without this, Dashboard unmounts when on DEX, misses balance update events, */}
              {/* then refetches stale data from API when remounted - causing "two balances" bug */}
              <div style={{ display: currentScreen === 'dashboard' ? 'block' : 'none' }}>
                <Dashboard key="dashboard-stable" onNavigateToSend={handleCoinSendClick} />
              </div>
              {currentScreen === 'transactions' && <TransactionScreenV2 currentBalance={nodeData.balance} />}
              {/* v2.3.12-beta: Keep DexScreen mounted to preserve swap state */}
              <div style={{ display: currentScreen === 'dex' ? 'block' : 'none' }}>
                <DexScreen />
              </div>
              {currentScreen === 'explorer' && <ExplorerScreen />}
              {currentScreen === 'mining' && <MiningScreen />}
              {currentScreen === 'vm' && <VittuaVMScreen />}
              {/* Keep AIChatScreen mounted to preserve state (messages, currentChatId, isGenerating) */}
              <div style={{ display: currentScreen === 'aichat' ? 'block' : 'none' }}>
                <AIChatScreen />
              </div>
              {currentScreen === 'download' && <DownloadNodeScreen />}
              {currentScreen === 'settings' && <SettingsScreen onLogout={handleLogout} />}
            </main>

            {/* AI Worker Panel - Floating bottom-right */}
            <div style={{ position: 'fixed', bottom: '20px', right: '20px', zIndex: 1000 }}>
              <AIWorkerDemo />
            </div>
          </div>
        </div>
      </div>
    </AnimatedBorder>
  );
}

export default App
