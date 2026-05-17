import { useState, useEffect, useRef, lazy, Suspense } from 'react';
import { createPortal } from 'react-dom';
import LoginScreen from './components/LoginScreen';
import Dashboard from './components/Dashboard';
import DexScreen from './components/DexScreen';
import AIChatScreen from './components/AIChatScreen';
import ChatScreen from './components/ChatScreen';
import Navigation from './components/Navigation';
import TopBar from './components/TopBar';
import TokenBar from './components/TokenBar';
import QuantumBackground from './components/QuantumBackground';
import AnimatedBorder from './components/AnimatedBorder';
import OAuthConsentPage from './components/OAuthConsentPage';
import MinerLoginPage from './components/MinerLoginPage';
import POSMode from './components/POSMode';
import { sseManager } from './services/sseManager';
import IncomingMemoModal from './components/IncomingMemoModal';
import './App.css';

// Lazy-loaded screens — split into separate chunks, loaded on first navigation
const TransactionScreenV2 = lazy(() => import('./components/TransactionScreenV2'));
const ExplorerScreen = lazy(() => import('./components/ExplorerScreen'));
const MiningScreen = lazy(() => import('./components/MiningScreen'));
const VittuaVMScreen = lazy(() => import('./components/VittuaVMScreen'));
const DownloadNodeScreen = lazy(() => import('./components/DownloadNodeScreen'));
const SettingsScreen = lazy(() => import('./components/SettingsScreen'));
const RwaMarketplaceScreen = lazy(() => import('./components/RwaMarketplaceScreen'));
const GameItemsScreen = lazy(() => import('./components/GameItemsScreen'));
const EmailScreen = lazy(() => import('./components/EmailScreen'));
const AnalyticsScreen = lazy(() => import('./components/AnalyticsScreen'));
const DeployControlPanel = lazy(() => import('./components/DeployControlPanel'));
const NodeSettingsModal = lazy(() => import('./components/NodeSettingsModal'));
const AIWheelButton = lazy(() => import('./components/AIWheelButton'));
const BountyModal = lazy(() => import('./components/BountyModal'));
const BountySiteModal = lazy(() => import('./components/BountyModal').then(m => ({ default: m.BountySiteModal })));
const MapScreen = lazy(() => import('./components/MapScreen'));
const BankScreen = lazy(() => import('./components/BankScreen'));
const TorrentTab = lazy(() => import('./components/TorrentTab'));

// Loading spinner for lazy-loaded screen transitions
const LoadingSpinner = () => (
  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '200px' }}>
    <div style={{
      width: 36, height: 36, borderRadius: '50%',
      border: '3px solid rgba(212, 175, 55, 0.2)',
      borderTopColor: '#d4af37',
      animation: 'spin 0.8s linear infinite',
    }} />
  </div>
);

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

type Screen = 'dashboard' | 'transactions' | 'explorer' | 'dex' | 'mining' | 'vm' | 'rwamarket' | 'gameitems' | 'download' | 'aichat' | 'email' | 'analytics' | 'settings' | 'map' | 'bank' | 'chat' | 'torrent';

function App() {
  console.log('🚀 App function executing - TOP OF FUNCTION');

  // v7.0.0: NETWORK CHANGE DETECTION - Clear all cached data when switching networks
  // This prevents stale testnet balances/tokens from showing up on mainnet
  (() => {
    const CURRENT_NETWORK = 'mainnet-genesis'; // Must match Q_NETWORK_ID
    const lastNetwork = localStorage.getItem('lastNetworkId');
    if (lastNetwork && lastNetwork !== CURRENT_NETWORK) {
      console.warn(`🔄 [App] Network changed: ${lastNetwork} → ${CURRENT_NETWORK}. Clearing ALL cached data.`);
      // Clear all balance/token caches
      localStorage.removeItem('cachedBalance');
      localStorage.removeItem('cachedQugusdBalance');
      localStorage.removeItem('walletBalanceHistory');
      localStorage.removeItem('dexLockedBalance');
      localStorage.removeItem('dexCooldownUntil');
      localStorage.removeItem('protectedTokenBalances');
      localStorage.removeItem('customTokensCache');
      localStorage.removeItem('customTokensCooldownUntil');
      localStorage.removeItem('selectedToken');
      localStorage.removeItem('dexSettings');
      // Force re-login to fetch fresh data
      localStorage.removeItem('authenticated');
      localStorage.removeItem('authToken');
    }
    localStorage.setItem('lastNetworkId', CURRENT_NETWORK);
  })();

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

  // Incoming transaction with memo — shows notification modal
  const [incomingMemoTx, setIncomingMemoTx] = useState<{
    amount: number;
    fromAddress: string;
    memo: string;
    txHash: string;
    timestamp: number;
  } | null>(null);
  const shownMemoTxIds = useRef(new Set<string>());

  // v5.6.0: Track server version for refresh banner after deploys
  const [newVersionBanner, setNewVersionBanner] = useState<string | null>(null);

  // Bounty modal — opened by TopBar/GlobalTopBar button via custom event
  const [showBountyModal, setShowBountyModal] = useState(false);
  const [showBountySite, setShowBountySite] = useState(false);
  useEffect(() => {
    const handler = () => setShowBountyModal(true);
    window.addEventListener('open-bounty-modal', handler);
    return () => window.removeEventListener('open-bounty-modal', handler);
  }, []);

  // Global incoming call overlay — ChatScreen fires qnk-incoming-call even when
  // it is hidden (display:none), so we render the ringing modal at App level.
  const [globalIncomingCall, setGlobalIncomingCall] = useState<{
    from: string; callType: 'audio' | 'video';
  } | null>(null);
  const ringAudioRef = useRef<AudioContext | null>(null);
  const ringIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startRing = () => {
    try {
      const ctx = new AudioContext();
      ringAudioRef.current = ctx;
      const playBeep = () => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.setValueAtTime(880, ctx.currentTime);
        gain.gain.setValueAtTime(0.3, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.5);
      };
      playBeep();
      ringIntervalRef.current = setInterval(playBeep, 1500);
    } catch { /* AudioContext not available */ }
  };

  const stopRing = () => {
    if (ringIntervalRef.current) {
      clearInterval(ringIntervalRef.current);
      ringIntervalRef.current = null;
    }
    ringAudioRef.current?.close().catch(() => {});
    ringAudioRef.current = null;
  };

  useEffect(() => {
    console.log('[App] qnk-incoming-call listeners registered');
    const onCall = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      console.log('[App] qnk-incoming-call received:', detail);
      setGlobalIncomingCall({ from: detail.from, callType: detail.callType });
      startRing();
      // Browser notification for background tabs
      if (Notification.permission === 'granted') {
        new Notification(`Incoming ${detail.callType} call`, {
          body: `From: ${(detail.from as string).slice(0, 20)}…`,
          icon: '/favicon.ico',
        });
      } else if (Notification.permission !== 'denied') {
        Notification.requestPermission().then(p => {
          if (p === 'granted') {
            new Notification(`Incoming ${detail.callType} call`, {
              body: `From: ${(detail.from as string).slice(0, 20)}…`,
              icon: '/favicon.ico',
            });
          }
        });
      }
    };
    const onClear = () => {
      setGlobalIncomingCall(null);
      stopRing();
    };
    window.addEventListener('qnk-incoming-call', onCall);
    window.addEventListener('qnk-incoming-call-cleared', onClear);
    return () => {
      window.removeEventListener('qnk-incoming-call', onCall);
      window.removeEventListener('qnk-incoming-call-cleared', onClear);
      stopRing();
    };
  }, []);

  // Chat message toast notification — shown when a message arrives outside the Chat screen
  const [chatMsgAlert, setChatMsgAlert] = useState<{ from: string; content: string; contactName?: string } | null>(null);
  const chatMsgTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const currentScreenRef = useRef(currentScreen);
  useEffect(() => { currentScreenRef.current = currentScreen; }, [currentScreen]);
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail as { from: string; content: string; contactName?: string };
      if (currentScreenRef.current === 'chat') return; // already on chat, ChatScreen handles it
      setChatMsgAlert(detail);
      if (chatMsgTimerRef.current) clearTimeout(chatMsgTimerRef.current);
      chatMsgTimerRef.current = setTimeout(() => setChatMsgAlert(null), 7000);
    };
    window.addEventListener('qnk-new-chat-message', handler);
    return () => window.removeEventListener('qnk-new-chat-message', handler);
  }, []);

  // v2.3.11-beta: Track when DEX swap just happened to ignore stale SSE updates
  // SSE balance updates from server can be stale and overwrite correct DEX swap balance
  const dexSwapInProgressRef = useRef(false);

  // v8.1.6: Monotonic block-height tracking to prevent balance zigzag
  // Only accept balance updates from same or higher block height
  const lastBalanceHeightRef = useRef(0);

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
  // v6.0.3: Use ref for nodeData.balance to avoid circular dependency (fixes React Error #185)
  const nodeDataBalanceRef = useRef(nodeData.balance);
  nodeDataBalanceRef.current = nodeData.balance;

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

    const currentBalance = nodeDataBalanceRef.current;
    const isBalanceIncrease = pendingBalanceUpdate > currentBalance;

    // v2.9.24-beta: INSTANT updates for receiving coins (balance increases)
    if (isBalanceIncrease) {
      console.log('⚡ [BALANCE DEBUG] INSTANT balance increase (receiving coins):', {
        oldBalance: currentBalance,
        newBalance: pendingBalanceUpdate,
        increase: pendingBalanceUpdate - currentBalance
      });
      setNodeData(prev => ({ ...prev, balance: pendingBalanceUpdate }));
      safeCacheBalance(pendingBalanceUpdate);
      setPendingBalanceUpdate(null);
      return;
    }

    // v2.9.24-beta: Fast 50ms debounce for balance decreases (sending coins)
    console.log('⏱️ [BALANCE DEBUG] Pending balance decrease queued:', {
      pendingValue: pendingBalanceUpdate,
      currentValue: currentBalance,
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
        oldBalance: nodeDataBalanceRef.current,
        newBalance: pendingBalanceUpdate,
        source: 'debounced-50ms'
      });
      setNodeData(prev => ({ ...prev, balance: pendingBalanceUpdate }));
      safeCacheBalance(pendingBalanceUpdate);
      setPendingBalanceUpdate(null);
    }, 50);  // v2.9.24-beta: Reduced from 300ms to 50ms for faster UX

    return () => clearTimeout(timer);
  }, [pendingBalanceUpdate]);

  // Fetch initial node data and set up SSE for real-time updates
  useEffect(() => {
    if (!authenticated) return;

    console.log('🎬 App.tsx: Setting up authenticated SSE for real-time balance updates');

    let mounted = true;

    // v5.1.1: Exponential backoff for SSE reconnection (3s → 6s → 12s → 24s → 30s cap)
    let sseReconnectAttempts = 0;
    const getReconnectDelay = () => {
      const base = 3000;
      const delay = Math.min(base * Math.pow(2, sseReconnectAttempts), 30000);
      sseReconnectAttempts++;
      return delay;
    };
    const resetReconnectBackoff = () => { sseReconnectAttempts = 0; };

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

        // v8.1.6: Custom events don't carry block height, so check if SSE already
        // provided a more recent update. If SSE is active with height tracking,
        // skip custom events to avoid stale overwrites.
        const eventHeight = customEvent.detail?.blockHeight || 0;
        if (eventHeight > 0 && eventHeight < lastBalanceHeightRef.current) {
          console.log('🚫 [BALANCE] Rejecting stale custom balance-update (height regression):', {
            eventHeight,
            lastHeight: lastBalanceHeightRef.current,
            staleBalance: newBalance,
            source
          });
          return;
        }
        if (eventHeight > 0) {
          lastBalanceHeightRef.current = eventHeight;
        }

        console.log('🟡 [BALANCE DEBUG] Custom balance-update event:', {
          newBalance: newBalance,
          currentBalance: nodeData.balance,
          source: source,
          blockHeight: eventHeight
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

    // v8.4.4: Use shared SSE manager — single connection, multiple subscribers
    // Replaces fetch-based SSE that had broken reconnection and no auto-retry
    const currentWalletAddress = localStorage.getItem('walletAddress') || '';
    console.log('📡 App.tsx: Starting shared SSE manager for:', currentWalletAddress);

    // v8.4.4: Use sseManager — native EventSource with auto-reconnect,
    // visibility-aware, health-checked, single connection shared by all components.
    // Replaces broken fetch()-based SSE that had no auto-retry and stale backoff.
    const unsubs: (() => void)[] = [];
    sseManager.start(currentWalletAddress);

    // Helper to normalize wallet addresses for comparison
    const normalizeAddr = (addr: string) =>
      (addr?.startsWith('qnk') ? addr.substring(3) : addr)?.toLowerCase();
    const myHex = normalizeAddr(currentWalletAddress);

    // --- balance-updated ---
    unsubs.push(sseManager.on('balance-updated', (parsedData: any) => {
      if (!mounted) return;
      const balanceData = parsedData.data || parsedData;
      const eventHex = normalizeAddr(balanceData.wallet_address || '');

      if (myHex && eventHex === myHex) {
        const changeReason = balanceData.change_reason || '';
        const confirmationStatus = balanceData.confirmation_status || '';

        // Incoming transaction with memo — show notification modal before any filters
        if (changeReason === 'transaction_received' && balanceData.memo && balanceData.tx_hash) {
          const txId = balanceData.tx_hash as string;
          if (!shownMemoTxIds.current.has(txId)) {
            shownMemoTxIds.current.add(txId);
            setIncomingMemoTx({
              amount: (balanceData.new_balance - balanceData.old_balance) * 1e10,
              fromAddress: balanceData.from_address || '',
              memo: balanceData.memo,
              txHash: txId,
              timestamp: Date.now(),
            });
          }
        }

        // Skip pending-only balance events — they inflate nodeDataBalanceRef temporarily
        // during restart bursts. Confirmed balance updates come from block processing.
        // The PendingMiningReward SSE event handles UI notification separately.
        if (changeReason === 'p2p_mining_reward_pending' || confirmationStatus === 'pending') return;

        // Skip zero-balance initial events — wallet_balances cache is empty at node startup
        // and previously sent balance=0 on SSE connect, overriding the valid cached balance.
        // Now handled at server (reads RocksDB instead), but keep this as frontend safety net.
        const newBalanceValue = balanceData.new_balance ?? 0;
        if (newBalanceValue === 0 && changeReason === 'SSE connection established') return;

        const isP2PMiningReward = changeReason === 'p2p_mining_reward' || changeReason === 'pending_mining_reward';

        const sseGlobalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
        if (dexSwapInProgressRef.current || Date.now() < sseGlobalCooldownUntil) return;

        const eventBlockHeight = balanceData.block_height || 0;
        if (eventBlockHeight > 0 && eventBlockHeight < lastBalanceHeightRef.current) return;
        if (eventBlockHeight > 0) lastBalanceHeightRef.current = eventBlockHeight;

        setPendingBalanceUpdate(balanceData.new_balance);
        const rewardAmount = isP2PMiningReward ? (balanceData.new_balance || 0) - (balanceData.old_balance || 0) : undefined;
        window.dispatchEvent(new CustomEvent('wallet-balance-updated', {
          detail: {
            symbol: 'QUG', balance: balanceData.new_balance, oldBalance: balanceData.old_balance,
            reason: changeReason, rewardAmount, blockHeight: balanceData.block_height,
            blockHash: balanceData.block_hash, walletAddress: balanceData.wallet_address,
            timestamp: balanceData.timestamp,
          }
        }));
      }
    }));

    // --- token-balance-updated ---
    unsubs.push(sseManager.on('token-balance-updated', (parsedData: any) => {
      if (!mounted) return;
      const tokenData = parsedData.data || parsedData;
      const globalCooldownUntil = parseInt(localStorage.getItem('customTokensCooldownUntil') || '0');
      if (Date.now() < globalCooldownUntil) return;

      const eventHex = normalizeAddr(tokenData.wallet_address || '');
      if (myHex && eventHex === myHex) {
        window.dispatchEvent(new CustomEvent('token-balance-updated', {
          detail: { tokenAddress: tokenData.token_address, tokenSymbol: tokenData.token_symbol,
            oldBalance: tokenData.old_balance, newBalance: tokenData.new_balance,
            reason: tokenData.change_reason, blockHeight: tokenData.block_height,
            confirmationStatus: tokenData.confirmation_status, source: 'backend-sse' }
        }));
      }
    }));

    // --- token_price_update ---
    unsubs.push(sseManager.on('token_price_update', (parsedData: any) => {
      if (!mounted) return;
      const priceData = parsedData.data || parsedData;
      window.dispatchEvent(new CustomEvent('token-price-updated', {
        detail: { token_symbol: priceData.token_symbol, token_address: priceData.token_address,
          price: priceData.price, change_1h: priceData.change_1h, change_24h: priceData.change_24h,
          change_7d: priceData.change_7d, volume_24h: priceData.volume_24h, source: 'app-sse-forward' }
      }));
    }));

    // --- loan-approved ---
    unsubs.push(sseManager.on('loan-approved', (parsedData: any) => {
      if (!mounted) return;
      const d = parsedData.data || parsedData;
      window.dispatchEvent(new CustomEvent('loan-approved', {
        detail: { loanId: d.loan_id, amount: d.amount, interestRate: d.interest_rate,
          termMonths: d.term_months, monthlyPayment: d.monthly_payment,
          collateralAmount: d.collateral_amount, collateralType: d.collateral_type || 'QUG' }
      }));
    }));

    // --- pending_mining_reward ---
    unsubs.push(sseManager.on('pending_mining_reward', (parsedData: any) => {
      if (!mounted) return;
      const sseGlobalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
      if (dexSwapInProgressRef.current || Date.now() < sseGlobalCooldownUntil) return;

      const rewardData = parsedData.data || parsedData;
      const eventHex = normalizeAddr(rewardData.miner_address || '');
      if (myHex && eventHex === myHex) {
        const rewardQnk = rewardData.pending_reward_qnk || 0;
        window.dispatchEvent(new CustomEvent('wallet-balance-updated', {
          detail: { symbol: 'QUG', balance: nodeDataBalanceRef.current + rewardQnk,
            oldBalance: nodeDataBalanceRef.current, reason: 'pending_mining_reward',
            rewardAmount: rewardQnk, walletAddress: eventHex,
            timestamp: new Date().toISOString(), infoOnly: true }
        }));
      }
    }));

    // --- mining_stats ---
    unsubs.push(sseManager.on('mining_stats', (parsedData: any) => {
      if (!mounted) return;
      window.dispatchEvent(new CustomEvent('mining-stats-updated', { detail: parsedData.data || parsedData }));
    }));

    // --- server-version ---
    unsubs.push(sseManager.on('server-version', (parsedData: any) => {
      if (!mounted) return;
      const versionData = parsedData.data || parsedData;
      const cachedVersion = localStorage.getItem('serverVersion');
      if (cachedVersion && cachedVersion !== versionData.version) setNewVersionBanner(versionData.version);
      localStorage.setItem('serverVersion', versionData.version);
    }));

    // --- email events ---
    unsubs.push(sseManager.on('email-received', (p: any) => { if (mounted) window.dispatchEvent(new CustomEvent('email-received', { detail: p.data || p })); }));
    unsubs.push(sseManager.on('email-sent', (p: any) => { if (mounted) window.dispatchEvent(new CustomEvent('email-sent', { detail: p.data || p })); }));
    unsubs.push(sseManager.on('email-unread-count', (p: any) => { if (mounted) window.dispatchEvent(new CustomEvent('email-unread-count', { detail: p.data || p })); }));

    // --- calendar events ---
    unsubs.push(sseManager.on('calendar-event-created', (p: any) => { if (mounted) window.dispatchEvent(new CustomEvent('calendar-event-created', { detail: p.data || p })); }));
    unsubs.push(sseManager.on('calendar-reminder', (p: any) => { if (mounted) window.dispatchEvent(new CustomEvent('calendar-reminder', { detail: p.data || p })); }));
    unsubs.push(sseManager.on('scheduled-tx-executed', (p: any) => { if (mounted) window.dispatchEvent(new CustomEvent('scheduled-tx-executed', { detail: p.data || p })); }));

    return () => {
      console.log('App.tsx: useEffect cleanup - stopping SSE');
      mounted = false;
      unsubs.forEach(u => u());
      sseManager.stop();
      window.removeEventListener('balance-update', handleBalanceUpdate);
      window.removeEventListener('dex-cooldown-expired', handleDexCooldownExpired);
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
    localStorage.removeItem('walletPublicKey');
    // Clear encrypted wallet keys and password hash
    localStorage.removeItem('walletEncryptedKey');
    localStorage.removeItem('walletEncryptedMnemonic');
    localStorage.removeItem('walletPasswordHash');
    localStorage.removeItem('walletEncryptedAegisKey');
    localStorage.removeItem('walletAegisPublicKey');
    localStorage.removeItem('walletEncryptedSQIsignKey');
    localStorage.removeItem('walletSQIsignPublicKey');
    localStorage.removeItem('walletEncryptedDilithium5Key');
    localStorage.removeItem('walletDilithium5PublicKey');
    // v3.9.2-beta: Clear ALL balance/token caches to prevent stale data on new login
    localStorage.removeItem('cachedBalance');
    localStorage.removeItem('cachedQugusdBalance');
    localStorage.removeItem('walletBalanceHistory');
    localStorage.removeItem('dexLockedBalance');
    localStorage.removeItem('dexCooldownUntil');
    localStorage.removeItem('protectedTokenBalances');
    localStorage.removeItem('customTokensCooldownUntil');
    localStorage.removeItem('customTokensCache');
    localStorage.removeItem('authToken');
    // Clear session storage
    sessionStorage.removeItem('walletSession');
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

  // v7.3.3: Backward-compat — old bounty-dapp opened popup to #/connect-bounty.
  // Now bounty uses OAuth2 redirect. Show message and auto-close stale popups.
  if (window.location.hash === '#/connect-bounty') {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: '#0f172a', color: '#94a3b8', fontFamily: 'sans-serif', textAlign: 'center', padding: 32 }}>
        <div>
          <p style={{ fontSize: 18, marginBottom: 12 }}>Wallet connect has been upgraded to OAuth2.</p>
          <p style={{ fontSize: 14 }}>Please refresh the bounty page and click "Connect Wallet" again.</p>
          <button onClick={() => window.close()} style={{ marginTop: 20, padding: '8px 24px', background: '#7c3aed', color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer' }}>Close</button>
        </div>
      </div>
    );
  }

  // v7.3.0: OAuth2 consent page intercept — render standalone consent page
  // when URL path is /oauth/consent (redirected from /api/v1/oauth2/authorize)
  if (window.location.pathname === '/oauth/consent') {
    return <OAuthConsentPage />;
  }

  // v8.5.9: Miner device login page — user opens this from the miner to link their wallet
  if (window.location.pathname === '/miner-login') {
    const urlParams = new URLSearchParams(window.location.search);
    const deviceCode = urlParams.get('code') || '';
    return <MinerLoginPage deviceCode={deviceCode} />;
  }

  // v9.5.0: Merchant POS mode — full-screen point-of-sale for accepting QR payments
  if (window.location.pathname === '/pos') {
    const posWalletAddress = localStorage.getItem('walletAddress') || '';
    if (!posWalletAddress) {
      // Merchant not logged in — redirect to main app to authenticate first
      window.history.pushState(null, '', '/');
      window.location.reload();
      return null;
    }
    return <POSMode walletAddress={posWalletAddress} serverUrl="" />;
  }

  if (!authenticated) {
    console.log('🔓 Rendering LoginScreen');
    // v3.4.2-beta: Login page gets full quality - no frame, always show QuantumBackground
    return (
      <div className="min-h-screen relative overflow-hidden bg-gradient-to-br from-slate-950 via-blue-950 to-slate-900">
        {/* Always show QuantumBackground on login for best visual quality */}
        <QuantumBackground />
        <LoginScreen onAuthenticate={() => {
          // v6.3.0: Reset balance state on login to prevent stale cached values
          // from a previous wallet showing in the new session
          const cachedBal = localStorage.getItem('cachedBalance');
          const freshBalance = cachedBal ? parseFloat(cachedBal) : 0;
          const validFresh = (!isNaN(freshBalance) && isFinite(freshBalance) && freshBalance >= 0 && freshBalance <= 21_000_000) ? freshBalance : 0;
          setNodeData(prev => ({ ...prev, balance: validFresh }));
          setPendingBalanceUpdate(null);
          setAuthenticated(true);
        }} />
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
    <>
    <AnimatedBorder>
      {/* Background and content are INSIDE the border so they don't cover the ornate frame */}
      <div className="min-h-full relative overflow-hidden" style={{ background: 'transparent' }}>
        {/* v2.4.0: Skip QuantumBackground in performance mode for better frame rates */}
        {!performanceMode && <QuantumBackground />}
        <div className="relative z-10 flex flex-col min-h-full">
          {/* v5.6.0: New version refresh banner */}
          {newVersionBanner && (
            <div
              onClick={() => window.location.reload()}
              className="fixed top-0 left-0 right-0 z-[9999] flex items-center justify-center gap-2 py-2 px-4 cursor-pointer"
              style={{
                background: 'linear-gradient(90deg, rgba(16, 185, 129, 0.95) 0%, rgba(52, 211, 153, 0.95) 100%)',
                color: 'white',
                fontSize: '14px',
                fontWeight: 600,
                textShadow: '0 1px 2px rgba(0,0,0,0.2)',
              }}
            >
              New version v{newVersionBanner} available. Click to refresh.
            </div>
          )}

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

          {/* v5.1.1: Deploy Control Panel - master wallet + node admin */}
          <Suspense fallback={null}><DeployControlPanel /></Suspense>
          {/* v7.3.0: Node Settings Modal - admin wallet OAuth2 + node info */}
          <Suspense fallback={null}><NodeSettingsModal /></Suspense>
          {/* Bounty Campaign Modal */}
          {showBountyModal && (
            <Suspense fallback={null}>
              <BountyModal
                onClose={() => setShowBountyModal(false)}
                onStartEarning={() => setShowBountySite(true)}
              />
            </Suspense>
          )}
          {/* Bounty Site Modal — rendered at top level above all other modals */}
          {showBountySite && (
            <Suspense fallback={null}>
              <BountySiteModal onClose={() => setShowBountySite(false)} />
            </Suspense>
          )}
          {/* Token Bar - Below TopBar */}
          <TokenBar onTokenClick={handleTokenClick} />

          <div className="flex flex-1 lg:flex-row">
            <Navigation
              currentScreen={currentScreen}
              onNavigate={setCurrentScreen}
              className="lg:w-20 xl:w-64"
              walletAddress={localStorage.getItem('walletAddress') || ''}
            />

            <main className="flex-1 p-4 lg:p-8 pb-20 lg:pb-8">
              {/* v2.3.12-beta: Keep Dashboard mounted to receive wallet-balance-updated events while on DEX */}
              {/* Without this, Dashboard unmounts when on DEX, misses balance update events, */}
              {/* then refetches stale data from API when remounted - causing "two balances" bug */}
              <div style={{ display: currentScreen === 'dashboard' ? 'block' : 'none' }}>
                <Dashboard key="dashboard-stable" onNavigateToSend={handleCoinSendClick} liveBalance={nodeData.balance} onNavigateToChat={() => setCurrentScreen('chat')} />
              </div>
              {/* v2.3.12-beta: Keep DexScreen mounted to preserve swap state */}
              <div style={{ display: currentScreen === 'dex' ? 'block' : 'none' }}>
                <DexScreen isActive={currentScreen === 'dex'} />
              </div>
              {/* Keep ChatScreen mounted to preserve call/meeting state */}
              <div style={{ display: currentScreen === 'chat' ? 'block' : 'none', height: '100%' }}>
                <ChatScreen />
              </div>
              {/* Keep AIChatScreen mounted to preserve state (messages, currentChatId, isGenerating) */}
              <div style={{ display: currentScreen === 'aichat' ? 'block' : 'none' }}>
                <AIChatScreen />
              </div>
              {/* Keep ExplorerScreen mounted — on remount, 7 optional-metric requests contend with core stats for rate-limiter slots, causing stats to never load */}
              <div style={{ display: currentScreen === 'explorer' ? 'block' : 'none' }}>
                <Suspense fallback={null}>
                  <ExplorerScreen isActive={currentScreen === 'explorer'} />
                </Suspense>
              </div>
              <Suspense fallback={<LoadingSpinner />}>
                {currentScreen === 'transactions' && <TransactionScreenV2 currentBalance={nodeData.balance} />}
                {currentScreen === 'mining' && <MiningScreen />}
                {currentScreen === 'vm' && <VittuaVMScreen />}
                {currentScreen === 'map' && <MapScreen />}
                {currentScreen === 'bank' && <BankScreen />}
                {currentScreen === 'rwamarket' && <RwaMarketplaceScreen />}
                {currentScreen === 'gameitems' && <GameItemsScreen />}
                {currentScreen === 'email' && <EmailScreen />}
                {currentScreen === 'analytics' && <AnalyticsScreen />}
                {currentScreen === 'download' && <DownloadNodeScreen />}
                {currentScreen === 'settings' && <SettingsScreen onLogout={handleLogout} />}
                {currentScreen === 'torrent' && <TorrentTab />}
              </Suspense>
            </main>

          </div>
        </div>
      </div>
      {/* v8.9.0: AI Wheel Button — floating AI assistant with radial tool wheel */}
      <Suspense fallback={null}><AIWheelButton /></Suspense>
    </AnimatedBorder>

    {/* Incoming transaction memo notification */}
    <IncomingMemoModal
      tx={incomingMemoTx}
      onClose={() => setIncomingMemoTx(null)}
    />

    {/* Global incoming call modal — rendered via portal directly into document.body
        so it escapes the AnimatedBorder isolation:isolate stacking context and
        is guaranteed to sit above everything else in the viewport. */}
    {globalIncomingCall && createPortal(
      <div
        style={{
          position: 'fixed', inset: 0, zIndex: 2147483647,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: 'rgba(0,0,0,0.72)', backdropFilter: 'blur(8px)',
          fontFamily: 'system-ui, sans-serif',
        }}
      >
        <div style={{
          background: 'linear-gradient(135deg, #0f172a, #1e293b)',
          border: '1.5px solid rgba(212,175,55,0.5)',
          borderRadius: 24, padding: '40px 44px', textAlign: 'center',
          boxShadow: '0 0 80px rgba(212,175,55,0.2), 0 32px 64px rgba(0,0,0,0.7)',
          minWidth: 320, maxWidth: 420,
        }}>
          <div style={{
            width: 80, height: 80, borderRadius: '50%', margin: '0 auto 24px',
            background: 'linear-gradient(135deg, rgba(212,175,55,0.3), rgba(255,215,0,0.1))',
            border: '2px solid rgba(212,175,55,0.6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 36,
          }}>
            {globalIncomingCall.callType === 'video' ? '📹' : '📞'}
          </div>
          <p style={{ color: 'rgba(212,175,55,0.7)', fontSize: 11, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 2 }}>
            Incoming {globalIncomingCall.callType} call
          </p>
          <p style={{ color: '#f1f5f9', fontSize: 15, fontWeight: 600, marginBottom: 32, wordBreak: 'break-all', lineHeight: 1.4 }}>
            {globalIncomingCall.from.length > 20
              ? `${globalIncomingCall.from.slice(0, 10)}…${globalIncomingCall.from.slice(-8)}`
              : globalIncomingCall.from}
          </p>
          <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
            <button
              onClick={() => {
                stopRing();
                setGlobalIncomingCall(null);
                window.dispatchEvent(new CustomEvent('qnk-reject-call'));
              }}
              style={{
                padding: '14px 32px', borderRadius: 14, border: '1px solid rgba(239,68,68,0.5)',
                background: 'rgba(239,68,68,0.2)', color: '#fca5a5',
                fontWeight: 700, fontSize: 14, cursor: 'pointer', letterSpacing: 0.5,
              }}
            >
              Decline
            </button>
            <button
              onClick={() => {
                stopRing();
                setGlobalIncomingCall(null);
                setCurrentScreen('chat');
                window.dispatchEvent(new CustomEvent('qnk-accept-call'));
              }}
              style={{
                padding: '14px 32px', borderRadius: 14, border: '1px solid rgba(34,197,94,0.5)',
                background: 'rgba(34,197,94,0.25)', color: '#86efac',
                fontWeight: 700, fontSize: 14, cursor: 'pointer', letterSpacing: 0.5,
              }}
            >
              Accept
            </button>
          </div>
        </div>
      </div>,
      document.body
    )}
    {/* Chat message toast — appears when a message arrives while on another screen */}
    {chatMsgAlert && createPortal(
      <div
        style={{
          position: 'fixed', top: 20, right: 20, zIndex: 2147483646,
          maxWidth: 340, fontFamily: 'system-ui, sans-serif',
          animation: 'slideInRight 0.3s ease-out',
        }}
      >
        <style>{`@keyframes slideInRight{from{transform:translateX(110%);opacity:0}to{transform:translateX(0);opacity:1}}`}</style>
        <div style={{
          background: 'linear-gradient(135deg, rgba(14,10,40,0.98), rgba(28,18,56,0.98))',
          border: '1.5px solid rgba(212,175,55,0.35)',
          borderRadius: 16, padding: '14px 16px',
          boxShadow: '0 8px 32px rgba(0,0,0,0.6), 0 0 24px rgba(212,175,55,0.1)',
        }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10, marginBottom: 10 }}>
            <div style={{
              width: 36, height: 36, borderRadius: '50%', flexShrink: 0,
              background: 'linear-gradient(135deg, rgba(212,175,55,0.3), rgba(255,165,0,0.15))',
              border: '1.5px solid rgba(212,175,55,0.4)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 16, color: '#d4af37', fontWeight: 700,
            }}>
              {(chatMsgAlert.contactName || chatMsgAlert.from).slice(0, 2).toUpperCase()}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <p style={{ color: '#fef3c7', fontSize: 13, fontWeight: 600, marginBottom: 2 }}>
                {chatMsgAlert.contactName || `${chatMsgAlert.from.slice(0, 8)}…${chatMsgAlert.from.slice(-6)}`}
              </p>
              <p style={{ color: 'rgba(203,213,225,0.75)', fontSize: 12, lineHeight: 1.4,
                overflow: 'hidden', display: '-webkit-box', WebkitLineClamp: 2,
                WebkitBoxOrient: 'vertical' as any,
              }}>
                {chatMsgAlert.content}
              </p>
            </div>
            <button
              onClick={() => { setChatMsgAlert(null); if (chatMsgTimerRef.current) clearTimeout(chatMsgTimerRef.current); }}
              style={{ background: 'none', border: 'none', color: 'rgba(212,175,55,0.4)', cursor: 'pointer', padding: 2, flexShrink: 0, fontSize: 16, lineHeight: 1 }}
            >✕</button>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={() => {
                setChatMsgAlert(null);
                if (chatMsgTimerRef.current) clearTimeout(chatMsgTimerRef.current);
              }}
              style={{
                flex: 1, padding: '7px 0', borderRadius: 10,
                background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(212,175,55,0.15)',
                color: 'rgba(212,175,55,0.6)', fontSize: 12, fontWeight: 600, cursor: 'pointer',
              }}
            >Dismiss</button>
            <button
              onClick={() => {
                const addr = chatMsgAlert.from;
                setChatMsgAlert(null);
                if (chatMsgTimerRef.current) clearTimeout(chatMsgTimerRef.current);
                setCurrentScreen('chat');
                setTimeout(() => window.dispatchEvent(new CustomEvent('qnk-open-conversation', { detail: { address: addr } })), 80);
              }}
              style={{
                flex: 1, padding: '7px 0', borderRadius: 10,
                background: 'linear-gradient(135deg, rgba(212,175,55,0.4), rgba(255,215,0,0.25))',
                border: '1px solid rgba(212,175,55,0.5)',
                color: '#fef3c7', fontSize: 12, fontWeight: 700, cursor: 'pointer',
              }}
            >Reply →</button>
          </div>
        </div>
      </div>,
      document.body
    )}
    </>
  );
}

export default App
