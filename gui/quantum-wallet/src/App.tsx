import { useState, useEffect } from 'react';
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
import './App.css';

type Screen = 'dashboard' | 'transactions' | 'explorer' | 'dex' | 'mining' | 'vm' | 'download' | 'aichat' | 'settings';

function App() {
  console.log('🚀 App function executing - TOP OF FUNCTION');

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
  const [nodeData, setNodeData] = useState(() => {
    const cachedBalance = localStorage.getItem('cachedBalance');
    const initialBalance = cachedBalance ? parseFloat(cachedBalance) : 0;
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

  // Debounce balance updates - only apply after 300ms of stability
  useEffect(() => {
    if (pendingBalanceUpdate === null) return;

    console.log('⏱️ [BALANCE DEBUG] Pending balance update queued:', {
      pendingValue: pendingBalanceUpdate,
      currentValue: nodeData.balance,
      willUpdateIn: '300ms'
    });

    const timer = setTimeout(() => {
      console.log('✅ [BALANCE DEBUG] Applying debounced balance update:', {
        oldBalance: nodeData.balance,
        newBalance: pendingBalanceUpdate,
        source: 'debounced'
      });
      setNodeData(prev => ({ ...prev, balance: pendingBalanceUpdate }));
      localStorage.setItem('cachedBalance', pendingBalanceUpdate.toString());
      setPendingBalanceUpdate(null);
    }, 300);

    return () => clearTimeout(timer);
  }, [pendingBalanceUpdate, nodeData.balance]);

  // Fetch initial node data and set up SSE for real-time updates
  useEffect(() => {
    if (!authenticated) return;

    console.log('🎬 App.tsx: Setting up authenticated SSE for real-time balance updates');

    let mounted = true;

    const fetchNodeStatus = async () => {
      try {
        const response = await fetch('/api/v1/node/status');
        if (!response.ok) throw new Error('Failed to fetch node status');

        const data = await response.json();
        if (!mounted) return;

        if (data.success && data.data) {
          // Don't fetch balance here - SSE will provide it
          // Using cached balance prevents flickering between API (RocksDB) and SSE (in-memory) values
          const cachedBalance = localStorage.getItem('cachedBalance');
          let walletBalance = cachedBalance ? parseFloat(cachedBalance) : 0;
          console.log('💰 App.tsx: Using cached balance (SSE will update):', walletBalance);

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
        console.log('🟡 [BALANCE DEBUG] Custom balance-update event:', {
          newBalance: newBalance,
          currentBalance: nodeData.balance,
          source: 'custom-event'
        });

        // Use debounced update to prevent flickering
        setPendingBalanceUpdate(newBalance);
        console.log('⏱️ [BALANCE DEBUG] Balance update queued from custom event (debounced):', newBalance);
      } else {
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
                console.log('🟢 [BALANCE DEBUG] SSE balance-updated event:', {
                  oldBalance: balanceData.old_balance,
                  newBalance: balanceData.new_balance,
                  currentNodeDataBalance: nodeData.balance,
                  reason: balanceData.change_reason,
                  source: 'SSE'
                });
                // Use debounced update to prevent flickering
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
                console.log('📢 App.tsx: Dispatched wallet-balance-updated event for Dashboard');
              } else {
                console.log('❌ App.tsx: Balance update IGNORED (not for current wallet)', {
                  eventWallet: eventHex,
                  currentWallet: currentHex
                });
              }
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
          }
        };

        // Start reading the stream
        readStream();

      } catch (error) {
        console.error('❌ App.tsx: Failed to establish authenticated SSE connection:', error);
      }
    };

    // Initialize authenticated SSE connection
    setupAuthenticatedSSE();

    return () => {
      console.log('🎬 App.tsx: useEffect cleanup - closing SSE');
      mounted = false;
      window.removeEventListener('balance-update', handleBalanceUpdate);
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
    localStorage.removeItem('faucetTransactions'); // Clear transaction history
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
    return (
      <div className="min-h-screen bg-quantum-dark relative overflow-hidden">
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
    <div className="min-h-screen bg-quantum-dark relative overflow-hidden">
      <QuantumBackground />

      <div className="relative z-10 flex flex-col min-h-screen">
        {/* Global TopBar */}
        <TopBar
          currentBalance={nodeData.balance}
          nodeId={nodeData.nodeId}
          blockHeight={nodeData.blockHeight}
          peers={nodeData.peers}
          isOnline={nodeData.isOnline}
          qci={nodeData.qci}
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
            {currentScreen === 'dashboard' && <Dashboard key="dashboard-stable" onNavigateToSend={handleCoinSendClick} />}
            {currentScreen === 'transactions' && <TransactionScreenV2 currentBalance={nodeData.balance} />}
            {currentScreen === 'dex' && <DexScreen />}
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
        </div>
      </div>
    </div>
  );
}

export default App
