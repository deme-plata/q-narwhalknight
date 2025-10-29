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
  const [nodeData, setNodeData] = useState({
    balance: 0,
    nodeId: '',
    blockHeight: 0,
    peers: 0,
    isOnline: false,
    qci: 0.42, // Quantum Coherence Index (42%)
  });

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
          const currentWalletAddress = localStorage.getItem('walletAddress');
          let walletBalance = 0;

          if (currentWalletAddress) {
            try {
              const balanceResponse = await fetch(`/api/v1/wallets/${currentWalletAddress}/balance`);
              const balanceData = await balanceResponse.json();

              if (balanceData.success && balanceData.data) {
                walletBalance = balanceData.data.balance_qnk || 0;
                // Cache balance for use after refresh
                localStorage.setItem('cachedBalance', walletBalance.toString());
                console.log('💰 App.tsx: Cached balance from API:', walletBalance);
              } else {
                // Authentication failed - calculate from faucet transactions
                console.warn('⚠️ App.tsx: Balance fetch failed, calculating from transaction history');

                // First try cached balance
                const cachedBalance = localStorage.getItem('cachedBalance');
                if (cachedBalance && parseFloat(cachedBalance) > 0) {
                  walletBalance = parseFloat(cachedBalance);
                  console.log('💰 App.tsx: Using cached balance:', walletBalance);
                } else {
                  // Calculate from faucet transactions as fallback
                  try {
                    const storedTxs = localStorage.getItem('faucetTransactions');
                    if (storedTxs) {
                      const transactions = JSON.parse(storedTxs);
                      walletBalance = transactions.reduce((total: number, tx: any) => {
                        return total + (tx.type === 'receive' ? tx.amount : 0);
                      }, 0);
                      console.log('💰 App.tsx: Calculated balance from faucet transactions:', walletBalance, 'QNK');
                      // Cache the calculated balance
                      localStorage.setItem('cachedBalance', walletBalance.toString());
                    }
                  } catch (txErr) {
                    console.error('Failed to calculate balance from transactions:', txErr);
                  }
                }
              }
            } catch (balanceErr) {
              console.warn('Failed to fetch wallet balance:', balanceErr);

              // Fallback 1: use cached balance from localStorage
              const cachedBalance = localStorage.getItem('cachedBalance');
              if (cachedBalance && parseFloat(cachedBalance) > 0) {
                walletBalance = parseFloat(cachedBalance);
                console.log('💰 App.tsx: Using cached balance (error fallback):', walletBalance);
              } else {
                // Fallback 2: calculate from faucet transactions
                try {
                  const storedTxs = localStorage.getItem('faucetTransactions');
                  if (storedTxs) {
                    const transactions = JSON.parse(storedTxs);
                    walletBalance = transactions.reduce((total: number, tx: any) => {
                      return total + (tx.type === 'receive' ? tx.amount : 0);
                    }, 0);
                    console.log('💰 App.tsx: Calculated balance from faucet transactions (error fallback):', walletBalance, 'QNK');
                    // Cache the calculated balance
                    localStorage.setItem('cachedBalance', walletBalance.toString());
                  }
                } catch (txErr) {
                  console.error('Failed to calculate balance from transactions:', txErr);
                }
              }
            }
          }

          if (mounted) {
            setNodeData({
              balance: walletBalance,
              nodeId: data.data.node_id || '',
              blockHeight: data.data.current_height || 0,
              peers: data.data.connected_peers || 0,
              isOnline: data.data.network_health === 'healthy',
              qci: 0.42
            });
          }
        }
      } catch (err) {
        console.error('Error fetching node status:', err);
      }
    };

    // Initial fetch
    fetchNodeStatus();

    // Listen for custom balance update events from Dashboard
    const handleBalanceUpdate = (event: Event) => {
      const customEvent = event as CustomEvent;
      console.log('💰 App.tsx: Received custom balance-update event:', customEvent.detail);
      if (customEvent.detail?.balance !== undefined) {
        setNodeData(prev => ({ ...prev, balance: customEvent.detail.balance }));
      } else {
        // If no balance in event, refresh from API
        fetchNodeStatus();
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

              console.log('💰 App.tsx: Balance update SSE event:', {
                eventWallet: eventHex,
                currentWallet: currentHex,
                match: eventHex === currentHex,
                newBalance: balanceData.new_balance,
                reason: balanceData.change_reason
              });

              // Only update if this balance event is for the current wallet
              if (currentHex && eventHex === currentHex) {
                console.log('✅ App.tsx: Balance update applied:', balanceData.new_balance);
                setNodeData(prev => ({ ...prev, balance: balanceData.new_balance }));
                // Also update cached balance
                localStorage.setItem('cachedBalance', balanceData.new_balance.toString());
              } else {
                console.log('❌ App.tsx: Balance update ignored (not for current wallet)');
              }
            } else if (type === 'faucet-dispensed') {
              console.log('🚰 App.tsx: Faucet dispensed - refreshing balance');
              fetchNodeStatus();
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
            {currentScreen === 'dashboard' && <Dashboard key="dashboard-stable" />}
            {currentScreen === 'transactions' && <TransactionScreenV2 currentBalance={nodeData.balance} />}
            {currentScreen === 'dex' && <DexScreen />}
            {currentScreen === 'explorer' && <ExplorerScreen />}
            {currentScreen === 'mining' && <MiningScreen />}
            {currentScreen === 'vm' && <VittuaVMScreen />}
            {currentScreen === 'aichat' && <AIChatScreen />}
            {currentScreen === 'download' && <DownloadNodeScreen />}
            {currentScreen === 'settings' && <SettingsScreen onLogout={handleLogout} />}
        </main>
        </div>
      </div>
    </div>
  );
}

export default App
