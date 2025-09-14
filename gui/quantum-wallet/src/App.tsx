import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import LoginScreen from './components/LoginScreen';
import Dashboard from './components/Dashboard';
import TransactionScreenV2 from './components/TransactionScreenV2';
import ExplorerScreen from './components/ExplorerScreen';
import SettingsScreen from './components/SettingsScreen';
import Navigation from './components/Navigation';
import TopBar from './components/TopBar';
import QuantumBackground from './components/QuantumBackground';
import { qnkAPI } from './services/api';
import './App.css';

type Screen = 'dashboard' | 'transactions' | 'explorer' | 'settings';

function App() {
  // Load authentication state from localStorage on mount
  const [authenticated, setAuthenticated] = useState(() => {
    const saved = localStorage.getItem('authenticated');
    return saved === 'true';
  });
  const [currentScreen, setCurrentScreen] = useState<Screen>('dashboard');
  const [nodeData, setNodeData] = useState({
    balance: 0,
    nodeId: '',
    blockHeight: 0,
    peers: 0,
    isOnline: false,
    qci: 0.42, // Quantum Coherence Index (42%)
  });

  // Save authentication state to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('authenticated', String(authenticated));
  }, [authenticated]);

  // Fetch initial node data and set up SSE for real-time updates
  useEffect(() => {
    if (!authenticated) return;

    const fetchInitialData = async () => {
      try {
        // Get node status for basic info
        const nodeResponse = await qnkAPI.getNodeStatus();
        if (nodeResponse.success && nodeResponse.data) {
          // Get wallet address from localStorage
          const walletAddress = localStorage.getItem('walletAddress');
          let walletBalance = 0;
          
          // If we have a wallet address, fetch its balance
          if (walletAddress) {
            try {
              const balanceResponse = await qnkAPI.getWalletBalance(walletAddress);
              if (balanceResponse.success && balanceResponse.data) {
                walletBalance = balanceResponse.data.balance_qnk || 0;
              }
            } catch (error) {
              console.error('Failed to fetch wallet balance:', error);
            }
          }

          setNodeData({
            balance: walletBalance,
            nodeId: nodeResponse.data.node_id,
            blockHeight: nodeResponse.data.current_height,
            peers: nodeResponse.data.connected_peers,
            isOnline: nodeResponse.data.network_health === 'healthy',
            qci: 0.42, // Quantum Coherence Index (42%)
          });
        }
      } catch (error) {
        console.error('Failed to fetch initial data:', error);
      }
    };

    fetchInitialData();

    // Set up SSE connection for real-time updates - use Vite proxy
    console.log('🔗 Setting up SSE connection to:', '/api/v1/events');
    const eventSource = new EventSource('/api/v1/events');
    
    // Add connection event listeners for debugging
    eventSource.onopen = () => {
      console.log('✅ SSE connection opened successfully');
    };
    
    eventSource.onerror = (error) => {
      console.error('❌ SSE connection error:', error);
      console.log('SSE readyState:', eventSource.readyState);
    };
    
    // Listen for balance updates
    eventSource.addEventListener('balance-updated', (event) => {
      console.log('📨 Received balance-updated event:', event);
      try {
        const data = JSON.parse(event.data);
        console.log('📊 Parsed balance update data:', data);
        const walletAddress = localStorage.getItem('walletAddress');
        console.log('💼 Current wallet address:', walletAddress);
        console.log('🔍 Event wallet address:', data.data?.wallet_address);
        
        // Only update if this balance update is for our wallet
        // Handle flexible wallet address matching (with/without trailing characters)
        const isMatchingWallet = walletAddress && data.data && (
          data.data.wallet_address === walletAddress ||
          data.data.wallet_address?.startsWith(walletAddress) ||
          walletAddress?.startsWith(data.data.wallet_address)
        );
        
        if (isMatchingWallet) {
          console.log('✅ Balance updated via SSE:', data.data.new_balance);
          setNodeData(prev => ({ ...prev, balance: data.data.new_balance }));
        } else {
          console.log('⚠️ Balance update ignored - wallet address mismatch or missing data');
          console.log('Expected:', walletAddress, 'Got:', data.data?.wallet_address);
        }
      } catch (error) {
        console.error('❌ Error parsing balance update event:', error);
        console.error('Raw event data:', event.data);
      }
    });

    // Listen for faucet events (immediate update)
    eventSource.addEventListener('faucet-dispensed', (event) => {
      console.log('🚰 Received faucet-dispensed event:', event);
      try {
        const data = JSON.parse(event.data);
        console.log('💰 Parsed faucet data:', data);
        const walletAddress = localStorage.getItem('walletAddress');
        console.log('💼 Current wallet address:', walletAddress);
        console.log('🔍 Faucet wallet address:', data.data?.wallet_address);
        
        // Only update if this faucet event is for our wallet
        // Handle flexible wallet address matching (with/without trailing characters)
        const isMatchingWallet = walletAddress && data.data && (
          data.data.wallet_address === walletAddress ||
          data.data.wallet_address?.startsWith(walletAddress) ||
          walletAddress?.startsWith(data.data.wallet_address)
        );
        
        if (isMatchingWallet) {
          console.log('✅ Faucet dispensed via SSE:', data.data.balance_after);
          setNodeData(prev => ({ ...prev, balance: data.data.balance_after }));
        } else {
          console.log('⚠️ Faucet event ignored - wallet address mismatch or missing data');
          console.log('Expected:', walletAddress, 'Got:', data.data?.wallet_address);
        }
      } catch (error) {
        console.error('❌ Error parsing faucet event:', error);
        console.error('Raw event data:', event.data);
      }
    });

    // Handle SSE connection errors
    eventSource.onerror = (error) => {
      console.error('SSE connection error:', error);
    };

    eventSource.onopen = () => {
      console.log('SSE connection established for real-time balance updates');
    };

    // Fallback: Poll balance every 5 seconds as backup to SSE (DISABLED - SSE works better)
    // DISABLED because polling was overriding correct SSE balance with wrong wallet address

    return () => {
      eventSource.close();
      // clearInterval(pollInterval); // Disabled
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
    return (
      <div className="min-h-screen bg-quantum-dark relative overflow-hidden">
        <QuantumBackground />
        <LoginScreen onAuthenticate={() => setAuthenticated(true)} />
      </div>
    );
  }

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
        
        <div className="flex flex-1 lg:flex-row">
          <Navigation 
            currentScreen={currentScreen} 
            onNavigate={setCurrentScreen}
            className="lg:w-20 xl:w-64"
          />
          
          <main className="flex-1 p-4 lg:p-8 pb-20 lg:pb-8">
          <AnimatePresence mode="wait">
            {currentScreen === 'dashboard' && (
              <motion.div
                key="dashboard"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
              >
                <Dashboard />
              </motion.div>
            )}
            
            {currentScreen === 'transactions' && (
              <motion.div
                key="transactions"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
              >
                <TransactionScreenV2 currentBalance={nodeData.balance} />
              </motion.div>
            )}
            
            {currentScreen === 'explorer' && (
              <motion.div
                key="explorer"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
              >
                <ExplorerScreen />
              </motion.div>
            )}
            
            {currentScreen === 'settings' && (
              <motion.div
                key="settings"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
              >
                <SettingsScreen onLogout={handleLogout} />
              </motion.div>
            )}
          </AnimatePresence>
        </main>
        </div>
      </div>
    </div>
  );
}

export default App
