import { useState, useEffect } from 'react';
import LoginScreen from './components/LoginScreen';
import Dashboard from './components/Dashboard';
import TransactionScreenV2 from './components/TransactionScreenV2';
import ExplorerScreen from './components/ExplorerScreen';
import DexScreen from './components/DexScreen';
import MiningScreen from './components/MiningScreen';
import VittuaVMScreen from './components/VittuaVMScreen';
import DownloadNodeScreen from './components/DownloadNodeScreen';
import SettingsScreen from './components/SettingsScreen';
import Navigation from './components/Navigation';
import TopBar from './components/TopBar';
import QuantumBackground from './components/QuantumBackground';
import './App.css';

type Screen = 'dashboard' | 'transactions' | 'explorer' | 'dex' | 'mining' | 'vm' | 'download' | 'settings';

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

    console.log('🎬 App.tsx: useEffect running for authenticated user');
    // No SSE or data fetching - Dashboard handles everything

    return () => {
      console.log('🎬 App.tsx: useEffect cleanup');
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
    console.log('🔓 Rendering LoginScreen');
    return (
      <div className="min-h-screen bg-quantum-dark relative overflow-hidden">
        <QuantumBackground />
        <LoginScreen onAuthenticate={() => setAuthenticated(true)} />
      </div>
    );
  }

  console.log('✅ Authenticated - Rendering main app');

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
            {currentScreen === 'dashboard' && <Dashboard key="dashboard-stable" />}
            {currentScreen === 'transactions' && <TransactionScreenV2 currentBalance={nodeData.balance} />}
            {currentScreen === 'dex' && <DexScreen />}
            {currentScreen === 'explorer' && <ExplorerScreen />}
            {currentScreen === 'mining' && <MiningScreen />}
            {currentScreen === 'vm' && <VittuaVMScreen />}
            {currentScreen === 'download' && <DownloadNodeScreen />}
            {currentScreen === 'settings' && <SettingsScreen onLogout={handleLogout} />}
        </main>
        </div>
      </div>
    </div>
  );
}

export default App
