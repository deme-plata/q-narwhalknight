/**
 * React + TypeScript Integration Example
 *
 * This example shows how to integrate Quillon Wallet OAuth2
 * into a React application with TypeScript.
 */

import React, { useState, useEffect, createContext, useContext } from 'react';
import QullionOAuth2Client from '@quillon/oauth2-sdk';

// ============================================================================
// Context Setup
// ============================================================================

interface TokenBalance {
  symbol: string;
  balance: number;
  balance_base_units: string;
  name: string;
}

interface QullionContextType {
  client: QullionOAuth2Client;
  isAuthenticated: boolean;
  userInfo: any | null;
  balance: number | null;
  tokens: Record<string, TokenBalance>;
  login: () => Promise<void>;
  logout: () => Promise<void>;
  refreshBalance: () => Promise<void>;
  swapTokens: (from: string, to: string, amount: number) => Promise<any>;
}

const QullionContext = createContext<QullionContextType | null>(null);

export function useQuillon() {
  const context = useContext(QullionContext);
  if (!context) {
    throw new Error('useQuillon must be used within QullionProvider');
  }
  return context;
}

// ============================================================================
// Provider Component
// ============================================================================

interface QullionProviderProps {
  clientId: string;
  clientSecret?: string;
  redirectUri: string;
  scopes?: string[];
  children: React.ReactNode;
}

export function QullionProvider({
  clientId,
  clientSecret,
  redirectUri,
  scopes = ['read:balance', 'send:transaction', 'read:transactions'],
  children
}: QullionProviderProps) {
  const [client] = useState(() => new QullionOAuth2Client({
    clientId,
    clientSecret,
    redirectUri,
    scopes,
    walletUrl: import.meta.env.VITE_QUILLON_WALLET_URL || 'https://wallet.quillon.xyz',
    apiUrl: import.meta.env.VITE_QUILLON_API_URL || 'https://api.quillon.xyz'
  }));

  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userInfo, setUserInfo] = useState<any | null>(null);
  const [balance, setBalance] = useState<number | null>(null);
  const [tokens, setTokens] = useState<Record<string, TokenBalance>>({});

  // Handle OAuth2 callback
  useEffect(() => {
    const handleCallback = async () => {
      const urlParams = new URLSearchParams(window.location.search);
      if (urlParams.has('code')) {
        try {
          await client.handleCallback();
          window.history.replaceState({}, document.title, window.location.pathname);
          setIsAuthenticated(true);
        } catch (error) {
          console.error('OAuth2 callback error:', error);
        }
      } else if (client.isAuthenticated()) {
        setIsAuthenticated(true);
      }
    };

    handleCallback();
  }, [client]);

  // Load user data when authenticated
  useEffect(() => {
    if (isAuthenticated) {
      loadUserData();
      setupSSE();
    }
  }, [isAuthenticated]);

  // Setup Server-Sent Events for real-time updates
  const setupSSE = () => {
    const apiUrl = import.meta.env.VITE_QUILLON_API_URL || 'https://api.quillon.xyz';
    const eventSource = new EventSource(`${apiUrl}/api/v1/stream`);

    eventSource.addEventListener('balance-updated', (event: MessageEvent) => {
      console.log('💰 Balance updated via SSE');
      loadUserData(); // Refresh balance when update detected
    });

    eventSource.addEventListener('transaction-confirmed', (event: MessageEvent) => {
      console.log('✅ Transaction confirmed via SSE');
      loadUserData(); // Refresh on transaction confirmation
    });

    eventSource.onerror = (error) => {
      console.warn('SSE connection error:', error);
      eventSource.close();
      // Retry connection after 5 seconds
      setTimeout(setupSSE, 5000);
    };

    return () => eventSource.close();
  };

  const loadUserData = async () => {
    try {
      const info = await client.getUserInfo();
      setUserInfo(info);

      // Fetch QUG balance
      const bal = await client.getBalance('QUG');
      setBalance(bal);

      // Fetch all token balances
      const apiUrl = import.meta.env.VITE_QUILLON_API_URL || 'https://api.quillon.xyz';
      const accessToken = await client.getAccessToken();

      const response = await fetch(`${apiUrl}/api/v1/wallet/tokens`, {
        headers: {
          'Authorization': `Bearer ${accessToken}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        const tokenBalances: Record<string, TokenBalance> = {};

        // Parse token data
        if (data.tokens) {
          Object.entries(data.tokens).forEach(([key, value]: [string, any]) => {
            tokenBalances[key] = {
              symbol: value.symbol || key.toUpperCase(),
              balance: parseFloat(value.balance) || 0,
              balance_base_units: value.balance_base_units || '0',
              name: value.name || key
            };
          });
        }

        setTokens(tokenBalances);
      }
    } catch (error) {
      console.error('Failed to load user data:', error);
      setIsAuthenticated(false);
    }
  };

  const login = async () => {
    await client.authorize();
  };

  const logout = async () => {
    await client.revoke();
    setIsAuthenticated(false);
    setUserInfo(null);
    setBalance(null);
  };

  const refreshBalance = async () => {
    await loadUserData();
  };

  const swapTokens = async (from: string, to: string, amount: number) => {
    const apiUrl = import.meta.env.VITE_QUILLON_API_URL || 'https://api.quillon.xyz';
    const accessToken = await client.getAccessToken();

    const response = await fetch(`${apiUrl}/api/v1/dex/swap`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${accessToken}`
      },
      body: JSON.stringify({
        from_token: from,
        to_token: to,
        amount_in: Math.floor(amount * 100000000) // Convert to base units
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Swap failed');
    }

    const result = await response.json();

    // Trigger balance refresh after swap
    await loadUserData();

    return result;
  };

  return (
    <QullionContext.Provider
      value={{
        client,
        isAuthenticated,
        userInfo,
        balance,
        tokens,
        login,
        logout,
        refreshBalance,
        swapTokens
      }}
    >
      {children}
    </QullionContext.Provider>
  );
}

// ============================================================================
// Example Components
// ============================================================================

export function QullionLoginButton() {
  const { isAuthenticated, login, logout, userInfo } = useQuillon();

  if (isAuthenticated && userInfo) {
    return (
      <div className="flex items-center gap-4">
        <div className="text-sm">
          <div className="font-mono text-xs text-gray-600">
            {userInfo.wallet_address.substring(0, 16)}...
          </div>
        </div>
        <button
          onClick={logout}
          className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600"
        >
          Disconnect
        </button>
      </div>
    );
  }

  return (
    <button
      onClick={login}
      className="px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:from-purple-600 hover:to-pink-600"
    >
      Connect Quillon Wallet
    </button>
  );
}

export function QullionBalanceDisplay() {
  const { balance, tokens, refreshBalance } = useQuillon();

  if (balance === null) {
    return <div>Loading balance...</div>;
  }

  return (
    <div className="bg-gradient-to-br from-purple-100 to-pink-100 rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="text-sm text-gray-600">Your Balances</div>
        <button
          onClick={refreshBalance}
          className="px-3 py-1 text-xs bg-white text-purple-600 rounded-lg hover:bg-gray-50"
        >
          🔄 Refresh
        </button>
      </div>

      {/* Primary QUG Balance */}
      <div className="text-3xl font-bold text-purple-600 mb-4">
        {(balance / 100000000).toFixed(8)} QUG
      </div>

      {/* Other Token Balances */}
      <div className="space-y-2">
        {Object.entries(tokens).map(([key, token]) => (
          <div key={key} className="flex items-center justify-between p-3 bg-white/70 rounded-lg">
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-gray-700">{token.symbol}</span>
              <span className="text-xs text-gray-500">{token.name}</span>
            </div>
            <span className="text-sm font-mono text-gray-800">
              {token.balance.toFixed(8)}
            </span>
          </div>
        ))}
      </div>

      <div className="mt-4 text-xs text-gray-500 text-center">
        💡 Balances update automatically via real-time SSE
      </div>
    </div>
  );
}

export function QullionSendForm() {
  const { client, refreshBalance } = useQuillon();
  const [recipient, setRecipient] = useState('');
  const [amount, setAmount] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<{ message: string; isError: boolean } | null>(null);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!recipient || !amount) {
      setStatus({ message: 'Please fill in all fields', isError: true });
      return;
    }

    setLoading(true);
    setStatus(null);

    try {
      await client.sendTransaction({
        to: recipient,
        amount: Math.floor(parseFloat(amount) * 100000000),
        token: 'QUG'
      });

      setStatus({ message: 'Transaction sent successfully!', isError: false });
      setRecipient('');
      setAmount('');
      await refreshBalance();
    } catch (error: any) {
      setStatus({ message: error.message || 'Transaction failed', isError: true });
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSend} className="bg-white rounded-xl p-6 shadow-lg">
      <h3 className="text-xl font-bold mb-4">Send QUG</h3>

      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Recipient Address
        </label>
        <input
          type="text"
          value={recipient}
          onChange={(e) => setRecipient(e.target.value)}
          placeholder="0x..."
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
        />
      </div>

      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Amount (QUG)
        </label>
        <input
          type="number"
          value={amount}
          onChange={(e) => setAmount(e.target.value)}
          placeholder="10.5"
          step="0.00000001"
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
        />
      </div>

      {status && (
        <div
          className={`mb-4 p-3 rounded-lg ${
            status.isError ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
          }`}
        >
          {status.message}
        </div>
      )}

      <button
        type="submit"
        disabled={loading}
        className="w-full px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:from-purple-600 hover:to-pink-600 disabled:opacity-50"
      >
        {loading ? 'Sending...' : 'Send Transaction'}
      </button>
    </form>
  );
}

export function QullionTransactionHistory() {
  const { client } = useQuillon();
  const [transactions, setTransactions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadTransactions();
  }, []);

  const loadTransactions = async () => {
    try {
      const txs = await client.getTransactionHistory(20);
      setTransactions(txs);
    } catch (error) {
      console.error('Failed to load transactions:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div>Loading transactions...</div>;
  }

  if (transactions.length === 0) {
    return <div className="text-gray-500">No transactions yet</div>;
  }

  return (
    <div className="bg-white rounded-xl p-6 shadow-lg">
      <h3 className="text-xl font-bold mb-4">Transaction History</h3>
      <div className="space-y-3">
        {transactions.map((tx) => (
          <div
            key={tx.id}
            className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
          >
            <div>
              <div className="font-medium">
                {(tx.amount / 100000000).toFixed(8)} {tx.token}
              </div>
              <div className="text-sm text-gray-500">
                {tx.from.substring(0, 16)}... → {tx.to.substring(0, 16)}...
              </div>
            </div>
            <div
              className={`px-3 py-1 rounded-full text-xs font-medium ${
                tx.status === 'confirmed'
                  ? 'bg-green-100 text-green-700'
                  : tx.status === 'pending'
                  ? 'bg-yellow-100 text-yellow-700'
                  : 'bg-red-100 text-red-700'
              }`}
            >
              {tx.status}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function QullionSwapForm() {
  const { tokens, swapTokens } = useQuillon();
  const [fromToken, setFromToken] = useState('qug');
  const [toToken, setToToken] = useState('qugusd');
  const [amount, setAmount] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<{ message: string; isError: boolean } | null>(null);

  const availableTokens = [
    { id: 'qug', symbol: 'QUG', name: 'Quillon' },
    { id: 'qugusd', symbol: 'QUGUSD', name: 'Quillon USD' },
    ...Object.entries(tokens).map(([key, token]) => ({
      id: key,
      symbol: token.symbol,
      name: token.name
    }))
  ];

  const handleSwap = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!amount || parseFloat(amount) <= 0) {
      setStatus({ message: 'Please enter a valid amount', isError: true });
      return;
    }

    setLoading(true);
    setStatus(null);

    try {
      const result = await swapTokens(fromToken, toToken, parseFloat(amount));

      setStatus({
        message: `✅ Swap successful! Received ${(result.data?.amount_out / 100000000 || 0).toFixed(8)} ${toToken.toUpperCase()}`,
        isError: false
      });
      setAmount('');

      // Show success for 3 seconds
      setTimeout(() => setStatus(null), 3000);
    } catch (error: any) {
      setStatus({ message: error.message || 'Swap failed', isError: true });
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSwap} className="bg-white rounded-xl p-6 shadow-lg">
      <h3 className="text-xl font-bold mb-4">🔄 Swap Tokens</h3>

      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          From
        </label>
        <div className="flex gap-2">
          <input
            type="number"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
            placeholder="0.0"
            step="0.00000001"
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
          <select
            value={fromToken}
            onChange={(e) => setFromToken(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            {availableTokens.map(token => (
              <option key={token.id} value={token.id}>
                {token.symbol}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="flex justify-center my-3">
        <div className="p-2 bg-purple-100 rounded-full">
          <svg className="w-5 h-5 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
        </div>
      </div>

      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          To
        </label>
        <select
          value={toToken}
          onChange={(e) => setToToken(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
        >
          {availableTokens.filter(t => t.id !== fromToken).map(token => (
            <option key={token.id} value={token.id}>
              {token.symbol} - {token.name}
            </option>
          ))}
        </select>
      </div>

      {status && (
        <div
          className={`mb-4 p-3 rounded-lg ${
            status.isError ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
          }`}
        >
          {status.message}
        </div>
      )}

      <button
        type="submit"
        disabled={loading}
        className="w-full px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:from-purple-600 hover:to-pink-600 disabled:opacity-50"
      >
        {loading ? '⏳ Swapping...' : '🔄 Swap Tokens'}
      </button>

      <div className="mt-3 text-xs text-gray-500 text-center">
        💡 Balances will update automatically after swap
      </div>
    </form>
  );
}

// ============================================================================
// Example App
// ============================================================================

export default function App() {
  return (
    <QullionProvider
      clientId="your-client-id"
      redirectUri={window.location.origin + '/callback'}
      scopes={['read:balance', 'send:transaction', 'read:transactions']}
    >
      <div className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-50 p-8">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="flex items-center justify-between mb-8">
            <h1 className="text-3xl font-bold text-gray-900">
              My Quillon App
            </h1>
            <QullionLoginButton />
          </div>

          {/* Protected Content */}
          <ProtectedContent />
        </div>
      </div>
    </QullionProvider>
  );
}

function ProtectedContent() {
  const { isAuthenticated } = useQuillon();

  if (!isAuthenticated) {
    return (
      <div className="text-center py-20">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Welcome to Quillon App
        </h2>
        <p className="text-gray-600 mb-8">
          Connect your Quillon Wallet to get started
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Balance */}
      <div className="md:col-span-2">
        <QullionBalanceDisplay />
      </div>

      {/* Swap Form - NEW! */}
      <QullionSwapForm />

      {/* Send Form */}
      <QullionSendForm />

      {/* Transaction History */}
      <div className="md:col-span-2">
        <QullionTransactionHistory />
      </div>
    </div>
  );
}
