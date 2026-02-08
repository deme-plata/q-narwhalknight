import React, { useState, useEffect } from 'react';
import { Shield, Check, X, AlertTriangle, ExternalLink, Lock, Eye, Send } from 'lucide-react';

interface OAuth2Client {
  client_id: string;
  name: string;
  website: string;
  logo_url?: string;
  scopes: string[];
}

interface ConsentScreenProps {
  clientId?: string;
  redirectUri?: string;
  scope?: string;
  state?: string;
}

const SCOPE_DESCRIPTIONS: Record<string, { name: string; description: string; icon: React.FC<any> }> = {
  'read:balance': {
    name: 'Read Balance',
    description: 'View your QUG, QUGUSD, and token balances',
    icon: Eye
  },
  'send:transaction': {
    name: 'Send Transactions',
    description: 'Send QUG and tokens on your behalf',
    icon: Send
  },
  'read:transactions': {
    name: 'Read Transaction History',
    description: 'View your past transactions',
    icon: Eye
  },
  'manage:tokens': {
    name: 'Manage Tokens',
    description: 'Create and manage custom tokens',
    icon: Lock
  }
};

export default function OAuth2ConsentScreen({ clientId, redirectUri, scope, state }: ConsentScreenProps) {
  const [client, setClient] = useState<OAuth2Client | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [walletAddress, setWalletAddress] = useState<string>('');
  const [processing, setProcessing] = useState(false);

  // Parse query parameters from URL
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const urlClientId = clientId || urlParams.get('client_id');

    if (!urlClientId) {
      setError('Missing client_id parameter');
      setLoading(false);
      return;
    }

    // Fetch client information
    fetchClientInfo(urlClientId);

    // Get current wallet address from session/localStorage
    const storedWallet = localStorage.getItem('currentWalletAddress');
    if (storedWallet) {
      setWalletAddress(storedWallet);
    } else {
      setError('No wallet connected. Please log in first.');
      setLoading(false);
    }
  }, [clientId, redirectUri, scope, state]);

  const fetchClientInfo = async (clientId: string) => {
    try {
      // In production, this would fetch from /api/v1/oauth2/clients/{clientId}
      // For now, we'll mock the response
      const response = await fetch(`/api/v1/oauth2/clients/${clientId}`);

      if (!response.ok) {
        throw new Error('Client not found');
      }

      const data = await response.json();

      if (data.success) {
        setClient(data.data);
      } else {
        setError(data.error || 'Failed to load client information');
      }
    } catch (err) {
      console.error('Failed to fetch client info:', err);
      setError('Failed to load application information');
    } finally {
      setLoading(false);
    }
  };

  const handleApprove = async () => {
    if (!client || !walletAddress) return;

    setProcessing(true);
    try {
      const urlParams = new URLSearchParams(window.location.search);
      const scopes = (scope || urlParams.get('scope') || 'read:balance').split(' ');

      const response = await fetch('/api/v1/oauth2/consent', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          wallet_address: walletAddress,
          client_id: client.client_id,
          scopes: scopes,
          approved: true,
          auth_request_id: urlParams.get('state') || '',
          redirect_uri: redirectUri || urlParams.get('redirect_uri') || undefined,
          code_challenge: urlParams.get('code_challenge') || undefined,
          code_challenge_method: urlParams.get('code_challenge_method') || undefined,
        })
      });

      const data = await response.json();

      if (data.success) {
        // Redirect back to application with authorization code
        const authCode = data.data;
        const redirectUrl = new URL(redirectUri || urlParams.get('redirect_uri') || '');
        redirectUrl.searchParams.set('code', authCode);
        if (state || urlParams.get('state')) {
          redirectUrl.searchParams.set('state', state || urlParams.get('state') || '');
        }

        window.location.href = redirectUrl.toString();
      } else {
        setError(data.error || 'Failed to grant consent');
        setProcessing(false);
      }
    } catch (err) {
      console.error('Consent error:', err);
      setError('Failed to process consent');
      setProcessing(false);
    }
  };

  const handleDeny = () => {
    const urlParams = new URLSearchParams(window.location.search);
    const redirectUrl = new URL(redirectUri || urlParams.get('redirect_uri') || '');
    redirectUrl.searchParams.set('error', 'access_denied');
    redirectUrl.searchParams.set('error_description', 'User denied access');
    if (state || urlParams.get('state')) {
      redirectUrl.searchParams.set('state', state || urlParams.get('state') || '');
    }

    window.location.href = redirectUrl.toString();
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-quantum-dark via-quantum-indigo/20 to-quantum-dark flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-quantum-cyan border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-white">Loading application information...</p>
        </div>
      </div>
    );
  }

  if (error && !client) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-quantum-dark via-quantum-indigo/20 to-quantum-dark flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-quantum-indigo/20 backdrop-blur-xl border border-red-500/50 rounded-2xl p-8 text-center">
          <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-white mb-2">Authorization Error</h2>
          <p className="text-gray-300 mb-6">{error}</p>
          <button
            onClick={() => window.close()}
            className="px-6 py-3 bg-red-500 hover:bg-red-600 text-white font-bold rounded-xl transition-all"
          >
            Close Window
          </button>
        </div>
      </div>
    );
  }

  const urlParams = new URLSearchParams(window.location.search);
  const requestedScopes = (scope || urlParams.get('scope') || 'read:balance').split(' ');

  return (
    <div className="min-h-screen bg-gradient-to-br from-quantum-dark via-quantum-indigo/20 to-quantum-dark flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-4 mb-4">
            <div className="w-16 h-16 bg-gradient-to-br from-quantum-cyan via-quantum-purple to-quantum-pink rounded-2xl flex items-center justify-center">
              <Shield className="w-8 h-8 text-white" />
            </div>
            <X className="w-6 h-6 text-gray-500" />
            {client?.logo_url ? (
              <img src={client.logo_url} alt={client.name} className="w-16 h-16 rounded-2xl object-cover" />
            ) : (
              <div className="w-16 h-16 bg-gray-700 rounded-2xl flex items-center justify-center">
                <ExternalLink className="w-8 h-8 text-gray-400" />
              </div>
            )}
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">
            Authorize Application
          </h1>
          <p className="text-gray-400">
            <span className="text-quantum-cyan font-bold">{client?.name}</span> wants to access your Quillon Wallet
          </p>
        </div>

        {/* Consent Card */}
        <div className="bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl p-8 mb-6">
          {/* Application Info */}
          <div className="mb-6 pb-6 border-b border-quantum-purple/30">
            <h3 className="text-white font-bold mb-2">Application Details</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Name:</span>
                <span className="text-white font-medium">{client?.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Website:</span>
                <a
                  href={client?.website}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-quantum-cyan hover:underline flex items-center gap-1"
                >
                  {client?.website}
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </div>
          </div>

          {/* Wallet Info */}
          <div className="mb-6 pb-6 border-b border-quantum-purple/30">
            <h3 className="text-white font-bold mb-2">Your Wallet</h3>
            <div className="p-3 bg-quantum-dark/30 rounded-lg">
              <p className="text-quantum-cyan font-mono text-sm break-all">{walletAddress}</p>
            </div>
          </div>

          {/* Permissions */}
          <div className="mb-6">
            <h3 className="text-white font-bold mb-4 flex items-center gap-2">
              <Lock className="w-5 h-5 text-quantum-purple" />
              Requested Permissions
            </h3>
            <div className="space-y-3">
              {requestedScopes.map((scopeId) => {
                const scopeInfo = SCOPE_DESCRIPTIONS[scopeId] || {
                  name: scopeId,
                  description: 'Unknown permission',
                  icon: AlertTriangle
                };
                const Icon = scopeInfo.icon;

                return (
                  <div
                    key={scopeId}
                    className="flex items-start gap-3 p-4 bg-quantum-dark/30 rounded-lg border border-quantum-purple/20"
                  >
                    <div className="w-10 h-10 bg-quantum-purple/20 rounded-lg flex items-center justify-center flex-shrink-0">
                      <Icon className="w-5 h-5 text-quantum-purple" />
                    </div>
                    <div className="flex-1">
                      <h4 className="text-white font-bold mb-1">{scopeInfo.name}</h4>
                      <p className="text-gray-400 text-sm">{scopeInfo.description}</p>
                    </div>
                    <Check className="w-5 h-5 text-quantum-green flex-shrink-0 mt-2" />
                  </div>
                );
              })}
            </div>
          </div>

          {/* Security Notice */}
          <div className="p-4 bg-quantum-cyan/10 border border-quantum-cyan/30 rounded-lg mb-6">
            <div className="flex items-start gap-3">
              <Shield className="w-5 h-5 text-quantum-cyan flex-shrink-0 mt-0.5" />
              <div className="text-sm">
                <p className="text-white font-bold mb-1">Quantum-Secure Connection</p>
                <p className="text-gray-300">
                  This authorization uses post-quantum cryptography (Kyber1024) to protect your data.
                  You can revoke access at any time from your wallet settings.
                </p>
              </div>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="p-4 bg-red-500/10 border border-red-500/50 rounded-lg mb-6">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-red-500" />
                <p className="text-red-400 text-sm">{error}</p>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-4">
            <button
              onClick={handleDeny}
              disabled={processing}
              className="flex-1 py-4 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-white font-bold rounded-xl transition-all flex items-center justify-center gap-2"
            >
              <X className="w-5 h-5" />
              Deny Access
            </button>
            <button
              onClick={handleApprove}
              disabled={processing}
              className="flex-1 py-4 bg-gradient-to-r from-quantum-cyan to-quantum-purple hover:from-quantum-cyan/80 hover:to-quantum-purple/80 disabled:from-gray-700 disabled:to-gray-800 text-white font-bold rounded-xl transition-all flex items-center justify-center gap-2"
            >
              {processing ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  Authorizing...
                </>
              ) : (
                <>
                  <Check className="w-5 h-5" />
                  Authorize
                </>
              )}
            </button>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-sm text-gray-500">
          <p>
            By authorizing, you allow <strong className="text-white">{client?.name}</strong> to perform the actions listed above.
          </p>
          <p className="mt-2">
            Powered by <span className="text-quantum-cyan">Quillon Wallet</span>
          </p>
        </div>
      </div>
    </div>
  );
}
