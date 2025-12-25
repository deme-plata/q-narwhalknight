import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Coins, Send, ChevronDown, ChevronUp, Loader2, AlertCircle } from 'lucide-react';
import { qnkAPI } from '../services/api';

interface CustomToken {
  symbol: string;
  name: string;
  balance: number;
  contractAddress: string;
  decimals?: number;
}

interface CustomTokensCardProps {
  onSendToken: (tokenSymbol: string, contractAddress: string) => void;
}

export default function CustomTokensCard({ onSendToken }: CustomTokensCardProps) {
  const [customTokens, setCustomTokens] = useState<CustomToken[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(true);

  useEffect(() => {
    fetchCustomTokens();

    // Refresh every 10 seconds
    const interval = setInterval(fetchCustomTokens, 10000);

    // v1.4.10-beta: Listen for token balance updates via SSE for instant refresh
    const handleTokenBalanceUpdate = (event: CustomEvent) => {
      const { tokenSymbol, newBalance, reason } = event.detail;
      console.log('🪙 [CustomTokens] Balance update received via SSE:', { tokenSymbol, newBalance, reason });

      // Update the balance for the matching token immediately
      setCustomTokens(prev => prev.map(token => {
        if (token.symbol === tokenSymbol) {
          console.log(`✅ [CustomTokens] Updated ${token.symbol} balance: ${token.balance} → ${newBalance}`);
          return { ...token, balance: newBalance };
        }
        return token;
      }));
    };

    window.addEventListener('token-balance-updated', handleTokenBalanceUpdate as EventListener);

    return () => {
      clearInterval(interval);
      window.removeEventListener('token-balance-updated', handleTokenBalanceUpdate as EventListener);
    };
  }, []);

  const fetchCustomTokens = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await qnkAPI.getMultiTokenBalance();

      if (response.success && response.data && response.data.tokens) {
        const tokensObj = response.data.tokens;

        // Filter out native tokens (QUG, QUGUSD) and extract custom tokens
        const customTokensList: CustomToken[] = [];

        for (const [symbol, tokenData] of Object.entries(tokensObj)) {
          const upperSymbol = symbol.toUpperCase();

          // Skip native tokens
          if (upperSymbol === 'QUG' || upperSymbol === 'QUGUSD') {
            continue;
          }

          // Add custom token
          const token = tokenData as any;
          customTokensList.push({
            symbol: upperSymbol,
            name: token.name || upperSymbol,
            balance: parseFloat(token.balance || '0'),
            contractAddress: token.contract_address || '',
            decimals: token.decimals || 8,
          });
        }

        setCustomTokens(customTokensList);
        console.log('🎨 [CustomTokens] Loaded', customTokensList.length, 'custom tokens');
      } else {
        console.warn('⚠️ [CustomTokens] Failed to fetch custom tokens:', response.error);
        setError(response.error || 'Failed to load custom tokens');
      }
    } catch (err) {
      console.error('❌ [CustomTokens] Error fetching custom tokens:', err);
      setError('Failed to load custom tokens');
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      className="rounded-2xl p-6 border-2"
      style={{
        background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(168, 85, 247, 0.05))',
        borderColor: 'rgba(139, 92, 246, 0.3)',
      }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{
              background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(168, 85, 247, 0.15))',
              border: '2px solid rgba(139, 92, 246, 0.3)',
            }}
          >
            <Coins className="w-5 h-5 text-purple-300" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">Custom Tokens</h3>
            <p className="text-sm text-gray-400">
              {loading ? 'Loading...' : `${customTokens.length} token${customTokens.length !== 1 ? 's' : ''}`}
            </p>
          </div>
        </div>

        {/* Expand/Collapse Button */}
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setIsExpanded(!isExpanded)}
          className="p-2 rounded-lg transition-colors"
          style={{
            background: 'rgba(139, 92, 246, 0.1)',
            border: '1px solid rgba(139, 92, 246, 0.2)',
          }}
        >
          {isExpanded ? (
            <ChevronUp className="w-5 h-5 text-purple-300" />
          ) : (
            <ChevronDown className="w-5 h-5 text-purple-300" />
          )}
        </motion.button>
      </div>

      {/* Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            {loading && customTokens.length === 0 ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 text-purple-400 animate-spin" />
                <span className="ml-3 text-purple-300">Loading custom tokens...</span>
              </div>
            ) : error ? (
              <div className="flex items-center gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/30">
                <AlertCircle className="w-5 h-5 text-red-400" />
                <div>
                  <p className="text-sm font-medium text-red-300">Failed to load tokens</p>
                  <p className="text-xs text-red-400/80 mt-1">{error}</p>
                </div>
              </div>
            ) : customTokens.length === 0 ? (
              <div className="text-center py-8">
                <Coins className="w-12 h-12 text-purple-400/50 mx-auto mb-3" />
                <p className="text-purple-300/70 text-sm">No custom tokens yet</p>
                <p className="text-purple-400/50 text-xs mt-1">
                  Custom tokens will appear here when you receive them
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {customTokens.map((token) => (
                  <motion.div
                    key={token.contractAddress}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="p-4 rounded-xl border"
                    style={{
                      background: 'rgba(139, 92, 246, 0.05)',
                      borderColor: 'rgba(139, 92, 246, 0.2)',
                    }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <h4 className="font-semibold text-white">{token.symbol}</h4>
                          <span className="text-xs text-gray-500">•</span>
                          <span className="text-xs text-gray-400">{token.name}</span>
                        </div>
                        <p className="text-2xl font-bold text-purple-300">
                          {token.balance.toLocaleString(undefined, {
                            minimumFractionDigits: 2,
                            maximumFractionDigits: 8,
                          })}
                        </p>
                        <p className="text-xs text-gray-500 mt-1 font-mono truncate">
                          {token.contractAddress.substring(0, 10)}...{token.contractAddress.substring(token.contractAddress.length - 8)}
                        </p>
                      </div>

                      {/* Send Button */}
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => onSendToken(token.symbol, token.contractAddress)}
                        className="px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2"
                        style={{
                          background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(37, 99, 235, 0.15))',
                          border: '2px solid rgba(59, 130, 246, 0.3)',
                          color: 'rgb(147, 197, 253)',
                        }}
                        disabled={token.balance === 0}
                      >
                        <Send className="w-4 h-4" />
                        Send
                      </motion.button>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
