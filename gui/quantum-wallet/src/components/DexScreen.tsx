import { useState, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowDownUp, Search, TrendingUp, TrendingDown, Settings, Info } from 'lucide-react';
import TokenDetailsModal from './TokenDetailsModal';
import { qnkAPI } from '../services/api';

interface Token {
  id: string;
  symbol: string;
  name: string;
  balance: number;
  price: number;
  change24h: number;
  volume24h: number;
  liquidity: number;
  icon: string;
  marketCap: number;
  totalSupply: number;
  circulatingSupply: number;
  holders: number;
  features: {
    reflection: boolean;
    autoLiquidity: boolean;
    buybackAndBurn: boolean;
    antiWhale: boolean;
    quantumSecured: boolean;
  };
  fees: {
    buy: number;
    sell: number;
    transfer: number;
  };
  description: string;
  website?: string;
  whitepaper?: string;
}

export default function DexScreen() {
  const [swapFrom, setSwapFrom] = useState('QNK');
  const [swapTo, setSwapTo] = useState('ORBUSD');
  const [swapAmount, setSwapAmount] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'symbol' | 'price' | 'change24h' | 'volume24h' | 'liquidity'>('volume24h');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [filterBy, setFilterBy] = useState<'all' | 'gainers' | 'losers'>('all');
  const [selectedToken, setSelectedToken] = useState<Token | null>(null);
  const [tokens, setTokens] = useState<Token[]>([]);
  const [loading, setLoading] = useState(true);

  // Fetch real tokens from API
  useEffect(() => {
    const fetchTokens = async () => {
      try {
        const response = await qnkAPI.getSupportedTokens();
        if (response.success && response.data) {
          // Convert API token data to Token interface with additional metadata
          const enrichedTokens: Token[] = response.data.map(apiToken => ({
            id: apiToken.address,
            symbol: apiToken.symbol,
            name: apiToken.name,
            balance: 0, // Will be fetched per wallet
            price: apiToken.symbol === 'ORBUSD' ? 1.00 : 42.50, // ORBUSD is $1, QNK estimated
            change24h: apiToken.symbol === 'ORBUSD' ? 0.01 : 8.5,
            volume24h: apiToken.symbol === 'QNK' ? 1250000 : 450000,
            liquidity: parseFloat(apiToken.total_supply) || 5000000,
            marketCap: apiToken.symbol === 'QNK' ? 425000000 : 95000000,
            totalSupply: parseFloat(apiToken.total_supply) || 10000000,
            circulatingSupply: parseFloat(apiToken.total_supply) || 10000000,
            holders: apiToken.symbol === 'QNK' ? 15432 : 2500,
            icon: apiToken.symbol === 'QNK' ? '⚛️' : '💵',
            features: {
              reflection: apiToken.symbol === 'QNK',
              autoLiquidity: apiToken.symbol === 'QNK',
              buybackAndBurn: apiToken.symbol === 'QNK',
              antiWhale: apiToken.symbol === 'QNK',
              quantumSecured: true,
            },
            fees: {
              buy: apiToken.symbol === 'QNK' ? 3 : 0,
              sell: apiToken.symbol === 'QNK' ? 5 : 0,
              transfer: apiToken.symbol === 'QNK' ? 1 : 0,
            },
            description: apiToken.symbol === 'QNK'
              ? 'Q-NarwhalKnight is a quantum-enhanced consensus layer utilizing DAG-BFT technology with post-quantum cryptographic security. The token powers the network through staking, governance, and transaction fees.'
              : 'ORBUSD is a quantum-secured stablecoin pegged 1:1 to USD, backed by collateralized assets and maintained through physics-inspired economic algorithms. Features zero-knowledge privacy and quantum-resistant cryptography.',
            website: apiToken.symbol === 'QNK' ? 'https://q-narwhalknight.dev' : 'https://quillon.xyz',
            whitepaper: apiToken.audit_report,
          }));
          setTokens(enrichedTokens);
        }
      } catch (error) {
        console.error('Failed to fetch tokens:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchTokens();
  }, []);

  // Filter tokens based on search and filter
  const filteredTokens = tokens
    .filter(token => {
      const matchesSearch = token.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           token.name.toLowerCase().includes(searchQuery.toLowerCase());

      if (filterBy === 'gainers') return matchesSearch && token.change24h > 0;
      if (filterBy === 'losers') return matchesSearch && token.change24h < 0;
      return matchesSearch;
    })
    .sort((a, b) => {
      const multiplier = sortDirection === 'asc' ? 1 : -1;
      const aVal = a[sortBy] as number;
      const bVal = b[sortBy] as number;
      return (aVal - bVal) * multiplier;
    });

  const handleSort = (field: typeof sortBy) => {
    if (sortBy === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortDirection('desc');
    }
  };

  const formatNumber = (num: number, decimals: number = 2) => {
    if (num >= 1000000000) return `$${(num / 1000000000).toFixed(decimals)}B`;
    if (num >= 1000000) return `$${(num / 1000000).toFixed(decimals)}M`;
    if (num >= 1000) return `$${(num / 1000).toFixed(decimals)}K`;
    return `$${num.toFixed(decimals)}`;
  };

  const swapTokens = () => {
    const temp = swapFrom;
    setSwapFrom(swapTo);
    setSwapTo(temp);
  };

  const handleCloseModal = useCallback(() => {
    setSelectedToken(null);
  }, []);

  return (
    <>
      {/* Token Details Modal */}
      {selectedToken && (
        <TokenDetailsModal
          token={selectedToken}
          onClose={handleCloseModal}
        />
      )}

      <div className="space-y-6">

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-black bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink bg-clip-text text-transparent">
            Quantum DEX
          </h1>
          <p className="text-gray-400 mt-1">Decentralized exchange with quantum security</p>
        </div>
      </motion.div>

      {loading ? (
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="text-center">
            <div className="inline-block w-12 h-12 border-4 border-quantum-cyan/30 border-t-quantum-cyan rounded-full animate-spin mb-4" />
            <p className="text-gray-400">Loading tokens...</p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Swap Interface */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-1"
          >
            <div className="relative group">
              {/* Glow effect */}
              <div className="absolute -inset-0.5 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-2xl blur-lg opacity-30 group-hover:opacity-50 transition-opacity" />

              <div className="relative bg-black/60 backdrop-blur-xl rounded-2xl border border-quantum-cyan/20 p-6 space-y-4">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-bold text-white">Swap Tokens</h2>
                  <button className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors">
                    <Settings className="w-5 h-5 text-gray-400" />
                  </button>
                </div>

              {/* From Token */}
              <div className="space-y-2">
                <label className="text-sm text-gray-400">From</label>
                <div className="relative">
                  <input
                    type="number"
                    value={swapAmount}
                    onChange={(e) => setSwapAmount(e.target.value)}
                    placeholder="0.0"
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-4 text-white text-xl focus:outline-none focus:border-quantum-cyan/50 transition-colors"
                  />
                  <select
                    value={swapFrom}
                    onChange={(e) => setSwapFrom(e.target.value)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 bg-quantum-purple/20 border border-quantum-purple/30 rounded-lg px-3 py-2 text-white font-bold cursor-pointer focus:outline-none"
                  >
                    {tokens.map(token => (
                      <option key={token.id} value={token.symbol}>{token.symbol}</option>
                    ))}
                  </select>
                </div>
                <div className="text-xs text-gray-500">
                  Balance: {tokens.find(t => t.symbol === swapFrom)?.balance.toFixed(4) || '0.0000'}
                </div>
              </div>

              {/* Swap Button */}
              <div className="flex justify-center -my-2">
                <button
                  onClick={swapTokens}
                  className="p-3 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-full hover:scale-110 transition-transform"
                >
                  <ArrowDownUp className="w-5 h-5 text-white" />
                </button>
              </div>

              {/* To Token */}
              <div className="space-y-2">
                <label className="text-sm text-gray-400">To</label>
                <div className="relative">
                  <input
                    type="text"
                    value={swapAmount ? (parseFloat(swapAmount) * 0.95).toFixed(4) : ''}
                    placeholder="0.0"
                    readOnly
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-4 text-white text-xl focus:outline-none"
                  />
                  <select
                    value={swapTo}
                    onChange={(e) => setSwapTo(e.target.value)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 bg-quantum-purple/20 border border-quantum-purple/30 rounded-lg px-3 py-2 text-white font-bold cursor-pointer focus:outline-none"
                  >
                    {tokens.map(token => (
                      <option key={token.id} value={token.symbol}>{token.symbol}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Swap Info */}
              <div className="space-y-2 text-sm p-4 bg-white/5 rounded-xl">
                <div className="flex justify-between text-gray-400">
                  <span>Rate</span>
                  <span className="text-white">1 {swapFrom} ≈ 0.95 {swapTo}</span>
                </div>
                <div className="flex justify-between text-gray-400">
                  <span>Slippage</span>
                  <span className="text-white">0.5%</span>
                </div>
                <div className="flex justify-between text-gray-400">
                  <span>Fee</span>
                  <span className="text-white">0.3%</span>
                </div>
              </div>

              {/* Swap Button */}
              <button className="w-full py-4 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-xl font-bold text-white hover:shadow-lg hover:shadow-quantum-cyan/50 transition-all">
                Swap Tokens
              </button>
            </div>
          </div>
        </motion.div>

        {/* Token Table */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="lg:col-span-2"
        >
          <div className="relative group">
            {/* Glow effect */}
            <div className="absolute -inset-0.5 bg-gradient-to-r from-quantum-purple to-quantum-pink rounded-2xl blur-lg opacity-30 group-hover:opacity-50 transition-opacity" />

            <div className="relative bg-black/60 backdrop-blur-xl rounded-2xl border border-quantum-purple/20 p-6">
              <h2 className="text-xl font-bold text-white mb-6">Available Tokens</h2>

              {/* Search and Filters */}
              <div className="flex flex-col sm:flex-row gap-4 mb-6">
                {/* Search */}
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search tokens..."
                    className="w-full bg-white/5 border border-white/10 rounded-xl pl-10 pr-4 py-3 text-white focus:outline-none focus:border-quantum-cyan/50 transition-colors"
                  />
                </div>

                {/* Filter Buttons */}
                <div className="flex gap-2">
                  {(['all', 'gainers', 'losers'] as const).map((filter) => (
                    <button
                      key={filter}
                      onClick={() => setFilterBy(filter)}
                      className={`px-4 py-3 rounded-xl font-medium transition-all ${
                        filterBy === filter
                          ? 'bg-gradient-to-r from-quantum-cyan to-quantum-purple text-white'
                          : 'bg-white/5 text-gray-400 hover:bg-white/10'
                      }`}
                    >
                      {filter.charAt(0).toUpperCase() + filter.slice(1)}
                    </button>
                  ))}
                </div>
              </div>

              {/* Table */}
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/10">
                      <th className="text-left py-3 px-4 text-gray-400 font-medium text-sm">Token</th>
                      <th
                        onClick={() => handleSort('price')}
                        className="text-right py-3 px-4 text-gray-400 font-medium text-sm cursor-pointer hover:text-white transition-colors"
                      >
                        Price {sortBy === 'price' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th
                        onClick={() => handleSort('change24h')}
                        className="text-right py-3 px-4 text-gray-400 font-medium text-sm cursor-pointer hover:text-white transition-colors"
                      >
                        24h Change {sortBy === 'change24h' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th
                        onClick={() => handleSort('volume24h')}
                        className="text-right py-3 px-4 text-gray-400 font-medium text-sm cursor-pointer hover:text-white transition-colors"
                      >
                        Volume {sortBy === 'volume24h' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th
                        onClick={() => handleSort('liquidity')}
                        className="text-right py-3 px-4 text-gray-400 font-medium text-sm cursor-pointer hover:text-white transition-colors"
                      >
                        Liquidity {sortBy === 'liquidity' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th className="text-right py-3 px-4 text-gray-400 font-medium text-sm">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredTokens.map((token, index) => (
                      <motion.tr
                        key={token.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="border-b border-white/5 hover:bg-white/5 transition-colors cursor-pointer"
                        onClick={() => setSelectedToken(token)}
                      >
                        {/* Token Info */}
                        <td className="py-4 px-4">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 bg-gradient-to-br from-quantum-cyan to-quantum-purple rounded-full flex items-center justify-center text-xl">
                              {token.icon}
                            </div>
                            <div>
                              <div className="font-bold text-white">{token.symbol}</div>
                              <div className="text-sm text-gray-400">{token.name}</div>
                            </div>
                          </div>
                        </td>

                        {/* Price */}
                        <td className="py-4 px-4 text-right text-white font-medium">
                          ${token.price.toLocaleString()}
                        </td>

                        {/* 24h Change */}
                        <td className="py-4 px-4 text-right">
                          <div className={`flex items-center justify-end gap-1 ${
                            token.change24h > 0 ? 'text-quantum-green' : 'text-red-500'
                          }`}>
                            {token.change24h > 0 ? (
                              <TrendingUp className="w-4 h-4" />
                            ) : (
                              <TrendingDown className="w-4 h-4" />
                            )}
                            <span className="font-medium">
                              {token.change24h > 0 ? '+' : ''}{token.change24h.toFixed(2)}%
                            </span>
                          </div>
                        </td>

                        {/* Volume */}
                        <td className="py-4 px-4 text-right text-white font-medium">
                          {formatNumber(token.volume24h)}
                        </td>

                        {/* Liquidity */}
                        <td className="py-4 px-4 text-right text-white font-medium">
                          {formatNumber(token.liquidity)}
                        </td>

                        {/* Actions */}
                        <td className="py-4 px-4 text-right" onClick={(e) => e.stopPropagation()}>
                          <button
                            onClick={() => {
                              setSwapFrom(token.symbol);
                              window.scrollTo({ top: 0, behavior: 'smooth' });
                            }}
                            className="px-4 py-2 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-lg text-white font-medium hover:shadow-lg hover:shadow-quantum-cyan/50 transition-all"
                          >
                            Trade
                          </button>
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>

                {filteredTokens.length === 0 && (
                  <div className="text-center py-12">
                    <Info className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                    <p className="text-gray-400">No tokens found matching your criteria</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      </div>
      )}
      </div>
    </>
  );
}
