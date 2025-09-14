import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Copy, Check, ExternalLink, Hash, User, Blocks, Shield } from 'lucide-react';
// import { qnkAPI } from '../services/api'; // For future real API integration

interface SearchResult {
  type: 'transaction' | 'block' | 'address' | 'node';
  id: string;
  title: string;
  subtitle?: string;
  hash?: string;
}

interface TopBarProps {
  currentBalance: number;
  nodeId: string;
  blockHeight: number;
  peers: number;
  isOnline: boolean;
  qci: number; // Quantum Coherence Index
}

export default function TopBar({ currentBalance, nodeId, blockHeight, peers, isOnline, qci }: TopBarProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Mock search function - replace with real API calls
  const performSearch = async (query: string) => {
    if (query.length < 3) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    
    // Simulate API search delay
    await new Promise(resolve => setTimeout(resolve, 300));
    
    const results: SearchResult[] = [];
    
    // Check if it looks like a hash (64 hex characters)
    if (query.match(/^[a-fA-F0-9]{8,64}$/)) {
      if (query.length === 64) {
        results.push({
          type: 'transaction',
          id: query,
          title: 'Transaction',
          subtitle: `${query.substring(0, 16)}...${query.substring(48)}`,
          hash: query
        });
        results.push({
          type: 'block',
          id: query,
          title: 'Block Hash',
          subtitle: `${query.substring(0, 16)}...${query.substring(48)}`,
          hash: query
        });
      } else {
        results.push({
          type: 'address',
          id: query,
          title: 'Address',
          subtitle: `${query.substring(0, 8)}...${query.substring(-8)}`,
          hash: query
        });
      }
    }
    
    // Check if it looks like a block number
    if (query.match(/^\\d+$/)) {
      const blockNum = parseInt(query);
      results.push({
        type: 'block',
        id: blockNum.toString(),
        title: `Block #${blockNum}`,
        subtitle: blockNum <= blockHeight ? 'Confirmed' : 'Future block',
      });
    }
    
    // Check if it matches node ID pattern
    if (query.toLowerCase().includes(nodeId.substring(0, 8).toLowerCase())) {
      results.push({
        type: 'node',
        id: nodeId,
        title: 'Current Node',
        subtitle: `${nodeId.substring(0, 16)}...`,
        hash: nodeId
      });
    }

    // Add some mock recent transactions/blocks
    if (query.toLowerCase().includes('recent') || query.toLowerCase().includes('latest')) {
      for (let i = 0; i < 3; i++) {
        results.push({
          type: 'block',
          id: (blockHeight - i).toString(),
          title: `Block #${blockHeight - i}`,
          subtitle: `${i === 0 ? 'Latest' : `${i} blocks ago`}`,
        });
      }
    }

    setSearchResults(results);
    setIsSearching(false);
  };

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (searchQuery.trim()) {
        performSearch(searchQuery);
      } else {
        setSearchResults([]);
      }
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [searchQuery]);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const getQCIColor = (qci: number) => {
    if (qci >= 0.9) return 'from-quantum-green to-quantum-cyan';
    if (qci >= 0.8) return 'from-quantum-yellow to-quantum-green';
    return 'from-quantum-pink to-quantum-yellow';
  };

  const getQCIStatus = (qci: number) => {
    if (qci >= 0.9) return 'Sublime';
    if (qci >= 0.8) return 'Coherent';
    return 'Stabilizing';
  };

  const getResultIcon = (type: SearchResult['type']) => {
    switch (type) {
      case 'transaction': return <Hash className="w-4 h-4" />;
      case 'block': return <Blocks className="w-4 h-4" />;
      case 'address': return <User className="w-4 h-4" />;
      case 'node': return <User className="w-4 h-4" />;
    }
  };

  const getResultColor = (type: SearchResult['type']) => {
    switch (type) {
      case 'transaction': return 'text-quantum-green';
      case 'block': return 'text-quantum-cyan';
      case 'address': return 'text-quantum-purple';
      case 'node': return 'text-quantum-pink';
    }
  };

  return (
    <div className="bg-quantum-indigo/30 backdrop-blur-xl border-b border-quantum-purple/20 px-6 py-4 relative z-50">
      <div className="flex items-center justify-between">
        {/* Left: Search */}
        <div className="flex-1 max-w-md relative">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setShowResults(true);
              }}
              onBlur={() => setTimeout(() => setShowResults(false), 200)}
              onFocus={() => setShowResults(true)}
              placeholder="Search transactions, blocks, addresses..."
              className="w-full pl-10 pr-4 py-2 bg-quantum-dark/50 border border-quantum-purple/30 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-quantum-cyan transition-colors"
            />
            {isSearching && (
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                className="absolute right-3 top-1/2 transform -translate-y-1/2"
              >
                <Search className="w-4 h-4 text-quantum-cyan" />
              </motion.div>
            )}
          </div>

          {/* Search Results Dropdown */}
          <AnimatePresence>
            {showResults && searchResults.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute top-full mt-2 w-full bg-quantum-dark/95 backdrop-blur-xl border border-quantum-purple/30 rounded-lg shadow-2xl max-h-96 overflow-y-auto z-50"
              >
                {searchResults.map((result, index) => (
                  <motion.div
                    key={`${result.type}-${result.id}-${index}`}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="flex items-center justify-between p-4 hover:bg-quantum-purple/10 border-b border-quantum-purple/10 last:border-b-0 cursor-pointer group"
                    onClick={() => {
                      // Handle navigation to result
                      console.log('Navigate to:', result);
                      setShowResults(false);
                      setSearchQuery('');
                    }}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg bg-quantum-purple/20 ${getResultColor(result.type)}`}>
                        {getResultIcon(result.type)}
                      </div>
                      <div>
                        <div className="text-white font-medium">{result.title}</div>
                        {result.subtitle && (
                          <div className="text-gray-400 text-sm">{result.subtitle}</div>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      {result.hash && (
                        <motion.button
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.9 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            copyToClipboard(result.hash!, `${result.type}-${result.id}`);
                          }}
                          className="p-1 text-gray-400 hover:text-quantum-cyan transition-colors"
                        >
                          {copiedId === `${result.type}-${result.id}` ? (
                            <Check className="w-4 h-4 text-quantum-green" />
                          ) : (
                            <Copy className="w-4 h-4" />
                          )}
                        </motion.button>
                      )}
                      <ExternalLink className="w-4 h-4 text-gray-400" />
                    </div>
                  </motion.div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Center: Network Status */}
        <div className="flex items-center gap-6">
          <div className="text-center">
            <div className="text-white font-bold text-lg">
              {currentBalance.toLocaleString()} QNK
            </div>
            <div className="text-gray-400 text-sm">Total Balance</div>
          </div>
          
          <div className="h-8 w-px bg-quantum-purple/30" />
          
          <div className="flex items-center gap-3">
            <div className={`w-2 h-2 rounded-full ${isOnline ? 'bg-quantum-green' : 'bg-red-500'}`} />
            <div className="text-white text-sm">
              Block #{blockHeight} • {peers} peers
            </div>
          </div>
        </div>

        {/* Right: Coherence Index */}
        <div className="flex items-center gap-3">
          <Shield className="w-4 h-4 text-quantum-cyan" />
          <div className="text-right">
            <div className="flex items-center gap-2">
              <span className="text-white text-sm font-bold">
                {(qci * 100).toFixed(0)}%
              </span>
              <span className={`text-xs font-medium bg-gradient-to-r ${getQCIColor(qci)} bg-clip-text text-transparent`}>
                {getQCIStatus(qci)}
              </span>
            </div>
            <div className="text-gray-400 text-xs">Quantum Coherence</div>
          </div>
          <div className="relative w-12 h-2 bg-quantum-dark/50 rounded-full overflow-hidden">
            <motion.div
              className={`h-full bg-gradient-to-r ${getQCIColor(qci)}`}
              initial={{ width: 0 }}
              animate={{ width: `${qci * 100}%` }}
              transition={{ duration: 1 }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}