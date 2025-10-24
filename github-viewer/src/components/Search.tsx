import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search as SearchIcon, Filter, X, FileText, Folder, Code, ArrowUpDown } from 'lucide-react';
import type { FileTreeNode } from '../types/github';
import { getFileExtension, getLanguageFromExtension, formatFileSize } from '../api/github';

interface SearchProps {
  fileTree: FileTreeNode | null;
  onFileSelect: (path: string) => void;
}

interface SearchResult {
  path: string;
  name: string;
  type: 'file' | 'folder';
  size?: number;
  language?: string;
  matches: number;
}

type SortOption = 'relevance' | 'name' | 'size' | 'type';
type FilterType = 'all' | 'rust' | 'typescript' | 'markdown' | 'json' | 'toml' | 'other';

export function Search({ fileTree, onFileSelect }: SearchProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [filteredResults, setFilteredResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [sortBy, setSortBy] = useState<SortOption>('relevance');
  const [filterType, setFilterType] = useState<FilterType>('all');
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Debounced search
  useEffect(() => {
    const searchTimeout = setTimeout(() => {
      if (query.trim().length >= 2) {
        performSearch(query);
      } else {
        setResults([]);
        setFilteredResults([]);
      }
    }, 300);

    return () => clearTimeout(searchTimeout);
  }, [query, fileTree]);

  // Apply filters and sorting when results, sortBy, or filterType changes
  useEffect(() => {
    let filtered = [...results];

    // Apply type filter
    if (filterType !== 'all') {
      filtered = filtered.filter(result => {
        if (result.type === 'folder') return filterType === 'other';
        const ext = getFileExtension(result.name);
        const lang = getLanguageFromExtension(ext);
        return lang === filterType || (filterType === 'other' && !['rust', 'typescript', 'markdown', 'json', 'toml'].includes(lang));
      });
    }

    // Apply sorting
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'relevance':
          return b.matches - a.matches;
        case 'name':
          return a.name.localeCompare(b.name);
        case 'size':
          return (b.size || 0) - (a.size || 0);
        case 'type':
          if (a.type !== b.type) return a.type === 'folder' ? -1 : 1;
          return a.name.localeCompare(b.name);
        default:
          return 0;
      }
    });

    setFilteredResults(filtered);
  }, [results, sortBy, filterType]);

  // Focus input when opening
  useEffect(() => {
    if (isOpen && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [isOpen]);

  const performSearch = (searchQuery: string) => {
    if (!fileTree) return;

    setIsSearching(true);
    const searchResults: SearchResult[] = [];
    const lowerQuery = searchQuery.toLowerCase();

    const searchNode = (node: FileTreeNode, depth: number = 0) => {
      const lowerName = node.name.toLowerCase();
      const lowerPath = node.path.toLowerCase();

      // Calculate relevance score
      let matches = 0;
      if (lowerName.includes(lowerQuery)) matches += 10;
      if (lowerPath.includes(lowerQuery)) matches += 5;
      if (lowerName.startsWith(lowerQuery)) matches += 20;

      // Boost results that are closer to root
      matches += Math.max(0, 10 - depth);

      if (matches > 0) {
        const ext = getFileExtension(node.name);
        searchResults.push({
          path: node.path,
          name: node.name,
          type: node.type,
          size: node.size,
          language: node.type === 'file' ? getLanguageFromExtension(ext) : undefined,
          matches,
        });
      }

      if (node.children) {
        node.children.forEach(child => searchNode(child, depth + 1));
      }
    };

    searchNode(fileTree);
    setResults(searchResults);
    setIsSearching(false);
  };

  const handleResultClick = (result: SearchResult) => {
    if (result.type === 'file') {
      onFileSelect(result.path);
      setIsOpen(false);
      setQuery('');
    }
  };

  const getFileIcon = (result: SearchResult) => {
    if (result.type === 'folder') return <Folder size={16} className="text-yellow-400" />;

    const ext = getFileExtension(result.name);
    const lang = getLanguageFromExtension(ext);

    const iconColors: Record<string, string> = {
      rust: 'text-orange-400',
      typescript: 'text-blue-400',
      javascript: 'text-yellow-400',
      markdown: 'text-gray-400',
      json: 'text-green-400',
      toml: 'text-purple-400',
    };

    return <FileText size={16} className={iconColors[lang] || 'text-gray-400'} />;
  };

  const highlightMatch = (text: string) => {
    if (!query.trim()) return text;

    const parts = text.split(new RegExp(`(${query})`, 'gi'));
    return (
      <>
        {parts.map((part, i) =>
          part.toLowerCase() === query.toLowerCase() ? (
            <span key={i} className="bg-cyan-500/30 text-cyan-300 font-semibold">
              {part}
            </span>
          ) : (
            <span key={i}>{part}</span>
          )
        )}
      </>
    );
  };

  return (
    <>
      {/* Search Button */}
      <button
        onClick={() => setIsOpen(true)}
        className="flex items-center gap-2 px-4 py-2 bg-cyan-500/10 hover:bg-cyan-500/20
                 text-cyan-400 rounded-lg transition-all duration-200 font-mono text-sm
                 border border-cyan-500/30 hover:border-cyan-500/50"
        title="Search files (Ctrl+K)"
      >
        <SearchIcon size={18} />
        <span className="hidden md:inline">Search</span>
        <kbd className="hidden md:inline px-2 py-0.5 bg-cyan-500/20 rounded text-xs">Ctrl+K</kbd>
      </button>

      {/* Search Modal */}
      <AnimatePresence>
        {isOpen && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-start justify-center z-50 p-4 pt-20">
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="bg-[#050714] border-2 border-cyan-500/50 rounded-xl max-w-3xl w-full max-h-[70vh] overflow-hidden shadow-2xl flex flex-col"
            >
              {/* Search Input */}
              <div className="p-4 border-b border-cyan-500/30">
                <div className="relative">
                  <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-cyan-400" />
                  <input
                    ref={searchInputRef}
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Search files and folders..."
                    className="w-full pl-11 pr-11 py-3 bg-quantum-indigo/30 border border-cyan-500/30 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 transition-all font-mono"
                  />
                  {query && (
                    <button
                      onClick={() => setQuery('')}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-cyan-400 transition-colors"
                    >
                      <X size={20} />
                    </button>
                  )}
                  <button
                    onClick={() => setIsOpen(false)}
                    className="absolute -right-12 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-cyan-400 transition-colors"
                    title="Close (Esc)"
                  >
                    <X size={24} />
                  </button>
                </div>

                {/* Filters and Sorting */}
                {results.length > 0 && (
                  <div className="flex items-center gap-4 mt-3 flex-wrap">
                    {/* Type Filter */}
                    <div className="flex items-center gap-2">
                      <Filter size={14} className="text-gray-400" />
                      <select
                        value={filterType}
                        onChange={(e) => setFilterType(e.target.value as FilterType)}
                        className="bg-quantum-indigo/30 border border-cyan-500/30 rounded px-2 py-1 text-xs text-cyan-400 focus:outline-none focus:border-cyan-400 font-mono"
                      >
                        <option value="all">All Files</option>
                        <option value="rust">Rust (.rs)</option>
                        <option value="typescript">TypeScript (.ts)</option>
                        <option value="markdown">Markdown (.md)</option>
                        <option value="json">JSON (.json)</option>
                        <option value="toml">TOML (.toml)</option>
                        <option value="other">Other</option>
                      </select>
                    </div>

                    {/* Sort */}
                    <div className="flex items-center gap-2">
                      <ArrowUpDown size={14} className="text-gray-400" />
                      <select
                        value={sortBy}
                        onChange={(e) => setSortBy(e.target.value as SortOption)}
                        className="bg-quantum-indigo/30 border border-cyan-500/30 rounded px-2 py-1 text-xs text-cyan-400 focus:outline-none focus:border-cyan-400 font-mono"
                      >
                        <option value="relevance">Relevance</option>
                        <option value="name">Name</option>
                        <option value="size">Size</option>
                        <option value="type">Type</option>
                      </select>
                    </div>

                    {/* Results Count */}
                    <div className="text-xs text-gray-400 font-mono ml-auto">
                      {filteredResults.length} result{filteredResults.length !== 1 ? 's' : ''}
                      {filteredResults.length !== results.length && ` (${results.length} total)`}
                    </div>
                  </div>
                )}
              </div>

              {/* Results */}
              <div className="flex-1 overflow-y-auto p-2">
                {isSearching ? (
                  <div className="flex items-center justify-center py-12">
                    <div className="text-cyan-400 font-mono text-sm animate-pulse">Searching...</div>
                  </div>
                ) : filteredResults.length > 0 ? (
                  <div className="space-y-1">
                    {filteredResults.map((result) => (
                      <motion.div
                        key={result.path}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="p-3 hover:bg-cyan-500/10 rounded-lg cursor-pointer transition-colors border border-transparent hover:border-cyan-500/30"
                        onClick={() => handleResultClick(result)}
                      >
                        <div className="flex items-start gap-3">
                          <div className="mt-0.5">{getFileIcon(result)}</div>
                          <div className="flex-1 min-w-0">
                            <div className="text-cyan-300 font-mono text-sm font-medium truncate">
                              {highlightMatch(result.name)}
                            </div>
                            <div className="text-gray-400 font-mono text-xs truncate mt-0.5">
                              {highlightMatch(result.path)}
                            </div>
                            <div className="flex items-center gap-3 mt-1">
                              {result.language && (
                                <span className="text-xs font-mono text-gray-500 uppercase">
                                  {result.language}
                                </span>
                              )}
                              {result.size && (
                                <span className="text-xs font-mono text-gray-500">
                                  {formatFileSize(result.size)}
                                </span>
                              )}
                              {sortBy === 'relevance' && (
                                <span className="text-xs font-mono text-cyan-500/50">
                                  Relevance: {result.matches}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                ) : query.trim().length >= 2 ? (
                  <div className="flex flex-col items-center justify-center py-12">
                    <SearchIcon size={48} className="text-gray-600 mb-4" />
                    <p className="text-gray-400 font-mono text-sm">No results found for "{query}"</p>
                    <p className="text-gray-500 font-mono text-xs mt-2">Try different keywords or filters</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12">
                    <Code size={48} className="text-cyan-500/30 mb-4" />
                    <p className="text-gray-400 font-mono text-sm">Start typing to search files</p>
                    <p className="text-gray-500 font-mono text-xs mt-2">Minimum 2 characters</p>
                  </div>
                )}
              </div>

              {/* Footer */}
              <div className="p-3 border-t border-cyan-500/30 bg-quantum-indigo/20">
                <div className="flex items-center justify-between text-xs font-mono text-gray-500">
                  <div className="flex items-center gap-4">
                    <span>↑↓ Navigate</span>
                    <span>Enter Select</span>
                    <span>Esc Close</span>
                  </div>
                  <div>
                    Powered by Fuse.js
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </>
  );
}
