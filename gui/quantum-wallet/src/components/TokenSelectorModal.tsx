import React, { useState, useMemo } from 'react';
import TokenIcon from './TokenIcon';
import './TokenSelectorModal.css';

interface Token {
  id: string;
  symbol: string;
  name: string;
  balance: number;
  price: number;
  change1h: number;
  change24h: number;
  change7d: number;
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
  nitroPoints?: number;
}

interface TokenSelectorModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectToken: (token: Token) => void;
  tokens: Token[];
  boostedTokens: Map<string, number>;
  currentToken?: Token;
}

type SortOption = 'name' | 'balance' | 'price' | 'change' | 'nitro';
type FilterOption = 'all' | 'boosted' | 'favorites';

const TokenSelectorModal: React.FC<TokenSelectorModalProps> = ({
  isOpen,
  onClose,
  onSelectToken,
  tokens,
  boostedTokens,
  currentToken,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<SortOption>('balance');
  const [sortAsc, setSortAsc] = useState(false);
  const [filter, setFilter] = useState<FilterOption>('all');
  const [favorites, setFavorites] = useState<Set<string>>(new Set(['native-qug']));

  // Merge tokens with boost data
  const tokensWithBoosts = useMemo(() => {
    return tokens.map(token => ({
      ...token,
      nitroPoints: boostedTokens.get(token.id) || 0,
    }));
  }, [tokens, boostedTokens]);

  // Filter tokens
  const filteredTokens = useMemo(() => {
    let result = tokensWithBoosts;

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        token =>
          token.name.toLowerCase().includes(query) ||
          token.symbol.toLowerCase().includes(query) ||
          token.id.toLowerCase().includes(query)
      );
    }

    // Category filter
    if (filter === 'boosted') {
      result = result.filter(token => (token.nitroPoints || 0) > 0);
    } else if (filter === 'favorites') {
      result = result.filter(token => favorites.has(token.id));
    }

    return result;
  }, [tokensWithBoosts, searchQuery, filter, favorites]);

  // Sort tokens
  const sortedTokens = useMemo(() => {
    const result = [...filteredTokens];

    result.sort((a, b) => {
      let compareValue = 0;

      switch (sortBy) {
        case 'name':
          compareValue = a.name.localeCompare(b.name);
          break;
        case 'balance':
          compareValue = (a.balance || 0) - (b.balance || 0);
          break;
        case 'price':
          compareValue = (a.price || 0) - (b.price || 0);
          break;
        case 'change':
          compareValue = (a.change24h || 0) - (b.change24h || 0);
          break;
        case 'nitro':
          compareValue = (a.nitroPoints || 0) - (b.nitroPoints || 0);
          break;
      }

      return sortAsc ? compareValue : -compareValue;
    });

    return result;
  }, [filteredTokens, sortBy, sortAsc]);

  const handleToggleFavorite = (tokenId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setFavorites(prev => {
      const newFavorites = new Set(prev);
      if (newFavorites.has(tokenId)) {
        newFavorites.delete(tokenId);
      } else {
        newFavorites.add(tokenId);
      }
      return newFavorites;
    });
  };

  const handleSelectToken = (token: Token) => {
    onSelectToken(token);
    onClose();
  };

  const handleSortToggle = (option: SortOption) => {
    if (sortBy === option) {
      setSortAsc(!sortAsc);
    } else {
      setSortBy(option);
      setSortAsc(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="token-selector-overlay" onClick={onClose}>
      <div className="token-selector-modal" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="modal-header">
          <h2 className="modal-title">
            <span className="title-icon">🪙</span>
            Select Token
          </h2>
          <button className="close-button" onClick={onClose}>
            ✕
          </button>
        </div>

        {/* Search Bar */}
        <div className="search-container">
          <div className="search-input-wrapper">
            <span className="search-icon">🔍</span>
            <input
              type="text"
              className="search-input"
              placeholder="Search by name, symbol, or address..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              autoFocus
            />
            {searchQuery && (
              <button
                className="clear-search"
                onClick={() => setSearchQuery('')}
              >
                ✕
              </button>
            )}
          </div>
        </div>

        {/* Filter Tabs */}
        <div className="filter-tabs">
          <button
            className={`filter-tab ${filter === 'all' ? 'active' : ''}`}
            onClick={() => setFilter('all')}
          >
            <span className="tab-icon">📋</span>
            All Tokens
            <span className="tab-count">{tokensWithBoosts.length}</span>
          </button>
          <button
            className={`filter-tab ${filter === 'boosted' ? 'active' : ''}`}
            onClick={() => setFilter('boosted')}
          >
            <span className="tab-icon">⚡</span>
            Nitro Boosted
            <span className="tab-count">
              {tokensWithBoosts.filter(t => (t.nitroPoints || 0) > 0).length}
            </span>
          </button>
          <button
            className={`filter-tab ${filter === 'favorites' ? 'active' : ''}`}
            onClick={() => setFilter('favorites')}
          >
            <span className="tab-icon">⭐</span>
            Favorites
            <span className="tab-count">{favorites.size}</span>
          </button>
        </div>

        {/* Sort Options */}
        <div className="sort-options">
          <span className="sort-label">Sort by:</span>
          {[
            { value: 'name' as SortOption, label: 'Name', icon: '🔤' },
            { value: 'balance' as SortOption, label: 'Balance', icon: '💰' },
            { value: 'price' as SortOption, label: 'Price', icon: '💵' },
            { value: 'change' as SortOption, label: '24h', icon: '📈' },
            { value: 'nitro' as SortOption, label: 'Nitro', icon: '⚡' },
          ].map(option => (
            <button
              key={option.value}
              className={`sort-button ${sortBy === option.value ? 'active' : ''}`}
              onClick={() => handleSortToggle(option.value)}
            >
              <span className="sort-icon">{option.icon}</span>
              {option.label}
              {sortBy === option.value && (
                <span className="sort-arrow">{sortAsc ? '↑' : '↓'}</span>
              )}
            </button>
          ))}
        </div>

        {/* Token List */}
        <div className="token-list">
          {sortedTokens.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">🔍</div>
              <div className="empty-text">
                {searchQuery
                  ? `No tokens found for "${searchQuery}"`
                  : 'No tokens available'}
              </div>
            </div>
          ) : (
            sortedTokens.map(token => {
              const isSelected = currentToken?.id === token.id;
              const isFavorite = favorites.has(token.id);
              const hasNitro = (token.nitroPoints || 0) > 0;

              return (
                <div
                  key={token.id}
                  className={`token-item ${isSelected ? 'selected' : ''} ${
                    hasNitro ? 'boosted' : ''
                  }`}
                  onClick={() => handleSelectToken(token)}
                >
                  <div className="token-main">
                    <div className="token-logo">
                      <div className="token-icon">
                        <TokenIcon
                          symbol={token.symbol}
                          icon={token.icon}
                          logoUrl={(token as any).logoUrl}
                          size={40}
                        />
                      </div>
                      {hasNitro && (
                        <div className="nitro-badge" title="Nitro Boosted">
                          ⚡
                        </div>
                      )}
                    </div>

                    <div className="token-info">
                      <div className="token-name-row">
                        <span className="token-name">{token.name}</span>
                        <button
                          className={`favorite-button ${isFavorite ? 'active' : ''}`}
                          onClick={e => handleToggleFavorite(token.id, e)}
                          title={isFavorite ? 'Remove from favorites' : 'Add to favorites'}
                        >
                          {isFavorite ? '⭐' : '☆'}
                        </button>
                      </div>
                      <div className="token-symbol">{token.symbol}</div>
                      {hasNitro && (
                        <div className="token-nitro">
                          <span className="nitro-icon">⚡</span>
                          <span className="nitro-text">
                            {token.nitroPoints} Nitro Points
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="token-stats">
                    <div className="token-balance">
                      <div className="stat-label">Balance</div>
                      <div className="stat-value">
                        {token.balance.toLocaleString(undefined, {
                          minimumFractionDigits: 2,
                          maximumFractionDigits: 6,
                        })}
                      </div>
                    </div>

                    <div className="token-price">
                      <div className="stat-label">Price</div>
                      <div className="stat-value">
                        ${token.price.toFixed(4)}
                      </div>
                    </div>

                    <div
                      className={`token-change ${
                        token.change24h >= 0 ? 'positive' : 'negative'
                      }`}
                    >
                      <div className="stat-label">24h</div>
                      <div className="stat-value">
                        {token.change24h >= 0 ? '+' : ''}
                        {token.change24h.toFixed(2)}%
                      </div>
                    </div>
                  </div>

                  {isSelected && (
                    <div className="selected-indicator">
                      <span className="checkmark">✓</span>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div className="modal-footer">
          <div className="footer-info">
            <span className="info-icon">ℹ️</span>
            <span className="info-text">
              Showing {sortedTokens.length} of {tokensWithBoosts.length} tokens
              {filter === 'boosted' && ' with Nitro boost'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TokenSelectorModal;
