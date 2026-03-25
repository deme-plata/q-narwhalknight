import React from 'react';
import { View, ScrollView, StyleSheet, TouchableOpacity } from 'react-native';
import { Text, Surface } from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import { COLORS, tokenColor } from '../theme';
import { formatNumber } from '../utils/formatBalance';
import { TokenBalance } from '../stores/walletStore';

interface TokenListProps {
  tokens: TokenBalance[];
  onTokenPress?: (token: TokenBalance) => void;
}

function TokenIcon({ symbol }: { symbol: string }) {
  const { bg, fg } = tokenColor(symbol);
  return (
    <View style={[styles.tokenIcon, { backgroundColor: bg }]}>
      <Text style={[styles.tokenIconText, { color: fg }]}>
        {symbol.slice(0, 2).toUpperCase()}
      </Text>
    </View>
  );
}

export function TokenList({ tokens, onTokenPress }: TokenListProps) {
  if (tokens.length === 0) {
    return null;
  }

  return (
    <View style={styles.container}>
      <Text variant="titleSmall" style={styles.sectionTitle}>
        Tokens
      </Text>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        {tokens.map((token) => (
          <TouchableOpacity
            key={token.symbol}
            onPress={() => onTokenPress?.(token)}
            activeOpacity={0.7}
          >
            <View style={styles.tokenCardOuter}>
              <LinearGradient
                colors={['rgba(0, 188, 212, 0.06)', 'rgba(26, 26, 38, 0.8)']}
                start={{ x: 0, y: 0 }}
                end={{ x: 0, y: 1 }}
                style={styles.tokenCard}
              >
                <TokenIcon symbol={token.symbol} />
                <Text variant="labelMedium" style={styles.tokenSymbol} numberOfLines={1}>
                  {token.symbol}
                </Text>
                <Text variant="bodySmall" style={styles.tokenBalance} numberOfLines={1}>
                  {formatNumber(token.balance, { maxDecimals: 4, compact: true })}
                </Text>
                {token.valueUsd > 0 && (
                  <Text variant="labelSmall" style={styles.tokenValue}>
                    ${formatNumber(token.valueUsd, { maxDecimals: 2 })}
                  </Text>
                )}
              </LinearGradient>
            </View>
          </TouchableOpacity>
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginTop: 20,
  },
  sectionTitle: {
    color: 'rgba(255, 255, 255, 0.6)',
    marginLeft: 16,
    marginBottom: 12,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  scrollContent: {
    paddingHorizontal: 16,
    gap: 12,
  },
  tokenCardOuter: {
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(0, 188, 212, 0.08)',
    overflow: 'hidden',
  },
  tokenCard: {
    width: 100,
    padding: 12,
    alignItems: 'center',
  },
  tokenIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  tokenIconText: {
    fontWeight: '700',
    fontSize: 14,
  },
  tokenSymbol: {
    color: '#FFFFFF',
    fontWeight: '600',
    marginBottom: 2,
  },
  tokenBalance: {
    color: 'rgba(255, 255, 255, 0.65)',
    fontSize: 12,
  },
  tokenValue: {
    color: 'rgba(255, 255, 255, 0.4)',
    marginTop: 2,
  },
  emptyContainer: {
    padding: 20,
    alignItems: 'center',
  },
  emptyText: {
    color: 'rgba(255, 255, 255, 0.4)',
  },
});
