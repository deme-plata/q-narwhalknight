import React, { useEffect, useCallback, useState } from 'react';
import { View, StyleSheet, FlatList, RefreshControl } from 'react-native';
import { Text, Chip, Searchbar } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import { COLORS } from '../../src/theme';
import { useWalletStore } from '../../src/stores/walletStore';
import { TransactionItem } from '../../src/components/TransactionItem';
import { TransactionItemSkeleton } from '../../src/components/SkeletonLoader';
import type { Transaction } from '../../src/services/api';

type TxFilter = 'all' | 'sent' | 'received' | 'pending';

export default function HistoryScreen() {
  const address = useWalletStore((s) => s.address);
  const transactions = useWalletStore((s) => s.transactions);
  const txLoading = useWalletStore((s) => s.txLoading);
  const txTotal = useWalletStore((s) => s.txTotal);
  const txPage = useWalletStore((s) => s.txPage);
  const refreshHistory = useWalletStore((s) => s.refreshHistory);

  const [filter, setFilter] = useState<TxFilter>('all');
  const [search, setSearch] = useState('');
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    if (address) refreshHistory(1);
  }, [address, refreshHistory]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await refreshHistory(1);
    setRefreshing(false);
  }, [refreshHistory]);

  const onEndReached = useCallback(() => {
    if (!txLoading && transactions.length < txTotal) {
      refreshHistory(txPage + 1);
    }
  }, [txLoading, transactions.length, txTotal, txPage, refreshHistory]);

  const filteredTransactions = transactions.filter((tx) => {
    if (filter === 'sent' && tx.from.toLowerCase() !== address?.toLowerCase()) return false;
    if (filter === 'received' && tx.to.toLowerCase() !== address?.toLowerCase()) return false;
    if (filter === 'pending' && tx.status !== 'pending') return false;

    if (search) {
      const s = search.toLowerCase();
      return (
        tx.id.toLowerCase().includes(s) ||
        tx.from.toLowerCase().includes(s) ||
        tx.to.toLowerCase().includes(s)
      );
    }

    return true;
  });

  const handleTxPress = useCallback((tx: Transaction) => {
    router.push({ pathname: '/tx/[id]', params: { id: tx.id } });
  }, []);

  const renderHeader = () => (
    <View style={styles.headerContainer}>
      <Searchbar
        placeholder="Search by hash or address..."
        onChangeText={setSearch}
        value={search}
        style={styles.searchBar}
        inputStyle={styles.searchInput}
        iconColor="rgba(255, 255, 255, 0.5)"
        placeholderTextColor="rgba(255, 255, 255, 0.3)"
      />

      <View style={styles.filterRow}>
        {(['all', 'sent', 'received', 'pending'] as TxFilter[]).map((f) => (
          <Chip
            key={f}
            selected={filter === f}
            onPress={() => setFilter(f)}
            style={[styles.filterChip, filter === f && styles.filterChipActive]}
            textStyle={[styles.filterChipText, filter === f && styles.filterChipTextActive]}
            compact
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </Chip>
        ))}
      </View>
    </View>
  );

  const renderEmpty = () => (
    <View style={styles.emptyState}>
      <MaterialCommunityIcons name="receipt" size={48} color="rgba(255, 255, 255, 0.15)" />
      <Text variant="bodyMedium" style={styles.emptyText}>
        {filter !== 'all' || search ? 'No matching transactions' : 'No transaction history'}
      </Text>
    </View>
  );

  const renderTransaction = ({ item }: { item: Transaction }) => (
    <TransactionItem
      transaction={item}
      walletAddress={address ?? ''}
      onPress={handleTxPress}
    />
  );

  const renderFooter = () => {
    if (!txLoading) return null;
    return (
      <View style={styles.footer}>
        <TransactionItemSkeleton />
        <TransactionItemSkeleton />
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <Text variant="headlineSmall" style={styles.title}>
        History
      </Text>

      <FlatList
        data={filteredTransactions}
        renderItem={renderTransaction}
        keyExtractor={(item) => item.id}
        ListHeaderComponent={renderHeader}
        ListEmptyComponent={txLoading && transactions.length === 0 ? (
          <View>
            <TransactionItemSkeleton />
            <TransactionItemSkeleton />
            <TransactionItemSkeleton />
            <TransactionItemSkeleton />
            <TransactionItemSkeleton />
          </View>
        ) : renderEmpty}
        ListFooterComponent={renderFooter}
        onEndReached={onEndReached}
        onEndReachedThreshold={0.3}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={COLORS.cyan}
            colors={[COLORS.cyan]}
          />
        }
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.listContent}
        ItemSeparatorComponent={() => <View style={styles.separator} />}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.darkBg,
  },
  title: {
    color: '#FFFFFF',
    fontWeight: '700',
    paddingHorizontal: 24,
    paddingTop: 8,
    paddingBottom: 4,
  },
  listContent: {
    paddingBottom: 24,
  },
  headerContainer: {
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  searchBar: {
    backgroundColor: COLORS.surfaceBg,
    borderRadius: 12,
    elevation: 0,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: 'rgba(0, 188, 212, 0.06)',
  },
  searchInput: {
    color: '#FFFFFF',
    fontSize: 14,
  },
  filterRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 8,
  },
  filterChip: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.12)',
  },
  filterChipActive: {
    backgroundColor: 'rgba(0, 188, 212, 0.15)',
    borderColor: COLORS.cyan,
  },
  filterChipText: {
    color: 'rgba(255, 255, 255, 0.5)',
    fontSize: 12,
  },
  filterChipTextActive: {
    color: COLORS.cyan,
  },
  separator: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.04)',
    marginHorizontal: 16,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
    gap: 12,
  },
  emptyText: {
    color: 'rgba(255, 255, 255, 0.4)',
  },
  footer: {
    paddingVertical: 8,
  },
});
