import React, { useCallback } from 'react';
import { View, StyleSheet, FlatList, RefreshControl } from 'react-native';
import { Text, IconButton, Chip } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import { COLORS } from '../../src/theme';
import { useWalletStore } from '../../src/stores/walletStore';
import { useNetworkStore } from '../../src/stores/networkStore';
import { useBalance } from '../../src/hooks/useBalance';
import { useSSE } from '../../src/hooks/useSSE';
import { BalanceCard } from '../../src/components/BalanceCard';
import { TokenList } from '../../src/components/TokenList';
import { TransactionItem } from '../../src/components/TransactionItem';
import { BalanceCardSkeleton, TransactionItemSkeleton } from '../../src/components/SkeletonLoader';
import type { Transaction } from '../../src/services/api';

export default function DashboardScreen() {
  const address = useWalletStore((s) => s.address);
  const transactions = useWalletStore((s) => s.transactions);
  const txLoading = useWalletStore((s) => s.txLoading);
  const refreshHistory = useWalletStore((s) => s.refreshHistory);
  const isOnline = useNetworkStore((s) => s.isOnline);
  const height = useNetworkStore((s) => s.height);

  const { qugBalance, tokenBalances, refresh: refreshBalance } = useBalance();
  const sseState = useSSE(address);

  const [refreshing, setRefreshing] = React.useState(false);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await Promise.all([refreshBalance(), refreshHistory(1)]);
    setRefreshing(false);
  }, [refreshBalance, refreshHistory]);

  React.useEffect(() => {
    if (address) {
      refreshHistory(1);
    }
  }, [address, refreshHistory]);

  const handleTxPress = useCallback((tx: Transaction) => {
    router.push({ pathname: '/tx/[id]', params: { id: tx.id } });
  }, []);

  const recentTxs = transactions.slice(0, 10);

  const renderHeader = () => (
    <View>
      {/* Top Bar */}
      <View style={styles.topBar}>
        <View style={styles.topBarLeft}>
          <View style={[styles.statusDot, { backgroundColor: isOnline ? COLORS.green : COLORS.red }]} />
          <Text variant="labelSmall" style={styles.statusText}>
            {isOnline ? `Block ${height.toLocaleString()}` : 'Offline'}
          </Text>
          {sseState.connected && (
            <View style={styles.liveBadge}>
              <View style={styles.livePulse} />
              <Text style={styles.liveText}>LIVE</Text>
            </View>
          )}
        </View>
        <View style={styles.topBarRight}>
          <IconButton
            icon="pickaxe"
            size={22}
            iconColor={COLORS.cyan}
            onPress={() => router.push('/mining')}
          />
          <IconButton
            icon="cog"
            size={22}
            iconColor="rgba(255, 255, 255, 0.6)"
            onPress={() => router.push('/settings')}
          />
        </View>
      </View>

      {/* Balance Card */}
      {qugBalance !== undefined ? (
        <BalanceCard balance={qugBalance} />
      ) : (
        <BalanceCardSkeleton />
      )}

      {/* Token Scroll */}
      <TokenList tokens={[
        { symbol: 'QUG', name: 'Quillon', balance: qugBalance ?? 0, rawBalance: String(qugBalance ?? 0), decimals: 24, priceUsd: 0, valueUsd: 0 },
        ...tokenBalances,
      ]} />

      {/* Recent Transactions Header */}
      <View style={styles.sectionHeader}>
        <Text variant="titleSmall" style={styles.sectionTitle}>
          Recent Activity
        </Text>
        {transactions.length > 0 && (
          <Text
            variant="labelSmall"
            style={styles.viewAll}
            onPress={() => router.push('/(tabs)/history')}
          >
            View All
          </Text>
        )}
      </View>
    </View>
  );

  const renderEmpty = () => (
    <View style={styles.emptyState}>
      <MaterialCommunityIcons name="inbox-outline" size={48} color="rgba(255, 255, 255, 0.15)" />
      <Text variant="bodyMedium" style={styles.emptyText}>
        No transactions yet
      </Text>
      <Text variant="bodySmall" style={styles.emptySubtext}>
        Send or receive QUG to see activity here
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

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <FlatList
        data={recentTxs}
        renderItem={renderTransaction}
        keyExtractor={(item) => item.id}
        ListHeaderComponent={renderHeader}
        ListEmptyComponent={txLoading ? (
          <View>
            <TransactionItemSkeleton />
            <TransactionItemSkeleton />
            <TransactionItemSkeleton />
          </View>
        ) : renderEmpty}
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
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.darkBg,
  },
  listContent: {
    paddingBottom: 24,
  },
  topBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 4,
  },
  topBarLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  topBarRight: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  statusText: {
    color: 'rgba(255, 255, 255, 0.6)',
  },
  liveBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
    backgroundColor: 'rgba(76, 175, 80, 0.15)',
    borderWidth: 1,
    borderColor: 'rgba(76, 175, 80, 0.25)',
  },
  livePulse: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: COLORS.green,
  },
  liveText: {
    color: COLORS.green,
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    marginTop: 24,
    marginBottom: 8,
  },
  sectionTitle: {
    color: 'rgba(255, 255, 255, 0.6)',
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  viewAll: {
    color: COLORS.cyan,
    fontWeight: '600',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 40,
    gap: 8,
  },
  emptyText: {
    color: 'rgba(255, 255, 255, 0.5)',
  },
  emptySubtext: {
    color: 'rgba(255, 255, 255, 0.3)',
  },
});
