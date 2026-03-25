import React, { useEffect, useState, useCallback } from 'react';
import { View, StyleSheet, ScrollView, RefreshControl } from 'react-native';
import { Text, Surface, Chip } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { COLORS } from '../src/theme';
import { useWalletStore } from '../src/stores/walletStore';
import { useNetworkStore } from '../src/stores/networkStore';
import * as api from '../src/services/api';
import { formatNumber, formatBalance } from '../src/utils/formatBalance';
import { SkeletonLoader } from '../src/components/SkeletonLoader';

interface MiningData {
  hashrate: number;
  blocks_found: number;
  reward_total: string;
  difficulty: number;
  network_hashrate: number;
  last_block_time: number;
}

function StatCard({
  icon,
  label,
  value,
  color = COLORS.cyan,
}: {
  icon: string;
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <Surface style={styles.statCard} elevation={1}>
      <MaterialCommunityIcons name={icon as any} size={24} color={color} />
      <Text variant="labelSmall" style={styles.statLabel}>
        {label}
      </Text>
      <Text variant="titleMedium" style={[styles.statValue, { color }]}>
        {value}
      </Text>
    </Surface>
  );
}

export default function MiningScreen() {
  const address = useWalletStore((s) => s.address);
  const height = useNetworkStore((s) => s.height);
  const [stats, setStats] = useState<MiningData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchStats = useCallback(async () => {
    if (!address) return;
    try {
      const data = await api.getMiningStats(address);
      setStats(data);
    } catch (err) {
      console.error('[Mining] Stats fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, [address]);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 15_000);
    return () => clearInterval(interval);
  }, [fetchStats]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await fetchStats();
    setRefreshing(false);
  }, [fetchStats]);

  const lastBlockAgo = stats?.last_block_time
    ? Math.floor((Date.now() / 1000 - stats.last_block_time) / 60)
    : null;

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.scrollContent}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={onRefresh}
          tintColor={COLORS.cyan}
          colors={[COLORS.cyan]}
        />
      }
    >
      {/* Header */}
      <View style={styles.header}>
        <MaterialCommunityIcons name="pickaxe" size={32} color={COLORS.cyan} />
        <Text variant="headlineSmall" style={styles.title}>
          Mining Monitor
        </Text>
        <Chip
          compact
          style={styles.heightChip}
          textStyle={styles.heightChipText}
        >
          Block {height.toLocaleString()}
        </Chip>
      </View>

      <Text variant="bodySmall" style={styles.infoText}>
        This screen shows mining statistics from your connected node. Mining itself runs on your
        desktop/server node, not on your mobile device.
      </Text>

      {/* Stats Grid */}
      {loading ? (
        <View style={styles.statsGrid}>
          <SkeletonLoader width="48%" height={100} borderRadius={16} />
          <SkeletonLoader width="48%" height={100} borderRadius={16} />
          <SkeletonLoader width="48%" height={100} borderRadius={16} />
          <SkeletonLoader width="48%" height={100} borderRadius={16} />
        </View>
      ) : stats ? (
        <>
          <View style={styles.statsGrid}>
            <StatCard
              icon="speedometer"
              label="Your Hashrate"
              value={`${formatNumber(stats.hashrate, { compact: true })} H/s`}
              color={COLORS.cyan}
            />
            <StatCard
              icon="cube-outline"
              label="Blocks Found"
              value={stats.blocks_found.toLocaleString()}
              color={COLORS.green}
            />
            <StatCard
              icon="cash-multiple"
              label="Total Rewards"
              value={formatBalance(parseFloat(stats.reward_total), 'QUG', 0)}
              color={COLORS.gold}
            />
            <StatCard
              icon="clock-outline"
              label="Last Block"
              value={lastBlockAgo !== null ? `${lastBlockAgo}m ago` : 'N/A'}
              color={COLORS.purple}
            />
          </View>

          {/* Network Stats */}
          <Text variant="titleSmall" style={styles.sectionTitle}>
            Network
          </Text>
          <Surface style={styles.networkCard} elevation={1}>
            <View style={styles.networkRow}>
              <Text style={styles.networkLabel}>Network Hashrate</Text>
              <Text style={styles.networkValue}>
                {formatNumber(stats.network_hashrate, { compact: true })} H/s
              </Text>
            </View>
            <View style={styles.networkDivider} />
            <View style={styles.networkRow}>
              <Text style={styles.networkLabel}>Difficulty</Text>
              <Text style={styles.networkValue}>
                {formatNumber(stats.difficulty, { compact: true })}
              </Text>
            </View>
            <View style={styles.networkDivider} />
            <View style={styles.networkRow}>
              <Text style={styles.networkLabel}>Your Share</Text>
              <Text style={styles.networkValue}>
                {stats.network_hashrate > 0
                  ? `${((stats.hashrate / stats.network_hashrate) * 100).toFixed(4)}%`
                  : 'N/A'}
              </Text>
            </View>
          </Surface>
        </>
      ) : (
        <View style={styles.emptyState}>
          <MaterialCommunityIcons name="pickaxe" size={48} color="rgba(255,255,255,0.15)" />
          <Text variant="bodyMedium" style={styles.emptyText}>
            No mining data available
          </Text>
          <Text variant="bodySmall" style={styles.emptySubtext}>
            Start mining on your node to see stats here
          </Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.darkBg,
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 8,
  },
  title: {
    color: '#FFFFFF',
    fontWeight: '700',
    flex: 1,
  },
  heightChip: {
    backgroundColor: 'rgba(0, 188, 212, 0.15)',
  },
  heightChipText: {
    color: COLORS.cyan,
    fontSize: 11,
  },
  infoText: {
    color: 'rgba(255, 255, 255, 0.4)',
    marginBottom: 24,
    lineHeight: 18,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    justifyContent: 'space-between',
    marginBottom: 24,
  },
  statCard: {
    width: '48%',
    padding: 16,
    borderRadius: 16,
    backgroundColor: COLORS.cardBg,
    alignItems: 'center',
    gap: 6,
  },
  statLabel: {
    color: 'rgba(255, 255, 255, 0.5)',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    textAlign: 'center',
  },
  statValue: {
    fontWeight: '700',
    textAlign: 'center',
  },
  sectionTitle: {
    color: 'rgba(255, 255, 255, 0.7)',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 12,
  },
  networkCard: {
    borderRadius: 16,
    padding: 16,
    backgroundColor: COLORS.cardBg,
  },
  networkRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
  },
  networkLabel: {
    color: 'rgba(255, 255, 255, 0.5)',
    fontSize: 14,
  },
  networkValue: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
  networkDivider: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
    gap: 12,
  },
  emptyText: {
    color: 'rgba(255, 255, 255, 0.5)',
  },
  emptySubtext: {
    color: 'rgba(255, 255, 255, 0.3)',
  },
});
