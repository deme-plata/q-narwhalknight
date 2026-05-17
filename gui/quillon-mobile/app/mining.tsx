import React, { useEffect, useState, useCallback } from 'react';
import { View, StyleSheet, ScrollView, RefreshControl, TouchableOpacity } from 'react-native';
import { Text, Surface, Chip, Button } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { router } from 'expo-router';
import { COLORS } from '../src/theme';
import { useWalletStore } from '../src/stores/walletStore';
import { useNetworkStore } from '../src/stores/networkStore';
import * as api from '../src/services/api';
import { formatNumber } from '../src/utils/formatBalance';
import { SkeletonLoader } from '../src/components/SkeletonLoader';

function formatHashrate(hs: number): string {
  if (hs >= 1e12) return `${(hs / 1e12).toFixed(2)} TH/s`;
  if (hs >= 1e9) return `${(hs / 1e9).toFixed(2)} GH/s`;
  if (hs >= 1e6) return `${(hs / 1e6).toFixed(2)} MH/s`;
  if (hs >= 1e3) return `${(hs / 1e3).toFixed(2)} KH/s`;
  return `${hs.toFixed(0)} H/s`;
}

function formatActivityTime(secs: number): string {
  if (secs >= 86400) return `${Math.floor(secs / 86400)}d ago`;
  if (secs >= 3600) return `${Math.floor(secs / 3600)}h ago`;
  if (secs >= 60) return `${Math.floor(secs / 60)}m ago`;
  return 'just now';
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

function WorkerCard({ worker }: { worker: api.WorkerStats }) {
  const name = worker.worker_name || worker.worker_id || 'Unknown';
  // Truncate long worker IDs (e.g. "p2p:12D3KooW..." → "p2p:12D3Ko...")
  const displayName = name.length > 20 ? name.slice(0, 18) + '...' : name;

  return (
    <Surface style={styles.workerCard} elevation={1}>
      <View style={styles.workerHeader}>
        <View style={styles.workerNameRow}>
          <MaterialCommunityIcons
            name="server"
            size={18}
            color={worker.is_active ? COLORS.green : 'rgba(255,255,255,0.3)'}
          />
          <Text variant="titleSmall" style={styles.workerName} numberOfLines={1}>
            {displayName}
          </Text>
        </View>
        <Chip
          compact
          style={[
            styles.statusChip,
            worker.is_active ? styles.statusActive : styles.statusInactive,
          ]}
          textStyle={[
            styles.statusText,
            worker.is_active ? styles.statusTextActive : styles.statusTextInactive,
          ]}
        >
          {worker.is_active ? 'Active' : 'Offline'}
        </Chip>
      </View>

      <View style={styles.workerStats}>
        <View style={styles.workerStatItem}>
          <Text style={styles.workerStatLabel}>Hashrate</Text>
          <Text style={[styles.workerStatValue, { color: COLORS.cyan }]}>
            {formatHashrate(worker.hash_rate)}
          </Text>
        </View>
        <View style={styles.workerStatItem}>
          <Text style={styles.workerStatLabel}>Blocks</Text>
          <Text style={[styles.workerStatValue, { color: COLORS.green }]}>
            {worker.blocks_found.toLocaleString()}
          </Text>
        </View>
        <View style={styles.workerStatItem}>
          <Text style={styles.workerStatLabel}>Rewards</Text>
          <Text style={[styles.workerStatValue, { color: COLORS.gold }]}>
            {worker.rewards_earned}
          </Text>
        </View>
      </View>

      <View style={styles.workerFooter}>
        <Text style={styles.workerFooterText}>
          {worker.solutions_submitted.toLocaleString()} solutions
        </Text>
        <Text style={styles.workerFooterText}>
          {worker.is_active ? formatActivityTime(worker.last_activity_secs) : 'inactive'}
        </Text>
      </View>
    </Surface>
  );
}

export default function MiningScreen() {
  const address = useWalletStore((s) => s.address);
  const height = useNetworkStore((s) => s.height);
  const difficulty = useNetworkStore((s) => s.difficulty);
  const [stats, setStats] = useState<api.MiningStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchStats = useCallback(async () => {
    if (!address) return;
    try {
      const data = await api.getMiningStats(address);
      setStats(data);
      setError(null);
    } catch (err) {
      console.error('[Mining] Stats fetch error:', err);
      setError(err instanceof Error ? err.message : 'Failed to load mining stats');
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

  const activeWorkers = stats?.workers.filter((w) => w.is_active) ?? [];
  const inactiveWorkers = stats?.workers.filter((w) => !w.is_active) ?? [];

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
        <TouchableOpacity
          style={styles.scanButton}
          onPress={() => router.push('/miner-scan')}
          activeOpacity={0.7}
        >
          <MaterialCommunityIcons name="qrcode-scan" size={18} color={COLORS.cyan} />
          <Text style={styles.scanButtonText}>Link</Text>
        </TouchableOpacity>
        <Chip compact style={styles.heightChip} textStyle={styles.heightChipText}>
          Block {height.toLocaleString()}
        </Chip>
      </View>

      <Text variant="bodySmall" style={styles.infoText}>
        Miners connected to your wallet address. Mining runs on your desktop/server node.
      </Text>

      {/* Stats Grid */}
      {loading ? (
        <View style={styles.statsGrid}>
          <SkeletonLoader width="48%" height={100} borderRadius={16} />
          <SkeletonLoader width="48%" height={100} borderRadius={16} />
          <SkeletonLoader width="48%" height={100} borderRadius={16} />
          <SkeletonLoader width="48%" height={100} borderRadius={16} />
        </View>
      ) : error ? (
        <View style={styles.emptyState}>
          <MaterialCommunityIcons name="alert-circle-outline" size={48} color={COLORS.orange} />
          <Text variant="bodyMedium" style={styles.emptyText}>
            {error}
          </Text>
          <Text variant="bodySmall" style={styles.emptySubtext}>
            Pull down to retry
          </Text>
        </View>
      ) : stats ? (
        <>
          {/* Aggregate Stats */}
          <View style={styles.statsGrid}>
            <StatCard
              icon="speedometer"
              label="Total Hashrate"
              value={formatHashrate(stats.hash_rate)}
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
              value={stats.rewards_earned}
              color={COLORS.gold}
            />
            <StatCard
              icon="account-group"
              label="Workers"
              value={`${activeWorkers.length} / ${stats.total_workers}`}
              color={stats.is_active ? COLORS.cyan : COLORS.orange}
            />
          </View>

          {/* Network Info */}
          <Text variant="titleSmall" style={styles.sectionTitle}>
            Network
          </Text>
          <Surface style={styles.networkCard} elevation={1}>
            <View style={styles.networkRow}>
              <Text style={styles.networkLabel}>Difficulty</Text>
              <Text style={styles.networkValue}>
                {formatNumber(difficulty, { compact: true })}
              </Text>
            </View>
            <View style={styles.networkDivider} />
            <View style={styles.networkRow}>
              <Text style={styles.networkLabel}>Status</Text>
              <Text
                style={[
                  styles.networkValue,
                  { color: stats.is_active ? COLORS.green : COLORS.orange },
                ]}
              >
                {stats.is_active ? 'Mining Active' : 'Inactive'}
              </Text>
            </View>
            {stats.last_activity_secs < 86400 * 365 && (
              <>
                <View style={styles.networkDivider} />
                <View style={styles.networkRow}>
                  <Text style={styles.networkLabel}>Last Activity</Text>
                  <Text style={styles.networkValue}>
                    {formatActivityTime(stats.last_activity_secs)}
                  </Text>
                </View>
              </>
            )}
          </Surface>

          {/* Active Workers */}
          {activeWorkers.length > 0 && (
            <>
              <Text variant="titleSmall" style={styles.sectionTitle}>
                Active Miners ({activeWorkers.length})
              </Text>
              {activeWorkers.map((worker, i) => (
                <WorkerCard key={`${worker.worker_id}-${i}`} worker={worker} />
              ))}
            </>
          )}

          {/* Inactive Workers */}
          {inactiveWorkers.length > 0 && (
            <>
              <Text variant="titleSmall" style={styles.sectionTitle}>
                Offline Miners ({inactiveWorkers.length})
              </Text>
              {inactiveWorkers.map((worker, i) => (
                <WorkerCard key={`${worker.worker_id}-${i}`} worker={worker} />
              ))}
            </>
          )}

          {/* No workers but has stats */}
          {stats.total_workers === 0 && stats.blocks_found > 0 && (
            <View style={styles.emptyState}>
              <MaterialCommunityIcons name="lan-disconnect" size={36} color="rgba(255,255,255,0.2)" />
              <Text variant="bodySmall" style={styles.emptySubtext}>
                No miners currently connected. Historical stats shown above.
              </Text>
            </View>
          )}
        </>
      ) : (
        <View style={styles.emptyState}>
          <MaterialCommunityIcons name="pickaxe" size={48} color="rgba(255,255,255,0.15)" />
          <Text variant="bodyMedium" style={styles.emptyText}>
            No mining data available
          </Text>
          <Text variant="bodySmall" style={styles.emptySubtext}>
            Start mining on your node, then scan the QR code to link it to this wallet
          </Text>
          <Button
            mode="contained"
            onPress={() => router.push('/miner-scan')}
            style={styles.linkMinerButton}
            buttonColor={COLORS.cyan}
            textColor="#000"
            icon="qrcode-scan"
          >
            Scan Miner QR
          </Button>
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
  scanButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: 'rgba(0, 188, 212, 0.12)',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'rgba(0, 188, 212, 0.25)',
    marginRight: 8,
  },
  scanButtonText: {
    color: COLORS.cyan,
    fontSize: 12,
    fontWeight: '600',
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

  // Stats grid
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
    fontSize: 10,
  },
  statValue: {
    fontWeight: '700',
    textAlign: 'center',
  },

  // Section titles
  sectionTitle: {
    color: 'rgba(255, 255, 255, 0.7)',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 12,
    marginTop: 8,
  },

  // Network card
  networkCard: {
    borderRadius: 16,
    padding: 16,
    backgroundColor: COLORS.cardBg,
    marginBottom: 8,
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

  // Worker cards
  workerCard: {
    borderRadius: 14,
    padding: 14,
    backgroundColor: COLORS.cardBg,
    marginBottom: 10,
  },
  workerHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  workerNameRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    flex: 1,
    marginRight: 8,
  },
  workerName: {
    color: '#FFFFFF',
    fontWeight: '600',
    flex: 1,
  },
  statusChip: {
    height: 24,
  },
  statusActive: {
    backgroundColor: 'rgba(76, 175, 80, 0.15)',
  },
  statusInactive: {
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
  },
  statusText: {
    fontSize: 10,
    lineHeight: 12,
  },
  statusTextActive: {
    color: COLORS.green,
  },
  statusTextInactive: {
    color: 'rgba(255, 255, 255, 0.4)',
  },
  workerStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  workerStatItem: {
    flex: 1,
    alignItems: 'center',
  },
  workerStatLabel: {
    color: 'rgba(255, 255, 255, 0.4)',
    fontSize: 10,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 4,
  },
  workerStatValue: {
    fontSize: 13,
    fontWeight: '700',
  },
  workerFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.06)',
    paddingTop: 8,
  },
  workerFooterText: {
    color: 'rgba(255, 255, 255, 0.3)',
    fontSize: 11,
  },

  // Empty state
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
    textAlign: 'center',
    paddingHorizontal: 24,
  },
  linkMinerButton: {
    borderRadius: 12,
    marginTop: 16,
  },
});
