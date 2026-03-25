import React from 'react';
import { View, StyleSheet, ScrollView } from 'react-native';
import { Text, Surface, Button, Chip } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useLocalSearchParams } from 'expo-router';
import * as Clipboard from 'expo-clipboard';
import * as Haptics from 'expo-haptics';
import { COLORS } from '../../src/theme';
import { useWalletStore } from '../../src/stores/walletStore';
import { truncateAddress } from '../../src/utils/validateAddress';
import { formatBalance } from '../../src/utils/formatBalance';

function DetailRow({
  label,
  value,
  copyable = false,
  color,
}: {
  label: string;
  value: string;
  copyable?: boolean;
  color?: string;
}) {
  const handleCopy = async () => {
    await Clipboard.setStringAsync(value);
    await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  };

  return (
    <View style={styles.detailRow}>
      <Text style={styles.detailLabel}>{label}</Text>
      <View style={styles.detailValueRow}>
        <Text
          style={[styles.detailValue, color ? { color } : {}]}
          numberOfLines={1}
          selectable
        >
          {value}
        </Text>
        {copyable && (
          <MaterialCommunityIcons
            name="content-copy"
            size={16}
            color="rgba(255, 255, 255, 0.4)"
            onPress={handleCopy}
          />
        )}
      </View>
    </View>
  );
}

export default function TransactionDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const address = useWalletStore((s) => s.address);
  const transactions = useWalletStore((s) => s.transactions);

  const tx = transactions.find((t) => t.id === id);

  if (!tx) {
    return (
      <View style={styles.container}>
        <View style={styles.emptyState}>
          <MaterialCommunityIcons
            name="file-search-outline"
            size={48}
            color="rgba(255, 255, 255, 0.2)"
          />
          <Text variant="bodyMedium" style={styles.emptyText}>
            Transaction not found
          </Text>
          <Text variant="bodySmall" style={styles.emptySubtext}>
            Hash: {truncateAddress(id ?? '', 12)}
          </Text>
        </View>
      </View>
    );
  }

  const isSend = tx.direction === 'sent';
  const date = new Date(tx.timestamp * 1000);

  const statusColor =
    tx.status === 'confirmed'
      ? COLORS.green
      : tx.status === 'pending'
      ? COLORS.gold
      : COLORS.red;

  const statusIcon =
    tx.status === 'confirmed'
      ? 'check-circle'
      : tx.status === 'pending'
      ? 'clock-outline'
      : 'close-circle';

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.scrollContent}>
      {/* Status Header */}
      <View style={styles.statusHeader}>
        <MaterialCommunityIcons
          name={isSend ? 'arrow-up-circle' : 'arrow-down-circle'}
          size={56}
          color={isSend ? COLORS.orange : COLORS.green}
        />
        <Text variant="headlineMedium" style={styles.amount}>
          {isSend ? '-' : '+'}{formatBalance(tx.amount, tx.token_symbol || 'QUG')}
        </Text>
        <Chip
          icon={() => (
            <MaterialCommunityIcons name={statusIcon} size={14} color={statusColor} />
          )}
          style={[styles.statusChip, { borderColor: statusColor }]}
          textStyle={[styles.statusChipText, { color: statusColor }]}
        >
          {tx.status.charAt(0).toUpperCase() + tx.status.slice(1)}
        </Chip>
      </View>

      {/* Details */}
      <Surface style={styles.detailsCard} elevation={1}>
        <DetailRow
          label="Transaction Hash"
          value={tx.id}
          copyable
        />
        <View style={styles.detailDivider} />
        <DetailRow
          label="From"
          value={isSend ? 'You' : truncateAddress(tx.from, 10)}
          copyable={!isSend}
        />
        <View style={styles.detailDivider} />
        <DetailRow
          label="To"
          value={isSend ? truncateAddress(tx.to, 10) : 'You'}
          copyable={isSend}
        />
        <View style={styles.detailDivider} />
        <DetailRow
          label="Amount"
          value={formatBalance(tx.amount, tx.token_symbol || 'QUG')}
          color={isSend ? COLORS.orange : COLORS.green}
        />
        <View style={styles.detailDivider} />
        <DetailRow label="Type" value={tx.tx_type} />
        <View style={styles.detailDivider} />
        <DetailRow label="Block Height" value={tx.block_height.toLocaleString()} />
        <View style={styles.detailDivider} />
        <DetailRow
          label="Date"
          value={date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
          })}
        />
      </Surface>

      <Button
        mode="outlined"
        onPress={async () => {
          await Clipboard.setStringAsync(tx.id);
          await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        }}
        style={styles.copyHashButton}
        textColor={COLORS.cyan}
        icon="content-copy"
      >
        Copy Transaction Hash
      </Button>
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
  statusHeader: {
    alignItems: 'center',
    paddingVertical: 24,
    gap: 12,
  },
  amount: {
    color: '#FFFFFF',
    fontWeight: '700',
  },
  statusChip: {
    backgroundColor: 'transparent',
    borderWidth: 1,
  },
  statusChipText: {
    fontWeight: '600',
    fontSize: 13,
  },
  detailsCard: {
    borderRadius: 16,
    padding: 16,
    backgroundColor: COLORS.cardBg,
    marginBottom: 20,
  },
  detailRow: {
    paddingVertical: 12,
  },
  detailLabel: {
    color: 'rgba(255, 255, 255, 0.4)',
    fontSize: 12,
    marginBottom: 4,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  detailValueRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  detailValue: {
    color: '#FFFFFF',
    fontSize: 14,
    flex: 1,
  },
  detailDivider: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
  },
  copyHashButton: {
    borderRadius: 14,
    borderColor: COLORS.cyan,
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    gap: 12,
  },
  emptyText: {
    color: 'rgba(255, 255, 255, 0.5)',
  },
  emptySubtext: {
    color: 'rgba(255, 255, 255, 0.3)',
    fontFamily: 'monospace',
  },
});
