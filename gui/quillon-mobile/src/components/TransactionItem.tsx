import React from 'react';
import { View, StyleSheet, TouchableOpacity } from 'react-native';
import { Text } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { COLORS } from '../theme';
import { formatBalance } from '../utils/formatBalance';
import { truncateAddress } from '../utils/validateAddress';
import type { Transaction } from '../services/api';

interface TransactionItemProps {
  transaction: Transaction;
  walletAddress: string;
  onPress?: (tx: Transaction) => void;
}

export function TransactionItem({ transaction, walletAddress, onPress }: TransactionItemProps) {
  const isSwap = transaction.tx_type === 'swap' || transaction.direction === 'swap';
  const isSend = transaction.direction === 'sent';
  const isMiningReward = transaction.tx_type === 'mining_reward';
  const isFailed = transaction.status === 'failed';

  const icon = isSwap
    ? 'swap-horizontal-circle'
    : isMiningReward
    ? 'pickaxe'
    : isSend
    ? 'arrow-up-circle'
    : 'arrow-down-circle';
  const iconColor = isFailed
    ? COLORS.red
    : isSwap
    ? COLORS.purple
    : isSend
    ? COLORS.orange
    : COLORS.green;

  const amountPrefix = isSend ? '-' : '+';
  const amountColor = isFailed
    ? COLORS.red
    : isSend
    ? COLORS.orange
    : COLORS.green;

  const token = transaction.token_symbol || 'QUG';
  const counterparty = isSend ? transaction.to : transaction.from;
  const label = isSwap
    ? `Swap ${transaction.token_in ?? ''} → ${transaction.token_out ?? ''}`
    : isMiningReward
    ? 'Mining Reward'
    : isSend
    ? 'Sent'
    : 'Received';

  const date = new Date(transaction.timestamp * 1000);
  const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const dateStr = date.toLocaleDateString([], { month: 'short', day: 'numeric' });

  return (
    <TouchableOpacity
      style={styles.container}
      onPress={() => onPress?.(transaction)}
      activeOpacity={0.6}
    >
      <View style={[styles.iconContainer, { backgroundColor: `${iconColor}12` }]}>
        <MaterialCommunityIcons name={icon} size={24} color={iconColor} />
      </View>

      <View style={styles.details}>
        <View style={styles.topRow}>
          <View style={styles.labelRow}>
            <Text variant="bodyMedium" style={styles.label}>
              {label}
            </Text>
            {isFailed && (
              <View style={styles.failedBadge}>
                <Text style={styles.failedBadgeText}>Failed</Text>
              </View>
            )}
          </View>
          <Text variant="bodyMedium" style={[styles.amount, { color: amountColor }]}>
            {amountPrefix}{formatBalance(transaction.amount, token)}
          </Text>
        </View>
        <View style={styles.bottomRow}>
          <Text variant="bodySmall" style={styles.address}>
            {isSwap ? `Block #${transaction.block_height}` : truncateAddress(counterparty, 6)}
          </Text>
          <Text variant="bodySmall" style={styles.time}>
            {dateStr} {timeStr}
          </Text>
        </View>
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
  },
  iconContainer: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  details: {
    flex: 1,
  },
  topRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  labelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  bottomRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 4,
  },
  label: {
    color: '#FFFFFF',
    fontWeight: '500',
  },
  amount: {
    fontWeight: '600',
  },
  address: {
    color: 'rgba(255, 255, 255, 0.5)',
  },
  time: {
    color: 'rgba(255, 255, 255, 0.35)',
  },
  pendingBadge: {
    paddingHorizontal: 6,
    paddingVertical: 1,
    borderRadius: 6,
    backgroundColor: 'rgba(255, 215, 0, 0.12)',
    borderWidth: 1,
    borderColor: 'rgba(255, 215, 0, 0.2)',
  },
  pendingBadgeText: {
    color: COLORS.gold,
    fontSize: 10,
    fontWeight: '600',
  },
  failedBadge: {
    paddingHorizontal: 6,
    paddingVertical: 1,
    borderRadius: 6,
    backgroundColor: 'rgba(244, 67, 54, 0.12)',
    borderWidth: 1,
    borderColor: 'rgba(244, 67, 54, 0.2)',
  },
  failedBadgeText: {
    color: COLORS.red,
    fontSize: 10,
    fontWeight: '600',
  },
});
