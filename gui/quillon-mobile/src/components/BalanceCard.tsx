import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Text, IconButton } from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { COLORS } from '../theme';
import { formatBalance, formatFiat } from '../utils/formatBalance';
import { useSettingsStore } from '../stores/settingsStore';

interface BalanceCardProps {
  balance: number;
  fiatValue?: number;
  currency?: string;
}

export function BalanceCard({ balance, fiatValue = 0, currency = 'USD' }: BalanceCardProps) {
  const hideBalances = useSettingsStore((s) => s.hideBalances);
  const setHideBalances = useSettingsStore((s) => s.setHideBalances);

  return (
    <View
      style={styles.outerContainer}
      accessible={true}
      accessibilityRole="summary"
      accessibilityLabel={hideBalances
        ? 'Total balance hidden'
        : `Total balance: ${formatBalance(balance, 'QUG', 0)}`}
    >
      <LinearGradient
        colors={['rgba(0, 188, 212, 0.12)', 'rgba(124, 77, 255, 0.08)', 'rgba(20, 20, 32, 0.95)']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.gradient}
      >
        {/* Top accent line */}
        <LinearGradient
          colors={[COLORS.cyan, COLORS.purple]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
          style={styles.accentLine}
        />

        <View style={styles.header}>
          <Text variant="labelMedium" style={styles.label}>
            Total Balance
          </Text>
          <IconButton
            icon={hideBalances ? 'eye-off' : 'eye'}
            size={20}
            iconColor={COLORS.cyan}
            onPress={() => setHideBalances(!hideBalances)}
            style={styles.eyeButton}
            accessibilityLabel={hideBalances ? 'Show balance' : 'Hide balance'}
            accessibilityRole="button"
          />
        </View>

        <Text
          variant="headlineLarge"
          style={styles.balance}
          accessibilityElementsHidden={hideBalances}
        >
          {hideBalances ? '****' : formatBalance(balance, 'QUG', 0)}
        </Text>

        {fiatValue > 0 && (
          <Text variant="bodyMedium" style={styles.fiat}>
            {hideBalances ? '****' : formatFiat(fiatValue, currency)}
          </Text>
        )}

        <View style={styles.divider} />

        <View style={styles.statsRow}>
          <View style={styles.stat}>
            <Text variant="labelSmall" style={styles.statLabel}>
              24h Change
            </Text>
            <Text variant="bodySmall" style={styles.statValueGreen}>
              +0.00%
            </Text>
          </View>
          <View style={styles.stat}>
            <Text variant="labelSmall" style={styles.statLabel}>
              Security
            </Text>
            <View style={styles.pqRow}>
              <MaterialCommunityIcons name="shield-check" size={12} color={COLORS.green} />
              <Text variant="bodySmall" style={styles.statValueGreen}>
                PQ Hybrid
              </Text>
            </View>
          </View>
          <View style={styles.stat}>
            <Text variant="labelSmall" style={styles.statLabel}>
              Portfolio
            </Text>
            <Text variant="bodySmall" style={styles.statValue}>
              100%
            </Text>
          </View>
        </View>
      </LinearGradient>
    </View>
  );
}

const styles = StyleSheet.create({
  outerContainer: {
    marginHorizontal: 16,
    marginTop: 16,
    borderRadius: 20,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(0, 188, 212, 0.12)',
  },
  gradient: {
    padding: 20,
  },
  accentLine: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 2,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 4,
  },
  label: {
    color: 'rgba(255, 255, 255, 0.5)',
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  eyeButton: {
    margin: -8,
  },
  balance: {
    color: '#FFFFFF',
    fontWeight: '700',
    marginTop: 8,
    fontSize: 34,
  },
  fiat: {
    color: 'rgba(255, 255, 255, 0.45)',
    marginTop: 4,
  },
  divider: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
    marginVertical: 16,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  stat: {
    alignItems: 'center',
  },
  statLabel: {
    color: 'rgba(255, 255, 255, 0.4)',
    marginBottom: 4,
  },
  statValue: {
    color: '#FFFFFF',
  },
  statValueGreen: {
    color: COLORS.green,
  },
  pqRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 3,
  },
});
