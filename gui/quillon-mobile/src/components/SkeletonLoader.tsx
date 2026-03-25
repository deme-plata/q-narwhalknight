import React, { useEffect, useRef } from 'react';
import { View, StyleSheet, Animated } from 'react-native';
import { COLORS } from '../theme';

interface SkeletonLoaderProps {
  width: number | string;
  height: number;
  borderRadius?: number;
  style?: object;
}

export function SkeletonLoader({
  width,
  height,
  borderRadius = 8,
  style,
}: SkeletonLoaderProps) {
  const opacity = useRef(new Animated.Value(0.3)).current;

  useEffect(() => {
    const animation = Animated.loop(
      Animated.sequence([
        Animated.timing(opacity, {
          toValue: 0.7,
          duration: 800,
          useNativeDriver: true,
        }),
        Animated.timing(opacity, {
          toValue: 0.3,
          duration: 800,
          useNativeDriver: true,
        }),
      ])
    );
    animation.start();
    return () => animation.stop();
  }, [opacity]);

  return (
    <Animated.View
      style={[
        styles.skeleton,
        {
          width: width as number,
          height,
          borderRadius,
          opacity,
        },
        style,
      ]}
    />
  );
}

/**
 * Pre-built skeleton for a balance card.
 */
export function BalanceCardSkeleton() {
  return (
    <View style={styles.cardContainer}>
      <SkeletonLoader width={120} height={14} />
      <SkeletonLoader width={200} height={36} style={{ marginTop: 12 }} />
      <SkeletonLoader width={100} height={14} style={{ marginTop: 8 }} />
      <View style={styles.cardDivider} />
      <View style={styles.cardRow}>
        <SkeletonLoader width={80} height={14} />
        <SkeletonLoader width={80} height={14} />
      </View>
    </View>
  );
}

/**
 * Pre-built skeleton for a transaction list item.
 */
export function TransactionItemSkeleton() {
  return (
    <View style={styles.txContainer}>
      <SkeletonLoader width={44} height={44} borderRadius={22} />
      <View style={styles.txDetails}>
        <View style={styles.txRow}>
          <SkeletonLoader width={60} height={14} />
          <SkeletonLoader width={80} height={14} />
        </View>
        <View style={[styles.txRow, { marginTop: 6 }]}>
          <SkeletonLoader width={120} height={12} />
          <SkeletonLoader width={60} height={12} />
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  skeleton: {
    backgroundColor: COLORS.surfaceBg,
  },
  cardContainer: {
    marginHorizontal: 16,
    marginTop: 16,
    padding: 20,
    borderRadius: 20,
    backgroundColor: COLORS.cardBg,
  },
  cardDivider: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    marginVertical: 16,
  },
  cardRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  txContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
  },
  txDetails: {
    flex: 1,
    marginLeft: 12,
  },
  txRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
});
