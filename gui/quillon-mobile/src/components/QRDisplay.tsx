import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Text } from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import QRCode from 'react-native-qrcode-svg';
import { COLORS } from '../theme';

interface QRDisplayProps {
  value: string;
  size?: number;
  label?: string;
}

export function QRDisplay({ value, size = 200, label }: QRDisplayProps) {
  if (!value) {
    return (
      <View style={[styles.container, { width: size, height: size }]}>
        <Text style={styles.placeholder}>No address</Text>
      </View>
    );
  }

  return (
    <View style={styles.wrapper}>
      <View style={styles.gradientBorderOuter}>
        <LinearGradient
          colors={[COLORS.cyan, COLORS.purple]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.gradientBorder}
        >
          <View style={styles.qrInner}>
            <QRCode
              value={value}
              size={size}
              backgroundColor="#FFFFFF"
              color="#000000"
              quietZone={12}
            />
          </View>
        </LinearGradient>
      </View>
      {label && (
        <Text variant="bodySmall" style={styles.label}>
          {label}
        </Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    alignItems: 'center',
  },
  container: {
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 16,
    borderWidth: 2,
    borderColor: 'rgba(0, 188, 212, 0.3)',
  },
  gradientBorderOuter: {
    borderRadius: 16,
    overflow: 'hidden',
  },
  gradientBorder: {
    padding: 3,
    borderRadius: 16,
  },
  qrInner: {
    borderRadius: 13,
    overflow: 'hidden',
    backgroundColor: '#FFFFFF',
  },
  placeholder: {
    color: 'rgba(255, 255, 255, 0.4)',
    textAlign: 'center',
  },
  label: {
    color: 'rgba(255, 255, 255, 0.5)',
    marginTop: 12,
    textAlign: 'center',
  },
});
