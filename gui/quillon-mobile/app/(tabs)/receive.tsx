import React, { useState, useCallback } from 'react';
import { View, StyleSheet, ScrollView, Share, Dimensions, Platform } from 'react-native';
import { Text, Button, Snackbar } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as Clipboard from 'expo-clipboard';
import * as Haptics from 'expo-haptics';
import { COLORS } from '../../src/theme';
import { useWalletStore } from '../../src/stores/walletStore';
import { truncateAddress } from '../../src/utils/validateAddress';
import { QRDisplay } from '../../src/components/QRDisplay';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const QR_SIZE = Math.min(SCREEN_WIDTH - 96, 240);

export default function ReceiveScreen() {
  const address = useWalletStore((s) => s.address);
  const [copied, setCopied] = useState(false);
  const [posMode, setPosMode] = useState(false);

  const handleCopy = useCallback(async () => {
    if (!address) return;
    await Clipboard.setStringAsync(address);
    await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [address]);

  const handleShare = useCallback(async () => {
    if (!address) return;
    try {
      await Share.share({
        message: `My Quillon address: ${address}`,
        title: 'Quillon Wallet Address',
      });
    } catch (err) {
      // User cancelled share
    }
  }, [address]);

  if (posMode) {
    return (
      <View style={styles.posContainer}>
        <View style={styles.posContent}>
          <QRDisplay value={address ?? ''} size={SCREEN_WIDTH - 80} />
          <Text variant="headlineMedium" style={styles.posTitle}>
            Scan to Pay
          </Text>
          <Text variant="bodyLarge" style={styles.posAddress}>
            {truncateAddress(address ?? '', 10)}
          </Text>
        </View>
        <Button
          mode="text"
          onPress={() => setPosMode(false)}
          textColor={COLORS.cyan}
          style={styles.posExitButton}
        >
          Exit PoS Mode
        </Button>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Text variant="headlineSmall" style={styles.title}>
          Receive QUG
        </Text>

        <Text variant="bodyMedium" style={styles.subtitle}>
          Share your address or QR code to receive payments
        </Text>

        {/* QR Code with gradient border */}
        <View style={styles.qrContainer}>
          <QRDisplay
            value={address ?? ''}
            size={QR_SIZE}
            label="Scan to send QUG to this wallet"
          />
        </View>

        {/* Address Display */}
        <View style={styles.addressCardOuter}>
          <LinearGradient
            colors={['rgba(0, 188, 212, 0.08)', 'rgba(26, 26, 38, 0.95)']}
            start={{ x: 0, y: 0 }}
            end={{ x: 0, y: 1 }}
            style={styles.addressCard}
          >
            <Text variant="labelSmall" style={styles.addressLabel}>
              YOUR WALLET ADDRESS
            </Text>
            <Text
              variant="bodyMedium"
              style={styles.addressText}
              selectable
              numberOfLines={2}
            >
              {address ?? ''}
            </Text>

            <View style={styles.addressActions}>
              <Button
                mode="contained"
                onPress={handleCopy}
                icon={copied ? 'check' : 'content-copy'}
                style={styles.copyButton}
                buttonColor={copied ? COLORS.green : COLORS.cyan}
                textColor="#000000"
                compact
                labelStyle={{ fontWeight: '700' }}
              >
                {copied ? 'Copied!' : 'Copy'}
              </Button>
              <Button
                mode="outlined"
                onPress={handleShare}
                icon="share-variant"
                style={styles.shareButton}
                textColor={COLORS.cyan}
                compact
              >
                Share
              </Button>
            </View>
          </LinearGradient>
        </View>

        {/* PQ Security Badge */}
        <View style={styles.pqBadge}>
          <MaterialCommunityIcons name="shield-check" size={14} color={COLORS.green} />
          <Text style={styles.pqBadgeText}>
            Post-Quantum Protected Address
          </Text>
        </View>

        {/* PoS Mode Button */}
        <Button
          mode="outlined"
          onPress={() => setPosMode(true)}
          icon="storefront"
          style={styles.posButton}
          textColor={COLORS.purple}
          contentStyle={styles.posButtonContent}
        >
          Point of Sale Mode
        </Button>
        <Text variant="bodySmall" style={styles.posHint}>
          Full-screen QR for in-store payments
        </Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.darkBg,
  },
  scrollContent: {
    padding: 24,
    alignItems: 'center',
  },
  title: {
    color: '#FFFFFF',
    fontWeight: '700',
    marginBottom: 8,
  },
  subtitle: {
    color: 'rgba(255, 255, 255, 0.5)',
    textAlign: 'center',
    marginBottom: 32,
  },
  qrContainer: {
    marginBottom: 24,
  },
  addressCardOuter: {
    width: '100%',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(0, 188, 212, 0.1)',
    overflow: 'hidden',
    marginBottom: 16,
  },
  addressCard: {
    padding: 16,
  },
  addressLabel: {
    color: 'rgba(255, 255, 255, 0.4)',
    marginBottom: 8,
    letterSpacing: 1,
  },
  addressText: {
    color: '#FFFFFF',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    fontSize: 13,
    lineHeight: 20,
    marginBottom: 16,
  },
  addressActions: {
    flexDirection: 'row',
    gap: 12,
  },
  copyButton: {
    flex: 1,
    borderRadius: 10,
  },
  shareButton: {
    flex: 1,
    borderRadius: 10,
    borderColor: 'rgba(0, 188, 212, 0.4)',
  },
  pqBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    backgroundColor: 'rgba(76, 175, 80, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(76, 175, 80, 0.2)',
    marginBottom: 24,
  },
  pqBadgeText: {
    color: COLORS.green,
    fontSize: 11,
    fontWeight: '600',
    letterSpacing: 0.3,
  },
  posButton: {
    width: '100%',
    borderRadius: 14,
    borderColor: 'rgba(124, 77, 255, 0.4)',
    borderWidth: 1.5,
    marginBottom: 8,
  },
  posButtonContent: {
    paddingVertical: 6,
  },
  posHint: {
    color: 'rgba(255, 255, 255, 0.3)',
  },
  // PoS Mode
  posContainer: {
    flex: 1,
    backgroundColor: '#000000',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  posContent: {
    alignItems: 'center',
  },
  posTitle: {
    color: '#FFFFFF',
    fontWeight: '700',
    marginTop: 32,
    marginBottom: 8,
  },
  posAddress: {
    color: 'rgba(255, 255, 255, 0.6)',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  posExitButton: {
    position: 'absolute',
    bottom: 40,
  },
});
