import React, { useState, useCallback } from 'react';
import { View, StyleSheet, ScrollView, Alert, TouchableOpacity } from 'react-native';
import { Text, List, Switch, Divider, Button, Surface, Snackbar } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { router } from 'expo-router';
import * as Clipboard from 'expo-clipboard';
import * as Haptics from 'expo-haptics';
import { COLORS } from '../src/theme';
import { useSettingsStore } from '../src/stores/settingsStore';
import { useWalletStore } from '../src/stores/walletStore';
import { useNetworkStore } from '../src/stores/networkStore';
import { checkBiometrics, setPin, setBiometricAuth } from '../src/services/auth';
import { truncateAddress } from '../src/utils/validateAddress';
import { getMnemonic, getWalletAddress, getDilithium5PublicKey, hasPQKeys } from '../src/services/secureStorage';
import { derivePublicKey } from '../src/services/wallet';
import { sha3_256 } from '@noble/hashes/sha3.js';
import { bytesToHex } from '@noble/hashes/utils.js';

export default function SettingsScreen() {
  const address = useWalletStore((s) => s.address);
  const logout = useWalletStore((s) => s.logout);
  const version = useNetworkStore((s) => s.version);
  const networkId = useNetworkStore((s) => s.networkId);

  const biometricEnabled = useSettingsStore((s) => s.biometricEnabled);
  const setBiometricEnabled = useSettingsStore((s) => s.setBiometricEnabled);
  const autoLockMinutes = useSettingsStore((s) => s.autoLockMinutes);
  const setAutoLockMinutes = useSettingsStore((s) => s.setAutoLockMinutes);
  const hideBalances = useSettingsStore((s) => s.hideBalances);
  const setHideBalances = useSettingsStore((s) => s.setHideBalances);
  const notificationsEnabled = useSettingsStore((s) => s.notificationsEnabled);
  const setNotificationsEnabled = useSettingsStore((s) => s.setNotificationsEnabled);

  const [error, setError] = useState('');
  const [snackMsg, setSnackMsg] = useState('');
  const [showMnemonic, setShowMnemonic] = useState(false);
  const [mnemonic, setMnemonic] = useState<string | null>(null);
  const [showPrivateKey, setShowPrivateKey] = useState(false);
  const [privateKeyHex, setPrivateKeyHex] = useState<string | null>(null);
  const [publicKeyHex, setPublicKeyHex] = useState<string | null>(null);
  const [pqAvailable, setPqAvailable] = useState(false);
  const [dilithiumPk, setDilithiumPk] = useState<string | null>(null);

  const copyToClipboard = useCallback(async (value: string, label: string) => {
    await Clipboard.setStringAsync(value);
    await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    setSnackMsg(`${label} copied`);
  }, []);

  const handleRevealMnemonic = useCallback(() => {
    Alert.alert(
      'Reveal Seed Phrase',
      'Anyone with your seed phrase can steal your funds. Never share it.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'I Understand',
          style: 'destructive',
          onPress: async () => {
            const phrase = await getMnemonic();
            if (phrase) {
              setMnemonic(phrase);
              setShowMnemonic(true);
            } else {
              setError('No seed phrase found');
            }
          },
        },
      ]
    );
  }, []);

  const handleRevealPrivateKey = useCallback(() => {
    Alert.alert(
      'Reveal Private Key',
      'Your private key gives full access to your wallet. Never share it.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'I Understand',
          style: 'destructive',
          onPress: async () => {
            const phrase = await getMnemonic();
            if (!phrase) {
              setError('No wallet found');
              return;
            }
            const mnemonicBytes = new TextEncoder().encode(phrase);
            const pk = sha3_256(mnemonicBytes);
            setPrivateKeyHex(bytesToHex(pk));
            pk.fill(0);
            setShowPrivateKey(true);

            // Also derive public key
            const pubHex = await derivePublicKey(phrase);
            setPublicKeyHex(pubHex);

            // Check PQ keys
            const hasPQ = await hasPQKeys();
            setPqAvailable(hasPQ);
            if (hasPQ) {
              const dilPk = await getDilithium5PublicKey();
              setDilithiumPk(dilPk);
            }
          },
        },
      ]
    );
  }, []);

  const handleHideSecrets = useCallback(() => {
    setShowMnemonic(false);
    setShowPrivateKey(false);
    setMnemonic(null);
    setPrivateKeyHex(null);
  }, []);

  const handleToggleBiometric = async (value: boolean) => {
    try {
      if (value) {
        const { available, enrolled } = await checkBiometrics();
        if (!available || !enrolled) {
          setError('Biometrics not available or not enrolled on this device');
          return;
        }
      }
      await setBiometricAuth(value);
      await setBiometricEnabled(value);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to toggle biometrics');
    }
  };

  const handleLogout = () => {
    Alert.alert(
      'Logout',
      'This will remove your wallet from this device. Make sure you have backed up your seed phrase!',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Logout',
          style: 'destructive',
          onPress: async () => {
            await logout();
            router.replace('/(auth)/login');
          },
        },
      ]
    );
  };

  const autoLockOptions = [
    { label: '1 minute', value: 1 },
    { label: '5 minutes', value: 5 },
    { label: '15 minutes', value: 15 },
    { label: '30 minutes', value: 30 },
    { label: 'Never', value: 0 },
  ];

  const currentAutoLock = autoLockOptions.find((o) => o.value === autoLockMinutes)?.label ?? '5 minutes';

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.scrollContent}>
      {/* Wallet Info */}
      <Surface style={styles.section} elevation={1}>
        <List.Item
          title="Wallet Address"
          description={truncateAddress(address ?? '', 12)}
          left={(props) => <List.Icon {...props} icon="wallet" color={COLORS.cyan} />}
          titleStyle={styles.itemTitle}
          descriptionStyle={styles.itemDescription}
        />
      </Surface>

      {/* Security */}
      <Text variant="labelMedium" style={styles.sectionLabel}>
        SECURITY
      </Text>
      <Surface style={styles.section} elevation={1}>
        <List.Item
          title="Biometric Unlock"
          description="Use fingerprint or Face ID"
          left={(props) => <List.Icon {...props} icon="fingerprint" color={COLORS.cyan} />}
          right={() => (
            <Switch
              value={biometricEnabled}
              onValueChange={handleToggleBiometric}
              color={COLORS.cyan}
            />
          )}
          titleStyle={styles.itemTitle}
          descriptionStyle={styles.itemDescription}
        />
        <Divider style={styles.divider} />
        <List.Item
          title="Auto-Lock"
          description={currentAutoLock}
          left={(props) => <List.Icon {...props} icon="timer-outline" color={COLORS.cyan} />}
          right={(props) => <List.Icon {...props} icon="chevron-right" color="rgba(255,255,255,0.3)" />}
          onPress={() => {
            const nextIdx = (autoLockOptions.findIndex((o) => o.value === autoLockMinutes) + 1) % autoLockOptions.length;
            setAutoLockMinutes(autoLockOptions[nextIdx].value);
          }}
          titleStyle={styles.itemTitle}
          descriptionStyle={styles.itemDescription}
        />
      </Surface>

      {/* Preferences */}
      <Text variant="labelMedium" style={styles.sectionLabel}>
        PREFERENCES
      </Text>
      <Surface style={styles.section} elevation={1}>
        <List.Item
          title="Hide Balances"
          description="Mask balance values"
          left={(props) => <List.Icon {...props} icon="eye-off" color={COLORS.cyan} />}
          right={() => (
            <Switch
              value={hideBalances}
              onValueChange={setHideBalances}
              color={COLORS.cyan}
            />
          )}
          titleStyle={styles.itemTitle}
          descriptionStyle={styles.itemDescription}
        />
        <Divider style={styles.divider} />
        <List.Item
          title="Notifications"
          description="Transaction alerts"
          left={(props) => <List.Icon {...props} icon="bell-outline" color={COLORS.cyan} />}
          right={() => (
            <Switch
              value={notificationsEnabled}
              onValueChange={setNotificationsEnabled}
              color={COLORS.cyan}
            />
          )}
          titleStyle={styles.itemTitle}
          descriptionStyle={styles.itemDescription}
        />
      </Surface>

      {/* Network */}
      <Text variant="labelMedium" style={styles.sectionLabel}>
        NETWORK
      </Text>
      <Surface style={styles.section} elevation={1}>
        <List.Item
          title="Node Version"
          description={version || 'Unknown'}
          left={(props) => <List.Icon {...props} icon="server" color={COLORS.cyan} />}
          titleStyle={styles.itemTitle}
          descriptionStyle={styles.itemDescription}
        />
        <Divider style={styles.divider} />
        <List.Item
          title="Network"
          description={networkId || 'mainnet2026.1'}
          left={(props) => <List.Icon {...props} icon="earth" color={COLORS.cyan} />}
          titleStyle={styles.itemTitle}
          descriptionStyle={styles.itemDescription}
        />
      </Surface>

      {/* Wallet Keys & Secrets */}
      <Text variant="labelMedium" style={styles.sectionLabel}>
        WALLET KEYS
      </Text>
      <Surface style={styles.section} elevation={1}>
        {/* Seed Phrase */}
        <List.Item
          title="Seed Phrase"
          description={showMnemonic ? 'Tap to hide' : 'Reveal your 12-word recovery phrase'}
          left={(props) => <List.Icon {...props} icon="key-variant" color={COLORS.orange} />}
          right={(props) =>
            showMnemonic ? (
              <List.Icon {...props} icon="eye-off" color="rgba(255,255,255,0.3)" />
            ) : (
              <List.Icon {...props} icon="eye" color="rgba(255,255,255,0.3)" />
            )
          }
          onPress={showMnemonic ? handleHideSecrets : handleRevealMnemonic}
          titleStyle={styles.itemTitle}
          descriptionStyle={styles.itemDescription}
        />
        {showMnemonic && mnemonic && (
          <View style={styles.secretContainer}>
            <View style={styles.mnemonicGrid}>
              {mnemonic.split(' ').map((word, i) => (
                <View key={i} style={styles.mnemonicWord}>
                  <Text style={styles.mnemonicIndex}>{i + 1}</Text>
                  <Text style={styles.mnemonicText}>{word}</Text>
                </View>
              ))}
            </View>
            <TouchableOpacity
              style={styles.copyBtn}
              onPress={() => copyToClipboard(mnemonic, 'Seed phrase')}
            >
              <MaterialCommunityIcons name="content-copy" size={16} color={COLORS.cyan} />
              <Text style={styles.copyBtnText}>Copy Seed Phrase</Text>
            </TouchableOpacity>
          </View>
        )}

        <Divider style={styles.divider} />

        {/* Private Key */}
        <List.Item
          title="Private Key"
          description={showPrivateKey ? 'Tap to hide' : 'Reveal your Ed25519 private key'}
          left={(props) => <List.Icon {...props} icon="shield-key" color={COLORS.red} />}
          right={(props) =>
            showPrivateKey ? (
              <List.Icon {...props} icon="eye-off" color="rgba(255,255,255,0.3)" />
            ) : (
              <List.Icon {...props} icon="eye" color="rgba(255,255,255,0.3)" />
            )
          }
          onPress={showPrivateKey ? handleHideSecrets : handleRevealPrivateKey}
          titleStyle={styles.itemTitle}
          descriptionStyle={styles.itemDescription}
        />
        {showPrivateKey && privateKeyHex && (
          <View style={styles.secretContainer}>
            <Text style={styles.secretLabel}>Ed25519 Private Key</Text>
            <View style={styles.secretBox}>
              <Text style={styles.secretValue} selectable numberOfLines={2}>
                {privateKeyHex}
              </Text>
            </View>
            <TouchableOpacity
              style={styles.copyBtn}
              onPress={() => copyToClipboard(privateKeyHex, 'Private key')}
            >
              <MaterialCommunityIcons name="content-copy" size={16} color={COLORS.cyan} />
              <Text style={styles.copyBtnText}>Copy Private Key</Text>
            </TouchableOpacity>

            {publicKeyHex && (
              <>
                <Text style={[styles.secretLabel, { marginTop: 16 }]}>Ed25519 Public Key</Text>
                <View style={styles.secretBox}>
                  <Text style={styles.secretValue} selectable numberOfLines={2}>
                    {publicKeyHex}
                  </Text>
                </View>
                <TouchableOpacity
                  style={styles.copyBtn}
                  onPress={() => copyToClipboard(publicKeyHex, 'Public key')}
                >
                  <MaterialCommunityIcons name="content-copy" size={16} color={COLORS.cyan} />
                  <Text style={styles.copyBtnText}>Copy Public Key</Text>
                </TouchableOpacity>
              </>
            )}

            {pqAvailable && dilithiumPk && (
              <>
                <Text style={[styles.secretLabel, { marginTop: 16 }]}>Dilithium5 Public Key (PQ)</Text>
                <View style={styles.secretBox}>
                  <Text style={styles.secretValue} selectable numberOfLines={2}>
                    {dilithiumPk.substring(0, 64)}...
                  </Text>
                </View>
                <TouchableOpacity
                  style={styles.copyBtn}
                  onPress={() => copyToClipboard(dilithiumPk, 'Dilithium5 public key')}
                >
                  <MaterialCommunityIcons name="content-copy" size={16} color={COLORS.cyan} />
                  <Text style={styles.copyBtnText}>Copy Dilithium5 Key</Text>
                </TouchableOpacity>
              </>
            )}

            <View style={styles.warningBanner}>
              <MaterialCommunityIcons name="alert" size={18} color={COLORS.orange} />
              <Text style={styles.warningText}>
                Never share your private key. Anyone with it can access your funds.
              </Text>
            </View>
          </View>
        )}

        <Divider style={styles.divider} />

        {/* Wallet Address (full, copyable) */}
        <List.Item
          title="Full Address"
          description={address ?? 'Not available'}
          descriptionNumberOfLines={2}
          left={(props) => <List.Icon {...props} icon="wallet" color={COLORS.cyan} />}
          right={(props) => <List.Icon {...props} icon="content-copy" color="rgba(255,255,255,0.3)" />}
          onPress={() => address && copyToClipboard(address, 'Address')}
          titleStyle={styles.itemTitle}
          descriptionStyle={[styles.itemDescription, { fontFamily: 'monospace', fontSize: 11 }]}
        />
      </Surface>

      {/* Danger Zone */}
      <Text variant="labelMedium" style={styles.sectionLabel}>
        ACCOUNT
      </Text>
      <Surface style={styles.section} elevation={1}>
        <List.Item
          title="Logout"
          description="Remove wallet from device"
          left={(props) => <List.Icon {...props} icon="logout" color={COLORS.red} />}
          onPress={handleLogout}
          titleStyle={[styles.itemTitle, { color: COLORS.red }]}
          descriptionStyle={styles.itemDescription}
        />
      </Surface>

      <Text variant="bodySmall" style={styles.versionText}>
        Quillon Wallet v1.0.0
      </Text>

      <Snackbar visible={!!error} onDismiss={() => setError('')} duration={3000} style={{ backgroundColor: COLORS.red }}>
        {error}
      </Snackbar>
      <Snackbar visible={!!snackMsg} onDismiss={() => setSnackMsg('')} duration={2000}>
        {snackMsg}
      </Snackbar>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.darkBg,
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 40,
  },
  sectionLabel: {
    color: 'rgba(255, 255, 255, 0.4)',
    marginTop: 24,
    marginBottom: 8,
    marginLeft: 4,
    letterSpacing: 1,
  },
  section: {
    borderRadius: 16,
    backgroundColor: COLORS.cardBg,
    overflow: 'hidden',
  },
  itemTitle: {
    color: '#FFFFFF',
  },
  itemDescription: {
    color: 'rgba(255, 255, 255, 0.5)',
  },
  divider: {
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
  },
  secretContainer: {
    paddingHorizontal: 16,
    paddingBottom: 16,
  },
  mnemonicGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 12,
  },
  mnemonicWord: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 188, 212, 0.08)',
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderWidth: 1,
    borderColor: 'rgba(0, 188, 212, 0.15)',
    minWidth: '28%' as unknown as number,
  },
  mnemonicIndex: {
    color: 'rgba(255, 255, 255, 0.3)',
    fontSize: 11,
    marginRight: 6,
    minWidth: 16,
  },
  mnemonicText: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '500',
  },
  secretLabel: {
    color: 'rgba(255, 255, 255, 0.4)',
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 6,
  },
  secretBox: {
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    borderRadius: 8,
    padding: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    marginBottom: 8,
  },
  secretValue: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 12,
    fontFamily: 'monospace',
    lineHeight: 18,
  },
  copyBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    gap: 6,
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 8,
    backgroundColor: 'rgba(0, 188, 212, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(0, 188, 212, 0.2)',
  },
  copyBtnText: {
    color: COLORS.cyan,
    fontSize: 12,
    fontWeight: '600',
  },
  warningBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginTop: 16,
    padding: 12,
    borderRadius: 8,
    backgroundColor: 'rgba(255, 152, 0, 0.08)',
    borderWidth: 1,
    borderColor: 'rgba(255, 152, 0, 0.2)',
  },
  warningText: {
    color: 'rgba(255, 152, 0, 0.8)',
    fontSize: 12,
    flex: 1,
    lineHeight: 16,
  },
  versionText: {
    color: 'rgba(255, 255, 255, 0.2)',
    textAlign: 'center',
    marginTop: 24,
  },
});
