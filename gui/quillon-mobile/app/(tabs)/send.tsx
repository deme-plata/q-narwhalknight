import React, { useState, useCallback } from 'react';
import { View, StyleSheet, ScrollView, KeyboardAvoidingView, Platform } from 'react-native';
import { Text, TextInput, Button, IconButton, Chip, Snackbar } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Network from 'expo-network';
import { COLORS } from '../../src/theme';
import { useWalletStore } from '../../src/stores/walletStore';
import { validateAddress, parseQnkUri } from '../../src/utils/validateAddress';
import { formatBalance } from '../../src/utils/formatBalance';
import { getMnemonic } from '../../src/services/secureStorage';
import * as api from '../../src/services/api';

export default function SendScreen() {
  const address = useWalletStore((s) => s.address);
  const qugBalance = useWalletStore((s) => s.qugBalance);
  const refreshBalance = useWalletStore((s) => s.refreshBalance);

  const [recipient, setRecipient] = useState('');
  const [amount, setAmount] = useState('');
  const [token] = useState('QUG');
  const [scanning, setScanning] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [step, setStep] = useState<'input' | 'review'>('input');

  const [permission, requestPermission] = useCameraPermissions();

  const recipientValidation = recipient ? validateAddress(recipient) : { valid: false };
  const amountNum = parseFloat(amount) || 0;
  const isValidAmount = amountNum > 0 && amountNum <= qugBalance;
  const canProceed = recipientValidation.valid && isValidAmount;

  const handleScan = useCallback(() => {
    if (!permission?.granted) {
      requestPermission();
      return;
    }
    setScanning(true);
  }, [permission, requestPermission]);

  const handleBarCodeScanned = useCallback((result: { data: string }) => {
    setScanning(false);
    const parsed = parseQnkUri(result.data);
    if (parsed) {
      setRecipient(parsed.address);
      if (parsed.amount) setAmount(parsed.amount);
    } else if (validateAddress(result.data).valid) {
      setRecipient(result.data);
    } else {
      setError('Invalid QR code - not a valid QNK address');
    }
  }, []);

  const handleMaxAmount = () => {
    const maxAmount = Math.max(0, qugBalance - 0.001);
    setAmount(maxAmount.toFixed(8));
  };

  const handleSend = async () => {
    if (!address || !canProceed) return;

    try {
      setSending(true);
      setError('');

      const networkState = await Network.getNetworkStateAsync();
      if (!networkState.isConnected || !networkState.isInternetReachable) {
        throw new Error('No network connection. Please connect to the internet to send transactions.');
      }

      // Mnemonic is used for client-derived signing; OAuth users don't have one
      // (the server authenticates via the Bearer token and signs server-side)
      const mnemonic = await getMnemonic() ?? '';

      const result = await api.transfer({
        from: address,
        to: recipient,
        amount: amount,
        token,
        mnemonic,
      });

      setSuccess(`Transaction sent! Hash: ${result.tx_hash.slice(0, 16)}...`);
      setRecipient('');
      setAmount('');
      setStep('input');

      setTimeout(() => refreshBalance(), 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send transaction');
    } finally {
      setSending(false);
    }
  };

  if (scanning) {
    return (
      <SafeAreaView style={styles.container} edges={['top']}>
        <View style={styles.scannerContainer}>
          <CameraView
            style={styles.camera}
            onBarcodeScanned={handleBarCodeScanned}
            barcodeScannerSettings={{ barcodeTypes: ['qr'] }}
          />
          <View style={styles.scannerOverlay}>
            <View style={styles.scannerFrame} />
            <Text variant="bodyMedium" style={styles.scannerText}>
              Scan a QNK address QR code
            </Text>
          </View>
          <Button
            mode="contained"
            onPress={() => setScanning(false)}
            style={styles.cancelScanButton}
            buttonColor={COLORS.surfaceBg}
          >
            Cancel
          </Button>
        </View>
      </SafeAreaView>
    );
  }

  if (step === 'review') {
    return (
      <SafeAreaView style={styles.container} edges={['top']}>
        <ScrollView contentContainerStyle={styles.scrollContent}>
          <Text variant="headlineSmall" style={styles.title}>
            Review Transaction
          </Text>

          <View style={styles.reviewCardOuter}>
            <LinearGradient
              colors={['rgba(0, 188, 212, 0.08)', 'rgba(20, 20, 32, 0.95)']}
              start={{ x: 0, y: 0 }}
              end={{ x: 0, y: 1 }}
              style={styles.reviewCard}
            >
              {/* Top accent */}
              <LinearGradient
                colors={[COLORS.cyan, COLORS.purple]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
                style={styles.reviewAccent}
              />

              <View style={styles.reviewRow}>
                <Text style={styles.reviewLabel}>To</Text>
                <Text style={styles.reviewValue} numberOfLines={1}>
                  {recipient.slice(0, 20)}...{recipient.slice(-8)}
                </Text>
              </View>
              <View style={styles.reviewDivider} />
              <View style={styles.reviewRow}>
                <Text style={styles.reviewLabel}>Amount</Text>
                <Text style={styles.reviewAmount}>
                  {formatBalance(amountNum, token, 0)}
                </Text>
              </View>
              <View style={styles.reviewDivider} />
              <View style={styles.reviewRow}>
                <Text style={styles.reviewLabel}>Network Fee</Text>
                <Text style={styles.reviewValue}>~0.001 QUG</Text>
              </View>
              <View style={styles.reviewDivider} />
              <View style={styles.reviewRow}>
                <Text style={styles.reviewLabel}>Security</Text>
                <View style={styles.pqRow}>
                  <MaterialCommunityIcons name="shield-check" size={14} color={COLORS.green} />
                  <Text style={styles.pqText}>Hybrid PQ</Text>
                </View>
              </View>
              <View style={styles.reviewDivider} />
              <View style={styles.reviewRow}>
                <Text style={styles.reviewLabel}>Total</Text>
                <Text style={styles.reviewTotal}>
                  {formatBalance(amountNum + 0.001, token, 0)}
                </Text>
              </View>
            </LinearGradient>
          </View>

          <Button
            mode="contained"
            onPress={handleSend}
            loading={sending}
            disabled={sending}
            style={styles.confirmButton}
            buttonColor={COLORS.cyan}
            textColor="#000000"
            icon="send"
            labelStyle={{ fontWeight: '700' }}
          >
            Confirm & Send
          </Button>

          <Button
            mode="text"
            onPress={() => setStep('input')}
            textColor={COLORS.cyan}
            disabled={sending}
          >
            Go Back
          </Button>
        </ScrollView>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <ScrollView contentContainerStyle={styles.scrollContent}>
          <Text variant="headlineSmall" style={styles.title}>
            Send {token}
          </Text>

          {/* Recipient */}
          <Text variant="labelMedium" style={styles.fieldLabel}>
            Recipient Address
          </Text>
          <View style={styles.inputRow}>
            <TextInput
              value={recipient}
              onChangeText={(text) => {
                setRecipient(text.trim().toLowerCase());
                setError('');
              }}
              mode="outlined"
              placeholder="qnk..."
              placeholderTextColor="rgba(255, 255, 255, 0.3)"
              style={styles.addressInput}
              outlineColor="rgba(0, 188, 212, 0.15)"
              activeOutlineColor={COLORS.cyan}
              textColor="#FFFFFF"
              autoCapitalize="none"
              autoCorrect={false}
              error={!!recipient && !recipientValidation.valid}
            />
            <IconButton
              icon="qrcode-scan"
              size={28}
              iconColor={COLORS.cyan}
              onPress={handleScan}
              style={styles.scanButton}
            />
          </View>
          {recipient && !recipientValidation.valid && (
            <Text variant="bodySmall" style={styles.fieldError}>
              {recipientValidation.error}
            </Text>
          )}

          {/* Amount */}
          <Text variant="labelMedium" style={[styles.fieldLabel, { marginTop: 20 }]}>
            Amount
          </Text>
          <View style={styles.amountRow}>
            <TextInput
              value={amount}
              onChangeText={(text) => {
                if (text === '' || /^\d*\.?\d*$/.test(text)) {
                  setAmount(text);
                }
              }}
              mode="outlined"
              placeholder="0.00"
              placeholderTextColor="rgba(255, 255, 255, 0.3)"
              style={styles.amountInput}
              outlineColor="rgba(0, 188, 212, 0.15)"
              activeOutlineColor={COLORS.cyan}
              textColor="#FFFFFF"
              keyboardType="decimal-pad"
              right={<TextInput.Affix text={token} textStyle={{ color: COLORS.cyan }} />}
            />
            <Chip
              compact
              onPress={handleMaxAmount}
              style={styles.maxChip}
              textStyle={styles.maxChipText}
            >
              MAX
            </Chip>
          </View>
          <Text variant="bodySmall" style={styles.balanceHint}>
            Available: {formatBalance(qugBalance, token, 0)}
          </Text>
          {amountNum > qugBalance && (
            <Text variant="bodySmall" style={styles.fieldError}>
              Insufficient balance
            </Text>
          )}

          {/* Send Button */}
          <Button
            mode="contained"
            onPress={() => setStep('review')}
            disabled={!canProceed}
            style={styles.sendButton}
            buttonColor={COLORS.cyan}
            textColor="#000000"
            icon="arrow-up-circle"
            contentStyle={styles.buttonContent}
            labelStyle={{ fontWeight: '700' }}
          >
            Review Transaction
          </Button>
        </ScrollView>
      </KeyboardAvoidingView>

      <Snackbar visible={!!error} onDismiss={() => setError('')} duration={4000} style={styles.errorSnackbar}>
        {error}
      </Snackbar>
      <Snackbar visible={!!success} onDismiss={() => setSuccess('')} duration={4000} style={styles.successSnackbar}>
        {success}
      </Snackbar>
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
  },
  title: {
    color: '#FFFFFF',
    fontWeight: '700',
    marginBottom: 24,
  },
  fieldLabel: {
    color: 'rgba(255, 255, 255, 0.6)',
    marginBottom: 8,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  addressInput: {
    flex: 1,
    backgroundColor: COLORS.surfaceBg,
    fontSize: 14,
  },
  scanButton: {
    marginTop: 0,
  },
  amountRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  amountInput: {
    flex: 1,
    backgroundColor: COLORS.surfaceBg,
    fontSize: 18,
  },
  maxChip: {
    backgroundColor: 'rgba(0, 188, 212, 0.15)',
    marginTop: 6,
  },
  maxChipText: {
    color: COLORS.cyan,
    fontWeight: '700',
    fontSize: 12,
  },
  balanceHint: {
    color: 'rgba(255, 255, 255, 0.4)',
    marginTop: 6,
  },
  fieldError: {
    color: COLORS.red,
    marginTop: 4,
  },
  sendButton: {
    marginTop: 32,
    borderRadius: 14,
    elevation: 4,
    shadowColor: COLORS.cyan,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  buttonContent: {
    paddingVertical: 6,
  },
  // Scanner
  scannerContainer: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  scannerOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scannerFrame: {
    width: 250,
    height: 250,
    borderWidth: 3,
    borderColor: COLORS.cyan,
    borderRadius: 24,
    backgroundColor: 'transparent',
  },
  scannerText: {
    color: '#FFFFFF',
    marginTop: 20,
    textAlign: 'center',
  },
  cancelScanButton: {
    position: 'absolute',
    bottom: 40,
    alignSelf: 'center',
    borderRadius: 14,
  },
  // Review
  reviewCardOuter: {
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(0, 188, 212, 0.12)',
    overflow: 'hidden',
    marginBottom: 24,
  },
  reviewCard: {
    padding: 16,
  },
  reviewAccent: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 2,
  },
  reviewRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
  },
  reviewLabel: {
    color: 'rgba(255, 255, 255, 0.5)',
    fontSize: 14,
  },
  reviewValue: {
    color: '#FFFFFF',
    fontSize: 14,
    maxWidth: '60%',
  },
  reviewAmount: {
    color: COLORS.cyan,
    fontSize: 18,
    fontWeight: '700',
  },
  reviewTotal: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },
  reviewDivider: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
  },
  pqRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  pqText: {
    color: COLORS.green,
    fontSize: 13,
    fontWeight: '600',
  },
  confirmButton: {
    borderRadius: 14,
    marginBottom: 12,
    elevation: 4,
    shadowColor: COLORS.cyan,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  errorSnackbar: {
    backgroundColor: COLORS.red,
  },
  successSnackbar: {
    backgroundColor: COLORS.green,
  },
});
