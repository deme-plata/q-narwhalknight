import React, { useState, useRef, useEffect } from 'react';
import { View, StyleSheet, Alert, Linking } from 'react-native';
import { Text, Button, Surface, ActivityIndicator } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { router } from 'expo-router';
import { COLORS } from '../src/theme';
import { useWalletStore } from '../src/stores/walletStore';

/**
 * v10.1.9: QR Scanner for linking standalone miner to mobile wallet.
 *
 * Flow:
 * 1. Miner displays QR code with: quillon://miner-login?code=<device_code>&server=<url>
 * 2. Mobile app scans QR → extracts device_code + server URL
 * 3. App sends POST /api/v1/miner/device-login/complete with wallet address
 * 4. Miner polls and receives wallet → starts mining to that wallet
 */
export default function MinerScanScreen() {
  const address = useWalletStore((s) => s.address);
  const [permission, requestPermission] = useCameraPermissions();
  const [scanned, setScanned] = useState(false);
  const [linking, setLinking] = useState(false);
  const [result, setResult] = useState<'success' | 'error' | null>(null);
  const [errorMsg, setErrorMsg] = useState('');
  const hasProcessed = useRef(false);

  // Handle deep links (if app was opened via quillon:// URL)
  useEffect(() => {
    const handleDeepLink = (event: { url: string }) => {
      processQRData(event.url);
    };
    const sub = Linking.addEventListener('url', handleDeepLink);
    // Check if app was opened with a URL
    Linking.getInitialURL().then((url) => {
      if (url) processQRData(url);
    });
    return () => sub.remove();
  }, []);

  async function processQRData(data: string) {
    if (hasProcessed.current) return;
    hasProcessed.current = true;
    setScanned(true);

    // Parse quillon://miner-login?code=<code>&server=<url>
    let deviceCode: string | null = null;
    let serverUrl: string | null = null;

    try {
      // Handle both URL formats
      const url = new URL(data);
      if (url.protocol === 'quillon:' && url.hostname === 'miner-login') {
        deviceCode = url.searchParams.get('code');
        serverUrl = url.searchParams.get('server');
      } else if (data.includes('miner-login') && data.includes('code=')) {
        // Fallback: parse query params from any URL format
        const params = new URLSearchParams(data.split('?')[1] || '');
        deviceCode = params.get('code');
        serverUrl = params.get('server');
      }
    } catch {
      // Try manual parsing for non-standard URLs
      const codeMatch = data.match(/code=([^&]+)/);
      const serverMatch = data.match(/server=([^&]+)/);
      if (codeMatch) deviceCode = codeMatch[1];
      if (serverMatch) serverUrl = decodeURIComponent(serverMatch[1]);
    }

    if (!deviceCode) {
      setResult('error');
      setErrorMsg('Invalid QR code. Please scan the QR code shown on your miner.');
      return;
    }

    if (!address) {
      setResult('error');
      setErrorMsg('No wallet address found. Please log in first.');
      return;
    }

    // Complete the device login
    setLinking(true);
    try {
      // Use the server URL from QR code, or fallback to default
      const baseUrl = serverUrl || 'https://quillon.xyz';
      const resp = await fetch(`${baseUrl}/api/v1/miner/device-login/complete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          device_code: deviceCode,
          wallet_address: address,
        }),
      });

      const json = await resp.json();
      if (json.data?.status === 'complete' || json.data?.message) {
        setResult('success');
      } else {
        setResult('error');
        setErrorMsg(json.error || json.message || 'Failed to link miner. Code may have expired.');
      }
    } catch (err) {
      setResult('error');
      setErrorMsg(
        err instanceof Error
          ? `Connection failed: ${err.message}`
          : 'Could not reach the server. Check your connection.'
      );
    } finally {
      setLinking(false);
    }
  }

  // Permission not yet determined
  if (!permission) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color={COLORS.cyan} />
      </View>
    );
  }

  // Permission denied
  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Surface style={styles.card} elevation={2}>
          <MaterialCommunityIcons name="camera-off" size={48} color={COLORS.orange} />
          <Text variant="titleMedium" style={styles.title}>
            Camera Permission Required
          </Text>
          <Text variant="bodyMedium" style={styles.subtitle}>
            Allow camera access to scan QR codes from your miner.
          </Text>
          <Button
            mode="contained"
            onPress={requestPermission}
            style={styles.button}
            buttonColor={COLORS.cyan}
            textColor="#000"
          >
            Allow Camera
          </Button>
        </Surface>
      </View>
    );
  }

  // Success state
  if (result === 'success') {
    return (
      <View style={styles.container}>
        <Surface style={styles.card} elevation={2}>
          <MaterialCommunityIcons name="check-circle" size={64} color={COLORS.green} />
          <Text variant="headlineSmall" style={[styles.title, { color: COLORS.green }]}>
            Miner Linked!
          </Text>
          <Text variant="bodyMedium" style={styles.subtitle}>
            Your miner is now connected to this wallet. Mining rewards will be sent here automatically.
          </Text>
          <Text variant="bodySmall" style={styles.addressText} numberOfLines={1}>
            {address}
          </Text>
          <Button
            mode="contained"
            onPress={() => router.back()}
            style={styles.button}
            buttonColor={COLORS.cyan}
            textColor="#000"
          >
            Done
          </Button>
        </Surface>
      </View>
    );
  }

  // Error state
  if (result === 'error') {
    return (
      <View style={styles.container}>
        <Surface style={styles.card} elevation={2}>
          <MaterialCommunityIcons name="alert-circle" size={64} color={COLORS.orange} />
          <Text variant="titleMedium" style={[styles.title, { color: COLORS.orange }]}>
            Link Failed
          </Text>
          <Text variant="bodyMedium" style={styles.subtitle}>
            {errorMsg}
          </Text>
          <Button
            mode="contained"
            onPress={() => {
              setScanned(false);
              setResult(null);
              setErrorMsg('');
              hasProcessed.current = false;
            }}
            style={styles.button}
            buttonColor={COLORS.cyan}
            textColor="#000"
          >
            Try Again
          </Button>
          <Button
            mode="text"
            onPress={() => router.back()}
            textColor="rgba(255,255,255,0.6)"
            style={{ marginTop: 8 }}
          >
            Cancel
          </Button>
        </Surface>
      </View>
    );
  }

  // Linking in progress
  if (linking) {
    return (
      <View style={styles.container}>
        <Surface style={styles.card} elevation={2}>
          <ActivityIndicator size="large" color={COLORS.cyan} />
          <Text variant="titleMedium" style={styles.title}>
            Linking Miner...
          </Text>
          <Text variant="bodyMedium" style={styles.subtitle}>
            Sending wallet address to your miner node.
          </Text>
        </Surface>
      </View>
    );
  }

  // Camera scanner
  return (
    <View style={styles.container}>
      <CameraView
        style={StyleSheet.absoluteFillObject}
        facing="back"
        barcodeScannerSettings={{
          barcodeTypes: ['qr'],
        }}
        onBarcodeScanned={scanned ? undefined : (result) => {
          if (result.data) processQRData(result.data);
        }}
      />
      {/* Overlay with scan frame */}
      <View style={styles.overlay}>
        <View style={styles.overlayTop} />
        <View style={styles.overlayMiddle}>
          <View style={styles.overlaySide} />
          <View style={styles.scanFrame}>
            {/* Corner markers */}
            <View style={[styles.corner, styles.cornerTL]} />
            <View style={[styles.corner, styles.cornerTR]} />
            <View style={[styles.corner, styles.cornerBL]} />
            <View style={[styles.corner, styles.cornerBR]} />
          </View>
          <View style={styles.overlaySide} />
        </View>
        <View style={styles.overlayBottom}>
          <Text variant="titleMedium" style={styles.scanTitle}>
            Scan Miner QR Code
          </Text>
          <Text variant="bodySmall" style={styles.scanSubtitle}>
            Point your camera at the QR code displayed on your standalone miner
          </Text>
          <Button
            mode="text"
            onPress={() => router.back()}
            textColor="rgba(255,255,255,0.6)"
            style={{ marginTop: 16 }}
          >
            Cancel
          </Button>
        </View>
      </View>
    </View>
  );
}

const SCAN_SIZE = 260;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.darkBg,
    justifyContent: 'center',
    alignItems: 'center',
  },
  card: {
    backgroundColor: COLORS.cardBg,
    borderRadius: 24,
    padding: 32,
    alignItems: 'center',
    gap: 16,
    marginHorizontal: 24,
    maxWidth: 400,
  },
  title: {
    color: '#FFFFFF',
    fontWeight: '700',
    textAlign: 'center',
  },
  subtitle: {
    color: 'rgba(255,255,255,0.5)',
    textAlign: 'center',
    lineHeight: 22,
  },
  addressText: {
    color: COLORS.cyan,
    fontFamily: 'monospace',
    fontSize: 11,
    textAlign: 'center',
    paddingHorizontal: 16,
  },
  button: {
    borderRadius: 12,
    marginTop: 8,
    paddingHorizontal: 24,
  },
  // Camera overlay
  overlay: {
    ...StyleSheet.absoluteFillObject,
  },
  overlayTop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.6)',
  },
  overlayMiddle: {
    flexDirection: 'row',
    height: SCAN_SIZE,
  },
  overlaySide: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.6)',
  },
  scanFrame: {
    width: SCAN_SIZE,
    height: SCAN_SIZE,
  },
  overlayBottom: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.6)',
    alignItems: 'center',
    paddingTop: 32,
  },
  scanTitle: {
    color: '#FFFFFF',
    fontWeight: '700',
  },
  scanSubtitle: {
    color: 'rgba(255,255,255,0.5)',
    textAlign: 'center',
    marginTop: 8,
    paddingHorizontal: 32,
  },
  // Corner markers
  corner: {
    position: 'absolute',
    width: 28,
    height: 28,
    borderColor: COLORS.cyan,
  },
  cornerTL: {
    top: 0,
    left: 0,
    borderTopWidth: 3,
    borderLeftWidth: 3,
    borderTopLeftRadius: 4,
  },
  cornerTR: {
    top: 0,
    right: 0,
    borderTopWidth: 3,
    borderRightWidth: 3,
    borderTopRightRadius: 4,
  },
  cornerBL: {
    bottom: 0,
    left: 0,
    borderBottomWidth: 3,
    borderLeftWidth: 3,
    borderBottomLeftRadius: 4,
  },
  cornerBR: {
    bottom: 0,
    right: 0,
    borderBottomWidth: 3,
    borderRightWidth: 3,
    borderBottomRightRadius: 4,
  },
});
