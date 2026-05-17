import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  Animated,
  ActivityIndicator,
} from 'react-native';
import { Text, Button, TextInput, Snackbar } from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { router } from 'expo-router';
import { COLORS, GRADIENTS } from '../../src/theme';
import { useWalletStore } from '../../src/stores/walletStore';
import { useNetworkStore } from '../../src/stores/networkStore';
import { runOAuthFlow } from '../../src/services/oauth';

// ============================================================================
// Pulsing Logo Component
// ============================================================================

function PulsingLogo() {
  const pulse = useRef(new Animated.Value(1)).current;
  const glow = useRef(new Animated.Value(0.3)).current;

  useEffect(() => {
    const pulseAnim = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, { toValue: 1.08, duration: 2000, useNativeDriver: true }),
        Animated.timing(pulse, { toValue: 1, duration: 2000, useNativeDriver: true }),
      ])
    );
    const glowAnim = Animated.loop(
      Animated.sequence([
        Animated.timing(glow, { toValue: 0.6, duration: 2000, useNativeDriver: true }),
        Animated.timing(glow, { toValue: 0.3, duration: 2000, useNativeDriver: true }),
      ])
    );
    pulseAnim.start();
    glowAnim.start();
    return () => { pulseAnim.stop(); glowAnim.stop(); };
  }, [pulse, glow]);

  return (
    <View style={styles.logoOuter}>
      <Animated.View style={[styles.logoGlow, { opacity: glow, transform: [{ scale: pulse }] }]} />
      <LinearGradient
        colors={['rgba(0, 188, 212, 0.15)', 'rgba(124, 77, 255, 0.1)']}
        style={styles.logoContainer}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
      >
        <MaterialCommunityIcons name="atom-variant" size={72} color={COLORS.cyan} />
      </LinearGradient>
    </View>
  );
}

// ============================================================================
// Login Screen
// ============================================================================

type ScreenMode = 'main' | 'import';

export default function LoginScreen() {
  const [mode, setMode] = useState<ScreenMode>('main');
  const [mnemonic, setMnemonic] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [oauthBusy, setOauthBusy] = useState(false);

  const createWallet = useWalletStore((s) => s.createWallet);
  const importWallet = useWalletStore((s) => s.importWallet);
  const loginWithOAuth = useWalletStore((s) => s.loginWithOAuth);
  const checkHealth = useNetworkStore((s) => s.checkHealth);

  // ---- Create Wallet ----
  const handleCreateWallet = async () => {
    setLoading(true);
    try {
      const newMnemonic = await createWallet();
      router.replace({
        pathname: '/(auth)/backup',
        params: { mnemonic: newMnemonic },
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create wallet');
    } finally {
      setLoading(false);
    }
  };

  // ---- Import Wallet ----
  const handleImportWallet = async () => {
    const trimmed = mnemonic.trim().toLowerCase();
    const words = trimmed.split(/\s+/);

    if (words.length !== 12) {
      setError('Please enter a 12 word mnemonic phrase');
      return;
    }

    setLoading(true);
    try {
      await importWallet(trimmed);
      router.replace('/(tabs)');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Invalid mnemonic phrase');
    } finally {
      setLoading(false);
    }
  };

  // ---- OAuth2 Server Login ----
  const handleVaultLogin = async () => {
    setOauthBusy(true);
    setError('');

    try {
      const result = await runOAuthFlow();

      if (result.success) {
        console.log('[OAUTH] Authenticated!', result.walletInfo?.walletAddress ?? '');
        await loginWithOAuth(result.tokens, result.walletInfo);
        // Immediately verify connectivity so the dashboard shows Online
        checkHealth();
        router.replace('/(tabs)');
      } else {
        if (result.error.code !== 'cancelled') {
          setError(result.error.message);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'OAuth flow failed');
    } finally {
      setOauthBusy(false);
    }
  };

  // ---- Import View ----
  if (mode === 'import') {
    return (
      <KeyboardAvoidingView
        style={styles.container}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        <LinearGradient colors={GRADIENTS.background} style={styles.gradientBg}>
          <ScrollView contentContainerStyle={styles.scrollContent}>
            <View style={styles.importIconRow}>
              <LinearGradient
                colors={['rgba(0, 188, 212, 0.15)', 'rgba(124, 77, 255, 0.1)']}
                style={styles.importIconCircle}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
              >
                <MaterialCommunityIcons name="key-variant" size={40} color={COLORS.cyan} />
              </LinearGradient>
            </View>

            <Text variant="headlineMedium" style={styles.title}>
              Import Wallet
            </Text>

            <Text variant="bodyMedium" style={styles.subtitle}>
              Enter your 12 word recovery phrase
            </Text>

            <TextInput
              label="Seed Phrase"
              value={mnemonic}
              onChangeText={(text) => { setMnemonic(text); setError(''); }}
              mode="outlined"
              multiline
              numberOfLines={4}
              style={styles.seedInput}
              outlineColor="rgba(0, 188, 212, 0.2)"
              activeOutlineColor={COLORS.cyan}
              textColor="#FFFFFF"
              placeholder="word1 word2 word3 ..."
              placeholderTextColor="rgba(255, 255, 255, 0.25)"
              autoCapitalize="none"
              autoCorrect={false}
            />

            <GradientButton
              label="Import Wallet"
              icon="import"
              onPress={handleImportWallet}
              loading={loading}
              disabled={loading || mnemonic.trim().length === 0}
            />

            <Button
              mode="text"
              onPress={() => { setMode('main'); setMnemonic(''); setError(''); }}
              textColor={COLORS.cyan}
              style={styles.backButton}
            >
              Back
            </Button>
          </ScrollView>
        </LinearGradient>

        <Snackbar visible={!!error} onDismiss={() => setError('')} duration={3000} style={styles.snackbar}>
          {error}
        </Snackbar>
      </KeyboardAvoidingView>
    );
  }

  // ---- Main Login View ----
  return (
    <View style={styles.container}>
      <LinearGradient colors={GRADIENTS.background} style={styles.gradientBg}>
        {/* Hero Glow */}
        <LinearGradient
          colors={GRADIENTS.heroGlow}
          style={styles.heroGlow}
          start={{ x: 0.5, y: 0 }}
          end={{ x: 0.5, y: 1 }}
        />

        <View style={styles.heroSection}>
          <PulsingLogo />

          <Text variant="headlineLarge" style={styles.heroTitle}>
            Quillon
          </Text>

          <Text variant="bodyLarge" style={styles.heroSubtitle}>
            Quantum-Resistant Blockchain Wallet
          </Text>

          <View style={styles.pqBadge}>
            <MaterialCommunityIcons name="shield-check" size={14} color={COLORS.green} />
            <Text style={styles.pqBadgeText}>
              Dilithium5 + Kyber1024 (NIST Level 5)
            </Text>
          </View>
        </View>

        <View style={styles.buttonsSection}>
          <GradientButton
            label="Create New Wallet"
            icon="plus-circle"
            onPress={handleCreateWallet}
            loading={loading}
            disabled={loading || oauthBusy}
          />

          <Button
            mode="outlined"
            onPress={() => setMode('import')}
            style={styles.secondaryButton}
            textColor={COLORS.cyan}
            icon="import"
            contentStyle={styles.buttonContent}
            disabled={loading || oauthBusy}
          >
            Import Mnemonic
          </Button>

          <Button
            mode="outlined"
            onPress={handleVaultLogin}
            style={styles.tertiaryButton}
            textColor={COLORS.purple}
            icon={oauthBusy ? undefined : 'shield-lock'}
            contentStyle={styles.buttonContent}
            disabled={loading || oauthBusy}
          >
            {oauthBusy ? 'Authenticating...' : 'Server Vault (OAuth2)'}
          </Button>

          {oauthBusy && (
            <View style={styles.oauthSpinner}>
              <ActivityIndicator size="small" color={COLORS.purple} />
              <Text style={styles.oauthSpinnerText}>
                Opening secure browser...
              </Text>
            </View>
          )}
        </View>

        <Text variant="bodySmall" style={styles.footer}>
          Powered by DAG-Knight Consensus
        </Text>
      </LinearGradient>

      <Snackbar visible={!!error} onDismiss={() => setError('')} duration={3000} style={styles.snackbar}>
        {error}
      </Snackbar>
    </View>
  );
}

// ============================================================================
// Reusable Gradient Button
// ============================================================================

function GradientButton({
  label, icon, onPress, loading, disabled,
}: {
  label: string; icon: string; onPress: () => void; loading?: boolean; disabled?: boolean;
}) {
  return (
    <Button
      mode="contained"
      onPress={onPress}
      loading={loading}
      disabled={disabled}
      style={styles.primaryButton}
      buttonColor={COLORS.cyan}
      textColor="#000000"
      icon={icon}
      contentStyle={styles.buttonContent}
      labelStyle={{ fontWeight: '700' }}
    >
      {label}
    </Button>
  );
}

// ============================================================================
// Styles
// ============================================================================

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.darkBg,
  },
  gradientBg: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    padding: 24,
    justifyContent: 'center',
  },
  heroGlow: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: '60%',
  },
  heroSection: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 24,
  },
  logoOuter: {
    width: 140,
    height: 140,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 24,
  },
  logoGlow: {
    position: 'absolute',
    width: 160,
    height: 160,
    borderRadius: 80,
    backgroundColor: COLORS.cyan,
  },
  logoContainer: {
    width: 120,
    height: 120,
    borderRadius: 60,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: 'rgba(0, 188, 212, 0.3)',
  },
  heroTitle: {
    color: '#FFFFFF',
    fontWeight: '700',
    letterSpacing: 3,
    fontSize: 36,
  },
  heroSubtitle: {
    color: 'rgba(255, 255, 255, 0.5)',
    marginTop: 8,
    textAlign: 'center',
  },
  pqBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: 16,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    backgroundColor: 'rgba(76, 175, 80, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(76, 175, 80, 0.2)',
  },
  pqBadgeText: {
    color: COLORS.green,
    fontSize: 11,
    fontWeight: '600',
    letterSpacing: 0.5,
  },
  importIconRow: {
    alignItems: 'center',
    marginBottom: 16,
  },
  importIconCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(0, 188, 212, 0.2)',
  },
  title: {
    color: '#FFFFFF',
    fontWeight: '700',
    textAlign: 'center',
    marginBottom: 8,
  },
  subtitle: {
    color: 'rgba(255, 255, 255, 0.5)',
    textAlign: 'center',
    marginBottom: 24,
  },
  seedInput: {
    backgroundColor: 'rgba(26, 26, 38, 0.8)',
    marginBottom: 20,
    minHeight: 100,
  },
  buttonsSection: {
    paddingHorizontal: 24,
    paddingBottom: 32,
    gap: 12,
  },
  primaryButton: {
    borderRadius: 14,
    elevation: 4,
    shadowColor: COLORS.cyan,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  secondaryButton: {
    borderRadius: 14,
    borderColor: 'rgba(0, 188, 212, 0.4)',
    borderWidth: 1.5,
  },
  tertiaryButton: {
    borderRadius: 14,
    borderColor: 'rgba(124, 77, 255, 0.3)',
    borderWidth: 1.5,
  },
  backButton: {
    marginTop: 8,
  },
  buttonContent: {
    paddingVertical: 6,
  },
  oauthSpinner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingTop: 4,
  },
  oauthSpinnerText: {
    color: 'rgba(255, 255, 255, 0.4)',
    fontSize: 12,
  },
  footer: {
    color: 'rgba(255, 255, 255, 0.25)',
    textAlign: 'center',
    paddingBottom: 24,
    letterSpacing: 1,
  },
  snackbar: {
    backgroundColor: COLORS.red,
  },
});
