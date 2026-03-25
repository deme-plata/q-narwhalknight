import React, { useState, useEffect, useCallback } from 'react';
import { View, StyleSheet, Vibration } from 'react-native';
import { Text, Button, TextInput } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { COLORS, GRADIENTS } from '../theme';
import { authenticateWithBiometrics, verifyPin, checkBiometrics, getPinLockoutRemaining } from '../services/auth';
import { isBiometricEnabled } from '../services/secureStorage';

interface BiometricGateProps {
  onUnlocked: () => void;
}

export function BiometricGate({ onUnlocked }: BiometricGateProps) {
  const [pin, setPin] = useState('');
  const [error, setError] = useState('');
  const [lockoutMs, setLockoutMs] = useState(0);
  const [showPinInput, setShowPinInput] = useState(false);
  const [bioAvailable, setBioAvailable] = useState(false);

  useEffect(() => {
    checkBiometricAvailability();
  }, []);

  // Countdown timer for lockout
  useEffect(() => {
    if (lockoutMs <= 0) return;
    const timer = setInterval(() => {
      setLockoutMs((prev) => {
        const next = prev - 1000;
        if (next <= 0) {
          setError('');
          return 0;
        }
        return next;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [lockoutMs > 0]);

  const checkBiometricAvailability = async () => {
    // Check existing lockout
    const remaining = await getPinLockoutRemaining();
    if (remaining > 0) {
      setLockoutMs(remaining);
      setError(`Locked out. Try again in ${Math.ceil(remaining / 1000)}s`);
      setShowPinInput(true);
      return;
    }

    const enabled = await isBiometricEnabled();
    const { available, enrolled } = await checkBiometrics();
    const canUseBio = enabled && available && enrolled;
    setBioAvailable(canUseBio);

    if (canUseBio) {
      attemptBiometric();
    } else {
      setShowPinInput(true);
    }
  };

  const attemptBiometric = async () => {
    try {
      const success = await authenticateWithBiometrics('Unlock Quillon Wallet');
      if (success) {
        onUnlocked();
      } else {
        setShowPinInput(true);
      }
    } catch {
      setShowPinInput(true);
    }
  };

  const handlePinSubmit = useCallback(async () => {
    if (pin.length < 4) {
      setError('PIN must be at least 4 digits');
      return;
    }

    if (lockoutMs > 0) {
      setError(`Locked out. Try again in ${Math.ceil(lockoutMs / 1000)}s`);
      return;
    }

    const result = await verifyPin(pin);
    if (result.success) {
      setError('');
      onUnlocked();
    } else {
      setPin('');
      Vibration.vibrate(200);

      if (result.lockoutMs > 0) {
        setLockoutMs(result.lockoutMs);
        setError(`Too many attempts. Locked for ${Math.ceil(result.lockoutMs / 1000)}s`);
      } else {
        setError('Incorrect PIN');
      }
    }
  }, [pin, lockoutMs, onUnlocked]);

  const isLocked = lockoutMs > 0;

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={GRADIENTS.background}
        style={StyleSheet.absoluteFillObject}
      />

      <View style={styles.card}>
        {/* Gradient top accent */}
        <LinearGradient
          colors={[COLORS.cyan, COLORS.purple]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
          style={styles.cardAccent}
        />

        <View style={styles.iconContainer}>
          <LinearGradient
            colors={['rgba(0, 188, 212, 0.15)', 'rgba(124, 77, 255, 0.1)']}
            style={styles.iconCircle}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
          >
            <MaterialCommunityIcons
              name="shield-lock"
              size={48}
              color={COLORS.cyan}
            />
          </LinearGradient>
        </View>

        <Text variant="headlineSmall" style={styles.title}>
          Wallet Locked
        </Text>

        <Text variant="bodyMedium" style={styles.subtitle}>
          Authenticate to access your wallet
        </Text>

        {showPinInput && (
          <View style={styles.pinContainer}>
            <TextInput
              label="Enter PIN"
              value={pin}
              onChangeText={(text) => {
                setPin(text.replace(/\D/g, '').slice(0, 8));
                if (!isLocked) setError('');
              }}
              onSubmitEditing={handlePinSubmit}
              keyboardType="numeric"
              secureTextEntry
              maxLength={8}
              mode="outlined"
              style={styles.pinInput}
              outlineColor="rgba(0, 188, 212, 0.15)"
              activeOutlineColor={COLORS.cyan}
              textColor="#FFFFFF"
              error={!!error}
              disabled={isLocked}
            />

            {error ? (
              <Text variant="bodySmall" style={styles.error}>
                {error}
              </Text>
            ) : null}

            <Button
              mode="contained"
              onPress={handlePinSubmit}
              style={styles.unlockButton}
              buttonColor={COLORS.cyan}
              textColor="#000000"
              disabled={pin.length < 4 || isLocked}
              labelStyle={{ fontWeight: '700' }}
            >
              {isLocked ? `Locked (${Math.ceil(lockoutMs / 1000)}s)` : 'Unlock'}
            </Button>
          </View>
        )}

        {bioAvailable && (
          <Button
            mode="outlined"
            onPress={attemptBiometric}
            style={styles.bioButton}
            textColor={COLORS.cyan}
            icon="fingerprint"
          >
            Use Biometrics
          </Button>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: COLORS.darkBg,
    padding: 24,
  },
  card: {
    width: '100%',
    maxWidth: 360,
    padding: 32,
    borderRadius: 20,
    alignItems: 'center',
    backgroundColor: COLORS.cardBg,
    borderWidth: 1,
    borderColor: 'rgba(0, 188, 212, 0.1)',
    overflow: 'hidden',
  },
  cardAccent: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 2,
  },
  iconContainer: {
    marginBottom: 16,
    marginTop: 8,
  },
  iconCircle: {
    width: 88,
    height: 88,
    borderRadius: 44,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(0, 188, 212, 0.2)',
  },
  title: {
    color: '#FFFFFF',
    fontWeight: '700',
    marginBottom: 8,
  },
  subtitle: {
    color: 'rgba(255, 255, 255, 0.5)',
    textAlign: 'center',
    marginBottom: 24,
  },
  pinContainer: {
    width: '100%',
    marginBottom: 16,
  },
  pinInput: {
    backgroundColor: COLORS.surfaceBg,
    marginBottom: 8,
  },
  error: {
    color: COLORS.red,
    textAlign: 'center',
    marginBottom: 8,
  },
  unlockButton: {
    marginTop: 8,
    borderRadius: 12,
  },
  bioButton: {
    marginTop: 8,
    borderColor: 'rgba(0, 188, 212, 0.4)',
    borderRadius: 12,
  },
});
