import React, { useState } from 'react';
import { View, StyleSheet, ScrollView } from 'react-native';
import { Text, Button, Checkbox, Surface, Chip } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { router, useLocalSearchParams } from 'expo-router';
import { COLORS } from '../../src/theme';

export default function BackupScreen() {
  const { mnemonic } = useLocalSearchParams<{ mnemonic: string }>();
  const words = (mnemonic ?? '').split(' ');
  const [confirmed, setConfirmed] = useState(false);

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.scrollContent}>
      <MaterialCommunityIcons
        name="alert-circle"
        size={48}
        color={COLORS.orange}
        style={styles.icon}
      />

      <Text variant="headlineSmall" style={styles.title}>
        Save Your Seed Phrase
      </Text>

      <Text variant="bodyMedium" style={styles.subtitle}>
        Write down these {words.length} words in order. This is the ONLY way to recover your wallet.
        Never share it with anyone.
      </Text>

      <Surface style={styles.wordsContainer} elevation={1}>
        <View style={styles.wordsGrid}>
          {words.map((word, index) => (
            <View key={index} style={styles.wordItem}>
              <Text style={styles.wordNumber}>{index + 1}</Text>
              <Chip
                style={styles.wordChip}
                textStyle={styles.wordChipText}
                compact
              >
                {word}
              </Chip>
            </View>
          ))}
        </View>
      </Surface>

      <Surface style={styles.warningBox} elevation={0}>
        <MaterialCommunityIcons name="alert" size={20} color={COLORS.orange} />
        <Text variant="bodySmall" style={styles.warningText}>
          If you lose this phrase, you will permanently lose access to your funds.
          Store it offline in a safe place.
        </Text>
      </Surface>

      <View style={styles.checkboxRow}>
        <Checkbox.Android
          status={confirmed ? 'checked' : 'unchecked'}
          onPress={() => setConfirmed(!confirmed)}
          color={COLORS.cyan}
          uncheckedColor="rgba(255, 255, 255, 0.5)"
        />
        <Text
          variant="bodySmall"
          style={styles.checkboxLabel}
          onPress={() => setConfirmed(!confirmed)}
        >
          I have written down my seed phrase and stored it safely
        </Text>
      </View>

      <Button
        mode="contained"
        onPress={() => router.replace('/(tabs)')}
        disabled={!confirmed}
        style={styles.primaryButton}
        buttonColor={COLORS.cyan}
        textColor="#000000"
      >
        Continue
      </Button>

      <Button
        mode="text"
        onPress={() => router.replace('/(tabs)')}
        textColor="rgba(255, 255, 255, 0.5)"
      >
        Skip for now
      </Button>
    </ScrollView>
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
  icon: {
    alignSelf: 'center',
    marginBottom: 16,
  },
  title: {
    color: '#FFFFFF',
    fontWeight: '700',
    textAlign: 'center',
    marginBottom: 8,
  },
  subtitle: {
    color: 'rgba(255, 255, 255, 0.6)',
    textAlign: 'center',
    marginBottom: 24,
    lineHeight: 20,
  },
  wordsContainer: {
    borderRadius: 16,
    padding: 16,
    backgroundColor: COLORS.cardBg,
    marginBottom: 16,
  },
  wordsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  wordItem: {
    flexDirection: 'row',
    alignItems: 'center',
    width: '48%',
    marginBottom: 4,
  },
  wordNumber: {
    color: 'rgba(255, 255, 255, 0.3)',
    fontSize: 12,
    width: 24,
    textAlign: 'right',
    marginRight: 8,
  },
  wordChip: {
    backgroundColor: COLORS.surfaceBg,
    flex: 1,
  },
  wordChipText: {
    color: '#FFFFFF',
    fontSize: 13,
  },
  warningBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    padding: 12,
    borderRadius: 12,
    backgroundColor: 'rgba(255, 110, 64, 0.1)',
    marginBottom: 16,
    gap: 8,
  },
  warningText: {
    color: COLORS.orange,
    flex: 1,
    lineHeight: 18,
  },
  checkboxRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  checkboxLabel: {
    color: 'rgba(255, 255, 255, 0.7)',
    flex: 1,
  },
  primaryButton: {
    borderRadius: 14,
    marginBottom: 12,
  },
});
