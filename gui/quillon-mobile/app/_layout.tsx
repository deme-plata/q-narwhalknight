import React, { useEffect } from 'react';
import { View, ActivityIndicator, StyleSheet } from 'react-native';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { PaperProvider } from 'react-native-paper';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { quilTheme } from '../src/theme';
import { COLORS } from '../src/theme';
import { useWalletStore } from '../src/stores/walletStore';
import { useSettingsStore } from '../src/stores/settingsStore';
import { useNetworkStore } from '../src/stores/networkStore';
import { useAuth } from '../src/hooks/useAuth';
import { BiometricGate } from '../src/components/BiometricGate';

function AuthGate({ children }: { children: React.ReactNode }) {
  const { needsAuth } = useAuth();
  const unlock = useWalletStore((s) => s.unlock);

  if (needsAuth) {
    return <BiometricGate onUnlocked={unlock} />;
  }

  return <>{children}</>;
}

export default function RootLayout() {
  const initialize = useWalletStore((s) => s.initialize);
  const loadSettings = useSettingsStore((s) => s.loadSettings);
  const checkHealth = useNetworkStore((s) => s.checkHealth);
  const isLoading = useWalletStore((s) => s.isLoading);
  const isLoggedIn = useWalletStore((s) => s.isLoggedIn);

  useEffect(() => {
    initialize();
    loadSettings();
    checkHealth();

    // Periodic health checks
    const interval = setInterval(() => {
      checkHealth();
    }, 60_000);

    return () => clearInterval(interval);
  }, [initialize, loadSettings, checkHealth]);

  // Wait for SecureStore to be read before deciding which screen to show.
  // Without this gate, isLoggedIn defaults to false and the auth screen
  // flashes briefly even when the user has a saved wallet.
  if (isLoading) {
    return (
      <GestureHandlerRootView style={{ flex: 1 }}>
        <SafeAreaProvider>
          <PaperProvider theme={quilTheme}>
            <StatusBar style="light" />
            <View style={splashStyles.container}>
              <ActivityIndicator size="large" color={COLORS.cyan} />
            </View>
          </PaperProvider>
        </SafeAreaProvider>
      </GestureHandlerRootView>
    );
  }

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaProvider>
        <PaperProvider theme={quilTheme}>
          <StatusBar style="light" />
          <AuthGate>
            <Stack
              screenOptions={{
                headerShown: false,
                contentStyle: { backgroundColor: quilTheme.colors.background },
                animation: 'slide_from_right',
              }}
            >
              {!isLoggedIn ? (
                <Stack.Screen name="(auth)" options={{ headerShown: false }} />
              ) : (
                <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
              )}
              <Stack.Screen
                name="settings"
                options={{
                  headerShown: true,
                  title: 'Settings',
                  headerStyle: { backgroundColor: quilTheme.colors.surface },
                  headerTintColor: quilTheme.colors.onSurface,
                  presentation: 'modal',
                }}
              />
              <Stack.Screen
                name="mining"
                options={{
                  headerShown: true,
                  title: 'Mining',
                  headerStyle: { backgroundColor: quilTheme.colors.surface },
                  headerTintColor: quilTheme.colors.onSurface,
                  presentation: 'modal',
                }}
              />
              <Stack.Screen
                name="miner-scan"
                options={{
                  headerShown: true,
                  title: 'Link Miner',
                  headerStyle: { backgroundColor: 'transparent' },
                  headerTintColor: '#FFFFFF',
                  headerTransparent: true,
                  presentation: 'fullScreenModal',
                }}
              />
              <Stack.Screen
                name="tx/[id]"
                options={{
                  headerShown: true,
                  title: 'Transaction',
                  headerStyle: { backgroundColor: quilTheme.colors.surface },
                  headerTintColor: quilTheme.colors.onSurface,
                }}
              />
            </Stack>
          </AuthGate>
        </PaperProvider>
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}

const splashStyles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: COLORS.darkBg,
  },
});
