import React from 'react';
import { Stack } from 'expo-router';
import { quilTheme } from '../../src/theme';

export default function AuthLayout() {
  return (
    <Stack
      screenOptions={{
        headerShown: false,
        contentStyle: { backgroundColor: quilTheme.colors.background },
        animation: 'fade',
      }}
    >
      <Stack.Screen name="login" />
      <Stack.Screen
        name="backup"
        options={{
          headerShown: true,
          title: 'Backup Seed Phrase',
          headerStyle: { backgroundColor: quilTheme.colors.surface },
          headerTintColor: quilTheme.colors.onSurface,
          presentation: 'modal',
        }}
      />
    </Stack>
  );
}
