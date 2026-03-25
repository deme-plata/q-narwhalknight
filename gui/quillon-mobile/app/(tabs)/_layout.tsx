import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Tabs } from 'expo-router';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { COLORS } from '../../src/theme';

type IconName = React.ComponentProps<typeof MaterialCommunityIcons>['name'];

function TabIcon({ name, color, focused }: { name: IconName; color: string; focused: boolean }) {
  return (
    <View style={styles.iconWrapper}>
      {focused && (
        <View style={styles.activeGlow} />
      )}
      <MaterialCommunityIcons name={name} color={color} size={24} />
    </View>
  );
}

export default function TabsLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: COLORS.cyan,
        tabBarInactiveTintColor: 'rgba(255, 255, 255, 0.35)',
        tabBarStyle: {
          backgroundColor: COLORS.cardBg,
          borderTopColor: 'rgba(0, 188, 212, 0.1)',
          borderTopWidth: 1,
          height: 64,
          paddingBottom: 8,
          paddingTop: 6,
          elevation: 0,
        },
        tabBarLabelStyle: {
          fontSize: 11,
          fontWeight: '600',
          letterSpacing: 0.3,
        },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: 'Home',
          tabBarIcon: ({ color, focused }) => (
            <TabIcon name="home" color={color} focused={focused} />
          ),
        }}
      />
      <Tabs.Screen
        name="send"
        options={{
          title: 'Send',
          tabBarIcon: ({ color, focused }) => (
            <TabIcon name="arrow-up-circle" color={color} focused={focused} />
          ),
        }}
      />
      <Tabs.Screen
        name="receive"
        options={{
          title: 'Receive',
          tabBarIcon: ({ color, focused }) => (
            <TabIcon name="arrow-down-circle" color={color} focused={focused} />
          ),
        }}
      />
      <Tabs.Screen
        name="dex"
        options={{
          title: 'DEX',
          tabBarIcon: ({ color, focused }) => (
            <TabIcon name="swap-horizontal" color={color} focused={focused} />
          ),
        }}
      />
      <Tabs.Screen
        name="history"
        options={{
          title: 'History',
          tabBarIcon: ({ color, focused }) => (
            <TabIcon name="history" color={color} focused={focused} />
          ),
        }}
      />
    </Tabs>
  );
}

const styles = StyleSheet.create({
  iconWrapper: {
    width: 36,
    height: 28,
    justifyContent: 'center',
    alignItems: 'center',
  },
  activeGlow: {
    position: 'absolute',
    width: 36,
    height: 28,
    borderRadius: 14,
    backgroundColor: 'rgba(0, 188, 212, 0.12)',
  },
});
