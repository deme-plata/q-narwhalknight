import { create } from 'zustand';
import * as api from '../services/api';
import { sseManager } from '../services/sse';

interface NetworkState {
  height: number;
  peers: number;
  tps: number;
  difficulty: number;
  version: string;
  networkId: string;
  isOnline: boolean;
  lastChecked: number;
  activeEndpoint: string;

  // Actions
  refreshNetworkStats: () => Promise<void>;
  checkHealth: () => Promise<void>;
  setOnline: (online: boolean) => void;
}

export const useNetworkStore = create<NetworkState>((set) => ({
  height: 0,
  peers: 0,
  tps: 0,
  difficulty: 0,
  version: '',
  networkId: '',
  isOnline: false,
  lastChecked: 0,
  activeEndpoint: api.getActiveEndpoint(),

  refreshNetworkStats: async () => {
    try {
      const stats = await api.getNetworkStats();
      set({
        height: stats.height,
        peers: stats.peers,
        tps: stats.tps,
        difficulty: stats.difficulty,
        isOnline: true,
        lastChecked: Date.now(),
      });
    } catch {
      set({ isOnline: false, lastChecked: Date.now() });
    }
  },

  checkHealth: async () => {
    try {
      const health = await api.getHealth();
      set({
        version: health.version,
        networkId: health.network_id,
        height: health.height,
        peers: health.peers,
        tps: health.tps,
        isOnline: true,
        lastChecked: Date.now(),
        activeEndpoint: api.getActiveEndpoint(),
      });
    } catch {
      set({ isOnline: false, lastChecked: Date.now() });
    }
  },

  setOnline: (online) => set({ isOnline: online }),
}));

// SSE connection → auto-update isOnline.
sseManager.on('connection', (event) => {
  const data = event.data as { status: string };
  if (data.status === 'connected') {
    useNetworkStore.getState().setOnline(true);
    useNetworkStore.getState().checkHealth();
  }
});

// SSE new-block → instant height update without REST call
sseManager.on('new-block', (event) => {
  const data = event.data as { height?: number; block_height?: number };
  const height = data.height ?? data.block_height;
  if (typeof height === 'number' && height > 0) {
    useNetworkStore.setState({
      height,
      isOnline: true,
      lastChecked: Date.now(),
    });
  }
});
