import { useEffect, useRef, useState } from 'react';
import { sseManager, SSEEvent } from '../services/sse';

interface SSEConnectionState {
  connected: boolean;
  reconnecting: boolean;
  lastEventTime: number;
}

/**
 * Hook to manage SSE connection lifecycle and subscribe to events.
 */
export function useSSE(
  address: string | null,
  onEvent?: (event: SSEEvent) => void
): SSEConnectionState {
  const [state, setState] = useState<SSEConnectionState>({
    connected: false,
    reconnecting: false,
    lastEventTime: 0,
  });
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  useEffect(() => {
    if (!address) return;

    sseManager.connect(address);

    const unsubConnection = sseManager.on('connection', (event) => {
      const data = event.data as { status: string };
      setState({
        connected: data.status === 'connected',
        reconnecting: data.status === 'reconnecting',
        lastEventTime: event.timestamp,
      });
    });

    const unsubAll = sseManager.on('*', (event) => {
      if (event.type !== 'connection') {
        setState((prev) => ({ ...prev, lastEventTime: event.timestamp }));
        onEventRef.current?.(event);
      }
    });

    return () => {
      unsubConnection();
      unsubAll();
      sseManager.disconnect();
    };
  }, [address]);

  return state;
}

/**
 * Hook to subscribe to a specific SSE event type.
 */
export function useSSEEvent(
  eventType: string,
  callback: (data: unknown) => void
): void {
  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  useEffect(() => {
    const unsub = sseManager.on(eventType, (event) => {
      callbackRef.current(event.data);
    });
    return unsub;
  }, [eventType]);
}
