/**
 * Tor Status Indicator Component
 *
 * 🧅 Shows the current Tor connection status in the UI
 *
 * This is a small, non-intrusive indicator that shows users
 * their traffic is being routed through Tor for privacy.
 *
 * States:
 * - Connecting: Yellow pulsing onion
 * - Connected: Green onion with latency
 * - Disconnected: Red onion with error
 *
 * @module TorStatusIndicator
 * @version 1.0.0
 */

import React, { useState, useEffect } from 'react'
import type { TorConnectionState } from '../libp2p/torConfig'
import {
  getTorState,
  formatTorStatus,
  isTorHealthy,
} from '../libp2p/torConfig'

interface TorStatusIndicatorProps {
  // Show compact version (just icon) or full (icon + text)
  compact?: boolean
  // Show in toolbar style
  toolbar?: boolean
  // Custom className
  className?: string
}

export function TorStatusIndicator({
  compact = false,
  toolbar = false,
  className = '',
}: TorStatusIndicatorProps) {
  const [torState, setTorState] = useState<TorConnectionState>(getTorState())
  const [isAnimating, setIsAnimating] = useState(false)

  // Listen for Tor state changes
  useEffect(() => {
    const handleStateChange = (event: CustomEvent<TorConnectionState>) => {
      setTorState(event.detail)
      // Trigger animation on state change
      setIsAnimating(true)
      setTimeout(() => setIsAnimating(false), 500)
    }

    window.addEventListener('tor-state-changed', handleStateChange as EventListener)
    return () => {
      window.removeEventListener('tor-state-changed', handleStateChange as EventListener)
    }
  }, [])

  // Determine status
  const isConnected = torState.isConnected && torState.circuitEstablished
  const isConnecting = !isConnected && torState.reconnectAttempts < 5
  const hasError = torState.errors.length > 0

  // Get status color
  const getStatusColor = () => {
    if (isConnected) return '#4ade80' // green-400
    if (isConnecting) return '#facc15' // yellow-400
    return '#f87171' // red-400
  }

  // Get status text
  const getStatusText = () => {
    if (isConnected) {
      const latency = torState.bridgeLatency
      return latency ? `Tor (${latency}ms)` : 'Tor Connected'
    }
    if (isConnecting) return 'Connecting...'
    return 'Tor Offline'
  }

  // Toolbar style (horizontal, compact)
  if (toolbar) {
    return (
      <div
        className={`tor-status-toolbar ${className}`}
        title={formatTorStatus()}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '6px',
          padding: '4px 8px',
          borderRadius: '4px',
          backgroundColor: 'rgba(0, 0, 0, 0.2)',
          fontSize: '12px',
          color: getStatusColor(),
        }}
      >
        <span
          style={{
            animation: isConnecting ? 'pulse 1.5s infinite' : undefined,
          }}
        >
          🧅
        </span>
        {!compact && (
          <span style={{ color: '#e5e5e5' }}>{getStatusText()}</span>
        )}
      </div>
    )
  }

  // Default style (vertical card)
  return (
    <div
      className={`tor-status-indicator ${className}`}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '8px',
        padding: '12px',
        borderRadius: '8px',
        backgroundColor: 'rgba(0, 0, 0, 0.3)',
        border: `1px solid ${getStatusColor()}40`,
      }}
    >
      {/* Onion icon with status */}
      <div
        style={{
          fontSize: '24px',
          animation: isConnecting ? 'pulse 1.5s infinite' : isAnimating ? 'bounce 0.5s' : undefined,
        }}
      >
        🧅
      </div>

      {/* Status text */}
      {!compact && (
        <div style={{ textAlign: 'center' }}>
          <div
            style={{
              fontSize: '14px',
              fontWeight: 'bold',
              color: getStatusColor(),
            }}
          >
            {isConnected ? 'Protected' : isConnecting ? 'Connecting' : 'Offline'}
          </div>
          <div
            style={{
              fontSize: '11px',
              color: '#a3a3a3',
              marginTop: '2px',
            }}
          >
            {isConnected
              ? `Traffic routed via Tor${torState.bridgeLatency ? ` (${torState.bridgeLatency}ms)` : ''}`
              : isConnecting
                ? 'Establishing Tor circuit...'
                : 'Tor connection lost'}
          </div>
        </div>
      )}

      {/* Error indicator */}
      {hasError && !compact && (
        <div
          style={{
            fontSize: '10px',
            color: '#f87171',
            marginTop: '4px',
          }}
        >
          ⚠️ {torState.errors[torState.errors.length - 1]}
        </div>
      )}

      {/* CSS for animations */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        @keyframes bounce {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.2); }
        }
      `}</style>
    </div>
  )
}

/**
 * Hook to get Tor status
 */
export function useTorStatus() {
  const [torState, setTorState] = useState<TorConnectionState>(getTorState())

  useEffect(() => {
    const handleStateChange = (event: CustomEvent<TorConnectionState>) => {
      setTorState(event.detail)
    }

    window.addEventListener('tor-state-changed', handleStateChange as EventListener)
    return () => {
      window.removeEventListener('tor-state-changed', handleStateChange as EventListener)
    }
  }, [])

  return {
    ...torState,
    isHealthy: isTorHealthy(),
    statusText: formatTorStatus(),
    isConnected: torState.isConnected && torState.circuitEstablished,
  }
}

export default TorStatusIndicator
