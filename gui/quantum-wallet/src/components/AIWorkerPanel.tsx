/**
 * AI Worker Panel Component
 *
 * Provides user interface for managing browser-based AI compute worker:
 * - Enable/disable worker mode
 * - View real-time stats (peers, latency, earnings)
 * - Monitor capability detection (WebGPU vs WASM)
 * - Track inference requests processed
 */

import React, { useState } from 'react'
import {
  useAIWorker,
  formatUptime,
  formatEarnings,
  getCapabilityDisplayName,
  getCapabilityColor
} from '../hooks/useAIWorker'

interface AIWorkerPanelProps {
  className?: string
}

export const AIWorkerPanel: React.FC<AIWorkerPanelProps> = ({ className = '' }) => {
  const {
    worker,
    stats,
    isInitializing,
    error,
    startWorker,
    stopWorker
  } = useAIWorker()

  const [showAdvanced, setShowAdvanced] = useState(false)

  const handleToggleWorker = async () => {
    if (stats.isRunning) {
      await stopWorker()
    } else {
      await startWorker()
    }
  }

  const getLatencyColor = (latency: number): string => {
    if (latency === 0) return '#6b7280' // gray-500
    if (latency < 50) return '#10b981' // green-500
    if (latency < 100) return '#f59e0b' // amber-500
    return '#ef4444' // red-500
  }

  const getConnectionQuality = (latency: number): string => {
    if (latency === 0) return 'Unknown'
    if (latency < 50) return 'Excellent'
    if (latency < 100) return 'Good'
    if (latency < 200) return 'Fair'
    return 'Poor'
  }

  return (
    <div className={`ai-worker-panel ${className}`}>
      {/* Header */}
      <div className="panel-header">
        <div className="title-section">
          <h2>🤖 AI Compute Worker</h2>
          <p className="subtitle">
            Earn QUG by contributing compute resources to distributed AI inference
          </p>
        </div>

        {/* Status Indicator */}
        <div className="status-indicator">
          <div
            className={`status-dot ${stats.isRunning ? 'running' : 'stopped'}`}
            style={{
              backgroundColor: stats.isRunning ? '#10b981' : '#6b7280',
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              marginRight: '8px'
            }}
          />
          <span className="status-text">
            {stats.isRunning ? 'Running' : 'Stopped'}
          </span>
        </div>
      </div>

      {/* Main Controls */}
      <div className="controls-section">
        <button
          onClick={handleToggleWorker}
          disabled={isInitializing}
          className={`toggle-button ${stats.isRunning ? 'stop' : 'start'}`}
          style={{
            backgroundColor: stats.isRunning ? '#ef4444' : '#10b981',
            color: 'white',
            padding: '12px 24px',
            borderRadius: '8px',
            border: 'none',
            cursor: isInitializing ? 'not-allowed' : 'pointer',
            fontSize: '16px',
            fontWeight: 'bold',
            opacity: isInitializing ? 0.6 : 1,
            transition: 'all 0.2s'
          }}
        >
          {isInitializing
            ? '⏳ Initializing...'
            : stats.isRunning
            ? '⏹️ Stop Worker'
            : '▶️ Start Worker'}
        </button>

        {error && (
          <div
            className="error-banner"
            style={{
              backgroundColor: '#fee2e2',
              border: '1px solid #ef4444',
              borderRadius: '8px',
              padding: '12px',
              marginTop: '16px',
              color: '#991b1b'
            }}
          >
            <strong>Error:</strong> {error.message}
          </div>
        )}
      </div>

      {/* Stats Grid */}
      {stats.isRunning && (
        <div className="stats-grid" style={{ marginTop: '24px' }}>
          {/* Capability Card */}
          <div className="stat-card">
            <div className="stat-label">Compute Capability</div>
            <div
              className="stat-value"
              style={{
                color: getCapabilityColor(stats.capability),
                fontWeight: 'bold'
              }}
            >
              {getCapabilityDisplayName(stats.capability)}
            </div>
            {stats.capability.type === 'WebGPU' && (
              <div className="stat-detail">
                ⚡ GPU Acceleration Enabled
              </div>
            )}
            {stats.capability.type === 'WASM_CPU' && (
              <div className="stat-detail">
                🔧 CPU Inference Mode
              </div>
            )}
          </div>

          {/* Peer Connections Card */}
          <div className="stat-card">
            <div className="stat-label">Connected Peers</div>
            <div className="stat-value" style={{ fontSize: '32px', fontWeight: 'bold' }}>
              {stats.connectedPeers}
            </div>
            <div className="stat-detail">
              Avg Latency: {stats.avgLatency}ms
              <span
                style={{
                  color: getLatencyColor(stats.avgLatency),
                  marginLeft: '8px',
                  fontWeight: 'bold'
                }}
              >
                ({getConnectionQuality(stats.avgLatency)})
              </span>
            </div>
          </div>

          {/* Requests Processed Card */}
          <div className="stat-card">
            <div className="stat-label">Inference Requests</div>
            <div className="stat-value" style={{ fontSize: '32px', fontWeight: 'bold' }}>
              {stats.requestsProcessed}
            </div>
            <div className="stat-detail">
              {stats.tokensGenerated.toLocaleString()} tokens generated
            </div>
          </div>

          {/* Earnings Card */}
          <div className="stat-card">
            <div className="stat-label">Earnings</div>
            <div
              className="stat-value"
              style={{
                fontSize: '28px',
                fontWeight: 'bold',
                color: '#10b981'
              }}
            >
              {formatEarnings(stats.earningsQUG)}
            </div>
            <div className="stat-detail">
              Uptime: {formatUptime(stats.uptime)}
            </div>
          </div>
        </div>
      )}

      {/* Advanced Info Section */}
      {stats.isRunning && (
        <div className="advanced-section" style={{ marginTop: '24px' }}>
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="advanced-toggle"
            style={{
              background: 'none',
              border: 'none',
              color: '#3b82f6',
              cursor: 'pointer',
              fontSize: '14px',
              textDecoration: 'underline'
            }}
          >
            {showAdvanced ? '▼' : '▶'} Advanced Info
          </button>

          {showAdvanced && (
            <div
              className="advanced-content"
              style={{
                backgroundColor: '#f9fafb',
                borderRadius: '8px',
                padding: '16px',
                marginTop: '12px',
                fontFamily: 'monospace',
                fontSize: '12px'
              }}
            >
              <div><strong>Peer ID:</strong> {stats.peerId || 'Not connected'}</div>
              <div style={{ marginTop: '8px' }}>
                <strong>Capability Details:</strong>
              </div>
              <pre style={{ marginTop: '8px', overflow: 'auto' }}>
                {JSON.stringify(stats.capability, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}

      {/* Information Banner */}
      {!stats.isRunning && (
        <div
          className="info-banner"
          style={{
            backgroundColor: '#eff6ff',
            border: '1px solid #3b82f6',
            borderRadius: '8px',
            padding: '16px',
            marginTop: '24px',
            color: '#1e40af'
          }}
        >
          <h3 style={{ margin: '0 0 8px 0' }}>💡 How It Works</h3>
          <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
            <li>Your browser becomes a compute node in the P2P network</li>
            <li>Process AI inference requests using WebGPU or WASM</li>
            <li>Earn QUG tokens for each token generated</li>
            <li>Rate: 10,000 tokens = 1 QUG</li>
            <li>All compute happens locally - your data stays private</li>
          </ul>
          <div style={{ marginTop: '12px', fontSize: '12px', opacity: 0.8 }}>
            <strong>Note:</strong> Worker mode requires an active internet connection.
            Your browser will connect to bootstrap peers via WebSocket.
          </div>
        </div>
      )}

      {/* CSS Styles */}
      <style>{`
        .ai-worker-panel {
          background: white;
          border-radius: 12px;
          padding: 24px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 24px;
          padding-bottom: 16px;
          border-bottom: 2px solid #e5e7eb;
        }

        .title-section h2 {
          margin: 0;
          font-size: 24px;
          font-weight: bold;
          color: #1f2937;
        }

        .subtitle {
          margin: 4px 0 0 0;
          font-size: 14px;
          color: #6b7280;
        }

        .status-indicator {
          display: flex;
          align-items: center;
          font-size: 14px;
          font-weight: 600;
        }

        .controls-section {
          display: flex;
          flex-direction: column;
          align-items: center;
        }

        .stats-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 16px;
        }

        .stat-card {
          background: #f9fafb;
          border-radius: 8px;
          padding: 16px;
          border: 1px solid #e5e7eb;
        }

        .stat-label {
          font-size: 12px;
          color: #6b7280;
          text-transform: uppercase;
          font-weight: 600;
          margin-bottom: 8px;
        }

        .stat-value {
          font-size: 24px;
          font-weight: bold;
          color: #1f2937;
          margin-bottom: 4px;
        }

        .stat-detail {
          font-size: 12px;
          color: #6b7280;
        }

        .toggle-button:hover:not(:disabled) {
          opacity: 0.9;
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }

        @media (max-width: 768px) {
          .panel-header {
            flex-direction: column;
            align-items: flex-start;
          }

          .status-indicator {
            margin-top: 12px;
          }

          .stats-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  )
}

export default AIWorkerPanel
