/**
 * P2P Status Component
 *
 * Displays the current P2P network connection status,
 * including peer count, topics, and connection quality.
 *
 * This component provides real-time visibility into the
 * browser's P2P node connectivity.
 */

import { useLibP2P } from '../contexts/LibP2PContext'

/**
 * P2P Status Display Component
 */
export function P2PStatus() {
  const {
    peerId,
    peerCount,
    connectionCount,
    topics,
    isReady,
    isConnecting,
    error,
    refresh,
  } = useLibP2P()

  /**
   * Render connection status indicator
   */
  const renderStatusIndicator = () => {
    if (error) {
      return (
        <div className="flex items-center gap-2 text-red-500">
          <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
          <span>P2P Error</span>
        </div>
      )
    }

    if (isConnecting) {
      return (
        <div className="flex items-center gap-2 text-yellow-500">
          <div className="w-3 h-3 bg-yellow-500 rounded-full animate-pulse" />
          <span>Connecting...</span>
        </div>
      )
    }

    if (!isReady) {
      return (
        <div className="flex items-center gap-2 text-gray-500">
          <div className="w-3 h-3 bg-gray-500 rounded-full" />
          <span>P2P Inactive</span>
        </div>
      )
    }

    if (peerCount === 0) {
      return (
        <div className="flex items-center gap-2 text-orange-500">
          <div className="w-3 h-3 bg-orange-500 rounded-full animate-pulse" />
          <span>No Peers</span>
        </div>
      )
    }

    return (
      <div className="flex items-center gap-2 text-green-500">
        <div className="w-3 h-3 bg-green-500 rounded-full" />
        <span>P2P Active</span>
      </div>
    )
  }

  /**
   * Render peer count with color coding
   */
  const renderPeerCount = () => {
    let colorClass = 'text-gray-500'

    if (peerCount >= 10) {
      colorClass = 'text-green-500'
    } else if (peerCount >= 5) {
      colorClass = 'text-blue-500'
    } else if (peerCount >= 1) {
      colorClass = 'text-orange-500'
    } else {
      colorClass = 'text-red-500'
    }

    return <span className={`font-bold ${colorClass}`}>{peerCount}</span>
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4 border border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          P2P Network Status
        </h3>
        {renderStatusIndicator()}
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 rounded border border-red-200 dark:border-red-800">
          <p className="text-sm text-red-600 dark:text-red-400 font-medium">
            Connection Error
          </p>
          <p className="text-xs text-red-500 dark:text-red-300 mt-1">
            {error.message}
          </p>
        </div>
      )}

      {isReady && (
        <div className="space-y-3">
          {/* Peer Information */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Connected Peers
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {renderPeerCount()}
              </p>
            </div>

            <div>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Connections
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {connectionCount}
              </p>
            </div>
          </div>

          {/* Peer ID */}
          {peerId && (
            <div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                Peer ID
              </p>
              <p className="text-xs font-mono text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded truncate">
                {peerId}
              </p>
            </div>
          )}

          {/* Subscribed Topics */}
          {topics.length > 0 && (
            <div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                Subscribed Topics ({topics.length})
              </p>
              <div className="space-y-1">
                {topics.slice(0, 3).map((topic) => (
                  <p
                    key={topic}
                    className="text-xs font-mono text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded truncate"
                  >
                    {topic}
                  </p>
                ))}
                {topics.length > 3 && (
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    +{topics.length - 3} more
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Refresh Button */}
          <button
            onClick={refresh}
            className="w-full mt-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white text-sm font-medium rounded transition-colors"
          >
            Refresh Status
          </button>
        </div>
      )}

      {isConnecting && (
        <div className="flex flex-col items-center justify-center py-8">
          <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <p className="mt-4 text-sm text-gray-600 dark:text-gray-400">
            Connecting to P2P network...
          </p>
        </div>
      )}
    </div>
  )
}

/**
 * Compact P2P Status Badge (for navbar)
 */
export function P2PStatusBadge() {
  const { peerCount, isReady, isConnecting, error } = useLibP2P()

  if (error) {
    return (
      <div className="flex items-center gap-2 px-3 py-1 bg-red-100 dark:bg-red-900/20 rounded-full">
        <div className="w-2 h-2 bg-red-500 rounded-full" />
        <span className="text-xs font-medium text-red-600 dark:text-red-400">
          P2P Error
        </span>
      </div>
    )
  }

  if (isConnecting) {
    return (
      <div className="flex items-center gap-2 px-3 py-1 bg-yellow-100 dark:bg-yellow-900/20 rounded-full">
        <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
        <span className="text-xs font-medium text-yellow-600 dark:text-yellow-400">
          Connecting...
        </span>
      </div>
    )
  }

  if (!isReady) {
    return null // Don't show badge if not ready
  }

  return (
    <div className="flex items-center gap-2 px-3 py-1 bg-green-100 dark:bg-green-900/20 rounded-full">
      <div className="w-2 h-2 bg-green-500 rounded-full" />
      <span className="text-xs font-medium text-green-600 dark:text-green-400">
        {peerCount} peer{peerCount !== 1 ? 's' : ''}
      </span>
    </div>
  )
}
