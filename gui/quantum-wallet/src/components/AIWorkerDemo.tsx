/**
 * AI Worker Demo Component
 *
 * Simple demonstration component showing how to use the AI Worker Panel.
 * This can be added to any page to enable distributed AI compute functionality.
 *
 * Usage:
 * import { AIWorkerDemo } from './components/AIWorkerDemo'
 *
 * function MyPage() {
 *   return (
 *     <div>
 *       <h1>My Page</h1>
 *       <AIWorkerDemo />
 *     </div>
 *   )
 * }
 */

import { useState } from 'react'
import AIWorkerPanel from './AIWorkerPanel'

export function AIWorkerDemo() {
  const [isExpanded, setIsExpanded] = useState(false)

  return (
    <div className="ai-worker-demo" style={{ marginTop: '24px' }}>
      {/* Toggle Button */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        style={{
          background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
          color: 'white',
          padding: '12px 24px',
          borderRadius: '8px',
          border: 'none',
          cursor: 'pointer',
          fontSize: '16px',
          fontWeight: 'bold',
          boxShadow: '0 4px 12px rgba(16, 185, 129, 0.3)',
          transition: 'all 0.2s',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.transform = 'translateY(-2px)'
          e.currentTarget.style.boxShadow = '0 6px 16px rgba(16, 185, 129, 0.4)'
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = 'translateY(0)'
          e.currentTarget.style.boxShadow = '0 4px 12px rgba(16, 185, 129, 0.3)'
        }}
      >
        <span style={{ fontSize: '20px' }}>🤖</span>
        <span>{isExpanded ? 'Hide' : 'Show'} AI Compute Worker</span>
        <span style={{ fontSize: '12px' }}>
          {isExpanded ? '▲' : '▼'}
        </span>
      </button>

      {/* Expandable Panel */}
      {isExpanded && (
        <div
          className="ai-worker-container"
          style={{
            marginTop: '16px',
            animation: 'slideDown 0.3s ease-out'
          }}
        >
          <AIWorkerPanel />
        </div>
      )}

      <style>{`
        @keyframes slideDown {
          from {
            opacity: 0;
            transform: translateY(-20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  )
}

export default AIWorkerDemo
