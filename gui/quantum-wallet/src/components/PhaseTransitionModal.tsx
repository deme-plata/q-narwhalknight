import React, { useState, useEffect } from 'react';
import './PhaseTransitionModal.css';

interface PhaseTransitionModalProps {
  onClose: () => void;
}

const PhaseTransitionModal: React.FC<PhaseTransitionModalProps> = ({ onClose }) => {
  const [activeTab, setActiveTab] = useState<'announcement' | 'faq'>('announcement');
  const [showDetails, setShowDetails] = useState(false);

  // Mark modal as seen in localStorage
  useEffect(() => {
    localStorage.setItem('v0918betaModalSeen', 'true');
  }, []);

  return (
    <div className="phase-transition-overlay">
      <div className="phase-transition-modal">
        {/* Header */}
        <div className="modal-header">
          <span className="quantum-logo">⚛️</span>
          <h1 className="modal-title">v0.9.18-beta Released!</h1>
          <button className="close-btn" onClick={onClose} aria-label="Close">&times;</button>
        </div>

        {/* Content */}
        <div className="modal-content">
          {activeTab === 'announcement' && (
            <>
              {/* Status Badge */}
              <div className="status-badge success">
                ✅ v0.9.18-beta: Network Hashrate Reporting + Improved Sync
              </div>

              {/* New Features */}
              <div className="info-box warning">
                <h3>🎯 What's New in v0.9.18-beta?</h3>
                <p className="lead">
                  Major improvements to mining transparency and network synchronization!
                </p>
                <ul className="bullet-list">
                  <li>📊 <strong>Network Hashrate Reporting</strong> - Miners now report their hashrate to the network</li>
                  <li>🌐 <strong>Live Network Statistics</strong> - Explorer page shows total network KH/s, MH/s, or GH/s</li>
                  <li>⚡ <strong>Enhanced Sync Performance</strong> - Faster block synchronization and reduced memory usage</li>
                  <li>🔧 <strong>Improved Stability</strong> - Better error handling and connection management</li>
                </ul>
              </div>

              {/* The Update */}
              <div className="detail-section highlight">
                <h3>✨ v0.9.18-beta: Key Features</h3>
                <ul className="bullet-list">
                  <li>⛏️ <strong>Miner Hashrate Tracking</strong> - Your mining contributions are now visible on the network</li>
                  <li>📈 <strong>Real-time Network Stats</strong> - See total network computing power on the Explorer page</li>
                  <li>🚀 <strong>Faster Sync</strong> - Optimized block propagation and validation</li>
                  <li>🔒 <strong>Production Ready</strong> - Enhanced stability for long-running nodes</li>
                  <li>💎 <strong>Better Mining Experience</strong> - More accurate statistics and monitoring</li>
                </ul>
              </div>

              {/* Phase 4 Features */}
              <div className="detail-section">
                <h3>🚀 Testnet Phase 4: Stable & Growing</h3>
                <p>The network is running smoothly with enhanced features:</p>
                <ul className="bullet-list">
                  <li>✅ Stable network with consistent block production</li>
                  <li>✅ Height monotonicity enforcement (blocks never deleted)</li>
                  <li>✅ Active network: <code>testnet-phase4</code></li>
                  <li>✅ Real-time hashrate monitoring</li>
                  <li>✅ Improved peer discovery and connectivity</li>
                </ul>
              </div>

              {/* Download Section */}
              <div className="action-section">
                <h3>📥 Upgrade to v0.9.18-beta</h3>
                <p>Download the latest versions with network hashrate reporting:</p>
                <div className="download-links">
                  <a href="/downloads/q-api-server-v0.9.18-beta" className="download-btn" download style={{marginRight: '10px'}}>
                    📦 Node v0.9.18-beta
                  </a>
                  <a href="/downloads/q-miner-linux-x64" className="download-btn" download>
                    ⛏️ Miner v0.9.18-beta
                  </a>
                </div>
                <div className="code-block">
                  <code>
                    # Stop your node<br/>
                    killall q-api-server<br/><br/>
                    # Download latest versions<br/>
                    wget https://quillon.xyz/downloads/q-api-server-v0.9.18-beta<br/>
                    wget https://quillon.xyz/downloads/q-miner-linux-x64<br/>
                    chmod +x q-api-server-v0.9.18-beta q-miner-linux-x64<br/><br/>
                    # Start node (no database reset needed!)<br/>
                    ./q-api-server-v0.9.18-beta --port 8080<br/><br/>
                    # Start mining with hashrate reporting<br/>
                    ./q-miner-linux-x64 --mode solo --wallet YOUR_WALLET --threads 4
                  </code>
                </div>
              </div>

              {/* Technical Details (Collapsible) */}
              <button
                className="toggle-details-btn"
                onClick={() => setShowDetails(!showDetails)}
              >
                {showDetails ? '▼ Hide Technical Details' : '▶ Show Technical Details'}
              </button>

              {showDetails && (
                <div className="bootstrap-info">
                  <h4>🔬 Technical Implementation Details</h4>
                  <p>
                    The v0.9.18-beta release adds network hashrate aggregation and improved synchronization:
                  </p>
                  <div className="code-block">
                    <code>
                      // NEW: Miner reports hashrate with each solution<br/>
                      let solution = serde_json::json!(&#123;<br/>
                      &nbsp;&nbsp;"miner_address": wallet,<br/>
                      &nbsp;&nbsp;"nonce": nonce,<br/>
                      &nbsp;&nbsp;"hash": hex::encode(hash),<br/>
                      &nbsp;&nbsp;"hash_rate": hashrate_khs  // NEW FIELD!<br/>
                      &#125;);<br/><br/>
                      // API aggregates hashrates from all active miners<br/>
                      let network_khash = mining_stats.calculate_network_hashrate();<br/>
                      // Returns sum of all miners active in last 5 minutes
                    </code>
                  </div>
                  <p className="lead" style={{marginTop: '1rem'}}>
                    <strong>Server Beta Bootstrap Node:</strong>
                  </p>
                  <div className="code-block">
                    <code>
                      /ip4/185.182.185.227/tcp/9001/p2p/12D3KooWRX3GGK9Fs3iM3BfqYNNJiHBDujac7EHwqWjaK1n1kzPN<br/>
                      Network ID: testnet-phase4<br/>
                      API: https://quillon.xyz
                    </code>
                  </div>
                </div>
              )}

              {/* Important Reminder */}
              <div className="info-box warning">
                <h3>💡 Important Reminder</h3>
                <p>
                  <strong>Testnet balances have NO VALUE.</strong> This is a TEST network. All balances reset
                  when we find critical bugs like this. Your testing helps us build a bulletproof mainnet.
                </p>
              </div>
            </>
          )}

          {activeTab === 'faq' && (
            <div className="faq-section">
              <h2>Frequently Asked Questions</h2>

              <div className="faq-item">
                <h4>Q: What is network hashrate reporting?</h4>
                <p>
                  A: Miners now send their current hashrate (in KH/s) when submitting solutions. The API aggregates
                  this from all active miners to show total network computing power on the Explorer page.
                </p>
              </div>

              <div className="faq-item">
                <h4>Q: Do I need to update my miner?</h4>
                <p>
                  A: Yes! The new v0.9.18-beta miner includes hashrate reporting. Download it from the Mining page
                  or use the links in this modal. Old miners will still work but won't contribute to network statistics.
                </p>
              </div>

              <div className="faq-item">
                <h4>Q: Do I need to reset my database?</h4>
                <p>
                  A: No! This is a minor update with no breaking changes. Your existing database and balances
                  are safe. Just stop your node, replace the binaries, and restart.
                </p>
              </div>

              <div className="faq-item">
                <h4>Q: Where can I see the network hashrate?</h4>
                <p>
                  A: Go to the Explorer page in the wallet UI. Look for the "Network Hashrate" stat in the
                  Network Overview section. It shows total network computing power in KH/s, MH/s, or GH/s.
                </p>
              </div>

              <div className="faq-item">
                <h4>Q: How do I verify my miner is reporting hashrate?</h4>
                <p>
                  A: Check the miner logs for messages like "⛏️ Mining │ X.XX MH/s". The new miner automatically
                  reports this to the network every 5 seconds when submitting solutions.
                </p>
              </div>

              <div className="faq-item">
                <h4>Q: What other improvements are in v0.9.18-beta?</h4>
                <p>
                  A: Enhanced sync performance, better error handling, reduced memory usage during synchronization,
                  and improved connection stability for long-running nodes.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Footer with Tabs */}
        <div className="modal-footer">
          <div style={{display: 'flex', gap: '10px'}}>
            <button
              className={`tab-btn ${activeTab === 'announcement' ? 'active' : ''}`}
              onClick={() => setActiveTab('announcement')}
            >
              📢 Announcement
            </button>
            <button
              className={`tab-btn ${activeTab === 'faq' ? 'active' : ''}`}
              onClick={() => setActiveTab('faq')}
            >
              ❓ FAQ
            </button>
          </div>
          <button className="primary-btn" onClick={onClose}>
            Start Using v0.9.18-beta →
          </button>
        </div>
      </div>
    </div>
  );
};

export default PhaseTransitionModal;
