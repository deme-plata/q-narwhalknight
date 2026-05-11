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
    localStorage.setItem('mainnet2026.2_transition_seen', 'true');
  }, []);

  return (
    <div className="phase-transition-overlay">
      <div className="phase-transition-modal">
        {/* Header */}
        <div className="modal-header">
          <span className="quantum-logo">⚛️</span>
          <h1 className="modal-title">Mainnet 2026.2 — Fresh Launch</h1>
          <button className="close-btn" onClick={onClose} aria-label="Close">&times;</button>
        </div>

        {/* Content */}
        <div className="modal-content">
          {activeTab === 'announcement' && (
            <>
              {/* Status Badge */}
              <div className="status-badge success">
                ⚛️ Q-NarwhalKnight v7.3.0 — Mainnet 2026.2 (Genesis: February 22, 2026 12:00 UTC)
              </div>

              {/* Key Highlights */}
              <div className="info-box warning">
                <h3>⚛️ Mainnet 2026.2: Zero-Contamination Relaunch</h3>
                <p className="lead">
                  <strong>Fresh data directory, new identity keys, clean genesis.</strong> All mainnet2026.1 data stays behind.
                </p>
                <div style={{background: 'rgba(76,175,80,0.1)', padding: '15px', borderRadius: '8px', margin: '10px 0'}}>
                  <h4 style={{color: '#51cf66', marginTop: 0}}>What Changed</h4>
                  <table style={{width: '100%', borderCollapse: 'collapse', marginTop: '10px'}}>
                    <thead>
                      <tr style={{borderBottom: '2px solid rgba(255,255,255,0.2)'}}>
                        <th style={{textAlign: 'left', padding: '8px'}}>Detail</th>
                        <th style={{textAlign: 'right', padding: '8px'}}>Mainnet 2026.1</th>
                        <th style={{textAlign: 'right', padding: '8px', color: '#51cf66'}}>Mainnet 2026.2</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr style={{borderBottom: '1px solid rgba(255,255,255,0.1)'}}>
                        <td style={{padding: '8px'}}>Network ID</td>
                        <td style={{textAlign: 'right', padding: '8px'}}>mainnet2026.1</td>
                        <td style={{textAlign: 'right', padding: '8px', color: '#51cf66', fontWeight: 'bold'}}>mainnet2026.2</td>
                      </tr>
                      <tr style={{borderBottom: '1px solid rgba(255,255,255,0.1)'}}>
                        <td style={{padding: '8px'}}>Chain ID</td>
                        <td style={{textAlign: 'right', padding: '8px'}}>999</td>
                        <td style={{textAlign: 'right', padding: '8px', color: '#51cf66', fontWeight: 'bold'}}>1000</td>
                      </tr>
                      <tr style={{borderBottom: '1px solid rgba(255,255,255,0.1)'}}>
                        <td style={{padding: '8px'}}>Data Directory</td>
                        <td style={{textAlign: 'right', padding: '8px'}}>data-mainnet2026.1</td>
                        <td style={{textAlign: 'right', padding: '8px', color: '#51cf66', fontWeight: 'bold'}}>data-mainnet2026.2</td>
                      </tr>
                      <tr style={{borderBottom: '1px solid rgba(255,255,255,0.1)'}}>
                        <td style={{padding: '8px'}}>Genesis</td>
                        <td style={{textAlign: 'right', padding: '8px'}}>Feb 15, 2026</td>
                        <td style={{textAlign: 'right', padding: '8px', color: '#51cf66', fontWeight: 'bold'}}>Feb 22, 2026 12:00 UTC</td>
                      </tr>
                      <tr>
                        <td style={{padding: '8px', fontWeight: 'bold'}}>Emission</td>
                        <td style={{textAlign: 'right', padding: '8px'}}>2,625,000 QUG/year</td>
                        <td style={{textAlign: 'right', padding: '8px', color: '#51cf66', fontWeight: 'bold'}}>2,625,000 QUG/year (same)</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Upgrade Instructions */}
              <div className="detail-section highlight">
                <h3>✨ How to Upgrade</h3>
                <ul className="bullet-list">
                  <li>📥 <strong>Download v10.5.3</strong> — Latest binary with gap-aware sync + mining load balancing</li>
                  <li>📂 <strong>Fresh data directory</strong> — Automatically uses data-mainnet2026.2</li>
                  <li>🔑 <strong>New identity keys</strong> — Generated on first boot</li>
                  <li>🌐 <strong>P2P isolation</strong> — Old mainnet2026.1 nodes cannot connect</li>
                  <li>⛏️ <strong>Same emission model</strong> — 2,625,000 QUG/year (Era 0), 21M max supply, 4-year halving</li>
                </ul>
              </div>

              {/* Download Section */}
              <div className="action-section">
                <h3>📥 Download v10.5.3</h3>
                <p>Download the latest binary:</p>
                <div className="download-links">
                  <a href="https://quillon.xyz/downloads/q-api-server-v10.5.3" className="download-btn" download style={{marginRight: '10px'}}>
                    📦 Node v10.5.3
                  </a>
                  <a href="https://quillon.xyz/downloads/q-miner-v10.5.3" className="download-btn" download>
                    ⛏️ Miner v10.5.3
                  </a>
                </div>
                <div className="code-block">
                  <code>
                    # Stop your old node<br/>
                    pkill -f q-api-server<br/><br/>
                    # Download v10.5.3<br/>
                    wget https://quillon.xyz/downloads/q-api-server-v10.5.3<br/>
                    wget https://quillon.xyz/downloads/q-miner-v10.5.3<br/>
                    chmod +x q-api-server-v10.5.3 q-miner-v10.5.3<br/><br/>
                    # Start node<br/>
                    ./q-api-server-v10.5.3 --port 8080<br/><br/>
                    # Start mining<br/>
                    ./q-miner-v10.5.3 --mode solo --wallet YOUR_WALLET --threads 4
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
                  <h4>Technical Details</h4>
                  <div className="code-block">
                    <code>
                      // Mainnet 2026.2 Configuration<br/>
                      Network ID: mainnet2026.2<br/>
                      Chain ID: 1000<br/>
                      Genesis: Feb 22, 2026 12:00:00 UTC (1771761600)<br/>
                      Data Path: ./data-mainnet2026.2<br/>
                      Max Supply: 21,000,000 QUG<br/><br/>
                      // Gossipsub Topics<br/>
                      /qnk/mainnet2026.2/blocks<br/>
                      /qnk/mainnet2026.2/peer-heights<br/>
                      /qnk/mainnet2026.2/turbo-sync-request<br/>
                      /qnk/mainnet2026.2/turbo-sync-response
                    </code>
                  </div>
                  <p className="lead" style={{marginTop: '1rem'}}>
                    <strong>Bootstrap Nodes:</strong>
                  </p>
                  <div className="code-block">
                    <code>
                      Server Beta: /ip4/185.182.185.227/tcp/9001<br/>
                      Server Gamma: /ip4/109.205.176.60/tcp/9001<br/>
                      API: https://quillon.xyz
                    </code>
                  </div>
                </div>
              )}

              {/* Important Reminder */}
              <div className="info-box warning">
                <h3>Fresh Network Launch</h3>
                <p>
                  <strong>Mainnet 2026.2 is a clean relaunch.</strong> Mainnet 2026.1 balances do NOT carry over.
                  Everyone starts fresh. Old nodes cannot connect to the new network.
                </p>
                <p style={{marginTop: '10px'}}>
                  <strong>Why?</strong> Mainnet 2026.1 had data contamination from the testnet transition.
                  A fresh directory with new identity keys ensures zero contamination and correct emission from genesis.
                </p>
              </div>
            </>
          )}

          {activeTab === 'faq' && (
            <div className="faq-section">
              <h2>Frequently Asked Questions</h2>

              <div className="faq-item">
                <h4>Q: What is mainnet2026.2?</h4>
                <p>
                  A: Mainnet 2026.2 is a clean relaunch of Q-NarwhalKnight with a fresh data directory,
                  new libp2p identity keys, and zero contamination from the previous network. Same emission model
                  (2,625,000 QUG/year, 21M max supply), better isolation.
                </p>
              </div>

              <div className="faq-item">
                <h4>Q: What happened to my mainnet2026.1 balance?</h4>
                <p>
                  A: Mainnet 2026.1 balances do NOT transfer. This is a fresh network — everyone starts equal.
                  The previous network had data contamination issues that made a clean restart necessary.
                </p>
              </div>

              <div className="faq-item">
                <h4>Q: Do I need to delete my old data?</h4>
                <p>
                  A: No. The new binary automatically uses data-mainnet2026.2 as its data directory.
                  Your old data-mainnet2026.1 directory remains untouched but is no longer used.
                </p>
              </div>

              <div className="faq-item">
                <h4>Q: Can old nodes connect to mainnet2026.2?</h4>
                <p>
                  A: No. The protocol handshake validates network_id. Old nodes running mainnet2026.1 will be
                  rejected during connection. Gossipsub topics are also different. All nodes must upgrade to v7.3.0.
                </p>
              </div>

              <div className="faq-item">
                <h4>Q: Is the emission model the same?</h4>
                <p>
                  A: Yes. Same 21M max supply, same 2,625,000 QUG/year target, same 4-year halving cycle.
                  The adaptive emission controller ensures consistent daily emission regardless of block rate.
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
              ⚛️ Announcement
            </button>
            <button
              className={`tab-btn ${activeTab === 'faq' ? 'active' : ''}`}
              onClick={() => setActiveTab('faq')}
            >
              ❓ FAQ
            </button>
          </div>
          <button className="primary-btn" onClick={onClose}>
            Start Mining on Mainnet 2026.2 →
          </button>
        </div>
      </div>
    </div>
  );
};

export default PhaseTransitionModal;
