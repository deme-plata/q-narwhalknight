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
    localStorage.setItem('phase12PQCSecurityModalSeen', 'true');
  }, []);

  return (
    <div className="phase-transition-overlay">
      <div className="phase-transition-modal">
        {/* Header */}
        <div className="modal-header">
          <span className="quantum-logo">🔐</span>
          <h1 className="modal-title">v1.0.12-beta: Phase 12 - Post-Quantum Security (0.05 QUG/block)</h1>
          <button className="close-btn" onClick={onClose} aria-label="Close">&times;</button>
        </div>

        {/* Content */}
        <div className="modal-content">
          {activeTab === 'announcement' && (
            <>
              {/* Status Badge */}
              <div className="status-badge success">
                🔐 v1.0.12-beta: Phase 12 - Post-Quantum Security + ZK Proofs (0.05 QUG/block)
              </div>

              {/* Critical PQC Security Highlight */}
              <div className="info-box warning">
                <h3>🔐 Phase 12: Post-Quantum Security ACTIVE!</h3>
                <p className="lead">
                  <strong>Phase 12 achieves QUANTUM RESISTANCE!</strong> Active Dilithium5 signatures, encrypted key storage, and ZK untrusted setup!
                </p>
                <div style={{background: 'rgba(76,175,80,0.1)', padding: '15px', borderRadius: '8px', margin: '10px 0'}}>
                  <h4 style={{color: '#51cf66', marginTop: 0}}>🔐 Post-Quantum Security Features</h4>
                  <table style={{width: '100%', borderCollapse: 'collapse', marginTop: '10px'}}>
                    <thead>
                      <tr style={{borderBottom: '2px solid rgba(255,255,255,0.2)'}}>
                        <th style={{textAlign: 'left', padding: '8px'}}>Feature</th>
                        <th style={{textAlign: 'right', padding: '8px'}}>Phase 11 (Pre-PQC)</th>
                        <th style={{textAlign: 'right', padding: '8px', color: '#51cf66'}}>Phase 12 (PQC ACTIVE!) ✅</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr style={{borderBottom: '1px solid rgba(255,255,255,0.1)'}}>
                        <td style={{padding: '8px'}}>Quantum Resistance</td>
                        <td style={{textAlign: 'right', padding: '8px', color: '#ff6b6b'}}>Classical only ❌</td>
                        <td style={{textAlign: 'right', padding: '8px', color: '#51cf66', fontWeight: 'bold'}}>Dilithium5 ACTIVE ✅</td>
                      </tr>
                      <tr style={{borderBottom: '1px solid rgba(255,255,255,0.1)'}}>
                        <td style={{padding: '8px'}}>Key Storage</td>
                        <td style={{textAlign: 'right', padding: '8px', color: '#ff6b6b', fontWeight: 'bold'}}>Plaintext (vulnerable) ❌</td>
                        <td style={{textAlign: 'right', padding: '8px', color: '#51cf66', fontWeight: 'bold'}}>AES-256-GCM encrypted ✅</td>
                      </tr>
                      <tr style={{borderBottom: '1px solid rgba(255,255,255,0.1)'}}>
                        <td style={{padding: '8px'}}>ZK Proofs</td>
                        <td style={{textAlign: 'right', padding: '8px', color: '#ff6b6b', fontWeight: 'bold'}}>Not implemented ❌</td>
                        <td style={{textAlign: 'right', padding: '8px', color: '#51cf66', fontWeight: 'bold'}}>STARK + SNARK ✅</td>
                      </tr>
                      <tr>
                        <td style={{padding: '8px', fontWeight: 'bold'}}>Security Level</td>
                        <td style={{textAlign: 'right', padding: '8px', color: '#ff6b6b', fontWeight: 'bold'}}>NIST Level 1 ❌</td>
                        <td style={{textAlign: 'right', padding: '8px', color: '#51cf66', fontWeight: 'bold'}}>NIST Level 5 (maximum) ✅</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <p style={{marginTop: '15px', fontSize: '0.95em'}}>
                  Phase 11 used classical Ed25519 signatures - vulnerable to quantum computers!
                  Phase 12 implements <strong>active Dilithium5 signatures</strong>, encrypted key storage, and ZK proofs for maximum security.
                </p>
              </div>

              {/* The Update */}
              <div className="detail-section highlight">
                <h3>✨ v1.0.12-beta: Post-Quantum Security Active</h3>
                <ul className="bullet-list">
                  <li>🔐 <strong>Dilithium5 Signatures</strong> - NIST Level 5 post-quantum signatures actively verified on every block</li>
                  <li>🔒 <strong>Encrypted Key Storage</strong> - AES-256-GCM + Argon2 password-based encryption</li>
                  <li>⚡ <strong>ZK Untrusted Setup</strong> - STARK + SNARK proofs without trusted setup ceremonies</li>
                  <li>🛡️ <strong>Quantum Resistance</strong> - Protected against Shor's algorithm and quantum attacks</li>
                  <li>✅ <strong>Active Integration</strong> - Signatures verified in consensus, not just scaffolding</li>
                  <li>🌐 <strong>New Network: testnet-phase12</strong> - Fresh start with PQC security</li>
                  <li>💎 <strong>Same Economics: 0.05 QUG/block</strong> - Proven sustainable scarcity model</li>
                  <li>🎯 <strong>Security Milestone</strong> - First quantum-ready consensus network</li>
                </ul>
              </div>

              {/* Phase 12 PQC Details */}
              <div className="detail-section">
                <h3>🔐 Post-Quantum Cryptography - Active Security</h3>
                <p><strong>Phase 11's limitation:</strong> Classical Ed25519 signatures vulnerable to quantum computers!</p>
                <p><strong>Phase 12's solution:</strong> Active Dilithium5 integration with complete security stack:</p>
                <ul className="bullet-list">
                  <li>🔐 <strong>Dilithium5 Signatures:</strong> Every block signed and verified with NIST Level 5 PQC</li>
                  <li>🔒 <strong>Encrypted Keys:</strong> AES-256-GCM + Argon2id key derivation (NIST FIPS 197 compliant)</li>
                  <li>⚡ <strong>STARK Proofs:</strong> Transparent, post-quantum secure (~100 KB proof size)</li>
                  <li>🚀 <strong>SNARK Proofs:</strong> Succinct, Halo2-style recursive (~1-2 KB proof size)</li>
                  <li>✅ <strong>Automatic Zeroization:</strong> Secret keys automatically cleared from memory</li>
                  <li>🌐 <strong>Network ID: testnet-phase12</strong> - Fresh start with PQC active</li>
                  <li>✅ <strong>Gossipsub topics:</strong> <code>/qnk/testnet-phase12/*</code></li>
                  <li>📂 <strong>Database: data-mine12</strong> - New clean database with PQC security</li>
                </ul>
                <div style={{background: 'rgba(76,175,80,0.1)', padding: '12px', borderRadius: '6px', marginTop: '15px'}}>
                  <p style={{margin: 0, fontSize: '0.95em'}}>
                    <strong>💡 Key Achievement:</strong> Phase 12 is the first blockchain testnet with ACTIVE post-quantum signature verification.
                    Dilithium5 signatures are verified on EVERY block - not just scaffolding, but real consensus security!
                  </p>
                </div>
              </div>

              {/* Download Section */}
              <div className="action-section">
                <h3>📥 Upgrade to v1.0.12-beta (Phase 12 - PQC Security!)</h3>
                <p>Download Phase 12 binaries with active post-quantum cryptography:</p>
                <div className="download-links">
                  <a href="/downloads/q-api-server-v1.0.12-beta" className="download-btn" download style={{marginRight: '10px'}}>
                    📦 Node v1.0.12-beta
                  </a>
                  <a href="/downloads/q-miner-linux-x64" className="download-btn" download>
                    ⛏️ Miner (Latest)
                  </a>
                </div>
                <div className="code-block">
                  <code>
                    # Stop your node<br/>
                    killall q-api-server<br/><br/>
                    # Download Phase 12 versions<br/>
                    wget https://quillon.xyz/downloads/q-api-server-v1.0.12-beta<br/>
                    wget https://quillon.xyz/downloads/q-miner-linux-x64<br/>
                    chmod +x q-api-server-v1.0.12-beta q-miner-linux-x64<br/><br/>
                    # Start node (automatic fresh database: data-mine12)<br/>
                    ./q-api-server-v1.0.12-beta --port 8080<br/><br/>
                    # Start mining on Phase 12 network (PQC SECURITY: 0.05 QUG/block!)<br/>
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
                    The v1.0.12-beta release implements Phase 12 with active post-quantum cryptography and ZK proofs:
                  </p>
                  <div className="code-block">
                    <code>
                      // Phase 12 Network Configuration<br/>
                      const NETWORK_ID: &str = "testnet-phase12";<br/>
                      const DEFAULT_PHASE: u8 = 12;<br/>
                      const FIXED_BLOCK_REWARD: u64 = 5_000_000; // 0.05 QUG (8 decimals)<br/>
                      const DATABASE_PATH: &str = "./data-mine12";<br/>
                      const HALVING_BLOCKS: u64 = 210_000; // ~4 years<br/><br/>
                      // Post-Quantum Security (v1.0.12-beta)<br/>
                      // Dilithium5 signatures: ACTIVE (NIST Level 5) ✅<br/>
                      // Encrypted keys: AES-256-GCM + Argon2 ✅<br/>
                      // ZK proofs: STARK + SNARK (untrusted setup) ✅<br/><br/>
                      // Gossipsub Topics (Phase 12)<br/>
                      /qnk/testnet-phase12/blocks<br/>
                      /qnk/testnet-phase12/peer-heights<br/>
                      /qnk/testnet-phase12/block-pack-requests<br/>
                      /qnk/testnet-phase12/block-pack-responses
                    </code>
                  </div>
                  <p className="lead" style={{marginTop: '1rem'}}>
                    <strong>Server Beta Bootstrap Node (Phase 12):</strong>
                  </p>
                  <div className="code-block">
                    <code>
                      /ip4/185.182.185.227/tcp/9001/p2p/12D3KooWRX3GGK9Fs3iM3BfqYNNJiHBDujac7EHwqWjaK1n1kzPN<br/>
                      Network ID: testnet-phase12<br/>
                      API: https://quillon.xyz<br/>
                      Genesis Height: 0 (fresh start)<br/>
                      Block Reward: 0.05 QUG (same scarcity model)<br/>
                      Security: Post-quantum ready (Dilithium5 active)
                    </code>
                  </div>
                </div>
              )}

              {/* Important Reminder */}
              <div className="info-box warning">
                <h3>💡 Important: Fresh Network, Post-Quantum Security</h3>
                <p>
                  <strong>Phase 12 is a complete network reset with ACTIVE PQC.</strong> Phase 11 balances do NOT transfer to Phase 12.
                  Everyone starts equal - this is a FRESH START with quantum-ready security. Phase 11 nodes CANNOT connect to Phase 12 network.
                  All nodes must upgrade to v1.0.12-beta.
                </p>
                <p style={{marginTop: '10px'}}>
                  <strong>Why fresh start?</strong> Phase 12 implements fundamental security upgrades - active Dilithium5 signatures, encrypted key storage, and ZK proofs.
                  Fresh starts ensure Phase 12's quantum-ready architecture works correctly from genesis with maximum security.
                </p>
              </div>
            </>
          )}

          {activeTab === 'faq' && (
            <div className="faq-section">
              <h2>Frequently Asked Questions</h2>

              <div className="faq-item">
                <h4>Q: What is Phase 12?</h4>
                <p>
                  A: Phase 12 is Q-NarwhalKnight's Post-Quantum Security testnet with ACTIVE Dilithium5 signatures, encrypted key storage, and ZK proofs.
                  This is a complete network reset (network ID: testnet-phase12) that achieves quantum resistance with NIST Level 5 cryptography.
                </p>
              </div>

              <div className="faq-item">
                <h4>Q: What's new in Phase 12?</h4>
                <p>
                  A: Phase 12 introduces three major security upgrades: 1) Active Dilithium5 signatures verified on every block (quantum-resistant),
                  2) AES-256-GCM encrypted key storage with Argon2 password derivation, 3) ZK untrusted setup with STARK + SNARK proofs.
                  Phase 11 only had classical Ed25519 signatures - vulnerable to quantum computers!
                </p>
              </div>

              <div className="faq-item">
                <h4>Q: What happened to my Phase 11 balance?</h4>
                <p>
                  A: Phase 11 balances do NOT transfer to Phase 12. This is a fresh network with PQC security - everyone starts equal (fair launch!).
                  Testnet balances have NO VALUE - this is expected. Phase 12 tests REAL mainnet quantum resistance.
                </p>
              </div>

              <div className="faq-item">
                <h4>Q: Do I need to reset my database?</h4>
                <p>
                  A: Phase 12 automatically uses a new database path (data-mine12). Your Phase 11 database remains intact in the old location.
                  No manual reset needed - just download v1.0.12-beta and start mining on the quantum-ready network!
                </p>
              </div>

              <div className="faq-item">
                <h4>Q: What is Dilithium5?</h4>
                <p>
                  A: Dilithium5 is NIST's Level 5 post-quantum digital signature algorithm - the HIGHEST security level available.
                  It's resistant to quantum computer attacks (Shor's algorithm). Phase 12 uses Dilithium5 to sign and verify EVERY block,
                  making it the first blockchain testnet with active post-quantum signature verification!
                </p>
              </div>

              <div className="faq-item">
                <h4>Q: Will mainnet use post-quantum cryptography?</h4>
                <p>
                  A: YES! Phase 12's Dilithium5 signatures, encrypted key storage, and ZK proofs will be the mainnet model.
                  We're testing PQC security in Phase 12 before mainnet launch. If you run Phase 12, you're practicing with REAL mainnet quantum resistance!
                </p>
              </div>

              <div className="faq-item">
                <h4>Q: Can Phase 11 nodes connect to Phase 12?</h4>
                <p>
                  A: No! Phase 11 (testnet-phase11) and Phase 12 (testnet-phase12) use different network IDs, gossipsub topics, databases, and cryptographic protocols.
                  They cannot communicate. All nodes must upgrade to v1.0.12-beta to join Phase 12.
                </p>
              </div>

              <div className="faq-item">
                <h4>Q: What are ZK proofs?</h4>
                <p>
                  A: Zero-Knowledge proofs let you prove you possess secret keys without revealing them. Phase 12 implements both STARK (transparent, ~100 KB) and
                  SNARK (succinct, ~1-2 KB) proofs WITHOUT trusted setup ceremonies. This enables privacy-preserving validator registration, threshold signatures,
                  and anonymous voting!
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
              🔐 Announcement
            </button>
            <button
              className={`tab-btn ${activeTab === 'faq' ? 'active' : ''}`}
              onClick={() => setActiveTab('faq')}
            >
              ❓ FAQ
            </button>
          </div>
          <button className="primary-btn" onClick={onClose}>
            Start Mining Phase 12 (PQC SECURITY!) →
          </button>
        </div>
      </div>
    </div>
  );
};

export default PhaseTransitionModal;
