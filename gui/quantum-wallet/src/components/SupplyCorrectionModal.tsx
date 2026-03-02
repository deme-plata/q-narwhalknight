import { useState, useEffect } from 'react';
import './SupplyCorrectionModal.css';

const STORAGE_KEY = 'supply_correction_v857_seen';

interface EmissionStats {
  total_supply_qug: number;
  cumulative_target_qug: number;
  budget_deviation_pct: number;
  correction_factor: number;
  stock_to_flow: number;
  max_supply_qug: number;
  annual_target_qug: number;
  genesis_timestamp: number;
  block_rate_bps: number;
  era: number;
  blocks_produced: number;
}

const SupplyCorrectionModal: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<'notice' | 'proof'>('notice');
  const [stats, setStats] = useState<EmissionStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Listen for custom event to open
  useEffect(() => {
    const handler = () => setIsOpen(true);
    window.addEventListener('open-supply-correction', handler);
    return () => window.removeEventListener('open-supply-correction', handler);
  }, []);

  // Fetch emission stats when proof tab is opened
  useEffect(() => {
    if (!isOpen || activeTab !== 'proof' || stats) return;

    const fetchStats = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch('/api/v1/emission/stats?days=7');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        const data = json.data || json;
        const s = data.summary || {};
        const history = data.daily_history || [];
        const totalBlocks = history.reduce((sum: number, r: any) => sum + (r.blocks || 0), 0);

        setStats({
          total_supply_qug: s.total_supply_qug || 0,
          cumulative_target_qug: s.cumulative_target_qug || 0,
          budget_deviation_pct: s.budget_deviation_pct || 0,
          correction_factor: s.correction_factor || 1.0,
          stock_to_flow: s.stock_to_flow || 0,
          max_supply_qug: s.max_supply_qug || 21_000_000,
          annual_target_qug: s.annual_target_qug || 2_625_000,
          genesis_timestamp: s.genesis_timestamp || 1771761600,
          block_rate_bps: s.block_rate_bps || 0,
          era: s.current_era ?? 0,
          blocks_produced: totalBlocks || s.today_blocks || 0,
        });
      } catch (e: any) {
        setError(e.message || 'Failed to fetch emission data');
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, [isOpen, activeTab, stats]);

  const handleClose = () => {
    localStorage.setItem(STORAGE_KEY, 'true');
    setIsOpen(false);
  };

  if (!isOpen) return null;

  return (
    <div className="supply-correction-overlay" onClick={(e) => { if (e.target === e.currentTarget) handleClose(); }}>
      <div className="supply-correction-modal">
        {/* Header */}
        <div className="sc-header">
          <span className="sc-header-icon">&#x26A0;&#xFE0F;</span>
          <h1 className="sc-header-title">Supply Correction Notice</h1>
          <button className="sc-close-btn" onClick={handleClose} aria-label="Close">&times;</button>
        </div>

        {/* Content */}
        <div className="sc-content">
          {activeTab === 'notice' && (
            <>
              <div className="sc-badge">
                v8.5.7 — Emission Integrity Fix
              </div>

              <div className="sc-info-box">
                <h3>What Happened</h3>
                <p>
                  A supply inflation bug caused the network to display approximately <strong>~144,000 QUG</strong> in
                  total circulating supply, when the mathematically correct amount based on the emission schedule is
                  approximately <strong>~31,600 QUG</strong>.
                </p>
              </div>

              <div className="sc-info-box">
                <h3>Why It Happened</h3>
                <p>
                  The static reward formula assumed a block production rate of 1 block per second, but the actual
                  network rate was approximately 2 blocks per second. This resulted in roughly <strong>4.5x overcrediting</strong> of
                  mining rewards over the affected period.
                </p>
              </div>

              <div className="sc-info-box" style={{ borderLeftColor: '#00ff88', background: 'rgba(0, 255, 136, 0.08)' }}>
                <h3 style={{ color: '#00ff88' }}>What Changed</h3>
                <p>
                  All balances have been recalculated using the emission controller's exact tracked totals,
                  distributed <strong>proportionally by each miner's block count</strong>. Your share of mining
                  work is fully preserved — only the total reward pool was corrected to match the emission
                  schedule.
                </p>
              </div>

              <div style={{
                background: 'rgba(0, 0, 0, 0.3)',
                padding: '16px',
                borderRadius: '10px',
                margin: '16px 0',
                border: '1px solid rgba(255, 255, 255, 0.08)',
              }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid rgba(255, 255, 255, 0.15)' }}>
                      <th style={{ textAlign: 'left', padding: '8px', color: 'rgba(255,255,255,0.5)', fontSize: '12px', textTransform: 'uppercase' }}>Detail</th>
                      <th style={{ textAlign: 'right', padding: '8px', color: '#ff6b6b', fontSize: '12px', textTransform: 'uppercase' }}>Before (Bug)</th>
                      <th style={{ textAlign: 'right', padding: '8px', color: '#00ff88', fontSize: '12px', textTransform: 'uppercase' }}>After (Corrected)</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                      <td style={{ padding: '10px 8px', fontSize: '14px' }}>Total Supply</td>
                      <td style={{ textAlign: 'right', padding: '10px 8px', color: '#ff6b6b', fontFamily: 'monospace' }}>~144,000 QUG</td>
                      <td style={{ textAlign: 'right', padding: '10px 8px', color: '#00ff88', fontWeight: 700, fontFamily: 'monospace' }}>~31,600 QUG</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                      <td style={{ padding: '10px 8px', fontSize: '14px' }}>Reward per Block</td>
                      <td style={{ textAlign: 'right', padding: '10px 8px', color: '#ff6b6b', fontFamily: 'monospace' }}>Inflated</td>
                      <td style={{ textAlign: 'right', padding: '10px 8px', color: '#00ff88', fontWeight: 700, fontFamily: 'monospace' }}>Emission-tracked</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                      <td style={{ padding: '10px 8px', fontSize: '14px' }}>Your Mining Share</td>
                      <td style={{ textAlign: 'right', padding: '10px 8px', color: 'rgba(255,255,255,0.6)' }}>X%</td>
                      <td style={{ textAlign: 'right', padding: '10px 8px', color: '#00ff88', fontWeight: 700 }}>X% (preserved)</td>
                    </tr>
                    <tr>
                      <td style={{ padding: '10px 8px', fontSize: '14px', fontWeight: 700 }}>Max Supply</td>
                      <td style={{ textAlign: 'right', padding: '10px 8px', fontFamily: 'monospace' }}>21,000,000</td>
                      <td style={{ textAlign: 'right', padding: '10px 8px', color: '#00ff88', fontWeight: 700, fontFamily: 'monospace' }}>21,000,000</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="sc-info-box" style={{ borderLeftColor: '#00ccff', background: 'rgba(0, 204, 255, 0.06)' }}>
                <h3 style={{ color: '#00ccff' }}>Emission Schedule</h3>
                <p>
                  <strong>Annual emission (Era 0):</strong> 2,625,000 QUG/year<br />
                  <strong>Halving cycle:</strong> Every 4 years<br />
                  <strong>Hard cap:</strong> 21,000,000 QUG (never exceeded)<br />
                  <strong>Formula:</strong> Annual = 2,625,000 / 2<sup>era</sup>
                </p>
                <p style={{ marginTop: '10px' }}>
                  Click the <strong>"Blockchain Proof"</strong> tab below to see live on-chain verification data.
                </p>
              </div>
            </>
          )}

          {activeTab === 'proof' && (
            <>
              <div className="sc-badge">
                Live On-Chain Data — /api/v1/emission/stats
              </div>

              {loading && (
                <div className="sc-loading">
                  <div className="sc-spinner" />
                  <p>Fetching emission data from blockchain...</p>
                </div>
              )}

              {error && (
                <div className="sc-info-box" style={{ borderLeftColor: '#ff4444', background: 'rgba(255,68,68,0.08)' }}>
                  <h3 style={{ color: '#ff4444' }}>Failed to Load</h3>
                  <p>{error}</p>
                  <button
                    onClick={() => { setStats(null); setError(null); }}
                    style={{
                      marginTop: '10px',
                      background: 'rgba(255,255,255,0.1)',
                      color: '#fff',
                      border: '1px solid rgba(255,255,255,0.2)',
                      padding: '6px 16px',
                      borderRadius: '6px',
                      cursor: 'pointer',
                    }}
                  >
                    Retry
                  </button>
                </div>
              )}

              {stats && !loading && (
                <>
                  <div className="sc-proof-section">
                    <h3>Emission Verification</h3>
                    <div className="sc-proof-grid">
                      <div className="sc-proof-item">
                        <div className="sc-proof-label">Total Supply (Corrected)</div>
                        <div className="sc-proof-value">{stats.total_supply_qug.toLocaleString(undefined, { maximumFractionDigits: 2 })} QUG</div>
                      </div>
                      <div className="sc-proof-item">
                        <div className="sc-proof-label">Target Supply</div>
                        <div className="sc-proof-value neutral">{stats.cumulative_target_qug.toLocaleString(undefined, { maximumFractionDigits: 2 })} QUG</div>
                      </div>
                      <div className="sc-proof-item">
                        <div className="sc-proof-label">Budget Deviation</div>
                        <div className="sc-proof-value highlight">{stats.budget_deviation_pct >= 0 ? '+' : ''}{stats.budget_deviation_pct.toFixed(2)}%</div>
                      </div>
                      <div className="sc-proof-item">
                        <div className="sc-proof-label">Correction Factor</div>
                        <div className="sc-proof-value neutral">{stats.correction_factor.toFixed(4)}x</div>
                      </div>
                      <div className="sc-proof-item">
                        <div className="sc-proof-label">Stock-to-Flow Ratio</div>
                        <div className="sc-proof-value">{stats.stock_to_flow.toFixed(2)}</div>
                      </div>
                      <div className="sc-proof-item">
                        <div className="sc-proof-label">Current Era</div>
                        <div className="sc-proof-value neutral">Era {stats.era}</div>
                      </div>
                    </div>
                  </div>

                  <div className="sc-proof-section" style={{ borderColor: 'rgba(0, 204, 255, 0.2)', background: 'rgba(0, 204, 255, 0.03)' }}>
                    <h3 style={{ color: '#00ccff' }}>Hard Limits</h3>
                    <div className="sc-proof-grid">
                      <div className="sc-proof-item">
                        <div className="sc-proof-label">Max Supply (Hard Cap)</div>
                        <div className="sc-proof-value">{stats.max_supply_qug.toLocaleString()} QUG</div>
                      </div>
                      <div className="sc-proof-item">
                        <div className="sc-proof-label">Annual Emission (Era 0)</div>
                        <div className="sc-proof-value neutral">{stats.annual_target_qug.toLocaleString()} QUG</div>
                      </div>
                      <div className="sc-proof-item">
                        <div className="sc-proof-label">Block Rate</div>
                        <div className="sc-proof-value highlight">{stats.block_rate_bps.toFixed(1)} bps</div>
                      </div>
                      <div className="sc-proof-item">
                        <div className="sc-proof-label">Blocks (7d)</div>
                        <div className="sc-proof-value neutral">{stats.blocks_produced.toLocaleString()}</div>
                      </div>
                    </div>
                  </div>

                  <div className="sc-formula-box">
                    <code>
                      {/* Emission formula */}
                      Emission Formula:<br />
                      &nbsp;&nbsp;Annual = 2,625,000 QUG / 2^era<br />
                      &nbsp;&nbsp;Era 0: 2,625,000 QUG/year (current)<br />
                      &nbsp;&nbsp;Era 1: 1,312,500 QUG/year (after 4 years)<br />
                      &nbsp;&nbsp;Era 2: &nbsp;&nbsp;656,250 QUG/year (after 8 years)<br />
                      &nbsp;&nbsp;...<br />
                      &nbsp;&nbsp;Hard cap: 21,000,000 QUG (never exceeded)<br /><br />
                      Halving Interval: 4 years<br />
                      Adaptive Controller: Adjusts per-block reward<br />
                      &nbsp;&nbsp;to match target emission rate regardless<br />
                      &nbsp;&nbsp;of actual block production speed.
                    </code>
                  </div>

                  <div className="sc-info-box" style={{ borderLeftColor: '#00ff88', background: 'rgba(0, 255, 136, 0.06)' }}>
                    <h3 style={{ color: '#00ff88' }}>Verification</h3>
                    <p>
                      You can independently verify this data by querying the endpoint:<br />
                      <code style={{ color: '#00ccff' }}>GET /api/v1/emission/stats?days=30</code><br /><br />
                      The emission controller tracks every block's reward and ensures the total supply
                      never deviates from the mathematically correct target.
                    </p>
                  </div>
                </>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="sc-footer">
          <div style={{ display: 'flex', gap: '10px' }}>
            <button
              className={`sc-tab-btn ${activeTab === 'notice' ? 'active' : ''}`}
              onClick={() => setActiveTab('notice')}
            >
              Notice
            </button>
            <button
              className={`sc-tab-btn ${activeTab === 'proof' ? 'active' : ''}`}
              onClick={() => setActiveTab('proof')}
            >
              Blockchain Proof
            </button>
          </div>
          <button className="sc-got-it-btn" onClick={handleClose}>
            Got it
          </button>
        </div>
      </div>
    </div>
  );
};

export default SupplyCorrectionModal;
