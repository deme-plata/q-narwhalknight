import React, { useState, useEffect } from 'react';

interface MinerLoginPageProps {
  deviceCode: string;
}

// Theme colors matching Slint wallet oauth_consent.slint
const T = {
  bgPrimary: '#0a0e1a',
  bgSecondary: '#111827',
  bgInput: '#1e2642',
  accentBlue: '#3b82f6',
  accentGreen: '#10b981',
  accentRed: '#ef4444',
  accentCyan: '#06b6d4',
  textPrimary: '#f1f5f9',
  textSecondary: '#94a3b8',
  textMuted: '#64748b',
  border: '#2a3550',
  radiusMd: '12px',
  radiusSm: '8px',
};

const scopes = [
  { name: 'Mining Rewards', description: 'Receive mining rewards to your wallet' },
  { name: 'Read Balance', description: 'View your current wallet balance' },
  { name: 'Submit Work', description: 'Submit proof-of-work on your behalf' },
];

const MinerLoginPage: React.FC<MinerLoginPageProps> = ({ deviceCode }) => {
  const [walletAddress, setWalletAddress] = useState('');
  const [manualAddress, setManualAddress] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const addr = localStorage.getItem('walletAddress') || '';
    setWalletAddress(addr);
  }, []);

  const doComplete = async (code: string, wallet: string): Promise<{ success: boolean; error?: string }> => {
    const resp = await fetch(`/api/v1/miner/device-login/complete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ device_code: code, wallet_address: wallet }),
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => '');
      if (text.startsWith('<!') || text.startsWith('<html')) {
        throw new Error(`Server busy (${resp.status}). Retrying...`);
      }
      throw new Error(text || `HTTP ${resp.status}`);
    }
    const ct = resp.headers.get('content-type') || '';
    if (!ct.includes('json')) {
      throw new Error(`Server busy. Retrying...`);
    }
    return resp.json();
  };

  const handleApprove = async () => {
    if (!deviceCode) {
      setError('Invalid login link. Please restart the miner.');
      return;
    }
    if (!walletAddress) {
      setError('No wallet found. Please log in first.');
      return;
    }

    setLoading(true);
    setError('');

    // Retry up to 3 times (server may be momentarily busy)
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const data = await doComplete(deviceCode, walletAddress);
        if (data.success) {
          setSuccess(true);
          return;
        } else {
          setError((data as any).message || (data as any).error || 'Authorization failed. Code may have expired.');
          setLoading(false);
          return;
        }
      } catch (err: any) {
        if (attempt < 2 && (err.message || '').includes('Retrying')) {
          await new Promise(r => setTimeout(r, 1500));
          continue;
        }
        setError(err.message || 'Network error. Please try again.');
        setLoading(false);
        return;
      }
    }

    setLoading(false);
  };

  const handleDeny = () => {
    window.close();
    // Fallback if window.close() is blocked
    setError('Denied. You can close this tab.');
  };

  // Success screen
  if (success) {
    return (
      <div style={containerStyle}>
        <div style={{ ...cardOuter, border: `1px solid ${T.accentGreen}40` }}>
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>&#x2705;</div>
          <div style={{ color: T.accentGreen, fontSize: '20px', fontWeight: 700, marginBottom: '8px' }}>
            Miner Authorized
          </div>
          <div style={{ color: T.textSecondary, fontSize: '14px', lineHeight: '1.6' }}>
            Mining rewards will go to your wallet.
            <br />You can close this tab.
          </div>
        </div>
      </div>
    );
  }

  // Not logged in — let user enter wallet address directly
  if (!walletAddress) {
    const handleManualApprove = async () => {
      const addr = manualAddress.trim();
      if (!addr) {
        setError('Please enter your wallet address.');
        return;
      }
      if (!addr.startsWith('qnk') || addr.length < 60) {
        setError('Invalid wallet address. Must start with "qnk" and be at least 60 characters.');
        return;
      }
      if (!deviceCode) {
        setError('Invalid login link. Please restart the miner.');
        return;
      }
      setLoading(true);
      setError('');
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          const data = await doComplete(deviceCode, addr);
          if (data.success) {
            setSuccess(true);
            return;
          } else {
            setError((data as any).message || (data as any).error || 'Authorization failed. Code may have expired.');
            setLoading(false);
            return;
          }
        } catch (err: any) {
          if (attempt < 2 && (err.message || '').includes('Retrying')) {
            await new Promise(r => setTimeout(r, 1500));
            continue;
          }
          setError(err.message || 'Network error. Please try again.');
          setLoading(false);
          return;
        }
      }
      setLoading(false);
    };

    return (
      <div style={containerStyle}>
        <div style={cardOuter}>
          <div style={{ color: T.accentCyan, fontSize: '20px', fontWeight: 700, marginBottom: '20px', textAlign: 'center' }}>
            Authorization Request
          </div>

          <div style={cardInner}>
            <div style={{ color: T.textPrimary, fontSize: '18px', fontWeight: 600, marginBottom: '6px' }}>
              Quillon Miner
            </div>
            <div style={{ color: T.textSecondary, fontSize: '14px', marginBottom: '16px' }}>
              wants to send mining rewards to your wallet
            </div>

            {/* Scopes preview */}
            {scopes.map((scope, i) => (
              <div key={i} style={{
                height: '40px',
                borderRadius: T.radiusSm,
                background: T.bgSecondary,
                display: 'flex',
                alignItems: 'center',
                padding: '0 12px',
                gap: '10px',
                marginBottom: i < scopes.length - 1 ? '6px' : '0',
              }}>
                <div style={{
                  width: '20px', height: '20px', borderRadius: '4px',
                  background: `${T.accentGreen}33`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                }}>
                  <span style={{ color: T.accentGreen, fontSize: '12px' }}>&#x2713;</span>
                </div>
                <div style={{ textAlign: 'left' }}>
                  <div style={{ color: T.textPrimary, fontSize: '12px', fontWeight: 500 }}>{scope.name}</div>
                  <div style={{ color: T.textMuted, fontSize: '10px' }}>{scope.description}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Wallet address input */}
          <div style={cardInner}>
            <div style={{ color: T.textPrimary, fontSize: '14px', fontWeight: 600, marginBottom: '10px', textAlign: 'left' }}>
              Enter your wallet address
            </div>
            <input
              type="text"
              placeholder="qnk..."
              value={manualAddress}
              onChange={(e) => { setManualAddress(e.target.value); setError(''); }}
              style={{
                width: '100%',
                padding: '10px 12px',
                background: T.bgInput,
                border: `1px solid ${T.border}`,
                borderRadius: T.radiusSm,
                color: T.textPrimary,
                fontSize: '13px',
                fontFamily: 'monospace',
                outline: 'none',
                boxSizing: 'border-box',
              }}
              onFocus={(e) => { e.target.style.borderColor = T.accentBlue; }}
              onBlur={(e) => { e.target.style.borderColor = T.border; }}
            />
            <div style={{ color: T.textMuted, fontSize: '11px', marginTop: '6px', textAlign: 'left' }}>
              Paste the wallet address where mining rewards should be sent.
            </div>
          </div>

          {/* Error */}
          {error && (
            <div style={{
              background: `${T.accentRed}18`,
              border: `1px solid ${T.accentRed}40`,
              borderRadius: T.radiusSm,
              padding: '10px 14px',
              color: T.accentRed,
              fontSize: '13px',
            }}>
              {error}
            </div>
          )}

          {/* Buttons */}
          <div style={{ display: 'flex', gap: '12px' }}>
            <button onClick={handleDeny} style={{ ...btnStyle, background: T.accentRed, flex: 1 }}>
              Deny
            </button>
            <button
              onClick={handleManualApprove}
              disabled={loading}
              style={{ ...btnStyle, background: loading ? T.bgSecondary : T.accentGreen, flex: 1, cursor: loading ? 'wait' : 'pointer' }}
            >
              {loading ? 'Approving...' : 'Approve'}
            </button>
          </div>

          <div style={{ color: T.textMuted, fontSize: '11px', textAlign: 'center' }}>
            This app will NOT have access to your private keys.
          </div>
        </div>
      </div>
    );
  }

  // Consent screen — matches Slint oauth_consent.slint layout
  return (
    <div style={containerStyle}>
      <div style={cardOuter}>

        {/* Title */}
        <div style={{ color: T.accentCyan, fontSize: '20px', fontWeight: 700, marginBottom: '20px', textAlign: 'center' }}>
          Authorization Request
        </div>

        {/* App card */}
        <div style={cardInner}>
          <div style={{ color: T.textPrimary, fontSize: '18px', fontWeight: 600, marginBottom: '6px' }}>
            Quillon Miner
          </div>
          <div style={{ color: T.textSecondary, fontSize: '14px', marginBottom: '16px' }}>
            wants to access your wallet
          </div>

          {/* Wallet address badge */}
          <div style={{
            height: '36px',
            borderRadius: T.radiusSm,
            background: T.bgInput,
            border: `1px solid ${T.border}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            overflow: 'hidden',
          }}>
            <span style={{ color: T.accentBlue, fontSize: '12px', fontFamily: 'monospace', padding: '0 12px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {walletAddress}
            </span>
          </div>
        </div>

        {/* Scopes card */}
        <div style={cardInner}>
          <div style={{ color: T.textPrimary, fontSize: '14px', fontWeight: 600, marginBottom: '12px', textAlign: 'left' }}>
            Requested Permissions
          </div>

          {scopes.map((scope, i) => (
            <div key={i} style={{
              height: '44px',
              borderRadius: T.radiusSm,
              background: T.bgSecondary,
              display: 'flex',
              alignItems: 'center',
              padding: '0 12px',
              gap: '12px',
              marginBottom: i < scopes.length - 1 ? '8px' : '0',
            }}>
              {/* Checkmark */}
              <div style={{
                width: '22px',
                height: '22px',
                borderRadius: '4px',
                background: `${T.accentGreen}33`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
              }}>
                <span style={{ color: T.accentGreen, fontSize: '13px' }}>&#x2713;</span>
              </div>

              <div>
                <div style={{ color: T.textPrimary, fontSize: '13px', fontWeight: 500 }}>{scope.name}</div>
                <div style={{ color: T.textMuted, fontSize: '11px' }}>{scope.description}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Error */}
        {error && (
          <div style={{
            background: `${T.accentRed}18`,
            border: `1px solid ${T.accentRed}40`,
            borderRadius: T.radiusSm,
            padding: '10px 14px',
            color: T.accentRed,
            fontSize: '13px',
          }}>
            {error}
          </div>
        )}

        {/* Buttons — Deny / Approve side by side */}
        <div style={{ display: 'flex', gap: '12px' }}>
          <button onClick={handleDeny} style={{ ...btnStyle, background: T.accentRed, flex: 1 }}>
            Deny
          </button>
          <button
            onClick={handleApprove}
            disabled={loading}
            style={{ ...btnStyle, background: loading ? T.bgSecondary : T.accentGreen, flex: 1, cursor: loading ? 'wait' : 'pointer' }}
          >
            {loading ? 'Approving...' : 'Approve'}
          </button>
        </div>

        {/* Security note */}
        <div style={{ color: T.textMuted, fontSize: '11px', textAlign: 'center' }}>
          This app will NOT have access to your private keys.
        </div>
      </div>
    </div>
  );
};

const containerStyle: React.CSSProperties = {
  minHeight: '100vh',
  background: T.bgPrimary,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
};

const cardOuter: React.CSSProperties = {
  background: T.bgSecondary,
  border: `1px solid ${T.border}`,
  borderRadius: T.radiusMd,
  padding: '32px',
  maxWidth: '420px',
  width: '90%',
  display: 'flex',
  flexDirection: 'column',
  gap: '16px',
};

const cardInner: React.CSSProperties = {
  background: T.bgPrimary,
  border: `1px solid ${T.border}`,
  borderRadius: T.radiusSm,
  padding: '16px',
  textAlign: 'center',
};

const btnStyle: React.CSSProperties = {
  padding: '12px',
  border: 'none',
  borderRadius: T.radiusMd,
  color: T.textPrimary,
  fontSize: '15px',
  fontWeight: 600,
  cursor: 'pointer',
};

export default MinerLoginPage;
