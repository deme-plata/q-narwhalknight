// v7.3.0: OAuth2 Consent Page — Visually striking authorization screen
// Renders when URL path is /oauth/consent with query params from the OAuth2 authorize endpoint
// Flow: Third-party app → /api/v1/oauth2/authorize → redirect here → user approves → redirect back with code

import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, Lock, Eye, Send, X, Check, AlertTriangle, Globe, Fingerprint, Sparkles, Zap } from 'lucide-react';

// ── Types ─────────────────────────────────────────────────────────────────────

interface ClientInfo {
  client_id: string;
  name: string;
  description: string;
  website: string;
  logo_url: string | null;
  scopes: string[];
}

interface ConsentParams {
  client_id: string;
  redirect_uri: string;
  scope: string;
  state: string;
  code_challenge?: string;
  code_challenge_method?: string;
}

const SCOPE_META: Record<string, { icon: React.ReactNode; label: string; description: string; risk: 'low' | 'medium' | 'high' }> = {
  'read:balance': {
    icon: <Eye className="w-5 h-5" />,
    label: 'View Balance',
    description: 'See your QUG balance and token holdings',
    risk: 'low',
  },
  'read:transactions': {
    icon: <Eye className="w-5 h-5" />,
    label: 'View Transactions',
    description: 'See your transaction history',
    risk: 'low',
  },
  'send:transaction': {
    icon: <Send className="w-5 h-5" />,
    label: 'Send Transactions',
    description: 'Create and submit transactions on your behalf',
    risk: 'high',
  },
  'read:profile': {
    icon: <Fingerprint className="w-5 h-5" />,
    label: 'View Profile',
    description: 'See your wallet address and public profile',
    risk: 'low',
  },
};

const RISK_COLORS = {
  low: { bg: 'from-emerald-500/20 to-green-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', badge: 'bg-emerald-500/20 text-emerald-300' },
  medium: { bg: 'from-amber-500/20 to-yellow-500/10', border: 'border-amber-500/30', text: 'text-amber-400', badge: 'bg-amber-500/20 text-amber-300' },
  high: { bg: 'from-red-500/20 to-orange-500/10', border: 'border-red-500/30', text: 'text-red-400', badge: 'bg-red-500/20 text-red-300' },
};

// ── Animated Background ───────────────────────────────────────────────────────

function ConsentBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;
    const particles: { x: number; y: number; vx: number; vy: number; size: number; alpha: number; hue: number }[] = [];

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    // Create floating particles
    for (let i = 0; i < 60; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        size: Math.random() * 2.5 + 0.5,
        alpha: Math.random() * 0.4 + 0.1,
        hue: 200 + Math.random() * 60, // Blue-cyan range
      });
    }

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw connection lines between nearby particles
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 150) {
            const alpha = (1 - dist / 150) * 0.08;
            ctx.strokeStyle = `hsla(220, 60%, 60%, ${alpha})`;
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }

      // Draw particles
      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${p.hue}, 70%, 65%, ${p.alpha})`;
        ctx.fill();

        // Glow
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * 3, 0, Math.PI * 2);
        const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size * 3);
        grad.addColorStop(0, `hsla(${p.hue}, 70%, 65%, ${p.alpha * 0.3})`);
        grad.addColorStop(1, 'transparent');
        ctx.fillStyle = grad;
        ctx.fill();
      }

      animId = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return <canvas ref={canvasRef} className="fixed inset-0 z-0 pointer-events-none" />;
}

// ── Connection Beam Animation ─────────────────────────────────────────────────

function ConnectionBeam() {
  return (
    <div className="relative w-full h-16 flex items-center justify-center my-2">
      {/* Static beam track */}
      <div className="absolute w-48 h-[2px] bg-gradient-to-r from-transparent via-blue-500/20 to-transparent rounded-full" />
      {/* Animated pulse traveling along the beam */}
      <motion.div
        className="absolute h-[2px] w-12 rounded-full"
        style={{ background: 'linear-gradient(90deg, transparent, #3b82f6, #06b6d4, transparent)' }}
        animate={{ x: [-100, 100] }}
        transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
      />
      {/* Center lock icon */}
      <motion.div
        className="relative z-10 w-10 h-10 rounded-full bg-slate-800/80 border border-blue-500/40 flex items-center justify-center backdrop-blur-sm"
        animate={{ boxShadow: ['0 0 15px rgba(59,130,246,0.2)', '0 0 25px rgba(59,130,246,0.4)', '0 0 15px rgba(59,130,246,0.2)'] }}
        transition={{ duration: 2, repeat: Infinity }}
      >
        <Lock className="w-4 h-4 text-blue-400" />
      </motion.div>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function OAuthConsentPage() {
  const [params, setParams] = useState<ConsentParams | null>(null);
  const [client, setClient] = useState<ClientInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<'approved' | 'denied' | null>(null);
  const [walletAddress, setWalletAddress] = useState('');
  const [walletFromStorage, setWalletFromStorage] = useState(false);

  // Parse query params on mount
  useEffect(() => {
    const search = new URLSearchParams(window.location.search);
    const clientId = search.get('client_id');
    const redirectUri = search.get('redirect_uri') || '';
    const scope = search.get('scope') || 'read:balance';
    const state = search.get('state') || '';
    const codeChallenge = search.get('code_challenge') || undefined;
    const codeChallengeMethod = search.get('code_challenge_method') || undefined;

    if (!clientId) {
      setError('Missing client_id parameter');
      setLoading(false);
      return;
    }

    setParams({ client_id: clientId, redirect_uri: redirectUri, scope, state, code_challenge: codeChallenge, code_challenge_method: codeChallengeMethod });

    // Load wallet address — try localStorage first, fall back to URL param (cross-subdomain flow)
    const storedWallet = localStorage.getItem('walletAddress') || '';
    const walletFromUrl = search.get('wallet') || '';
    const resolvedWallet = storedWallet || walletFromUrl;
    setWalletAddress(resolvedWallet);
    setWalletFromStorage(!!resolvedWallet);

    // Fetch client info
    fetch(`/api/v1/oauth2/clients/${clientId}`)
      .then(r => r.json())
      .then(data => {
        if (data.success && data.data) {
          setClient(data.data);
        } else {
          setError(data.error || 'Unknown application');
        }
      })
      .catch(() => setError('Failed to load application details'))
      .finally(() => setLoading(false));
  }, []);

  const scopes = params?.scope.split(/[\s,]+/).filter(Boolean) || [];
  const hasHighRisk = scopes.some(s => SCOPE_META[s]?.risk === 'high');

  const handleApprove = useCallback(async () => {
    if (!params || !walletAddress) return;
    setSubmitting(true);

    try {
      const res = await fetch('/api/v1/oauth2/consent', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Wallet-Auth': walletAddress,
        },
        body: JSON.stringify({
          wallet_address: walletAddress,
          client_id: params.client_id,
          scopes,
          approved: true,
          auth_request_id: params.state,
          redirect_uri: params.redirect_uri,
          code_challenge: params.code_challenge,
          code_challenge_method: params.code_challenge_method,
        }),
      });

      const data = await res.json();
      if (data.success && data.data) {
        // v7.4.0: consent endpoint now returns {auth_code, consent_hash, consent_tx_data}
        const authCode = typeof data.data === 'string' ? data.data : data.data.auth_code;
        setResult('approved');
        // Redirect back to the app after a brief success animation
        setTimeout(() => {
          const separator = params.redirect_uri.includes('?') ? '&' : '?';
          const redirectUrl = `${params.redirect_uri}${separator}code=${encodeURIComponent(authCode)}&state=${encodeURIComponent(params.state)}`;
          window.location.href = redirectUrl;
        }, 1500);
      } else {
        setError(data.error || 'Failed to authorize');
      }
    } catch {
      setError('Network error - please try again');
    }
    setSubmitting(false);
  }, [params, walletAddress, scopes]);

  const handleDeny = useCallback(() => {
    if (!params) return;
    setResult('denied');
    setTimeout(() => {
      const separator = params.redirect_uri.includes('?') ? '&' : '?';
      window.location.href = `${params.redirect_uri}${separator}error=access_denied&state=${encodeURIComponent(params.state)}`;
    }, 800);
  }, [params]);

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-[#0a1628] to-slate-950 flex items-center justify-center p-4 overflow-hidden">
      <ConsentBackground />

      {/* Top ambient glow */}
      <div className="fixed top-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] bg-blue-500/5 rounded-full blur-[100px] pointer-events-none" />
      <div className="fixed bottom-0 left-1/2 -translate-x-1/2 w-[400px] h-[200px] bg-cyan-500/5 rounded-full blur-[80px] pointer-events-none" />

      <AnimatePresence mode="wait">
        {loading ? (
          <LoadingState key="loading" />
        ) : error ? (
          <ErrorState key="error" error={error} />
        ) : result ? (
          <ResultState key="result" result={result} clientName={client?.name || 'Application'} />
        ) : (
          <motion.div
            key="consent"
            initial={{ opacity: 0, y: 30, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20, scale: 0.95 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            className="relative z-10 w-full max-w-lg"
          >
            {/* Outer glow ring */}
            <div className="absolute -inset-[1px] rounded-3xl bg-gradient-to-br from-blue-500/30 via-transparent to-cyan-500/30 blur-sm" />

            {/* Main card */}
            <div className="relative bg-gradient-to-br from-slate-900/95 via-slate-800/95 to-slate-900/95 backdrop-blur-xl rounded-3xl border border-white/[0.08] shadow-2xl shadow-blue-500/10 overflow-hidden">

              {/* Animated top accent line */}
              <div className="relative h-[2px] overflow-hidden">
                <motion.div
                  className="absolute inset-y-0 w-1/3"
                  style={{ background: 'linear-gradient(90deg, transparent, #3b82f6, #06b6d4, transparent)' }}
                  animate={{ x: ['-33%', '133%'] }}
                  transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
                />
              </div>

              {/* Header */}
              <div className="px-8 pt-8 pb-4">
                {/* App identity section */}
                <div className="flex flex-col items-center text-center">
                  {/* App + Wallet connection visual */}
                  <div className="flex items-center gap-0 mb-2">
                    {/* Third-party app */}
                    <motion.div
                      className="w-16 h-16 rounded-2xl bg-gradient-to-br from-slate-700/80 to-slate-800/80 border border-white/10 flex items-center justify-center shadow-lg overflow-hidden"
                      initial={{ x: -20, opacity: 0 }}
                      animate={{ x: 0, opacity: 1 }}
                      transition={{ delay: 0.2 }}
                    >
                      {client?.logo_url ? (
                        <img src={client.logo_url} alt={client.name} className="w-10 h-10 rounded-lg object-cover" />
                      ) : (
                        <Globe className="w-8 h-8 text-slate-400" />
                      )}
                    </motion.div>

                    <ConnectionBeam />

                    {/* Quillon wallet */}
                    <motion.div
                      className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-600/30 to-cyan-600/30 border border-blue-500/30 flex items-center justify-center shadow-lg shadow-blue-500/10"
                      initial={{ x: 20, opacity: 0 }}
                      animate={{ x: 0, opacity: 1 }}
                      transition={{ delay: 0.3 }}
                    >
                      <Shield className="w-8 h-8 text-blue-400" />
                    </motion.div>
                  </div>

                  {/* App name + request text */}
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                  >
                    <h1 className="text-xl font-bold text-white mt-3">
                      {client?.name || 'Application'}
                    </h1>
                    <p className="text-sm text-slate-400 mt-1">wants to connect to your wallet</p>
                    {client?.website && (
                      <a
                        href={client.website}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 text-xs text-blue-400/70 hover:text-blue-400 mt-1.5 transition-colors"
                      >
                        <Globe className="w-3 h-3" />
                        {new URL(client.website).hostname}
                      </a>
                    )}
                  </motion.div>
                </div>

                {/* Your wallet (show display if wallet came from storage) */}
                {walletFromStorage && walletAddress && (
                  <motion.div
                    className="mt-5 px-4 py-2.5 bg-slate-800/50 border border-slate-700/50 rounded-xl flex items-center gap-3"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                  >
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-amber-500/20 to-yellow-500/20 border border-amber-500/30 flex items-center justify-center flex-shrink-0">
                      <Fingerprint className="w-4 h-4 text-amber-400" />
                    </div>
                    <div className="min-w-0">
                      <div className="text-xs text-slate-500">Your Wallet</div>
                      <div className="text-sm text-slate-300 font-mono truncate">
                        {walletAddress.slice(0, 8)}...{walletAddress.slice(-8)}
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>

              {/* Permissions */}
              <div className="px-8 pb-4">
                <motion.div
                  className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.6 }}
                >
                  <Lock className="w-3 h-3" />
                  Permissions Requested
                </motion.div>

                <div className="space-y-2.5">
                  {scopes.map((scope, i) => {
                    const meta = SCOPE_META[scope] || {
                      icon: <Zap className="w-5 h-5" />,
                      label: scope,
                      description: `Access to ${scope}`,
                      risk: 'medium' as const,
                    };
                    const colors = RISK_COLORS[meta.risk];

                    return (
                      <motion.div
                        key={scope}
                        className={`relative bg-gradient-to-r ${colors.bg} border ${colors.border} rounded-xl px-4 py-3 flex items-start gap-3 overflow-hidden`}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.7 + i * 0.1, duration: 0.4 }}
                      >
                        <div className={`flex-shrink-0 mt-0.5 ${colors.text}`}>
                          {meta.icon}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-semibold text-white">{meta.label}</span>
                            <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium uppercase tracking-wide ${colors.badge}`}>
                              {meta.risk}
                            </span>
                          </div>
                          <p className="text-xs text-slate-400 mt-0.5">{meta.description}</p>
                        </div>
                      </motion.div>
                    );
                  })}
                </div>
              </div>

              {/* High-risk warning */}
              {hasHighRisk && (
                <motion.div
                  className="mx-8 mb-4 px-4 py-3 bg-red-500/10 border border-red-500/20 rounded-xl flex items-start gap-3"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  transition={{ delay: 1.0 }}
                >
                  <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                  <p className="text-xs text-red-300/90 leading-relaxed">
                    This application is requesting <strong>high-risk</strong> permissions that allow it to send transactions from your wallet. Only approve if you trust this application.
                  </p>
                </motion.div>
              )}

              {/* Wallet address input for users not logged in on quillon.xyz (e.g. from bounty subdomain) */}
              {!walletFromStorage && (
                <motion.div
                  className="mx-8 mb-4"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.8 }}
                >
                  <div className="px-4 py-4 bg-slate-800/40 border border-blue-500/20 rounded-xl space-y-3">
                    <div className="flex items-start gap-2">
                      <Fingerprint className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                      <p className="text-xs text-slate-300/90 leading-relaxed">
                        Enter your Quillon wallet address to authorize this application.
                      </p>
                    </div>
                    <input
                      type="text"
                      value={walletAddress}
                      onChange={(e) => setWalletAddress(e.target.value.trim())}
                      placeholder="qnk... or hex wallet address"
                      className="w-full px-3 py-2.5 bg-slate-900/80 border border-slate-600/50 rounded-lg text-sm text-white placeholder-slate-500 font-mono focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all"
                    />
                    {walletAddress && (
                      <div className="flex items-center gap-1.5 text-[10px] text-emerald-400/80">
                        <Check className="w-3 h-3" />
                        <span>Wallet address set</span>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}

              {/* PQ Security badge */}
              <motion.div
                className="mx-8 mb-5 flex items-center justify-center gap-2 text-[10px] text-slate-500"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1.1 }}
              >
                <Sparkles className="w-3 h-3 text-purple-400/60" />
                <span>Secured with post-quantum cryptography (Kyber1024)</span>
              </motion.div>

              {/* Action buttons */}
              <motion.div
                className="px-8 pb-8 flex gap-3"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.2 }}
              >
                <button
                  onClick={handleDeny}
                  disabled={submitting}
                  className="flex-1 py-3.5 px-6 rounded-xl bg-slate-800/80 border border-slate-600/50 text-slate-300 font-medium text-sm hover:bg-slate-700/80 hover:border-slate-500/50 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  <X className="w-4 h-4" />
                  Deny
                </button>
                <button
                  onClick={handleApprove}
                  disabled={submitting || !walletAddress}
                  className="flex-[1.5] py-3.5 px-6 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-600 text-white font-semibold text-sm hover:from-blue-500 hover:to-cyan-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-blue-500/20 hover:shadow-blue-500/30 flex items-center justify-center gap-2 relative overflow-hidden group"
                >
                  {submitting ? (
                    <motion.div
                      className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full"
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    />
                  ) : (
                    <>
                      {/* Shimmer effect on hover */}
                      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
                      <Check className="w-4 h-4 relative z-10" />
                      <span className="relative z-10">Authorize</span>
                    </>
                  )}
                </button>
              </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Sub-states ────────────────────────────────────────────────────────────────

function LoadingState() {
  return (
    <motion.div
      className="relative z-10 flex flex-col items-center gap-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div
        className="w-16 h-16 rounded-2xl bg-slate-800/80 border border-blue-500/30 flex items-center justify-center"
        animate={{
          boxShadow: [
            '0 0 20px rgba(59,130,246,0.1)',
            '0 0 40px rgba(59,130,246,0.3)',
            '0 0 20px rgba(59,130,246,0.1)',
          ],
        }}
        transition={{ duration: 2, repeat: Infinity }}
      >
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
        >
          <Shield className="w-8 h-8 text-blue-400" />
        </motion.div>
      </motion.div>
      <p className="text-slate-400 text-sm">Loading authorization request...</p>
    </motion.div>
  );
}

function ErrorState({ error }: { error: string }) {
  return (
    <motion.div
      className="relative z-10 w-full max-w-md"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0 }}
    >
      <div className="bg-slate-900/90 backdrop-blur-xl border border-red-500/30 rounded-3xl p-8 text-center shadow-2xl">
        <div className="w-16 h-16 rounded-2xl bg-red-500/10 border border-red-500/30 flex items-center justify-center mx-auto mb-4">
          <AlertTriangle className="w-8 h-8 text-red-400" />
        </div>
        <h2 className="text-lg font-bold text-white mb-2">Authorization Failed</h2>
        <p className="text-sm text-slate-400 mb-6">{error}</p>
        <button
          onClick={() => window.history.back()}
          className="px-6 py-2.5 rounded-xl bg-slate-800 border border-slate-600/50 text-slate-300 text-sm font-medium hover:bg-slate-700 transition-colors"
        >
          Go Back
        </button>
      </div>
    </motion.div>
  );
}

function ResultState({ result, clientName }: { result: 'approved' | 'denied'; clientName: string }) {
  const isApproved = result === 'approved';

  return (
    <motion.div
      className="relative z-10 flex flex-col items-center gap-4"
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0 }}
      transition={{ type: 'spring', stiffness: 200, damping: 15 }}
    >
      <motion.div
        className={`w-20 h-20 rounded-full flex items-center justify-center ${
          isApproved
            ? 'bg-gradient-to-br from-emerald-500/20 to-green-500/20 border-2 border-emerald-500/50'
            : 'bg-gradient-to-br from-red-500/20 to-orange-500/20 border-2 border-red-500/50'
        }`}
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: 'spring', stiffness: 300, damping: 12, delay: 0.1 }}
      >
        {isApproved ? (
          <Check className="w-10 h-10 text-emerald-400" />
        ) : (
          <X className="w-10 h-10 text-red-400" />
        )}
      </motion.div>

      <motion.div
        className="text-center"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <h2 className="text-xl font-bold text-white">
          {isApproved ? 'Authorized' : 'Denied'}
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          {isApproved
            ? `Redirecting back to ${clientName}...`
            : `Access denied. Returning to ${clientName}...`}
        </p>
      </motion.div>

      {/* Ripple effect */}
      {isApproved && (
        <motion.div
          className="absolute w-40 h-40 rounded-full border border-emerald-500/20"
          initial={{ scale: 0.5, opacity: 1 }}
          animate={{ scale: 3, opacity: 0 }}
          transition={{ duration: 1.5, ease: 'easeOut' }}
        />
      )}
    </motion.div>
  );
}
