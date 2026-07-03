import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X, Trophy, Users, CreditCard, DollarSign, Coins, CheckCircle,
  AlertCircle, Loader2, TrendingUp, Award, Star, Crown, Gem,
  ChevronRight, Vote, Gift
} from 'lucide-react';
import { generateAuthHeader, walletSession } from '../services/walletAuth';

// ─── Types ──────────────────────────────────────────────────────────────────

interface CampaignPerks {
  reduced_trading_fees: boolean;
  governance_voting: boolean;
  airdrop_multiplier: number;
  early_access: boolean;
  vip_support: boolean;
  nft_badge: boolean;
}

interface Campaign {
  campaign_id: string;
  exchange_name: string;
  exchange_logo: string;
  target_usd: number;
  raised_usd: number;
  contributor_count: number;
  early_bird_slots: number;
  early_bird_claimed: number;
  status: 'funding' | 'funded' | 'listed' | 'cancelled';
  tier: string;
  description: string;
  perks: CampaignPerks;
}

interface LeaderboardEntry {
  rank: number;
  wallet_address: string;
  amount_usd: number;
  share_percent: number;
  tier: 'diamond' | 'gold' | 'silver' | 'bronze';
  multiplier: number;
  is_early_bird: boolean;
}

interface MyPerksData {
  contributed: boolean;
  total_contributed_usd: number;
  share_percent: number;
  tier: 'diamond' | 'gold' | 'silver' | 'bronze';
  multiplier: number;
  is_early_bird: boolean;
  governance_eligible: boolean;
  perks_unlocked: string[];
}

interface XListCrowdfundModalProps {
  campaign: Campaign;
  onClose: () => void;
  walletAddress: string;
}

type TabType = 'campaign' | 'leaderboard' | 'perks';
type PaymentMethod = 'qug' | 'qugusd' | 'stripe_usd';

// ─── Tier Helpers ───────────────────────────────────────────────────────────

const TIER_CONFIG: Record<string, { icon: string; label: string; color: string; bg: string; minUsd: number; multiplier: number }> = {
  diamond: { icon: '\u{1F48E}', label: 'Diamond', color: '#b9f2ff', bg: 'rgba(185, 242, 255, 0.12)', minUsd: 10000, multiplier: 3.0 },
  gold:    { icon: '\u{1F947}', label: 'Gold',    color: '#ffd700', bg: 'rgba(255, 215, 0, 0.10)',    minUsd: 5000,  multiplier: 2.0 },
  silver:  { icon: '\u{1F948}', label: 'Silver',  color: '#c0c0c0', bg: 'rgba(192, 192, 192, 0.10)',  minUsd: 1000,  multiplier: 1.5 },
  bronze:  { icon: '\u{1F949}', label: 'Bronze',  color: '#cd7f32', bg: 'rgba(205, 127, 50, 0.10)',   minUsd: 1,     multiplier: 1.0 },
};

const truncateWallet = (addr: string): string => {
  if (!addr || addr.length < 14) return addr;
  return `${addr.slice(0, 8)}...${addr.slice(-6)}`;
};

const formatUsd = (value: number): string => {
  if (value == null || isNaN(value)) return '$0';
  return value.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
};

// ─── Component ──────────────────────────────────────────────────────────────

const XListCrowdfundModal: React.FC<XListCrowdfundModalProps> = ({ campaign, onClose, walletAddress }) => {
  const [activeTab, setActiveTab] = useState<TabType>('campaign');
  const [amount, setAmount] = useState<string>('');
  const [paymentMethod, setPaymentMethod] = useState<PaymentMethod>('qug');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // Helper: generate proper X-Wallet-Auth header using Ed25519 session key
  const getAuthHeaders = useCallback(async (path: string): Promise<Record<string, string>> => {
    const session = walletSession.getSession();
    if (!session) return {};
    const header = await generateAuthHeader(session.privateKey, session.address, path);
    return { 'X-Wallet-Auth': header };
  }, []);

  // Local optimistic state for progress bar
  const [localRaised, setLocalRaised] = useState(campaign.raised_usd);
  const [localContributors, setLocalContributors] = useState(campaign.contributor_count);
  const [localEarlyClaimed, setLocalEarlyClaimed] = useState(campaign.early_bird_claimed);

  // Leaderboard
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [leaderboardLoading, setLeaderboardLoading] = useState(false);

  // My perks
  const [myPerks, setMyPerks] = useState<MyPerksData | null>(null);
  const [perksLoading, setPerksLoading] = useState(false);

  // Sync from prop on mount
  useEffect(() => {
    setLocalRaised(campaign.raised_usd);
    setLocalContributors(campaign.contributor_count);
    setLocalEarlyClaimed(campaign.early_bird_claimed);
  }, [campaign]);

  // Fetch leaderboard when switching to that tab
  const fetchLeaderboard = useCallback(async () => {
    setLeaderboardLoading(true);
    try {
      const res = await fetch(`/api/v1/contracts/listing/campaigns/${campaign.campaign_id}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.success && data.data?.leaderboard) {
        setLeaderboard(data.data.leaderboard);
      }
    } catch (e) {
      console.warn('Failed to fetch leaderboard:', e);
    } finally {
      setLeaderboardLoading(false);
    }
  }, [campaign.campaign_id, walletAddress]);

  // Fetch my perks when switching to that tab
  const fetchMyPerks = useCallback(async () => {
    setPerksLoading(true);
    try {
      const perksPath = `/api/v1/contracts/listing/campaigns/${campaign.campaign_id}/my-perks`;
      const authHeaders = await getAuthHeaders(perksPath);
      const res = await fetch(perksPath, {
        headers: authHeaders,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.success && data.data) {
        setMyPerks(data.data);
      }
    } catch (e) {
      console.warn('Failed to fetch perks:', e);
    } finally {
      setPerksLoading(false);
    }
  }, [campaign.campaign_id, walletAddress]);

  useEffect(() => {
    if (activeTab === 'leaderboard') fetchLeaderboard();
    if (activeTab === 'perks') fetchMyPerks();
  }, [activeTab, fetchLeaderboard, fetchMyPerks]);

  // Contribute handler
  const handleContribute = async () => {
    setError(null);
    setSuccessMsg(null);
    const amountNum = parseFloat(amount);
    if (!amountNum || amountNum < 1) {
      setError('Minimum contribution is $1.');
      return;
    }
    setIsSubmitting(true);
    try {
      const contributePath = '/api/v1/contracts/listing/campaigns/contribute';
      const authHeaders = await getAuthHeaders(contributePath);
      const res = await fetch(contributePath, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...authHeaders,
        },
        body: JSON.stringify({
          campaign_id: campaign.campaign_id,
          payment_method: paymentMethod,
          amount_usd: amountNum,
        }),
      });
      const text = await res.text();
      let data: any;
      try { data = JSON.parse(text); } catch {
        throw new Error(res.ok ? 'Invalid server response' : `Server error (${res.status})`);
      }
      if (data.success) {
        setSuccessMsg(`Contributed ${formatUsd(amountNum)} successfully!`);
        setAmount('');
        // Optimistic update
        setLocalRaised(prev => prev + amountNum);
        setLocalContributors(prev => prev + 1);
        if (localEarlyClaimed < campaign.early_bird_slots) {
          setLocalEarlyClaimed(prev => prev + 1);
        }
      } else {
        setError(data.error || 'Contribution failed. Please try again.');
      }
    } catch (e: any) {
      setError(e.message || 'Network error. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const percentFunded = Math.min((localRaised / campaign.target_usd) * 100, 100);
  const isFunded = campaign.status === 'funded' || campaign.status === 'listed';
  const earlyBirdRemaining = campaign.early_bird_slots - localEarlyClaimed;

  // ─── Status Badge ───────────────────────────────────────────────────────

  const StatusBadge = () => {
    const configs: Record<string, { label: string; color: string; glow: string; pulse: boolean; icon: React.ReactNode }> = {
      funding: { label: 'FUNDING', color: '#00d4ff', glow: 'rgba(0, 212, 255, 0.3)', pulse: true, icon: <TrendingUp size={14} /> },
      funded:  { label: 'FUNDED',  color: '#ffd700', glow: 'rgba(255, 215, 0, 0.3)',  pulse: false, icon: <Trophy size={14} /> },
      listed:  { label: 'LISTED',  color: '#3b82f6', glow: 'rgba(59, 130, 246, 0.3)', pulse: false, icon: <CheckCircle size={14} /> },
      cancelled: { label: 'CANCELLED', color: '#ef4444', glow: 'rgba(239, 68, 68, 0.3)', pulse: false, icon: <AlertCircle size={14} /> },
    };
    const cfg = configs[campaign.status] || configs.funding;

    return (
      <motion.span
        animate={cfg.pulse ? { boxShadow: [`0 0 8px ${cfg.glow}`, `0 0 20px ${cfg.glow}`, `0 0 8px ${cfg.glow}`] } : {}}
        transition={cfg.pulse ? { duration: 2, repeat: Infinity } : {}}
        style={{
          display: 'inline-flex', alignItems: 'center', gap: 6,
          padding: '4px 12px', borderRadius: 20,
          fontSize: 12, fontWeight: 700, letterSpacing: 1,
          color: cfg.color,
          background: `${cfg.color}15`,
          border: `1px solid ${cfg.color}40`,
        }}
      >
        {cfg.icon} {cfg.label}
      </motion.span>
    );
  };

  // ─── Campaign Tab ─────────────────────────────────────────────────────────

  const CampaignTab = () => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Funded celebration */}
      {isFunded && (
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          style={{
            textAlign: 'center', padding: 16, borderRadius: 12,
            background: 'linear-gradient(135deg, rgba(255, 215, 0, 0.08), rgba(255, 165, 0, 0.06))',
            border: '1px solid rgba(255, 215, 0, 0.25)',
          }}
        >
          <div style={{ fontSize: 28, fontWeight: 800, color: '#ffd700', marginBottom: 4 }}>
            WE DID IT!
          </div>
          <div style={{ fontSize: 13, color: 'rgba(255, 215, 0, 0.7)' }}>
            {campaign.exchange_name} listing is fully funded
          </div>
        </motion.div>
      )}

      {/* Exchange header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
        <div style={{
          width: 52, height: 52, borderRadius: 14, overflow: 'hidden',
          background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
        }}>
          {campaign.exchange_logo ? (
            <img src={campaign.exchange_logo} alt={campaign.exchange_name} style={{ width: 40, height: 40, objectFit: 'contain' }} />
          ) : (
            <Star size={24} style={{ color: '#00d4ff' }} />
          )}
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#ffffff' }}>{campaign.exchange_name}</div>
          <div style={{ fontSize: 12, color: 'rgba(255,255,255,0.45)', marginTop: 2 }}>
            {localContributors} contributors
          </div>
        </div>
        <StatusBadge />
      </div>

      {/* Progress bar */}
      <div style={{
        padding: 16, borderRadius: 12,
        background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(0, 212, 255, 0.12)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
          <span style={{ fontSize: 13, color: 'rgba(255,255,255,0.6)' }}>
            {formatUsd(localRaised)} raised
          </span>
          <span style={{ fontSize: 13, fontWeight: 600, color: '#00d4ff' }}>
            {percentFunded.toFixed(1)}%
          </span>
        </div>
        <div style={{
          width: '100%', height: 14, borderRadius: 7,
          background: 'rgba(255,255,255,0.06)', overflow: 'hidden',
          position: 'relative',
        }}>
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${percentFunded}%` }}
            transition={{ duration: 1.2, ease: 'easeOut' }}
            style={{
              height: '100%', borderRadius: 7,
              background: 'linear-gradient(90deg, #00d4ff, #00e676)',
              boxShadow: '0 0 12px rgba(0, 212, 255, 0.4)',
            }}
          />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6 }}>
          <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)' }}>$0</span>
          <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)' }}>
            Goal: {formatUsd(campaign.target_usd)}
          </span>
        </div>
      </div>

      {/* Early bird badge */}
      {earlyBirdRemaining > 0 && !isFunded && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '8px 14px', borderRadius: 10,
          background: 'rgba(255, 193, 7, 0.08)', border: '1px solid rgba(255, 193, 7, 0.2)',
        }}>
          <span style={{ fontSize: 16 }}>{'\u{1F426}'}</span>
          <span style={{ fontSize: 13, color: '#ffc107', fontWeight: 600 }}>
            {localEarlyClaimed}/{campaign.early_bird_slots} Early Bird slots left
          </span>
          <span style={{ fontSize: 11, color: 'rgba(255, 193, 7, 0.6)', marginLeft: 'auto' }}>
            +0.5x bonus
          </span>
        </div>
      )}

      {/* Description */}
      <div style={{ fontSize: 13, lineHeight: 1.6, color: 'rgba(255,255,255,0.55)', padding: '0 2px' }}>
        {campaign.description}
      </div>

      {/* Perk tiers */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {Object.entries(TIER_CONFIG).map(([key, cfg]) => (
          <div key={key} style={{
            padding: '10px 12px', borderRadius: 10,
            background: cfg.bg, border: `1px solid ${cfg.color}25`,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
              <span style={{ fontSize: 15 }}>{cfg.icon}</span>
              <span style={{ fontSize: 12, fontWeight: 700, color: cfg.color }}>{cfg.label}</span>
            </div>
            <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.4)' }}>
              {cfg.multiplier}x multiplier &middot; {formatUsd(cfg.minUsd)}+
            </div>
          </div>
        ))}
      </div>

      {/* Contribution form */}
      {!isFunded ? (
        <div style={{
          padding: 16, borderRadius: 12,
          background: 'rgba(0, 212, 255, 0.04)', border: '1px solid rgba(0, 212, 255, 0.15)',
        }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: '#ffffff', marginBottom: 10 }}>
            Contribute to this campaign
          </div>

          {/* Amount input */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            padding: '10px 12px', borderRadius: 10,
            background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.1)',
            marginBottom: 10,
          }}>
            <DollarSign size={16} style={{ color: 'rgba(255,255,255,0.3)', flexShrink: 0 }} />
            <input
              type="number"
              min="1"
              step="1"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              placeholder="Amount in USD (min $1)"
              style={{
                flex: 1, background: 'transparent', border: 'none', outline: 'none',
                color: '#ffffff', fontSize: 15, fontFamily: 'monospace',
              }}
            />
          </div>

          {/* Payment method selector */}
          <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
            {([
              { key: 'qug' as PaymentMethod, label: 'QUG', icon: <Coins size={14} /> },
              { key: 'qugusd' as PaymentMethod, label: 'QUGUSD', icon: <DollarSign size={14} /> },
              { key: 'stripe_usd' as PaymentMethod, label: 'Card', icon: <CreditCard size={14} /> },
            ]).map(({ key, label, icon }) => (
              <button
                key={key}
                onClick={() => setPaymentMethod(key)}
                style={{
                  flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                  padding: '8px 0', borderRadius: 8, cursor: 'pointer',
                  fontSize: 12, fontWeight: 600, transition: 'all 0.2s',
                  background: paymentMethod === key ? 'rgba(0, 212, 255, 0.15)' : 'rgba(255,255,255,0.03)',
                  border: paymentMethod === key ? '1px solid rgba(0, 212, 255, 0.4)' : '1px solid rgba(255,255,255,0.08)',
                  color: paymentMethod === key ? '#00d4ff' : 'rgba(255,255,255,0.45)',
                }}
              >
                {icon} {label}
              </button>
            ))}
          </div>

          {/* Error / Success */}
          {error && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '8px 12px', borderRadius: 8, marginBottom: 10,
              background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.2)',
            }}>
              <AlertCircle size={14} style={{ color: '#ef4444', flexShrink: 0 }} />
              <span style={{ fontSize: 12, color: '#fca5a5' }}>{error}</span>
            </div>
          )}
          {successMsg && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '8px 12px', borderRadius: 8, marginBottom: 10,
              background: 'rgba(34, 197, 94, 0.08)', border: '1px solid rgba(34, 197, 94, 0.2)',
            }}>
              <CheckCircle size={14} style={{ color: '#22c55e', flexShrink: 0 }} />
              <span style={{ fontSize: 12, color: '#86efac' }}>{successMsg}</span>
            </div>
          )}

          {/* Contribute button */}
          <button
            onClick={handleContribute}
            disabled={isSubmitting || !amount || parseFloat(amount) < 1}
            style={{
              width: '100%', padding: '12px 0', borderRadius: 10, cursor: 'pointer',
              fontSize: 14, fontWeight: 700, color: '#ffffff', border: 'none',
              background: isSubmitting ? 'rgba(0, 212, 255, 0.2)'
                : 'linear-gradient(135deg, #00d4ff, #00b4d8)',
              boxShadow: isSubmitting ? 'none' : '0 0 20px rgba(0, 212, 255, 0.2)',
              opacity: (!amount || parseFloat(amount) < 1) ? 0.4 : 1,
              transition: 'all 0.2s',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
            }}
          >
            {isSubmitting ? (
              <>
                <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} />
                Processing...
              </>
            ) : (
              <>
                <ChevronRight size={16} />
                Contribute {amount ? formatUsd(parseFloat(amount)) : ''}
              </>
            )}
          </button>
        </div>
      ) : (
        /* Funded state — form replaced with celebration */
        <div style={{
          padding: 20, borderRadius: 12, textAlign: 'center',
          background: 'linear-gradient(135deg, rgba(255, 215, 0, 0.06), rgba(0, 212, 255, 0.04))',
          border: '1px solid rgba(255, 215, 0, 0.2)',
        }}>
          <Trophy size={32} style={{ color: '#ffd700', marginBottom: 8 }} />
          <div style={{ fontSize: 16, fontWeight: 700, color: '#ffd700' }}>Campaign Funded!</div>
          <div style={{ fontSize: 12, color: 'rgba(255,255,255,0.45)', marginTop: 4 }}>
            {formatUsd(localRaised)} raised from {localContributors} contributors
          </div>
        </div>
      )}
    </div>
  );

  // ─── Leaderboard Tab ──────────────────────────────────────────────────────

  const LeaderboardTab = () => {
    if (leaderboardLoading) {
      return (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 40, gap: 10 }}>
          <Loader2 size={20} style={{ color: '#00d4ff', animation: 'spin 1s linear infinite' }} />
          <span style={{ color: 'rgba(255,255,255,0.5)', fontSize: 13 }}>Loading leaderboard...</span>
        </div>
      );
    }

    if (leaderboard.length === 0) {
      return (
        <div style={{ textAlign: 'center', padding: 40, color: 'rgba(255,255,255,0.4)', fontSize: 13 }}>
          No contributions yet. Be the first!
        </div>
      );
    }

    const rankBg = (rank: number): string => {
      if (rank === 1) return 'rgba(255, 215, 0, 0.08)';
      if (rank === 2) return 'rgba(192, 192, 192, 0.06)';
      if (rank === 3) return 'rgba(205, 127, 50, 0.06)';
      return 'transparent';
    };

    const rankBorder = (rank: number): string => {
      if (rank === 1) return '1px solid rgba(255, 215, 0, 0.2)';
      if (rank === 2) return '1px solid rgba(192, 192, 192, 0.15)';
      if (rank === 3) return '1px solid rgba(205, 127, 50, 0.15)';
      return '1px solid rgba(255,255,255,0.04)';
    };

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {/* Header row */}
        <div style={{
          display: 'grid', gridTemplateColumns: '36px 1fr 80px 60px 70px 60px',
          padding: '8px 12px', fontSize: 10, fontWeight: 700, letterSpacing: 0.5,
          color: 'rgba(255,255,255,0.3)', textTransform: 'uppercase',
        }}>
          <span>#</span>
          <span>Wallet</span>
          <span style={{ textAlign: 'right' }}>Amount</span>
          <span style={{ textAlign: 'right' }}>Share</span>
          <span style={{ textAlign: 'center' }}>Tier</span>
          <span style={{ textAlign: 'right' }}>Multi</span>
        </div>

        {/* Entries */}
        {leaderboard.map((entry) => {
          const tierCfg = TIER_CONFIG[entry.tier] || TIER_CONFIG.bronze;
          return (
            <motion.div
              key={entry.rank}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: entry.rank * 0.03 }}
              style={{
                display: 'grid', gridTemplateColumns: '36px 1fr 80px 60px 70px 60px',
                alignItems: 'center', padding: '10px 12px', borderRadius: 8,
                background: rankBg(entry.rank), border: rankBorder(entry.rank),
                fontSize: 12,
              }}
            >
              <span style={{
                fontWeight: 700, fontSize: entry.rank <= 3 ? 14 : 12,
                color: entry.rank <= 3 ? tierCfg.color : 'rgba(255,255,255,0.4)',
              }}>
                {entry.rank}
              </span>
              <span style={{ color: 'rgba(255,255,255,0.7)', fontFamily: 'monospace', fontSize: 11 }}>
                {truncateWallet(entry.wallet_address)}
                {entry.is_early_bird && <span style={{ marginLeft: 4 }}>{'\u{1F426}'}</span>}
              </span>
              <span style={{ textAlign: 'right', color: '#ffffff', fontWeight: 600, fontFamily: 'monospace' }}>
                {formatUsd(entry.amount_usd)}
              </span>
              <span style={{ textAlign: 'right', color: 'rgba(255,255,255,0.5)', fontFamily: 'monospace' }}>
                {(entry.share_percent ?? 0).toFixed(1)}%
              </span>
              <span style={{ textAlign: 'center' }}>
                <span style={{
                  display: 'inline-flex', alignItems: 'center', gap: 3,
                  padding: '2px 8px', borderRadius: 6, fontSize: 10, fontWeight: 600,
                  background: tierCfg.bg, color: tierCfg.color,
                  border: `1px solid ${tierCfg.color}30`,
                }}>
                  {tierCfg.icon} {tierCfg.label}
                </span>
              </span>
              <span style={{ textAlign: 'right', color: '#00d4ff', fontWeight: 600, fontFamily: 'monospace' }}>
                {(entry.multiplier ?? 0).toFixed(1)}x
              </span>
            </motion.div>
          );
        })}
      </div>
    );
  };

  // ─── My Perks Tab ─────────────────────────────────────────────────────────

  const MyPerksTab = () => {
    if (perksLoading) {
      return (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 40, gap: 10 }}>
          <Loader2 size={20} style={{ color: '#00d4ff', animation: 'spin 1s linear infinite' }} />
          <span style={{ color: 'rgba(255,255,255,0.5)', fontSize: 13 }}>Loading perks...</span>
        </div>
      );
    }

    if (!myPerks || !myPerks.contributed) {
      return (
        <div style={{
          textAlign: 'center', padding: 40,
          display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12,
        }}>
          <Gift size={36} style={{ color: 'rgba(255,255,255,0.15)' }} />
          <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: 14 }}>
            You haven't contributed yet
          </div>
          <button
            onClick={() => setActiveTab('campaign')}
            style={{
              padding: '8px 20px', borderRadius: 8, cursor: 'pointer',
              background: 'rgba(0, 212, 255, 0.12)', border: '1px solid rgba(0, 212, 255, 0.3)',
              color: '#00d4ff', fontSize: 13, fontWeight: 600,
            }}
          >
            Join the campaign
          </button>
        </div>
      );
    }

    const tierCfg = TIER_CONFIG[myPerks.tier] || TIER_CONFIG.bronze;

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {/* Tier card */}
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          style={{
            padding: 20, borderRadius: 14, textAlign: 'center',
            background: `linear-gradient(135deg, ${tierCfg.bg}, rgba(0, 212, 255, 0.03))`,
            border: `1px solid ${tierCfg.color}30`,
            position: 'relative', overflow: 'hidden',
          }}
        >
          <div style={{ fontSize: 40, marginBottom: 6 }}>{tierCfg.icon}</div>
          <div style={{ fontSize: 22, fontWeight: 800, color: tierCfg.color }}>
            {tierCfg.label} Tier
          </div>
          <div style={{
            display: 'inline-block', marginTop: 8, padding: '4px 16px', borderRadius: 20,
            background: 'rgba(0, 212, 255, 0.1)', border: '1px solid rgba(0, 212, 255, 0.2)',
            fontSize: 14, fontWeight: 700, color: '#00d4ff',
          }}>
            {(myPerks.multiplier ?? 0).toFixed(1)}x airdrop multiplier
            {myPerks.is_early_bird && (
              <span style={{ marginLeft: 6, fontSize: 12, color: '#ffc107' }}>
                (incl. +0.5x early bird)
              </span>
            )}
          </div>
        </motion.div>

        {/* Stats row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <div style={{
            padding: '12px 14px', borderRadius: 10,
            background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)',
          }}>
            <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.35)', textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Contributed
            </div>
            <div style={{ fontSize: 18, fontWeight: 700, color: '#ffffff', marginTop: 4, fontFamily: 'monospace' }}>
              {formatUsd(myPerks.total_contributed_usd)}
            </div>
          </div>
          <div style={{
            padding: '12px 14px', borderRadius: 10,
            background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)',
          }}>
            <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.35)', textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Pool Share
            </div>
            <div style={{ fontSize: 18, fontWeight: 700, color: '#00d4ff', marginTop: 4, fontFamily: 'monospace' }}>
              {(myPerks.share_percent ?? 0).toFixed(2)}%
            </div>
          </div>
        </div>

        {/* Governance status */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '10px 14px', borderRadius: 10,
          background: myPerks.governance_eligible ? 'rgba(34, 197, 94, 0.06)' : 'rgba(255,255,255,0.02)',
          border: myPerks.governance_eligible ? '1px solid rgba(34, 197, 94, 0.15)' : '1px solid rgba(255,255,255,0.05)',
        }}>
          <Vote size={16} style={{ color: myPerks.governance_eligible ? '#22c55e' : 'rgba(255,255,255,0.2)', flexShrink: 0 }} />
          <div>
            <div style={{ fontSize: 12, fontWeight: 600, color: myPerks.governance_eligible ? '#22c55e' : 'rgba(255,255,255,0.4)' }}>
              Governance Voting
            </div>
            <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)' }}>
              {myPerks.governance_eligible ? 'Eligible to vote on listing decisions' : 'Requires Gold tier or above'}
            </div>
          </div>
        </div>

        {/* Unlocked perks list */}
        <div>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'rgba(255,255,255,0.5)', marginBottom: 8 }}>
            Unlocked Perks
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {myPerks.perks_unlocked.length > 0 ? myPerks.perks_unlocked.map((perk, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '8px 12px', borderRadius: 8,
                background: 'rgba(34, 197, 94, 0.04)', border: '1px solid rgba(34, 197, 94, 0.1)',
              }}>
                <CheckCircle size={14} style={{ color: '#22c55e', flexShrink: 0 }} />
                <span style={{ fontSize: 12, color: 'rgba(255,255,255,0.7)' }}>{perk}</span>
              </div>
            )) : (
              <div style={{ fontSize: 12, color: 'rgba(255,255,255,0.3)', padding: '8px 0' }}>
                Contribute more to unlock additional perks.
              </div>
            )}
          </div>
        </div>

        {/* Early bird badge */}
        {myPerks.is_early_bird && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            padding: '8px 14px', borderRadius: 10,
            background: 'rgba(255, 193, 7, 0.06)', border: '1px solid rgba(255, 193, 7, 0.15)',
          }}>
            <span style={{ fontSize: 16 }}>{'\u{1F426}'}</span>
            <span style={{ fontSize: 12, color: '#ffc107', fontWeight: 600 }}>Early Bird Contributor</span>
            <span style={{ fontSize: 11, color: 'rgba(255, 193, 7, 0.6)', marginLeft: 'auto' }}>+0.5x bonus applied</span>
          </div>
        )}
      </div>
    );
  };

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        style={{
          position: 'fixed', inset: 0, zIndex: 50,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}
      >
        {/* Backdrop */}
        <div style={{
          position: 'absolute', inset: 0,
          background: 'rgba(0, 0, 0, 0.75)',
          backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)',
        }} />

        {/* Modal */}
        <motion.div
          initial={{ scale: 0.92, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.92, opacity: 0, y: 20 }}
          transition={{ type: 'spring', damping: 28, stiffness: 300 }}
          onClick={(e) => e.stopPropagation()}
          style={{
            position: 'relative', width: '100%', maxWidth: 540,
            maxHeight: '90vh', margin: '0 16px',
            borderRadius: 16, overflow: 'hidden',
            background: 'linear-gradient(135deg, rgba(10, 10, 15, 0.98), rgba(10, 15, 25, 0.96))',
            border: '1px solid rgba(0, 212, 255, 0.18)',
            boxShadow: '0 0 60px rgba(0, 212, 255, 0.08), 0 25px 50px rgba(0, 0, 0, 0.5)',
            display: 'flex', flexDirection: 'column',
          }}
        >
          {/* Header */}
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '16px 20px',
            borderBottom: '1px solid rgba(0, 212, 255, 0.1)',
            flexShrink: 0,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{
                width: 36, height: 36, borderRadius: 10,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                background: 'linear-gradient(135deg, #00d4ff, #0099cc)',
              }}>
                <Crown size={18} style={{ color: '#ffffff' }} />
              </div>
              <div>
                <div style={{ fontSize: 15, fontWeight: 700, color: '#ffffff' }}>Exchange Listing Fund</div>
                <div style={{ fontSize: 11, color: 'rgba(0, 212, 255, 0.5)' }}>Crowdfunding Campaign</div>
              </div>
            </div>
            <button
              onClick={onClose}
              style={{
                padding: 6, borderRadius: 8, cursor: 'pointer',
                background: 'transparent', border: 'none',
                color: 'rgba(255,255,255,0.4)', display: 'flex',
              }}
            >
              <X size={18} />
            </button>
          </div>

          {/* Tabs */}
          <div style={{
            display: 'flex', gap: 4, padding: '10px 20px 0',
            borderBottom: '1px solid rgba(255,255,255,0.04)',
            flexShrink: 0,
          }}>
            {([
              { key: 'campaign' as TabType, label: 'Campaign' },
              { key: 'leaderboard' as TabType, label: 'Leaderboard' },
              { key: 'perks' as TabType, label: 'My Perks' },
            ]).map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                style={{
                  padding: '8px 16px', borderRadius: '8px 8px 0 0', cursor: 'pointer',
                  fontSize: 13, fontWeight: 600, transition: 'all 0.2s',
                  background: activeTab === key ? 'rgba(0, 212, 255, 0.1)' : 'transparent',
                  border: activeTab === key ? '1px solid rgba(0, 212, 255, 0.2)' : '1px solid transparent',
                  borderBottom: 'none',
                  color: activeTab === key ? '#00d4ff' : 'rgba(255,255,255,0.4)',
                }}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Tab content — scrollable */}
          <div style={{
            flex: 1, overflow: 'auto', padding: 20,
            minHeight: 0,
          }}>
            {activeTab === 'campaign' && <CampaignTab />}
            {activeTab === 'leaderboard' && <LeaderboardTab />}
            {activeTab === 'perks' && <MyPerksTab />}
          </div>
        </motion.div>
      </motion.div>

      {/* Keyframe for spinner — injected once */}
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </AnimatePresence>
  );
};

export default XListCrowdfundModal;
