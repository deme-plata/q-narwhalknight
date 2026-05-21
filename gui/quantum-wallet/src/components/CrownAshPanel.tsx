// CrownAshPanel — agent-detail UI for the Crown & Ash grand-strategy game.
//
// Three sub-tabs:
//   • Realm     — my faction summary (provinces, treasury, prestige, armies)
//   • World     — full snapshot (turn, all factions, treaty list)
//   • Diplomacy — propose alliance + treaty list with action buttons
//
// All data flows over the existing /api/v1/crown-ash/* endpoints. The
// agent's wallet address is read from localStorage (`walletAddress`),
// matching the rest of the app's auth pattern.

import { useState, useEffect, useCallback } from 'react';
import { Crown, Sword, Shield, Scroll, Handshake, RefreshCw, AlertCircle } from 'lucide-react';

interface CrownAshPanelProps {
  walletAddress?: string;
  className?: string;
}

interface FactionLite {
  id: number;
  name: string;
  religion?: string;
  player_wallet?: string | null;
}

interface WorldMeta {
  turn?: number;
  player_count?: number;
  initialized?: boolean;
  sim_version?: string;
  genesis_block?: number;
}

interface WorldSnapshot {
  meta?: WorldMeta;
  factions?: FactionLite[];
  treaties?: Array<{ faction_a: number; faction_b: number; treaty_type: string; signed_turn?: number }>;
  wars?: Array<{ attacker: number; defender: number }>;
  provinces?: any[];
  armies?: any[];
}

interface RealmSnapshot {
  faction?: number;
  faction_id?: number;
  faction_name?: string;
  treasury?: number;
  prestige?: number;
  provinces?: Array<{ id: number; name?: string; population?: number; tax_rate?: number }>;
  armies?: Array<{ id: number; size?: number; location?: number }>;
  treaties?: Array<{ with: number; type: string }>;
  wars?: Array<{ with: number }>;
}

const TREATY_OPTIONS = [
  'DefensiveAlliance',
  'NonAggression',
  'TradeAgreement',
  'Marriage',
  'Vassalization',
  'WhitePeace',
] as const;

type Tab = 'realm' | 'world' | 'diplomacy';

export default function CrownAshPanel({ walletAddress, className = '' }: CrownAshPanelProps) {
  const wallet = walletAddress || (typeof localStorage !== 'undefined' ? localStorage.getItem('walletAddress') : null) || '';

  const [tab, setTab] = useState<Tab>('realm');
  const [world, setWorld] = useState<WorldSnapshot | null>(null);
  const [realm, setRealm] = useState<RealmSnapshot | null>(null);
  const [realmError, setRealmError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [actionStatus, setActionStatus] = useState<string | null>(null);

  // Diplomacy form state
  const [targetFaction, setTargetFaction] = useState<number>(0);
  const [treatyType, setTreatyType] = useState<string>('DefensiveAlliance');

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const wr = await fetch('/api/v1/crown-ash/world', { signal: AbortSignal.timeout(5000) });
      if (wr.ok) {
        const wj = await wr.json();
        setWorld(wj.data ?? wj);
      }
      if (wallet) {
        try {
          const rr = await fetch(`/api/v1/crown-ash/realm/${wallet}`, { signal: AbortSignal.timeout(5000) });
          if (rr.ok) {
            const rj = await rr.json();
            if (rj.success === false || rj.ok === false) {
              setRealmError(rj.error ?? 'no realm yet');
              setRealm(null);
            } else {
              setRealm(rj.data ?? rj);
              setRealmError(null);
            }
          } else {
            setRealmError(`HTTP ${rr.status}`);
            setRealm(null);
          }
        } catch (e: any) {
          setRealmError(e.message || 'fetch failed');
        }
      } else {
        setRealmError('no wallet — log in first');
      }
    } finally {
      setLoading(false);
    }
  }, [wallet]);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 12_000);
    return () => clearInterval(id);
  }, [refresh]);

  const submitAction = async (action: Record<string, any>) => {
    setActionStatus('submitting…');
    try {
      const r = await fetch('/api/v1/crown-ash/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ wallet, action }),
        signal: AbortSignal.timeout(8000),
      });
      const j = await r.json();
      if (j.success === false || j.ok === false) {
        setActionStatus(`✗ ${j.error ?? 'rejected'}`);
      } else {
        const d = j.data ?? j;
        setActionStatus(`✓ queued (position ${d.queue_position}, turn ${d.turn})`);
        setTimeout(refresh, 800);
      }
    } catch (e: any) {
      setActionStatus(`✗ ${e.message}`);
    }
    setTimeout(() => setActionStatus(null), 6000);
  };

  const joinFaction = async (factionId: number) => {
    setActionStatus('joining…');
    try {
      const r = await fetch('/api/v1/crown-ash/join', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ wallet, faction: factionId }),
        signal: AbortSignal.timeout(8000),
      });
      const j = await r.json();
      if (j.success === false || j.ok === false) {
        setActionStatus(`✗ ${j.error ?? 'rejected'}`);
      } else {
        setActionStatus(`✓ joined faction ${factionId}`);
        setTimeout(refresh, 800);
      }
    } catch (e: any) {
      setActionStatus(`✗ ${e.message}`);
    }
    setTimeout(() => setActionStatus(null), 6000);
  };

  return (
    <div className={`bg-gradient-to-br from-amber-950/30 via-purple-950/20 to-slate-950 border border-amber-700/30 rounded-2xl p-5 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <Crown className="w-6 h-6 text-amber-400" />
          <div>
            <h3 className="text-lg font-bold text-amber-100">Crown & Ash</h3>
            <p className="text-xs text-amber-300/60">
              Medieval grand strategy · {world?.meta?.turn !== undefined ? `Turn ${world.meta.turn}` : 'loading…'} ·{' '}
              {world?.meta?.player_count ?? '?'} players
            </p>
          </div>
        </div>
        <button
          onClick={refresh}
          className="p-2 hover:bg-amber-500/20 rounded-lg transition-colors"
          title="Refresh"
        >
          <RefreshCw className={`w-4 h-4 text-amber-400 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-4 border-b border-amber-700/20 pb-2">
        {(['realm', 'world', 'diplomacy'] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-1.5 rounded-t-lg text-sm font-medium transition-all ${
              tab === t
                ? 'bg-amber-500/20 text-amber-200 border border-amber-500/40 border-b-transparent'
                : 'text-amber-300/50 hover:text-amber-300 hover:bg-amber-500/5'
            }`}
          >
            {t === 'realm' && <Shield className="w-3.5 h-3.5 inline mr-1" />}
            {t === 'world' && <Scroll className="w-3.5 h-3.5 inline mr-1" />}
            {t === 'diplomacy' && <Handshake className="w-3.5 h-3.5 inline mr-1" />}
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {/* Action status banner */}
      {actionStatus && (
        <div
          className={`mb-3 p-2 rounded-lg text-xs font-medium ${
            actionStatus.startsWith('✓')
              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
              : actionStatus.startsWith('✗')
                ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                : 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
          }`}
        >
          {actionStatus}
        </div>
      )}

      {/* ── REALM tab ── */}
      {tab === 'realm' && (
        <div>
          {!wallet ? (
            <div className="flex items-center gap-2 text-amber-300/70 text-sm py-4">
              <AlertCircle className="w-4 h-4" /> Log in to view your realm.
            </div>
          ) : realmError ? (
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-amber-300/70 text-sm">
                <AlertCircle className="w-4 h-4" /> No realm yet — claim a faction:
              </div>
              <div className="grid grid-cols-2 gap-2">
                {(world?.factions ?? []).slice(0, 7).map((f) => {
                  const claimed = !!f.player_wallet;
                  return (
                    <button
                      key={f.id}
                      onClick={() => joinFaction(f.id)}
                      disabled={claimed}
                      className={`text-left p-2 rounded-lg border text-xs transition-all ${
                        claimed
                          ? 'bg-slate-800/40 border-slate-700 text-slate-500 cursor-not-allowed'
                          : 'bg-amber-500/10 border-amber-500/30 text-amber-200 hover:bg-amber-500/20 hover:border-amber-400'
                      }`}
                    >
                      <div className="font-semibold">
                        #{f.id} {f.name}
                      </div>
                      <div className="text-[10px] opacity-70">
                        {f.religion ?? 'unknown'} {claimed ? '· claimed' : '· available'}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <Stat label="Faction" value={`#${realm?.faction ?? realm?.faction_id ?? '?'} ${realm?.faction_name ?? ''}`} />
                <Stat label="Treasury" value={String(realm?.treasury ?? '?')} />
                <Stat label="Prestige" value={String(realm?.prestige ?? '?')} />
                <Stat label="Armies" value={String(realm?.armies?.length ?? 0)} />
                <Stat label="Provinces" value={String(realm?.provinces?.length ?? 0)} />
                <Stat label="Treaties" value={String(realm?.treaties?.length ?? 0)} />
              </div>
              {realm?.provinces && realm.provinces.length > 0 && (
                <div className="mt-3">
                  <div className="text-xs text-amber-300/60 mb-1">Provinces</div>
                  <div className="max-h-32 overflow-y-auto space-y-1 pr-1">
                    {realm.provinces.slice(0, 12).map((p) => (
                      <div
                        key={p.id}
                        className="flex items-center justify-between text-xs bg-slate-800/40 border border-slate-700/40 rounded p-1.5"
                      >
                        <span className="text-amber-200">
                          #{p.id} {p.name ?? '?'}
                        </span>
                        <span className="text-amber-300/60">
                          pop {p.population ?? '?'} · tax {p.tax_rate ?? '?'}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── WORLD tab ── */}
      {tab === 'world' && (
        <div className="space-y-3">
          <div className="grid grid-cols-3 gap-2">
            <Stat label="Turn" value={String(world?.meta?.turn ?? '?')} />
            <Stat label="Players" value={String(world?.meta?.player_count ?? 0)} />
            <Stat label="Sim" value={world?.meta?.sim_version ?? '?'} />
            <Stat label="Provinces" value={String((world?.provinces ?? []).length)} />
            <Stat label="Factions" value={String((world?.factions ?? []).length)} />
            <Stat label="Active wars" value={String((world?.wars ?? []).length)} />
          </div>
          <div>
            <div className="text-xs text-amber-300/60 mb-1">Factions</div>
            <div className="grid grid-cols-2 gap-1.5">
              {(world?.factions ?? []).slice(0, 14).map((f) => {
                const me = !!f.player_wallet && f.player_wallet === wallet;
                return (
                  <div
                    key={f.id}
                    className={`p-1.5 rounded border text-xs ${
                      me ? 'bg-amber-500/20 border-amber-400/60 text-amber-100' : 'bg-slate-800/30 border-slate-700/30 text-amber-200/80'
                    }`}
                  >
                    <span className="font-mono">#{f.id}</span> {f.name}{' '}
                    <span className="text-[10px] opacity-60">{f.player_wallet ? '🧙' : ''}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* ── DIPLOMACY tab ── */}
      {tab === 'diplomacy' && (
        <div className="space-y-3">
          <div className="bg-slate-900/40 border border-amber-700/20 rounded-lg p-3 space-y-2">
            <div className="text-sm font-semibold text-amber-200 flex items-center gap-2">
              <Handshake className="w-4 h-4" /> Propose treaty
            </div>
            <div className="grid grid-cols-3 gap-2 items-end">
              <label className="text-xs text-amber-300/70">
                Target faction
                <select
                  value={targetFaction}
                  onChange={(e) => setTargetFaction(parseInt(e.target.value))}
                  className="mt-0.5 w-full bg-slate-800 border border-amber-700/30 rounded text-amber-100 text-sm p-1"
                >
                  {(world?.factions ?? []).map((f) => (
                    <option key={f.id} value={f.id}>
                      #{f.id} {f.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-xs text-amber-300/70">
                Type
                <select
                  value={treatyType}
                  onChange={(e) => setTreatyType(e.target.value)}
                  className="mt-0.5 w-full bg-slate-800 border border-amber-700/30 rounded text-amber-100 text-sm p-1"
                >
                  {TREATY_OPTIONS.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              </label>
              <button
                onClick={() => submitAction({ ProposeTreaty: { target: targetFaction, treaty: treatyType } })}
                disabled={!wallet || !realm}
                className="px-3 py-1.5 rounded bg-gradient-to-r from-amber-500/30 to-orange-500/30 border border-amber-500/50 text-amber-100 text-sm font-semibold hover:from-amber-500/40 hover:to-orange-500/40 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
              >
                Propose
              </button>
            </div>
            <p className="text-[10px] text-amber-300/50">
              Proposal queues on the chain and resolves on the next sim tick. {realm ? '' : 'You must own a realm first.'}
            </p>
          </div>

          <div className="bg-slate-900/40 border border-amber-700/20 rounded-lg p-3">
            <div className="text-sm font-semibold text-amber-200 mb-2 flex items-center gap-2">
              <Sword className="w-4 h-4" /> Active treaties
            </div>
            <div className="max-h-32 overflow-y-auto space-y-1">
              {(world?.treaties ?? []).length === 0 ? (
                <div className="text-xs text-amber-300/50">No treaties yet — the world is unclaimed.</div>
              ) : (
                (world?.treaties ?? []).slice(0, 12).map((t, i) => (
                  <div key={i} className="text-xs flex items-center justify-between bg-slate-800/40 border border-slate-700/40 rounded p-1.5">
                    <span className="text-amber-200">
                      #{t.faction_a} ↔ #{t.faction_b}
                    </span>
                    <span className="text-amber-300/70">
                      {t.treaty_type}
                      {t.signed_turn !== undefined && ` · turn ${t.signed_turn}`}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-800/40 border border-amber-700/20 rounded-lg p-2">
      <div className="text-[10px] uppercase tracking-wider text-amber-300/50">{label}</div>
      <div className="text-sm font-semibold text-amber-100 truncate" title={value}>
        {value}
      </div>
    </div>
  );
}
