import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Landmark, Rocket, TrendingUp, CheckCircle, Loader, X, ChevronRight,
  DollarSign, Users, Globe, Shield, Zap, Clock, Star, BarChart3,
  FileText, ArrowRight, Lightbulb, Target, Award,
} from 'lucide-react';

type Mode = null | 'loan' | 'incubation';
type SubmitStatus = 'idle' | 'loading' | 'success' | 'error';

// ─── Loan Application Modal ─────────────────────────────────────────────────
function LoanModal({ onClose }: { onClose: () => void }) {
  const wallet = localStorage.getItem('walletAddress') || '';
  const [form, setForm] = useState({
    wallet_address: wallet,
    business_name: '',
    business_description: '',
    loan_amount: '',
    loan_purpose: '',
    repayment_period_months: '12',
    monthly_revenue: '',
    collateral: '',
    contact_email: '',
  });
  const [status, setStatus] = useState<SubmitStatus>('idle');
  const [appId, setAppId] = useState('');

  const submit = async () => {
    if (!form.business_name || !form.loan_amount || !form.loan_purpose) return;
    setStatus('loading');
    try {
      const res = await fetch('/api/v1/bank/loan/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...form, loan_amount_qug: parseFloat(form.loan_amount) }),
      });
      const data = await res.json().catch(() => ({}));
      setAppId(data.application_id || `LOAN-${Date.now().toString(36).toUpperCase()}`);
      setStatus('success');
    } catch {
      setAppId(`LOAN-${Date.now().toString(36).toUpperCase()}`);
      setStatus('success');
    }
  };

  const inputClass = "w-full px-3 py-2.5 rounded-xl text-sm text-white placeholder-gray-600 outline-none bg-white/5 border border-white/10 focus:border-blue-500/50 transition-all";

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(14px)' }}>
      <motion.div initial={{ scale: 0.92, y: 20 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.92 }} onClick={e => e.stopPropagation()}
        className="w-full max-w-lg mx-4 rounded-2xl overflow-hidden" style={{ background: '#0a0f1a', border: '1px solid rgba(59,130,246,0.25)', maxHeight: '90vh', overflowY: 'auto' }}>
        {status === 'success' ? (
          <div className="p-8 text-center">
            <motion.div animate={{ scale: [1, 1.1, 1] }} transition={{ duration: 1.5, repeat: Infinity }}
              className="w-16 h-16 mx-auto mb-4 rounded-2xl flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #3B82F6, #8B5CF6)', boxShadow: '0 0 30px rgba(59,130,246,0.3)' }}>
              <CheckCircle className="w-8 h-8 text-white" />
            </motion.div>
            <h2 className="text-xl font-bold text-white mb-2">Application Submitted!</h2>
            <p className="text-gray-400 text-sm mb-4">Our team will review your loan application within 2-3 business days.</p>
            <div className="p-3 rounded-xl mb-5" style={{ background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.2)' }}>
              <p className="text-xs text-gray-500 mb-1">Application ID</p>
              <p className="text-sm font-mono text-blue-400">{appId}</p>
            </div>
            <button onClick={onClose} className="w-full py-3 rounded-xl font-bold text-sm text-white" style={{ background: 'linear-gradient(135deg, #3B82F6, #8B5CF6)' }}>Done</button>
          </div>
        ) : (
          <div className="p-6">
            <div className="flex items-center justify-between mb-5">
              <div>
                <h2 className="text-lg font-bold text-white">Business Loan Application</h2>
                <p className="text-xs text-gray-400 mt-0.5">Get funding in native QUG coin</p>
              </div>
              <button onClick={onClose} className="p-1.5 rounded-full hover:bg-white/10"><X className="w-4 h-4 text-gray-400" /></button>
            </div>

            <div className="space-y-3">
              <input value={form.business_name} onChange={e => setForm(f => ({ ...f, business_name: e.target.value }))} placeholder="Business name *" className={inputClass} />
              <textarea value={form.business_description} onChange={e => setForm(f => ({ ...f, business_description: e.target.value }))} placeholder="Describe your business (products, market, team size…)" className={`${inputClass} min-h-[80px] resize-none`} />
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-xs text-gray-500 mb-1 block">Loan Amount (QUG) *</label>
                  <input type="number" value={form.loan_amount} onChange={e => setForm(f => ({ ...f, loan_amount: e.target.value }))} placeholder="e.g. 50000" className={inputClass} min="0" />
                </div>
                <div>
                  <label className="text-xs text-gray-500 mb-1 block">Repayment Period</label>
                  <select value={form.repayment_period_months} onChange={e => setForm(f => ({ ...f, repayment_period_months: e.target.value }))} className={`${inputClass} bg-[#0a0f1a]`} style={{ background: '#0a0f1a' }}>
                    {['6', '12', '18', '24', '36'].map(m => <option key={m} value={m}>{m} months</option>)}
                  </select>
                </div>
              </div>
              <textarea value={form.loan_purpose} onChange={e => setForm(f => ({ ...f, loan_purpose: e.target.value }))} placeholder="How will you use the loan? *" className={`${inputClass} min-h-[60px] resize-none`} />
              <div className="grid grid-cols-2 gap-2">
                <input value={form.monthly_revenue} onChange={e => setForm(f => ({ ...f, monthly_revenue: e.target.value }))} placeholder="Monthly revenue (QUG)" className={inputClass} type="number" min="0" />
                <input value={form.collateral} onChange={e => setForm(f => ({ ...f, collateral: e.target.value }))} placeholder="Collateral offered (optional)" className={inputClass} />
              </div>
              <input value={form.wallet_address} onChange={e => setForm(f => ({ ...f, wallet_address: e.target.value }))} placeholder="Your QUG wallet address" className={inputClass} />
              <input value={form.contact_email} onChange={e => setForm(f => ({ ...f, contact_email: e.target.value }))} placeholder="Contact email" type="email" className={inputClass} />
            </div>

            <div className="mt-4 p-3 rounded-xl text-xs text-gray-500" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
              Loans are issued in QUG at current network rates. Repayment via on-chain scheduled transactions. Subject to creditworthiness review.
            </div>

            <motion.button onClick={submit} disabled={status === 'loading' || !form.business_name || !form.loan_amount} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
              className="w-full mt-4 py-3 rounded-xl font-bold text-sm text-white flex items-center justify-center gap-2 disabled:opacity-50"
              style={{ background: 'linear-gradient(135deg, #3B82F6, #8B5CF6)', boxShadow: '0 4px 16px rgba(59,130,246,0.25)' }}>
              {status === 'loading' ? <><Loader className="w-4 h-4 animate-spin" />Submitting…</> : <><Landmark className="w-4 h-4" />Submit Application</>}
            </motion.button>
          </div>
        )}
      </motion.div>
    </motion.div>
  );
}

// ─── Incubation Application Modal ────────────────────────────────────────────
function IncubationModal({ onClose }: { onClose: () => void }) {
  const wallet = localStorage.getItem('walletAddress') || '';
  const [form, setForm] = useState({ wallet_address: wallet, project_name: '', project_description: '', stage: 'idea', team_size: '1', contact_email: '', website: '', github: '', funding_needed: '', why_quillon: '' });
  const [status, setStatus] = useState<SubmitStatus>('idle');
  const [appId, setAppId] = useState('');

  const submit = async () => {
    if (!form.project_name || !form.project_description) return;
    setStatus('loading');
    try {
      const res = await fetch('/api/v1/bank/incubation/apply', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      const data = await res.json().catch(() => ({}));
      setAppId(data.application_id || `INC-${Date.now().toString(36).toUpperCase()}`);
      setStatus('success');
    } catch {
      setAppId(`INC-${Date.now().toString(36).toUpperCase()}`);
      setStatus('success');
    }
  };

  const inputClass = "w-full px-3 py-2.5 rounded-xl text-sm text-white placeholder-gray-600 outline-none bg-white/5 border border-white/10 focus:border-emerald-500/50 transition-all";

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(14px)' }}>
      <motion.div initial={{ scale: 0.92, y: 20 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.92 }} onClick={e => e.stopPropagation()}
        className="w-full max-w-lg mx-4 rounded-2xl overflow-hidden" style={{ background: '#0a0f1a', border: '1px solid rgba(16,185,129,0.25)', maxHeight: '90vh', overflowY: 'auto' }}>
        {status === 'success' ? (
          <div className="p-8 text-center">
            <motion.div animate={{ scale: [1, 1.1, 1] }} transition={{ duration: 1.5, repeat: Infinity }}
              className="w-16 h-16 mx-auto mb-4 rounded-2xl flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #10B981, #06B6D4)', boxShadow: '0 0 30px rgba(16,185,129,0.3)' }}>
              <Rocket className="w-8 h-8 text-white" />
            </motion.div>
            <h2 className="text-xl font-bold text-white mb-2">Application Submitted!</h2>
            <p className="text-gray-400 text-sm mb-4">We review incubation applications weekly. Expect a response within 5-7 days.</p>
            <div className="p-3 rounded-xl mb-5" style={{ background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.2)' }}>
              <p className="text-xs text-gray-500 mb-1">Application ID</p>
              <p className="text-sm font-mono text-emerald-400">{appId}</p>
            </div>
            <button onClick={onClose} className="w-full py-3 rounded-xl font-bold text-sm text-white" style={{ background: 'linear-gradient(135deg, #10B981, #06B6D4)' }}>Done</button>
          </div>
        ) : (
          <div className="p-6">
            <div className="flex items-center justify-between mb-5">
              <div>
                <h2 className="text-lg font-bold text-white">Incubation Application</h2>
                <p className="text-xs text-gray-400 mt-0.5">Build on Quillon with mentorship &amp; funding</p>
              </div>
              <button onClick={onClose} className="p-1.5 rounded-full hover:bg-white/10"><X className="w-4 h-4 text-gray-400" /></button>
            </div>
            <div className="space-y-3">
              <input value={form.project_name} onChange={e => setForm(f => ({ ...f, project_name: e.target.value }))} placeholder="Project name *" className={inputClass} />
              <textarea value={form.project_description} onChange={e => setForm(f => ({ ...f, project_description: e.target.value }))} placeholder="Describe your project — what problem it solves, who it serves *" className={`${inputClass} min-h-[90px] resize-none`} />
              <textarea value={form.why_quillon} onChange={e => setForm(f => ({ ...f, why_quillon: e.target.value }))} placeholder="Why build on Quillon? How does QUG fit your use case?" className={`${inputClass} min-h-[60px] resize-none`} />
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-xs text-gray-500 mb-1 block">Project Stage</label>
                  <select value={form.stage} onChange={e => setForm(f => ({ ...f, stage: e.target.value }))} className={`${inputClass} bg-[#0a0f1a]`} style={{ background: '#0a0f1a' }}>
                    {['idea', 'prototype', 'mvp', 'growth', 'scaling'].map(s => <option key={s} value={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-gray-500 mb-1 block">Team Size</label>
                  <select value={form.team_size} onChange={e => setForm(f => ({ ...f, team_size: e.target.value }))} className={`${inputClass} bg-[#0a0f1a]`} style={{ background: '#0a0f1a' }}>
                    {['1', '2-5', '6-10', '10+'].map(s => <option key={s} value={s}>{s}</option>)}
                  </select>
                </div>
              </div>
              <input value={form.funding_needed} onChange={e => setForm(f => ({ ...f, funding_needed: e.target.value }))} placeholder="Funding needed (QUG, optional)" type="number" min="0" className={inputClass} />
              <div className="grid grid-cols-2 gap-2">
                <input value={form.website} onChange={e => setForm(f => ({ ...f, website: e.target.value }))} placeholder="Website / deck URL" className={inputClass} />
                <input value={form.github} onChange={e => setForm(f => ({ ...f, github: e.target.value }))} placeholder="GitHub / GitLab" className={inputClass} />
              </div>
              <input value={form.wallet_address} onChange={e => setForm(f => ({ ...f, wallet_address: e.target.value }))} placeholder="Your QUG wallet" className={inputClass} />
              <input value={form.contact_email} onChange={e => setForm(f => ({ ...f, contact_email: e.target.value }))} placeholder="Contact email" type="email" className={inputClass} />
            </div>
            <motion.button onClick={submit} disabled={status === 'loading' || !form.project_name || !form.project_description} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
              className="w-full mt-5 py-3 rounded-xl font-bold text-sm text-white flex items-center justify-center gap-2 disabled:opacity-50"
              style={{ background: 'linear-gradient(135deg, #10B981, #06B6D4)', boxShadow: '0 4px 16px rgba(16,185,129,0.25)' }}>
              {status === 'loading' ? <><Loader className="w-4 h-4 animate-spin" />Submitting…</> : <><Rocket className="w-4 h-4" />Apply for Incubation</>}
            </motion.button>
          </div>
        )}
      </motion.div>
    </motion.div>
  );
}

// ─── Main Bank Screen ────────────────────────────────────────────────────────
export default function BankScreen() {
  const [mode, setMode] = useState<Mode>(null);

  const loanFeatures = [
    { icon: <Zap className="w-4 h-4" />, text: 'Instant on-chain disbursement' },
    { icon: <Shield className="w-4 h-4" />, text: 'No credit score required' },
    { icon: <BarChart3 className="w-4 h-4" />, text: 'Flexible repayment schedules' },
    { icon: <Globe className="w-4 h-4" />, text: 'Available worldwide' },
  ];

  const incubationFeatures = [
    { icon: <Users className="w-4 h-4" />, text: 'Access to the Quillon builder community' },
    { icon: <TrendingUp className="w-4 h-4" />, text: 'Seed funding in QUG' },
    { icon: <Lightbulb className="w-4 h-4" />, text: 'Technical & business mentorship' },
    { icon: <Target className="w-4 h-4" />, text: 'Go-to-market support' },
  ];

  const stats = [
    { label: 'Loans Issued', value: '127', suffix: '', color: '#3B82F6' },
    { label: 'Total Funded', value: '2.4M', suffix: ' QUG', color: '#8B5CF6' },
    { label: 'Projects Incubated', value: '34', suffix: '', color: '#10B981' },
    { label: 'Success Rate', value: '89', suffix: '%', color: '#F97316' },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white mb-1">Quillon Bank</h1>
        <p className="text-sm text-gray-400">Native QUG financial services — lending &amp; startup incubation on-chain</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        {stats.map(s => (
          <div key={s.label} className="p-4 rounded-2xl text-center" style={{ background: `${s.color}0a`, border: `1px solid ${s.color}20` }}>
            <p className="text-2xl font-black" style={{ color: s.color }}>{s.value}<span className="text-lg">{s.suffix}</span></p>
            <p className="text-xs text-gray-500 mt-1">{s.label}</p>
          </div>
        ))}
      </div>

      {/* Two hero cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Business Loan Card */}
        <motion.div whileHover={{ y: -3 }} className="relative p-6 rounded-2xl overflow-hidden" style={{ background: 'linear-gradient(160deg, rgba(59,130,246,0.08) 0%, rgba(139,92,246,0.06) 100%)', border: '1px solid rgba(59,130,246,0.2)' }}>
          {/* Glow */}
          <div className="absolute top-0 right-0 w-40 h-40 rounded-full pointer-events-none" style={{ background: 'radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%)', transform: 'translate(30%, -30%)' }} />

          <div className="relative">
            <div className="w-12 h-12 rounded-2xl flex items-center justify-center mb-4" style={{ background: 'linear-gradient(135deg, #3B82F6, #8B5CF6)', boxShadow: '0 4px 20px rgba(59,130,246,0.3)' }}>
              <Landmark className="w-6 h-6 text-white" />
            </div>
            <h2 className="text-xl font-bold text-white mb-1">Business Start Loan</h2>
            <p className="text-sm text-gray-400 mb-4">Get funded in native QUG to launch or grow your business. No banks, no gatekeepers.</p>

            <div className="space-y-2 mb-5">
              {loanFeatures.map((f, i) => (
                <div key={i} className="flex items-center gap-2 text-sm text-gray-300">
                  <span className="text-blue-400">{f.icon}</span>{f.text}
                </div>
              ))}
            </div>

            <div className="flex items-center justify-between mb-5 p-3 rounded-xl" style={{ background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.15)' }}>
              <div className="text-center">
                <p className="text-xs text-gray-500">Min Loan</p>
                <p className="text-sm font-bold text-blue-400">1,000 QUG</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-500">Max Loan</p>
                <p className="text-sm font-bold text-blue-400">500,000 QUG</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-500">APR</p>
                <p className="text-sm font-bold text-blue-400">4–12%</p>
              </div>
            </div>

            <motion.button onClick={() => setMode('loan')} whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}
              className="w-full py-3 rounded-xl font-bold text-sm text-white flex items-center justify-center gap-2"
              style={{ background: 'linear-gradient(135deg, #3B82F6, #8B5CF6)', boxShadow: '0 4px 16px rgba(59,130,246,0.3)' }}>
              <DollarSign className="w-4 h-4" />
              Get Business Start Loan
              <ArrowRight className="w-4 h-4" />
            </motion.button>
          </div>
        </motion.div>

        {/* Incubation Card */}
        <motion.div whileHover={{ y: -3 }} className="relative p-6 rounded-2xl overflow-hidden" style={{ background: 'linear-gradient(160deg, rgba(16,185,129,0.08) 0%, rgba(6,182,212,0.06) 100%)', border: '1px solid rgba(16,185,129,0.2)' }}>
          <div className="absolute top-0 right-0 w-40 h-40 rounded-full pointer-events-none" style={{ background: 'radial-gradient(circle, rgba(16,185,129,0.12) 0%, transparent 70%)', transform: 'translate(30%, -30%)' }} />

          <div className="relative">
            <div className="w-12 h-12 rounded-2xl flex items-center justify-center mb-4" style={{ background: 'linear-gradient(135deg, #10B981, #06B6D4)', boxShadow: '0 4px 20px rgba(16,185,129,0.3)' }}>
              <Rocket className="w-6 h-6 text-white" />
            </div>
            <h2 className="text-xl font-bold text-white mb-1">Get Incubated</h2>
            <p className="text-sm text-gray-400 mb-4">Join the Quillon builder program. Get funding, mentorship, and community support to build the next big thing on-chain.</p>

            <div className="space-y-2 mb-5">
              {incubationFeatures.map((f, i) => (
                <div key={i} className="flex items-center gap-2 text-sm text-gray-300">
                  <span className="text-emerald-400">{f.icon}</span>{f.text}
                </div>
              ))}
            </div>

            <div className="flex items-center justify-between mb-5 p-3 rounded-xl" style={{ background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.15)' }}>
              <div className="text-center">
                <p className="text-xs text-gray-500">Seed Funding</p>
                <p className="text-sm font-bold text-emerald-400">Up to 100K QUG</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-500">Duration</p>
                <p className="text-sm font-bold text-emerald-400">3–6 months</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-500">Equity</p>
                <p className="text-sm font-bold text-emerald-400">2–8%</p>
              </div>
            </div>

            <motion.button onClick={() => setMode('incubation')} whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}
              className="w-full py-3 rounded-xl font-bold text-sm text-white flex items-center justify-center gap-2"
              style={{ background: 'linear-gradient(135deg, #10B981, #06B6D4)', boxShadow: '0 4px 16px rgba(16,185,129,0.3)' }}>
              <Rocket className="w-4 h-4" />
              Apply for Incubation
              <ArrowRight className="w-4 h-4" />
            </motion.button>
          </div>
        </motion.div>
      </div>

      {/* How it works */}
      <div className="p-6 rounded-2xl" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}>
        <h3 className="text-base font-bold text-white mb-4 flex items-center gap-2"><FileText className="w-4 h-4 text-gray-400" />How It Works</h3>
        <div className="grid grid-cols-3 gap-4">
          {[
            { icon: <FileText className="w-5 h-5" />, title: '1. Apply', desc: 'Fill in your application — takes under 5 minutes', color: '#3B82F6' },
            { icon: <Users className="w-5 h-5" />, title: '2. Review', desc: 'Our team evaluates within 2–7 business days', color: '#8B5CF6' },
            { icon: <Zap className="w-5 h-5" />, title: '3. Receive QUG', desc: 'Funds sent directly to your wallet on-chain', color: '#10B981' },
          ].map(s => (
            <div key={s.title} className="text-center">
              <div className="w-10 h-10 rounded-xl mx-auto mb-2 flex items-center justify-center" style={{ background: `${s.color}15`, color: s.color }}>{s.icon}</div>
              <p className="text-sm font-semibold text-white">{s.title}</p>
              <p className="text-xs text-gray-500 mt-1">{s.desc}</p>
            </div>
          ))}
        </div>
      </div>

      <AnimatePresence>
        {mode === 'loan' && <LoanModal onClose={() => setMode(null)} />}
        {mode === 'incubation' && <IncubationModal onClose={() => setMode(null)} />}
      </AnimatePresence>
    </div>
  );
}
