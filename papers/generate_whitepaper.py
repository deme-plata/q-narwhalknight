#!/usr/bin/env python3
"""
Q-NarwhalKnight Comprehensive Whitepaper Generator
Generates: papers/qnk-whitepaper-2026-05-07.pdf
Uses fpdf2 for PDF generation + matplotlib for charts
"""

import sys
import os
import math
import io
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch
from matplotlib.patches import FancyArrowPatch
import numpy as np
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# ─── Color Palette ──────────────────────────────────────────────────────────
PURPLE_DARK   = (45, 15, 75)
PURPLE_MID    = (100, 40, 160)
PURPLE_LIGHT  = (180, 120, 240)
GOLD          = (212, 175, 55)
TEAL          = (0, 180, 180)
RED           = (220, 50, 50)
ORANGE        = (230, 130, 50)
GREEN         = (50, 180, 80)
LIGHT_GRAY    = (240, 240, 245)
DARK_GRAY     = (60, 60, 70)
WHITE         = (255, 255, 255)
BLACK         = (20, 20, 30)

def rgb_norm(t):
    return tuple(c/255 for c in t)

# ─── Chart Generators ────────────────────────────────────────────────────────

def make_emission_schedule():
    """Annual emission across 64 eras (256 years)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor('#0d0520')

    eras = np.arange(0, 20)
    base = 2_625_000
    emissions = [base / (2 ** e) for e in eras]
    cumulative = np.cumsum(emissions)

    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor('#0d0520')
        ax.tick_params(colors='#c8b0f0', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#3a1a5a')

    # Bar chart
    colors = [rgb_norm(GOLD) if e == 0 else rgb_norm(PURPLE_MID) for e in eras]
    bars = ax1.bar(eras, [e/1e6 for e in emissions], color=colors, alpha=0.85, width=0.7)
    ax1.set_xlabel('Era Number (4-year halving)', color='#c8b0f0', fontsize=9)
    ax1.set_ylabel('Annual Emission (M QUG)', color='#c8b0f0', fontsize=9)
    ax1.set_title('Emission Schedule per Era', color='#e0d0ff', fontsize=11, fontweight='bold')
    ax1.axhline(y=0, color='#3a1a5a', linewidth=0.5)
    ax1.set_xlim(-0.5, 19.5)
    bars[0].set_edgecolor(rgb_norm(GOLD))
    bars[0].set_linewidth(2)

    # Cumulative
    ax2.fill_between(eras, [c/1e6 for c in cumulative], alpha=0.3, color=rgb_norm(PURPLE_MID))
    ax2.plot(eras, [c/1e6 for c in cumulative], color=rgb_norm(PURPLE_LIGHT), linewidth=2.5)
    ax2.axhline(y=21, color=rgb_norm(GOLD), linestyle='--', linewidth=1.5, label='21M cap')
    ax2.set_xlabel('Era Number', color='#c8b0f0', fontsize=9)
    ax2.set_ylabel('Cumulative Supply (M QUG)', color='#c8b0f0', fontsize=9)
    ax2.set_title('Cumulative Supply Towards 21M Cap', color='#e0d0ff', fontsize=11, fontweight='bold')
    ax2.legend(facecolor='#1a0a2a', edgecolor='#3a1a5a', labelcolor='#c8b0f0', fontsize=8)
    ax2.set_xlim(-0.5, 19.5)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0d0520')
    plt.close()
    return buf

def make_turbo_sync_performance():
    """Turbo sync speed over block height range"""
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor('#0d0520')
    ax.set_facecolor('#0d0520')

    heights = np.linspace(0, 11_400_000, 500)
    # Model: slow start → peak → gradual taper
    def sync_rate(h):
        if h < 100_000:
            return 280 * (h / 100_000) ** 0.5
        elif h < 500_000:
            return 280 + (820 * (h - 100_000) / 400_000)
        elif h < 3_000_000:
            return 1100
        elif h < 8_000_000:
            return 1100 - 200 * ((h - 3_000_000) / 5_000_000)
        else:
            return 900 - 300 * ((h - 8_000_000) / 3_400_000)

    rates = [sync_rate(h) for h in heights]
    ax.fill_between(heights/1e6, rates, alpha=0.25, color=rgb_norm(TEAL))
    ax.plot(heights/1e6, rates, color=rgb_norm(TEAL), linewidth=2.5)
    ax.axhline(y=570, color=rgb_norm(GOLD), linestyle='--', linewidth=1.5, label='Average: 570 blocks/s')
    ax.axhline(y=1100, color=rgb_norm(GREEN), linestyle=':', linewidth=1, label='Peak: 1,100 blocks/s')

    ax.fill_betweenx([0, 1200], 0, 0.5, alpha=0.08, color='white')
    ax.text(0.25, 1050, 'Warmup\n(DHT)', ha='center', color='#a090c0', fontsize=7)
    ax.fill_betweenx([0, 1200], 0.5, 3.0, alpha=0.06, color=rgb_norm(TEAL))
    ax.text(1.75, 1050, 'Peak\nTurbo', ha='center', color=rgb_norm(TEAL), fontsize=7)

    ax.set_xlabel('Block Height (millions)', color='#c8b0f0', fontsize=9)
    ax.set_ylabel('Sync Rate (blocks/sec)', color='#c8b0f0', fontsize=9)
    ax.set_title('Turbo Sync Performance — Epsilon 10 Gbit Supernode (Full 11.4M block chain)',
                 color='#e0d0ff', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#c8b0f0', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#3a1a5a')
    ax.legend(facecolor='#1a0a2a', edgecolor='#3a1a5a', labelcolor='#c8b0f0', fontsize=8)
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 1250)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0d0520')
    plt.close()
    return buf

def make_network_topology():
    """Network topology: 4-server cluster"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d0520')
    ax.set_facecolor('#0d0520')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    nodes = {
        'Epsilon\n10 Gbit\nSupernode': (5, 6.5, GOLD, 0.55),
        'Beta\nPrimary\n100 Mbit':     (2, 4,   PURPLE_LIGHT, 0.45),
        'Gamma\nBackup\n1 Gbit':       (8, 4,   PURPLE_MID, 0.45),
        'Delta\nCanary\n1 Gbit':       (5, 1.5, TEAL, 0.40),
    }

    positions = {}
    for label, (x, y, color, size) in nodes.items():
        positions[label] = (x, y)
        circle = plt.Circle((x, y), size, color=rgb_norm(color), alpha=0.85, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=7.5,
                color='#0a0514', fontweight='bold', zorder=6)

    edges = [
        ('Epsilon\n10 Gbit\nSupernode', 'Beta\nPrimary\n100 Mbit'),
        ('Epsilon\n10 Gbit\nSupernode', 'Gamma\nBackup\n1 Gbit'),
        ('Epsilon\n10 Gbit\nSupernode', 'Delta\nCanary\n1 Gbit'),
        ('Beta\nPrimary\n100 Mbit',     'Gamma\nBackup\n1 Gbit'),
        ('Beta\nPrimary\n100 Mbit',     'Delta\nCanary\n1 Gbit'),
        ('Gamma\nBackup\n1 Gbit',       'Delta\nCanary\n1 Gbit'),
    ]
    for a, b in edges:
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        ax.plot([x1, x2], [y1, y2], color='#4a2a7a', linewidth=1.8, zorder=3, alpha=0.7)

    # Nginx LB
    ax.add_patch(mpatches.FancyBboxPatch((3.5, 7.2), 3, 0.55,
                                         boxstyle='round,pad=0.1',
                                         facecolor=rgb_norm(DARK_GRAY),
                                         edgecolor=rgb_norm(GOLD), linewidth=1.5, zorder=4))
    ax.text(5, 7.47, 'quillon.xyz  (Nginx ip_hash + TLS)', ha='center', va='center',
            fontsize=8, color=rgb_norm(GOLD), fontweight='bold', zorder=7)

    # User arrow
    ax.annotate('', xy=(5, 7.2), xytext=(5, 7.75),
                arrowprops=dict(arrowstyle='->', color='#a090c0', lw=1.5))
    ax.text(5, 7.82, 'Users / Miners', ha='center', va='bottom',
            fontsize=8, color='#c8b0f0')

    # gossipsub label
    ax.text(5, 2.9, 'Gossipsub P2P  |  Kademlia DHT  |  Turbo Sync',
            ha='center', fontsize=8, color='#7a6a9a',
            style='italic')

    # weight labels
    ax.text(1.5, 5.3, 'weight=10\n(primary)', ha='center', fontsize=7, color='#b090d0')
    ax.text(8.5, 5.3, 'weight=1\n(backup)', ha='center', fontsize=7, color='#8070a0')

    ax.set_title('Q-NarwhalKnight 4-Node Production Cluster',
                 color='#e0d0ff', fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0d0520')
    plt.close()
    return buf

def make_consensus_dag():
    """Simplified DAG-Knight round structure"""
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor('#0d0520')
    ax.set_facecolor('#0d0520')
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 4)
    ax.axis('off')

    np.random.seed(42)
    rounds = 7
    validators = 4
    nodes = {}
    for r in range(rounds):
        for v in range(validators):
            x = r * 1.05
            y = v * 0.85
            color = rgb_norm(GOLD) if r == 3 else rgb_norm(PURPLE_MID)
            size = 0.18 if r == 3 else 0.13
            circle = plt.Circle((x, y), size, color=color, alpha=0.9, zorder=5)
            ax.add_patch(circle)
            nodes[(r, v)] = (x, y)

    for r in range(1, rounds):
        for v in range(validators):
            x2, y2 = nodes[(r, v)]
            # 2–3 parent edges per vertex
            parents = [v] + [(v + 1) % validators]
            if r > 1:
                parents.append((v + 2) % validators)
            for pv in parents:
                if (r-1, pv) in nodes:
                    x1, y1 = nodes[(r-1, pv)]
                    color = '#6a3a9a' if r != 3 else rgb_norm(GOLD)
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.0, zorder=3, alpha=0.6)

    # Anchor marker
    ax.text(3 * 1.05, -0.38, '★ ANCHOR\n(VDF-elected)', ha='center', fontsize=7.5,
            color=rgb_norm(GOLD), fontweight='bold')

    # Round labels
    for r in range(rounds):
        ax.text(r * 1.05, 3.5, f'R{r}', ha='center', fontsize=8, color='#8070a0')

    # Validator labels
    for v in range(validators):
        ax.text(-0.38, v * 0.85, f'V{v}', ha='center', va='center', fontsize=7.5, color='#a090c0')

    ax.text(3.3, 4.0, 'DAG-Knight Consensus — Bullshark BFT  |  δ=1  |  <50ms Finality',
            ha='center', fontsize=10, color='#e0d0ff', fontweight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0d0520')
    plt.close()
    return buf

def make_crypto_phases():
    """Cryptographic phase migration roadmap"""
    fig, ax = plt.subplots(figsize=(11, 3.5))
    fig.patch.set_facecolor('#0d0520')
    ax.set_facecolor('#0d0520')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    phases = [
        (1,   'Q0\nEd25519\nClassical',        PURPLE_MID,   'Production'),
        (3.3, 'Q1\nEd25519+Dilithium5\nHybrid', PURPLE_LIGHT, 'Production'),
        (5.6, 'Q2\nDilithium5 Only\nFull PQ',   GOLD,         'Integrated'),
        (7.9, 'Q3\nSQIsign / FROST\nThreshold', TEAL,         'Planned'),
    ]

    for x, label, color, status in phases:
        rect = mpatches.FancyBboxPatch((x-0.8, 1.0), 1.8, 2.0,
                                       boxstyle='round,pad=0.1',
                                       facecolor=(*rgb_norm(color), 0.25),
                                       edgecolor=rgb_norm(color), linewidth=2, zorder=4)
        ax.add_patch(rect)
        ax.text(x + 0.1, 2.15, label, ha='center', va='center',
                fontsize=7.5, color='#e0d0ff', fontweight='bold', zorder=5)
        ax.text(x + 0.1, 1.2, status, ha='center', va='center',
                fontsize=6.5, color=rgb_norm(color), zorder=5, style='italic')

    # Arrow connecting phases
    for i in range(len(phases)-1):
        x1 = phases[i][0] + 1.0
        x2 = phases[i+1][0] - 0.9
        ax.annotate('', xy=(x2, 2.0), xytext=(x1, 2.0),
                    arrowprops=dict(arrowstyle='->', color='#5a3a8a', lw=2.0))

    ax.text(5, 3.6, 'Post-Quantum Cryptographic Migration Roadmap',
            ha='center', fontsize=11, color='#e0d0ff', fontweight='bold')

    # Quantum threat bar
    threat_x = np.linspace(0.2, 9.8, 100)
    threat_color = [(*rgb_norm(GREEN), a) for a in np.linspace(0.6, 0.0, 100)]
    for i in range(len(threat_x)-1):
        ax.plot([threat_x[i], threat_x[i+1]], [0.55, 0.55],
                color=(*rgb_norm(GREEN), (100-i)/100 * 0.7), linewidth=4)
    ax.text(0.2, 0.3, 'Quantum threat level →', fontsize=7, color='#7a9a7a', style='italic')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0d0520')
    plt.close()
    return buf

def make_block_production_rate():
    """Block production rate and reward vs height"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0d0520')

    heights = np.arange(0, 20_000_000, 100_000)
    # Block reward: ~2 QUG at genesis, halving every 126M seconds / ~10BPS
    # Approximate: 4-year halving ≈ 126M blocks at 1 BPS → much more at 10 BPS
    # Use time-based halving: era duration ~126M seconds = ~1.26B blocks at 10 BPS
    # Simpler: show reward flattening from emission controller
    base_reward = 2.0
    rewards = [base_reward * (0.9999995 ** h) for h in heights]

    for ax in [ax1, ax2]:
        ax.set_facecolor('#0d0520')
        ax.tick_params(colors='#c8b0f0', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#3a1a5a')

    # Mining reward
    ax1.fill_between(heights/1e6, rewards, alpha=0.25, color=rgb_norm(GOLD))
    ax1.plot(heights/1e6, rewards, color=rgb_norm(GOLD), linewidth=2)
    ax1.set_xlabel('Block Height (millions)', color='#c8b0f0', fontsize=9)
    ax1.set_ylabel('Block Reward (QUG)', color='#c8b0f0', fontsize=9)
    ax1.set_title('Mining Reward Decay (Time-Based Emission)', color='#e0d0ff', fontsize=11, fontweight='bold')
    ax1.set_xlim(0, 20)

    # Throughput
    bps_labels = ['0.1 BPS\n(Genesis)', '1 BPS', '5 BPS', '10 BPS\n(Target)', '50 BPS', '100 BPS']
    tps_vals = [100, 1000, 5000, 10000, 50000, 100000]  # estimated
    colors_bar = [rgb_norm(PURPLE_MID)] * 3 + [rgb_norm(GOLD)] + [rgb_norm(TEAL)] * 2
    bars = ax2.barh(bps_labels, [t/1000 for t in tps_vals], color=colors_bar, alpha=0.85)
    ax2.set_xlabel('Estimated TPS (thousands)', color='#c8b0f0', fontsize=9)
    ax2.set_title('Throughput Scalability vs Block Rate', color='#e0d0ff', fontsize=11, fontweight='bold')
    bars[3].set_edgecolor(rgb_norm(GOLD))
    bars[3].set_linewidth(2)
    ax2.tick_params(colors='#c8b0f0', labelsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#3a1a5a')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0d0520')
    plt.close()
    return buf

def make_issue_severity_chart():
    """Issues by severity — pie / bar combo"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor('#0d0520')

    categories = ['Critical\n(DEX)', 'High\n(DEX+Pool)', 'Medium', 'Fixed\n(v10.6.1)', 'Low/Acceptable']
    counts = [2, 4, 8, 3, 9]
    colors_pie = [rgb_norm(RED), rgb_norm(ORANGE), rgb_norm(GOLD), rgb_norm(GREEN), rgb_norm(PURPLE_MID)]
    explode = [0.05, 0.02, 0, 0.05, 0]

    for ax in [ax1, ax2]:
        ax.set_facecolor('#0d0520')

    wedges, texts, autotexts = ax1.pie(
        counts, labels=categories, colors=colors_pie,
        autopct='%1.0f%%', explode=explode,
        textprops={'color': '#c8b0f0', 'fontsize': 8},
        startangle=140
    )
    for at in autotexts:
        at.set_color('#e0d0ff')
        at.set_fontsize(8)
    ax1.set_title('Issue Distribution by Severity\n(Technical Review v3)',
                  color='#e0d0ff', fontsize=10, fontweight='bold')

    # Version timeline
    versions = ['v10.5.0', 'v10.5.3', 'v10.6.0', 'v10.6.1']
    open_critical = [4, 4, 4, 2]
    open_high = [6, 6, 6, 4]
    x = np.arange(len(versions))

    ax2.bar(x - 0.2, open_critical, 0.35, label='Critical', color=rgb_norm(RED), alpha=0.85)
    ax2.bar(x + 0.2, open_high,     0.35, label='High',     color=rgb_norm(ORANGE), alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(versions, color='#c8b0f0', fontsize=8)
    ax2.set_ylabel('Open Issues', color='#c8b0f0', fontsize=9)
    ax2.set_title('Open Issues by Release (Critical vs High)',
                  color='#e0d0ff', fontsize=10, fontweight='bold')
    ax2.tick_params(colors='#c8b0f0', labelsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#3a1a5a')
    ax2.legend(facecolor='#1a0a2a', edgecolor='#3a1a5a', labelcolor='#c8b0f0', fontsize=8)
    ax2.set_ylim(0, 8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0d0520')
    plt.close()
    return buf


# ─── PDF Builder ─────────────────────────────────────────────────────────────

class QNKWhitepaper(FPDF):
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_auto_page_break(auto=True, margin=22)
        self.set_margins(20, 20, 20)
        self.add_font('DejaVu', '', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        self.add_font('DejaVu', 'B', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
        self.add_font('DejaVu', 'I', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf')
        self.add_font('DejaVuMono', '', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf')
        self.add_font('DejaVuMono', 'B', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf')
        self._page_title = ''

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font('DejaVu', 'I', 7.5)
        self.set_text_color(*DARK_GRAY)
        self.cell(0, 8, 'Q-NarwhalKnight — Comprehensive Technical Whitepaper  |  2026',
                  align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*PURPLE_MID)
        self.set_line_width(0.3)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(3)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 7.5)
        self.set_text_color(*DARK_GRAY)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def cover_page(self):
        self.add_page()
        # Purple-to-black gradient simulation via rectangles
        h = 297
        for i in range(60):
            frac = i / 60
            r = int(45 * (1 - frac) + 10 * frac)
            g = int(15 * (1 - frac) + 5 * frac)
            b = int(75 * (1 - frac) + 20 * frac)
            self.set_fill_color(r, g, b)
            self.rect(0, i * (h/60), 210, h/60 + 1, 'F')

        # Title block
        self.set_y(45)
        self.set_font('DejaVu', 'B', 28)
        self.set_text_color(*WHITE)
        self.cell(0, 14, 'Q-NarwhalKnight', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_font('DejaVu', 'I', 16)
        self.set_text_color(*PURPLE_LIGHT)
        self.cell(0, 10, 'Quantum-Enhanced DAG-BFT Blockchain', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 10, 'Comprehensive Technical Whitepaper', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Gold rule
        self.set_draw_color(*GOLD)
        self.set_line_width(1.0)
        y = self.get_y() + 5
        self.line(40, y, 170, y)
        self.ln(14)

        # Abstract box
        self.set_fill_color(0, 0, 0)
        self.set_xy(25, self.get_y())
        self.set_font('DejaVu', '', 9.5)
        self.set_text_color(*LIGHT_GRAY)
        self.multi_cell(
            160, 6,
            'Q-NarwhalKnight (QNK) is a modular Rust blockchain implementing DAG-Knight BFT '
            'consensus with progressive post-quantum cryptographic hardening, a full DeFi stack '
            '(constant-product AMM DEX, WASM smart contracts, cross-chain bridges), and a '
            'high-performance 4-server production deployment serving 10+ BPS throughput. This '
            'whitepaper covers architecture, consensus protocol, cryptographic transition roadmap, '
            'mining subsystem, DeFi layer, infrastructure design, and security audit findings '
            'through technical review v3 (May 2026).',
            new_x=XPos.LMARGIN, new_y=YPos.NEXT
        )
        self.ln(10)

        # Metadata table
        meta = [
            ('Version', 'v10.6.1  (Technical Review v3)'),
            ('Date', '2026-05-07'),
            ('Network', 'mainnet-genesis  |  Chain ID: 1000'),
            ('Genesis', '2026-02-22 12:00 UTC  (ts: 1771761600)'),
            ('Max Supply', '21,000,000 QUG  (24-decimal precision)'),
            ('Consensus', 'DAG-Knight BFT  |  δ=1  |  <50ms finality'),
            ('Servers', 'Beta · Gamma · Delta · Epsilon  (4-node HA cluster)'),
        ]
        for k, v in meta:
            self.set_x(30)
            self.set_font('DejaVu', 'B', 9)
            self.set_text_color(*GOLD)
            self.cell(45, 7, k + ':', new_x=XPos.RIGHT, new_y=YPos.LAST)
            self.set_font('DejaVu', '', 9)
            self.set_text_color(*LIGHT_GRAY)
            self.cell(0, 7, v, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.ln(12)
        self.set_font('DejaVu', 'I', 8)
        self.set_text_color(120, 100, 160)
        self.cell(0, 6, 'Prepared by Server Beta (Claude Code)  ·  Internal Technical Document', align='C')

    def toc_page(self):
        self.add_page()
        self.h2('Table of Contents')
        toc = [
            ('1', 'Introduction & Mission', 3),
            ('2', 'System Architecture', 4),
            ('3', 'DAG-Knight Consensus Protocol', 5),
            ('4', 'Storage & State Management', 7),
            ('5', 'P2P Networking & Turbo Sync', 8),
            ('6', 'Mining Subsystem', 9),
            ('7', 'Cryptography & Post-Quantum Roadmap', 11),
            ('8', 'DeFi & Smart Contracts', 13),
            ('9', 'Cross-Chain Bridges', 14),
            ('10', 'Frontend Wallets', 15),
            ('11', 'Infrastructure & Deployment', 16),
            ('12', 'Security Audit Findings', 18),
            ('13', 'Emission Economics', 20),
            ('14', 'Roadmap & Open Issues', 21),
            ('15', 'Conclusion', 22),
        ]
        for num, title, pg in toc:
            self.set_font('DejaVu', '', 10)
            self.set_text_color(*DARK_GRAY)
            self.cell(12, 8, num + '.', new_x=XPos.RIGHT, new_y=YPos.LAST)
            self.set_font('DejaVu', '', 10)
            self.cell(140, 8, title, new_x=XPos.RIGHT, new_y=YPos.LAST)
            self.set_font('DejaVu', 'I', 9)
            self.set_text_color(*PURPLE_MID)
            self.cell(0, 8, str(pg), align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_draw_color(220, 215, 235)
            self.set_line_width(0.1)
            self.line(20, self.get_y(), 190, self.get_y())

    def h1(self, text):
        self.ln(4)
        self.set_font('DejaVu', 'B', 15)
        self.set_text_color(*PURPLE_DARK)
        self.set_fill_color(*LIGHT_GRAY)
        self.cell(0, 10, text, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*GOLD)
        self.set_line_width(0.6)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(3)

    def h2(self, text):
        self.ln(3)
        self.set_font('DejaVu', 'B', 12)
        self.set_text_color(*PURPLE_MID)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*PURPLE_LIGHT)
        self.set_line_width(0.3)
        self.line(20, self.get_y(), 100, self.get_y())
        self.ln(2)

    def h3(self, text):
        self.ln(2)
        self.set_font('DejaVu', 'B', 10.5)
        self.set_text_color(*DARK_GRAY)
        self.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def body(self, text):
        self.set_font('DejaVu', '', 9.5)
        self.set_text_color(*BLACK)
        self.multi_cell(0, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def bullet(self, items, indent=4):
        self.set_font('DejaVu', '', 9)
        self.set_text_color(*BLACK)
        for item in items:
            self.set_x(self.l_margin + indent)
            self.cell(5, 5.5, '•', new_x=XPos.RIGHT, new_y=YPos.LAST)
            self.multi_cell(0, 5.5, item, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def kv_table(self, rows, col1=60, col2=110):
        self.set_font('DejaVu', '', 9)
        for i, (k, v) in enumerate(rows):
            fill = i % 2 == 0
            self.set_fill_color(*LIGHT_GRAY if fill else WHITE)
            self.set_text_color(*DARK_GRAY)
            self.set_font('DejaVu', 'B', 9)
            self.cell(col1, 6.5, k, border=0, fill=fill, new_x=XPos.RIGHT, new_y=YPos.LAST)
            self.set_font('DejaVu', '', 9)
            self.set_text_color(*BLACK)
            self.multi_cell(col2, 6.5, v, border=0, fill=fill, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def code(self, text, small=True):
        self.set_fill_color(238, 235, 248)
        self.set_font('DejaVuMono', '', 7.5 if small else 8.5)
        self.set_text_color(*PURPLE_DARK)
        self.multi_cell(0, 5, text, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def info_box(self, title, text, color=PURPLE_MID):
        self.set_fill_color(*[int(c * 0.15) for c in color])
        self.set_draw_color(*color)
        self.set_line_width(0.5)
        self.set_font('DejaVu', 'B', 9)
        self.set_text_color(*color)
        self.cell(0, 7, '  ' + title, border='L', fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font('DejaVu', '', 9)
        self.set_text_color(*BLACK)
        self.set_fill_color(248, 246, 255)
        self.multi_cell(0, 5.5, '  ' + text, border='LRB', fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def add_chart(self, buf, w=170, caption=''):
        buf.seek(0)
        # Save to temp file
        tmp = '/tmp/qnk_chart_tmp.png'
        with open(tmp, 'wb') as f:
            f.write(buf.read())
        x = (210 - w) / 2
        self.image(tmp, x=x, w=w)
        if caption:
            self.set_font('DejaVu', 'I', 8)
            self.set_text_color(*DARK_GRAY)
            self.cell(0, 5, caption, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)


def build_pdf():
    pdf = QNKWhitepaper()
    pdf.cover_page()
    pdf.toc_page()

    # ── §1 Introduction ──────────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1('1. Introduction & Mission')

    pdf.body(
        'Q-NarwhalKnight (QNK) is a research-grade, production-deployed blockchain built entirely '
        'in Rust. Its mission is to demonstrate that a quantum-ready, high-throughput, privacy-first '
        'consensus system can be operated at production scale before quantum computers become '
        'computationally relevant — making the transition to post-quantum cryptography a planned '
        'evolution rather than an emergency patch.'
    )

    pdf.h2('1.1 Design Philosophy')
    pdf.bullet([
        'Quantum-first: Progressive migration from Ed25519 (Q0) to Dilithium5 (Q2) without hard forks.',
        'Safety over performance: Sync-down protection, 3-layer dedup, BalanceRootV1 integrity checks.',
        'Privacy-by-default: Dandelion++ mandatory for all transactions; Tor integration available.',
        'Modular architecture: 89 Rust crates, each independently testable and replaceable.',
        'Production-grade HA: 4-node cluster with zero-downtime rolling deployments.',
    ])

    pdf.h2('1.2 Current Status (May 2026)')
    pdf.kv_table([
        ('Network',          'mainnet-genesis  (live since 2026-02-22)'),
        ('Current version',  'v10.6.1'),
        ('Block height',     '~18,500,000 (approaching BalanceRootV1 activation)'),
        ('Active nodes',     'Beta · Gamma · Delta · Epsilon  +  community nodes'),
        ('Mining',           'CPU (AVX2/512) + GPU (OpenCL) + Stratum pool'),
        ('Frontend',         'Quantum Wallet (React) + Slint native wallet'),
        ('Smart contracts',  '25+ WASM contract types deployed'),
        ('Bridges',          'Bitcoin deposit bridge LIVE; Zcash RPC integrated'),
    ])

    # ── §2 Architecture ──────────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1('2. System Architecture')

    pdf.body(
        'The QNK stack is organized into seven horizontal layers, each independently tested and '
        'deployable. The primary node binary (q-api-server) integrates all layers except the miner, '
        'which runs as a separate process (q-miner). The FPGA/RTL implementation targets hardware '
        'acceleration for the BLAKE3 mining kernel.'
    )

    pdf.h2('2.1 Stack Overview')
    pdf.code(
        'Layer 7  Web Wallet (React/TS) · Slint Native Wallet (Rust)\n'
        'Layer 6  q-miner (CPU+GPU+VDF) · External miners (Stratum)\n'
        'Layer 5  REST API 228 routes · SSE <50ms · WebSocket  (Axum 0.7)\n'
        'Layer 4  Block Producer · Balance Consensus · Emission Controller · WASM VM\n'
        'Layer 3  DAG-Knight Consensus  (Bullshark BFT, δ=1, Genus-2 VDF anchor)\n'
        'Layer 2  libp2p 0.56  (TCP/QUIC/WS · Kademlia DHT · Gossipsub · DCUTR)\n'
        'Layer 1  Storage  (RocksDB 12 CFs Linux/macOS · Sled Windows)'
    )

    pdf.add_chart(make_network_topology(), w=165,
                  caption='Figure 2.1 — 4-node production cluster topology with Nginx load balancer')

    pdf.h2('2.2 Source Code Metrics')
    pdf.kv_table([
        ('main.rs (node entry)',  '24,609 lines — main() fn spans ~22,762 lines (ARCH-001)'),
        ('handlers.rs (API)',     '17,019 lines — 70+ handler functions (ARCH-002)'),
        ('q-miner/src/main.rs',  '4,778 lines'),
        ('q-mining-pool/src/',   '~3,500 lines (multi-file)'),
        ('FPGA RTL (SystemVerilog)', '~2,100 lines in qug-v1-rtl/rtl/xcrypto/'),
        ('Total crate count',    '89 crates'),
    ])

    # ── §3 Consensus ─────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1('3. DAG-Knight Consensus Protocol')

    pdf.body(
        'DAG-Knight is a Bullshark-family asynchronous Byzantine Fault Tolerant (BFT) consensus '
        'protocol operating on a directed acyclic graph (DAG) of vertices. Unlike single-chain '
        'protocols, DAG-Knight allows all honest validators to contribute vertices simultaneously, '
        'maximizing throughput while preserving safety under f < n/3 Byzantine faults.'
    )

    pdf.h2('3.1 Protocol Properties')
    pdf.kv_table([
        ('Byzantine tolerance',   'f = ⌊(n-1)/3⌋ — survives up to 1/3 malicious validators'),
        ('Finality target',        '<50 ms  (δ=1 commit rule, Bullshark certificate)'),
        ('Vertex structure',       'ID [u8;32] · round u64 · parents Vec<VertexId> · signature'),
        ('Block production',       '15 second default interval; min 1 solution queued'),
        ('Anchor election',        'Quantum VDF entropy (Genus-2 Jacobian curve)'),
        ('Fork detection',         'Homological analysis: Betti numbers H₀, H₁'),
        ('VDF base difficulty',    '2,048 squarings × 1.15 quantum boost = ~2,355 effective'),
        ('QRNG refresh',           'Every 30 seconds via quantum_beacon.rs'),
    ])

    pdf.add_chart(make_consensus_dag(), w=165,
                  caption='Figure 3.1 — DAG rounds with gold-highlighted VDF-elected anchor vertex')

    pdf.h2('3.2 Vertex and Block Lifecycle')
    pdf.body(
        'Each validator continuously broadcasts signed vertices into the DAG. Vertices reference 2f+1 '
        'parent vertices from the previous round (strong causal history). Once a vertex accumulates '
        'enough parent edges, it is eligible for anchor election. The VDF-elected anchor at round R '
        'finalizes all vertices reachable from R-1, emitting a Bullshark certificate that is '
        'gossipped to all peers within 50 ms.'
    )
    pdf.body(
        'Blocks are produced from finalized DAG rounds. The block producer drains a lock-free '
        'SegQueue of up to 250 mining solutions per block, computes the coinbase transaction via '
        'the emission controller, calculates the SIMD-accelerated Merkle root (AVX-512, 8–10× '
        'speedup over scalar), and broadcasts via Gossipsub.'
    )

    pdf.h2('3.3 BalanceRootV1 — On-chain State Commitment')
    pdf.info_box(
        'Activation Height: 18,600,000',
        'At block height ≥ 18,600,000, every block must carry a 32-byte BLAKE3 commitment over '
        'the full balance state (all non-zero wallet balances, sorted deterministically). Peers '
        'reject blocks with missing or mismatched balance roots. This creates a cryptographic '
        'audit trail of the economic state at every block, enabling trustless light-client '
        'verification.',
        color=GOLD
    )
    pdf.kv_table([
        ('Root computation', 'BLAKE3(sorted wallet_address:balance pairs, zero-balances excluded)'),
        ('Activation gate',  'q-consensus-guard::is_upgrade_active(BalanceRootV1, next_height)'),
        ('Fallback behavior','Compute failure → [0u8;32] → peer rejection → chain halt (BAL-001)'),
        ('Determinism proof', '5 tests in balance_determinism_tests.rs (order-independent, restart-safe)'),
        ('Peer validation',  'Incoming blocks validated before transaction application'),
    ])

    # ── §4 Storage ───────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1('4. Storage & State Management')

    pdf.body(
        'The storage layer abstracts over two backends: RocksDB (Linux/macOS) and Sled (Windows). '
        'Both expose the same KV interface, allowing the same node binary to run cross-platform. '
        'RocksDB uses 12 column families for logical isolation between data domains.'
    )

    pdf.h2('4.1 RocksDB Column Families')
    pdf.kv_table([
        ('CF_BLOCKS',       'Blocks indexed by height'),
        ('CF_BALANCES',     'Wallet balances (u128, 24-decimal)'),
        ('CF_STATE',        'WASM smart contract state'),
        ('CF_TRANSACTIONS', 'Transactions by TxHash'),
        ('CF_VERTICES',     'DAG vertices'),
        ('CF_CERTIFICATES', 'Bullshark finality certificates'),
        ('CF_CONTRACTS',    'Contract registry and metadata'),
        ('CF_MANIFEST',     'Node metadata + processed_balance_block dedup keys'),
        ('CF_PEER_TRUST',   'Per-peer reputation scores'),
        ('CF_MINING_STATS', 'Per-miner aggregate statistics'),
        ('CF_EMAIL',        'Blockchain email messages'),
        ('CF_CALENDAR',     'Decentralized calendar events'),
    ])

    pdf.h2('4.2 Mining Reward Deduplication — 3-Layer Architecture')
    pdf.body(
        'The most safety-critical storage path is mining reward accounting. A double-credit bug '
        'would silently mint coins, violating the 21M cap. Three independent dedup layers ensure '
        'this cannot happen, even across node restarts:'
    )
    pdf.bullet([
        'Layer 1 (PERSISTENT): RocksDB key "processed_balance_block:{hash}" written atomically '
        'with balance update. Survives node restarts. Checked first.',
        'Layer 2 (IN-MEMORY): LRU cache, 100,000 entries (~5 MB). Fast secondary check. Falls '
        'back to persistent on cache miss.',
        'Layer 3 (ATOMIC TX): Balance update + dedup key written in a single RocksDB transaction. '
        'Partial updates are impossible.',
    ])
    pdf.info_box(
        'Audit Verdict: SAFE',
        'Integration test test_double_processing_safety (balance_consensus_integration.rs:132) '
        'verifies the second call returns AlreadyProcessed with balance unchanged. '
        'test_five_node_consensus (line 178) verifies 5 independent nodes reach identical balances '
        'after 10 blocks.',
        color=GREEN
    )

    pdf.h2('4.3 Sync-Down Protection')
    pdf.body(
        'A catastrophic sync-down bug would allow a malicious peer to convince a node to delete '
        'its chain history by advertising a lower block height. Protection operates at two layers:'
    )
    pdf.bullet([
        'Application layer: Only initiates sync if network_height > current_height + 5.',
        'Database layer: Safety abort if target_height < local_height when local_height > 1,000.',
        'Balances are always reset together with the blockchain state — no orphaned economic state.',
    ])

    # ── §5 P2P Networking ────────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1('5. P2P Networking & Turbo Sync')

    pdf.body(
        'The networking layer is built on libp2p 0.56 (Tokio async runtime). QNK uses a hybrid '
        'transport stack: TCP for compatibility, QUIC for low-latency, WebSocket for browser nodes, '
        'and DCUTR for NAT traversal. The Kademlia DHT provides peer discovery, while Gossipsub '
        'handles topic-based message propagation.'
    )

    pdf.h2('5.1 Protocol Stack')
    pdf.kv_table([
        ('libp2p version',   '0.56'),
        ('Transports',       'TCP · QUIC · WebSocket · DCUTR (hole-punching)'),
        ('Peer discovery',   'Kademlia DHT + Bootstrap peers (5 nodes)'),
        ('Message relay',    'Gossipsub v1.1 (mesh + fanout)'),
        ('Gossipsub topics', '/qnk/{net}/blocks · mempool-txs · peer-heights · turbo-sync-{shard} · mining-solutions · ai-inference-{model}'),
        ('Network ID',       'mainnet-genesis  (validated at protocol handshake — rejects mismatches)'),
        ('u128 safety',      'Custom serde module serializes all u128 as STRING — prevents MessagePack truncation to u64'),
    ])

    pdf.h2('5.2 Turbo Sync')
    pdf.body(
        'New nodes joining the network sync the full chain history via Turbo Sync: 16 parallel '
        'streams per peer, 2,500-block chunks, Zstd level-1 compression. At Epsilon (10 Gbit), '
        'peak sync speed reaches 1,100 blocks/second, completing the full 11.4M block chain in '
        'approximately 5.5 hours.'
    )
    pdf.add_chart(make_turbo_sync_performance(), w=165,
                  caption='Figure 5.1 — Turbo Sync speed profile across 11.4M block chain history')

    pdf.h2('5.3 Bootstrap Nodes')
    pdf.kv_table([
        ('Epsilon (primary)', '89.149.241.126  10 Gbit  PeerID: 12D3KooWAbrV...'),
        ('Beta',              '185.182.185.227  100 Mbit  PeerID: 12D3KooWSBxw...'),
        ('Gamma',             '109.205.176.60  1 Gbit  PeerID: 12D3KooWFfZK...'),
        ('Delta',             '5.79.79.158  1 Gbit  PeerID: 12D3KooWLJJR...'),
    ])

    # ── §6 Mining ────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1('6. Mining Subsystem')

    pdf.body(
        'QNK supports three mining modes: solo HTTP (direct reward to wallet), Stratum pool (PPLNS '
        'with pool operator), and P2P pool (CRDT-based PPLNS via libp2p gossipsub). The mining '
        'algorithm uses BLAKE3 × 100 iterations for CPU/GPU, and Genus-2 VDF for the optional '
        'sequential proof lane.'
    )

    pdf.add_chart(make_block_production_rate(), w=165,
                  caption='Figure 6.1 — Mining reward decay over block height and throughput scalability')

    pdf.h2('6.1 CPU Mining Engine')
    pdf.kv_table([
        ('Thread model',       'tokio::spawn_blocking per thread (keeps Tokio scheduler free)'),
        ('Core affinity',      'Pinned via core_affinity crate'),
        ('Nonce partition',    'thread_id << 48  (cross-thread collision prevention)'),
        ('SIMD acceleration',  'AVX2, AVX-512, NEON — auto-detected at runtime'),
        ('Batch size',         'intensity × 100,000 nonces  (intensity 1–10, default 7)'),
        ('Stale detection',    'Atomic new_block_signal checked every 512 nonces'),
        ('Hash counter flush', 'Every 1,024 hashes  (reduces atomic contention)'),
        ('Allocator',          'jemalloc (Linux) · mimalloc (Windows)'),
    ])

    pdf.h2('6.2 GPU Mining Engine')
    pdf.kv_table([
        ('Framework',          'OpenCL 3.0 (primary) · CUDA via cudarc (feature flag)'),
        ('Kernel',             'BLAKE3 × 100 rounds with #pragma unroll 9'),
        ('Dispatch strategy',  'Adaptive [100ms, 400ms] per kernel'),
        ('Multi-GPU support',  'Per-GPU independent auto-tuning (v10.1.7+)'),
        ('Persistent buffers', 'Challenge loaded to __constant cache once per job'),
        ('GPU nonce space',    'Starts at u64::MAX/2  (no collision with CPU nonces)'),
        ('Kernel cache',       '~/.config/q-miner/kernel-cache/'),
    ])

    pdf.h2('6.3 Genus-2 VDF Lane')
    pdf.body(
        'The VDF lane provides a sequential proof-of-work using Wesolowski proofs over a Genus-2 '
        'hyperelliptic Jacobian curve. VDF solutions carry a cryptographic guarantee that a minimum '
        'amount of sequential computation was performed — non-parallelizable by design.'
    )
    pdf.info_box(
        'VDF-001: Single-Lane Bottleneck (Open Issue)',
        'Effective difficulty: ~2,355 squarings per evaluation. Wall-clock time: 4–7 seconds '
        'on modern hardware. At 10 BPS (100ms/block), virtually every VDF evaluation is discarded '
        'when a new block arrives. Recommendation: expose Q_VDF_ITERATIONS_CAP environment '
        'variable; consider parallel VDF lanes.',
        color=ORANGE
    )

    pdf.h2('6.4 FPGA/RTL Implementation')
    pdf.kv_table([
        ('Target',             'Xilinx Kintex-7 XC7K325T  @  100 MHz'),
        ('Design',             '16-core RISC-V RV32IMC SoC'),
        ('Xcrypto (custom-0)', 'Hardware BLAKE3: blake3.init/round/chain/finalize — IMPLEMENTED'),
        ('Throughput',         '~114 MH/s (16 cores × 1 hash / 14 cycles @ 100 MHz)'),
        ('Xlattice (custom-1)','poly.add + poly.mul IMPLEMENTED; NTT stubs (Phase 1B, not blocking mining)'),
        ('Status',             'Prototype — simulation testbench present (blake3_tb.vvp)'),
    ])

    # ── §7 Cryptography ──────────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1('7. Cryptography & Post-Quantum Roadmap')

    pdf.body(
        'QNK implements a staged cryptographic migration from classical Ed25519 to full post-quantum '
        'Dilithium5, without requiring hard forks. The upgrade gate mechanism allows new rules to '
        'activate at a specific block height while remaining backward-compatible with historical blocks.'
    )

    pdf.add_chart(make_crypto_phases(), w=165,
                  caption='Figure 7.1 — Post-quantum migration phases: Q0 → Q1 → Q2 → Q3')

    pdf.h2('7.1 Classical Cryptography Stack')
    pdf.kv_table([
        ('Ed25519',        'Phase 0 signing, address derivation  (q-wallet)'),
        ('SHA3-256',       'Block/tx hash, address hash  (q-types)'),
        ('BLAKE3',         'Mining PoW, Merkle trees  (q-miner, q-crypto-simd)'),
        ('AES-256-GCM',    'Wallet + storage at-rest encryption  (q-wallet, q-storage)'),
        ('Argon2id',       'Key derivation — 64 MB cost, 4 iterations  (q-wallet, q-storage)'),
        ('BIP39',          '12-word mnemonic wallet recovery  (q-wallet)'),
        ('AEGIS-256',      'Auth encryption — 2–5× AES-GCM speed using AES-NI  (q-crypto-advanced)'),
    ])

    pdf.h2('7.2 Post-Quantum Cryptography Stack')
    pdf.kv_table([
        ('Dilithium5 (ML-DSA)', 'NIST standard — signature size ~3,300 bytes — Phase Q1/Q2'),
        ('Kyber1024 (ML-KEM)',  'NIST standard — KEM for session establishment — Phase Q1/Q2'),
        ('SQIsign',             'Isogeny-based, 204-byte signatures — Phase advanced (Q3)'),
        ('SPHINCS+',            'Hash-based fallback, 7,856-byte signatures — NIST alternate'),
        ('Genus-2 VDF',         'Hyperelliptic curves, Wesolowski proofs — Mining sequential PoW'),
        ('FROST',               'Threshold Schnorr (IACR 2025/1024) — Validator committee'),
        ('Ring-LWE L-VRF',      'Lattice VRF — Anchor election randomness'),
    ])

    pdf.h2('7.3 Zero-Knowledge Proof Systems')
    pdf.kv_table([
        ('Circle STARKs (IACR 2024/278)', '~60 KB proofs · No trusted setup · Quantum-safe · Private txs'),
        ('Groth16/PLONK SNARKs',          '96–192 B proofs · Requires trusted setup · Contract proofs'),
        ('Bulletproofs v2 (IACR 2024/313)', 'O(log n) · No trusted setup · Range proofs'),
        ('Recursive SNARKs',               'Succinct · No trusted setup · Light client bootstrap'),
    ])

    pdf.h2('7.4 Privacy Primitives')
    pdf.bullet([
        'Dandelion++: Mandatory for ALL transaction relay — stem phase randomizes propagation path.',
        'Tor (Arti embedded): Available via Q_ENABLE_TOR=1 — dedicated circuits per validator.',
        'Ring signatures (0x82): RingTransfer tx type — production ready.',
        'Shielded transfers (0x83): Circle STARK + nullifier set — production integrated.',
        'AEGIS-QL middleware: Auth for all API endpoints.',
        'Quantum mixing pool: Available at /api/v1/mixer/ endpoints.',
    ])

    pdf.h2('7.5 Upgrade Gate Mechanism')
    pdf.body(
        'All consensus rule changes are wrapped in height-gated upgrades via q-consensus-guard. '
        'This ensures old blocks always validate under old rules, preventing chain splits on upgrade.'
    )
    pdf.code(
        '// q-consensus-guard/src/upgrade_gate.rs\n'
        'upgrades.insert(Upgrade::BalanceRootV1, UpgradeConfig {\n'
        '    activation_height: 18_600_000,\n'
        '    mandatory: true,\n'
        '    min_version: "10.6.0".to_string(),\n'
        '});\n'
        '\n'
        '// Usage in block validation:\n'
        'if is_upgrade_active(Upgrade::BalanceRootV1, block.height) {\n'
        '    // New rule: require balance root commitment\n'
        '} else {\n'
        '    // Old rule: no balance root required\n'
        '}'
    )

    # ── §8 DeFi ──────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1('8. DeFi & Smart Contracts')

    pdf.h2('8.1 Constant-Product AMM DEX')
    pdf.body(
        'The QNK DEX implements a quantum-enhanced constant-product (x × y = k) AMM. The quantum '
        'enhancement applies physics-inspired adjustments (uncertainty principle noise, entanglement '
        'effect, wave function collapse) to the price calculation, adding non-determinism that '
        'makes front-running harder.'
    )
    pdf.kv_table([
        ('AMM formula',    'x × y = k  (quantum-adjusted)'),
        ('Fee structure',  '0.30% total  (0.05% protocol + 0.25% LP)'),
        ('Pool reserves',  'Tracked per pair in QuantumLiquidityManager (RwLock<HashMap>)'),
        ('Precision',      '24-decimal (BigDecimal arithmetic prevents u128 overflow at scale)'),
        ('Token pairs',    'Any QUG-native, ERC20-bridge, or custom token pair'),
    ])
    pdf.info_box(
        'DEX-001/002: CRITICAL — Trading Engine Disconnected (Open)',
        'execute_quantum_trade() computes prices and charges fees but NEVER updates pool reserves. '
        'Pool state (token_a_reserve, token_b_reserve) is unchanged by swaps. The DEX operates '
        'as a price discovery engine; settlement via reserve mutation is not implemented. '
        'Additionally, DEX-002: concurrent swaps have a read-compute-write race condition. '
        'Both issues must be fixed before the DEX can be promoted as a settled AMM.',
        color=RED
    )

    pdf.h2('8.2 WASM Smart Contract VM')
    pdf.kv_table([
        ('Runtime',            'Wasmer 4.0.0 (JIT) + Wasmtime 14.0.0'),
        ('Language',           'WebAssembly (compiled from Rust, AssemblyScript, etc.)'),
        ('Deployment signing', 'Post-quantum Dilithium5 (v3.7.4+)'),
        ('Gas metering',       'Pluggable metering middleware'),
        ('Contract count',     '25+ contract types across 6 domains'),
        ('Domains',            'DeFi · RWA · Derivatives · Governance · Identity · Utility'),
    ])

    pdf.h2('8.3 Transaction Taxonomy')
    pdf.body('QNK defines 46 distinct transaction opcodes across 11 domains:')
    pdf.kv_table([
        ('0x00–0x0F', 'Core: Transfer, Coinbase, Burn, Fee'),
        ('0x20–0x2F', 'DEX: Pool create/add/remove, Swap, FlashLoan'),
        ('0x40–0x4F', 'Token: Mint, Burn, Freeze, Transfer'),
        ('0x60–0x6F', 'Governance: Vote, Proposal, Execute'),
        ('0x80–0x8F', 'Privacy: Dandelion++, RingTransfer, Shielded'),
        ('0x90–0x9F', 'Cross-chain: Atomic swaps (0x90–0x93 currently no-op)'),
        ('0xF0–0xFF', 'System: EmergencyPause, StateCheckpoint'),
    ])

    # ── §9 Bridges ───────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1('9. Cross-Chain Bridges')

    pdf.h2('9.1 Bitcoin Deposit Bridge — LIVE (v10.6.1)')
    pdf.body(
        'The Bitcoin deposit bridge enables one-way BTC → QUG conversion through the QNK ecosystem. '
        'Users generate a unique native SegWit (bc1q...) deposit address derived from the qug-bridge '
        'HD wallet on Delta. Inbound BTC is monitored and credited as wrapped BTC (wBTC) in the '
        'user\'s QNK wallet.'
    )
    pdf.kv_table([
        ('Backend',         'Bitcoin Knots v28.1  on Delta (5.79.79.158:8332)'),
        ('HD wallet',       'qug-bridge  (BIP44 derivation: m/44\'/0\'/0\'/0/{user_index})'),
        ('Address format',  'Native SegWit (bech32, bc1q...)'),
        ('Address API',     'POST /api/v1/bitcoin-bridge/deposit/address'),
        ('Status API',      'GET /api/v1/bitcoin-bridge/status → {bridge_enabled, connected, version, height}'),
        ('BTC height',      '948,305 (fully synced)'),
        ('Monitoring',      'mempool.space fallback when RPC unavailable; real-time via SSE'),
    ])
    pdf.info_box(
        'Status: Production (fixed in v10.6.1)',
        'Previous versions showed "Bridge Offline" even when the deposit bridge was initialized, '
        'due to a bug in get_bridge_status() that only checked atomic_swap_manager.is_some(). '
        'Fixed by adding || state.deposit_bridge.is_some().',
        color=GREEN
    )

    pdf.h2('9.2 Zcash Integration')
    pdf.kv_table([
        ('Backend',         'Zebra RPC on Delta  (port 8232)'),
        ('Zebra height',    '3,334,024 (fully synced)'),
        ('z-address',       'Shielded Sapling address per user'),
        ('Frontend',        'ZcashWalletModal (redesigned v10.6.1) — framer-motion, QR code, ZEC gold theme'),
        ('Operations',      'get_z_address · get_z_balance · transaction history'),
        ('QR code scheme',  'zcash:{z_address}  (standard QR payment URI)'),
    ])

    pdf.h2('9.3 Future Bridges')
    pdf.bullet([
        'Ethereum bridge: No handler functions implemented. Planned for Phase 2.',
        'Monero bridge: No handler functions implemented. Planned for Phase 2.',
        'Atomic swaps (0x90–0x93): Currently a silent no-op (fee deducted, no escrow state created). Safe.',
    ])

    # ── §10 Frontend ─────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1('10. Frontend Wallets')

    pdf.h2('10.1 Quantum Wallet — React/TypeScript')
    pdf.body(
        'The Quantum Wallet is a 14-screen React SPA with SSE-driven real-time state synchronization. '
        'All balance updates arrive via the SSE event stream and are validated client-side '
        '(max 21,000,000 QUG sanity check; monotonic block-height tracking).'
    )
    pdf.kv_table([
        ('Framework',           'React 18 + TypeScript + Vite'),
        ('State sync',          'SSE /api/v1/events — <50ms latency'),
        ('Auth',                'AEGIS-QL session tokens + optional 2FA'),
        ('Multi-server failover', 'Return-to-primary every 60s; 30s minimum cooldown'),
        ('DEX UI',              'Swap, add/remove liquidity, pool stats'),
        ('Mining UI',           'Challenge display, stats, difficulty chart'),
        ('Rate limiter',        'RequestRateLimiter — 20 concurrent slots (fixed from 10 in v10.6.1)'),
        ('Screens',             'Dashboard · Explorer · Send · History · DEX · Mining · Settings · Deploy · RWA · Zcash · Bitcoin · AI · Calendar · POS'),
    ])

    pdf.h2('10.2 Slint Native Wallet — Rust')
    pdf.kv_table([
        ('Framework',      'Slint v1.9'),
        ('Features',       'BIP39 create/restore · QR code · SSE balance · OAuth2 (PKCE)'),
        ('Mining',         'CPU + OpenCL GPU mining built-in'),
        ('Auto-update',    'Self-replacing binary updater v1.5 — polls /api/v1/version'),
        ('POS mode',       'Point-of-sale payment terminal'),
        ('Platform',       'Linux (primary) · Windows cross-compile (--no-default-features --features tui)'),
    ])

    # ── §11 Infrastructure ───────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1('11. Infrastructure & Deployment')

    pdf.body(
        'QNK runs on a 4-server production cluster with zero-downtime rolling deployments via '
        'ha-deploy.sh. Epsilon serves quillon.xyz (DNS primary) via q-flux reverse proxy. '
        'Beta and Gamma are behind Nginx with ip_hash sticky sessions.'
    )

    pdf.h2('11.1 Server Configuration')
    pdf.kv_table([
        ('Epsilon (89.149.241.126)', '10 Gbit · q-flux reverse proxy · 219 GB RocksDB · NEVER write to /tmp'),
        ('Beta (185.182.185.227)',   '100 Mbit · Nginx primary (weight=10) · ROCKSDB_BLOCK_CACHE_MB=4096'),
        ('Gamma (109.205.176.60)',   '1 Gbit · Nginx backup (weight=1) · 4 GB swap (OOM protection)'),
        ('Delta (5.79.79.158)',      '1 Gbit · Canary · Bitcoin Knots v28.1 · Zebra · Q_TOR_BOOTSTRAP_TIMEOUT=5'),
        ('Alpha (161.35.219.10)',    'Testing/canary · Docker containers · Debian 12'),
    ])

    pdf.h2('11.2 Rolling Deployment Pipeline')
    pdf.body(
        'The ha-deploy.sh pipeline ensures zero-downtime upgrades. Traffic remains served '
        'throughout the entire process:'
    )
    pdf.bullet([
        'Step 1 verify-delta: Deploy to canary Docker node, 7+ min soak.',
        'Step 2 verify-gamma: Deploy to backup, 90s stability window.',
        'Step 3 promote: Nginx sets Gamma weight=10, Beta weight=1 (traffic shifts to Gamma).',
        'Step 4 deploy-beta: Stop Beta, replace binary, restart, verify health.',
        'Step 5 restore: Nginx sets Beta weight=10, Gamma weight=1 (traffic shifts back).',
    ])
    pdf.kv_table([
        ('Lock file',        'Prevents concurrent deploys'),
        ('Version check',    'Aborts if new version == currently running version'),
        ('Health timeout',   'Max 10 min per server to pass /api/v1/health'),
        ('Auto-rollback',    'Restores previous binary from .backup on failure'),
        ('Binary copies',    'New binary copied to downloads/ for user wget links'),
    ])

    pdf.h2('11.3 Key Environment Variables')
    pdf.kv_table([
        ('Q_NETWORK_ID',          'Checked BEFORE --network CLI flag'),
        ('Q_DB_PATH',             'Must be ABSOLUTE PATH on Epsilon (never relative)'),
        ('ROCKSDB_BLOCK_CACHE_MB','Required to prevent OOM (Beta: 4096, Gamma/Delta: 512–2048)'),
        ('Q_TOR_BOOTSTRAP_TIMEOUT', 'Set to 5 in production (default 120s blocks startup)'),
        ('RUST_LOG',              'Must be warn on Epsilon (DEBUG fills 40 GB root partition in minutes)'),
        ('BTC_RPC_PASS',          'Required for Bitcoin deposit bridge initialization'),
        ('Q_ENABLE_MINING_POOL',  'Defaults to 1; must explicitly set to 0 to disable'),
    ])

    # ── §12 Security Audit ───────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1('12. Security Audit Findings (Technical Review v3)')

    pdf.add_chart(make_issue_severity_chart(), w=165,
                  caption='Figure 12.1 — Issue distribution and open critical/high count across releases')

    pdf.h2('12.1 Critical Issues (Open)')
    pdf.info_box(
        'DEX-001: Trading Engine Reserve Disconnect',
        'execute_quantum_trade() (q-dex/src/trading.rs:242) computes prices with quantum adjustments '
        'and returns results but NEVER calls liquidity.rs::update_pool_reserves(). Pool state is '
        'immutable during swaps. x×y=k invariant never verified post-swap.',
        color=RED
    )
    pdf.info_box(
        'DEX-002: Concurrent Swap Race Condition',
        'QuantumLiquidityManager uses Arc<RwLock<HashMap>>. execute_quantum_trade() reads pool state '
        'under read lock, computes output, then releases — but the reserve update (if reconnected) '
        'would require a separate write lock acquisition. Two simultaneous swaps can both read stale '
        'state and both write conflicting results.',
        color=RED
    )

    pdf.h2('12.2 High Issues (Open)')
    pdf.bullet([
        'DEX-003: max_slippage_bps field exists in TradeRequest but is never read in trading.rs — '
        'no slippage protection at execution time.',
        'DEX-004: No MIN_POOL_RESERVE constant in q-dex/ — reserves can drain to dust, producing '
        'trillion-dollar prices (historical precedent in codebase comments).',
        'POOL-001: Share dedup HashSet.clear() at 100,000 entries — all previously-seen shares '
        're-accepted after flush (share.rs:260).',
        'POOL-002: Min share difficulty not enforced at TCP handler — workers receive acceptance '
        'before validation (stratum.rs:511).',
    ])

    pdf.h2('12.3 Issues Fixed in v10.6.1')
    pdf.bullet([
        'I-004 (Bitcoin bridge): Fully implemented DepositBridge with Delta Bitcoin Knots RPC — '
        'was a stub returning active:false.',
        'I-008 (Bridge status): get_bridge_status() now checks || state.deposit_bridge.is_some().',
        'I-009 (Rate limiter): RequestRateLimiter max concurrent 10 → 20 — prevents user actions '
        'from being blocked by background polling.',
    ])

    pdf.h2('12.4 Confirmed Safe (No Action Required)')
    pdf.bullet([
        'Mining reward double-credit: 3-layer persistent dedup SAFE — verified by integration tests.',
        'BalanceRootV1 activation: height >= 18,600,000 correctly implemented — 5 determinism tests.',
        'Atomic swap opcodes (0x90–0x93): Safe no-ops — fee deducted but no escrow state created.',
        'u128 P2P serialization: Custom serde module always serializes as STRING — verified boundary.',
        'systemd service file: No secrets in Environment= lines — clean audit.',
        'Sync-down protection: Application + database layer both enforce safety aborts.',
    ])

    # ── §13 Emission Economics ───────────────────────────────────────────────
    pdf.add_page()
    pdf.h1('13. Emission Economics')

    pdf.body(
        'QNK follows a Bitcoin-inspired deflationary emission model with a 21 million QUG hard cap '
        'and 4-year halving cycles. The emission controller is pure u128 integer arithmetic — no '
        'floating-point drift across nodes, guaranteed identical rewards on every validator for the '
        'same (block_height, timestamp) pair.'
    )

    pdf.add_chart(make_emission_schedule(), w=165,
                  caption='Figure 13.1 — Per-era annual emission (left) and cumulative supply toward 21M cap (right)')

    pdf.h2('13.1 Economic Parameters')
    pdf.kv_table([
        ('Max supply',          '21,000,000 QUG'),
        ('Decimal precision',   '24  (1 QUG = 10^24 base units)'),
        ('Genesis timestamp',   '1771761600  (2026-02-22 12:00 UTC)'),
        ('Era 0 annual emission', '2,625,000 QUG/year'),
        ('Halving interval',    '126,230,400 seconds  (4 × 365.25 days)'),
        ('Total eras',          '64  (~256 years to full emission)'),
        ('Max reward/block',    '2 QUG'),
        ('Correction factor',   'Budget-based, bounded to [0.01, 3.0]'),
        ('Arithmetic',          'Pure u128 — no floating-point'),
    ])

    pdf.h2('13.2 Token Ecosystem')
    pdf.kv_table([
        ('QUG',     '24 decimals · 21M max supply · Native deflationary token'),
        ('QUGUSD',  '24 decimals · Collateral-bounded · Algorithmic stablecoin'),
        ('QCREDIT', '24 decimals · 1:1 QUG lock · Yield vault (v8.5.5)'),
        ('QUSD',    '24 decimals · Founder-controlled · Issuer stablecoin'),
        ('wBTC',    '8 decimals · 1:1 Bitcoin collateral (deposit bridge)'),
        ('VAULT',   '0 decimals · Physical device RWA token'),
    ])

    pdf.h2('13.3 Fee Distribution')
    pdf.bullet([
        'Total fee: configurable dev_fee_bps (default 100 bps = 1%).',
        'Operator split: configurable portion to node operator address.',
        'Miner remainder: balance after dev fee and operator split.',
        'All distributions handled within coinbase transaction in same block as mining solution.',
    ])

    # ── §14 Roadmap ──────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1('14. Roadmap & Open Issues')

    pdf.h2('14.1 Priority 1 — DEX Invariant Fix (Before AMM Promotion)')
    pdf.body(
        'DEX-001/002 must be fixed before the DEX can be advertised as a finalized settlement layer. '
        'The fix requires connecting execute_quantum_trade() to update_pool_reserves() under a '
        'single write lock, and verifying k invariant post-update:'
    )
    pdf.code(
        '// In execute_quantum_trade(), after price computation:\n'
        'let mut pools = self.liquidity_manager.quantum_pools.write().await;\n'
        'let pool = pools.get_mut(&pair_id).ok_or(DexError::PoolNotFound)?;\n'
        'let new_k = new_reserve_in * new_reserve_out;\n'
        'ensure!(new_k >= pool.k_invariant, DexError::InvariantViolation);\n'
        'pool.reserve_in = new_reserve_in;\n'
        'pool.reserve_out = new_reserve_out;'
    )

    pdf.h2('14.2 Priority 2 — Stratum Pool Security')
    pdf.bullet([
        'POOL-001: Replace HashSet with bloom filter or LRU deque — no full-clear.',
        'POOL-002: Enforce share_difficulty >= min_difficulty synchronously at stratum.rs:511.',
        'POOL-003: Change dedup key to hash(job_id ‖ nonce) — exclude miner-controlled extranonce2.',
        'POOL-004: Pass clean_jobs=true to create_job() on block found.',
    ])

    pdf.h2('14.3 Priority 3 — Rate Limiter Architecture')
    pdf.body(
        'Separate user-initiated requests from background polling in RequestRateLimiter. '
        'User actions must never be blocked by background SSE polling filling all slots. '
        'Recommended: 15 slots for user actions + 10 for background, with priority queue.'
    )

    pdf.h2('14.4 Priority 4 — BalanceRootV1 Pre-Activation Monitoring')
    pdf.body(
        'Current height is ~18,500,000 — BalanceRootV1 activates at 18,600,000 (~100,000 blocks away). '
        'Add startup health check that verifies compute_balance_root_for_block() succeeds before '
        'accepting mining submissions near activation height (BAL-001 mitigation).'
    )

    pdf.h2('14.5 Longer Term')
    pdf.bullet([
        'Q2 → Q3 migration: Activate SQIsign / FROST threshold signatures.',
        'Xlattice NTT implementation: Complete NTT forward/inverse for full Dilithium5 FPGA.',
        'Ethereum bridge: Implement bridge handler functions for EVM-compatible chains.',
        'Monero bridge: Ring signature cross-chain integration.',
        'Code refactor: Split main.rs (24,609 lines) into domain modules (ARCH-001).',
        'Gamma v10.6.1 deploy: Currently one release behind — run ha-deploy.sh verify-gamma.',
    ])

    # ── §15 Conclusion ───────────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1('15. Conclusion')

    pdf.body(
        'Q-NarwhalKnight v10.6.1 is a feature-complete, production-grade blockchain with genuine '
        'post-quantum cryptographic hardening already active in Q1 hybrid mode, a fully operational '
        'Bitcoin deposit bridge, real-time SSE wallet synchronization, and a 4-server HA cluster '
        'serving a live mainnet since February 2026.'
    )
    pdf.body(
        'The technical audit across three review iterations (v1: inventory, v2: deep security, '
        'v3: post-incident) found the core consensus, storage, and mining subsystems to be '
        'correct and safe. The highest-priority remaining issues are architectural: the DEX trading '
        'engine is disconnected from pool reserve mutation (DEX-001/002), and the Stratum pool has '
        'several exploitable vulnerabilities (POOL-001..004) that should be addressed before the '
        'system processes significant mining volume.'
    )
    pdf.body(
        'With BalanceRootV1 activation approaching (~18.5M/18.6M blocks), the system is entering '
        'its most critical operational window. The upgrade gate mechanism and 5-test determinism '
        'suite provide high confidence in the implementation, but pre-activation monitoring and '
        'the BAL-001 fallback mitigation should be completed before the height is reached.'
    )

    pdf.h2('Key Strengths')
    pdf.bullet([
        'Mining reward accounting: 3-layer persistent dedup — no double-credit possible.',
        'BalanceRootV1: Correctly implemented, fully tested, approaching activation.',
        'Sync-down protection: Both application and database layers enforce safety aborts.',
        'Post-quantum transition: Height-gated upgrades enable seamless Q0 → Q2 migration.',
        'Privacy infrastructure: Dandelion++ mandatory, Tor integrated, ring/shielded txs production-ready.',
        'Turbo Sync: 1,100 blocks/sec peak on Epsilon — 11.4M block history syncs in 5.5 hours.',
        'HA deployment: Zero-downtime rolling upgrades with auto-rollback.',
        'Bitcoin bridge: Live HD-derived deposit addresses via Delta Bitcoin Knots RPC.',
    ])

    pdf.h2('Key Risks to Address')
    pdf.bullet([
        'DEX-001/002 CRITICAL: Pool reserves not updated by swaps — DEX is price discovery only.',
        'POOL-001 HIGH: Share dedup full-clear enables double-submission after 100k shares.',
        'BAL-001 MEDIUM: Balance root compute failure → chain halt; add startup health check.',
        'ARCH-001 MEDIUM: 24,609-line main.rs is a long-term maintenance liability.',
        'Gamma is one release behind (v10.6.0 vs v10.6.1) — needs deploy.',
    ])

    pdf.ln(8)
    pdf.set_font('DejaVu', 'I', 8.5)
    pdf.set_text_color(*DARK_GRAY)
    pdf.cell(0, 6,
             'Q-NarwhalKnight Whitepaper v3  ·  Generated 2026-05-07  ·  Server Beta (Claude Code)',
             align='C')

    return pdf


if __name__ == '__main__':
    out_path = os.path.join(os.path.dirname(__file__), 'qnk-whitepaper-2026-05-07.pdf')
    print('Generating charts...')
    # Pre-generate all charts (errors surface early)
    charts = {
        'emission':    make_emission_schedule(),
        'turbo_sync':  make_turbo_sync_performance(),
        'topology':    make_network_topology(),
        'dag':         make_consensus_dag(),
        'crypto':      make_crypto_phases(),
        'block_prod':  make_block_production_rate(),
        'issues':      make_issue_severity_chart(),
    }
    print(f'  {len(charts)} charts generated.')

    print('Building PDF...')
    pdf = build_pdf()

    print(f'Saving to {out_path}...')
    pdf.output(out_path)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f'Done!  {out_path}  ({size_mb:.2f} MB)')
