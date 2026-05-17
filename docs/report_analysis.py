#!/usr/bin/env python3
"""Generate analysis graphs for Q-NarwhalKnight project report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ── 1. Weekly commit velocity ──────────────────────────────────────────────────
weeks = ['W09\n(Mar 2)', 'W10\n(Mar 9)', 'W11\n(Mar 16)', 'W12\n(Mar 23)',
         'W13\n(Mar 30)', 'W14\n(Apr 6)', 'W15\n(Apr 13)', 'W16\n(Apr 20)',
         'W17\n(Apr 27)', 'W18\n(May 4)', 'W19\n(May 11)']
counts = [78, 27, 11, 37, 8, 48, 71, 52, 28, 52, 4]
colors = ['#2196F3' if c < 50 else '#F44336' if c >= 70 else '#FF9800' for c in counts]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(weeks, counts, color=colors, edgecolor='white', linewidth=0.5)
ax.set_title('Q-NarwhalKnight — Weekly Commit Velocity (Mar–May 2026)', fontsize=13, fontweight='bold', pad=14)
ax.set_xlabel('Week', fontsize=10)
ax.set_ylabel('Commits', fontsize=10)
ax.set_ylim(0, 90)
for bar, v in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, str(v),
            ha='center', va='bottom', fontsize=9, fontweight='bold')
patches = [mpatches.Patch(color='#F44336', label='High velocity (≥70)'),
           mpatches.Patch(color='#FF9800', label='Medium (50–69)'),
           mpatches.Patch(color='#2196F3', label='Normal (<50)')]
ax.legend(handles=patches, fontsize=9)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'report_commits_weekly.png'), dpi=150)
plt.close()
print("Saved report_commits_weekly.png")

# ── 2. Bug category breakdown ──────────────────────────────────────────────────
categories = ['Sync / P2P\n(gap-fill, turbo-sync)', 'Balance /\nLedger integrity',
              'Storage / DB\n(RocksDB, keys)', 'UI / UX\n(frontend, SSE)',
              'Mining /\nVDF / GPU', 'Consensus /\nCryptography', 'Infrastructure\n(deploy, build)']
bug_counts = [29, 18, 12, 11, 12, 9, 7]
palette = ['#E53935', '#FB8C00', '#FDD835', '#43A047', '#1E88E5', '#8E24AA', '#00ACC1']

fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(bug_counts, labels=categories, colors=palette,
                                   autopct='%1.0f%%', startangle=140,
                                   pctdistance=0.78, labeldistance=1.12,
                                   wedgeprops=dict(edgecolor='white', linewidth=1.2))
for t in texts:
    t.set_fontsize(8.5)
for at in autotexts:
    at.set_fontsize(8)
    at.set_fontweight('bold')
ax.set_title('Fix Categories — Mar–May 2026\n(total: 98 bug-fix commits)', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'report_bug_categories.png'), dpi=150)
plt.close()
print("Saved report_bug_categories.png")

# ── 3. Network health / incident timeline ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.set_xlim(0, 71)
ax.set_ylim(-1.8, 2.2)
ax.axhline(0, color='#555', linewidth=1.2, zorder=1)

# Timeline events: (day_offset_from_Mar1, label, y_offset, color)
events = [
    (0,  'v10.2.x era\nbegins', 0.9, '#1E88E5'),
    (10, 'Epsilon disk\ncrisis', -1.0, '#E53935'),
    (14, 'v10.2.9: height\nstall fix', 0.9, '#43A047'),
    (30, 'v10.2.8\nkill-9 recovery', -1.0, '#E53935'),
    (43, 'v10.3.x\nLWMA + GPU', 0.9, '#1E88E5'),
    (49, 'Forward-seek\nDAG fix', 0.9, '#43A047'),
    (56, 'Balance divergence\nincident (62 wallets)', -1.2, '#E53935'),
    (59, 'v10.4.15\ncheckpoint', 0.9, '#1E88E5'),
    (62, 'SYNC-003/004\nbalance replay', 0.9, '#43A047'),
    (66, 'Balance replay\ncorruption (May 9)', -1.0, '#E53935'),
    (68, 'Max-wins guard\n+ SYNC-006 fix', 0.9, '#43A047'),
    (70, 'SYNC-005\nfail-fast (v10.8.3)', 0.9, '#43A047'),
]
for day, label, y, color in events:
    ax.annotate('', xy=(day, 0), xytext=(day, y * 0.85),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    va = 'bottom' if y > 0 else 'top'
    ax.text(day, y, label, ha='center', va=va, fontsize=7.5, color=color, fontweight='bold')

# Version bands
for x0, x1, label in [(0,42,'v10.2.x'), (43,55,'v10.3.x'), (56,60,'v10.4.x'),
                       (61,65,'v10.5–7.x'), (66,71,'v10.8.x')]:
    ax.axvspan(x0, x1, alpha=0.07, color='gray')
    ax.text((x0+x1)/2, -1.7, label, ha='center', fontsize=8, color='#444')

month_ticks = [0, 9, 30, 60, 71]
month_labels = ['Mar 1', 'Mar 10', 'Mar 30', 'May 1', 'May 11']
ax.set_xticks(month_ticks)
ax.set_xticklabels(month_labels, fontsize=9)
ax.set_yticks([])
ax.set_title('Q-NarwhalKnight Network Health Timeline — Mar–May 2026', fontsize=12, fontweight='bold')
green_p = mpatches.Patch(color='#43A047', label='Fix / improvement')
red_p   = mpatches.Patch(color='#E53935', label='Incident / regression')
blue_p  = mpatches.Patch(color='#1E88E5', label='Feature milestone')
ax.legend(handles=[green_p, red_p, blue_p], loc='upper right', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'report_timeline.png'), dpi=150)
plt.close()
print("Saved report_timeline.png")

print("All graphs generated.")
