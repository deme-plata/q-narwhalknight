import React, { useMemo, useState } from 'react';

export interface ScoreComponent {
  label: string;
  value: number;
  explanation?: string;
  weight?: number;
}

export interface ScoreData {
  total?: number;
  components?: ScoreComponent[];
  weights?: Record<string, number>;
}

interface ScoreBreakdownProps {
  score: ScoreData;
  title?: string;
}

const clampScore = (value: number): number => {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(10, value));
};

export default function ScoreBreakdown({ score, title = 'Score Breakdown' }: ScoreBreakdownProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const totalScore = useMemo(() => clampScore(score?.total ?? 0), [score?.total]);
  const components = Array.isArray(score?.components) ? score.components : [];
  const weights = score?.weights ?? {};

  const totalStyle =
    totalScore >= 7
      ? 'bg-emerald-500/20 text-emerald-300 border-emerald-400/40'
      : totalScore >= 4
        ? 'bg-amber-500/20 text-amber-300 border-amber-400/40'
        : 'bg-red-500/20 text-red-300 border-red-400/40';

  return (
    <div className="rounded-2xl p-5 border border-white/10 bg-black/20">
      <div className="flex items-center justify-between gap-3 mb-4">
        <h3 className="text-sm font-semibold text-white/90">{title}</h3>
        <div className={`px-3 py-1 rounded-full border text-sm font-bold ${totalStyle}`}>
          {totalScore.toFixed(1)} / 10
        </div>
      </div>

      <div className="space-y-3">
        {components.length > 0 ? (
          components.map((item, idx) => {
            const itemScore = clampScore(item?.value ?? 0);
            return (
              <div key={`${item.label}-${idx}`}>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-white/80">{item?.label || `Component ${idx + 1}`}</span>
                  <span className="text-white/60">{itemScore.toFixed(1)} / 10</span>
                </div>
                <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-violet-400 to-pink-400"
                    style={{ width: `${itemScore * 10}%` }}
                  />
                </div>
                {item?.explanation && (
                  <p className="text-[11px] mt-1 text-white/55">{item.explanation}</p>
                )}
              </div>
            );
          })
        ) : (
          <p className="text-xs text-white/50">No component details were provided.</p>
        )}
      </div>

      <div className="mt-4">
        <button
          type="button"
          onClick={() => setShowAdvanced((prev) => !prev)}
          className="text-xs text-cyan-300 hover:text-cyan-200 transition-colors"
        >
          {showAdvanced ? 'Hide' : 'Show'} advanced weights
        </button>

        {showAdvanced && (
          <div className="mt-2 rounded-xl border border-white/10 bg-white/5 p-3">
            {Object.keys(weights).length > 0 ? (
              <ul className="space-y-1 text-xs text-white/70">
                {Object.entries(weights).map(([key, value]) => (
                  <li key={key} className="flex justify-between gap-4">
                    <span className="capitalize">{key.replace(/_/g, ' ')}</span>
                    <span>{Number.isFinite(value) ? value : 0}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs text-white/50">No weight metadata available.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
