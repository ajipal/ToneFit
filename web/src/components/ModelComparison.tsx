"use client";

import { SUBTYPE_DISPLAY } from "@/lib/season-data";

// ── Types ──────────────────────────────────────────────────────────────────────

export interface RawFeatures {
  L_mean: number; a_mean: number; b_mean: number;
  ITA: number; H_mean: number; S_mean: number; V_mean: number;
}

export interface ModelResult {
  available: boolean;
  error?: string;
  season?: string;
  season_confidence?: number;
  season_probs?: Record<string, number>;
  subtype?: string;
  subtype_confidence?: number;
  top3?: Array<{ subtype: string; confidence: number }>;
  features_used?: string[];
  raw_features?: RawFeatures;
}

export interface CompareResponse {
  face_detected: boolean;
  final_season: string;
  final_subtype: string;
  farl: ModelResult;
  svm: ModelResult;
}

// ── Constants ──────────────────────────────────────────────────────────────────

export const SEASON_COLORS: Record<string, string> = {
  autumn: "#B84020", spring: "#E87850", summer: "#8898D0", winter: "#3A60D8",
};

export const SEASONS = ["autumn", "spring", "summer", "winter"];

const FEATURE_META: Record<string, { label: string; desc: string; min: number; max: number; warm?: boolean }> = {
  L_mean: { label: "L* (Lightness)",     desc: "0 = black, 100 = white",          min: 0,   max: 100 },
  a_mean: { label: "a* (Green–Red)",     desc: "− = cool green, + = warm red",    min: -30, max: 30, warm: true },
  b_mean: { label: "b* (Blue–Yellow)",   desc: "− = cool blue, + = warm yellow",  min: -30, max: 30, warm: true },
  ITA:    { label: "ITA (Undertone)",    desc: "− = dark/cool, + = light/warm",   min: -90, max: 90, warm: true },
  H_mean: { label: "Hue",               desc: "0–360°, 0/360 = red",             min: 0,   max: 360 },
  S_mean: { label: "Saturation",         desc: "0 = grey, 1 = vivid",             min: 0,   max: 1   },
  V_mean: { label: "Value (Brightness)", desc: "0 = dark, 1 = bright",            min: 0,   max: 1   },
};

// ── Helpers ────────────────────────────────────────────────────────────────────

export function deriveStats(probs: Record<string, number>) {
  const warm = (probs.autumn ?? 0) + (probs.spring ?? 0);
  const cool = (probs.summer ?? 0) + (probs.winter ?? 0);
  const sorted = Object.values(probs).sort((a, b) => b - a);
  const margin = sorted[0] - sorted[1];
  const entropy = -Object.values(probs).reduce(
    (s, p) => s + (p > 0 ? p * Math.log2(p) : 0), 0
  );
  const certainty = 1 - entropy / Math.log2(4);
  return { warm, cool, margin, entropy, certainty };
}

// ── Micro components ───────────────────────────────────────────────────────────

export function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[10px] font-bold tracking-widest uppercase text-neutral-400 mb-3">
      {children}
    </p>
  );
}

export function StatRow({ label, value, bar, color, note }: {
  label: string; value: string; bar?: number; color?: string; note?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between items-center">
        <span className="text-xs text-neutral-500">{label}</span>
        <span className="text-xs font-mono font-semibold text-black">{value}</span>
      </div>
      {bar !== undefined && (
        <div className="h-1.5 bg-neutral-100 rounded-full overflow-hidden">
          <div className="h-full rounded-full" style={{ width: `${Math.max(2, bar * 100)}%`, backgroundColor: color ?? "#000" }} />
        </div>
      )}
      {note && <p className="text-[10px] text-neutral-400">{note}</p>}
    </div>
  );
}

export function ConfBar({ label, value, color, highlight }: {
  label: string; value: number; color: string; highlight: boolean;
}) {
  return (
    <div className={`flex flex-col gap-1 ${highlight ? "" : "opacity-40"}`}>
      <div className="flex justify-between text-xs">
        <span className={`capitalize font-medium ${highlight ? "text-black" : "text-neutral-500"}`}>{label}</span>
        <span className="font-mono text-neutral-500">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-1.5 bg-neutral-100 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${value * 100}%`, backgroundColor: highlight ? color : "#e5e5e5" }} />
      </div>
    </div>
  );
}

export function FeatureGauge({ name, value }: { name: string; value: number }) {
  const meta = FEATURE_META[name];
  if (!meta) return null;
  const pct = (value - meta.min) / (meta.max - meta.min);
  const clamped = Math.max(0, Math.min(1, pct));
  const isWarm = meta.warm && value > 0;
  const isCool = meta.warm && value < 0;
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between items-center">
        <span className="text-xs text-neutral-600 font-medium">{meta.label}</span>
        <span className="text-xs font-mono text-black font-semibold">{value}</span>
      </div>
      <div className="h-1.5 bg-neutral-100 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{ width: `${clamped * 100}%`, backgroundColor: isWarm ? "#B84020" : isCool ? "#3A60D8" : "#6b7280" }}
        />
      </div>
      <p className="text-[10px] text-neutral-400">{meta.desc}</p>
    </div>
  );
}

// ── Model card ─────────────────────────────────────────────────────────────────

export function ModelCard({ title, subtitle, model, accent }: {
  title: string; subtitle: string; model: ModelResult; accent: string;
}) {
  if (!model.available) {
    return (
      <div className="rounded-2xl border border-black/8 p-6 flex flex-col gap-3">
        <div>
          <p className="text-xs font-bold tracking-widest uppercase text-neutral-400">{title}</p>
          <p className="text-sm text-neutral-500 mt-0.5">{subtitle}</p>
        </div>
        <div className="rounded-xl bg-neutral-50 p-4 text-sm text-neutral-400">
          Model not loaded{model.error ? `: ${model.error}` : ""}
        </div>
      </div>
    );
  }

  const topSeason = model.season!;
  const color = SEASON_COLORS[topSeason] ?? accent;
  const stats = model.season_probs ? deriveStats(model.season_probs) : null;

  return (
    <div className="rounded-2xl border border-black/8 p-6 flex flex-col gap-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-bold tracking-widest uppercase text-neutral-400">{title}</p>
          <p className="text-sm text-neutral-500 mt-0.5">{subtitle}</p>
        </div>
        <div className="px-3 py-1 rounded-full text-xs font-bold text-white capitalize shrink-0" style={{ backgroundColor: color }}>
          {topSeason}
        </div>
      </div>

      <div className="flex flex-col gap-2">
        <SectionTitle>Season Confidence</SectionTitle>
        {SEASONS.map((s) => (
          <ConfBar key={s} label={s} value={model.season_probs?.[s] ?? 0}
            color={SEASON_COLORS[s] ?? "#888"} highlight={s === topSeason} />
        ))}
      </div>

      {stats && (
        <div className="flex flex-col gap-3 rounded-xl bg-neutral-50 p-4">
          <SectionTitle>Prediction Analysis</SectionTitle>
          <StatRow label="Warm bias (Autumn + Spring)" value={`${(stats.warm * 100).toFixed(1)}%`} bar={stats.warm} color="#B84020" />
          <StatRow label="Cool bias (Summer + Winter)" value={`${(stats.cool * 100).toFixed(1)}%`} bar={stats.cool} color="#3A60D8" />
          <StatRow label="Confidence margin (1st − 2nd)" value={`${(stats.margin * 100).toFixed(1)}%`} bar={stats.margin} color={color}
            note={stats.margin > 0.3 ? "Decisive prediction" : stats.margin > 0.15 ? "Moderate confidence" : "Low confidence — close call"} />
          <StatRow label="Prediction certainty" value={`${(stats.certainty * 100).toFixed(1)}%`} bar={stats.certainty} color={color}
            note={`Entropy: ${stats.entropy.toFixed(3)} bits (max 2.0)`} />
        </div>
      )}

      {model.subtype && (
        <div className="flex flex-col gap-3">
          <SectionTitle>Sub-type Prediction</SectionTitle>
          <div className="rounded-xl border border-black/8 p-4 flex flex-col gap-1">
            <p className="text-base font-black">{SUBTYPE_DISPLAY[model.subtype] ?? model.subtype}</p>
            <p className="text-xs text-neutral-400 font-mono">{(model.subtype_confidence! * 100).toFixed(1)}% confidence</p>
          </div>
          {model.top3 && (
            <div className="flex flex-col gap-1.5">
              {model.top3.map((item, i) => (
                <div key={item.subtype} className="flex justify-between items-center text-sm">
                  <div className="flex items-center gap-2">
                    <span className="text-neutral-300 text-xs w-3">{i + 1}</span>
                    <span className={i === 0 ? "font-semibold text-black" : "text-neutral-500"}>
                      {SUBTYPE_DISPLAY[item.subtype] ?? item.subtype}
                    </span>
                  </div>
                  <span className="text-xs font-mono text-neutral-400">{(item.confidence * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {model.raw_features && (
        <div className="flex flex-col gap-3">
          <SectionTitle>CIELab / HSV Input Features</SectionTitle>
          <div className="flex flex-col gap-3">
            {Object.entries(model.raw_features).map(([k, v]) => (
              <FeatureGauge key={k} name={k} value={v} />
            ))}
          </div>
          <p className="text-[10px] text-neutral-400 leading-relaxed">
            Red bars = warm signal (a*, b*, ITA {'>'} 0). Blue bars = cool signal ({'<'} 0). Gray bars = neutral features.
          </p>
        </div>
      )}
    </div>
  );
}

// ── Comparison table ───────────────────────────────────────────────────────────

export function ComparisonTable({ farl, svm }: { farl: ModelResult; svm: ModelResult }) {
  if (!farl.available || !svm.available || !farl.season_probs || !svm.season_probs) return null;

  const fs = deriveStats(farl.season_probs);
  const ss = deriveStats(svm.season_probs);
  const agree = farl.season === svm.season;

  const rows = [
    { label: "Predicted season",     farl: farl.season ?? "—",                                   svm: svm.season ?? "—",                                   compare: farl.season === svm.season },
    { label: "Season confidence",    farl: `${((farl.season_confidence ?? 0)*100).toFixed(1)}%`,  svm: `${((svm.season_confidence ?? 0)*100).toFixed(1)}%`,  compare: null },
    { label: "Warm bias",            farl: `${(fs.warm*100).toFixed(1)}%`,                        svm: `${(ss.warm*100).toFixed(1)}%`,                       compare: null },
    { label: "Cool bias",            farl: `${(fs.cool*100).toFixed(1)}%`,                        svm: `${(ss.cool*100).toFixed(1)}%`,                       compare: null },
    { label: "Confidence margin",    farl: `${(fs.margin*100).toFixed(1)}%`,                      svm: `${(ss.margin*100).toFixed(1)}%`,                     compare: null },
    { label: "Prediction certainty", farl: `${(fs.certainty*100).toFixed(1)}%`,                   svm: `${(ss.certainty*100).toFixed(1)}%`,                  compare: null },
    { label: "Entropy (bits)",       farl: fs.entropy.toFixed(3),                                 svm: ss.entropy.toFixed(3),                                compare: null },
  ];

  return (
    <div className="rounded-2xl border border-black/8 overflow-hidden">
      <div className="px-6 py-4 border-b border-black/8 flex items-center justify-between">
        <p className="text-sm font-bold">Side-by-side Comparison</p>
        <div className={`px-3 py-1 rounded-full text-xs font-semibold ${agree ? "bg-green-50 text-green-700" : "bg-amber-50 text-amber-700"}`}>
          {agree ? "✓ Models agree" : "≠ Models disagree"}
        </div>
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-black/8 bg-neutral-50">
            <th className="text-left px-6 py-3 text-xs font-bold text-neutral-400 uppercase tracking-wider w-1/3">Metric</th>
            <th className="text-left px-6 py-3 text-xs font-bold text-neutral-400 uppercase tracking-wider">FaRL</th>
            <th className="text-left px-6 py-3 text-xs font-bold text-neutral-400 uppercase tracking-wider">SVM</th>
            <th className="px-6 py-3 w-8" />
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label} className="border-b border-black/5 last:border-0">
              <td className="px-6 py-3 text-neutral-500 text-xs">{row.label}</td>
              <td className="px-6 py-3 font-mono font-semibold text-black capitalize">{row.farl}</td>
              <td className="px-6 py-3 font-mono font-semibold text-black capitalize">{row.svm}</td>
              <td className="px-6 py-3 text-center">
                {row.compare !== null && (
                  <span className={row.compare ? "text-green-500" : "text-amber-500"}>{row.compare ? "✓" : "≠"}</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
