"use client";

import { useState } from "react";
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

const SEASON_TIPS: Record<string, string> = {
  autumn: "Probability assigned to Autumn. Higher = more golden, earthy, muted warmth detected. Autumn palettes suit warm, deep complexions with low contrast.",
  spring: "Probability assigned to Spring. Higher = clear, bright warmth detected. Spring palettes suit light-to-medium warm complexions with fresh, vivid coloring.",
  summer: "Probability assigned to Summer. Higher = cool, soft, muted tones detected. Summer palettes suit cool complexions with low contrast and ashy or rosy undertones.",
  winter: "Probability assigned to Winter. Higher = cool, clear, high-contrast coloring detected. Winter palettes suit cool complexions with strong contrast between features.",
};

const FEATURE_TIPS: Record<string, string> = {
  L_mean: "CIELab lightness (0–100). Lighter complexions (higher L*) lean Spring/Summer. Darker (lower L*) lean Autumn/Winter. The most direct measure of how light or dark the skin is.",
  a_mean: "CIELab red-green axis. Positive = warm pinkish/reddish hue — a Spring or Autumn signal. Negative = cool greenish hue — a Summer or Winter signal.",
  b_mean: "CIELab yellow-blue axis. Positive = warm golden/yellow cast — strong Autumn or Spring indicator. Negative = cool bluish cast — Winter or Summer indicator.",
  ITA:    "Individual Typology Angle — a combined undertone score derived from L* and b*. Above 0 = warm/light (Spring/Summer likely). Below 0 = cool/dark (Autumn/Winter likely). One of the strongest predictors in this model.",
  H_mean: "Dominant hue angle on the color wheel (0–360°). Skin sits in the 0–50° range. Closer to orange/yellow (higher) = warmer undertone. Closer to red (lower) = more neutral or cool-warm.",
  S_mean: "HSV saturation — how vivid the skin color is. Low = desaturated/neutral (Summer/Winter signal). Higher = vivid color intensity (Spring/Autumn signal).",
  V_mean: "HSV brightness — how much light is reflected. High brightness + warm hue = Spring. High brightness + cool = Summer. Low + warm = Autumn. Low + cool = Winter.",
};

const STAT_TIPS: Record<string, string> = {
  warm:    "Combined probability for warm seasons (Autumn + Spring). Above 50% means the model reads warm undertones — golden, peachy, or earthy tones in your complexion.",
  cool:    "Combined probability for cool seasons (Summer + Winter). Above 50% means the model reads cool undertones — ashy, rosy, or icy tones.",
  margin:  "Gap between the 1st and 2nd ranked seasons. >30% = decisive prediction. 15–30% = moderate confidence. <15% = close call — you may sit between two seasons.",
  cert:    "How concentrated probability is on one season. Derived from entropy (lower entropy = higher certainty). Low values mean the model is unsure and spread across multiple seasons.",
  entropy: "Information-theoretic uncertainty. 0 bits = completely certain. 2.0 bits = maximum uncertainty (random guess across 4 seasons). Lower is more confident.",
};

const TABLE_ROW_TIPS: Record<string, string> = {
  "Predicted season":     "The season label each model ultimately assigned to your photo.",
  "Season confidence":    "Probability the model places on its top prediction. 25% = random guess. 100% = completely certain. Above 50% is a strong signal.",
  "Warm bias":            "Both models' combined probability for warm seasons (Autumn + Spring). Compare to see if both models agree on temperature.",
  "Cool bias":            "Both models' combined probability for cool seasons (Summer + Winter). High cool bias means both models lean toward ashy/icy undertones.",
  "Confidence margin":    "Difference between top-1 and top-2 season probabilities. Higher = more decisive. Low margin means you are a close call between two seasons.",
  "Prediction certainty": "Derived from entropy. Higher = model is more confident. Lower = probability is spread across seasons and the prediction is less reliable.",
  "Entropy (bits)":       "Uncertainty measure (Shannon entropy over 4 seasons). 0 = certain, 2.0 = maximum uncertainty. Lower is better for prediction reliability.",
};

// ── Tooltip bubble ─────────────────────────────────────────────────────────────

export function TipBubble({ text, anchorRect }: { text: string; anchorRect: DOMRect }) {
  const left = anchorRect.left + anchorRect.width / 2;
  const top  = anchorRect.top - 8;
  return (
    <div style={{
      position: "fixed",
      left: `${left}px`,
      top: `${top}px`,
      transform: "translate(-50%, -100%)",
      background: "#111",
      color: "#fff",
      padding: "8px 12px",
      borderRadius: "8px",
      fontSize: "12px",
      lineHeight: "17px",
      maxWidth: "240px",
      width: "max-content",
      whiteSpace: "normal",
      zIndex: 9999,
      pointerEvents: "none",
      boxShadow: "0 4px 14px rgba(0,0,0,0.18)",
      textTransform: "none",
      letterSpacing: "normal",
      fontWeight: 400,
    }}>
      {text}
      <div style={{
        position: "absolute",
        top: "100%",
        left: "50%",
        transform: "translateX(-50%)",
        borderWidth: "5px",
        borderStyle: "solid",
        borderColor: "#111 transparent transparent transparent",
      }} />
    </div>
  );
}

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

export function StatRow({ label, value, bar, color, note, tooltip }: {
  label: string; value: string; bar?: number; color?: string; note?: string; tooltip?: string;
}) {
  const [rect, setRect] = useState<DOMRect | null>(null);
  return (
    <div
      className="flex flex-col gap-1"
      style={{ cursor: tooltip ? "help" : "default" }}
      onMouseEnter={(e) => tooltip && setRect(e.currentTarget.getBoundingClientRect())}
      onMouseLeave={() => setRect(null)}
    >
      {rect && tooltip && <TipBubble text={tooltip} anchorRect={rect} />}
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

export function ConfBar({ label, value, color, highlight, tooltip }: {
  label: string; value: number; color: string; highlight: boolean; tooltip?: string;
}) {
  const [rect, setRect] = useState<DOMRect | null>(null);
  return (
    <div
      className={`flex flex-col gap-1 ${highlight ? "" : "opacity-40"}`}
      style={{ cursor: tooltip ? "help" : "default" }}
      onMouseEnter={(e) => tooltip && setRect(e.currentTarget.getBoundingClientRect())}
      onMouseLeave={() => setRect(null)}
    >
      {rect && tooltip && <TipBubble text={tooltip} anchorRect={rect} />}
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
  const tip = FEATURE_TIPS[name];
  const [rect, setRect] = useState<DOMRect | null>(null);
  return (
    <div
      className="flex flex-col gap-1"
      style={{ cursor: "help" }}
      onMouseEnter={(e) => tip && setRect(e.currentTarget.getBoundingClientRect())}
      onMouseLeave={() => setRect(null)}
    >
      {rect && tip && <TipBubble text={tip} anchorRect={rect} />}
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
          <ConfBar
            key={s}
            label={s}
            value={model.season_probs?.[s] ?? 0}
            color={SEASON_COLORS[s] ?? "#888"}
            highlight={s === topSeason}
            tooltip={SEASON_TIPS[s]}
          />
        ))}
      </div>

      {stats && (
        <div className="flex flex-col gap-3 rounded-xl bg-neutral-50 p-4">
          <SectionTitle>Prediction Analysis</SectionTitle>
          <StatRow
            label="Warm bias (Autumn + Spring)" value={`${(stats.warm * 100).toFixed(1)}%`}
            bar={stats.warm} color="#B84020" tooltip={STAT_TIPS.warm}
          />
          <StatRow
            label="Cool bias (Summer + Winter)" value={`${(stats.cool * 100).toFixed(1)}%`}
            bar={stats.cool} color="#3A60D8" tooltip={STAT_TIPS.cool}
          />
          <StatRow
            label="Confidence margin (1st − 2nd)" value={`${(stats.margin * 100).toFixed(1)}%`}
            bar={stats.margin} color={color} tooltip={STAT_TIPS.margin}
            note={stats.margin > 0.3 ? "Decisive prediction" : stats.margin > 0.15 ? "Moderate confidence" : "Low confidence — close call"}
          />
          <StatRow
            label="Prediction certainty" value={`${(stats.certainty * 100).toFixed(1)}%`}
            bar={stats.certainty} color={color} tooltip={STAT_TIPS.cert}
            note={`Entropy: ${stats.entropy.toFixed(3)} bits (max 2.0)`}
          />
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
            <TableRow key={row.label} row={row} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TableRow({ row }: {
  row: { label: string; farl: string; svm: string; compare: boolean | null };
}) {
  const [rect, setRect] = useState<DOMRect | null>(null);
  const tip = TABLE_ROW_TIPS[row.label];
  return (
    <tr className="border-b border-black/5 last:border-0">
      <td
        className="px-6 py-3 text-neutral-500 text-xs"
        style={{ cursor: tip ? "help" : "default" }}
        onMouseEnter={(e) => tip && setRect(e.currentTarget.getBoundingClientRect())}
        onMouseLeave={() => setRect(null)}
      >
        {rect && tip && <TipBubble text={tip} anchorRect={rect} />}
        <span className={tip ? "border-b border-dashed border-neutral-300" : ""}>{row.label}</span>
      </td>
      <td className="px-6 py-3 font-mono font-semibold text-black capitalize">{row.farl}</td>
      <td className="px-6 py-3 font-mono font-semibold text-black capitalize">{row.svm}</td>
      <td className="px-6 py-3 text-center">
        {row.compare !== null && (
          <span className={row.compare ? "text-green-500" : "text-amber-500"}>{row.compare ? "✓" : "≠"}</span>
        )}
      </td>
    </tr>
  );
}
