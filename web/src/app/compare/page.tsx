"use client";

import { useState, useRef, useCallback } from "react";
import Nav from "@/components/Nav";
import { SUBTYPE_DISPLAY } from "@/lib/season-data";

// ─────────────────────────────────────────────
//  Types
// ─────────────────────────────────────────────

interface RawFeatures {
  L_mean: number; a_mean: number; b_mean: number;
  ITA: number; H_mean: number; S_mean: number; V_mean: number;
}

interface ModelResult {
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

interface CompareResponse {
  face_detected: boolean;
  final_season: string;
  final_subtype: string;
  farl: ModelResult;
  svm: ModelResult;
}

// ─────────────────────────────────────────────
//  Derived stats (computed from season_probs)
// ─────────────────────────────────────────────

function deriveStats(probs: Record<string, number>) {
  const warm = (probs.autumn ?? 0) + (probs.spring ?? 0);
  const cool = (probs.summer ?? 0) + (probs.winter ?? 0);
  const sorted = Object.values(probs).sort((a, b) => b - a);
  const margin = sorted[0] - sorted[1];
  const entropy = -Object.values(probs).reduce(
    (s, p) => s + (p > 0 ? p * Math.log2(p) : 0), 0
  );
  const maxEntropy = Math.log2(4); // 2 bits for 4 classes
  const certainty = 1 - entropy / maxEntropy;
  return { warm, cool, margin, entropy, certainty };
}

// ─────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────

const SEASON_COLORS: Record<string, string> = {
  autumn: "#B84020", spring: "#E87850", summer: "#8898D0", winter: "#3A60D8",
};
const SEASONS = ["autumn", "spring", "summer", "winter"];

// Feature metadata for display
const FEATURE_META: Record<string, { label: string; desc: string; min: number; max: number; warm?: boolean }> = {
  L_mean: { label: "L* (Lightness)",    desc: "0 = black, 100 = white",   min: 0,    max: 100  },
  a_mean: { label: "a* (Green–Red)",    desc: "− = cool green, + = warm red", min: -30, max: 30, warm: true },
  b_mean: { label: "b* (Blue–Yellow)",  desc: "− = cool blue, + = warm yellow", min: -30, max: 30, warm: true },
  ITA:    { label: "ITA (Undertone)",   desc: "− = dark/cool, + = light/warm", min: -90, max: 90, warm: true },
  H_mean: { label: "Hue",              desc: "0–360°, 0/360 = red",       min: 0,    max: 360  },
  S_mean: { label: "Saturation",        desc: "0 = grey, 1 = vivid",      min: 0,    max: 1    },
  V_mean: { label: "Value (Brightness)", desc: "0 = dark, 1 = bright",   min: 0,    max: 1    },
};

// ─────────────────────────────────────────────
//  Micro components
// ─────────────────────────────────────────────

function StatRow({ label, value, bar, color, note }: {
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
          <div
            className="h-full rounded-full"
            style={{ width: `${Math.max(2, bar * 100)}%`, backgroundColor: color ?? "#000" }}
          />
        </div>
      )}
      {note && <p className="text-[10px] text-neutral-400">{note}</p>}
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[10px] font-bold tracking-widest uppercase text-neutral-400 mb-3">
      {children}
    </p>
  );
}

function ConfBar({ label, value, color, highlight }: {
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

// ─────────────────────────────────────────────
//  Feature gauge (SVM raw values)
// ─────────────────────────────────────────────

function FeatureGauge({ name, value }: { name: string; value: number }) {
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
          style={{
            width: `${clamped * 100}%`,
            backgroundColor: isWarm ? "#B84020" : isCool ? "#3A60D8" : "#6b7280",
          }}
        />
      </div>
      <p className="text-[10px] text-neutral-400">{meta.desc}</p>
    </div>
  );
}

// ─────────────────────────────────────────────
//  Model card
// ─────────────────────────────────────────────

function ModelCard({ title, subtitle, model, accent }: {
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

      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-bold tracking-widest uppercase text-neutral-400">{title}</p>
          <p className="text-sm text-neutral-500 mt-0.5">{subtitle}</p>
        </div>
        <div
          className="px-3 py-1 rounded-full text-xs font-bold text-white capitalize shrink-0"
          style={{ backgroundColor: color }}
        >
          {topSeason}
        </div>
      </div>

      {/* Season confidence bars */}
      <div className="flex flex-col gap-2">
        <SectionTitle>Season Confidence</SectionTitle>
        {SEASONS.map((s) => (
          <ConfBar key={s} label={s} value={model.season_probs?.[s] ?? 0}
            color={SEASON_COLORS[s] ?? "#888"} highlight={s === topSeason} />
        ))}
      </div>

      {/* Derived stats */}
      {stats && (
        <div className="flex flex-col gap-3 rounded-xl bg-neutral-50 p-4">
          <SectionTitle>Prediction Analysis</SectionTitle>
          <StatRow
            label="Warm bias (Autumn + Spring)"
            value={`${(stats.warm * 100).toFixed(1)}%`}
            bar={stats.warm} color="#B84020"
          />
          <StatRow
            label="Cool bias (Summer + Winter)"
            value={`${(stats.cool * 100).toFixed(1)}%`}
            bar={stats.cool} color="#3A60D8"
          />
          <StatRow
            label="Confidence margin (1st − 2nd)"
            value={`${(stats.margin * 100).toFixed(1)}%`}
            bar={stats.margin} color={color}
            note={stats.margin > 0.3 ? "Decisive prediction" : stats.margin > 0.15 ? "Moderate confidence" : "Low confidence — close call"}
          />
          <StatRow
            label="Prediction certainty"
            value={`${(stats.certainty * 100).toFixed(1)}%`}
            bar={stats.certainty} color={color}
            note={`Entropy: ${stats.entropy.toFixed(3)} bits (max 2.0)`}
          />
        </div>
      )}

      {/* Subtype — FaRL only */}
      {model.subtype && (
        <div className="flex flex-col gap-3">
          <SectionTitle>Sub-type Prediction</SectionTitle>
          <div className="rounded-xl border border-black/8 p-4 flex flex-col gap-1">
            <p className="text-base font-black">{SUBTYPE_DISPLAY[model.subtype] ?? model.subtype}</p>
            <p className="text-xs text-neutral-400 font-mono">
              {(model.subtype_confidence! * 100).toFixed(1)}% confidence
            </p>
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
                  <span className="text-xs font-mono text-neutral-400">
                    {(item.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Raw features — SVM only */}
      {model.raw_features && (
        <div className="flex flex-col gap-3">
          <SectionTitle>CIELab / HSV Input Features</SectionTitle>
          <div className="flex flex-col gap-3">
            {Object.entries(model.raw_features).map(([k, v]) => (
              <FeatureGauge key={k} name={k} value={v} />
            ))}
          </div>
          <p className="text-[10px] text-neutral-400 leading-relaxed">
            Red bars = warm signal (a*, b*, ITA {'>'} 0). Blue bars = cool signal ({'<'} 0).
            Gray bars = neutral features.
          </p>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────
//  Comparison table
// ─────────────────────────────────────────────

function ComparisonTable({ farl, svm }: { farl: ModelResult; svm: ModelResult }) {
  if (!farl.available || !svm.available || !farl.season_probs || !svm.season_probs) return null;

  const fs = deriveStats(farl.season_probs);
  const ss = deriveStats(svm.season_probs);

  const rows = [
    { label: "Predicted season",    farl: farl.season ?? "—",                          svm: svm.season ?? "—",                        compare: farl.season === svm.season },
    { label: "Season confidence",   farl: `${((farl.season_confidence ?? 0)*100).toFixed(1)}%`, svm: `${((svm.season_confidence ?? 0)*100).toFixed(1)}%`, compare: null },
    { label: "Warm bias",           farl: `${(fs.warm*100).toFixed(1)}%`,               svm: `${(ss.warm*100).toFixed(1)}%`,            compare: null },
    { label: "Cool bias",           farl: `${(fs.cool*100).toFixed(1)}%`,               svm: `${(ss.cool*100).toFixed(1)}%`,            compare: null },
    { label: "Confidence margin",   farl: `${(fs.margin*100).toFixed(1)}%`,             svm: `${(ss.margin*100).toFixed(1)}%`,          compare: null },
    { label: "Prediction certainty",farl: `${(fs.certainty*100).toFixed(1)}%`,          svm: `${(ss.certainty*100).toFixed(1)}%`,       compare: null },
    { label: "Entropy (bits)",      farl: fs.entropy.toFixed(3),                        svm: ss.entropy.toFixed(3),                     compare: null },
  ];

  const agree = farl.season === svm.season;

  return (
    <div className="rounded-2xl border border-black/8 overflow-hidden">
      <div className="px-6 py-4 border-b border-black/8 flex items-center justify-between">
        <p className="text-sm font-bold">Side-by-side Comparison</p>
        <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold ${
          agree ? "bg-green-50 text-green-700" : "bg-amber-50 text-amber-700"
        }`}>
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
                  <span className={row.compare ? "text-green-500" : "text-amber-500"}>
                    {row.compare ? "✓" : "≠"}
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─────────────────────────────────────────────
//  Page
// ─────────────────────────────────────────────

export default function ComparePage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [photoPreview, setPhotoPreview] = useState<string | null>(null);
  const [photoData, setPhotoData]       = useState<string | null>(null);
  const [loading, setLoading]           = useState(false);
  const [result, setResult]             = useState<CompareResponse | null>(null);
  const [error, setError]               = useState<string | null>(null);
  const [dragOver, setDragOver]         = useState(false);

  const handleFile = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const data = e.target?.result as string;
      setPhotoPreview(data);
      setPhotoData(data);
      setResult(null);
      setError(null);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file?.type.startsWith("image/")) handleFile(file);
  }, [handleFile]);

  const handleAnalyze = async () => {
    if (!photoData || loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ photo: photoData }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ error: "Unknown error" }));
        const msg: string = body.error ?? `HTTP ${res.status}`;
        throw new Error(msg.toLowerCase().includes("no face") ? "no_face" : msg);
      }
      setResult(await res.json());
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setPhotoPreview(null);
    setPhotoData(null);
    setResult(null);
    setError(null);
  };

  const finalColor = result ? (SEASON_COLORS[result.final_season] ?? "#000") : "#000";

  return (
    <div className="min-h-screen bg-white text-black">
      <Nav />

      {/* ── Pre-result: centered hero ────────────────────────────────────── */}
      {!result && (
        <div
          className="flex flex-col items-center justify-center text-center"
          style={{ height: "calc(100vh - 88px)", paddingTop: "88px" }}
        >
          <p className="text-xs font-bold tracking-widest uppercase text-neutral-400 mb-3">Model Comparison</p>
          <h1 className="text-[48px] font-black leading-tight tracking-tight mb-3">FaRL vs SVM</h1>
          <p className="text-base text-neutral-500 mb-10 max-w-md">
            Deep learning (FaRL · CLIP ViT-B/16) vs classical CV (SVM · CIELab + HSV).
            Upload a photo to compare both models side by side.
          </p>

          {/* Upload zone */}
          <div className="flex flex-col gap-4 w-[380px]">
            {photoPreview ? (
              <div className="flex flex-col gap-4 items-center">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={photoPreview} alt="Uploaded" className="w-full max-w-xs rounded-2xl object-cover aspect-[3/4]" />
                <div className="flex gap-3 w-full">
                  <button
                    onClick={handleAnalyze} disabled={loading}
                    className="flex-1 py-3 bg-black text-white rounded-xl font-semibold text-sm hover:bg-neutral-800 transition-colors disabled:opacity-50"
                  >
                    {loading ? "Analyzing…" : "Compare Models"}
                  </button>
                  <button onClick={reset} className="px-4 py-3 border border-black/12 rounded-xl text-sm font-semibold text-neutral-600 hover:border-black/30 transition-colors">
                    Reset
                  </button>
                </div>
              </div>
            ) : (
              <div
                onDrop={handleDrop}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onClick={() => fileInputRef.current?.click()}
                className={`w-full aspect-[4/3] rounded-2xl border-2 border-dashed flex flex-col items-center justify-center gap-3 cursor-pointer transition-all ${
                  dragOver ? "border-black bg-neutral-50" : "border-black/15 hover:border-black/40"
                }`}
              >
                <svg viewBox="0 0 40 40" fill="none" className="w-10 h-10 text-neutral-300" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                  <rect x="4" y="10" width="32" height="24" rx="3" />
                  <circle cx="20" cy="22" r="6" />
                  <path d="M14 10l3-4h6l3 4" />
                </svg>
                <p className="text-sm font-medium text-neutral-400">Click or drag a photo to compare</p>
              </div>
            )}
            <input ref={fileInputRef} type="file" accept="image/*" className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
          </div>

          {/* Error */}
          {error && (
            <div className="mt-6">
              {error === "no_face" ? (
                <div className="rounded-2xl bg-neutral-50 border border-black/8 p-6 max-w-sm flex flex-col gap-3 text-left">
                  <div className="w-10 h-10 rounded-full bg-neutral-200 flex items-center justify-center">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" className="w-5 h-5 text-neutral-500">
                      <circle cx="12" cy="8" r="4" />
                      <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" />
                      <line x1="4" y1="4" x2="20" y2="20" />
                    </svg>
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-black mb-1">No face detected</p>
                    <p className="text-sm text-neutral-500 leading-relaxed">
                      Try a clear, well-lit front-facing photo with your full face visible.
                    </p>
                  </div>
                  <button onClick={reset} className="text-sm font-semibold text-black underline underline-offset-2 text-left">
                    Upload a different photo
                  </button>
                </div>
              ) : (
                <div className="rounded-2xl bg-red-50 border border-red-100 p-5 text-sm text-red-700 max-w-lg">{error}</div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── Post-result: full layout ─────────────────────────────────────── */}
      {result && (
        <div className="px-14 py-10 max-w-7xl mx-auto flex flex-col gap-8" style={{ paddingTop: "108px" }}>

          {/* Top row — image + model cards */}
          <div className="flex gap-6 items-start">

            {/* Photo + verdict */}
            <div className="w-56 shrink-0 flex flex-col gap-3 sticky top-8">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={photoPreview!} alt="Analyzed" className="w-full rounded-2xl object-cover aspect-[3/4]" />
              <div className="flex items-center gap-2 text-xs text-neutral-500">
                <div className={`w-2 h-2 rounded-full shrink-0 ${result.face_detected ? "bg-green-500" : "bg-amber-400"}`} />
                {result.face_detected ? "Face detected" : "No face — full image used"}
              </div>
              <div className="rounded-2xl p-4 flex flex-col gap-1 text-white" style={{ backgroundColor: finalColor }}>
                <p className="text-[10px] font-bold tracking-widest uppercase opacity-70">Final Prediction</p>
                <p className="text-xl font-black capitalize">{result.final_season}</p>
                <p className="text-xs opacity-80">{SUBTYPE_DISPLAY[result.final_subtype] ?? result.final_subtype}</p>
              </div>
              <button onClick={reset} className="text-sm font-semibold text-black underline underline-offset-2 text-left">
                Try another photo
              </button>
            </div>

            {/* Model cards */}
            <div className="flex-1 grid grid-cols-2 gap-5">
              <ModelCard title="FaRL" subtitle="CLIP ViT-B/16 · Deep Learning" model={result.farl} accent="#3A60D8" />
              <ModelCard title="SVM" subtitle="RBF kernel · CIELab + HSV" model={result.svm} accent="#B84020" />
            </div>
          </div>

          {/* Comparison table */}
          <ComparisonTable farl={result.farl} svm={result.svm} />
        </div>
      )}
    </div>
  );
}
