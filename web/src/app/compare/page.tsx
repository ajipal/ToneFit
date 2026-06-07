"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import Nav from "@/components/Nav";
import { SUBTYPE_DISPLAY } from "@/lib/season-data";
import {
  ModelCard, ComparisonTable, SEASON_COLORS, TipBubble,
  type CompareResponse,
} from "@/components/ModelComparison";

type Tab = "performance" | "demo";

const TABS: { id: Tab; label: string }[] = [
  { id: "performance", label: "Model Performance" },
  { id: "demo",        label: "Live Comparison" },
];

const NAV_H = 88;
const TAB_H = 48;

// ── Hardcoded test-set results ─────────────────────────────────────────────────

const SEASONS = ["autumn", "spring", "summer", "winter"] as const;
type Season = (typeof SEASONS)[number];

const FARL = {
  accuracy: 0.5636, f1_macro: 0.5546, top2_acc: 0.8377, autumn_recall: 0.4363, n_test: 912,
  best_epoch: 6,
  backbone: "FaRL (CLIP ViT-B/16, feature_dim=512)",
  per_season: {
    autumn: { precision: 0.5825, recall: 0.4363, f1: 0.4989, support: 259 },
    spring: { precision: 0.5000, recall: 0.4335, f1: 0.4644, support: 203 },
    summer: { precision: 0.5099, recall: 0.5538, f1: 0.5309, support: 186 },
    winter: { precision: 0.6176, recall: 0.7955, f1: 0.6954, support: 264 },
  } as Record<Season, { precision: number; recall: number; f1: number; support: number }>,
  confusion: [
    [113, 40, 22, 84],
    [34,  88, 63, 18],
    [16,  39, 103, 28],
    [31,   9,  14, 210],
  ],
};

const SVM = {
  accuracy: 0.4566, f1_macro: 0.4507, top2_acc: 0.7350, autumn_recall: 0.5309, n_test: 668,
  best_params: "RBF kernel, C=100, γ=scale",
  features: "L*, a*, b*, ITA, Hue, Saturation, Value",
  per_season: {
    autumn: { precision: 0.4503, recall: 0.5309, f1: 0.4873, support: 162 },
    spring: { precision: 0.3926, recall: 0.2928, f1: 0.3354, support: 181 },
    summer: { precision: 0.4477, recall: 0.4529, f1: 0.4503, support: 170 },
    winter: { precision: 0.5235, recall: 0.5742, f1: 0.5477, support: 155 },
  } as Record<Season, { precision: number; recall: number; f1: number; support: number }>,
  confusion: [
    [86, 27, 21, 28],
    [49, 53, 54, 25],
    [24, 41, 77, 28],
    [32, 14, 20, 89],
  ],
};

const PAPER = [
  { model: "FaRL-16",    source: "Stacchio et al. (2024)", accuracy: 0.525, f1: 0.516, top2: 0.815 },
  { model: "FaRL-64",    source: "Stacchio et al. (2024)", accuracy: 0.554, f1: 0.548, top2: 0.808 },
  { model: "ResNeXt50",  source: "Stacchio et al. (2024)", accuracy: 0.513, f1: 0.502, top2: 0.789 },
];

const FARL_TRAIN_ACC = [0.4678,0.5531,0.5841,0.6085,0.6233,0.6395,0.6440,0.6577,0.6564,0.6609,0.6697,0.6537,0.6607,0.6739,0.6784,0.6861,0.7003,0.7093,0.7088,0.7126,0.7136,0.7088,0.7223,0.7253,0.7330,0.7363,0.7398,0.7580,0.7657,0.7652,0.7657,0.7610,0.7567,0.7595,0.7747,0.7802,0.7882,0.7899,0.8039,0.7982,0.7997,0.7927,0.7959,0.7977,0.8036,0.8124,0.8141,0.8276,0.8221,0.8236];
const FARL_VAL_ACC   = [0.5285,0.5482,0.5428,0.5581,0.5285,0.5636,0.5471,0.5482,0.5450,0.5428,0.5428,0.5461,0.5296,0.5307,0.5482,0.5406,0.5274,0.5329,0.5351,0.5362,0.5329,0.5340,0.5362,0.5384,0.5351,0.5406,0.5482,0.5351,0.5461,0.5461,0.5417,0.5274,0.5395,0.5373,0.5285,0.5428,0.5406,0.5384,0.5384,0.5395,0.5395,0.5482,0.5329,0.5450,0.5362,0.5362,0.5395,0.5296,0.5384,0.5362];

// ── Sub-components ─────────────────────────────────────────────────────────────

function MetricBar({ value, max = 1, color }: { value: number; max?: number; color: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
      <div style={{ flex: 1, height: "6px", background: "#f0f0f0", borderRadius: "3px", overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${(value / max) * 100}%`, background: color, borderRadius: "3px" }} />
      </div>
      <span style={{ fontFamily: "monospace", fontSize: "12px", fontWeight: 600, color: "#111", minWidth: "42px", textAlign: "right" }}>
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p style={{ fontSize: "10px", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "#aaa", marginBottom: "16px" }}>
      {children}
    </p>
  );
}

function TippedText({ children, tip }: { children: React.ReactNode; tip: string }) {
  const [rect, setRect] = useState<DOMRect | null>(null);
  return (
    <span
      style={{ cursor: "help", borderBottom: "1px dashed #ccc", display: "inline-block" }}
      onMouseEnter={(e) => setRect(e.currentTarget.getBoundingClientRect())}
      onMouseLeave={() => setRect(null)}
    >
      {children}
      {rect && <TipBubble text={tip} anchorRect={rect} />}
    </span>
  );
}

function ConfCell({ val, rowTotal, ri, ci, maxVal }: {
  val: number; rowTotal: number; ri: number; ci: number; maxVal: number;
}) {
  const [rect, setRect] = useState<DOMRect | null>(null);
  const isDiag = ri === ci;
  const intensity = val / maxVal;
  const seasonColor = SEASON_COLORS[SEASONS[ri]];
  const pct = ((val / rowTotal) * 100).toFixed(1);
  const bg = isDiag
    ? `${seasonColor}${Math.round(40 + intensity * 180).toString(16).padStart(2, "0")}`
    : `rgba(0,0,0,${(intensity * 0.12).toFixed(2)})`;
  const tip = isDiag
    ? `✓ ${val} / ${rowTotal} ${SEASONS[ri]} images (${pct}%) correctly classified as ${SEASONS[ci]}.`
    : `${val} / ${rowTotal} ${SEASONS[ri]} images (${pct}%) misclassified as ${SEASONS[ci]}.`;
  return (
    <div
      style={{
        background: bg,
        color: isDiag && intensity > 0.55 ? "#fff" : "#111",
        textAlign: "center", padding: "7px 4px", borderRadius: "6px",
        fontWeight: isDiag ? 700 : 400, fontFamily: "monospace", fontSize: "12px", cursor: "help",
      }}
      onMouseEnter={(e) => setRect(e.currentTarget.getBoundingClientRect())}
      onMouseLeave={() => setRect(null)}
    >
      {rect && <TipBubble text={tip} anchorRect={rect} />}
      {val}
    </div>
  );
}

const CARD_METRIC_TIPS: Record<string, string> = {
  "Season Accuracy": "Fraction of test images correctly classified as the right season (4 classes). Chance baseline is 25%.",
  "F1 Macro": "Unweighted average F1 across all 4 seasons. Macro averaging treats each season equally regardless of sample count — important when classes are imbalanced.",
  "Top-2 Accuracy": "Correct season appears in the model's top-2 predictions. High top-2 with lower top-1 means the model often narrows it to two adjacent seasons.",
  "Autumn Recall": "Of all actual Autumn test images, what fraction the model correctly identified as Autumn. Low recall means many Autumn faces are misclassified. Autumn is the hardest class.",
};

const SEASON_TABLE_TIPS: Record<string, string> = {
  Precision: "Of all images predicted as this season, what fraction actually belongs to it. Low precision = too many false positives for this season.",
  Recall: "Of all actual images of this season, what fraction the model correctly identified. Low recall = the model frequently misses true examples.",
  F1: "Harmonic mean of precision and recall: 2 × (P × R) / (P + R). Penalizes both low precision and low recall equally — the single best per-class quality score.",
};

const BASELINE_TIPS: Record<string, string> = {
  Accuracy: "4-class season accuracy on the Deep Armocromia test set, as reported in each paper.",
  "F1 Macro": "Macro-averaged F1 across 4 seasons. Unweighted — each season contributes equally regardless of class size.",
  "Top-2": "Correct season appears in the model's top-2 predictions. Reflects how often the right answer is at least considered.",
};

function MetricRow({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div>
      <div style={{ marginBottom: "4px" }}>
        <TippedText tip={CARD_METRIC_TIPS[label] ?? ""}>{label}</TippedText>
      </div>
      <MetricBar value={value} color={color} />
    </div>
  );
}

function TrainingCurve() {
  const W = 560, H = 180;
  const PAD = { top: 16, right: 16, bottom: 36, left: 44 };
  const pw = W - PAD.left - PAD.right;
  const ph = H - PAD.top - PAD.bottom;
  const MIN_Y = 0.45, MAX_Y = 0.85;
  const n = FARL_TRAIN_ACC.length;

  const cx = (i: number) => PAD.left + (i / (n - 1)) * pw;
  const cy = (v: number) => PAD.top + ph - ((v - MIN_Y) / (MAX_Y - MIN_Y)) * ph;

  const trainPts = FARL_TRAIN_ACC.map((v, i) => `${cx(i).toFixed(1)},${cy(v).toFixed(1)}`).join(" ");
  const valPts   = FARL_VAL_ACC.map((v, i) => `${cx(i).toFixed(1)},${cy(v).toFixed(1)}`).join(" ");

  const bestIdx = 5;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }}>
      {/* Y grid lines */}
      {[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8].map(v => (
        <g key={v}>
          <line x1={PAD.left} y1={cy(v)} x2={W - PAD.right} y2={cy(v)} stroke="#f0f0f0" strokeWidth="1" />
          <text x={PAD.left - 5} y={cy(v)} textAnchor="end" dominantBaseline="middle" fill="#bbb" fontSize="9">
            {(v * 100).toFixed(0)}%
          </text>
        </g>
      ))}
      {/* X axis */}
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="#e8e8e8" strokeWidth="1" />
      {[1, 10, 20, 30, 40, 50].map(ep => {
        const i = ep - 1;
        return (
          <g key={ep}>
            <line x1={cx(i)} y1={H - PAD.bottom} x2={cx(i)} y2={H - PAD.bottom + 4} stroke="#ddd" strokeWidth="1" />
            <text x={cx(i)} y={H - PAD.bottom + 13} textAnchor="middle" fill="#bbb" fontSize="9">{ep}</text>
          </g>
        );
      })}
      {/* X axis label */}
      <text x={PAD.left + pw / 2} y={H - 2} textAnchor="middle" fill="#bbb" fontSize="9">Epoch</text>
      {/* Train curve */}
      <polyline points={trainPts} fill="none" stroke="#d0d0d0" strokeWidth="1.5" strokeDasharray="4 2" />
      {/* Val curve */}
      <polyline points={valPts} fill="none" stroke="#3A60D8" strokeWidth="2" />
      {/* Best epoch marker */}
      <circle cx={cx(bestIdx)} cy={cy(FARL_VAL_ACC[bestIdx])} r="4" fill="#3A60D8" />
      <text x={cx(bestIdx) + 7} y={cy(FARL_VAL_ACC[bestIdx]) - 5} fill="#3A60D8" fontSize="9" fontWeight="600">
        Best epoch {bestIdx + 1} — {(FARL_VAL_ACC[bestIdx] * 100).toFixed(1)}%
      </text>
      {/* Legend */}
      <line x1={W - 140} y1={PAD.top + 6} x2={W - 122} y2={PAD.top + 6} stroke="#d0d0d0" strokeWidth="1.5" strokeDasharray="4 2" />
      <text x={W - 118} y={PAD.top + 10} fill="#bbb" fontSize="9">Train acc</text>
      <line x1={W - 62} y1={PAD.top + 6} x2={W - 44} y2={PAD.top + 6} stroke="#3A60D8" strokeWidth="2" />
      <text x={W - 40} y={PAD.top + 10} fill="#3A60D8" fontSize="9">Val acc</text>
    </svg>
  );
}

function ConfusionMatrix({ matrix, title }: {
  matrix: number[][];
  title: string;
}) {
  const maxVal = Math.max(...matrix.flat());
  return (
    <div>
      <SectionLabel>{title}</SectionLabel>
      {/* Column headers */}
      <div style={{ display: "grid", gridTemplateColumns: "72px repeat(4, 1fr)", gap: "3px", marginBottom: "3px" }}>
        <div />
        {SEASONS.map(s => (
          <div key={s} style={{ textAlign: "center", fontSize: "10px", fontWeight: 700, textTransform: "capitalize", color: "#888", paddingBottom: "4px" }}>
            {s}
          </div>
        ))}
      </div>
      {matrix.map((row, ri) => (
        <div key={ri} style={{ display: "grid", gridTemplateColumns: "72px repeat(4, 1fr)", gap: "3px", marginBottom: "3px" }}>
          <div style={{ display: "flex", alignItems: "center", fontSize: "10px", fontWeight: 700, textTransform: "capitalize", color: "#888" }}>
            {SEASONS[ri]}
          </div>
          {row.map((val, ci) => (
            <ConfCell
              key={ci} val={val}
              rowTotal={row.reduce((a, b) => a + b, 0)}
              ri={ri} ci={ci} maxVal={maxVal}
            />
          ))}
        </div>
      ))}
      <p style={{ marginTop: "8px", fontSize: "10px", color: "#ccc" }}>Rows = actual · Columns = predicted · Diagonal = correct</p>
    </div>
  );
}

// ── Performance tab ────────────────────────────────────────────────────────────

function PerformanceTab() {
  const farlColor = "#3A60D8";
  const svmColor  = "#B84020";

  return (
    <div style={{ maxWidth: "900px", margin: "0 auto", padding: "40px 32px 80px" }}>

      {/* Header */}
      <div style={{ marginBottom: "48px" }}>
        <p style={{ fontSize: "11px", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "#bbb", marginBottom: "8px" }}>
          Research Question — Stacchio et al. (2024) Recommendation 3
        </p>
        <h1 style={{ fontSize: "36px", fontWeight: 900, lineHeight: "42px", color: "#111", margin: "0 0 12px 0" }}>
          FaRL vs SVM
        </h1>
        <p style={{ fontSize: "14px", color: "#888", lineHeight: "22px", maxWidth: "560px", margin: 0 }}>
          Deep learning (FaRL · CLIP ViT-B/16) vs classical computer vision (SVM · CIELab + HSV features) for
          personal color season classification on the Deep Armocromia dataset.
        </p>
      </div>

      {/* Overall metrics — hero cards */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginBottom: "48px" }}>
        {/* FaRL */}
        <div style={{ border: "1.5px solid #3A60D8", borderRadius: "16px", padding: "24px" }}>
          <p style={{ fontSize: "10px", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "#3A60D8", marginBottom: "4px" }}>Model A</p>
          <p style={{ fontSize: "20px", fontWeight: 900, color: "#111", margin: "0 0 2px 0" }}>FaRL</p>
          <p style={{ fontSize: "12px", color: "#aaa", margin: "0 0 20px 0" }}>CLIP ViT-B/16 · Deep Learning · n=912</p>
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            <MetricRow label="Season Accuracy" value={FARL.accuracy}       color={farlColor} />
            <MetricRow label="F1 Macro"        value={FARL.f1_macro}       color={farlColor} />
            <MetricRow label="Top-2 Accuracy"  value={FARL.top2_acc}       color={farlColor} />
            <MetricRow label="Autumn Recall"   value={FARL.autumn_recall}  color={farlColor} />
          </div>
        </div>

        {/* SVM */}
        <div style={{ border: "1.5px solid #e0e0e0", borderRadius: "16px", padding: "24px" }}>
          <p style={{ fontSize: "10px", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: svmColor, marginBottom: "4px" }}>Model B</p>
          <p style={{ fontSize: "20px", fontWeight: 900, color: "#111", margin: "0 0 2px 0" }}>SVM</p>
          <p style={{ fontSize: "12px", color: "#aaa", margin: "0 0 20px 0" }}>RBF kernel · CIELab + HSV · n=668</p>
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            <MetricRow label="Season Accuracy" value={SVM.accuracy}      color={svmColor} />
            <MetricRow label="F1 Macro"        value={SVM.f1_macro}      color={svmColor} />
            <MetricRow label="Top-2 Accuracy"  value={SVM.top2_acc}      color={svmColor} />
            <MetricRow label="Autumn Recall"   value={SVM.autumn_recall} color={svmColor} />
          </div>
        </div>
      </div>

      {/* Per-season breakdown */}
      <div style={{ marginBottom: "48px" }}>
        <SectionLabel>Per-Season Breakdown</SectionLabel>
        <div style={{ border: "1px solid #f0f0f0", borderRadius: "12px", overflow: "hidden" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
            <thead>
              <tr style={{ background: "#fafafa", borderBottom: "1px solid #f0f0f0" }}>
                <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 700, fontSize: "10px", letterSpacing: "0.08em", textTransform: "uppercase", color: "#aaa" }}>Season</th>
                <th colSpan={3} style={{ padding: "10px 16px", textAlign: "center", fontWeight: 700, fontSize: "10px", letterSpacing: "0.08em", textTransform: "uppercase", color: farlColor }}>FaRL</th>
                <th colSpan={3} style={{ padding: "10px 16px", textAlign: "center", fontWeight: 700, fontSize: "10px", letterSpacing: "0.08em", textTransform: "uppercase", color: svmColor }}>SVM</th>
              </tr>
              <tr style={{ background: "#fafafa", borderBottom: "1.5px solid #e8e8e8" }}>
                <th style={{ padding: "6px 16px" }} />
                {["Precision", "Recall", "F1"].map(h => (
                  <th key={`f-${h}`} style={{ padding: "6px 12px", textAlign: "right", fontWeight: 600, fontSize: "10px", color: "#bbb" }}>
                    <TippedText tip={SEASON_TABLE_TIPS[h]}>{h}</TippedText>
                  </th>
                ))}
                {["Precision", "Recall", "F1"].map(h => (
                  <th key={`s-${h}`} style={{ padding: "6px 12px", textAlign: "right", fontWeight: 600, fontSize: "10px", color: "#bbb" }}>
                    <TippedText tip={SEASON_TABLE_TIPS[h]}>{h}</TippedText>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {SEASONS.map((s, i) => {
                const fc = FARL.per_season[s];
                const sc = SVM.per_season[s];
                const color = SEASON_COLORS[s] ?? "#888";
                return (
                  <tr key={s} style={{ borderBottom: i < 3 ? "1px solid #f8f8f8" : "none" }}>
                    <td style={{ padding: "12px 16px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: color, flexShrink: 0 }} />
                        <span style={{ fontWeight: 600, textTransform: "capitalize", color: "#111" }}>{s}</span>
                        <span style={{ fontSize: "10px", color: "#ccc" }}>n={fc.support}/{sc.support}</span>
                      </div>
                    </td>
                    {[fc.precision, fc.recall, fc.f1].map((v, vi) => (
                      <td key={vi} style={{ padding: "12px", textAlign: "right", fontFamily: "monospace", fontWeight: v === Math.max(fc.precision, fc.recall, fc.f1) ? 700 : 400, color: "#111" }}>
                        {v.toFixed(4)}
                      </td>
                    ))}
                    {[sc.precision, sc.recall, sc.f1].map((v, vi) => (
                      <td key={vi} style={{ padding: "12px", textAlign: "right", fontFamily: "monospace", fontWeight: v === Math.max(sc.precision, sc.recall, sc.f1) ? 700 : 400, color: "#111" }}>
                        {v.toFixed(4)}
                      </td>
                    ))}
                  </tr>
                );
              })}
              {/* Macro avg row */}
              <tr style={{ background: "#fafafa", borderTop: "1.5px solid #e8e8e8" }}>
                <td style={{ padding: "12px 16px", fontWeight: 700, fontSize: "11px", color: "#888", textTransform: "uppercase", letterSpacing: "0.05em" }}>Macro Avg</td>
                {[
                  SEASONS.reduce((s, k) => s + FARL.per_season[k].precision, 0) / 4,
                  SEASONS.reduce((s, k) => s + FARL.per_season[k].recall, 0) / 4,
                  FARL.f1_macro,
                ].map((v, i) => (
                  <td key={i} style={{ padding: "12px", textAlign: "right", fontFamily: "monospace", fontWeight: 700, color: farlColor }}>{v.toFixed(4)}</td>
                ))}
                {[
                  SEASONS.reduce((s, k) => s + SVM.per_season[k].precision, 0) / 4,
                  SEASONS.reduce((s, k) => s + SVM.per_season[k].recall, 0) / 4,
                  SVM.f1_macro,
                ].map((v, i) => (
                  <td key={i} style={{ padding: "12px", textAlign: "right", fontFamily: "monospace", fontWeight: 700, color: svmColor }}>{v.toFixed(4)}</td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Paper baselines comparison */}
      <div style={{ marginBottom: "48px" }}>
        <SectionLabel>Comparison vs Paper Baselines</SectionLabel>
        <div style={{ border: "1px solid #f0f0f0", borderRadius: "12px", overflow: "hidden" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
            <thead>
              <tr style={{ background: "#fafafa", borderBottom: "1.5px solid #e8e8e8" }}>
                {["Model", "Source", "Accuracy", "F1 Macro", "Top-2"].map(h => (
                  <th key={h} style={{ padding: "10px 16px", textAlign: h === "Model" || h === "Source" ? "left" : "right", fontWeight: 700, fontSize: "10px", letterSpacing: "0.08em", textTransform: "uppercase", color: "#aaa" }}>
                    {BASELINE_TIPS[h] ? <TippedText tip={BASELINE_TIPS[h]}>{h}</TippedText> : h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {PAPER.map((row, i) => (
                <tr key={i} style={{ borderBottom: "1px solid #f8f8f8" }}>
                  <td style={{ padding: "11px 16px", fontWeight: 500, color: "#555" }}>{row.model}</td>
                  <td style={{ padding: "11px 16px", fontSize: "11px", color: "#aaa" }}>{row.source}</td>
                  <td style={{ padding: "11px 16px", textAlign: "right", fontFamily: "monospace", color: "#555" }}>{(row.accuracy * 100).toFixed(1)}%</td>
                  <td style={{ padding: "11px 16px", textAlign: "right", fontFamily: "monospace", color: "#555" }}>{(row.f1 * 100).toFixed(1)}%</td>
                  <td style={{ padding: "11px 16px", textAlign: "right", fontFamily: "monospace", color: "#555" }}>{(row.top2 * 100).toFixed(1)}%</td>
                </tr>
              ))}
              {/* Divider */}
              <tr style={{ borderTop: "2px solid #e0e0e0" }}>
                <td style={{ padding: "11px 16px", fontWeight: 700, color: "#111" }}>FaRL (Ours)</td>
                <td style={{ padding: "11px 16px", fontSize: "11px", color: "#aaa" }}>This study</td>
                <td style={{ padding: "11px 16px", textAlign: "right", fontFamily: "monospace", fontWeight: 700, color: farlColor }}>{(FARL.accuracy * 100).toFixed(1)}%</td>
                <td style={{ padding: "11px 16px", textAlign: "right", fontFamily: "monospace", fontWeight: 700, color: farlColor }}>{(FARL.f1_macro * 100).toFixed(1)}%</td>
                <td style={{ padding: "11px 16px", textAlign: "right", fontFamily: "monospace", fontWeight: 700, color: farlColor }}>{(FARL.top2_acc * 100).toFixed(1)}%</td>
              </tr>
              <tr>
                <td style={{ padding: "11px 16px", fontWeight: 700, color: "#111" }}>SVM (Ours)</td>
                <td style={{ padding: "11px 16px", fontSize: "11px", color: "#aaa" }}>This study</td>
                <td style={{ padding: "11px 16px", textAlign: "right", fontFamily: "monospace", fontWeight: 700, color: svmColor }}>{(SVM.accuracy * 100).toFixed(1)}%</td>
                <td style={{ padding: "11px 16px", textAlign: "right", fontFamily: "monospace", fontWeight: 700, color: svmColor }}>{(SVM.f1_macro * 100).toFixed(1)}%</td>
                <td style={{ padding: "11px 16px", textAlign: "right", fontFamily: "monospace", fontWeight: 700, color: svmColor }}>{(SVM.top2_acc * 100).toFixed(1)}%</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p style={{ marginTop: "10px", fontSize: "11px", color: "#bbb", lineHeight: "17px" }}>
          FaRL (Ours) achieves {((FARL.accuracy - 0.554) * 100).toFixed(1)}pp above FaRL-64 paper baseline.
          SVM provides a strong classical CV reference point. Test sets differ in size (FaRL n=912, SVM n=668).
        </p>
      </div>

      {/* FaRL training curve */}
      <div style={{ marginBottom: "48px" }}>
        <SectionLabel>FaRL Training Curve — 50 Epochs</SectionLabel>
        <div style={{ border: "1px solid #f0f0f0", borderRadius: "12px", padding: "20px 16px 8px" }}>
          <TrainingCurve />
          <p style={{ fontSize: "11px", color: "#bbb", marginTop: "8px", lineHeight: "17px" }}>
            Train accuracy (dashed) steadily increases to ~82%. Validation accuracy (blue) peaks at epoch 6 (56.4%) then plateaus — classic early stopping regime. Best model checkpoint saved at epoch 6.
          </p>
        </div>
      </div>

      {/* Confusion matrices */}
      <div style={{ marginBottom: "48px" }}>
        <SectionLabel>Confusion Matrices</SectionLabel>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>
          <div style={{ border: "1px solid #f0f0f0", borderRadius: "12px", padding: "20px" }}>
            <ConfusionMatrix matrix={FARL.confusion} title="FaRL" />
          </div>
          <div style={{ border: "1px solid #f0f0f0", borderRadius: "12px", padding: "20px" }}>
            <ConfusionMatrix matrix={SVM.confusion} title="SVM" />
          </div>
        </div>
      </div>

      {/* Architecture notes */}
      <div style={{ background: "#fafafa", borderRadius: "12px", padding: "20px 24px" }}>
        <SectionLabel>Model Architectures</SectionLabel>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", fontSize: "12px", color: "#555", lineHeight: "20px" }}>
          <div>
            <p style={{ fontWeight: 700, color: "#111", marginBottom: "6px" }}>FaRL (Model A)</p>
            <p style={{ margin: 0 }}>{FARL.backbone}<br />Frozen backbone · FC(512→256) → ReLU → Dropout(0.5) → FC(256→4)<br />Trained 50 epochs, batch size 64, best at epoch {FARL.best_epoch}</p>
          </div>
          <div>
            <p style={{ fontWeight: 700, color: "#111", marginBottom: "6px" }}>SVM (Model B)</p>
            <p style={{ margin: 0 }}>{SVM.best_params}<br />Features: {SVM.features}<br />class_weight=balanced · 5-fold CV · train_time 71.6s</p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Demo tab (existing upload + compare) ───────────────────────────────────────

function DemoTab() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [photoPreview, setPhotoPreview] = useState<string | null>(null);
  const [photoData,    setPhotoData]    = useState<string | null>(null);
  const [loading,      setLoading]      = useState(false);
  const [result,       setResult]       = useState<CompareResponse | null>(null);
  const [error,        setError]        = useState<string | null>(null);
  const [dragOver,     setDragOver]     = useState(false);

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
    setPhotoPreview(null); setPhotoData(null); setResult(null); setError(null);
  };

  const finalColor = result ? (SEASON_COLORS[result.final_season] ?? "#000") : "#000";

  return (
    <div>
      {/* Pre-result: centered upload hero */}
      {!result && (
        <div className="flex flex-col items-center justify-center text-center px-5 md:px-0" style={{ minHeight: `calc(100vh - ${NAV_H + TAB_H}px)`, paddingBottom: "40px" }}>
          <p className="text-xs font-bold tracking-widest uppercase text-neutral-400 mb-3">Run Both Models</p>
          <h2 className="text-[28px] md:text-[40px] font-black leading-tight tracking-tight mb-3">Compare on Your Photo</h2>
          <p className="text-sm md:text-base text-neutral-500 mb-8 md:mb-10 max-w-md">
            Upload a photo to run both FaRL and SVM on your image and see their predictions side by side.
          </p>

          <div className="flex flex-col gap-4 w-full max-w-[380px]">
            {photoPreview ? (
              <div className="flex flex-col gap-4 items-center">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={photoPreview} alt="Uploaded" className="w-full max-w-xs rounded-2xl object-cover aspect-[3/4]" />
                <div className="flex gap-3 w-full">
                  <button onClick={handleAnalyze} disabled={loading}
                    className="flex-1 py-3 bg-black text-white rounded-xl font-semibold text-sm hover:bg-neutral-800 transition-colors disabled:opacity-50">
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
                className={`w-full aspect-[4/3] rounded-2xl border-2 border-dashed flex flex-col items-center justify-center gap-3 cursor-pointer transition-all ${dragOver ? "border-black bg-neutral-50" : "border-black/15 hover:border-black/40"}`}
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

          {error && (
            <div className="mt-6">
              {error === "no_face" ? (
                <div className="rounded-2xl bg-neutral-50 border border-black/8 p-6 max-w-sm flex flex-col gap-3 text-left">
                  <p className="text-sm font-semibold text-black mb-1">No face detected</p>
                  <p className="text-sm text-neutral-500 leading-relaxed">Try a clear, well-lit front-facing photo with your full face visible.</p>
                  <button onClick={reset} className="text-sm font-semibold text-black underline underline-offset-2 text-left">Upload a different photo</button>
                </div>
              ) : (
                <div className="rounded-2xl bg-red-50 border border-red-100 p-5 text-sm text-red-700 max-w-lg">{error}</div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Post-result */}
      {result && (
        <div className="px-4 md:px-14 py-8 md:py-10 max-w-7xl mx-auto flex flex-col gap-6 md:gap-8">
          <div className="flex gap-6 items-start">
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
            <div className="flex-1 grid grid-cols-2 gap-5">
              <ModelCard title="FaRL" subtitle="CLIP ViT-B/16 · Deep Learning" model={result.farl} accent="#3A60D8" />
              <ModelCard title="SVM" subtitle="RBF kernel · CIELab + HSV" model={result.svm} accent="#B84020" />
            </div>
          </div>
          <ComparisonTable farl={result.farl} svm={result.svm} />
        </div>
      )}
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────────

export default function ComparePage() {
  const [activeTab,  setActiveTab]  = useState<Tab>("performance");
  const [navVisible, setNavVisible] = useState(true);
  const lastYRef = useRef(0);

  useEffect(() => {
    let ticking = false;
    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        const y = window.scrollY;
        if (y < 10)                       setNavVisible(true);
        else if (y > lastYRef.current + 4) setNavVisible(false);
        else if (y < lastYRef.current - 4) setNavVisible(true);
        lastYRef.current = y;
        ticking = false;
      });
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <div className="min-h-screen bg-white text-black">
      <Nav />

      {/* Tab bar — mirrors Nav's scroll-hide transform */}
      <div style={{
        position: "fixed",
        top: `${NAV_H}px`,
        left: 0, right: 0,
        height: `${TAB_H}px`,
        background: "#fff",
        borderBottom: "1px solid #e8e8e8",
        display: "flex", alignItems: "stretch", justifyContent: "center",
        zIndex: 40,
        transform: navVisible ? "translateY(0)" : `translateY(-${NAV_H + TAB_H}px)`,
        transition: "transform 300ms ease-in-out",
      }}>
        {TABS.map(tab => {
          const active = activeTab === tab.id;
          return (
            <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
              flex: 1, maxWidth: "240px", fontSize: "14px",
              fontWeight: active ? 600 : 400, color: active ? "#111" : "#888",
              background: "none", border: "none",
              borderBottom: active ? "2px solid #111" : "2px solid transparent",
              cursor: "pointer", transition: "color 0.15s, border-color 0.15s", whiteSpace: "nowrap",
            }}>
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Content pushed below both nav bars */}
      <div style={{ paddingTop: `${NAV_H + TAB_H}px` }}>
        {activeTab === "performance" && <PerformanceTab />}
        {activeTab === "demo"        && <DemoTab />}
      </div>
    </div>
  );
}
