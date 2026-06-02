"use client";

import { useState, useRef, useCallback } from "react";
import Nav from "@/components/Nav";
import { SUBTYPE_DISPLAY } from "@/lib/season-data";
import {
  ModelCard, ComparisonTable, SEASON_COLORS,
  type ModelResult, type CompareResponse,
} from "@/components/ModelComparison";

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
