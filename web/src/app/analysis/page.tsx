"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import Nav from "@/components/Nav";
import { SUBTYPE_DISPLAY } from "@/lib/season-data";
import {
  ModelCard, ComparisonTable, SEASON_COLORS,
  type CompareResponse,
} from "@/components/ModelComparison";

function AnalysisInner() {
  const params    = useSearchParams();
  const seasonKey = params.get("season");

  const [photo,   setPhoto]   = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [result,  setResult]  = useState<CompareResponse | null>(null);
  const [error,   setError]   = useState<string | null>(null);

  useEffect(() => {
    const savedPhoto = localStorage.getItem("tonefit_last_photo");
    if (!savedPhoto) {
      setError("No photo found. Please go back and complete the analysis first.");
      setLoading(false);
      return;
    }
    setPhoto(savedPhoto);

    fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ photo: savedPhoto }),
    })
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => ({ error: "Unknown error" }));
          const msg: string = body.error ?? `HTTP ${res.status}`;
          throw new Error(msg.toLowerCase().includes("no face") ? "no_face" : msg);
        }
        return res.json();
      })
      .then((data: CompareResponse) => setResult(data))
      .catch((err: unknown) => setError(err instanceof Error ? err.message : "Unexpected error"))
      .finally(() => setLoading(false));
  }, []);

  const finalColor = result ? (SEASON_COLORS[result.final_season] ?? "#000") : "#000";

  return (
    <div className="min-h-screen bg-white text-black">
      <Nav showFullLinks />

      <div className="px-14 py-10 max-w-7xl mx-auto flex flex-col gap-8" style={{ paddingTop: "108px" }}>

        {/* Back link — always visible */}
        <button
          onClick={() => {
            window.location.href = seasonKey ? `/results?season=${seasonKey}` : "/";
          }}
          className="flex items-center gap-1.5 text-sm font-semibold text-black hover:opacity-60 transition-opacity self-start"
        >
          ← Back to results
        </button>

        {/* Loading */}
        {loading && (
          <div className="flex flex-col items-center justify-center gap-4 py-24 text-neutral-400">
            <div className="w-8 h-8 rounded-full border-2 border-black/10 border-t-black animate-spin" />
            <p className="text-sm">Running model comparison…</p>
          </div>
        )}

        {/* Error */}
        {error && !loading && (
          <div className="rounded-2xl bg-neutral-50 border border-black/8 p-8 flex flex-col gap-4 max-w-md">
            <p className="text-sm font-semibold text-black">Analysis failed</p>
            <p className="text-sm text-neutral-500 leading-relaxed">
              {error === "no_face"
                ? "No face was detected in the saved photo. Please retake your analysis with a clear, front-facing photo."
                : error}
            </p>
            <button
              onClick={() => { window.location.href = seasonKey ? `/results?season=${seasonKey}` : "/"; }}
              className="text-sm font-semibold text-black underline underline-offset-2 self-start"
            >
              ← Back to results
            </button>
          </div>
        )}

        {/* Result */}
        {result && !loading && (
          <>
            <div className="flex gap-6 items-start">

              {/* Photo + verdict sidebar */}
              <div className="w-56 shrink-0 flex flex-col gap-3 sticky top-8">
                {photo && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={photo} alt="Analyzed" className="w-full rounded-2xl object-cover aspect-[3/4]" />
                )}
                <div className="flex items-center gap-2 text-xs text-neutral-500">
                  <div className={`w-2 h-2 rounded-full shrink-0 ${result.face_detected ? "bg-green-500" : "bg-amber-400"}`} />
                  {result.face_detected ? "Face detected" : "No face — full image used"}
                </div>
                <div className="rounded-2xl p-4 flex flex-col gap-1 text-white" style={{ backgroundColor: finalColor }}>
                  <p className="text-[10px] font-bold tracking-widest uppercase opacity-70">Final Prediction</p>
                  <p className="text-xl font-black capitalize">{result.final_season}</p>
                  <p className="text-xs opacity-80">{SUBTYPE_DISPLAY[result.final_subtype] ?? result.final_subtype}</p>
                </div>
              </div>

              {/* Model cards */}
              <div className="flex-1 grid grid-cols-2 gap-5">
                <ModelCard title="FaRL" subtitle="CLIP ViT-B/16 · Deep Learning" model={result.farl} accent="#3A60D8" />
                <ModelCard title="SVM" subtitle="RBF kernel · CIELab + HSV" model={result.svm} accent="#B84020" />
              </div>
            </div>

            <ComparisonTable farl={result.farl} svm={result.svm} />
          </>
        )}
      </div>
    </div>
  );
}

export default function AnalysisPage() {
  return (
    <Suspense>
      <AnalysisInner />
    </Suspense>
  );
}
