"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import Nav from "@/components/Nav";
import type { SeasonKey } from "@/lib/season-data";

const ANALYSIS_STEPS = [
  { number: "1", title: "Detecting face",        subtitle: "Locating facial features" },
  { number: "2", title: "Analyzing undertones",  subtitle: "Reading skin tone & contrast" },
  { number: "3", title: "Matching color season", subtitle: "Comparing against 12 seasons" },
  { number: "4", title: "Building style guide",  subtitle: "Generating outfit & palette" },
];

const DOTS = ["●", "●", "●"];

export default function ProcessingPage() {
  const router = useRouter();
  const [progress,    setProgress]    = useState(0);
  const [activeStep,  setActiveStep]  = useState(0);
  const [dotCount,    setDotCount]    = useState(0);
  const [error,       setError]       = useState<string | null>(null);

  useEffect(() => {
    // Guard: no photo means user skipped onboarding
    const photo = sessionStorage.getItem("tonefit_photo");
    if (!photo) {
      router.replace("/onboarding");
      return;
    }

    // These are local-scope flags so tryRedirect closes over them cleanly
    let subtypeResult: SeasonKey | null = null;
    let animationComplete = false;

    function tryRedirect() {
      if (subtypeResult && animationComplete) {
        // Persist photo for results page display, then clear session
        try { if (photo) localStorage.setItem("tonefit_last_photo", photo); } catch {}
        try { localStorage.setItem("tonefit_last_season", subtypeResult); } catch {}
        sessionStorage.removeItem("tonefit_photo");
        router.push(`/results?season=${subtypeResult}`);
      }
    }

    // ── Real ML API call ──────────────────────────────────────────────────────
    fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ photo }),
    })
      .then(async (r) => {
        if (!r.ok) {
          const body = await r.json().catch(() => ({}));
          throw new Error(body.error ?? `HTTP ${r.status}`);
        }
        return r.json();
      })
      .then((data: { subtype: SeasonKey; season: string }) => {
        subtypeResult = data.subtype ?? (data.season as SeasonKey);
        tryRedirect();
      })
      .catch((err: Error) => {
        console.error("[processing] API error:", err);
        const msg = err.message ?? "";
        if (msg.toLowerCase().includes("no face")) {
          setError("no_face");
        } else {
          setError("Could not reach the analysis server. Please try again.");
        }
      });

    // ── Progress bar animation (minimum 5 s for UX) ───────────────────────────
    const duration = 5_000;
    const start    = Date.now();

    const progressTimer = setInterval(() => {
      const pct = Math.min(((Date.now() - start) / duration) * 100, 100);
      setProgress(pct);
      setActiveStep(Math.min(Math.floor((pct / 100) * 4), 3));

      if (pct >= 100) {
        clearInterval(progressTimer);
        animationComplete = true;
        tryRedirect();
      }
    }, 40);

    const dotTimer = setInterval(() => {
      setDotCount((prev) => (prev + 1) % 4);
    }, 500);

    return () => {
      clearInterval(progressTimer);
      clearInterval(dotTimer);
    };
  }, [router]);

  const stepLabel = ANALYSIS_STEPS[activeStep]?.title ?? "";

  // ── Error state ───────────────────────────────────────────────────────────────
  if (error) {
    const isNoFace = error === "no_face";
    return (
      <div className="min-h-screen bg-white text-black flex flex-col">
        <Nav showFullLinks={false} />
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="text-center max-w-md flex flex-col gap-6">
            {isNoFace ? (
              <>
                <div className="w-16 h-16 rounded-full bg-neutral-100 flex items-center justify-center mx-auto">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" className="w-8 h-8 text-neutral-400">
                    <circle cx="12" cy="8" r="4" />
                    <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" />
                    <line x1="4" y1="4" x2="20" y2="20" />
                  </svg>
                </div>
                <div>
                  <h1 className="text-2xl font-black mb-2">No face detected</h1>
                  <p className="text-neutral-500 text-base leading-relaxed">
                    We couldn&apos;t find a face in your photo. Try a clear, well-lit front-facing photo with your full face visible.
                  </p>
                </div>
              </>
            ) : (
              <>
                <h1 className="text-2xl font-black">Analysis failed</h1>
                <p className="text-neutral-500 text-base">{error}</p>
              </>
            )}
            <Link
              href="/onboarding"
              className="px-8 py-4 bg-black text-white rounded-xl font-semibold text-base hover:bg-neutral-800 transition-colors inline-block"
            >
              Try Again
            </Link>
          </div>
        </div>
      </div>
    );
  }

  // ── Normal processing UI ──────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-white text-black flex flex-col">
      <Nav showFullLinks={false} />

      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-2xl flex flex-col items-center gap-8 text-center">
          <div>
            <h1 className="text-[42px] font-black leading-tight tracking-tight mb-3">
              Analyzing your features…
            </h1>
            <p className="text-base text-neutral-500">
              This usually takes 5–10 seconds. Please keep the app open.
            </p>
          </div>

          {/* Progress bar */}
          <div className="w-full h-1 bg-neutral-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-black rounded-full transition-all duration-100"
              style={{ width: `${progress}%` }}
            />
          </div>

          {/* Step cards */}
          <div className="w-full grid grid-cols-4 gap-3">
            {ANALYSIS_STEPS.map((s, i) => {
              const done   = i < activeStep;
              const active = i === activeStep;
              return (
                <div
                  key={s.number}
                  className={`rounded-2xl p-4 text-left transition-all ${
                    done   ? "bg-black text-white"
                    : active ? "bg-neutral-100 text-black"
                    :          "bg-neutral-50 text-neutral-400"
                  }`}
                >
                  <div className="flex items-center gap-1.5 mb-3">
                    {done ? (
                      <svg viewBox="0 0 16 16" fill="none" className="w-4 h-4 text-white" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M3 8l4 4 6-6" />
                      </svg>
                    ) : (
                      <span className={`text-sm font-bold ${active ? "text-black" : "text-neutral-400"}`}>
                        {s.number}
                      </span>
                    )}
                  </div>
                  <div className={`text-sm font-semibold leading-tight mb-1 ${done ? "text-white" : active ? "text-black" : "text-neutral-400"}`}>
                    {s.title}
                  </div>
                  <div className={`text-xs leading-snug ${done ? "text-white/60" : active ? "text-neutral-500" : "text-neutral-300"}`}>
                    {s.subtitle}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Animated status */}
          <p className="text-sm text-neutral-500">
            {stepLabel}{" "}
            <span className="tracking-widest">
              {DOTS.slice(0, dotCount).join(" ")}
            </span>
          </p>
        </div>
      </div>
    </div>
  );
}
