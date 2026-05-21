"use client";

import { Suspense, useState, useRef, useCallback, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Nav from "@/components/Nav";
import FaceCapture from "@/components/FaceCapture";

/* ── Step / photo-mode types ─────────────────────────────── */
type Step = "name" | "style" | "age" | "photo";
type PhotoMode = "select" | "camera" | "upload";
type CameraError = "permission" | "no-camera" | "unknown" | null;
const STEPS: Step[] = ["name", "style", "age", "photo"];
const STEP_LABELS: Record<Step, string> = {
  name: "Name",
  style: "Style",
  age: "Age",
  photo: "Photo",
};

const STYLE_OPTIONS = [
  { id: "casual", label: "Casual", subtitle: "Relaxed everyday looks", icon: <ShirtIcon /> },
  { id: "streetwear", label: "Streetwear", subtitle: "Urban, bold", icon: <CapIcon /> },
  { id: "smart_casual", label: "Smart Casual", subtitle: "Polished yet relaxed", icon: <JacketIcon /> },
  { id: "retro", label: "Retro", subtitle: "Vintage-inspired looks", icon: <RetroIcon /> },
  { id: "classic", label: "Classic", subtitle: "Timeless elegance", icon: <ClassicIcon /> },
  { id: "formal", label: "Formal", subtitle: "Professional attire", icon: <TieIcon /> },
];

const AGE_OPTIONS = ["Under 18", "18-24", "25-34", "35-44", "45+"];

/* ── Icon components ─────────────────────────────────────── */
function ShirtIcon() {
  return (
    <svg viewBox="0 0 28 28" fill="none" className="w-7 h-7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 8l5-4h10l5 4-4 3v13H8V11L4 8z" />
    </svg>
  );
}
function CapIcon() {
  return (
    <svg viewBox="0 0 28 28" fill="none" className="w-7 h-7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
      <path d="M5 14a9 9 0 0118 0" />
      <rect x="4" y="14" width="20" height="5" rx="2" />
    </svg>
  );
}
function JacketIcon() {
  return (
    <svg viewBox="0 0 28 28" fill="none" className="w-7 h-7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 6l4-2 5 5 5-5 4 2v18H5V6z" />
      <path d="M14 9v15" />
    </svg>
  );
}
function RetroIcon() {
  return (
    <svg viewBox="0 0 28 28" fill="none" className="w-7 h-7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
      <circle cx="14" cy="14" r="10" />
      <path d="M14 8v6l4 3" />
    </svg>
  );
}
function ClassicIcon() {
  return (
    <svg viewBox="0 0 28 28" fill="none" className="w-7 h-7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 24V8l9-4 9 4v16" />
      <rect x="10" y="15" width="8" height="9" />
    </svg>
  );
}
function TieIcon() {
  return (
    <svg viewBox="0 0 28 28" fill="none" className="w-7 h-7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 4h6l-2 8 4 12H9L13 12 11 4z" />
    </svg>
  );
}
function CheckIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 8l4 4 6-6" />
    </svg>
  );
}
function ArrowIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 8h10M9 4l4 4-4 4" />
    </svg>
  );
}
function CameraIcon() {
  return (
    <svg viewBox="0 0 28 28" fill="none" className="w-5 h-5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 9a2 2 0 012-2h2l2-3h6l2 3h2a2 2 0 012 2v12a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
      <circle cx="14" cy="15" r="4" />
    </svg>
  );
}
function UploadIcon() {
  return (
    <svg viewBox="0 0 28 28" fill="none" className="w-5 h-5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 18v4a2 2 0 002 2h16a2 2 0 002-2v-4" />
      <path d="M18 10l-4-4-4 4M14 6v14" />
    </svg>
  );
}
function ShieldIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 2l5 2v5c0 3-2.5 5-5 6C5.5 14 3 12 3 9V4L8 2z" />
    </svg>
  );
}

/* ── Progress stepper ────────────────────────────────────── */
function ProgressStepper({ currentStep, completedSteps }: { currentStep: Step; completedSteps: Set<Step> }) {
  return (
    <div className="flex items-center justify-center gap-1">
      {STEPS.map((step) => {
        const done = completedSteps.has(step);
        const active = step === currentStep;
        return (
          <div
            key={step}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-medium transition-all ${
              active
                ? "bg-black text-white"
                : done
                ? "bg-neutral-100 text-neutral-600"
                : "text-neutral-400"
            }`}
          >
            {done && <CheckIcon />}
            <span>{STEP_LABELS[step]}</span>
          </div>
        );
      })}
    </div>
  );
}

/* ── Resize + encode photo for sessionStorage ────────────────────────────── */
async function resizeAndEncode(file: File, maxPx = 1024): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      const ratio = Math.min(maxPx / img.width, maxPx / img.height, 1);
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(img.width * ratio);
      canvas.height = Math.round(img.height * ratio);
      canvas.getContext("2d")!.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      resolve(canvas.toDataURL("image/jpeg", 0.85));
    };
    img.src = url;
  });
}

/* ── Main page ───────────────────────────────────────────── */
function OnboardingInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [step, setStep] = useState<Step>("name");
  const [completed, setCompleted] = useState<Set<Step>>(new Set());

  const [name, setName] = useState("");
  const [style, setStyle] = useState<string[]>([]);
  const [age, setAge] = useState("");
  const [photo, setPhoto] = useState<File | null>(null);
  const [photoPreview, setPhotoPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const [photoMode, setPhotoMode] = useState<PhotoMode>("select");
  const [cameraError, setCameraError] = useState<CameraError>(null);
  const [photoError, setPhotoError] = useState<string | null>(null);

  // Jump to photo step if returning from results with existing profile
  useEffect(() => {
    if (searchParams.get("step") === "photo") {
      try {
        const saved = localStorage.getItem("tonefit_profile");
        if (saved) {
          const profile = JSON.parse(saved);
          if (profile.name)  setName(profile.name);
          if (profile.style) setStyle(profile.style);
          if (profile.age)   setAge(profile.age);
        }
      } catch {}
      setCompleted(new Set<Step>(["name", "style", "age"]));
      setStep("photo");
    }
  }, [searchParams]);

  const advance = (from: Step, to: Step) => {
    setCompleted((prev) => new Set([...prev, from]));
    setStep(to);
  };

  const handlePhotoFile = useCallback((file: File) => {
    setPhoto(file);
    setPhotoPreview(URL.createObjectURL(file));
    setPhotoError(null);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) handlePhotoFile(file);
    },
    [handlePhotoFile]
  );

  const handleAnalyze = async () => {
    if (!photo || preparing) return;
    setPreparing(true);
    setPhotoError(null);

    // Quick face check before navigating to processing
    try {
      const form = new FormData();
      form.append("file", photo, "photo.jpg");
      const res = await fetch("/api/check-face", { method: "POST", body: form });
      if (res.ok) {
        const { face_detected } = await res.json();
        if (!face_detected) {
          setPhotoError("no_face");
          setPreparing(false);
          return;
        }
      }
      // If check-face fails (server down etc.) proceed anyway — processing will catch it
    } catch {
      // proceed
    }

    localStorage.setItem("tonefit_profile", JSON.stringify({ name, style, age }));

    const base64 = await resizeAndEncode(photo);
    sessionStorage.setItem("tonefit_photo", base64);

    router.push("/processing");
  };

  return (
    <div className="min-h-screen bg-white text-black flex flex-col">
      <Nav showFullLinks={false} />

      <div className="flex-1 flex flex-col">
        {/* Stepper */}
        <div className="py-5 px-14 border-b border-black/8">
          <ProgressStepper currentStep={step} completedSteps={completed} />
        </div>

        {/* Step content */}
        <div className="flex-1 flex items-center justify-center p-8">
          {/* ── Step 1: Name ───────────────────────────── */}
          {step === "name" && (
            <div className="w-full max-w-xl flex flex-col items-center text-center gap-8">
              <h1 className="text-[40px] font-black leading-tight tracking-tight">
                What should we call you?
              </h1>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Enter your first name"
                onKeyDown={(e) => e.key === "Enter" && name.trim() && advance("name", "style")}
                className="w-full px-6 py-4 border-2 border-black/12 rounded-xl text-lg focus:outline-none focus:border-black transition-colors"
                autoFocus
              />
              <button
                onClick={() => advance("name", "style")}
                disabled={!name.trim()}
                className="flex items-center gap-2 px-8 py-4 bg-black text-white rounded-xl font-semibold text-base hover:bg-neutral-800 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
              >
                Next <ArrowIcon />
              </button>
            </div>
          )}

          {/* ── Step 2: Style ──────────────────────────── */}
          {step === "style" && (
            <div className="w-full max-w-2xl flex flex-col gap-8">
              <h1 className="text-[40px] font-black leading-tight tracking-tight text-center">
                What&apos;s your style?
              </h1>
              <div className="grid grid-cols-2 gap-4">
                {STYLE_OPTIONS.map((opt) => {
                  const selected = style.includes(opt.id);
                  return (
                    <button
                      key={opt.id}
                      onClick={() =>
                        setStyle((prev) =>
                          prev.includes(opt.id)
                            ? prev.filter((s) => s !== opt.id)
                            : [...prev, opt.id]
                        )
                      }
                      className={`flex items-center gap-4 p-6 rounded-2xl border-2 text-left transition-all ${
                        selected
                          ? "border-black bg-black text-white"
                          : "border-black/10 bg-white hover:border-black/30"
                      }`}
                    >
                      <div className={`flex-shrink-0 ${selected ? "text-white" : "text-neutral-500"}`}>
                        {opt.icon}
                      </div>
                      <div>
                        <div className="font-bold text-base leading-tight">{opt.label}</div>
                        <div className={`text-sm mt-0.5 ${selected ? "text-white/70" : "text-neutral-500"}`}>
                          {opt.subtitle}
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
              <div className="flex justify-center">
                <button
                  onClick={() => advance("style", "age")}
                  disabled={style.length === 0}
                  className="flex items-center gap-2 px-8 py-4 bg-black text-white rounded-xl font-semibold text-base hover:bg-neutral-800 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  Next <ArrowIcon />
                </button>
              </div>
            </div>
          )}

          {/* ── Step 3: Age ────────────────────────────── */}
          {step === "age" && (
            <div className="w-full max-w-sm flex flex-col items-center gap-8">
              <h1 className="text-[40px] font-black leading-tight tracking-tight text-center">
                How old are you?
              </h1>
              <div className="w-full flex flex-col gap-3">
                {AGE_OPTIONS.map((opt) => (
                  <button
                    key={opt}
                    onClick={() => {
                      setAge(opt);
                      advance("age", "photo");
                    }}
                    className={`w-full py-4 rounded-xl border-2 text-base font-semibold transition-all ${
                      age === opt
                        ? "border-black bg-black text-white"
                        : "border-black/10 hover:border-black/30 text-black"
                    }`}
                  >
                    {opt}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* ── Step 4: Photo ──────────────────────────── */}
          {step === "photo" && (
            <div className="w-full max-w-xs flex flex-col items-center gap-5">
              <div className="text-center">
                <h1 className="text-[32px] font-black leading-tight tracking-tight mb-1">
                  {photoPreview ? "Looking good!" : "Add your photo"}
                </h1>
                <p className="text-sm text-neutral-500">
                  {photoPreview
                    ? "Ready to analyze your color season."
                    : "Choose how you'd like to provide a photo."}
                </p>
              </div>

              {photoPreview ? (
                /* ── Captured / uploaded preview ── */
                <div className="w-full flex flex-col items-center gap-4">
                  <div className="w-full aspect-[3/4] rounded-3xl overflow-hidden bg-neutral-100 relative">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={photoPreview} alt="Preview" className="w-full h-full object-cover" />
                    {photoError === "no_face" && (
                      <div className="absolute inset-0 bg-black/50 flex items-end p-4">
                        <div className="w-full rounded-2xl bg-white p-4 flex flex-col gap-2">
                          <div className="flex items-center gap-2">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" className="w-5 h-5 text-neutral-500 shrink-0">
                              <circle cx="12" cy="8" r="4" />
                              <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" />
                              <line x1="4" y1="4" x2="20" y2="20" />
                            </svg>
                            <p className="text-sm font-bold text-black">No face detected</p>
                          </div>
                          <p className="text-xs text-neutral-500 leading-relaxed">
                            Try a clear, well-lit front-facing photo with your full face visible.
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                  {photoError === "no_face" ? (
                    <button
                      onClick={() => {
                        setPhoto(null);
                        setPhotoPreview(null);
                        setPhotoError(null);
                        setPhotoMode("upload");
                      }}
                      className="w-full py-3 bg-black text-white rounded-xl font-semibold text-sm hover:bg-neutral-800 transition-colors"
                    >
                      Upload a Different Photo
                    </button>
                  ) : (
                    <button
                      onClick={() => {
                        setPhoto(null);
                        setPhotoPreview(null);
                        setPhotoMode("select");
                        setCameraError(null);
                      }}
                      className="text-sm text-neutral-500 underline underline-offset-2 hover:text-black transition-colors"
                    >
                      Use a different photo
                    </button>
                  )}
                </div>

              ) : photoMode === "select" ? (
                /* ── Method selection ── */
                <div className="w-full flex flex-col gap-3">
                  <button
                    onClick={() => { setCameraError(null); setPhotoMode("camera"); }}
                    className="w-full flex items-center gap-5 p-5 rounded-2xl border-2 border-black/10 hover:border-black/40 transition-all text-left"
                  >
                    <div className="w-11 h-11 rounded-xl bg-neutral-100 flex items-center justify-center shrink-0">
                      <CameraIcon />
                    </div>
                    <div>
                      <p className="font-bold text-sm text-black">Take a Photo</p>
                      <p className="text-xs text-neutral-500 mt-0.5">Use your device camera for a live selfie</p>
                    </div>
                  </button>
                  <button
                    onClick={() => setPhotoMode("upload")}
                    className="w-full flex items-center gap-5 p-5 rounded-2xl border-2 border-black/10 hover:border-black/40 transition-all text-left"
                  >
                    <div className="w-11 h-11 rounded-xl bg-neutral-100 flex items-center justify-center shrink-0">
                      <UploadIcon />
                    </div>
                    <div>
                      <p className="font-bold text-sm text-black">Upload an Image</p>
                      <p className="text-xs text-neutral-500 mt-0.5">Choose a photo from your device</p>
                    </div>
                  </button>
                </div>

              ) : photoMode === "camera" ? (
                /* ── Live camera or camera error ── */
                cameraError ? (
                  <div className="w-full flex flex-col items-center gap-4">
                    <div className="w-full rounded-2xl bg-neutral-50 border border-black/8 p-6 flex flex-col items-center gap-3 text-center">
                      <div className="w-12 h-12 rounded-full bg-neutral-200 flex items-center justify-center">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" className="w-6 h-6 text-neutral-500">
                          <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z" />
                          <line x1="1" y1="1" x2="23" y2="23" />
                        </svg>
                      </div>
                      <div>
                        <p className="text-sm font-bold text-black mb-1">
                          {cameraError === "permission"
                            ? "Camera permission denied"
                            : cameraError === "no-camera"
                            ? "No camera found"
                            : "Camera unavailable"}
                        </p>
                        <p className="text-xs text-neutral-500 leading-relaxed">
                          {cameraError === "permission"
                            ? "Allow camera access in your browser settings and try again."
                            : cameraError === "no-camera"
                            ? "No camera device was detected on this device."
                            : "Could not start the camera. Please try again or upload a photo."}
                        </p>
                      </div>
                    </div>
                    <div className="w-full flex flex-col gap-2">
                      <button
                        onClick={() => { setCameraError(null); setPhotoMode("camera"); }}
                        className="w-full py-3 border-2 border-black/12 rounded-xl font-semibold text-sm hover:border-black/30 transition-colors"
                      >
                        Try Again
                      </button>
                      <button
                        onClick={() => { setCameraError(null); setPhotoMode("upload"); }}
                        className="w-full py-3 bg-black text-white rounded-xl font-semibold text-sm hover:bg-neutral-800 transition-colors"
                      >
                        Upload a Photo Instead
                      </button>
                    </div>
                    <button
                      onClick={() => { setCameraError(null); setPhotoMode("select"); }}
                      className="text-xs text-neutral-400 underline underline-offset-2 hover:text-neutral-600 transition-colors"
                    >
                      Go back
                    </button>
                  </div>
                ) : (
                  <div className="w-full flex flex-col items-center gap-4">
                    <FaceCapture
                      onCapture={(file, preview) => {
                        setPhoto(file);
                        setPhotoPreview(preview);
                      }}
                      onError={(reason) => setCameraError(reason)}
                    />
                    <button
                      onClick={() => setPhotoMode("select")}
                      className="text-xs text-neutral-400 underline underline-offset-2 hover:text-neutral-600 transition-colors"
                    >
                      Switch to upload instead
                    </button>
                  </div>
                )

              ) : (
                /* ── Upload ── */
                <div className="w-full flex flex-col items-center gap-4">
                  <div
                    onDrop={handleDrop}
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    onClick={() => fileInputRef.current?.click()}
                    className={`w-full aspect-[4/3] rounded-3xl border-2 border-dashed flex items-center justify-center cursor-pointer transition-all ${
                      dragOver ? "border-black bg-neutral-50" : "border-black/20 hover:border-black/40"
                    }`}
                  >
                    <div className="flex flex-col items-center gap-3 text-neutral-400 p-8 text-center">
                      <svg viewBox="0 0 48 48" fill="none" className="w-12 h-12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                        <rect x="4" y="12" width="40" height="30" rx="4" />
                        <circle cx="24" cy="27" r="8" />
                        <path d="M18 12l3-6h6l3 6" />
                      </svg>
                      <p className="text-sm font-medium">Click or drop a photo here</p>
                    </div>
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handlePhotoFile(file);
                    }}
                  />
                  <button
                    onClick={() => setPhotoMode("select")}
                    className="text-xs text-neutral-400 underline underline-offset-2 hover:text-neutral-600 transition-colors"
                  >
                    Go back
                  </button>
                </div>
              )}

              <p className="flex items-center gap-1.5 text-xs text-neutral-400">
                <ShieldIcon /> Your photo is never stored or shared
              </p>

              {photo && !photoError && (
                <button
                  onClick={handleAnalyze}
                  disabled={preparing}
                  className="w-full py-4 bg-black text-white rounded-xl font-semibold text-base hover:bg-neutral-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {preparing ? "Checking photo…" : "Analyze My Color Season"}
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function OnboardingPage() {
  return (
    <Suspense>
      <OnboardingInner />
    </Suspense>
  );
}
