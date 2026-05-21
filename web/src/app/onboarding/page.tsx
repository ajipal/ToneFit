"use client";

import { useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import Nav from "@/components/Nav";
import FaceCapture from "@/components/FaceCapture";
import PhotoCropper from "@/components/PhotoCropper";

/* ── Step types ─────────────────────────────────────────── */
type Step = "name" | "style" | "age" | "photo";
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
export default function OnboardingPage() {
  const router = useRouter();
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
  const [photoMode, setPhotoMode] = useState<"camera" | "upload" | "crop">("camera");
  const [rawImageSrc, setRawImageSrc] = useState<string | null>(null);

  const advance = (from: Step, to: Step) => {
    setCompleted((prev) => new Set([...prev, from]));
    setStep(to);
  };

  const handlePhotoFile = useCallback((file: File) => {
    const url = URL.createObjectURL(file);
    setRawImageSrc(url);
    setPhotoMode("crop");
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
                  {photoPreview ? "Looking good!" : "Take a selfie"}
                </h1>
                <p className="text-sm text-neutral-500">
                  {photoPreview
                    ? "Ready to analyze your color season."
                    : "Align your face in the oval and hold still."}
                </p>
              </div>

              {/* Camera capture or preview */}
              {photoPreview ? (
                /* ── Captured preview ── */
                <div className="w-full flex flex-col items-center gap-4">
                  <div className="w-full aspect-[3/4] rounded-3xl overflow-hidden bg-neutral-100">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={photoPreview} alt="Preview" className="w-full h-full object-cover" />
                  </div>
                  <button
                    onClick={() => {
                      setPhoto(null);
                      setPhotoPreview(null);
                      setRawImageSrc(null);
                      setPhotoMode("camera");
                    }}
                    className="text-sm text-neutral-500 underline underline-offset-2 hover:text-black transition-colors"
                  >
                    Retake photo
                  </button>
                </div>
              ) : photoMode === "crop" && rawImageSrc ? (
                /* ── Crop uploaded image ── */
                <PhotoCropper
                  imageSrc={rawImageSrc}
                  onCrop={(file, preview) => {
                    setPhoto(file);
                    setPhotoPreview(preview);
                    setPhotoMode("upload");
                  }}
                  onCancel={() => {
                    setRawImageSrc(null);
                    setPhotoMode("upload");
                  }}
                />
              ) : photoMode === "camera" ? (
                /* ── Live camera ── */
                <FaceCapture
                  onCapture={(file, preview) => {
                    setPhoto(file);
                    setPhotoPreview(preview);
                  }}
                  onError={() => setPhotoMode("upload")}
                />
              ) : (
                /* ── Upload fallback ── */
                <div className="w-full flex flex-col items-center gap-4">
                  <div
                    onDrop={handleDrop}
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    className={`w-full aspect-[4/3] rounded-3xl border-2 border-dashed flex items-center justify-center transition-all ${
                      dragOver ? "border-black bg-neutral-50" : "border-black/20"
                    }`}
                  >
                    <div className="flex flex-col items-center gap-3 text-neutral-400 p-8 text-center">
                      <svg viewBox="0 0 48 48" fill="none" className="w-12 h-12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                        <rect x="4" y="12" width="40" height="30" rx="4" />
                        <circle cx="24" cy="27" r="8" />
                        <path d="M18 12l3-6h6l3 6" />
                      </svg>
                      <p className="text-sm font-medium">Drop your photo here</p>
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
                    onClick={() => fileInputRef.current?.click()}
                    className="flex items-center gap-2 px-6 py-3 border-2 border-black/12 rounded-xl font-semibold text-sm hover:border-black/30 transition-colors"
                  >
                    <UploadIcon /> Choose a Photo
                  </button>
                </div>
              )}

              <p className="flex items-center gap-1.5 text-xs text-neutral-400">
                <ShieldIcon /> Your photo is never stored or shared
              </p>

              {photo && (
                <button
                  onClick={handleAnalyze}
                  disabled={preparing}
                  className="w-full py-4 bg-black text-white rounded-xl font-semibold text-base hover:bg-neutral-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {preparing ? "Preparing…" : "Analyze My Color Season"}
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
