"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/* ── Status ──────────────────────────────────────────────────────────────── */

type Status =
  | "loading"
  | "no-face"
  | "too-far"
  | "too-close"
  | "off-center"
  | "too-dark"
  | "too-bright"
  | "ready";

interface StatusConfig {
  text: string;
  sub: string;
  color: string;
  border: string;
}

const STATUS: Record<Status, StatusConfig> = {
  loading:      { text: "Starting camera…",              sub: "Please wait",                                    color: "#9ca3af", border: "#9ca3af" },
  "no-face":    { text: "No face detected",              sub: "Position your face inside the oval",             color: "#ef4444", border: "#ef4444" },
  "too-far":    { text: "Move closer",                   sub: "Your face is too small in the frame",            color: "#f97316", border: "#f97316" },
  "too-close":  { text: "Move back a little",            sub: "Your face is too close to the camera",           color: "#f97316", border: "#f97316" },
  "off-center": { text: "Center your face",              sub: "Align your face with the oval guide",            color: "#f97316", border: "#f97316" },
  "too-dark":   { text: "Lighting too low",              sub: "Move to a brighter area or face a window",       color: "#eab308", border: "#eab308" },
  "too-bright": { text: "Too much light",                sub: "Avoid bright light directly behind you",         color: "#eab308", border: "#eab308" },
  ready:        { text: "Ready to take photo",           sub: "Tap the button below to capture",                color: "#22c55e", border: "#22c55e" },
};

/* ── Condition icons ─────────────────────────────────────────────────────── */

function CheckCircle({ pass }: { pass: boolean | null }) {
  if (pass === null) return <span className="w-4 h-4 rounded-full bg-neutral-200 inline-block" />;
  return pass ? (
    <svg viewBox="0 0 16 16" className="w-4 h-4 text-green-500 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="8" cy="8" r="7" />
      <path d="M5 8l2.5 2.5L11 5.5" />
    </svg>
  ) : (
    <svg viewBox="0 0 16 16" className="w-4 h-4 text-red-400 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="8" cy="8" r="7" />
      <path d="M5.5 5.5l5 5M10.5 5.5l-5 5" />
    </svg>
  );
}

/* ── Props ───────────────────────────────────────────────────────────────── */

interface Props {
  onCapture: (file: File, preview: string) => void;
  onError: (reason: "permission" | "no-camera" | "unknown") => void;
}

/* ── Overlay renderer ────────────────────────────────────────────────────── */

function drawOverlay(canvas: HTMLCanvasElement, W: number, H: number, status: Status) {
  canvas.width  = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2, cy = H / 2;
  const rx = W * 0.36, ry = H * 0.45;

  // Dark backdrop
  ctx.fillStyle = "rgba(0,0,0,0.50)";
  ctx.fillRect(0, 0, W, H);

  // Punch oval cutout
  ctx.globalCompositeOperation = "destination-out";
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalCompositeOperation = "source-over";

  // Oval ring
  const borderColor = STATUS[status].border;
  ctx.strokeStyle = borderColor + (status === "ready" ? "ff" : "99");
  ctx.lineWidth   = status === "ready" ? 3.5 : 2.5;
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.stroke();

  // Ready glow
  if (status === "ready") {
    ctx.shadowColor = "#22c55e";
    ctx.shadowBlur  = 14;
    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth   = 2;
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // Corner ticks
  const tick  = W * 0.045;
  const ticks: [number, number, number][] = [
    [cx - rx * 0.55, cy - ry,  1],
    [cx + rx * 0.55, cy - ry, -1],
    [cx - rx * 0.55, cy + ry,  1],
    [cx + rx * 0.55, cy + ry, -1],
  ];
  ctx.strokeStyle = borderColor;
  ctx.lineWidth   = 3;
  ctx.lineCap     = "round";
  ticks.forEach(([x, y, dx]) => {
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + dx * tick, y);
    ctx.stroke();
  });
}

/* ── Component ───────────────────────────────────────────────────────────── */

export default function FaceCapture({ onCapture, onError }: Props) {
  const videoRef    = useRef<HTMLVideoElement>(null);
  const analysisRef = useRef<HTMLCanvasElement>(null);
  const overlayRef  = useRef<HTMLCanvasElement>(null);
  const streamRef   = useRef<MediaStream | null>(null);
  const detectorRef = useRef<any>(null);
  const rafRef      = useRef<number>(0);
  const doneRef     = useRef(false);

  const [status, setStatus] = useState<Status>("loading");

  // Derived condition booleans (null = not yet assessed)
  const [conditions, setConditions] = useState<{
    lighting: boolean | null;
    face: boolean | null;
    centered: boolean | null;
    distance: boolean | null;
  }>({ lighting: null, face: null, centered: null, distance: null });

  /* ── Capture ──────────────────────────────────────────────────────────── */
  const capture = useCallback(() => {
    if (doneRef.current || status !== "ready") return;
    doneRef.current = true;
    cancelAnimationFrame(rafRef.current);

    const video = videoRef.current!;
    const W = video.videoWidth  || 640;
    const H = video.videoHeight || 640;

    const snap = document.createElement("canvas");
    snap.width  = W;
    snap.height = H;
    const ctx = snap.getContext("2d")!;
    ctx.translate(W, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0);

    snap.toBlob(
      (blob) => {
        if (!blob) return;
        const file    = new File([blob], "selfie.jpg", { type: "image/jpeg" });
        const preview = URL.createObjectURL(blob);
        streamRef.current?.getTracks().forEach((t) => t.stop());
        onCapture(file, preview);
      },
      "image/jpeg",
      0.92,
    );
  }, [onCapture, status]);

  /* ── Per-frame analysis ───────────────────────────────────────────────── */
  const analyze = useCallback(async () => {
    const video    = videoRef.current;
    const analysis = analysisRef.current;
    const overlay  = overlayRef.current;

    if (!video || !analysis || !overlay || video.readyState < 2 || doneRef.current) {
      rafRef.current = requestAnimationFrame(analyze);
      return;
    }

    const W = video.videoWidth  || 640;
    const H = video.videoHeight || 640;

    analysis.width  = W;
    analysis.height = H;
    const ctx = analysis.getContext("2d")!;
    ctx.save();
    ctx.translate(W, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, W, H);
    ctx.restore();

    // ── Lighting check ────────────────────────────────────────────────────
    const cx = W / 2, cy = H / 2;
    const srx = Math.floor(W * 0.3), sry = Math.floor(H * 0.38);
    const data = ctx.getImageData(cx - srx, cy - sry, srx * 2, sry * 2).data;
    let lum = 0;
    for (let i = 0; i < data.length; i += 4)
      lum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    lum /= data.length / 4;

    const lightingOk = lum >= 40 && lum <= 210;

    if (!lightingOk) {
      const next: Status = lum < 40 ? "too-dark" : "too-bright";
      setStatus(next);
      setConditions({ lighting: false, face: null, centered: null, distance: null });
      drawOverlay(overlay, W, H, next);
      rafRef.current = requestAnimationFrame(analyze);
      return;
    }

    // ── Face detection ────────────────────────────────────────────────────
    let next: Status = "no-face";
    let faceOk = false, centeredOk = false, distanceOk = false;

    if (detectorRef.current) {
      try {
        const faces = await detectorRef.current.detect(video);
        if (faces.length > 0) {
          faceOk = true;
          const b      = faces[0].boundingBox;
          const faceCx = (b.x + b.width  / 2) / W;
          const faceCy = (b.y + b.height / 2) / H;
          const faceW  = b.width / W;

          distanceOk = faceW >= 0.22 && faceW <= 0.62;
          centeredOk = Math.abs(faceCx - 0.5) <= 0.13 && Math.abs(faceCy - 0.5) <= 0.15;

          if (!distanceOk) {
            next = faceW < 0.22 ? "too-far" : "too-close";
          } else if (!centeredOk) {
            next = "off-center";
          } else {
            next = "ready";
          }
        }
      } catch {
        // FaceDetector failed — use brightness proxy
        faceOk = centeredOk = distanceOk = true;
        next = "ready";
      }
    } else {
      // No FaceDetector API — brightness proxy only
      faceOk = centeredOk = distanceOk = true;
      next = "ready";
    }

    setStatus(next);
    setConditions({ lighting: lightingOk, face: faceOk, centered: centeredOk, distance: distanceOk });
    drawOverlay(overlay, W, H, next);

    rafRef.current = requestAnimationFrame(analyze);
  }, []);

  /* ── Start camera ─────────────────────────────────────────────────────── */
  useEffect(() => {
    async function start() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } },
          audio: false,
        });
        streamRef.current = stream;
        const video = videoRef.current!;
        video.srcObject = stream;
        await video.play();

        if ("FaceDetector" in window) {
          detectorRef.current = new (window as any).FaceDetector({ fastMode: true, maxDetectedFaces: 1 });
        }

        setStatus("no-face");
        rafRef.current = requestAnimationFrame(analyze);
      } catch (err: unknown) {
        const name = (err as DOMException)?.name ?? "";
        onError(
          name === "NotAllowedError" ? "permission" :
          name === "NotFoundError" || name === "DevicesNotFoundError" ? "no-camera" :
          "unknown"
        );
      }
    }
    start();
    return () => {
      cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [analyze, onError]);

  const isReady = status === "ready";

  /* ── Render ───────────────────────────────────────────────────────────── */
  return (
    <div className="flex flex-col items-center gap-4 w-full">

      {/* Camera viewport */}
      <div className="relative w-full max-w-xs aspect-[3/4] rounded-3xl overflow-hidden bg-black mx-auto">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="absolute inset-0 w-full h-full object-cover"
          style={{ transform: "scaleX(-1)" }}
        />
        <canvas ref={overlayRef} className="absolute inset-0 w-full h-full pointer-events-none" />

        {/* Status badge */}
        <div className="absolute bottom-4 left-3 right-3 flex justify-center pointer-events-none">
          <span
            className="px-4 py-2 rounded-full text-white text-xs font-semibold shadow-lg text-center"
            style={{ backgroundColor: STATUS[status].color + "dd" }}
          >
            {STATUS[status].text}
          </span>
        </div>

        {/* Loading spinner */}
        {status === "loading" && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60">
            <div className="w-8 h-8 border-2 border-white/20 border-t-white rounded-full animate-spin" />
          </div>
        )}

        <canvas ref={analysisRef} className="hidden" />
      </div>

      {/* Real-time condition checklist */}
      {status !== "loading" && (
        <div className="w-full max-w-xs bg-neutral-50 border border-black/8 rounded-2xl px-4 py-3 flex flex-col gap-2">
          <div className="flex items-center justify-between text-xs text-neutral-500">
            <span className="flex items-center gap-2">
              <CheckCircle pass={conditions.lighting} />
              Good lighting
            </span>
            <span className="flex items-center gap-2">
              <CheckCircle pass={conditions.face} />
              Face detected
            </span>
          </div>
          <div className="flex items-center justify-between text-xs text-neutral-500">
            <span className="flex items-center gap-2">
              <CheckCircle pass={conditions.centered} />
              Face centered
            </span>
            <span className="flex items-center gap-2">
              <CheckCircle pass={conditions.distance} />
              Good distance
            </span>
          </div>
          {status !== "ready" && (
            <p className="text-xs text-neutral-400 text-center pt-1 border-t border-black/6 mt-1">
              {STATUS[status].sub}
            </p>
          )}
        </div>
      )}

      {/* Capture button — only enabled when ready */}
      <button
        onClick={capture}
        disabled={!isReady}
        className={`w-full max-w-xs py-4 rounded-2xl font-semibold text-base transition-all ${
          isReady
            ? "bg-black text-white hover:bg-neutral-800 active:scale-95 shadow-lg"
            : "bg-neutral-100 text-neutral-400 cursor-not-allowed"
        }`}
      >
        {isReady ? "Take Photo" : "Waiting for cues…"}
      </button>
    </div>
  );
}
