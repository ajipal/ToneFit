"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/* ── Status types & messages ─────────────────────────────────────────────── */

type Status =
  | "loading"
  | "no-face"
  | "too-far"
  | "too-close"
  | "off-center"
  | "too-dark"
  | "too-bright"
  | "ready";

const MSG: Record<Status, { text: string; color: string }> = {
  loading:       { text: "Starting camera…",            color: "#9ca3af" },
  "no-face":     { text: "Place your face in the oval", color: "#ef4444" },
  "too-far":     { text: "Move closer",                 color: "#f97316" },
  "too-close":   { text: "Move back a little",          color: "#f97316" },
  "off-center":  { text: "Center your face",            color: "#f97316" },
  "too-dark":    { text: "Find better lighting",        color: "#eab308" },
  "too-bright":  { text: "Too much light behind you",   color: "#eab308" },
  ready:         { text: "Hold still…",                 color: "#22c55e" },
};

const READY_FRAMES = 55; // ~1.8 s at 30 fps before auto-capture

/* ── Props ───────────────────────────────────────────────────────────────── */

interface Props {
  onCapture: (file: File, preview: string) => void;
  onError: (reason: "permission" | "no-camera" | "unknown") => void;
}

/* ── Oval overlay renderer ───────────────────────────────────────────────── */

function drawOverlay(canvas: HTMLCanvasElement, W: number, H: number, status: Status, progress: number) {
  canvas.width  = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2;
  const cy = H / 2;
  const rx = W * 0.36;
  const ry = H * 0.45;

  // Dark backdrop
  ctx.fillStyle = "rgba(0,0,0,0.52)";
  ctx.fillRect(0, 0, W, H);

  // Punch oval cutout through backdrop
  ctx.globalCompositeOperation = "destination-out";
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalCompositeOperation = "source-over";

  // Oval border color
  const borderColor =
    status === "ready"                                   ? "#22c55e" :
    status === "loading" || status === "no-face"         ? "#ef4444" :
                                                           "#f97316";

  // Base oval ring
  ctx.strokeStyle = borderColor + "88";
  ctx.lineWidth   = 2.5;
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.stroke();

  // Auto-capture progress arc (draws on top)
  if (progress > 0) {
    const pct   = progress / 100;
    const start = -Math.PI / 2;
    const end   = start + pct * Math.PI * 2;
    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth   = 3.5;
    ctx.lineCap     = "round";
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, start, end);
    ctx.stroke();
  }

  // Corner guide ticks
  const tick = W * 0.045;
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
  const readyRef    = useRef(0);
  const doneRef     = useRef(false);

  const [status,   setStatus]   = useState<Status>("loading");
  const [progress, setProgress] = useState(0);

  /* ── Capture a still ──────────────────────────────────────────────────── */
  const capture = useCallback(() => {
    if (doneRef.current) return;
    doneRef.current = true;
    cancelAnimationFrame(rafRef.current);

    const video = videoRef.current!;
    const W = video.videoWidth  || 640;
    const H = video.videoHeight || 640;

    const snap = document.createElement("canvas");
    snap.width  = W;
    snap.height = H;
    const ctx = snap.getContext("2d")!;
    // Mirror to match what the user sees on screen
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
  }, [onCapture]);

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

    // Draw mirrored frame into hidden canvas
    analysis.width  = W;
    analysis.height = H;
    const ctx = analysis.getContext("2d")!;
    ctx.save();
    ctx.translate(W, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, W, H);
    ctx.restore();

    // Sample brightness in the oval region
    const cx   = W / 2;
    const cy   = H / 2;
    const srx  = Math.floor(W * 0.3);
    const sry  = Math.floor(H * 0.38);
    const data = ctx.getImageData(cx - srx, cy - sry, srx * 2, sry * 2).data;
    let lum = 0;
    for (let i = 0; i < data.length; i += 4)
      lum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    lum /= data.length / 4;

    let next: Status = "no-face";

    if (lum < 35) {
      next = "too-dark";
    } else if (lum > 215) {
      next = "too-bright";
    } else if (detectorRef.current) {
      try {
        const faces = await detectorRef.current.detect(video);
        if (faces.length === 0) {
          next = "no-face";
        } else {
          const b     = faces[0].boundingBox;
          const faceCx = (b.x + b.width  / 2) / W;
          const faceCy = (b.y + b.height / 2) / H;
          const faceW  = b.width / W;

          if (Math.abs(faceCx - 0.5) > 0.13 || Math.abs(faceCy - 0.5) > 0.15) {
            next = "off-center";
          } else if (faceW < 0.22) {
            next = "too-far";
          } else if (faceW > 0.62) {
            next = "too-close";
          } else {
            next = "ready";
          }
        }
      } catch {
        next = lum > 55 ? "ready" : "no-face";
      }
    } else {
      // No FaceDetector API — lighting proxy only
      next = lum > 55 && lum < 205 ? "ready" : "no-face";
    }

    // Update ready counter & progress
    if (next === "ready") {
      readyRef.current = Math.min(readyRef.current + 1, READY_FRAMES);
    } else {
      readyRef.current = Math.max(readyRef.current - 2, 0);
    }

    const pct = (readyRef.current / READY_FRAMES) * 100;
    setProgress(pct);
    setStatus(next);
    drawOverlay(overlay, W, H, next, pct);

    if (readyRef.current >= READY_FRAMES) {
      capture();
      return;
    }

    rafRef.current = requestAnimationFrame(analyze);
  }, [capture]);

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
          detectorRef.current = new (window as any).FaceDetector({
            fastMode: true,
            maxDetectedFaces: 1,
          });
        }

        setStatus("no-face");
        rafRef.current = requestAnimationFrame(analyze);
      } catch (err: unknown) {
        const name = (err as DOMException)?.name ?? "";
        const reason =
          name === "NotAllowedError" ? "permission" :
          name === "NotFoundError" || name === "DevicesNotFoundError" ? "no-camera" :
          "unknown";
        onError(reason);
      }
    }
    start();

    return () => {
      cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [analyze, onError]);

  /* ── Render ───────────────────────────────────────────────────────────── */
  return (
    <div className="flex flex-col items-center gap-4 w-full">
      {/* Camera viewport */}
      <div className="relative w-full max-w-xs aspect-[3/4] rounded-3xl overflow-hidden bg-black mx-auto">
        {/* Live video (mirrored like a mirror) */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="absolute inset-0 w-full h-full object-cover"
          style={{ transform: "scaleX(-1)" }}
        />

        {/* Oval overlay */}
        <canvas
          ref={overlayRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />

        {/* Feedback badge */}
        <div className="absolute bottom-5 left-0 right-0 flex justify-center pointer-events-none">
          <span
            className="px-4 py-1.5 rounded-full text-white text-xs font-semibold shadow-lg"
            style={{ backgroundColor: MSG[status].color + "dd" }}
          >
            {MSG[status].text}
          </span>
        </div>

        {/* Loading spinner */}
        {status === "loading" && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60">
            <div className="w-8 h-8 border-2 border-white/20 border-t-white rounded-full animate-spin" />
          </div>
        )}

        {/* Hidden analysis canvas */}
        <canvas ref={analysisRef} className="hidden" />
      </div>

      {/* Manual capture button */}
      <button
        onClick={capture}
        className="w-16 h-16 rounded-full bg-white border-4 border-neutral-200 shadow-md hover:scale-105 active:scale-95 transition-transform flex items-center justify-center"
        title="Capture photo"
      >
        <div className="w-10 h-10 rounded-full bg-neutral-800" />
      </button>

      <p className="text-xs text-neutral-400">
        {status === "ready"
          ? `Auto-capturing in ${Math.ceil(((READY_FRAMES - readyRef.current) / READY_FRAMES) * 1.8)}s…`
          : "Or tap the button to capture manually"}
      </p>
    </div>
  );
}
