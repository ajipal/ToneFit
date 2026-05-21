"use client";

import { useState, useCallback } from "react";
import Cropper from "react-easy-crop";
import type { Area } from "react-easy-crop";

interface Props {
  imageSrc: string;
  onCrop: (file: File, preview: string) => void;
  onCancel: () => void;
}

async function cropImageToFile(imageSrc: string, cropArea: Area): Promise<{ file: File; preview: string }> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const size = Math.min(cropArea.width, cropArea.height);
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d")!;

      // Clip to oval
      ctx.beginPath();
      ctx.ellipse(size / 2, size / 2, size / 2, size / 2, 0, 0, Math.PI * 2);
      ctx.clip();

      ctx.drawImage(
        img,
        cropArea.x, cropArea.y, cropArea.width, cropArea.height,
        0, 0, size, size,
      );

      canvas.toBlob(
        (blob) => {
          if (!blob) return;
          const file = new File([blob], "photo.jpg", { type: "image/jpeg" });
          const preview = URL.createObjectURL(blob);
          resolve({ file, preview });
        },
        "image/jpeg",
        0.92,
      );
    };
    img.src = imageSrc;
  });
}

export default function PhotoCropper({ imageSrc, onCrop, onCancel }: Props) {
  const [crop, setCrop]       = useState({ x: 0, y: 0 });
  const [zoom, setZoom]       = useState(1);
  const [cropArea, setCropArea] = useState<Area | null>(null);
  const [busy, setBusy]       = useState(false);

  const onCropComplete = useCallback((_: Area, croppedPixels: Area) => {
    setCropArea(croppedPixels);
  }, []);

  const handleConfirm = async () => {
    if (!cropArea || busy) return;
    setBusy(true);
    const { file, preview } = await cropImageToFile(imageSrc, cropArea);
    onCrop(file, preview);
  };

  return (
    <div className="w-full flex flex-col items-center gap-5">
      <p className="text-sm text-neutral-500 text-center">
        Drag to reposition · pinch or scroll to zoom
      </p>

      {/* Crop viewport */}
      <div className="relative w-full aspect-square rounded-3xl overflow-hidden bg-black">
        <Cropper
          image={imageSrc}
          crop={crop}
          zoom={zoom}
          aspect={1}
          cropShape="round"
          showGrid={false}
          onCropChange={setCrop}
          onZoomChange={setZoom}
          onCropComplete={onCropComplete}
          style={{
            containerStyle: { borderRadius: "1.5rem" },
            cropAreaStyle: {
              border: "2.5px solid #22c55e",
              boxShadow: "0 0 0 9999px rgba(0,0,0,0.55)",
            },
          }}
        />
      </div>

      {/* Zoom slider */}
      <input
        type="range"
        min={1}
        max={3}
        step={0.01}
        value={zoom}
        onChange={(e) => setZoom(Number(e.target.value))}
        className="w-full accent-black"
      />

      <div className="flex gap-3 w-full">
        <button
          onClick={onCancel}
          className="flex-1 py-3 border-2 border-black/12 rounded-xl font-semibold text-sm hover:border-black/30 transition-colors"
        >
          Cancel
        </button>
        <button
          onClick={handleConfirm}
          disabled={busy}
          className="flex-1 py-3 bg-black text-white rounded-xl font-semibold text-sm hover:bg-neutral-800 transition-colors disabled:opacity-50"
        >
          {busy ? "Cropping…" : "Use this crop"}
        </button>
      </div>
    </div>
  );
}
