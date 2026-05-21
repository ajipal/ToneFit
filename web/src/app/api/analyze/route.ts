import { NextRequest, NextResponse } from "next/server";

const PYTHON_API_URL = process.env.PYTHON_API_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  let photo: string;
  try {
    ({ photo } = await req.json());
  } catch {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  if (!photo) {
    return NextResponse.json({ error: "No photo provided" }, { status: 400 });
  }

  // Strip data URL prefix and decode to bytes
  const base64Data = photo.replace(/^data:image\/\w+;base64,/, "");
  const buffer = Buffer.from(base64Data, "base64");
  const blob = new Blob([buffer], { type: "image/jpeg" });

  const form = new FormData();
  form.append("file", blob, "photo.jpg");

  let response: Response;
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30_000);

    response = await fetch(`${PYTHON_API_URL}/predict`, {
      method: "POST",
      body: form,
      signal: controller.signal,
    });

    clearTimeout(timeout);
  } catch {
    return NextResponse.json(
      { error: "Could not reach the ML server. Make sure server.py is running." },
      { status: 502 }
    );
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    console.error("[/api/analyze] ML server error:", response.status, text);
    return NextResponse.json({ error: "ML server returned an error" }, { status: 502 });
  }

  const result = await response.json();
  return NextResponse.json(result);
}
