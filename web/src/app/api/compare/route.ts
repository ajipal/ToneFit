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

  const base64Data = photo.replace(/^data:image\/\w+;base64,/, "");
  const buffer = Buffer.from(base64Data, "base64");
  const blob = new Blob([buffer], { type: "image/jpeg" });

  const form = new FormData();
  form.append("file", blob, "photo.jpg");

  let response: Response;
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60_000);
    response = await fetch(`${PYTHON_API_URL}/compare`, {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    clearTimeout(timeout);
  } catch {
    return NextResponse.json(
      { error: "Could not reach the ML server. Make sure the backend is running." },
      { status: 502 }
    );
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    return NextResponse.json({ error: `ML server error: ${text}` }, { status: 502 });
  }

  return NextResponse.json(await response.json());
}
