import { NextRequest, NextResponse } from "next/server";

const PYTHON_API_URL = process.env.PYTHON_API_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const form = await req.formData().catch(() => null);
  if (!form) return NextResponse.json({ error: "Invalid request" }, { status: 400 });

  let response: Response;
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15_000);
    response = await fetch(`${PYTHON_API_URL}/check-face`, {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    clearTimeout(timeout);
  } catch {
    return NextResponse.json({ error: "Could not reach the ML server." }, { status: 502 });
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    return NextResponse.json({ error: `ML server error: ${text}` }, { status: 502 });
  }

  return NextResponse.json(await response.json());
}
