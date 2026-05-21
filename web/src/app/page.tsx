import Link from "next/link";
import Nav from "@/components/Nav";

const SEASON_PANELS = [
  { name: "Spring", className: "season-panel-spring", textColor: "text-amber-900" },
  { name: "Summer", className: "season-panel-summer", textColor: "text-sky-900" },
  { name: "Autumn", className: "season-panel-autumn", textColor: "text-amber-950" },
  { name: "Winter", className: "season-panel-winter", textColor: "text-white" },
];

const HOW_IT_WORKS = [
  {
    step: "Step 1",
    icon: (
      <svg viewBox="0 0 40 40" fill="none" className="w-10 h-10" stroke="currentColor" strokeWidth="1.8">
        <circle cx="20" cy="15" r="8" />
        <path d="M5 38c0-8.284 6.716-15 15-15s15 6.716 15 15" strokeLinecap="round" />
      </svg>
    ),
    title: "Take or Upload a Photo",
    description:
      "Use your camera or upload a clear face photo. Make sure you have good lighting and face the camera directly for the most accurate results.",
  },
  {
    step: "Step 2",
    icon: (
      <svg viewBox="0 0 40 40" fill="none" className="w-10 h-10" stroke="currentColor" strokeWidth="1.8">
        <circle cx="20" cy="20" r="16" />
        <path d="M20 8v4M20 28v4M8 20h4M28 20h4" strokeLinecap="round" />
        <circle cx="20" cy="20" r="6" />
      </svg>
    ),
    title: "AI Analyzes Your Features",
    description:
      "Our model analyzes your skin tone, undertone, and contrast to classify your personal color season from 12 possible sub-types.",
  },
  {
    step: "Step 3",
    icon: (
      <svg viewBox="0 0 40 40" fill="none" className="w-10 h-10" stroke="currentColor" strokeWidth="1.8">
        <rect x="6" y="8" width="28" height="28" rx="4" />
        <path d="M13 20l5 5 9-9" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
    title: "Get Your Full Style Guide",
    description:
      "Receive your season, color palette, outfit recommendations, accessories, and makeup tips — all tailored to your personal color season.",
  },
];

export default function HomePage() {
  return (
    <div className="min-h-screen bg-white text-black">
      <Nav />

      {/* ── Hero ─────────────────────────────────────────── */}
      <section className="flex h-[calc(100vh-88px)] min-h-[640px] overflow-hidden">
        {/* Left — text + CTAs */}
        <div className="flex flex-col justify-center px-14 flex-1 max-w-[640px] gap-0">
          <p className="text-sm font-semibold tracking-widest uppercase text-neutral-500 mb-6">
            AI-Powered Personal Color Analysis
          </p>
          <h1 className="text-[64px] font-black leading-[1.05] tracking-tight text-black mb-8">
            Discover Your Perfect Color Season
          </h1>
          <p className="text-lg leading-relaxed text-neutral-600 max-w-[460px] mb-10">
            Our AI analyzes your skin tone, undertone, and contrast to reveal your personal color
            season and give you a complete, personalized style guide.
          </p>
          <div className="flex items-center gap-4">
            <Link
              href="/onboarding"
              className="px-8 py-4 bg-black text-white rounded-xl font-semibold text-[15px] hover:bg-neutral-800 transition-colors"
            >
              Discover It Now
            </Link>
            <a
              href="#how-it-works"
              className="px-8 py-4 border border-black/20 rounded-xl font-semibold text-[15px] text-neutral-700 hover:bg-neutral-50 transition-colors"
            >
              How it Works
            </a>
          </div>
        </div>

        {/* Right — 4 season panels */}
        <div className="flex flex-1">
          {SEASON_PANELS.map((s) => (
            <div key={s.name} className={`flex-1 relative ${s.className}`}>
              <span
                className={`absolute bottom-6 left-5 text-base font-semibold tracking-wide ${s.textColor}`}
              >
                {s.name}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* ── Feature highlight ─────────────────────────────── */}
      <section id="about" className="flex min-h-[640px] border-t border-black/8">
        {/* Left — two stacked info blocks */}
        <div className="flex-1 flex flex-col border-r border-black/8">
          <div className="flex-1 p-14 border-b border-black/8 flex flex-col justify-center">
            <span className="text-xs font-bold tracking-widest uppercase text-neutral-400 mb-4">
              01
            </span>
            <h2 className="text-3xl font-black leading-tight mb-4">
              What Is Personal Color Analysis?
            </h2>
            <p className="text-base leading-relaxed text-neutral-600">
              Personal color analysis (Armocromia) identifies which seasonal color palette —
              Spring, Summer, Autumn, or Winter — best harmonizes with your natural skin tone,
              hair, and eye color. Wearing your season&apos;s colors makes you look more vibrant,
              healthier, and effortlessly put-together.
            </p>
          </div>
          <div className="flex-1 p-14 flex flex-col justify-center">
            <span className="text-xs font-bold tracking-widest uppercase text-neutral-400 mb-4">
              02
            </span>
            <h2 className="text-3xl font-black leading-tight mb-4">
              12 Sub-Types for Precision
            </h2>
            <p className="text-base leading-relaxed text-neutral-600">
              ToneFit goes beyond 4 seasons. We classify you into one of 12 sub-types —
              like Deep Autumn or Bright Winter — for a more nuanced, accurate color
              profile tailored specifically to your unique coloring.
            </p>
          </div>
        </div>

        {/* Right — visual card */}
        <div className="flex-1 p-14 flex items-center justify-center bg-neutral-50">
          <div className="w-full max-w-md rounded-3xl overflow-hidden shadow-xl aspect-[4/5] bg-gradient-to-br from-neutral-100 to-neutral-200 flex flex-col items-center justify-center gap-4 text-neutral-400">
            <svg viewBox="0 0 80 80" fill="none" className="w-16 h-16" stroke="currentColor" strokeWidth="1.5">
              <circle cx="40" cy="30" r="14" />
              <path d="M12 72c0-15.464 12.536-28 28-28s28 12.536 28 28" strokeLinecap="round" />
            </svg>
            <p className="text-sm font-medium">Your result preview</p>
          </div>
        </div>
      </section>

      {/* ── How it Works ─────────────────────────────────── */}
      <section id="how-it-works" className="border-t border-black/8 py-24 px-14">
        <div className="max-w-5xl mx-auto">
          <p className="text-xs font-bold tracking-widest uppercase text-neutral-400 mb-4 text-center">
            How it Works
          </p>
          <h2 className="text-[42px] font-black text-center mb-16 tracking-tight">
            Three steps to your season
          </h2>

          <div className="grid grid-cols-3 gap-8">
            {HOW_IT_WORKS.map((item) => (
              <div key={item.step} className="flex flex-col items-center text-center gap-6">
                {/* Step number */}
                <div className="w-full flex justify-center">
                  <span className="text-xs font-bold tracking-widest uppercase text-neutral-400">
                    {item.step}
                  </span>
                </div>

                {/* Icon circle */}
                <div className="w-24 h-24 rounded-full bg-neutral-100 flex items-center justify-center text-neutral-600">
                  {item.icon}
                </div>

                {/* Text */}
                <div>
                  <h3 className="text-xl font-bold mb-3">{item.title}</h3>
                  <p className="text-sm leading-relaxed text-neutral-600">{item.description}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="flex justify-center mt-16">
            <Link
              href="/onboarding"
              className="px-10 py-4 bg-black text-white rounded-xl font-semibold text-base hover:bg-neutral-800 transition-colors"
            >
              Discover Your Color Palette
            </Link>
          </div>
        </div>
      </section>

      {/* ── Footer ───────────────────────────────────────── */}
      <footer className="border-t border-black/8 px-14 py-8 flex items-center justify-between text-sm text-neutral-400">
        <span className="font-semibold text-black">ToneFit AI</span>
        <span>PUP Manila · Data Science Pilot Study · 2026</span>
      </footer>
    </div>
  );
}
