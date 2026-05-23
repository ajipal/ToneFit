import Link from "next/link";
import Nav from "@/components/Nav";

const SEASONS = [
  {
    name: "Spring", cls: "season-spring",
    // warm: golden yellow, hot coral, soft peach, deep amber
    blobs: ["#FFD700", "#FF4D6D", "#FFAA5E", "#FF6B00"],
    delays: ["0s", "-3.5s", "-6s", "-1.5s"],
  },
  {
    name: "Summer", cls: "season-summer",
    // cool muted: soft lavender, dusty rose, powder blue, lilac
    blobs: ["#C5B3F0", "#F4A8C7", "#7EC8E3", "#A78BFA"],
    delays: ["-2s", "-5s", "-1s", "-8s"],
  },
  {
    name: "Autumn", cls: "season-autumn",
    // earthy warm: golden, burnt orange, deep rust, olive amber
    blobs: ["#E1AD01", "#CC4400", "#8B1A00", "#C8860A"],
    delays: ["-1s", "-4s", "-7s", "-2.5s"],
  },
  {
    name: "Winter", cls: "season-winter",
    // cool deep: aurora teal, violet, icy blue, deep purple
    blobs: ["#00C9A7", "#7C3AED", "#38BDF8", "#9B1D6E"],
    delays: ["-3s", "-6s", "-2s", "-9s"],
  },
];

export default function HomePage() {
  return (
    <div className="min-h-screen bg-[#fafaf8] text-[#141412] pt-[88px]">
      <Nav />

      {/* ── Hero ─────────────────────────────────────────── */}
      <section className="flex items-start px-[58.5px] pt-[56px] bg-[#fafaf8] overflow-hidden min-h-[700px]">
        {/* Left — text + CTAs */}
        <div className="flex flex-col flex-1 min-w-0 mt-[24px] max-w-[566px]">
          <p className="text-[13.4px] font-medium tracking-[2.4px] uppercase text-[#a8a59e] mb-[29px]">
            AI-Powered Personal Color Analysis
          </p>
          <h1 className="font-display text-[82px] leading-[1.05] text-[#141412] mb-[29px]">
            Discover your perfect color season
          </h1>
          <p className="text-[19.5px] leading-[1.65] text-[#a8a59e] max-w-[463px] mb-[29px]">
            Our AI analyzes your skin tone, undertone, and contrast to reveal your personal color
            season and the palette that makes you glow.
          </p>
          <div className="flex items-center gap-[14.6px] pt-[9.75px]">
            <Link
              href="/onboarding"
              className="bg-[#1a1a1a] text-white px-[29px] h-[53.6px] rounded-full flex items-center font-medium text-[17px] hover:bg-neutral-800 transition-colors whitespace-nowrap"
            >
              Discover It Now
            </Link>
            <a
              href="#how-it-works"
              className="border border-black/[0.09] text-[#141412] px-[30px] h-[53.6px] rounded-full flex items-center font-medium text-[17px] hover:bg-black/5 transition-colors whitespace-nowrap"
            >
              How it Works
            </a>
          </div>
        </div>

        {/* Right — 4 season gradient cards */}
        <div className="flex gap-[10px] flex-shrink-0 mt-0 ml-auto">
          {SEASONS.map((s) => (
            <div
              key={s.name}
              className={`relative rounded-[19.5px] overflow-hidden flex-shrink-0 ${s.cls}`}
              style={{ width: "165.8px", height: "520px" }}
            >
              {/* top-left */}
              <div className="season-blob blob-a" style={{ width: "210px", height: "210px", background: s.blobs[0], top: "-10%",  left: "-25%", animationDelay: s.delays[0] }} />
              {/* bottom-right */}
              <div className="season-blob blob-b" style={{ width: "200px", height: "200px", background: s.blobs[1], bottom: "-8%", right: "-30%", animationDelay: s.delays[1] }} />
              {/* center */}
              <div className="season-blob blob-c" style={{ width: "180px", height: "180px", background: s.blobs[2], top: "30%",  left: "-10%", animationDelay: s.delays[2] }} />
              {/* top-right */}
              <div className="season-blob blob-d" style={{ width: "160px", height: "160px", background: s.blobs[3], top: "5%",   right: "-20%", animationDelay: s.delays[3] }} />
              <span className="absolute bottom-[19.5px] left-[19.5px] text-[14.6px] font-semibold text-white z-10">
                {s.name}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* ── About ────────────────────────────────────────── */}
      <section id="about" className="px-[59px] py-[72px] bg-[#fafaf8] flex items-center">
        <div className="flex gap-[18px] w-full">

          {/* Left column — two stacked dark cards */}
          <div className="flex flex-col gap-[18px] flex-shrink-0" style={{ width: "658px" }}>
            {/* Card 1 — What is Color Analysis? */}
            <div
              className="bg-[#141412] rounded-[29px] overflow-hidden relative flex-shrink-0"
              style={{ height: "329px" }}
            >
              <h2 className="font-display italic text-[50px] leading-[97.5px] text-white/60 text-center absolute top-[30px] left-1/2 -translate-x-1/2 whitespace-nowrap">
                What is Color Analysis?
              </h2>
              <p className="absolute text-[17px] leading-[28.2px] text-white/45 text-justify" style={{ top: "130px", left: "78px", width: "502px" }}>
                Color analysis — also known as personal color analysis or armocromia — is a method
                of identifying which colors naturally complement your unique skin tone, undertone,
                and contrast level (Caygill, 1980; Jackson, 1980). The result is your color season:
                a personal color profile that tells you which shades make you look your most vibrant
                and put-together.
              </p>
            </div>

            {/* Card 2 — Why It Matters */}
            <div
              className="bg-[#141412] rounded-[29px] overflow-hidden relative flex-shrink-0"
              style={{ height: "263px" }}
            >
              <h2 className="font-display italic text-[50px] leading-[97.5px] text-white/60 text-center absolute top-[16px] left-1/2 -translate-x-1/2 whitespace-nowrap">
                Why It Matters
              </h2>
              <p className="absolute text-[17px] leading-[28.2px] text-white/45 text-justify" style={{ top: "116px", left: "78.5px", width: "502px" }}>
                Wearing your best colors can make your skin look clearer, your eyes appear brighter,
                and your overall look more harmonious — without changing anything else about yourself.
              </p>
            </div>
          </div>

          {/* Right column — About the Color Seasons */}
          <div
            className="bg-[#141412] rounded-[29px] overflow-hidden flex-1 flex-shrink-0 flex flex-col items-center justify-center px-[69px] py-[48px]"
            style={{ height: "610px" }}
          >
            <h2 className="font-display italic text-[50px] leading-[55px] text-white/60 text-center mb-[40px] max-w-[302px]">
              About the Color Seasons
            </h2>
            <div className="text-[17px] leading-[28.2px] text-white/45 text-justify space-y-[28px] max-w-[448px]">
              <p>
                The color seasons have nothing to do with weather or the time of year. They are named
                after the four seasons of nature because each group shares visual qualities — warmth,
                coolness, brightness, or depth — that mirror what we see in spring blooms, summer
                haze, autumn leaves, and winter frost (Jackson, 1980; Kentner, 1985). Your color
                season is determined by your natural coloring, not the calendar.
              </p>
              <p>
                ToneFit uses a 12-season model, which expands the original four seasons into
                sub-groups for a more precise result (Findlay, 2012; Stacchio et al., 2024).
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── How it Works ─────────────────────────────────── */}
      <section id="how-it-works" className="px-[58.5px] py-[72px] bg-[#fafaf8] flex flex-col justify-center">
        {/* Section header */}
        <div className="text-center mb-[48px]">
          <p className="text-[13.4px] font-medium tracking-[2.4px] uppercase text-[#a8a59e] mb-4">
            How It Works
          </p>
          <h2 className="font-display text-[42px] leading-tight text-[#141412]">
            Three steps to your season
          </h2>
        </div>

        <div className="grid grid-cols-3" style={{ gap: "18px" }}>

          {/* Step 1 */}
          <div className="bg-white rounded-[29px] shadow-[0px_4px_24px_rgba(0,0,0,0.07)] px-[48px] pt-[48px] pb-[48px] flex flex-col items-center text-center">
            <p className="text-[12px] font-medium tracking-[2px] uppercase text-[#a8a59e] mb-5">Step 1</p>
            <div className="border border-black/10 bg-neutral-50 rounded-full flex items-center justify-center mb-6" style={{ width: "88px", height: "88px" }}>
              <svg className="w-9 h-9 text-[#141412]/40" fill="none" viewBox="0 0 40 40" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="11" width="34" height="23" rx="4" />
                <circle cx="20" cy="22.5" r="6" />
                <path d="M13.5 11 16 7h8l2.5 4" />
              </svg>
            </div>
            <h3 className="text-[18px] font-bold text-[#141412] mb-3">Take or Upload a Photo</h3>
            <p className="text-[15px] leading-[26px] text-neutral-500">
              Use your camera or upload a clear face photo. Make sure you have good lighting and face
              the camera directly for the most accurate results.
            </p>
          </div>

          {/* Step 2 */}
          <div className="bg-white rounded-[29px] shadow-[0px_4px_24px_rgba(0,0,0,0.07)] px-[48px] pt-[48px] pb-[48px] flex flex-col items-center text-center">
            <p className="text-[12px] font-medium tracking-[2px] uppercase text-[#a8a59e] mb-5">Step 2</p>
            <div className="border border-black/10 bg-neutral-50 rounded-full flex items-center justify-center mb-6" style={{ width: "88px", height: "88px" }}>
              <svg className="w-9 h-9 text-[#141412]/40" fill="none" viewBox="0 0 40 40" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M20 5 l2.8 9.5 9.5 2.8-9.5 2.8L20 30l-2.8-9.9L7.7 17.6l9.5-2.8z" />
                <path d="M32 26l1.5 4.5L38 32l-4.5 1.5L32 38l-1.5-4.5L26 32l4.5-1.5z" />
              </svg>
            </div>
            <h3 className="text-[18px] font-bold text-[#141412] mb-3">AI Analyzes Your Features</h3>
            <p className="text-[15px] leading-[26px] text-neutral-500">
              Our model analyzes your skin tone, undertone, and contrast to classify your personal
              color season from 12 possible sub-types.
            </p>
          </div>

          {/* Step 3 */}
          <div className="bg-white rounded-[29px] shadow-[0px_4px_24px_rgba(0,0,0,0.07)] px-[48px] pt-[48px] pb-[48px] flex flex-col items-center text-center">
            <p className="text-[12px] font-medium tracking-[2px] uppercase text-[#a8a59e] mb-5">Step 3</p>
            <div className="border border-black/10 bg-neutral-50 rounded-full flex items-center justify-center mb-6" style={{ width: "88px", height: "88px" }}>
              <svg className="w-9 h-9 text-[#141412]/40" fill="none" viewBox="0 0 40 40" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="6" y="8" width="28" height="28" rx="4" />
                <path d="M13 22l5 5 9-9" />
              </svg>
            </div>
            <h3 className="text-[18px] font-bold text-[#141412] mb-3">Get Your Full Style Guide</h3>
            <p className="text-[15px] leading-[26px] text-neutral-500">
              Receive your season, color palette, outfit recommendations, accessories, and makeup
              tips — all tailored to your personal color season.
            </p>
          </div>

        </div>

        {/* CTA */}
        <div className="flex justify-center mt-[48px]">
          <Link
            href="/onboarding"
            className="bg-[#1a1a1a] text-white px-[40px] h-[53.6px] rounded-full flex items-center font-medium text-[17px] hover:bg-neutral-800 transition-colors whitespace-nowrap"
          >
            Discover Your Color Palette
          </Link>
        </div>
      </section>

      {/* ── Footer ───────────────────────────────────────── */}
      <footer className="border-t border-black/8 px-[58.5px] py-8 flex items-center justify-between text-sm text-neutral-400">
        <span className="font-semibold text-[#141412]">ToneFit AI</span>
        <span>PUP Manila · Data Science Pilot Study · 2026</span>
      </footer>
    </div>
  );
}
