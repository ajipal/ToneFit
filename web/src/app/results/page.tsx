"use client";

import { useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Nav from "@/components/Nav";
import { SEASON_DATA, type SeasonKey } from "@/lib/season-data";

type Tab = "overview" | "colors" | "clothes" | "makeup";

const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "colors",   label: "Color Palette" },
  { id: "clothes",  label: "Clothes Style" },
  // { id: "makeup", label: "Makeup Style" }, // hidden — images not ready
];

/* ── Outfit image lookup — maps (style, subtype) → public path ───────────── */
const OUTFIT_IMAGES: Record<string, Partial<Record<string, string>>> = {
  casual: {
    spring_warm:   "/outfits/casual/spring_warm_casual.png",
    spring_light:  "/outfits/casual/spring_light_casual.png",
    spring_bright: "/outfits/casual/spring_bright_casual.png",
    summer_cool:   "/outfits/casual/cool_summer_casual.png",
    summer_light:  "/outfits/casual/light_summer_casual.png",
    summer_soft:   "/outfits/casual/soft_summer_casual.png",
    autumn_warm:   "/outfits/casual/warm_autumn_casual.png",
    autumn_soft:   "/outfits/casual/soft_autumn_casual.png",
    autumn_deep:   "/outfits/casual/deep_autumn_casual.png",
    winter_cool:   "/outfits/casual/cool_winter_casual.png",
    winter_deep:   "/outfits/casual/deep_winter_casual.png",
    winter_bright: "/outfits/casual/bright_winter_casual.png",
  },
  classic: {
    spring_warm:   "/outfits/classic/warm_spring_classic.png",
    spring_light:  "/outfits/classic/light_spring_classic.png",
    spring_bright: "/outfits/classic/bright_spring_classic.png",
    summer_cool:   "/outfits/classic/cool_summer_classic.png",
    summer_light:  "/outfits/classic/light_summer_classic.png",
    summer_soft:   "/outfits/classic/soft_summer_classic.png",
    autumn_warm:   "/outfits/classic/warm_autumn_classic.png",
    autumn_soft:   "/outfits/classic/soft_autumn_classic.png",
    autumn_deep:   "/outfits/classic/deep_autumn_classic.png",
    winter_cool:   "/outfits/classic/cool_winter_classic.png",
    winter_deep:   "/outfits/classic/deep_winter_classic.png",
    winter_bright: "/outfits/classic/bright_winter_classic.png",
  },
  formal: {
    spring_warm:   "/outfits/formal/outfitsspring_warmformal.png",
    spring_light:  "/outfits/formal/outfitsspring_lightformal.png",
    spring_bright: "/outfits/formal/outfitsspring_brightformal.png",
    summer_cool:   "/outfits/formal/outfitssummer_coolformal.png",
    summer_light:  "/outfits/formal/outfitssummer_lightformal.png",
    summer_soft:   "/outfits/formal/outfitssummer_softformal.png",
    autumn_warm:   "/outfits/formal/outfitsautumn_warmformal.png",
    autumn_soft:   "/outfits/formal/outfitsautumn_softformal.png",
    autumn_deep:   "/outfits/formal/outfitsautumn_deepformal.png",
    winter_cool:   "/outfits/formal/outfitswinter_coolformal.png",
    winter_deep:   "/outfits/formal/outfitswinter_deepformal.png",
    winter_bright: "/outfits/formal/outfitswinter_brightformal.png",
  },
  retro: {
    spring_warm:   "/outfits/retro/outfitsspring_warmretro.png",
    spring_light:  "/outfits/retro/outfitsspring_lightretro.png",
    spring_bright: "/outfits/retro/outfitsspring_brightretro.png",
    summer_cool:   "/outfits/retro/outfitssummer_coolretro.png",
    summer_light:  "/outfits/retro/outfitssummer_lightretro.jpg.png",
    summer_soft:   "/outfits/retro/outfitssummer_softretro.png",
    autumn_warm:   "/outfits/retro/outfitsautumn_warmretro.png",
    autumn_soft:   "/outfits/retro/outfitsautumn_softretro.png",
    autumn_deep:   "/outfits/retro/outfitsautumn_deepretro.png",
    winter_cool:   "/outfits/retro/outfitswinter_coolretro.png",
    winter_deep:   "/outfits/retro/outfitswinter_deepretro..png",
    winter_bright: "/outfits/retro/outfitswinter_brightretro.png",
  },
  smart_casual: {
    spring_warm:   "/outfits/smart casual/warm_spring_smart_casual.png",
    spring_light:  "/outfits/smart casual/light_spring_smart_casual.png",
    spring_bright: "/outfits/smart casual/bright_spring_smart_casual.png",
    summer_cool:   "/outfits/smart casual/cool_summer_smart_casual.png",
    summer_light:  "/outfits/smart casual/light_summer_smart_casual.png",
    summer_soft:   "/outfits/smart casual/soft_summer_smart_casual.png",
    autumn_warm:   "/outfits/smart casual/warm_autumn_smart_casual.png",
    autumn_soft:   "/outfits/smart casual/soft_autumn_smart_casual.png",
    autumn_deep:   "/outfits/smart casual/deep_autumn_smart_casual.png",
    winter_cool:   "/outfits/smart casual/cool_winter_smart_casual.png",
    winter_deep:   "/outfits/smart casual/deep_winter_smart_casual.png",
    winter_bright: "/outfits/smart casual/bright_winter_smart_casual.png",
  },
  streetwear: {
    spring_warm:   "/outfits/streetwear/warm_spring_streetwear.png",
    spring_light:  "/outfits/streetwear/light_spring_streetwear.png",
    spring_bright: "/outfits/streetwear/bright_spring_streetwear.png",
    summer_cool:   "/outfits/streetwear/cool_summer_streetwear.png",
    summer_light:  "/outfits/streetwear/light_summer_streetwear.png",
    summer_soft:   "/outfits/streetwear/soft_summer_streetwear.png",
  },
};

/* ── Season gradient palette ─────────────────────────────────────────────── */
const SEASON_THEME: Record<string, { nameColor: string; gradientFrom: string; gradientTo: string }> = {
  spring: { nameColor: "#d4621a", gradientFrom: "#e87850", gradientTo: "#fddeca" },
  summer: { nameColor: "#3a6880", gradientFrom: "#5a85a0", gradientTo: "#c8e0f0" },
  autumn: { nameColor: "#8a4e1e", gradientFrom: "#c4773a", gradientTo: "#f4dfc0" },
  winter: { nameColor: "#2244cc", gradientFrom: "#2244cc", gradientTo: "#aaddff" },
};

/* ── Nav heights ─────────────────────────────────────────────────────────── */
const MAIN_NAV_H  = 88;   // px — desktop nav height (64px on mobile via CSS)
const TAB_BAR_H   = 48;   // px — secondary tab strip
const TOTAL_NAV_H = MAIN_NAV_H + TAB_BAR_H;

/* ── Color swatch grid — 4 cols × 3 rows, fills parent height ────────────── */
function ColorGrid({ colors }: { colors: { name: string; hex: string }[] }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px", height: "100%" }}>
      {[colors.slice(0, 4), colors.slice(4, 8), colors.slice(8, 12)].map((row, ri) => (
        <div key={ri} style={{ display: "flex", gap: "12px", flex: 1, minHeight: 0 }}>
          {row.map((c) => (
            <div key={c.name} style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: "6px" }}>
              {/* Swatch fills available row height */}
              <div style={{ flex: 1, backgroundColor: c.hex, borderRadius: "12px", minHeight: 0 }} />
              <span style={{ fontSize: "11px", lineHeight: "14px", color: "#111", textAlign: "center", flexShrink: 0 }}>
                {c.name}
              </span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

/* ── Style card ──────────────────────────────────────────────────────────── */
function StyleCard({ title, wear, avoid, style }: { title: string; wear: string; avoid?: string; style?: React.CSSProperties }) {
  return (
    <div style={{ background: "#fff", border: "1px solid #e0e0e0", borderRadius: "12px", padding: "10px 16px", boxShadow: "0 2px 8px rgba(0,0,0,0.06)", display: "flex", flexDirection: "column", justifyContent: "center", ...style }}>
      <p style={{ fontWeight: 700, fontSize: "13px", lineHeight: "16px", color: "#0d0d0d", margin: "0 0 6px 0" }}>{title}</p>
      <div style={{ display: "flex", gap: "8px", alignItems: "flex-start", marginBottom: avoid ? "4px" : 0 }}>
        <span style={{ fontWeight: 700, fontSize: "11px", lineHeight: "16px", color: "#268c4d", flexShrink: 0 }}>✓ WEAR</span>
        <p style={{ fontSize: "11px", lineHeight: "16px", color: "#4d4d4d", margin: 0 }}>{wear}</p>
      </div>
      {avoid && (
        <div style={{ display: "flex", gap: "8px", alignItems: "flex-start" }}>
          <span style={{ fontWeight: 700, fontSize: "11px", lineHeight: "16px", color: "#bf3333", flexShrink: 0 }}>✗ AVOID</span>
          <p style={{ fontSize: "11px", lineHeight: "16px", color: "#808080", margin: 0 }}>{avoid}</p>
        </div>
      )}
    </div>
  );
}

/* ── Placeholder image card ──────────────────────────────────────────────── */
function PlaceholderCard({ label }: { label: string }) {
  return (
    <div style={{ width: "100%", height: "100%", background: "#f0f0f0", border: "1px solid #ccc", borderRadius: "20px", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "12px", color: "#aaa" }}>
        <svg viewBox="0 0 64 64" fill="none" width="56" height="56" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <rect x="6" y="6" width="52" height="52" rx="6" />
          <circle cx="22" cy="24" r="7" />
          <path d="M6 46l16-14 9 9 11-13 16 18" />
        </svg>
        <p style={{ fontSize: "14px", fontWeight: 500, margin: 0 }}>{label}</p>
      </div>
    </div>
  );
}

/* ── Section heading ─────────────────────────────────────────────────────── */
function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2
      className="font-cormorant font-bold text-black"
      style={{ fontSize: "43px", lineHeight: "48px", margin: "0 0 20px 0" }}
    >
      {children}
    </h2>
  );
}

/* ── Results inner ───────────────────────────────────────────────────────── */
function ResultsInner() {
  const searchParams = useSearchParams();
  const seasonKey = (searchParams.get("season") ?? "winter_cool") as SeasonKey;
  const data = SEASON_DATA[seasonKey] ?? SEASON_DATA["winter_cool"];

  const [userName, setUserName] = useState("friend");
  const [userStyle, setUserStyle] = useState("");
  const [activeTab, setActiveTab] = useState<Tab>("overview");

  useEffect(() => {
    if (typeof window !== "undefined") {
      try {
        const profile = JSON.parse(localStorage.getItem("tonefit_profile") ?? "{}");
        if (profile.name)  setUserName(profile.name);
        if (profile.style) setUserStyle(profile.style);
      } catch { /* no-op */ }
    }
  }, []);

  const outfitSrc = OUTFIT_IMAGES[userStyle]?.[seasonKey] ?? null;
  const baseSeason = seasonKey.split("_")[0];
  const theme = SEASON_THEME[baseSeason] ?? SEASON_THEME["winter"];

  /* Content area height = viewport minus both nav bars */
  const contentH = `calc(100vh - ${TOTAL_NAV_H}px)`;
  const contentPadding = "40px 109px";

  return (
    <div style={{ background: "#fff", color: "#000", minHeight: "100vh" }}>
      {/* ── Main nav ──────────────────────────────────────────────────────── */}
      <Nav variant="light" />

      {/* ── Tab bar — fixed just below main nav, centered & spread ────────── */}
      <div className="results-tab-bar" style={{
        position: "fixed",
        top: `${MAIN_NAV_H}px`,
        left: 0,
        right: 0,
        height: `${TAB_BAR_H}px`,
        background: "#fff",
        borderBottom: "1px solid #e8e8e8",
        display: "flex",
        alignItems: "stretch",
        justifyContent: "center",
        zIndex: 40,
      }}>
        {TABS.map((tab) => {
          const active = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                flex: 1,
                maxWidth: "240px",
                fontSize: "14px",
                fontWeight: active ? 600 : 400,
                color: active ? "#111" : "#888",
                background: "none",
                border: "none",
                borderBottom: active ? "2px solid #111" : "2px solid transparent",
                cursor: "pointer",
                transition: "color 0.15s, border-color 0.15s",
                whiteSpace: "nowrap",
              }}
            >
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* ── Content area — pushed below both navs ─────────────────────────── */}
      <div style={{ paddingTop: `${TOTAL_NAV_H}px`, height: "100vh", boxSizing: "border-box", overflow: "hidden" }}>

        {/* ── Overview ──────────────────────────────────────────────────── */}
        {activeTab === "overview" && (
          <div className="results-content-area results-two-col" style={{ height: contentH, padding: contentPadding, boxSizing: "border-box", display: "flex", gap: "40px", alignItems: "stretch" }}>

            {/* Left */}
            <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", justifyContent: "center" }}>
              <h1
                className="results-hero-heading font-cormorant font-bold"
                style={{ fontSize: "64px", lineHeight: "72px", color: theme.nameColor, margin: 0 }}
              >
                Hi, {userName}!
              </h1>
              <div
                className="results-hero-subheading font-cormorant font-bold"
                style={{
                  fontSize: "80px", lineHeight: "90px",
                  background: `linear-gradient(to right, ${theme.gradientFrom}, ${theme.gradientTo})`,
                  WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text",
                }}
              >
                <p style={{ margin: 0 }}>You Are</p>
                <p style={{ margin: 0 }}>{data.displayName}</p>
              </div>
              <div style={{ marginTop: "24px" }}>
                <p style={{ fontSize: "13px", lineHeight: "20px", color: "#8c8c8c", letterSpacing: "0.08px", marginBottom: "10px", textTransform: "uppercase", margin: "0 0 10px 0" }}>
                  Season Overview
                </p>
                <p style={{ fontSize: "17px", lineHeight: "26px", color: "#111", textAlign: "justify", margin: 0 }}>
                  {data.description}
                </p>
              </div>
            </div>

            {/* Right */}
            <div style={{ flex: 1, minWidth: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
              {outfitSrc ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={outfitSrc}
                  alt={`${data.displayName} ${userStyle} outfit`}
                  style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", borderRadius: "20px" }}
                />
              ) : (
                <PlaceholderCard label="Outfit photo" />
              )}
            </div>
          </div>
        )}

        {/* ── Color Palette ─────────────────────────────────────────────── */}
        {activeTab === "colors" && (
          <div className="results-content-area results-two-col" style={{ height: contentH, padding: contentPadding, boxSizing: "border-box", display: "flex", gap: "32px" }}>

            {/* Best Colors */}
            <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
              <h2 className="font-cormorant font-bold text-black" style={{ fontSize: "43px", lineHeight: "48px", margin: "0 0 16px 0", textAlign: "center" }}>
                Best Colors
              </h2>
              {/* Card fills remaining height, grid fills card */}
              <div style={{ flex: 1, minHeight: 0, background: "#fff", border: "1px solid #e0e0e0", borderRadius: "20px", padding: "24px", boxShadow: "0 4px 24px rgba(0,0,0,0.08)", boxSizing: "border-box" }}>
                <ColorGrid colors={data.bestColors} />
              </div>
            </div>

            {/* Avoid */}
            <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
              <h2 className="font-cormorant font-bold text-black" style={{ fontSize: "43px", lineHeight: "48px", margin: "0 0 16px 0", textAlign: "center" }}>
                Avoid
              </h2>
              <div style={{ flex: 1, minHeight: 0, background: "#fff", border: "1px solid #e0e0e0", borderRadius: "20px", padding: "24px", boxShadow: "0 4px 24px rgba(0,0,0,0.08)", boxSizing: "border-box" }}>
                <ColorGrid colors={data.avoidColors} />
              </div>
            </div>
          </div>
        )}

        {/* ── Clothes Style ─────────────────────────────────────────────── */}
        {activeTab === "clothes" && (
          <div className="results-content-area" style={{ height: contentH, padding: contentPadding, boxSizing: "border-box", display: "flex", flexDirection: "column" }}>
            <SectionHeading>Clothes Style</SectionHeading>
            <div className="results-two-col" style={{ display: "flex", gap: "32px", flex: 1, minHeight: 0, overflow: "hidden" }}>
              {/* Left — cards share height equally via flex:1 */}
              <div className="results-cards-col" style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: "8px", overflow: "hidden" }}>
                <StyleCard style={{ flex: 1 }} title="Tops"                   wear={data.clothes.tops.wear}        avoid={data.clothes.tops.avoid} />
                <StyleCard style={{ flex: 1 }} title="Bottoms"                wear={data.clothes.bottoms.wear}     avoid={data.clothes.bottoms.avoid} />
                <StyleCard style={{ flex: 1 }} title="Bags & Shoes"           wear={data.clothes.bagsShoes.wear}   avoid={data.clothes.bagsShoes.avoid} />
                <StyleCard style={{ flex: 1 }} title="Metals, Jewelry & Gems" wear={data.clothes.metals.wear}      avoid={data.clothes.metals.avoid} />
                <StyleCard style={{ flex: 1 }} title="Pairing Tips"           wear={data.clothes.pairingTips.wear} />
              </div>
              <div style={{ flex: 1, minWidth: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
                {outfitSrc ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={outfitSrc}
                    alt={`${data.displayName} ${userStyle} outfit`}
                    style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", borderRadius: "20px" }}
                  />
                ) : (
                  <PlaceholderCard label="Outfit example" />
                )}
              </div>
            </div>
          </div>
        )}

        {/* ── Makeup Style ──────────────────────────────────────────────── */}
        {activeTab === "makeup" && (
          <div style={{ height: contentH, padding: contentPadding, boxSizing: "border-box", display: "flex", flexDirection: "column" }}>
            <SectionHeading>Makeup Style</SectionHeading>
            <div style={{ display: "flex", gap: "32px", flex: 1, minHeight: 0, overflow: "hidden" }}>
              {/* Left — cards share height equally via flex:1 */}
              <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: "8px", overflow: "hidden" }}>
                <StyleCard style={{ flex: 1 }} title="LIPS"               wear={data.makeup.lips.wear}  avoid={data.makeup.lips.avoid} />
                <StyleCard style={{ flex: 1 }} title="EYES"               wear={data.makeup.eyes.wear}  avoid={data.makeup.eyes.avoid} />
                <StyleCard style={{ flex: 1 }} title="BLUSH & FOUNDATION" wear={data.makeup.blush.wear} avoid={data.makeup.blush.avoid} />
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <PlaceholderCard label="Makeup example" />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense fallback={
      <div style={{ minHeight: "100vh", background: "#fff", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <p style={{ color: "#8c8c8c" }}>Loading results…</p>
      </div>
    }>
      <ResultsInner />
    </Suspense>
  );
}
