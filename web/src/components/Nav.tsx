"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

interface NavProps {
  variant?: "light" | "dark";
  showFullLinks?: boolean;
}

export default function Nav({ variant = "light", showFullLinks = true }: NavProps) {
  const isDark = variant === "dark";
  const [visible, setVisible]   = useState(true);
  const [lastY, setLastY]       = useState(0);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    let ticking = false;
    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        const currentY = window.scrollY;
        if (currentY < 10)          setVisible(true);
        else if (currentY > lastY + 4) setVisible(false);
        else if (currentY < lastY - 4) setVisible(true);
        setLastY(currentY);
        ticking = false;
      });
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [lastY]);

  // Close menu on route change / resize
  useEffect(() => {
    const close = () => setMenuOpen(false);
    window.addEventListener("resize", close);
    return () => window.removeEventListener("resize", close);
  }, []);

  const linkCls = `text-[17px] font-normal hover:opacity-60 transition-opacity ${isDark ? "text-white/80" : "text-black"}`;
  const bgCls   = isDark
    ? "bg-black border-b border-white/10 text-white"
    : "bg-[#fafaf8] shadow-[0px_4.875px_7.313px_rgba(0,0,0,0.12)] text-[#141412]";

  return (
    <>
      <nav
        className={`fixed top-0 left-0 right-0 z-50 w-full transition-transform duration-300 ease-in-out ${
          visible ? "translate-y-0" : "-translate-y-full"
        } ${bgCls}`}
      >
        <div className="flex items-center justify-between px-5 md:px-[58.5px] h-[64px] md:h-[88px]">
          {/* Logo */}
          <Link
            href="/"
            className={`text-[17px] md:text-[19.5px] font-semibold tracking-tight leading-none ${isDark ? "text-white" : "text-[#141412]"}`}
          >
            ToneFit AI
          </Link>

          {/* Desktop links */}
          {showFullLinks ? (
            <div className="hidden md:flex items-center gap-10">
              <Link href="/#about"        className={linkCls}>About ToneFit</Link>
              <Link href="/#how-it-works" className={linkCls}>How It Works</Link>
              <Link href="/compare"       className={linkCls}>Model Comparison</Link>
              <Link
                href="/onboarding"
                className={`px-[29px] h-[53.6px] rounded-full text-[17px] font-medium transition-all flex items-center whitespace-nowrap ${
                  isDark ? "bg-white text-black hover:bg-white/90" : "bg-[#1a1a1a] text-white hover:bg-neutral-800"
                }`}
              >
                Discover Your Color Palette
              </Link>
            </div>
          ) : (
            <Link href="/#how-it-works" className={`hidden md:block ${linkCls}`}>
              How It Works
            </Link>
          )}

          {/* Mobile hamburger */}
          <button
            onClick={() => setMenuOpen((o) => !o)}
            className="md:hidden flex flex-col gap-[5px] p-2"
            aria-label="Toggle menu"
          >
            <span className={`block w-6 h-0.5 transition-all duration-200 ${isDark ? "bg-white" : "bg-black"} ${menuOpen ? "rotate-45 translate-y-[7px]" : ""}`} />
            <span className={`block w-6 h-0.5 transition-all duration-200 ${isDark ? "bg-white" : "bg-black"} ${menuOpen ? "opacity-0" : ""}`} />
            <span className={`block w-6 h-0.5 transition-all duration-200 ${isDark ? "bg-white" : "bg-black"} ${menuOpen ? "-rotate-45 -translate-y-[7px]" : ""}`} />
          </button>
        </div>

        {/* Mobile dropdown menu */}
        {menuOpen && (
          <div className={`md:hidden border-t ${isDark ? "border-white/10 bg-black" : "border-black/8 bg-[#fafaf8]"} px-5 py-4 flex flex-col gap-4`}>
            {showFullLinks && (
              <>
                <Link href="/#about"        className={linkCls} onClick={() => setMenuOpen(false)}>About ToneFit</Link>
                <Link href="/#how-it-works" className={linkCls} onClick={() => setMenuOpen(false)}>How It Works</Link>
                <Link href="/compare"       className={linkCls} onClick={() => setMenuOpen(false)}>Model Comparison</Link>
              </>
            )}
            {!showFullLinks && (
              <Link href="/#how-it-works" className={linkCls} onClick={() => setMenuOpen(false)}>How It Works</Link>
            )}
            <Link
              href="/onboarding"
              onClick={() => setMenuOpen(false)}
              className={`px-6 py-3 rounded-full text-[15px] font-medium text-center transition-all ${
                isDark ? "bg-white text-black" : "bg-[#1a1a1a] text-white"
              }`}
            >
              Discover Your Color Palette
            </Link>
          </div>
        )}
      </nav>
    </>
  );
}
