"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

interface NavProps {
  variant?: "light" | "dark";
  showFullLinks?: boolean;
}

export default function Nav({ variant = "light", showFullLinks = true }: NavProps) {
  const isDark = variant === "dark";
  const [visible, setVisible] = useState(true);
  const [lastY, setLastY] = useState(0);

  useEffect(() => {
    let ticking = false;

    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        const currentY = window.scrollY;
        // Always show at the very top
        if (currentY < 10) {
          setVisible(true);
        } else if (currentY > lastY + 4) {
          // Scrolling down — hide
          setVisible(false);
        } else if (currentY < lastY - 4) {
          // Scrolling up — show
          setVisible(true);
        }
        setLastY(currentY);
        ticking = false;
      });
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [lastY]);

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 w-full flex items-center justify-between px-[58.5px] h-[88px] transition-transform duration-300 ease-in-out ${
        visible ? "translate-y-0" : "-translate-y-full"
      } ${
        isDark
          ? "bg-black border-b border-white/10 text-white"
          : "bg-[#fafaf8] shadow-[0px_4.875px_7.313px_rgba(0,0,0,0.12)] text-[#141412]"
      }`}
    >
      <Link
        href="/"
        className={`text-[19.5px] font-semibold tracking-tight leading-none ${isDark ? "text-white" : "text-[#141412]"}`}
      >
        ToneFit AI
      </Link>

      {showFullLinks ? (
        <div className="flex items-center gap-10">
          <Link
            href="/#about"
            className={`text-[17px] font-normal hover:opacity-60 transition-opacity ${isDark ? "text-white/80" : "text-black"}`}
          >
            About ToneFit
          </Link>
          <Link
            href="/#how-it-works"
            className={`text-[17px] font-normal hover:opacity-60 transition-opacity ${isDark ? "text-white/80" : "text-black"}`}
          >
            How It Works
          </Link>
          <Link
            href="/compare"
            className={`text-[17px] font-normal hover:opacity-60 transition-opacity ${isDark ? "text-white/80" : "text-black"}`}
          >
            Model Comparison
          </Link>
          <Link
            href="/onboarding"
            className={`px-[29px] h-[53.6px] rounded-full text-[17px] font-medium transition-all flex items-center whitespace-nowrap ${
              isDark
                ? "bg-white text-black hover:bg-white/90"
                : "bg-[#1a1a1a] text-white hover:bg-neutral-800"
            }`}
          >
            Discover Your Color Palette
          </Link>
        </div>
      ) : (
        <Link
          href="/#how-it-works"
          className={`text-[17px] font-normal hover:opacity-60 transition-opacity ${isDark ? "text-white/80" : "text-black"}`}
        >
          How It Works
        </Link>
      )}
    </nav>
  );
}
