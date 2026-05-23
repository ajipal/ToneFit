import Link from "next/link";

interface NavProps {
  variant?: "light" | "dark";
  showFullLinks?: boolean;
}

export default function Nav({ variant = "light", showFullLinks = true }: NavProps) {
  const isDark = variant === "dark";
  return (
    <nav
      className={`w-full flex items-center justify-between px-14 h-[88px] border-b ${
        isDark
          ? "bg-black border-white/10 text-white"
          : "bg-white border-black/8 text-black"
      }`}
    >
      <Link
        href="/"
        className={`text-xl font-bold tracking-tight ${isDark ? "text-white" : "text-black"}`}
      >
        ToneFit AI
      </Link>

      {showFullLinks ? (
        <div className="flex items-center gap-10">
          <Link
            href="/#about"
            className={`text-[15px] font-medium hover:opacity-60 transition-opacity ${isDark ? "text-white/80" : "text-neutral-600"}`}
          >
            What is ToneFit
          </Link>
          <Link
            href="/#how-it-works"
            className={`text-[15px] font-medium hover:opacity-60 transition-opacity ${isDark ? "text-white/80" : "text-neutral-600"}`}
          >
            How It Works
          </Link>
          <Link
            href="/compare"
            className={`text-[15px] font-medium hover:opacity-60 transition-opacity ${isDark ? "text-white/80" : "text-neutral-600"}`}
          >
            Model Comparison
          </Link>
          <Link
            href="/onboarding"
            className={`px-6 py-3 rounded-xl text-[15px] font-semibold transition-all ${
              isDark
                ? "bg-white text-black hover:bg-white/90"
                : "bg-black text-white hover:bg-neutral-800"
            }`}
          >
            Discover Your Color Palette
          </Link>
        </div>
      ) : (
        <Link
          href="/#how-it-works"
          className={`text-[15px] font-medium hover:opacity-60 transition-opacity ${isDark ? "text-white/80" : "text-neutral-600"}`}
        >
          How It Works
        </Link>
      )}
    </nav>
  );
}
