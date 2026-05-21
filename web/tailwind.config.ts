import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
      },
      colors: {
        spring: { DEFAULT: "#E8856A", light: "#FDDECA", dark: "#C4614A" },
        summer: { DEFAULT: "#8BAEC4", light: "#D8E8F4", dark: "#5A85A0" },
        autumn: { DEFAULT: "#C4773A", light: "#F4DFC0", dark: "#8A4E1E" },
        winter: { DEFAULT: "#2D3E6E", light: "#C8D4F0", dark: "#0A1832" },
      },
    },
  },
  plugins: [],
};

export default config;
