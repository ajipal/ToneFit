import type { Metadata } from "next";
import { Inter, DM_Serif_Display, Cormorant } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const dmSerifDisplay = DM_Serif_Display({
  subsets: ["latin"],
  weight: "400",
  style: ["normal", "italic"],
  variable: "--font-dm-serif",
});
const cormorant = Cormorant({
  subsets: ["latin"],
  weight: ["700"],
  style: ["normal", "italic"],
  variable: "--font-cormorant",
});

export const metadata: Metadata = {
  title: "ToneFit AI — Discover Your Personal Color Season",
  description: "AI-powered personal color season analysis and complete style guide",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.variable} ${dmSerifDisplay.variable} ${cormorant.variable} font-sans antialiased`}>{children}</body>
    </html>
  );
}
