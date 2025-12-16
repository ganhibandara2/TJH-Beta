import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Header from "./components/Header";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "TJH Job Hunter",
  description: "A polished job search UI for JSearch and Indeed.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-linear-to-br from-[#020617] via-[#020617] to-[#0f172a] text-zinc-50 min-h-screen`}
      >
        <div className="relative overflow-hidden">
          <div className="pointer-events-none fixed inset-0 -z-10">
            <div className="absolute -top-40 -left-40 h-80 w-80 rounded-full bg-blue-500/20 blur-3xl" />
            <div className="absolute top-1/2 -right-40 h-96 w-96 rounded-full bg-emerald-500/10 blur-3xl" />
            <div className="absolute -bottom-40 left-1/2 h-72 w-72 -translate-x-1/2 rounded-full bg-indigo-500/20 blur-3xl" />
          </div>

          <Header />

          <main>{children}</main>

          <footer className="border-t border-white/5 bg-black/30 backdrop-blur-sm">
            <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-3 px-4 py-4 text-xs text-zinc-500 sm:flex-row sm:px-6 lg:px-8">
              <p>Built for technical job hunters.</p>
              <p className="text-[11px]">
                Powered by{" "}
                <span className="font-medium text-zinc-300">TJH </span>{" "}
              </p>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
