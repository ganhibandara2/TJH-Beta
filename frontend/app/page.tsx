"use client";

import { useLayoutEffect } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();

  useLayoutEffect(() => {
    // Check for authentication token
    const token = localStorage.getItem("access_token");

    // Redirect based on authentication status
    if (token) {
      router.push("/search");
    } else {
      router.push("/login");
    }
  }, [router]);

  // Return a minimal loading state or nothing while redirecting
  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-zinc-500 border-t-zinc-200"></div>
    </div>
  );
}
