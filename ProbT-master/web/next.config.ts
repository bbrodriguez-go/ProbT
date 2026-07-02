import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // Proxy /api/* to the engine service at runtime (Cloud Run)
  // Also allows NEXT_PUBLIC_API_URL override at build time
  async rewrites() {
    const api = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${api}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
