/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable React strict mode for better development experience
  reactStrictMode: true,

  // Generate source maps in production for Lighthouse and debugging
  productionBrowserSourceMaps: true,

  // Image optimization
  images: {
    domains: ['localhost'],
    unoptimized: process.env.NODE_ENV === 'development',
  },

  // Redirect trailing slashes
  trailingSlash: false,

  async redirects() {
    return [
      { source: '/docs/manual', destination: '/docs', permanent: true },
      { source: '/docs/getting-started', destination: '/docs/getting-started/installation', permanent: true },
    ];
  },

  // Enable experimental features
  experimental: {
    // Enable optimized package imports
    optimizePackageImports: ['lucide-react', 'framer-motion'],
  },
};

module.exports = nextConfig;
