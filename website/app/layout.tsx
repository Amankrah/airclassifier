import type { Metadata } from 'next';
import './globals.css';
import { Header } from '@/components/shared/header';
import { Footer } from '@/components/shared/footer';

export const metadata: Metadata = {
  title: {
    default: 'ProteinProcessIO - Protein Processing Simulation',
    template: '%s | ProteinProcessIO',
  },
  description:
    'Complete protein processing simulation software - from raw seed to fractionated flour. Simulate pretreatment, milling, and air classification with GPU-accelerated physics.',
  keywords: [
    'protein processing',
    'simulation',
    'pretreatment',
    'milling',
    'air classification',
    'pulse protein',
    'legume processing',
    'food science',
    'process engineering',
  ],
  authors: [{ name: 'mvgill' }],
  creator: 'mvgill',
  publisher: 'ProteinProcessIO',
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://proteinprocessio.com',
    siteName: 'ProteinProcessIO',
    title: 'ProteinProcessIO - Protein Processing Simulation',
    description:
      'Complete protein processing simulation software - from raw seed to fractionated flour.',
    images: [
      {
        url: '/images/og-image.png',
        width: 1200,
        height: 630,
        alt: 'ProteinProcessIO',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'ProteinProcessIO - Protein Processing Simulation',
    description:
      'Complete protein processing simulation software - from raw seed to fractionated flour.',
    images: ['/images/og-image.png'],
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-1">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
