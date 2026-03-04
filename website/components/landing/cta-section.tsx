'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { Download, ArrowRight, BookOpen } from 'lucide-react';

export function CTASection() {
  return (
    <section className="section relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-primary/5 to-transparent" />

      <div className="container-custom relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="max-w-4xl mx-auto"
        >
          {/* Card */}
          <div className="relative glass-card p-8 md:p-12 text-center overflow-hidden">
            {/* Glow Effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-primary/10 via-accent/10 to-primary/10 opacity-50" />
            <div className="absolute -top-24 left-1/2 -translate-x-1/2 w-96 h-96 bg-accent/20 rounded-full blur-3xl" />

            <div className="relative z-10">
              {/* Headline */}
              <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-white mb-4">
                Ready to optimize your{' '}
                <span className="gradient-text">process?</span>
              </h2>

              <p className="text-lg text-text-secondary max-w-2xl mx-auto mb-8">
                Download ProteinProcessIO today and start simulating your protein
                processing operations. Free for research and academic use.
              </p>

              {/* CTA Buttons */}
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Link
                  href="/download"
                  className="btn-accent text-base px-8 py-3 group"
                >
                  <Download className="w-5 h-5" />
                  Download for Free
                  <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
                </Link>
                <Link href="/docs" className="btn-outline text-base px-8 py-3">
                  <BookOpen className="w-5 h-5" />
                  Read the Docs
                </Link>
              </div>

              {/* Trust Signals */}
              <div className="mt-10 pt-8 border-t border-white/10 flex flex-wrap items-center justify-center gap-8">
                <div className="text-center">
                  <div className="text-2xl font-bold text-white">100%</div>
                  <div className="text-sm text-text-muted">Free & Open</div>
                </div>
                <div className="w-px h-10 bg-white/10 hidden sm:block" />
                <div className="text-center">
                  <div className="text-2xl font-bold text-white">3</div>
                  <div className="text-sm text-text-muted">
                    Processing Stages
                  </div>
                </div>
                <div className="w-px h-10 bg-white/10 hidden sm:block" />
                <div className="text-center">
                  <div className="text-2xl font-bold text-white">GPU</div>
                  <div className="text-sm text-text-muted">Accelerated</div>
                </div>
                <div className="w-px h-10 bg-white/10 hidden sm:block" />
                <div className="text-center">
                  <div className="flex items-center gap-1 justify-center">
                    <span className="text-2xl font-bold text-white">NRC</span>
                  </div>
                  <div className="text-sm text-text-muted">Validated</div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
