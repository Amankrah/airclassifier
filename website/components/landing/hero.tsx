'use client';

import { motion } from 'framer-motion';
import Image from 'next/image';
import Link from 'next/link';
import { ArrowRight, Download, BookOpen, Sparkles } from 'lucide-react';

export function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-gradient-hero" />
      <div className="absolute inset-0 bg-gradient-glow" />

      {/* Animated Grid Background */}
      <div className="absolute inset-0 opacity-20">
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `
              linear-gradient(rgba(37, 99, 235, 0.1) 1px, transparent 1px),
              linear-gradient(90deg, rgba(37, 99, 235, 0.1) 1px, transparent 1px)
            `,
            backgroundSize: '60px 60px',
          }}
        />
      </div>

      {/* Floating Particles Animation */}
      <div className="absolute inset-0 overflow-hidden">
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-2 h-2 rounded-full bg-accent/30"
            initial={{
              x: Math.random() * (typeof window !== 'undefined' ? window.innerWidth : 1000),
              y: Math.random() * (typeof window !== 'undefined' ? window.innerHeight : 800),
            }}
            animate={{
              y: [null, -100],
              opacity: [0, 1, 0],
            }}
            transition={{
              duration: 3 + Math.random() * 2,
              repeat: Infinity,
              delay: Math.random() * 2,
            }}
          />
        ))}
      </div>

      <div className="container-custom relative z-10">
        <div className="max-w-4xl mx-auto text-center">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 mb-8"
          >
            <Sparkles className="w-4 h-4 text-accent" />
            <span className="text-sm text-text-secondary">
              In collaboration with{' '}
              <span className="text-accent font-medium">NRC Canada</span>
            </span>
          </motion.div>

          {/* Main Headline */}
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight text-balance"
          >
            <span className="text-white">Simulate.</span>{' '}
            <span className="gradient-text">Optimize.</span>{' '}
            <span className="text-white">Process.</span>
          </motion.h1>

          {/* Subheadline */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="mt-6 text-xl md:text-2xl text-text-secondary max-w-2xl mx-auto text-balance"
          >
            Complete protein processing simulation — from raw seed to
            fractionated flour. Pretreatment, milling, and air classification in
            one powerful desktop application.
          </motion.p>

          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link
              href="/download"
              className="btn-accent text-base px-8 py-3 group"
            >
              <Download className="w-5 h-5" />
              Download Now
              <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
            </Link>
            <Link href="/docs" className="btn-outline text-base px-8 py-3">
              <BookOpen className="w-5 h-5" />
              View Documentation
            </Link>
          </motion.div>

          {/* Platform Badges */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.5 }}
            className="mt-8 flex items-center justify-center gap-6 text-text-muted text-sm"
          >
            <span className="flex items-center gap-2">
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                <path d="M0 3.449L9.75 2.1v9.451H0m10.949-9.602L24 0v11.4H10.949M0 12.6h9.75v9.451L0 20.699M10.949 12.6H24V24l-12.9-1.801" />
              </svg>
              Windows
            </span>
            <span className="flex items-center gap-2">
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.477 2 2 6.477 2 12s4.477 10 10 10 10-4.477 10-10S17.523 2 12 2zm0 18c-4.418 0-8-3.582-8-8s3.582-8 8-8 8 3.582 8 8-3.582 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z" />
              </svg>
              macOS
            </span>
            <span className="flex items-center gap-2">
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12.504 0c-.155 0-.315.008-.48.021-4.226.333-3.105 4.807-3.17 6.298-.076 1.092-.3 1.953-1.05 3.02-.885 1.051-2.127 2.75-2.716 4.521-.278.832-.41 1.684-.287 2.489.117.779.456 1.511 1.096 2.182l.096.091c.217.18.424.333.602.468.17.136.309.271.423.403.287.332.457.779.457 1.326 0 .742-.309 1.342-.882 1.697-.317.194-.632.339-.88.4-.238.058-.463.137-.615.232-.304.19-.474.455-.514.727-.042.276.017.556.144.812.254.511.72.936 1.267 1.257.546.32 1.171.55 1.8.686a6.34 6.34 0 0 0 1.484.179c1.031 0 1.893-.218 2.454-.569.27-.17.474-.379.609-.605.135-.228.2-.47.2-.705 0-.432-.188-.765-.469-1.028-.142-.133-.308-.252-.497-.365-.19-.114-.406-.221-.608-.332a3.695 3.695 0 0 1-.504-.33c-.144-.111-.256-.235-.33-.369-.146-.268-.133-.529.04-.76.174-.23.466-.434.82-.633.353-.2.768-.395 1.168-.609.4-.213.788-.448 1.092-.732.608-.57.894-1.301.894-2.12 0-.68-.211-1.284-.595-1.817-.385-.533-.922-.99-1.544-1.383-.623-.394-1.331-.726-2.07-1.004-.74-.278-1.51-.502-2.25-.678-.742-.176-1.454-.303-2.079-.387-.624-.085-1.16-.127-1.555-.134l-.114-.001c-.023 0-.039.001-.058.001-.064 0-.127.004-.191.01-.063.006-.129.016-.193.029a1.516 1.516 0 0 0-.314.088 1.095 1.095 0 0 0-.44.302 1.015 1.015 0 0 0-.222.381c-.045.145-.069.299-.069.457 0 .157.024.31.069.455.045.145.119.277.222.381.103.104.231.188.38.246.15.058.322.087.513.087.064 0 .127-.004.191-.01.064-.006.129-.016.193-.029a1.516 1.516 0 0 0 .314-.088 1.095 1.095 0 0 0 .44-.302 1.015 1.015 0 0 0 .222-.381c.045-.145.069-.299.069-.457 0-.157-.024-.31-.069-.455a1.015 1.015 0 0 0-.222-.381 1.095 1.095 0 0 0-.38-.246 1.43 1.43 0 0 0-.513-.087c-.166 0-.328.029-.476.088s-.281.148-.387.26a1.165 1.165 0 0 0-.256.397c-.062.152-.093.32-.093.502 0 .183.031.35.093.503.062.152.148.286.256.397.108.112.239.2.387.26.148.059.31.088.476.088.167 0 .328-.029.477-.088a1.2 1.2 0 0 0 .386-.26c.108-.111.194-.245.256-.397.062-.153.093-.32.093-.503 0-.183-.031-.35-.093-.502a1.165 1.165 0 0 0-.256-.397 1.2 1.2 0 0 0-.386-.26 1.413 1.413 0 0 0-.477-.088z" />
              </svg>
              Linux
            </span>
          </motion.div>
        </div>

        {/* Hero Image / Screenshot */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mt-16 md:mt-20 relative"
        >
          <div className="relative mx-auto max-w-5xl">
            {/* Glow Effect */}
            <div className="absolute -inset-4 bg-gradient-to-r from-primary/20 via-accent/20 to-primary/20 rounded-2xl blur-2xl opacity-50" />

            {/* Screenshot Container */}
            <div className="relative glass-card p-2 rounded-2xl overflow-hidden">
              {/* Window Chrome */}
              <div className="flex items-center gap-2 px-4 py-3 bg-bg-elevated rounded-t-xl border-b border-white/10">
                <div className="flex gap-2">
                  <div className="w-3 h-3 rounded-full bg-red-500/80" />
                  <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                  <div className="w-3 h-3 rounded-full bg-green-500/80" />
                </div>
                <span className="text-xs text-text-muted ml-2">
                  ProteinProcessIO — Air Classification Simulation
                </span>
              </div>

              {/* App screenshot */}
              <div className="aspect-[16/10] relative bg-bg-surface">
                <Image
                  src="/images/hero-screenshot.png"
                  alt="ProteinProcessIO application window showing air classification simulation"
                  fill
                  className="object-contain object-top"
                  sizes="(max-width: 1024px) 100vw, 1024px"
                  priority
                />
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Scroll Indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2"
      >
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="w-6 h-10 rounded-full border-2 border-white/20 flex items-start justify-center p-2"
        >
          <div className="w-1.5 h-1.5 rounded-full bg-white/50" />
        </motion.div>
      </motion.div>
    </section>
  );
}
