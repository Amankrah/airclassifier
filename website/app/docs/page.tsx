'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import {
  BookOpen,
  Rocket,
  Zap,
  Cog,
  Wind,
  Workflow,
  Keyboard,
  FileJson,
  ArrowRight,
} from 'lucide-react';

const sections = [
  {
    title: 'Getting Started',
    description: 'Installation, quick start guide, and your first simulation',
    icon: Rocket,
    color: 'from-green-500 to-emerald-500',
    links: [
      { title: 'Installation', href: '/docs/getting-started/installation' },
      { title: 'Quick Start', href: '/docs/getting-started/quick-start' },
      { title: 'First Simulation', href: '/docs/getting-started/first-simulation' },
    ],
  },
  {
    title: 'Pretreatment',
    description: 'GP-15 RF dielectric heating simulation guide',
    icon: Zap,
    color: 'from-orange-500 to-red-500',
    links: [
      { title: 'GP-15 Overview', href: '/docs/pretreatment/overview' },
      { title: 'Configuration', href: '/docs/pretreatment/configuration' },
      { title: 'Running Simulations', href: '/docs/pretreatment/simulation' },
      { title: 'Interpreting Results', href: '/docs/pretreatment/results' },
    ],
  },
  {
    title: 'Milling',
    description: 'Hammer mill simulation and PSD analysis',
    icon: Cog,
    color: 'from-primary to-blue-400',
    links: [
      { title: 'Hammer Mill Overview', href: '/docs/milling/overview' },
      { title: 'Configuration', href: '/docs/milling/configuration' },
      { title: 'Running Simulations', href: '/docs/milling/simulation' },
      { title: 'PSD Analysis', href: '/docs/milling/psd' },
    ],
  },
  {
    title: 'Air Classification',
    description: 'Multi-stage air classifier simulation',
    icon: Wind,
    color: 'from-accent to-cyan-300',
    links: [
      { title: 'System Overview', href: '/docs/classification/overview' },
      { title: 'Configuration', href: '/docs/classification/configuration' },
      { title: 'Running Simulations', href: '/docs/classification/simulation' },
      { title: 'Separation Efficiency', href: '/docs/classification/efficiency' },
    ],
  },
  {
    title: 'Pipeline Mode',
    description: 'Orchestrating multi-stage simulations',
    icon: Workflow,
    color: 'from-purple-500 to-pink-500',
    links: [
      { title: 'Orchestration Overview', href: '/docs/pipeline/overview' },
      { title: 'Stage Transfers', href: '/docs/pipeline/transfers' },
      { title: 'Mass Balance', href: '/docs/pipeline/mass-balance' },
    ],
  },
  {
    title: 'Reference',
    description: 'Keyboard shortcuts, file formats, and API reference',
    icon: BookOpen,
    color: 'from-gray-500 to-gray-400',
    links: [
      { title: 'Keyboard Shortcuts', href: '/docs/reference/shortcuts' },
      { title: 'File Formats', href: '/docs/reference/file-formats' },
      { title: 'Configuration Reference', href: '/docs/reference/config' },
    ],
  },
];

export default function DocsPage() {
  return (
    <div>
      {/* Hero Section */}
      <section className="pb-8">
        <div className="max-w-4xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-3xl mx-auto text-center"
          >
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-6">
              <span className="gradient-text">Documentation</span>
            </h1>
            <p className="text-xl text-text-secondary">
              Everything you need to get started with ProteinProcessIO and master
              protein processing simulation.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Quick Links */}
      <section className="section pt-0">
        <div className="container-custom">
          <div className="max-w-6xl mx-auto">
            {/* Quick Start Banner */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-12"
            >
              <Link href="/docs/getting-started/quick-start">
                <div className="glass-card p-6 md:p-8 flex flex-col md:flex-row items-center justify-between gap-6 group hover:border-accent/30 transition-colors">
                  <div className="flex items-center gap-4">
                    <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center">
                      <Rocket className="w-7 h-7 text-white" />
                    </div>
                    <div>
                      <h2 className="text-xl font-bold text-white mb-1">
                        Quick Start Guide
                      </h2>
                      <p className="text-text-secondary">
                        Get up and running in 5 minutes
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 text-accent group-hover:text-accent-glow transition-colors">
                    Start learning
                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                  </div>
                </div>
              </Link>
            </motion.div>

            {/* Documentation Sections */}
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {sections.map((section, index) => (
                <motion.div
                  key={section.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="glass-card p-6"
                >
                  {/* Header */}
                  <div className="flex items-center gap-3 mb-4">
                    <div
                      className={`w-10 h-10 rounded-lg bg-gradient-to-br ${section.color} flex items-center justify-center`}
                    >
                      <section.icon className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">
                        {section.title}
                      </h3>
                    </div>
                  </div>

                  <p className="text-sm text-text-secondary mb-4">
                    {section.description}
                  </p>

                  {/* Links */}
                  <ul className="space-y-2">
                    {section.links.map((link) => (
                      <li key={link.href}>
                        <Link
                          href={link.href}
                          className="flex items-center gap-2 text-sm text-text-muted hover:text-white transition-colors group"
                        >
                          <div className="w-1.5 h-1.5 rounded-full bg-white/30 group-hover:bg-accent transition-colors" />
                          {link.title}
                        </Link>
                      </li>
                    ))}
                  </ul>
                </motion.div>
              ))}
            </div>

            {/* Additional Resources */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="mt-12 grid md:grid-cols-2 gap-6"
            >
              <div className="glass-card p-6 flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center">
                  <Keyboard className="w-5 h-5 text-text-muted" />
                </div>
                <div>
                  <h3 className="font-semibold text-white mb-1">
                    Keyboard Shortcuts
                  </h3>
                  <p className="text-sm text-text-secondary mb-2">
                    Master the app with keyboard shortcuts for faster workflow.
                  </p>
                  <Link
                    href="/docs/reference/shortcuts"
                    className="text-sm text-accent hover:text-accent-glow transition-colors"
                  >
                    View all shortcuts →
                  </Link>
                </div>
              </div>

              <div className="glass-card p-6 flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center">
                  <FileJson className="w-5 h-5 text-text-muted" />
                </div>
                <div>
                  <h3 className="font-semibold text-white mb-1">
                    Project Files (.acproj)
                  </h3>
                  <p className="text-sm text-text-secondary mb-2">
                    Learn about project file format and sharing configurations.
                  </p>
                  <Link
                    href="/docs/reference/file-formats"
                    className="text-sm text-accent hover:text-accent-glow transition-colors"
                  >
                    Learn more →
                  </Link>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>
    </div>
  );
}
