'use client';

import { motion } from 'framer-motion';
import {
  Cpu,
  Box,
  LineChart,
  Workflow,
  Gauge,
  FileJson,
} from 'lucide-react';

const features = [
  {
    icon: Cpu,
    title: 'GPU-Accelerated Physics',
    description:
      'CUDA-powered simulations using NVIDIA Warp for real-time particle dynamics. Thousands of particles simulated simultaneously.',
    highlight: 'CUDA/Warp',
  },
  {
    icon: Box,
    title: 'Real-time 3D Visualization',
    description:
      'Interactive PyVista-based viewport with animated components. Watch particles flow through the system in real-time.',
    highlight: 'PyVista',
  },
  {
    icon: LineChart,
    title: 'Comprehensive Analytics',
    description:
      'Time-series plots, PSD analysis, separation efficiency metrics. Export results to CSV, JSON, or PDF reports.',
    highlight: 'Export Ready',
  },
  {
    icon: Workflow,
    title: 'Pipeline Orchestration',
    description:
      'Transfer outlet conditions between stages automatically. Mass balance tracking ensures process consistency.',
    highlight: 'New!',
  },
  {
    icon: Gauge,
    title: 'Process Optimization',
    description:
      'Calibrated physics models based on real machine data. Validated against NRC Canada experimental results.',
    highlight: 'Validated',
  },
  {
    icon: FileJson,
    title: 'Project Files',
    description:
      'Save and load complete simulation configurations. Share projects with colleagues or resume work later.',
    highlight: '.acproj',
  },
];

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.4 },
  },
};

export function FeaturesGrid() {
  return (
    <section className="section bg-bg-surface/50">
      <div className="container-custom">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center max-w-3xl mx-auto mb-16"
        >
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-white mb-4">
            Powerful Features
          </h2>
          <p className="text-lg text-text-secondary">
            Everything you need to simulate, analyze, and optimize protein
            processing operations.
          </p>
        </motion.div>

        {/* Features Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-50px' }}
          className="grid md:grid-cols-2 lg:grid-cols-3 gap-6"
        >
          {features.map((feature) => (
            <motion.div
              key={feature.title}
              variants={itemVariants}
              className="feature-card group"
            >
              {/* Icon */}
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center mb-4 group-hover:from-primary/30 group-hover:to-accent/30 transition-colors">
                <feature.icon className="w-6 h-6 text-accent" />
              </div>

              {/* Content */}
              <div className="flex items-start justify-between gap-4 mb-2">
                <h3 className="text-lg font-semibold text-white">
                  {feature.title}
                </h3>
                <span className="px-2 py-0.5 rounded text-xs font-medium bg-accent/10 text-accent whitespace-nowrap">
                  {feature.highlight}
                </span>
              </div>
              <p className="text-text-secondary text-sm">{feature.description}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
