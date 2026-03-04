'use client';

import { motion } from 'framer-motion';
import {
  Zap,
  Cpu,
  Box,
  BarChart3,
  Workflow,
  Target,
} from 'lucide-react';

const features = [
  {
    icon: Zap,
    title: 'Coupled Multi-Physics Engine',
    description:
      'Nine-step simulation loop coupling RF electromagnetic fields, heat conduction, moisture diffusion, and dielectric property updates. FDM Laplace solver with 100+ Lagrangian tracers for seed-level temperature and moisture tracking.',
    highlight: '9-Step Loop',
  },
  {
    icon: Cpu,
    title: 'GPU-Accelerated Computation',
    description:
      'Eight NVIDIA Warp GPU kernels with persistent memory allocation and batched launches — zero per-step allocations. Automatic CUDA/CPU device detection with adaptive CFL-based timestep control.',
    highlight: 'CUDA/Warp',
  },
  {
    icon: Box,
    title: 'Interactive 3D Digital Twin',
    description:
      'Over 40 parametric components with three cinematic camera modes — orbit, guided showcase tours, and spiral flythrough. Live particle rendering, mechanical animations, and per-component opacity controls.',
    highlight: '40+ Parts',
  },
  {
    icon: Workflow,
    title: 'Three-Stage Pipeline',
    description:
      'RF pretreatment, hammer milling, and air classification linked with automatic outlet-to-inlet state mapping. Multi-pass recirculation with attrition modeling and full mass balance tracking across stages.',
    highlight: 'End-to-End',
  },
  {
    icon: Target,
    title: 'Multi-Objective Optimization',
    description:
      'Derringer-Suich desirability scoring across five dimensions — thermal treatment, LOX inactivation, protein preservation, moisture retention, and energy efficiency. Grid search and gradient-based optimization via Warp Tape.',
    highlight: '5D Scoring',
  },
  {
    icon: BarChart3,
    title: 'Advanced Analytics & Export',
    description:
      'Interactive PSD charts with hover tooltips and log-scale toggle, real-time KPI dashboards with radial gauges, and dual-scale time-series plots. Export to VTK for 3D field post-processing, plus CSV, JSON, and NumPy formats.',
    highlight: 'VTK/CSV',
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
            Coupled multi-physics solvers, GPU-accelerated kernels, and
            experimentally validated models — from whole seed conditioning to
            micron-scale flour separation.
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
