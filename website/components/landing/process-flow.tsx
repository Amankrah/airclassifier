'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { ArrowRight, Zap, Cog, Wind } from 'lucide-react';

const stages = [
  {
    id: 'pretreatment',
    title: 'Pretreatment',
    subtitle: 'GP-15 RF Heating',
    description:
      'Dielectric heating for moisture control and enzyme inactivation. Simulates RF frequency, electrode gap, and belt conveyor dynamics.',
    icon: Zap,
    color: 'from-orange-500 to-red-500',
    href: '/features#pretreatment',
    metrics: ['Temperature profiles', 'Moisture control', 'Energy consumption'],
  },
  {
    id: 'milling',
    title: 'Milling',
    subtitle: 'Hammer Mill',
    description:
      'Impact grinding simulation with real-time particle size distribution. Configure rotor speed, screen aperture, and breakage parameters.',
    icon: Cog,
    color: 'from-primary to-blue-400',
    href: '/features#milling',
    metrics: ['PSD analysis', 'Power consumption', 'Throughput rates'],
  },
  {
    id: 'classification',
    title: 'Classification',
    subtitle: 'Air Classifier',
    description:
      'Particle separation by size and density. Includes venturi eductor, zigzag channel, wheel classifier, and cyclone stages.',
    icon: Wind,
    color: 'from-accent to-cyan-300',
    href: '/features#classification',
    metrics: ['Separation efficiency', 'Protein yield', 'Particle trajectories'],
  },
];

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.2,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 30 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5 },
  },
};

export function ProcessFlow() {
  return (
    <section className="section relative overflow-hidden">
      {/* Background Decoration */}
      <div className="absolute inset-0 bg-gradient-glow opacity-30" />

      <div className="container-custom relative z-10">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center max-w-3xl mx-auto mb-16"
        >
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-white mb-4">
            Complete Processing Pipeline
          </h2>
          <p className="text-lg text-text-secondary">
            Simulate the entire protein fractionation process. Transfer data
            seamlessly between stages with automatic parameter mapping.
          </p>
        </motion.div>

        {/* Process Flow Cards */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-100px' }}
          className="grid md:grid-cols-3 gap-6 md:gap-8"
        >
          {stages.map((stage, index) => (
            <motion.div
              key={stage.id}
              variants={itemVariants}
              className="relative group"
            >
              {/* Connection Line */}
              {index < stages.length - 1 && (
                <div className="hidden md:block absolute top-1/2 -right-4 w-8 h-0.5 bg-gradient-to-r from-white/20 to-transparent z-10">
                  <ArrowRight className="absolute -right-1 -top-2 w-4 h-4 text-white/30" />
                </div>
              )}

              <Link href={stage.href}>
                <div className="feature-card h-full relative overflow-hidden">
                  {/* Gradient Overlay */}
                  <div
                    className={`absolute inset-0 bg-gradient-to-br ${stage.color} opacity-0 group-hover:opacity-5 transition-opacity duration-300`}
                  />

                  {/* Stage Number */}
                  <div className="absolute top-4 right-4 w-8 h-8 rounded-full bg-white/5 flex items-center justify-center">
                    <span className="text-sm font-bold text-text-muted">
                      {index + 1}
                    </span>
                  </div>

                  {/* Icon */}
                  <div
                    className={`w-14 h-14 rounded-xl bg-gradient-to-br ${stage.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}
                  >
                    <stage.icon className="w-7 h-7 text-white" />
                  </div>

                  {/* Content */}
                  <h3 className="text-xl font-bold text-white mb-1">
                    {stage.title}
                  </h3>
                  <p className="text-sm text-accent mb-3">{stage.subtitle}</p>
                  <p className="text-text-secondary text-sm mb-4">
                    {stage.description}
                  </p>

                  {/* Metrics */}
                  <div className="space-y-2 pt-4 border-t border-white/10">
                    {stage.metrics.map((metric) => (
                      <div
                        key={metric}
                        className="flex items-center gap-2 text-xs text-text-muted"
                      >
                        <div className="w-1.5 h-1.5 rounded-full bg-accent/50" />
                        {metric}
                      </div>
                    ))}
                  </div>

                  {/* Learn More */}
                  <div className="mt-4 flex items-center gap-1 text-sm text-accent group-hover:text-accent-glow transition-colors">
                    Learn more
                    <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </motion.div>

        {/* Pipeline Transfer Feature */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.4 }}
          className="mt-12 text-center"
        >
          <div className="inline-flex items-center gap-3 px-6 py-3 rounded-full bg-white/5 border border-white/10">
            <div className="flex -space-x-1">
              {stages.map((stage, i) => (
                <div
                  key={stage.id}
                  className={`w-6 h-6 rounded-full bg-gradient-to-br ${stage.color} border-2 border-bg-dark`}
                  style={{ zIndex: stages.length - i }}
                />
              ))}
            </div>
            <span className="text-sm text-text-secondary">
              <span className="text-white font-medium">Pipeline Mode:</span>{' '}
              Automatic data transfer between stages
            </span>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
