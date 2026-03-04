'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import {
  Zap,
  Cog,
  Wind,
  ArrowRight,
  Thermometer,
  Droplets,
  Gauge,
  BarChart3,
  Layers,
  Settings,
  Activity,
  Target,
} from 'lucide-react';

const stages = [
  {
    id: 'pretreatment',
    title: 'Pretreatment',
    subtitle: 'GP-15 RF Dielectric Heating',
    icon: Zap,
    color: 'from-orange-500 to-red-500',
    description:
      'Simulate the GP-15 RF dielectric heating machine for moisture control and enzyme inactivation. The pretreatment stage conditions whole seeds before milling.',
    features: [
      {
        icon: Thermometer,
        title: 'Temperature Control',
        description:
          'RF heating at 27.12 MHz with real-time temperature field visualization. Track sensor readings and bulk averages.',
      },
      {
        icon: Droplets,
        title: 'Moisture Management',
        description:
          'Monitor moisture content (wet basis) across the bed. Achieve target drying with configurable evaporation models.',
      },
      {
        icon: Gauge,
        title: 'Process Parameters',
        description:
          'Configure electrode gap, belt speed, RF power, and extraction fan settings. Match your actual GP-15 configuration.',
      },
      {
        icon: BarChart3,
        title: 'Energy Analytics',
        description:
          'Track total energy consumption (kWh), specific energy (kWh/kg water), and anode current. Optimize efficiency.',
      },
    ],
    specs: [
      { label: 'RF Frequency', value: '27.12 MHz' },
      { label: 'Max Power', value: '15 kW' },
      { label: 'Belt Width', value: '800 mm' },
      { label: 'Oven Length', value: '1.5 m' },
    ],
  },
  {
    id: 'milling',
    title: 'Milling',
    subtitle: 'Hammer Mill Simulation',
    icon: Cog,
    color: 'from-primary to-blue-400',
    description:
      'High-fidelity hammer mill simulation with GPU-accelerated particle physics. Break down whole seeds into flour with realistic size reduction.',
    features: [
      {
        icon: Layers,
        title: 'Particle Breakage',
        description:
          'Selection and breakage functions calibrated to legume comminution. Multi-fragment breakage with mass conservation.',
      },
      {
        icon: Settings,
        title: 'Screen Classification',
        description:
          'Configurable screen aperture (0.75-2.0 mm) with size-dependent passage probability. Realistic retention and recirculation.',
      },
      {
        icon: Activity,
        title: 'Real-time PSD',
        description:
          'Watch the particle size distribution evolve. Track D10, D50, D90 percentiles and full mass fractions.',
      },
      {
        icon: Gauge,
        title: 'Thermal Modeling',
        description:
          'Track product temperature rise from impact energy. Temperature-dependent breakage and stickiness effects.',
      },
    ],
    specs: [
      { label: 'Rotor Speed', value: '3,000-7,200 RPM' },
      { label: 'Screen Aperture', value: '0.75-2.0 mm' },
      { label: 'Motor Power', value: '15-55 kW' },
      { label: 'Target D50', value: '~24 µm' },
    ],
  },
  {
    id: 'classification',
    title: 'Air Classification',
    subtitle: 'Multi-Stage Separator',
    icon: Wind,
    color: 'from-accent to-cyan-300',
    description:
      'Complete air classification system with venturi eductor, zigzag channel, wheel classifier, and 3-stage cyclone. Separate flour into protein and starch fractions.',
    features: [
      {
        icon: Wind,
        title: 'Venturi Eductor',
        description:
          'Optional preclassification stage. Air entrainment of flour with configurable throat ratio and bypass.',
      },
      {
        icon: Layers,
        title: 'Zigzag Classifier',
        description:
          'Multi-stage zigzag channel for preliminary separation. Configure channel dimensions and number of stages.',
      },
      {
        icon: Target,
        title: 'Wheel Classifier',
        description:
          'High-speed rotating wheel (up to 3,000 RPM) for fine classification. Cut size determined by wheel speed and air flow.',
      },
      {
        icon: Activity,
        title: 'Cyclone System',
        description:
          '3-stage cyclone for fines collection. Primary, secondary, and tertiary cyclones with configurable diameters.',
      },
    ],
    specs: [
      { label: 'Wheel Speed', value: 'Up to 3,000 RPM' },
      { label: 'Air Flow', value: '1,000-5,000 m³/h' },
      { label: 'Throughput', value: '100-1,000 kg/h' },
      { label: 'Protein Yield', value: '>55%' },
    ],
  },
];

export default function FeaturesPage() {
  return (
    <main className="pt-24">
      {/* Hero Section */}
      <section className="section pb-12">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-3xl mx-auto text-center"
          >
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-6">
              Powerful Simulation{' '}
              <span className="gradient-text">Capabilities</span>
            </h1>
            <p className="text-xl text-text-secondary">
              Three integrated processing stages with physics-based models
              validated against real equipment and experimental data.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Stage Sections */}
      {stages.map((stage, index) => (
        <section
          key={stage.id}
          id={stage.id}
          className={`section ${index % 2 === 1 ? 'bg-bg-surface/50' : ''}`}
        >
          <div className="container-custom">
            <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
              {/* Content */}
              <motion.div
                initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                className={index % 2 === 1 ? 'lg:order-2' : ''}
              >
                {/* Badge */}
                <div
                  className={`inline-flex items-center gap-2 px-3 py-1 rounded-full bg-gradient-to-r ${stage.color} bg-opacity-10 mb-4`}
                >
                  <stage.icon className="w-4 h-4 text-white" />
                  <span className="text-sm font-medium text-white">
                    Stage {index + 1}
                  </span>
                </div>

                <h2 className="text-3xl md:text-4xl font-bold text-white mb-2">
                  {stage.title}
                </h2>
                <p className="text-lg text-accent mb-4">{stage.subtitle}</p>
                <p className="text-text-secondary mb-8">{stage.description}</p>

                {/* Feature Cards */}
                <div className="grid sm:grid-cols-2 gap-4 mb-8">
                  {stage.features.map((feature) => (
                    <div
                      key={feature.title}
                      className="p-4 rounded-xl bg-white/5 border border-white/10"
                    >
                      <feature.icon className="w-5 h-5 text-accent mb-2" />
                      <h3 className="text-sm font-semibold text-white mb-1">
                        {feature.title}
                      </h3>
                      <p className="text-xs text-text-muted">
                        {feature.description}
                      </p>
                    </div>
                  ))}
                </div>

                <Link
                  href={`/docs/${stage.id}`}
                  className="inline-flex items-center gap-2 text-accent hover:text-accent-glow transition-colors"
                >
                  View documentation
                  <ArrowRight className="w-4 h-4" />
                </Link>
              </motion.div>

              {/* Specs Card */}
              <motion.div
                initial={{ opacity: 0, x: index % 2 === 0 ? 20 : -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                className={index % 2 === 1 ? 'lg:order-1' : ''}
              >
                <div className="glass-card p-8 relative overflow-hidden">
                  {/* Gradient Background */}
                  <div
                    className={`absolute inset-0 bg-gradient-to-br ${stage.color} opacity-5`}
                  />

                  {/* Large Icon */}
                  <div className="relative z-10">
                    <div
                      className={`w-20 h-20 rounded-2xl bg-gradient-to-br ${stage.color} flex items-center justify-center mb-6`}
                    >
                      <stage.icon className="w-10 h-10 text-white" />
                    </div>

                    <h3 className="text-xl font-semibold text-white mb-6">
                      Technical Specifications
                    </h3>

                    <div className="space-y-4">
                      {stage.specs.map((spec) => (
                        <div
                          key={spec.label}
                          className="flex items-center justify-between py-3 border-b border-white/10 last:border-0"
                        >
                          <span className="text-text-secondary">
                            {spec.label}
                          </span>
                          <span className="font-mono text-white">
                            {spec.value}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </motion.div>
            </div>
          </div>
        </section>
      ))}

      {/* Pipeline Mode Section */}
      <section className="section bg-gradient-to-b from-transparent to-bg-surface/50">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="max-w-4xl mx-auto text-center"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Pipeline Orchestration
            </h2>
            <p className="text-lg text-text-secondary mb-8">
              Transfer data seamlessly between stages. Outlet conditions from one
              stage automatically configure the next.
            </p>

            {/* Pipeline Visualization */}
            <div className="flex items-center justify-center gap-4 flex-wrap">
              {stages.map((stage, i) => (
                <div key={stage.id} className="flex items-center gap-4">
                  <div
                    className={`w-16 h-16 rounded-xl bg-gradient-to-br ${stage.color} flex items-center justify-center`}
                  >
                    <stage.icon className="w-8 h-8 text-white" />
                  </div>
                  {i < stages.length - 1 && (
                    <ArrowRight className="w-6 h-6 text-text-muted" />
                  )}
                </div>
              ))}
            </div>

            <div className="mt-8">
              <Link href="/docs/pipeline" className="btn-accent">
                Learn about Pipeline Mode
                <ArrowRight className="w-4 h-4" />
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </main>
  );
}
