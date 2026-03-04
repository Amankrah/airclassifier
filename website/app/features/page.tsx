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
  Monitor,
  Palette,
  Keyboard,
  Camera,
  TrendingUp,
  FileOutput,
} from 'lucide-react';

const stages = [
  {
    id: 'pretreatment',
    title: 'Pretreatment',
    subtitle: 'GP-15 RF Dielectric Heating',
    icon: Zap,
    color: 'from-orange-500 to-red-500',
    description:
      'Full digital twin of the GP-15 RF dielectric heating oven with a coupled 9-step physics loop — RF field solve, volumetric heating, thermal conduction, moisture diffusion, evaporation kinetics, and material property updates every timestep. Validated against NRC Canada experimental runs with PLC data, temperature strips, and NIR moisture measurements.',
    features: [
      {
        icon: Thermometer,
        title: 'Coupled RF Field Solver',
        description:
          'Red-Black Gauss-Seidel SOR for the Laplace equation with variable permittivity per cell. Voltage-constrained iteration maintains target RF power density at 27.12 MHz.',
      },
      {
        icon: Droplets,
        title: 'Moisture & Thermal Coupling',
        description:
          "Fick's law diffusion with Arrhenius-type effective diffusivity D_eff(T). Latent heat sink from evaporation. Moisture-dependent thermal conductivity via the Luikov model.",
      },
      {
        icon: Gauge,
        title: 'Adaptive PLC Control',
        description:
          'Electrode gap control with hysteresis, belt speed regulation, temperature setpoint tracking, and arc detection safety interlocks. Replicates the real GP-15 PLC logic.',
      },
      {
        icon: BarChart3,
        title: 'Desirability Scoring',
        description:
          'Five-dimensional Derringer-Suich functions — thermal treatment, LOX inactivation, protein preservation, moisture retention, and energy efficiency. Material-specific profiles for yellow pea, faba bean, and red lentil.',
      },
    ],
    specs: [
      { label: 'RF Frequency', value: '27.12 MHz' },
      { label: 'Field Solver', value: 'FDM Laplace (GPU)' },
      { label: 'Lagrangian Tracers', value: '100+ particles' },
      { label: 'Material Presets', value: '3 legumes' },
    ],
  },
  {
    id: 'milling',
    title: 'Milling',
    subtitle: 'Hammer Mill Simulation',
    icon: Cog,
    color: 'from-primary to-blue-400',
    description:
      'Energy-based comminution model with calibrated selection and breakage functions. Particles undergo transport, hammer impact, multi-fragment breakage, and aperture-dependent screen classification. Validated against NIH hammer mill trials — simulated D50 of 23.6 µm vs measured 23.7 µm at 6,000 RPM with 0.75 mm screen.',
    features: [
      {
        icon: Layers,
        title: 'Calibrated Breakage Model',
        description:
          'Velocity-dependent selection function with Rosin-Rammler daughter distributions. Parameters (k=0.6, d_ref=300 µm) validated against NIH legume comminution data with mass conservation.',
      },
      {
        icon: Settings,
        title: 'Screen Classification',
        description:
          'Aperture-based cutoff with (1-t)^4 passage taper. Size-ratio threshold filtering and velocity-dependent passage probability model realistic retention and discharge dynamics.',
      },
      {
        icon: Activity,
        title: 'Live PSD Evolution',
        description:
          'Interactive chart with hover tooltips, cumulative distribution toggle, and logarithmic X-axis. D10, D50, D90 percentiles update in real-time as product discharges.',
      },
      {
        icon: Gauge,
        title: 'Housing Thermal Model',
        description:
          '50 kg steel housing thermal mass prevents unrealistic temperature oscillations. Friction heating from rotor impacts balanced by ambient convective cooling.',
      },
    ],
    specs: [
      { label: 'Rotor Speed', value: '3,000-7,200 RPM' },
      { label: 'Screen Aperture', value: '0.75-2.0 mm' },
      { label: 'Validated D50', value: '23.6 µm (NIH: 23.7)' },
      { label: 'Breakage Model', value: 'v4 calibrated' },
    ],
  },
  {
    id: 'classification',
    title: 'Air Classification',
    subtitle: 'Multi-Stage Separator',
    icon: Wind,
    color: 'from-accent to-cyan-300',
    description:
      'Lagrangian particle tracking through a complete separation train — venturi eductor, zigzag preclassifier, high-speed wheel classifier, and 3-stage cyclone system. Two operating modes (full system or wheel-only) with configurable bypass ratio and multi-pass recirculation with attrition modeling.',
    features: [
      {
        icon: Wind,
        title: 'Lagrangian Particle Tracking',
        description:
          'Schiller-Naumann drag for spherical particles, Haider-Levenspiel for non-spherical. Gravity, buoyancy, and inelastic wall collisions with configurable restitution and friction coefficients.',
      },
      {
        icon: Layers,
        title: 'Multi-Path Configuration',
        description:
          'Full system (venturi → zigzag → dropout → wheel → cyclones → bag filter) or wheel-only mode. Adjustable bypass ratio (0-100%) for fine-tuning the separation cut point.',
      },
      {
        icon: Target,
        title: 'Wheel Classifier Physics',
        description:
          'Centrifugal force up to 5,000g at blade tips. Cut size emerges from the balance of centrifugal and aerodynamic drag forces acting on each individual particle.',
      },
      {
        icon: Activity,
        title: 'Multi-Stage Cyclone',
        description:
          'Primary, secondary, and tertiary cyclones with grade efficiency curves. Fines collection with bag filter exhaust cleaning. Configurable cyclone diameters and inlet velocities.',
      },
    ],
    specs: [
      { label: 'Wheel Speed', value: 'Up to 3,000 RPM' },
      { label: 'Drag Model', value: 'Schiller-Naumann' },
      { label: 'Cyclone Stages', value: '3 + bag filter' },
      { label: 'Recirculation', value: 'Multi-pass' },
    ],
  },
];

const guiFeatures = [
  {
    icon: Monitor,
    title: 'Glassmorphic Dark Theme',
    description:
      'Semi-transparent cards with backdrop blur, semantic color coding for every KPI, and smooth 400ms eased transitions throughout the interface.',
  },
  {
    icon: TrendingUp,
    title: 'Animated KPI Dashboards',
    description:
      'Radial gauges with threshold coloring, sparkline trend lines with Bezier interpolation, and delta badges showing real-time percentage changes.',
  },
  {
    icon: Camera,
    title: 'Cinematic 3D Camera',
    description:
      'Three camera modes — smooth orbit, guided 5-keyframe showcase tour, and spiral flythrough. Mouse interaction pauses the camera for manual inspection.',
  },
  {
    icon: Palette,
    title: '40+ Parametric Components',
    description:
      'Cyclones, blowers, wheel classifiers, ductwork, dampers, explosion vents, and instrumentation ports — all rendered with per-component color mapping and opacity controls.',
  },
  {
    icon: Keyboard,
    title: '20+ Keyboard Shortcuts',
    description:
      'F5 to run, F6 to pause, Ctrl+B to build, Ctrl+R for results. Waveform timeline with drag-to-scrub playback and speed control (0.5x to 4x).',
  },
  {
    icon: FileOutput,
    title: 'Multi-Format Export',
    description:
      'VTK structured grids for 3D field post-processing, CSV time-series, JSON for automation, and NumPy snapshots. One-click export from the results view.',
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
              Three integrated processing stages with coupled multi-physics
              models, GPU-accelerated solvers, and parameters validated against
              NRC Canada and NIH experimental data.
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

      {/* Professional GUI Section */}
      <section className="section bg-bg-surface/50">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="max-w-3xl mx-auto text-center mb-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Professional Desktop Experience
            </h2>
            <p className="text-lg text-text-secondary">
              A polished PySide6 application with glassmorphic design, animated
              dashboards, and a cinematic 3D viewport — built for engineers who
              care about their tools.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6"
          >
            {guiFeatures.map((feature) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="p-6 rounded-xl bg-white/5 border border-white/10 hover:border-accent/30 transition-colors"
              >
                <feature.icon className="w-6 h-6 text-accent mb-3" />
                <h3 className="text-sm font-semibold text-white mb-1">
                  {feature.title}
                </h3>
                <p className="text-xs text-text-muted">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

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
              Outlet temperature and moisture from pretreatment automatically map
              to milling feed conditions. Milling PSD flows into the classifier
              inlet. Multi-pass recirculation with attrition modeling and full
              mass balance tracking across all three stages.
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
