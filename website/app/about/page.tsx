'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import {
  Linkedin,
  Mail,
  ExternalLink,
  Target,
  Users,
  Microscope,
  Award,
} from 'lucide-react';

const timeline = [
  {
    year: '2023',
    title: 'Project Inception',
    description:
      'Initial concept and requirements gathering. Partnership with NRC Canada established.',
  },
  {
    year: '2024 Q1',
    title: 'Pretreatment Module',
    description:
      'GP-15 RF dielectric heating simulation developed and validated against experimental data.',
  },
  {
    year: '2024 Q2',
    title: 'Milling Module',
    description:
      'Hammer mill simulation with GPU-accelerated particle physics. Breakage models calibrated.',
  },
  {
    year: '2024 Q3',
    title: 'Air Classification',
    description:
      'Multi-stage classification system including venturi, zigzag, wheel, and cyclones.',
  },
  {
    year: '2024 Q4',
    title: 'Pipeline Integration',
    description:
      'Orchestration mode enabling seamless data transfer between stages. Mass balance tracking.',
  },
  {
    year: '2025',
    title: 'Public Release',
    description:
      'Version 1.0 released for research and academic use. Ongoing development continues.',
  },
];

const values = [
  {
    icon: Target,
    title: 'Accuracy',
    description:
      'Physics-based models validated against real equipment and experimental data from NRC Canada.',
  },
  {
    icon: Users,
    title: 'Accessibility',
    description:
      'Free for research and academic use. Empowering scientists and engineers worldwide.',
  },
  {
    icon: Microscope,
    title: 'Innovation',
    description:
      'GPU-accelerated simulations pushing the boundaries of what\'s possible in process modeling.',
  },
  {
    icon: Award,
    title: 'Quality',
    description:
      'Comprehensive documentation, regular updates, and responsive support for all users.',
  },
];

export default function AboutPage() {
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
              Built for <span className="gradient-text">Protein Science</span>
            </h1>
            <p className="text-xl text-text-secondary">
              ProteinProcessIO is a comprehensive simulation platform for protein
              processing operations, developed at McGill University in collaboration
              with the National Research Council Canada.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Mission Section */}
      <section className="section pt-0">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="max-w-4xl mx-auto"
          >
            <div className="glass-card p-8 md:p-12">
              <h2 className="text-2xl md:text-3xl font-bold text-white mb-6">
                Our Mission
              </h2>
              <p className="text-lg text-text-secondary mb-6">
                To accelerate innovation in plant protein processing by providing
                researchers and engineers with powerful, validated simulation
                tools. We believe that better process understanding leads to more
                efficient operations, higher quality products, and a more
                sustainable food system.
              </p>
              <p className="text-text-secondary">
                ProteinProcessIO enables virtual experimentation, reducing the
                need for costly physical trials while providing deep insights
                into process dynamics that would be impossible to observe in
                real equipment.
              </p>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Values Grid */}
      <section className="section bg-bg-surface/50">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Our Values
            </h2>
            <p className="text-text-secondary max-w-2xl mx-auto">
              The principles that guide our development and support
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {values.map((value, index) => (
              <motion.div
                key={value.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="feature-card text-center"
              >
                <div className="w-14 h-14 mx-auto rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center mb-4">
                  <value.icon className="w-7 h-7 text-accent" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  {value.title}
                </h3>
                <p className="text-sm text-text-secondary">{value.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section id="team" className="section">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="max-w-4xl mx-auto"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white text-center mb-12">
              Meet the Team
            </h2>

            <div className="grid gap-6 sm:grid-cols-2">
              {/* Ebenezer Miezah Kwofie - PI */}
              <div className="glass-card p-6">
                <div className="flex flex-col sm:flex-row items-center gap-4">
                  <div className="w-20 h-20 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center flex-shrink-0">
                    <span className="text-2xl font-bold text-white">EMK</span>
                  </div>
                  <div className="text-center sm:text-left flex-1">
                    <h3 className="text-lg font-bold text-white mb-1">
                      <a
                        href="https://www.mcgill.ca/bioeng/kwofie-ebenezer-miezah"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="hover:text-accent transition-colors"
                      >
                        Ebenezer Miezah Kwofie
                      </a>
                    </h3>
                    <p className="text-accent text-sm mb-2">Principal Investigator</p>
                    <p className="text-text-secondary text-sm mb-3">
                      Assistant Professor, Sustainable Food Systems Engineering at
                      McGill University. Co-director, Integrated Food and Bioprocess
                      Engineering Program.
                    </p>
                    <a
                      href="https://www.mcgill.ca/bioeng/kwofie-ebenezer-miezah"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-accent hover:text-accent-glow text-sm"
                    >
                      McGill profile <ExternalLink className="w-3.5 h-3.5" />
                    </a>
                  </div>
                </div>
              </div>

              {/* Emmanuel Amankrah Kwofie */}
              <div className="glass-card p-6">
                <div className="flex flex-col sm:flex-row items-center gap-4">
                  <div className="w-20 h-20 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center flex-shrink-0">
                    <span className="text-2xl font-bold text-white">EAK</span>
                  </div>
                  <div className="text-center sm:text-left flex-1">
                    <h3 className="text-lg font-bold text-white mb-1">
                      Emmanuel Amankrah Kwofie
                    </h3>
                    <p className="text-accent text-sm mb-2">Lead Developer &amp; Engineer</p>
                    <p className="text-text-secondary text-sm mb-3">
                      Process engineer and software developer at McGill University.
                      Specializing in CFD, particle physics, GPU computing, and
                      multiphysics simulation.
                    </p>
                    <div className="flex items-center justify-center sm:justify-start gap-2">
                      <a
                        href="https://www.eakwofie.com/"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-text-secondary hover:text-white transition-colors"
                        title="Personal Website"
                      >
                        <ExternalLink className="w-4 h-4" />
                      </a>
                      <a
                        href="https://linkedin.com/in/eakwofie"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-text-secondary hover:text-white transition-colors"
                        aria-label="LinkedIn"
                      >
                        <Linkedin className="w-4 h-4" />
                      </a>
                      <a
                        href="mailto:emmanuel.kwofie@mail.mcgill.ca"
                        className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-text-secondary hover:text-white transition-colors"
                        aria-label="Email"
                      >
                        <Mail className="w-4 h-4" />
                      </a>
                    </div>
                  </div>
                </div>
              </div>

              {/* Derrick Allotey */}
              <div className="glass-card p-6">
                <div className="flex flex-col sm:flex-row items-center gap-4">
                  <div className="w-20 h-20 rounded-xl bg-gradient-to-br from-primary/80 to-accent/80 flex items-center justify-center flex-shrink-0">
                    <span className="text-2xl font-bold text-white">DA</span>
                  </div>
                  <div className="text-center sm:text-left flex-1">
                    <h3 className="text-lg font-bold text-white mb-1">Derrick Allotey</h3>
                    <p className="text-accent text-sm mb-2">PhD Student</p>
                    <p className="text-text-secondary text-sm">
                      SASEL Lab, McGill University. Worked on this project.
                    </p>
                  </div>
                </div>
              </div>

              {/* Anthony Iheonye */}
              <div className="glass-card p-6">
                <div className="flex flex-col sm:flex-row items-center gap-4">
                  <div className="w-20 h-20 rounded-xl bg-gradient-to-br from-primary/80 to-accent/80 flex items-center justify-center flex-shrink-0">
                    <span className="text-2xl font-bold text-white">AI</span>
                  </div>
                  <div className="text-center sm:text-left flex-1">
                    <h3 className="text-lg font-bold text-white mb-1">Anthony Iheonye</h3>
                    <p className="text-accent text-sm mb-2">Postdoctoral Researcher</p>
                    <p className="text-text-secondary text-sm">
                      SASEL Lab, McGill University. Worked on this project.
                    </p>
                  </div>
                </div>
              </div>

              {/* Abrahim Sabiku */}
              <div className="glass-card p-6">
                <div className="flex flex-col sm:flex-row items-center gap-4">
                  <div className="w-20 h-20 rounded-xl bg-gradient-to-br from-primary/80 to-accent/80 flex items-center justify-center flex-shrink-0">
                    <span className="text-2xl font-bold text-white">AS</span>
                  </div>
                  <div className="text-center sm:text-left flex-1">
                    <h3 className="text-lg font-bold text-white mb-1">Abrahim Sabiku</h3>
                    <p className="text-accent text-sm mb-2">Masters Student</p>
                    <p className="text-text-secondary text-sm">
                      SASEL Lab, McGill University. Worked on this project.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* McGill University Section */}
      <section id="mcgill" className="section bg-bg-surface/50">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="max-w-4xl mx-auto"
          >
            <div className="glass-card p-8 md:p-12">
              <div className="flex flex-col md:flex-row items-center gap-8">
                {/* McGill Logo Placeholder */}
                <div className="w-32 h-32 rounded-2xl bg-white/10 flex items-center justify-center flex-shrink-0">
                  <div className="text-center">
                    <span className="text-2xl font-bold text-red-500">McGill</span>
                    <p className="text-xs text-text-muted mt-1">University</p>
                  </div>
                </div>

                {/* Info */}
                <div>
                  <h2 className="text-2xl font-bold text-white mb-4">
                    McGill University
                  </h2>
                  <p className="text-text-secondary mb-6">
                    ProteinProcessIO was developed at McGill University, one of
                    Canada's leading research universities. The project combines
                    expertise in food science, process engineering, and computational
                    modeling to create cutting-edge simulation tools for the plant
                    protein industry.
                  </p>
                  <a
                    href="https://www.mcgill.ca"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 text-accent hover:text-accent-glow transition-colors"
                  >
                    Visit McGill University
                    <ExternalLink className="w-4 h-4" />
                  </a>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* NRC Canada Section */}
      <section id="nrc" className="section">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="max-w-4xl mx-auto"
          >
            <div className="glass-card p-8 md:p-12">
              <div className="flex flex-col md:flex-row items-center gap-8">
                {/* NRC Logo Placeholder */}
                <div className="w-32 h-32 rounded-2xl bg-white/10 flex items-center justify-center flex-shrink-0">
                  <div className="text-center">
                    <span className="text-2xl font-bold text-accent">NRC</span>
                    <p className="text-xs text-text-muted mt-1">Canada</p>
                  </div>
                </div>

                {/* Info */}
                <div>
                  <h2 className="text-2xl font-bold text-white mb-4">
                    National Research Council Canada
                  </h2>
                  <p className="text-text-secondary mb-6">
                    ProteinProcessIO was developed in collaboration with NRC
                    Canada, leveraging their expertise in food processing
                    research and access to pilot-scale equipment for model
                    validation. This partnership ensures that our simulations
                    are grounded in real-world data and validated against
                    experimental results.
                  </p>
                  <a
                    href="https://nrc.canada.ca"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 text-accent hover:text-accent-glow transition-colors"
                  >
                    Visit NRC Canada
                    <ExternalLink className="w-4 h-4" />
                  </a>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Timeline Section */}
      <section className="section">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Development Timeline
            </h2>
            <p className="text-text-secondary max-w-2xl mx-auto">
              The journey from concept to release
            </p>
          </motion.div>

          <div className="max-w-3xl mx-auto">
            <div className="relative">
              {/* Timeline Line */}
              <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gradient-to-b from-primary via-accent to-primary" />

              {/* Timeline Items */}
              <div className="space-y-8">
                {timeline.map((item, index) => (
                  <motion.div
                    key={item.year}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.1 }}
                    className="relative pl-20"
                  >
                    {/* Dot */}
                    <div className="absolute left-6 w-5 h-5 rounded-full bg-bg-dark border-2 border-accent" />

                    {/* Content */}
                    <div className="glass-card p-6">
                      <div className="flex items-center gap-3 mb-2">
                        <span className="px-2 py-0.5 rounded text-xs font-medium bg-accent/10 text-accent">
                          {item.year}
                        </span>
                        <h3 className="text-lg font-semibold text-white">
                          {item.title}
                        </h3>
                      </div>
                      <p className="text-text-secondary text-sm">
                        {item.description}
                      </p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="section bg-gradient-to-b from-transparent to-bg-surface/50">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="max-w-2xl mx-auto text-center"
          >
            <h2 className="text-3xl font-bold text-white mb-4">
              Get in Touch
            </h2>
            <p className="text-text-secondary mb-8">
              Have questions about ProteinProcessIO? Interested in collaboration?
              We'd love to hear from you.
            </p>
            <Link href="/contact" className="btn-accent">
              Contact Us
            </Link>
          </motion.div>
        </div>
      </section>
    </main>
  );
}
