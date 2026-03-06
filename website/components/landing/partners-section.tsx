'use client';

import { motion } from 'framer-motion';
import Image from 'next/image';
import { ExternalLink } from 'lucide-react';

const SASEL_URL = 'https://sasellab.com/';

export function PartnersSection() {
  return (
    <section className="section relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-primary/5 to-transparent" />

      <div className="container-custom relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="max-w-4xl mx-auto text-center"
        >
          <h2 className="text-2xl md:text-3xl font-bold text-white mb-2">
            Partners & funding
          </h2>
          <p className="text-text-secondary mb-10">
            Developed at the Sustainable Agrifood Systems Engineering Lab (SASEL)
            at McGill University. Principal Investigator:{' '}
            <a
              href="https://www.mcgill.ca/bioeng/kwofie-ebenezer-miezah"
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent hover:underline"
            >
              Dr. Ebenezer Miezah Kwofie
            </a>
            . Funded by the National Research Council of Canada (NRC).
          </p>

          <div className="flex flex-wrap items-center justify-center gap-10 md:gap-14">
            {/* SASEL Lab — logo has its own black background */}
            <a
              href={SASEL_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="group flex flex-col items-center gap-2 text-text-secondary hover:text-white transition-colors"
              aria-label="SASEL Lab – Sustainable Agrifood Systems Engineering Lab at McGill"
            >
              <div className="relative h-28 w-52 md:h-32 md:w-60 flex items-center justify-center rounded-xl bg-white border border-white/20 group-hover:border-accent/40 transition-colors overflow-hidden">
                <Image
                  src="/images/logo.png"
                  alt="SASEL Lab – Sustainable Agrifood Systems Engineering Lab"
                  width={220}
                  height={120}
                  className="object-contain w-full h-full p-2"
                />
                <ExternalLink className="absolute top-2 right-2 w-4 h-4 text-accent opacity-0 group-hover:opacity-100 transition-opacity drop-shadow-md" />
              </div>
              <span className="text-xs text-text-muted">
                Sustainable Agrifood Systems Engineering Lab
              </span>
            </a>

            {/* McGill University — white background */}
            <a
              href="https://www.mcgill.ca/"
              target="_blank"
              rel="noopener noreferrer"
              className="group flex flex-col items-center gap-2"
              aria-label="McGill University"
            >
              <div className="relative h-28 w-52 md:h-32 md:w-60 flex items-center justify-center p-4 rounded-xl bg-white border border-white/20 group-hover:border-accent/40 transition-colors">
                <Image
                  src="/images/mcgill_sig_red.png"
                  alt="McGill University"
                  width={200}
                  height={80}
                  className="object-contain w-full h-full"
                />
              </div>
              <span className="text-xs text-text-muted">McGill University</span>
            </a>

            {/* NRC — white background */}
            <a
              href="https://nrc.canada.ca/"
              target="_blank"
              rel="noopener noreferrer"
              className="group flex flex-col items-center gap-2"
              aria-label="National Research Council Canada"
            >
              <div className="relative h-28 w-52 md:h-32 md:w-60 flex items-center justify-center p-4 rounded-xl bg-white border border-white/20 group-hover:border-accent/40 transition-colors">
                <Image
                  src="/images/nrc-cnrc-logo.png"
                  alt="National Research Council Canada"
                  width={200}
                  height={80}
                  className="object-contain w-full h-full"
                />
              </div>
              <span className="text-xs text-text-muted">NRC Canada</span>
            </a>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
