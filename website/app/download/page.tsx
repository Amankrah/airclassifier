'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Download,
  Monitor,
  Apple,
  Terminal,
  ChevronDown,
  ChevronUp,
  Check,
  AlertCircle,
  Cpu,
  HardDrive,
  MemoryStick,
} from 'lucide-react';
import { detectOS } from '@/lib/utils';

type Platform = 'windows' | 'macos' | 'linux';

const platforms: Record<Platform, {
  name: string;
  icon: typeof Monitor;
  version: string;
  size: string;
  filename: string;
  requirements: string;
}> = {
  windows: {
    name: 'Windows',
    icon: Monitor,
    version: 'Windows 10/11 (64-bit)',
    size: '~150 MB',
    filename: 'ProteinProcessIO-1.0.0-Setup.exe',
    requirements: 'Windows 10 or later',
  },
  macos: {
    name: 'macOS',
    icon: Apple,
    version: 'macOS 11+ (Intel & Apple Silicon)',
    size: '~160 MB',
    filename: 'ProteinProcessIO-1.0.0-macos.dmg',
    requirements: 'macOS 11 Big Sur or later',
  },
  linux: {
    name: 'Linux',
    icon: Terminal,
    version: 'Ubuntu 20.04+, Fedora 34+',
    size: '~145 MB',
    filename: 'ProteinProcessIO-1.0.0-linux.AppImage',
    requirements: 'glibc 2.31 or later',
  },
};

const systemRequirements = {
  minimum: [
    { icon: Cpu, label: 'Processor', value: 'Intel Core i5 / AMD Ryzen 5' },
    { icon: MemoryStick, label: 'Memory', value: '8 GB RAM' },
    { icon: HardDrive, label: 'Storage', value: '500 MB available space' },
    { icon: Monitor, label: 'Display', value: '1920x1080 resolution' },
  ],
  recommended: [
    { icon: Cpu, label: 'Processor', value: 'Intel Core i7 / AMD Ryzen 7' },
    { icon: MemoryStick, label: 'Memory', value: '16 GB RAM' },
    { icon: HardDrive, label: 'Storage', value: '1 GB available space' },
    { icon: Cpu, label: 'GPU', value: 'NVIDIA RTX 2060+ (CUDA 11.0+)' },
  ],
};

const installationSteps: Record<Platform, string[]> = {
  windows: [
    'Download the installer (.exe file)',
    'Run the installer (you may need to click "More info" → "Run anyway" if Windows SmartScreen appears)',
    'Follow the installation wizard - choose install location and shortcuts',
    'Launch ProteinProcessIO from Start Menu or Desktop shortcut',
  ],
  macos: [
    'Download the disk image (.dmg file)',
    'Open the .dmg file',
    'Drag ProteinProcessIO to Applications',
    'Launch from Applications folder',
  ],
  linux: [
    'Download the AppImage file',
    'Make it executable: chmod +x ProteinProcessIO*.AppImage',
    'Run the AppImage',
    'Optional: Add to applications menu',
  ],
};

const changelog = [
  {
    version: '1.0.0',
    date: '2025-03-01',
    changes: [
      'Initial release',
      'Pretreatment (GP-15) simulation',
      'Hammer mill simulation with GPU acceleration',
      'Air classification with 3-stage cyclone',
      'Pipeline orchestration mode',
      'Real-time 3D visualization',
      'Mass balance tracking',
    ],
  },
];

export default function DownloadPage() {
  const [selectedPlatform, setSelectedPlatform] = useState<Platform>('windows');
  const [showInstallGuide, setShowInstallGuide] = useState(false);

  useEffect(() => {
    const detected = detectOS();
    if (detected !== 'unknown') {
      setSelectedPlatform(detected);
    }
  }, []);

  const currentPlatform = platforms[selectedPlatform];

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
              Download <span className="gradient-text">ProteinProcessIO</span>
            </h1>
            <p className="text-xl text-text-secondary">
              Free for research and academic use. Get started with protein
              processing simulation today.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Download Section */}
      <section className="section pt-0">
        <div className="container-custom">
          <div className="max-w-4xl mx-auto">
            {/* Platform Tabs */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="flex justify-center gap-2 mb-8"
            >
              {(Object.keys(platforms) as Platform[]).map((platform) => {
                const PlatformIcon = platforms[platform].icon;
                return (
                  <button
                    key={platform}
                    onClick={() => setSelectedPlatform(platform)}
                    className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${
                      selectedPlatform === platform
                        ? 'bg-accent text-bg-dark'
                        : 'bg-white/5 text-text-secondary hover:bg-white/10 hover:text-white'
                    }`}
                  >
                    <PlatformIcon className="w-5 h-5" />
                    {platforms[platform].name}
                  </button>
                );
              })}
            </motion.div>

            {/* Download Card */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="glass-card p-8 md:p-12 text-center"
            >
              <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-primary to-accent flex items-center justify-center">
                <currentPlatform.icon className="w-10 h-10 text-white" />
              </div>

              <h2 className="text-2xl font-bold text-white mb-2">
                {currentPlatform.name}
              </h2>
              <p className="text-text-secondary mb-6">
                {currentPlatform.version}
              </p>

              <a
                href={`/downloads/${currentPlatform.filename}`}
                className="btn-accent text-lg px-10 py-4 mb-6 inline-flex"
              >
                <Download className="w-5 h-5" />
                Download ({currentPlatform.size})
              </a>

              <p className="text-sm text-text-muted">
                {currentPlatform.filename}
              </p>
            </motion.div>

            {/* Installation Guide */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="mt-8"
            >
              <button
                onClick={() => setShowInstallGuide(!showInstallGuide)}
                className="w-full flex items-center justify-between p-4 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
              >
                <span className="font-medium text-white">
                  Installation Guide
                </span>
                {showInstallGuide ? (
                  <ChevronUp className="w-5 h-5 text-text-muted" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-text-muted" />
                )}
              </button>

              {showInstallGuide && (
                <div className="mt-4 p-6 rounded-lg bg-bg-surface border border-white/10">
                  <ol className="space-y-4">
                    {installationSteps[selectedPlatform].map((step, index) => (
                      <li key={index} className="flex items-start gap-3">
                        <div className="w-6 h-6 rounded-full bg-accent/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                          <span className="text-xs font-bold text-accent">
                            {index + 1}
                          </span>
                        </div>
                        <span className="text-text-secondary">{step}</span>
                      </li>
                    ))}
                  </ol>
                </div>
              )}
            </motion.div>
          </div>
        </div>
      </section>

      {/* System Requirements */}
      <section className="section bg-bg-surface/50">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="max-w-4xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-white text-center mb-12">
              System Requirements
            </h2>

            <div className="grid md:grid-cols-2 gap-8">
              {/* Minimum */}
              <div className="glass-card p-6">
                <div className="flex items-center gap-2 mb-6">
                  <AlertCircle className="w-5 h-5 text-yellow-500" />
                  <h3 className="text-lg font-semibold text-white">Minimum</h3>
                </div>
                <div className="space-y-4">
                  {systemRequirements.minimum.map((req) => (
                    <div
                      key={req.label}
                      className="flex items-center gap-3 text-sm"
                    >
                      <req.icon className="w-4 h-4 text-text-muted" />
                      <span className="text-text-muted">{req.label}:</span>
                      <span className="text-text-secondary">{req.value}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Recommended */}
              <div className="glass-card p-6 border-accent/30">
                <div className="flex items-center gap-2 mb-6">
                  <Check className="w-5 h-5 text-green-500" />
                  <h3 className="text-lg font-semibold text-white">
                    Recommended
                  </h3>
                </div>
                <div className="space-y-4">
                  {systemRequirements.recommended.map((req) => (
                    <div
                      key={req.label}
                      className="flex items-center gap-3 text-sm"
                    >
                      <req.icon className="w-4 h-4 text-accent" />
                      <span className="text-text-muted">{req.label}:</span>
                      <span className="text-text-secondary">{req.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <p className="text-center text-text-muted text-sm mt-8">
              GPU acceleration requires NVIDIA GPU with CUDA 11.0 or later.
              CPU-only mode available for all simulations.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Changelog */}
      <section id="changelog" className="section">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="max-w-3xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-white text-center mb-12">
              Release Notes
            </h2>

            <div className="space-y-8">
              {changelog.map((release) => (
                <div key={release.version} className="glass-card p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-semibold text-white">
                      v{release.version}
                    </h3>
                    <span className="text-sm text-text-muted">
                      {release.date}
                    </span>
                  </div>
                  <ul className="space-y-2">
                    {release.changes.map((change, i) => (
                      <li
                        key={i}
                        className="flex items-center gap-2 text-text-secondary"
                      >
                        <div className="w-1.5 h-1.5 rounded-full bg-accent" />
                        {change}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>
    </main>
  );
}
