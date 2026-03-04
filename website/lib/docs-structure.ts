// Static documentation structure - safe to use in client components

export interface DocMeta {
  title: string;
  description: string;
  slug: string;
  order?: number;
}

export interface DocSection {
  title: string;
  slug: string;
  items: DocMeta[];
}

// Documentation structure
export const docsStructure: DocSection[] = [
  {
    title: 'Getting Started',
    slug: 'getting-started',
    items: [
      { title: 'Installation', description: 'Install ProteinProcessIO on your system', slug: 'installation', order: 1 },
      { title: 'Quick Start', description: 'Get up and running in 5 minutes', slug: 'quick-start', order: 2 },
      { title: 'First Simulation', description: 'Run your first protein processing simulation', slug: 'first-simulation', order: 3 },
    ],
  },
  {
    title: 'Pretreatment',
    slug: 'pretreatment',
    items: [
      { title: 'GP-15 Overview', description: 'Understanding RF dielectric heating', slug: 'overview', order: 1 },
      { title: 'Configuration', description: 'Configure pretreatment parameters', slug: 'configuration', order: 2 },
      { title: 'Running Simulations', description: 'Execute pretreatment simulations', slug: 'simulation', order: 3 },
      { title: 'Interpreting Results', description: 'Analyze pretreatment outputs', slug: 'results', order: 4 },
    ],
  },
  {
    title: 'Milling',
    slug: 'milling',
    items: [
      { title: 'Hammer Mill Overview', description: 'Understanding hammer mill simulation', slug: 'overview', order: 1 },
      { title: 'Configuration', description: 'Configure milling parameters', slug: 'configuration', order: 2 },
      { title: 'Running Simulations', description: 'Execute milling simulations', slug: 'simulation', order: 3 },
      { title: 'PSD Analysis', description: 'Particle size distribution analysis', slug: 'psd', order: 4 },
    ],
  },
  {
    title: 'Air Classification',
    slug: 'classification',
    items: [
      { title: 'System Overview', description: 'Multi-stage air classification system', slug: 'overview', order: 1 },
      { title: 'Configuration', description: 'Configure classifier parameters', slug: 'configuration', order: 2 },
      { title: 'Running Simulations', description: 'Execute classification simulations', slug: 'simulation', order: 3 },
      { title: 'Separation Efficiency', description: 'Analyze separation results', slug: 'efficiency', order: 4 },
    ],
  },
  {
    title: 'Pipeline Mode',
    slug: 'pipeline',
    items: [
      { title: 'Orchestration Overview', description: 'Multi-stage process orchestration', slug: 'overview', order: 1 },
      { title: 'Stage Transfers', description: 'Transfer data between stages', slug: 'transfers', order: 2 },
      { title: 'Mass Balance', description: 'Track mass through the pipeline', slug: 'mass-balance', order: 3 },
    ],
  },
  {
    title: 'Reference',
    slug: 'reference',
    items: [
      { title: 'Keyboard Shortcuts', description: 'Master the application shortcuts', slug: 'shortcuts', order: 1 },
      { title: 'File Formats', description: 'Project file and export formats', slug: 'file-formats', order: 2 },
      { title: 'Configuration Reference', description: 'Complete configuration options', slug: 'config', order: 3 },
    ],
  },
];

export function getAllDocSlugs(): { section: string; slug: string }[] {
  const slugs: { section: string; slug: string }[] = [];

  for (const section of docsStructure) {
    for (const item of section.items) {
      slugs.push({ section: section.slug, slug: item.slug });
    }
  }

  return slugs;
}

export function getDocNavigation(currentSection: string, currentSlug: string) {
  let prev: { section: string; slug: string; title: string } | null = null;
  let next: { section: string; slug: string; title: string } | null = null;

  const allDocs = getAllDocSlugs();
  const currentIndex = allDocs.findIndex(
    (d) => d.section === currentSection && d.slug === currentSlug
  );

  if (currentIndex > 0) {
    const prevDoc = allDocs[currentIndex - 1];
    const prevSection = docsStructure.find((s) => s.slug === prevDoc.section);
    const prevItem = prevSection?.items.find((i) => i.slug === prevDoc.slug);
    if (prevItem) {
      prev = { section: prevDoc.section, slug: prevDoc.slug, title: prevItem.title };
    }
  }

  if (currentIndex < allDocs.length - 1) {
    const nextDoc = allDocs[currentIndex + 1];
    const nextSection = docsStructure.find((s) => s.slug === nextDoc.section);
    const nextItem = nextSection?.items.find((i) => i.slug === nextDoc.slug);
    if (nextItem) {
      next = { section: nextDoc.section, slug: nextDoc.slug, title: nextItem.title };
    }
  }

  return { prev, next };
}
