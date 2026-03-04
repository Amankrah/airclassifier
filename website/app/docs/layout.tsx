'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, Menu, X, BookOpen } from 'lucide-react';
import { docsStructure } from '@/lib/docs-structure';

function SidebarSection({
  section,
  isOpen,
  onToggle,
  currentPath,
}: {
  section: (typeof docsStructure)[0];
  isOpen: boolean;
  onToggle: () => void;
  currentPath: string;
}) {
  return (
    <div className="mb-2">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-3 py-2 text-sm font-medium text-text-secondary hover:text-white transition-colors rounded-lg hover:bg-white/5"
      >
        <span>{section.title}</span>
        <ChevronDown
          className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
        />
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <ul className="mt-1 ml-3 border-l border-white/10 pl-3 space-y-1">
              {section.items.map((item) => {
                const href = `/docs/${section.slug}/${item.slug}`;
                const isActive = currentPath === href;

                return (
                  <li key={item.slug}>
                    <Link
                      href={href}
                      className={`block px-3 py-1.5 text-sm rounded-md transition-colors ${
                        isActive
                          ? 'bg-primary/20 text-accent border-l-2 border-accent -ml-[13px] pl-[23px]'
                          : 'text-text-muted hover:text-white hover:bg-white/5'
                      }`}
                    >
                      {item.title}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function DocsSidebar({
  className = '',
  onLinkClick,
}: {
  className?: string;
  onLinkClick?: () => void;
}) {
  const pathname = usePathname();
  const [openSections, setOpenSections] = useState<Record<string, boolean>>(() => {
    // Open the section that contains the current page
    const initial: Record<string, boolean> = {};
    docsStructure.forEach((section) => {
      const isCurrentSection = section.items.some(
        (item) => pathname === `/docs/${section.slug}/${item.slug}`
      );
      initial[section.slug] = isCurrentSection;
    });
    // Default to opening Getting Started if no section is active
    if (!Object.values(initial).some(Boolean)) {
      initial['getting-started'] = true;
    }
    return initial;
  });

  const toggleSection = (slug: string) => {
    setOpenSections((prev) => ({ ...prev, [slug]: !prev[slug] }));
  };

  return (
    <nav className={className}>
      <div className="mb-6">
        <Link
          href="/docs"
          className="flex items-center gap-2 text-white font-semibold hover:text-accent transition-colors"
          onClick={onLinkClick}
        >
          <BookOpen className="w-5 h-5" />
          Documentation
        </Link>
      </div>

      {docsStructure.map((section) => (
        <SidebarSection
          key={section.slug}
          section={section}
          isOpen={openSections[section.slug] || false}
          onToggle={() => toggleSection(section.slug)}
          currentPath={pathname}
        />
      ))}

      <div className="mt-8 pt-6 border-t border-white/10">
        <p className="px-3 text-xs text-text-muted mb-2">Resources</p>
        <ul className="space-y-1">
          <li>
            <Link
              href="/download"
              className="block px-3 py-1.5 text-sm text-text-muted hover:text-white transition-colors"
              onClick={onLinkClick}
            >
              Download App
            </Link>
          </li>
        </ul>
      </div>
    </nav>
  );
}

export default function DocsLayout({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen pt-16">
      {/* Mobile sidebar toggle */}
      <div className="lg:hidden fixed top-20 left-4 z-40">
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="p-2 rounded-lg bg-bg-surface border border-white/10 text-white"
        >
          {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {/* Mobile sidebar overlay */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="lg:hidden fixed inset-0 z-30 bg-black/50"
            onClick={() => setSidebarOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Mobile: fixed drawer sidebar */}
      <aside
        className={`lg:hidden fixed top-16 left-0 z-40 h-[calc(100vh-4rem)] w-72 bg-bg-dark border-r border-white/10 overflow-y-auto transform transition-transform ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="p-6">
          <DocsSidebar onLinkClick={() => setSidebarOpen(false)} />
        </div>
      </aside>

      {/* Desktop: flex layout so sidebar stays in flow and does not overlap footer */}
      <div className="lg:flex">
        {/* Desktop sidebar - sticky so it stays in view while scrolling but scrolls away before footer */}
        <aside className="hidden lg:block w-72 flex-shrink-0 border-r border-white/10 bg-bg-dark">
          <div className="sticky top-20 max-h-[calc(100vh-5rem)] overflow-y-auto p-6">
            <DocsSidebar />
          </div>
        </aside>

        {/* Main content */}
        <main className="min-w-0 flex-1">
          <div className="max-w-4xl mx-auto px-6 py-12">{children}</div>
        </main>
      </div>
    </div>
  );
}
