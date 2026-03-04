import Link from 'next/link';
import { Linkedin, Mail } from 'lucide-react';
import { Logo } from './logo';

const footerLinks = {
  product: [
    { label: 'Features', href: '/features' },
    { label: 'Download', href: '/download' },
    { label: 'Documentation', href: '/docs' },
    { label: 'Release Notes', href: '/download#changelog' },
  ],
  resources: [
    { label: 'Getting Started', href: '/docs/getting-started/installation' },
    { label: 'Pretreatment Guide', href: '/docs/pretreatment/overview' },
    { label: 'Milling Guide', href: '/docs/milling/overview' },
    { label: 'Reference', href: '/docs/reference/shortcuts' },
  ],
  company: [
    { label: 'About', href: '/about' },
    { label: 'Contact', href: '/contact' },
    { label: 'McGill University', href: '/about#mcgill' },
    { label: 'NRC Canada', href: '/about#nrc' },
  ],
};

const socialLinks = [
  {
    label: 'LinkedIn',
    href: 'https://linkedin.com/in/eakwofie',
    icon: Linkedin,
  },
  {
    label: 'Email',
    href: 'mailto:emmanuel.kwofie@mcgill.ca',
    icon: Mail,
  },
];

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-bg-surface border-t border-white/10">
      <div className="container-custom">
        {/* Main Footer Content */}
        <div className="py-12 md:py-16 grid grid-cols-2 md:grid-cols-5 gap-8 md:gap-12">
          {/* Brand Column */}
          <div className="col-span-2">
            <Logo size="lg" />
            <p className="mt-4 text-text-secondary text-sm max-w-xs">
              Complete protein processing simulation software. From raw seed to
              fractionated flour.
            </p>
            <div className="mt-6 flex items-center gap-4">
              {socialLinks.map((link) => (
                <a
                  key={link.label}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-text-secondary hover:text-white transition-colors"
                  aria-label={link.label}
                >
                  <link.icon className="w-5 h-5" />
                </a>
              ))}
            </div>
            {/* Institution Badges */}
            <div className="mt-6 pt-6 border-t border-white/10">
              <p className="text-xs text-text-muted mb-3">Developed at</p>
              <div className="flex items-center gap-2 mb-3">
                <div className="w-8 h-8 rounded bg-white/10 flex items-center justify-center">
                  <span className="text-xs font-bold text-red-400">M</span>
                </div>
                <span className="text-sm text-text-secondary">
                  McGill University
                </span>
              </div>
              <p className="text-xs text-text-muted mb-2">In collaboration with</p>
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded bg-white/10 flex items-center justify-center">
                  <span className="text-xs font-bold text-accent">NRC</span>
                </div>
                <span className="text-sm text-text-secondary">
                  National Research Council Canada
                </span>
              </div>
            </div>
          </div>

          {/* Product Links */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-4">Product</h3>
            <ul className="space-y-3">
              {footerLinks.product.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-sm text-text-secondary hover:text-white transition-colors"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Resources Links */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-4">Resources</h3>
            <ul className="space-y-3">
              {footerLinks.resources.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-sm text-text-secondary hover:text-white transition-colors"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Company Links */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-4">Company</h3>
            <ul className="space-y-3">
              {footerLinks.company.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-sm text-text-secondary hover:text-white transition-colors"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="py-6 border-t border-white/10 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-sm text-text-muted">
            &copy; {currentYear} ProteinProcessIO. Developed at McGill University by{' '}
            <a
              href="https://www.eakwofie.com/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent hover:text-accent-glow transition-colors"
            >
              Emmanuel Amankrah Kwofie
            </a>
          </p>
          <div className="flex items-center gap-6 text-sm text-text-muted">
            <Link
              href="/privacy"
              className="hover:text-white transition-colors"
            >
              Privacy Policy
            </Link>
            <Link
              href="/terms"
              className="hover:text-white transition-colors"
            >
              Terms of Service
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
