import { notFound, redirect } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft, ArrowRight, Clock } from 'lucide-react';
import { serialize } from 'next-mdx-remote/serialize';
import remarkGfm from 'remark-gfm';
import rehypeSlug from 'rehype-slug';
import rehypeHighlight from 'rehype-highlight';
import { getDocBySlug, getAllDocSlugs, getDocNavigation, docsStructure } from '@/lib/docs';
import { MDXContent } from '@/components/docs/mdx-content';

// Avoid prerender errors (undefined component / MDX) until resolved; docs render on first request
export const dynamic = 'force-dynamic';

interface PageProps {
  params: Promise<{
    slug: string[];
  }>;
}

export async function generateStaticParams() {
  const slugs = getAllDocSlugs();

  // Generate paths for both section-only and section/slug routes
  const params: { slug: string[] }[] = [];

  // Add section-level paths (e.g., /docs/getting-started)
  docsStructure.forEach((section) => {
    params.push({ slug: [section.slug] });
  });

  // Add full doc paths (e.g., /docs/getting-started/installation)
  slugs.forEach(({ section, slug }) => {
    params.push({ slug: [section, slug] });
  });

  return params;
}

export async function generateMetadata({ params }: PageProps) {
  const resolvedParams = await params;
  const slugParts = resolvedParams.slug;

  // Handle section-only URLs
  if (slugParts.length === 1) {
    const sectionData = docsStructure.find((s) => s.slug === slugParts[0]);
    if (sectionData) {
      return {
        title: `${sectionData.title} - ProteinProcessIO Docs`,
        description: `${sectionData.title} documentation`,
      };
    }
    return { title: 'Not Found' };
  }

  const [section, slug] = slugParts;
  const doc = getDocBySlug(section, slug);

  if (!doc) {
    return { title: 'Not Found' };
  }

  return {
    title: `${doc.meta.title} - ProteinProcessIO Docs`,
    description: doc.meta.description,
  };
}

export default async function DocPage({ params }: PageProps) {
  const resolvedParams = await params;
  const slugParts = resolvedParams.slug;

  // Handle section-only URLs - redirect to first item in section
  if (slugParts.length === 1) {
    const sectionSlug = slugParts[0];
    const sectionData = docsStructure.find((s) => s.slug === sectionSlug);

    if (sectionData && sectionData.items.length > 0) {
      redirect(`/docs/${sectionSlug}/${sectionData.items[0].slug}`);
    }
    notFound();
  }

  const [section, slug] = slugParts;
  const doc = getDocBySlug(section, slug);

  if (!doc) {
    notFound();
  }

  const mdxSource = await serialize(doc.content, {
    mdxOptions: {
      remarkPlugins: [remarkGfm],
      rehypePlugins: [rehypeSlug, rehypeHighlight],
    },
  });

  const { prev, next } = getDocNavigation(section, slug);
  const sectionData = docsStructure.find((s) => s.slug === section);

  return (
    <article className="prose prose-invert prose-lg max-w-none">
      {/* Breadcrumb */}
      <nav className="flex items-center gap-2 text-sm text-text-muted mb-8 not-prose">
        <Link href="/docs" className="hover:text-white transition-colors">
          Docs
        </Link>
        <span>/</span>
        <Link
          href={`/docs/${section}/${sectionData?.items[0]?.slug || slug}`}
          className="hover:text-white transition-colors"
        >
          {sectionData?.title}
        </Link>
        <span>/</span>
        <span className="text-text-secondary">{doc.meta.title}</span>
      </nav>

      {/* Header */}
      <header className="mb-8 not-prose">
        <h1 className="text-3xl md:text-4xl font-bold text-white mb-3">
          {doc.meta.title}
        </h1>
        {doc.meta.description && (
          <p className="text-lg text-text-secondary">{doc.meta.description}</p>
        )}
        <div className="flex items-center gap-4 mt-4 text-sm text-text-muted">
          <span className="flex items-center gap-1">
            <Clock className="w-4 h-4" />
            5 min read
          </span>
        </div>
      </header>

      {/* Content */}
      <div className="docs-content">
        <MDXContent source={mdxSource} />
      </div>

      {/* Navigation */}
      <nav className="flex items-center justify-between mt-12 pt-8 border-t border-white/10 not-prose">
        {prev ? (
          <Link
            href={`/docs/${prev.section}/${prev.slug}`}
            className="group flex items-center gap-2 text-text-secondary hover:text-white transition-colors"
          >
            <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
            <div className="text-right">
              <div className="text-xs text-text-muted">Previous</div>
              <div className="font-medium">{prev.title}</div>
            </div>
          </Link>
        ) : (
          <div />
        )}

        {next ? (
          <Link
            href={`/docs/${next.section}/${next.slug}`}
            className="group flex items-center gap-2 text-text-secondary hover:text-white transition-colors text-right"
          >
            <div>
              <div className="text-xs text-text-muted">Next</div>
              <div className="font-medium">{next.title}</div>
            </div>
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </Link>
        ) : (
          <div />
        )}
      </nav>
    </article>
  );
}
