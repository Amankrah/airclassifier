'use client';

import { MDXRemote, MDXRemoteSerializeResult } from 'next-mdx-remote';
import { serialize } from 'next-mdx-remote/serialize';
import { useEffect, useState } from 'react';
import remarkGfm from 'remark-gfm';
import rehypeSlug from 'rehype-slug';
import rehypeHighlight from 'rehype-highlight';
import { AlertCircle, Info, AlertTriangle, CheckCircle, Copy, Check } from 'lucide-react';

// Default HTML elements so MDX has a component for every tag (next-mdx-remote doesn't merge with defaults)
const htmlTags = [
  'p', 'div', 'span', 'em', 'strong', 'code', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
  'blockquote', 'pre', 'hr', 'br', 'ul', 'ol', 'li', 'img', 'table', 'thead', 'tbody', 'tr', 'td', 'th',
  'section', 'article', 'nav', 'header', 'footer', 'main', 'aside', 'del',
];
const defaultComponents = Object.fromEntries(htmlTags.map((tag) => [tag, tag])) as Record<string, string>;

// Custom components for MDX (spread after defaults so our overrides take precedence)
const components = {
  ...defaultComponents,
  // Callout component
  Callout: ({
    type = 'info',
    children,
  }: {
    type?: 'info' | 'warning' | 'error' | 'success';
    children: React.ReactNode;
  }) => {
    const styles = {
      info: {
        bg: 'bg-primary/10 border-primary/30',
        icon: <Info className="w-5 h-5 text-primary" />,
      },
      warning: {
        bg: 'bg-warning/10 border-warning/30',
        icon: <AlertTriangle className="w-5 h-5 text-warning" />,
      },
      error: {
        bg: 'bg-error/10 border-error/30',
        icon: <AlertCircle className="w-5 h-5 text-error" />,
      },
      success: {
        bg: 'bg-success/10 border-success/30',
        icon: <CheckCircle className="w-5 h-5 text-success" />,
      },
    };

    const style = styles[type];

    return (
      <div className={`flex gap-3 p-4 rounded-lg border ${style.bg} my-6`}>
        <div className="flex-shrink-0 mt-0.5">{style.icon}</div>
        <div className="text-sm text-text-secondary [&>p]:m-0">{children}</div>
      </div>
    );
  },

  // Code block with copy button
  pre: ({ children, ...props }: React.HTMLAttributes<HTMLPreElement>) => {
    const [copied, setCopied] = useState(false);

    const handleCopy = () => {
      const code = (children as React.ReactElement)?.props?.children;
      if (typeof code === 'string') {
        navigator.clipboard.writeText(code);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }
    };

    return (
      <div className="relative group">
        <pre
          {...props}
          className="bg-bg-elevated rounded-lg p-4 overflow-x-auto text-sm border border-white/10"
        >
          {children}
        </pre>
        <button
          onClick={handleCopy}
          className="absolute top-3 right-3 p-2 rounded-md bg-white/5 hover:bg-white/10 transition-colors opacity-0 group-hover:opacity-100"
          title="Copy code"
        >
          {copied ? (
            <Check className="w-4 h-4 text-success" />
          ) : (
            <Copy className="w-4 h-4 text-text-muted" />
          )}
        </button>
      </div>
    );
  },

  // Inline code
  code: ({ children, className, ...props }: React.HTMLAttributes<HTMLElement>) => {
    // If it has a language class, it's inside a pre block
    if (className?.includes('language-')) {
      return (
        <code className={className} {...props}>
          {children}
        </code>
      );
    }
    // Inline code
    return (
      <code
        className="px-1.5 py-0.5 rounded bg-white/10 text-accent font-mono text-sm"
        {...props}
      >
        {children}
      </code>
    );
  },

  // Tables
  table: ({ children }: { children: React.ReactNode }) => (
    <div className="overflow-x-auto my-6">
      <table className="w-full text-sm border-collapse">{children}</table>
    </div>
  ),
  th: ({ children }: { children: React.ReactNode }) => (
    <th className="border border-white/10 bg-white/5 px-4 py-2 text-left font-semibold text-white">
      {children}
    </th>
  ),
  td: ({ children }: { children: React.ReactNode }) => (
    <td className="border border-white/10 px-4 py-2 text-text-secondary">{children}</td>
  ),

  // Links
  a: ({ href, children, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
    <a
      href={href}
      className="text-accent hover:text-accent-glow underline underline-offset-4 transition-colors"
      {...props}
    >
      {children}
    </a>
  ),

  // Headings with anchor links (use full HTML props for MDXComponents compatibility)
  h2: (props: React.HTMLAttributes<HTMLHeadingElement>) => {
    const { children, id, ...rest } = props;
    return (
      <h2 id={id} className="group flex items-center gap-2 scroll-mt-24" {...rest}>
        {children}
        {id && (
          <a
            href={`#${id}`}
            className="opacity-0 group-hover:opacity-100 text-text-muted hover:text-accent transition-all"
          >
            #
          </a>
        )}
      </h2>
    );
  },
  h3: (props: React.HTMLAttributes<HTMLHeadingElement>) => {
    const { children, id, ...rest } = props;
    return (
      <h3 id={id} className="group flex items-center gap-2 scroll-mt-24" {...rest}>
        {children}
        {id && (
          <a
            href={`#${id}`}
            className="opacity-0 group-hover:opacity-100 text-text-muted hover:text-accent transition-all"
          >
            #
          </a>
        )}
      </h3>
    );
  },

  // Steps component
  Steps: ({ children }: { children: React.ReactNode }) => (
    <div className="steps-container my-6 ml-4 border-l-2 border-white/10 pl-6 space-y-6">
      {children}
    </div>
  ),
  Step: ({
    number,
    title,
    children,
  }: {
    number: number;
    title: string;
    children: React.ReactNode;
  }) => (
    <div className="relative">
      <div className="absolute -left-[34px] w-6 h-6 rounded-full bg-primary flex items-center justify-center text-xs font-bold text-white">
        {number}
      </div>
      <h4 className="font-semibold text-white mb-2">{title}</h4>
      <div className="text-text-secondary text-sm">{children}</div>
    </div>
  ),
};

type MDXContentProps =
  | { source: MDXRemoteSerializeResult }
  | { content: string };

export function MDXContent(props: MDXContentProps) {
  const [mdxSource, setMdxSource] = useState<MDXRemoteSerializeResult | null>(
    'source' in props ? props.source : null
  );
  const [error, setError] = useState<string | null>(null);
  const content = 'content' in props ? props.content : null;

  useEffect(() => {
    if (content === null) return;
    const contentToSerialize = content;

    let cancelled = false;
    async function processMDX() {
      try {
        const serialized = await serialize(contentToSerialize, {
          mdxOptions: {
            remarkPlugins: [remarkGfm],
            rehypePlugins: [rehypeSlug, rehypeHighlight as any],
          },
        });
        if (!cancelled) setMdxSource(serialized);
      } catch (err) {
        if (!cancelled) {
          setError('Failed to process documentation content');
          console.error(err);
        }
      }
    }

    processMDX();
    return () => {
      cancelled = true;
    };
  }, [content]);

  if (error) {
    return (
      <div className="p-4 rounded-lg bg-error/10 border border-error/30 text-error">
        {error}
      </div>
    );
  }

  const sourceToRender = 'source' in props ? props.source : mdxSource;

  if (!sourceToRender) {
    return (
      <div className="animate-pulse space-y-4">
        <div className="h-4 bg-white/10 rounded w-3/4" />
        <div className="h-4 bg-white/10 rounded w-1/2" />
        <div className="h-4 bg-white/10 rounded w-5/6" />
      </div>
    );
  }

  return (
    <MDXRemote
      {...sourceToRender}
      components={components as React.ComponentProps<typeof MDXRemote>['components']}
    />
  );
}
