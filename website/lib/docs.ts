// Server-side only functions for reading MDX files
// This file should only be imported in server components

import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { DocMeta, docsStructure, getAllDocSlugs, getDocNavigation } from './docs-structure';

// Re-export from docs-structure for convenience in server components
export { docsStructure, getAllDocSlugs, getDocNavigation };
export type { DocMeta, DocSection } from './docs-structure';

const docsDirectory = path.join(process.cwd(), 'content/docs');

export function getDocBySlug(section: string, slug: string): { content: string; meta: DocMeta } | null {
  try {
    const fullPath = path.join(docsDirectory, section, `${slug}.mdx`);

    if (!fs.existsSync(fullPath)) {
      return null;
    }

    const fileContents = fs.readFileSync(fullPath, 'utf8');
    const { data, content } = matter(fileContents);

    return {
      content,
      meta: {
        title: data.title || slug,
        description: data.description || '',
        slug,
        order: data.order,
      },
    };
  } catch {
    return null;
  }
}
