import { getCollection } from 'astro:content';
import type { APIContext } from 'astro';

export async function GET(context: APIContext) {
  const posts = await getCollection('posts');
  const ideas = await getCollection('ideas');
  const inventions = await getCollection('inventions');

  const items = [
    ...posts.map(post => ({
      title: post.data.title,
      description: post.data.description,
      url: `/blog/posts/${post.id.replace(/\.md$/, '')}/`,
      type: 'post' as const,
      tags: post.data.tags || [],
      date: post.data.pubDate.toLocaleDateString('zh-CN'),
    })),
    ...ideas.map(idea => ({
      title: idea.data.title,
      description: idea.data.description || '',
      url: `/blog/ideas/${idea.id.replace(/\.md$/, '')}/`,
      type: 'idea' as const,
      tags: idea.data.tags || [],
      date: idea.data.pubDate.toLocaleDateString('zh-CN'),
    })),
    ...inventions.map(inv => ({
      title: inv.data.title,
      description: inv.data.description,
      url: `/blog/inventions/${inv.id.replace(/\.md$/, '')}/`,
      type: 'invention' as const,
      tags: inv.data.tags || [],
      date: inv.data.pubDate.toLocaleDateString('zh-CN'),
    })),
  ];

  return new Response(JSON.stringify(items), {
    headers: { 'Content-Type': 'application/json' },
  });
}
