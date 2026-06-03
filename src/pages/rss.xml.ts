import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import type { APIContext } from 'astro';

export async function GET(context: APIContext) {
  const posts = await getCollection('posts');
  const ideas = await getCollection('ideas');
  const inventions = await getCollection('inventions');

  const allItems = [
    ...posts.map(post => ({
      title: `📝 ${post.data.title}`,
      pubDate: post.data.pubDate,
      description: post.data.description,
      link: `/blog/posts/${post.id.replace(/\.md$/, '')}/`,
    })),
    ...ideas.map(idea => ({
      title: `💡 ${idea.data.title}`,
      pubDate: idea.data.pubDate,
      description: idea.data.description || '',
      link: `/blog/ideas/${idea.id.replace(/\.md$/, '')}/`,
    })),
    ...inventions.map(inv => ({
      title: `🔧 ${inv.data.title}`,
      pubDate: inv.data.pubDate,
      description: inv.data.description,
      link: `/blog/inventions/${inv.id.replace(/\.md$/, '')}/`,
    })),
  ].sort((a, b) => b.pubDate.valueOf() - a.pubDate.valueOf());

  return rss({
    title: '赛博阿漆的AI学习博客',
    description: '从零开始，系统学习人工智能与机器学习',
    site: context.site!,
    items: allItems,
    customData: '<language>zh-CN</language>',
  });
}
