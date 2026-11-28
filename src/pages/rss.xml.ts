import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import type { APIContext } from 'astro';

export async function GET(context: APIContext) {
  const posts = await getCollection('posts');
  const ideas = await getCollection('ideas');
  const inventions = await getCollection('inventions');

  // 订阅源以 /blog/ 为根：item 链接用相对路径，channel <link> 才是站点首页
  const allItems = [
    ...posts.map(post => ({
      title: `📝 ${post.data.title}`,
      pubDate: post.data.pubDate,
      description: post.data.description,
      link: `posts/${post.id.replace(/\.md$/, '')}/`,
    })),
    ...ideas.map(idea => ({
      title: `💡 ${idea.data.title}`,
      pubDate: idea.data.pubDate,
      description: idea.data.description || '',
      link: `ideas/${idea.id.replace(/\.md$/, '')}/`,
    })),
    ...inventions.map(inv => ({
      title: `🔧 ${inv.data.title}`,
      pubDate: inv.data.pubDate,
      description: inv.data.description,
      link: `inventions/${inv.id.replace(/\.md$/, '')}/`,
    })),
  ].sort((a, b) => b.pubDate.valueOf() - a.pubDate.valueOf());

  const lastBuildDate = new Date().toUTCString();

  return rss({
    title: '赛博阿漆 · AI 核心概念与推理优化笔记',
    description: '用可运行代码讲清 AI 核心机制：Transformer、KV Cache、量化、MoE、RAG 与评测，每篇附可复现实验与真实输出。',
    site: new URL('/blog/', context.site!),
    items: allItems,
    xmlns: { atom: 'http://www.w3.org/2005/Atom' },
    customData:
      `<language>zh-CN</language>` +
      `<lastBuildDate>${lastBuildDate}</lastBuildDate>` +
      `<atom:link href="https://shenlv7.github.io/blog/rss.xml" rel="self" type="application/rss+xml"/>`,
  });
}
