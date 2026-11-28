import type { APIContext } from 'astro';
import { getCollection } from 'astro:content';
import { CONCEPT_GROUPS, GROUP_PREREQ, CONCEPT_ROUTES } from '../data/concept-groups';
import { CONCEPT_LABELS } from '../data/concept-labels';

export async function GET(context: APIContext) {
  const posts = await getCollection('posts');
  const numbered = posts
    .filter(p => p.data.series === 'ai-concepts' && p.data.seriesOrder)
    .sort((a, b) => (a.data.seriesOrder as number) - (b.data.seriesOrder as number));

  const byNum = new Map(numbered.map(p => [p.data.seriesOrder as number, p]));

  const nodes = numbered.map(p => {
    const body = p.body || '';
    const n = p.data.seriesOrder as number;
    const zh = (body.match(/[\u4e00-\u9fff]/g) || []).length;
    const demo = body.match(/https:\/\/github\.com\/shenlv7\/blog\/blob\/main\/demos\/[^\s)]+/);
    return {
      id: p.id.replace(/\.md$/, ''),
      n,
      label: CONCEPT_LABELS[n] || p.data.title.slice(0, 12),
      title: p.data.title.replace(/^\d+\s*\|\s*/, ''),
      desc: p.data.description,
      group: CONCEPT_GROUPS.find(g => g.nums.includes(n))?.key || 'X',
      difficulty: p.data.difficulty || 'beginner',
      minutes: Math.max(1, Math.round(zh / 450)),
      tags: p.data.tags || [],
      code: /```(python|py)/.test(body),
      demo: demo ? demo[0] : null,
      url: `/blog/posts/${p.id.replace(/\.md$/, '')}/`,
    };
  });

  const idByNum = new Map(nodes.map(n => [n.n, n.id]));

  // 顺序边：组内按学习路径的阅读顺序串起来
  const links: { s: string; t: string; kind: 'order' | 'semantic' }[] = [];
  for (const g of CONCEPT_GROUPS) {
    for (let i = 0; i < g.nums.length - 1; i++) {
      const s = idByNum.get(g.nums[i]);
      const t = idByNum.get(g.nums[i + 1]);
      if (s && t) links.push({ s, t, kind: 'order' });
    }
  }

  // 语义边：共享 ≥2 个标签，每个节点最多留 2 条，避免变成毛线团
  const tagMap = new Map<string, Set<string>>();
  nodes.forEach(n => n.tags.forEach(t => {
    if (!tagMap.has(t)) tagMap.set(t, new Set());
    tagMap.get(t)!.add(n.id);
  }));
  const pairs = new Map<string, { a: string; b: string; w: number }>();
  nodes.forEach(a => {
    const score = new Map<string, number>();
    a.tags.forEach(t => tagMap.get(t)!.forEach(bid => {
      if (bid === a.id) return;
      score.set(bid, (score.get(bid) || 0) + 1);
    }));
    [...score.entries()]
      .filter(([, w]) => w >= 2)
      .sort((x, y) => y[1] - x[1])
      .slice(0, 2)
      .forEach(([bid, w]) => {
        const key = [a.id, bid].sort().join('|');
        if (!pairs.has(key)) pairs.set(key, { a: a.id, b: bid, w });
      });
  });
  pairs.forEach(p => links.push({ s: p.a, t: p.b, kind: 'semantic' }));

  const groups = CONCEPT_GROUPS.map(g => ({
    key: g.key,
    name: g.name,
    desc: g.desc,
    color: g.color,
    prereq: GROUP_PREREQ[g.key] || [],
    count: g.nums.filter(n => byNum.has(n)).length,
  }));

  const routes = CONCEPT_ROUTES.map(r => ({
    ...r,
    count: r.groups.reduce((s, k) => s + (groups.find(g => g.key === k)?.count || 0), 0),
  }));

  return new Response(JSON.stringify({ nodes, links, groups, routes }), {
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
}
