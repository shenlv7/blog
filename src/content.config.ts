import { defineCollection, z } from 'astro:content';

const posts = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    tags: z.array(z.string()).optional(),
    cover: z.string().optional(),
    difficulty: z.enum(['beginner', 'intermediate', 'advanced']).optional(),
    series: z.string().optional(),
    seriesOrder: z.number().optional(),
    episode: z.number().optional(),
  }),
});

const ideas = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    pubDate: z.coerce.date(),
    tags: z.array(z.string()).optional(),
    ideaNumber: z.number(),
    status: z.enum(['idea', 'building', 'done']).optional(),
  }),
});

const inventions = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    tags: z.array(z.string()).optional(),
    cover: z.string().optional(),
    ideaId: z.string().optional(),
    inventionNumber: z.number(),
    status: z.enum(['prototype', 'beta', 'released']).optional(),
  }),
});

const shell = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    tags: z.array(z.string()).optional(),
    cover: z.string().optional(),
    shellNumber: z.number(),
    phase: z.string(),
    status: z.enum(['planning', 'building', 'done']).optional(),
    budget: z.string().optional(),
  }),
});

export const collections = { posts, ideas, inventions, shell };
