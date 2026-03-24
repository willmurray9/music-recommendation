/**
 * Catalog and embedding loaders for both the full server-side catalog and the
 * public visualization subset.
 */

import { promises as fs } from 'fs';
import path from 'path';

export interface Track {
  id: string;
  name: string;
  artist: string;
  album: string;
  artistId?: string;
  genres: string[];
  tags?: string[];
  popularity: number;
  playlistCount: number;
  support?: number;
  durationMs?: number;
  releaseYear?: number | null;
  modelVersion?: string;
}

export interface TracksData {
  tracks: Track[];
}

export interface EmbeddingsMeta {
  numTracks: number;
  dimensions: number;
}

export interface GenreData {
  genres: { name: string; count: number }[];
}

export interface RerankerModel {
  feature_names: string[];
  baseline: number;
  trees: {
    nodes: {
      value: number;
      feature_idx: number;
      num_threshold: number;
      left: number;
      right: number;
      is_leaf: boolean;
    }[];
  }[];
}

const PUBLIC_DATA_DIR = path.join(process.cwd(), 'public', 'data');
const SERVER_DATA_DIR = path.join(process.cwd(), 'data', 'server', 'current');

let fullTracksCache: Track[] | null = null;
let fullEmbeddingsCache: Float32Array | null = null;
let fullEmbeddingsMetaCache: EmbeddingsMeta | null = null;
let fullSearchIndexCache: Record<string, number[]> | null = null;
let fullGenresCache: { name: string; count: number }[] | null = null;

let vizTracksCache: Track[] | null = null;
let vizEmbeddingsCache: Float32Array | null = null;
let vizEmbeddingsMetaCache: EmbeddingsMeta | null = null;
let vizIndexCache: Map<string, number> | null = null;

let rerankerModelCache: RerankerModel | null | undefined = undefined;

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function resolveServerOrPublic(filename: string): Promise<string> {
  const serverPath = path.join(SERVER_DATA_DIR, filename);
  if (await fileExists(serverPath)) {
    return serverPath;
  }
  return path.join(PUBLIC_DATA_DIR, filename);
}

async function resolvePublic(filename: string): Promise<string> {
  return path.join(PUBLIC_DATA_DIR, filename);
}

async function loadTracksFrom(filePath: string): Promise<Track[]> {
  const data = await fs.readFile(filePath, 'utf-8');
  const parsed: TracksData = JSON.parse(data);
  return parsed.tracks;
}

async function loadEmbeddingsMetaFrom(filePath: string): Promise<EmbeddingsMeta> {
  const data = await fs.readFile(filePath, 'utf-8');
  return JSON.parse(data);
}

async function loadEmbeddingsFrom(
  filePath: string,
  meta: EmbeddingsMeta,
): Promise<Float32Array> {
  const buffer = await fs.readFile(filePath);
  return new Float32Array(
    buffer.buffer,
    buffer.byteOffset,
    meta.numTracks * meta.dimensions,
  );
}

export async function loadTracks(): Promise<Track[]> {
  if (fullTracksCache) return fullTracksCache;
  fullTracksCache = await loadTracksFrom(await resolveServerOrPublic('tracks.json'));
  return fullTracksCache;
}

export async function loadVisualizationTracks(): Promise<Track[]> {
  if (vizTracksCache) return vizTracksCache;
  vizTracksCache = await loadTracksFrom(await resolvePublic('tracks.json'));
  return vizTracksCache;
}

export async function loadEmbeddingsMeta(): Promise<EmbeddingsMeta> {
  if (fullEmbeddingsMetaCache) return fullEmbeddingsMetaCache;
  fullEmbeddingsMetaCache = await loadEmbeddingsMetaFrom(
    await resolveServerOrPublic('embeddings_meta.json'),
  );
  return fullEmbeddingsMetaCache;
}

export async function loadVisualizationEmbeddingsMeta(): Promise<EmbeddingsMeta> {
  if (vizEmbeddingsMetaCache) return vizEmbeddingsMetaCache;
  vizEmbeddingsMetaCache = await loadEmbeddingsMetaFrom(
    await resolvePublic('embeddings_meta.json'),
  );
  return vizEmbeddingsMetaCache;
}

export async function loadEmbeddings(): Promise<Float32Array> {
  if (fullEmbeddingsCache) return fullEmbeddingsCache;
  const meta = await loadEmbeddingsMeta();
  fullEmbeddingsCache = await loadEmbeddingsFrom(
    await resolveServerOrPublic('embeddings.bin'),
    meta,
  );
  return fullEmbeddingsCache;
}

export async function loadVisualizationEmbeddings(): Promise<Float32Array> {
  if (vizEmbeddingsCache) return vizEmbeddingsCache;
  const meta = await loadVisualizationEmbeddingsMeta();
  vizEmbeddingsCache = await loadEmbeddingsFrom(
    await resolvePublic('embeddings.bin'),
    meta,
  );
  return vizEmbeddingsCache;
}

export async function loadSearchIndex(): Promise<Record<string, number[]>> {
  if (fullSearchIndexCache) return fullSearchIndexCache;
  const filePath = await resolveServerOrPublic('search_index.json');
  const data = await fs.readFile(filePath, 'utf-8');
  fullSearchIndexCache = JSON.parse(data);
  return fullSearchIndexCache || {};
}

export async function loadGenres(): Promise<{ name: string; count: number }[]> {
  if (fullGenresCache) return fullGenresCache;
  const filePath = await resolveServerOrPublic('genres.json');
  const data = await fs.readFile(filePath, 'utf-8');
  const parsed: GenreData = JSON.parse(data);
  fullGenresCache = parsed.genres;
  return fullGenresCache;
}

export async function loadVisualizationIndexMap(): Promise<Map<string, number>> {
  if (vizIndexCache) return vizIndexCache;

  const explicitIndexPath = path.join(PUBLIC_DATA_DIR, 'viz_index.json');
  if (await fileExists(explicitIndexPath)) {
    const data = await fs.readFile(explicitIndexPath, 'utf-8');
    const parsed: Record<string, number> = JSON.parse(data);
    vizIndexCache = new Map(Object.entries(parsed).map(([id, index]) => [id, Number(index)]));
    return vizIndexCache;
  }

  const tracks = await loadVisualizationTracks();
  vizIndexCache = buildTrackIndex(tracks);
  return vizIndexCache;
}

export async function loadRerankerModel(): Promise<RerankerModel | null> {
  if (rerankerModelCache !== undefined) return rerankerModelCache ?? null;
  const filePath = path.join(SERVER_DATA_DIR, 'reranker_model.json');
  if (!(await fileExists(filePath))) {
    rerankerModelCache = null;
    return rerankerModelCache;
  }
  const data = await fs.readFile(filePath, 'utf-8');
  rerankerModelCache = JSON.parse(data);
  return rerankerModelCache ?? null;
}

export function getEmbedding(
  embeddings: Float32Array,
  index: number,
  dimensions: number,
): Float32Array {
  const start = index * dimensions;
  return embeddings.subarray(start, start + dimensions);
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function cosineSimilarityAtOffset(
  vector: Float32Array,
  embeddings: Float32Array,
  start: number,
  dimensions: number,
): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < dimensions; i++) {
    const value = embeddings[start + i];
    const vec = vector[i];
    dotProduct += vec * value;
    normA += vec * vec;
    normB += value * value;
  }

  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

export function computeCentroid(
  embeddings: Float32Array,
  indices: number[],
  dimensions: number,
): Float32Array {
  const centroid = new Float32Array(dimensions);

  for (const idx of indices) {
    const start = idx * dimensions;
    for (let i = 0; i < dimensions; i++) {
      centroid[i] += embeddings[start + i];
    }
  }

  for (let i = 0; i < dimensions; i++) {
    centroid[i] /= indices.length;
  }

  return centroid;
}

export function findSimilar(
  centroid: Float32Array,
  embeddings: Float32Array,
  numTracks: number,
  dimensions: number,
  topN: number = 500,
  excludeIndices: Set<number> = new Set(),
): { index: number; score: number }[] {
  const similarities: { index: number; score: number }[] = [];

  for (let i = 0; i < numTracks; i++) {
    if (excludeIndices.has(i)) continue;
    const start = i * dimensions;
    const score = cosineSimilarityAtOffset(centroid, embeddings, start, dimensions);
    similarities.push({ index: i, score });
  }

  similarities.sort((a, b) => b.score - a.score);
  return similarities.slice(0, topN);
}

export function buildTrackIndex(tracks: Track[]): Map<string, number> {
  const map = new Map<string, number>();
  tracks.forEach((track, index) => {
    map.set(track.id, index);
  });
  return map;
}
