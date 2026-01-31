/**
 * Embeddings library for loading and querying track embeddings.
 * Handles binary Float32Array loading and cosine similarity computations.
 */

import { promises as fs } from 'fs';
import path from 'path';

// Types
export interface Track {
  id: string;
  name: string;
  artist: string;
  album: string;
  genres: string[];
  popularity: number;
  playlistCount: number;
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

// Cache for loaded data
let tracksCache: Track[] | null = null;
let embeddingsCache: Float32Array | null = null;
let embeddingsMetaCache: EmbeddingsMeta | null = null;
let searchIndexCache: Record<string, number[]> | null = null;
let genresCache: { name: string; count: number }[] | null = null;

const DATA_DIR = path.join(process.cwd(), 'public', 'data');

/**
 * Load tracks metadata from JSON file.
 */
export async function loadTracks(): Promise<Track[]> {
  if (tracksCache) return tracksCache;
  
  const filePath = path.join(DATA_DIR, 'tracks.json');
  const data = await fs.readFile(filePath, 'utf-8');
  const parsed: TracksData = JSON.parse(data);
  tracksCache = parsed.tracks;
  return tracksCache;
}

/**
 * Load embeddings metadata.
 */
export async function loadEmbeddingsMeta(): Promise<EmbeddingsMeta> {
  if (embeddingsMetaCache) return embeddingsMetaCache;
  
  const filePath = path.join(DATA_DIR, 'embeddings_meta.json');
  const data = await fs.readFile(filePath, 'utf-8');
  embeddingsMetaCache = JSON.parse(data);
  return embeddingsMetaCache!;
}

/**
 * Load embeddings binary file into Float32Array.
 */
export async function loadEmbeddings(): Promise<Float32Array> {
  if (embeddingsCache) return embeddingsCache;
  
  const meta = await loadEmbeddingsMeta();
  const filePath = path.join(DATA_DIR, 'embeddings.bin');
  const buffer = await fs.readFile(filePath);
  
  // Convert Node.js Buffer to Float32Array
  embeddingsCache = new Float32Array(
    buffer.buffer,
    buffer.byteOffset,
    meta.numTracks * meta.dimensions
  );
  
  return embeddingsCache;
}

/**
 * Load search index.
 */
export async function loadSearchIndex(): Promise<Record<string, number[]>> {
  if (searchIndexCache) return searchIndexCache;
  
  const filePath = path.join(DATA_DIR, 'search_index.json');
  const data = await fs.readFile(filePath, 'utf-8');
  searchIndexCache = JSON.parse(data);
  return searchIndexCache!;
}

/**
 * Load genres list.
 */
export async function loadGenres(): Promise<{ name: string; count: number }[]> {
  if (genresCache) return genresCache;
  
  const filePath = path.join(DATA_DIR, 'genres.json');
  const data = await fs.readFile(filePath, 'utf-8');
  const parsed: GenreData = JSON.parse(data);
  genresCache = parsed.genres;
  return genresCache;
}

/**
 * Get embedding vector for a track by index.
 */
export function getEmbedding(
  embeddings: Float32Array,
  index: number,
  dimensions: number
): Float32Array {
  const start = index * dimensions;
  return embeddings.slice(start, start + dimensions);
}

/**
 * Compute cosine similarity between two vectors.
 */
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

/**
 * Compute centroid (mean) of multiple embedding vectors.
 */
export function computeCentroid(
  embeddings: Float32Array,
  indices: number[],
  dimensions: number
): Float32Array {
  const centroid = new Float32Array(dimensions);
  
  for (const idx of indices) {
    const vec = getEmbedding(embeddings, idx, dimensions);
    for (let i = 0; i < dimensions; i++) {
      centroid[i] += vec[i];
    }
  }
  
  // Divide by count to get mean
  for (let i = 0; i < dimensions; i++) {
    centroid[i] /= indices.length;
  }
  
  return centroid;
}

/**
 * Find top N most similar tracks to a centroid vector.
 */
export function findSimilar(
  centroid: Float32Array,
  embeddings: Float32Array,
  numTracks: number,
  dimensions: number,
  topN: number = 500,
  excludeIndices: Set<number> = new Set()
): { index: number; score: number }[] {
  const similarities: { index: number; score: number }[] = [];
  
  for (let i = 0; i < numTracks; i++) {
    if (excludeIndices.has(i)) continue;
    
    const vec = getEmbedding(embeddings, i, dimensions);
    const score = cosineSimilarity(centroid, vec);
    similarities.push({ index: i, score });
  }
  
  // Sort by score descending and take top N
  similarities.sort((a, b) => b.score - a.score);
  return similarities.slice(0, topN);
}

/**
 * Build track URI to index lookup map.
 */
export function buildTrackIndex(tracks: Track[]): Map<string, number> {
  const map = new Map<string, number>();
  tracks.forEach((track, index) => {
    map.set(track.id, index);
  });
  return map;
}
