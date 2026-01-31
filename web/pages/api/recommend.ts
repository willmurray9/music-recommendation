import type { NextApiRequest, NextApiResponse } from 'next';
import {
  loadTracks,
  loadEmbeddings,
  loadEmbeddingsMeta,
  buildTrackIndex,
  Track,
} from '@/lib/embeddings';
import { recommend, computePercentiles, RecommendationResult } from '@/lib/recommender';

interface RecommendRequest {
  seedTracks: string[]; // Track IDs
  n?: number;
  popularity?: number; // 0-1
  artistDiversity?: number; // 0-1
  exploration?: number; // 0-1
  genres?: string[];
  excludeGenres?: string[];
  excludeArtists?: string[];
}

interface RecommendResponse {
  recommendations: RecommendationResult[];
  seedTracks: { track: Track; index: number }[];
}

interface ErrorResponse {
  error: string;
}

// Cache for percentiles (computed once)
let playlistPercentilesCache: number[] | null = null;

/**
 * Get recommendations for seed tracks.
 * POST /api/recommend
 */
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<RecommendResponse | ErrorResponse>
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const body: RecommendRequest = req.body;
  
  if (!body.seedTracks || !Array.isArray(body.seedTracks) || body.seedTracks.length === 0) {
    return res.status(400).json({ error: 'seedTracks array is required' });
  }

  try {
    const [tracks, embeddings, meta] = await Promise.all([
      loadTracks(),
      loadEmbeddings(),
      loadEmbeddingsMeta(),
    ]);

    // Build track index if needed
    const trackIndex = buildTrackIndex(tracks);

    // Compute percentiles if needed
    if (!playlistPercentilesCache) {
      playlistPercentilesCache = computePercentiles(
        tracks.map(t => t.playlistCount)
      );
    }

    // Convert seed track IDs to indices
    const seedIndices: number[] = [];
    const validSeedTracks: { track: Track; index: number }[] = [];
    
    for (const id of body.seedTracks) {
      const idx = trackIndex.get(id);
      if (idx !== undefined) {
        seedIndices.push(idx);
        validSeedTracks.push({ track: tracks[idx], index: idx });
      }
    }

    if (seedIndices.length === 0) {
      return res.status(400).json({ error: 'No valid seed tracks found' });
    }

    // Get recommendations
    const recommendations = recommend(
      embeddings,
      tracks,
      meta.dimensions,
      playlistPercentilesCache,
      {
        seedIndices,
        n: body.n || 10,
        popularity: body.popularity ?? 0.5,
        artistDiversity: body.artistDiversity ?? 0.5,
        exploration: body.exploration ?? 0,
        genres: body.genres,
        excludeGenres: body.excludeGenres,
        excludeArtists: body.excludeArtists,
      }
    );

    res.status(200).json({
      recommendations,
      seedTracks: validSeedTracks,
    });
  } catch (error) {
    console.error('Recommendation error:', error);
    res.status(500).json({ error: 'Failed to generate recommendations' });
  }
}
