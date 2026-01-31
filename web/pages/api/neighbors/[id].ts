import type { NextApiRequest, NextApiResponse } from 'next';
import {
  loadTracks,
  loadEmbeddings,
  loadEmbeddingsMeta,
  buildTrackIndex,
  Track,
} from '@/lib/embeddings';
import { findNeighbors } from '@/lib/recommender';

interface NeighborResult {
  track: Track;
  index: number;
  similarity: number;
}

interface NeighborsResponse {
  target: { track: Track; index: number };
  neighbors: NeighborResult[];
}

interface ErrorResponse {
  error: string;
}

/**
 * Get nearest neighbors for a track (for visualization).
 * GET /api/neighbors/[id]?n=100
 */
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<NeighborsResponse | ErrorResponse>
) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { id, n = '100' } = req.query;
  
  if (!id || typeof id !== 'string') {
    return res.status(400).json({ error: 'Track ID is required' });
  }

  const numNeighbors = Math.min(parseInt(n as string, 10) || 100, 200);

  try {
    const [tracks, embeddings, meta] = await Promise.all([
      loadTracks(),
      loadEmbeddings(),
      loadEmbeddingsMeta(),
    ]);

    // Find track index
    const trackIndex = buildTrackIndex(tracks);
    const targetIdx = trackIndex.get(id);
    
    if (targetIdx === undefined) {
      return res.status(404).json({ error: 'Track not found' });
    }

    // Find neighbors
    const neighbors = findNeighbors(
      embeddings,
      tracks,
      meta.dimensions,
      targetIdx,
      numNeighbors
    );

    res.status(200).json({
      target: { track: tracks[targetIdx], index: targetIdx },
      neighbors,
    });
  } catch (error) {
    console.error('Neighbors error:', error);
    res.status(500).json({ error: 'Failed to find neighbors' });
  }
}
