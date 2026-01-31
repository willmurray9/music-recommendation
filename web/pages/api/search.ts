import type { NextApiRequest, NextApiResponse } from 'next';
import { loadTracks, loadSearchIndex, Track } from '@/lib/embeddings';

interface SearchResult {
  track: Track;
  index: number;
}

interface SearchResponse {
  results: SearchResult[];
  query: string;
}

interface ErrorResponse {
  error: string;
}

/**
 * Search for tracks by name or artist.
 * GET /api/search?q=<query>&limit=<limit>
 */
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<SearchResponse | ErrorResponse>
) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { q, limit = '20' } = req.query;
  
  if (!q || typeof q !== 'string') {
    return res.status(400).json({ error: 'Query parameter "q" is required' });
  }

  const maxResults = Math.min(parseInt(limit as string, 10) || 20, 100);

  try {
    const [tracks, searchIndex] = await Promise.all([
      loadTracks(),
      loadSearchIndex(),
    ]);

    // Tokenize query
    const queryTokens = q.toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter(t => t.length >= 2);

    if (queryTokens.length === 0) {
      return res.status(200).json({ results: [], query: q });
    }

    // Find matching track indices
    const matchCounts = new Map<number, number>();
    
    for (const token of queryTokens) {
      const matchingIndices = searchIndex[token] || [];
      for (const idx of matchingIndices) {
        matchCounts.set(idx, (matchCounts.get(idx) || 0) + 1);
      }
    }

    // Sort by match count (descending), then by playlist count (popularity)
    const sortedMatches = Array.from(matchCounts.entries())
      .map(([index, count]) => ({
        index,
        count,
        track: tracks[index],
      }))
      .sort((a, b) => {
        if (b.count !== a.count) return b.count - a.count;
        return b.track.playlistCount - a.track.playlistCount;
      })
      .slice(0, maxResults);

    const results: SearchResult[] = sortedMatches.map(({ index, track }) => ({
      track,
      index,
    }));

    res.status(200).json({ results, query: q });
  } catch (error) {
    console.error('Search error:', error);
    res.status(500).json({ error: 'Failed to search tracks' });
  }
}
