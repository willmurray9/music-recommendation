import type { NextApiRequest, NextApiResponse } from 'next';
import { loadGenres } from '@/lib/embeddings';

interface GenresResponse {
  genres: { name: string; count: number }[];
}

interface ErrorResponse {
  error: string;
}

/**
 * Get list of available genres.
 * GET /api/genres
 */
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<GenresResponse | ErrorResponse>
) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const genres = await loadGenres();
    res.status(200).json({ genres });
  } catch (error) {
    console.error('Genres error:', error);
    res.status(500).json({ error: 'Failed to load genres' });
  }
}
