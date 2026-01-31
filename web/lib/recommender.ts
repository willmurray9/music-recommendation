/**
 * Controllable Recommender ported from Python.
 * 
 * Control Parameters:
 * - popularity: 0.0 (obscure) to 1.0 (mainstream)
 * - artistDiversity: 0.0 (same artists) to 1.0 (new artists only)
 * - exploration: 0.0 (deterministic) to 1.0 (random sampling)
 * - genres: list of genres to include (empty = all)
 * - excludeGenres: list of genres to exclude
 */

import {
  Track,
  getEmbedding,
  computeCentroid,
  findSimilar,
  cosineSimilarity,
} from './embeddings';

export interface RecommendationParams {
  seedIndices: number[];
  n?: number;
  popularity?: number; // 0-1
  artistDiversity?: number; // 0-1
  exploration?: number; // 0-1
  genres?: string[];
  excludeGenres?: string[];
  excludeArtists?: string[];
  candidatePoolSize?: number;
}

export interface RecommendationResult {
  track: Track;
  index: number;
  score: number;
}

/**
 * Compute popularity score for a track (0-1).
 */
function computePopularityScore(
  track: Track,
  playlistPercentiles: number[],
): number {
  const playlistPercentile = 
    binarySearchPercentile(playlistPercentiles, track.playlistCount) / 100;
  const artistPercentile = track.popularity / 100;
  
  // Combine with weights
  return 0.6 * playlistPercentile + 0.4 * artistPercentile;
}

/**
 * Binary search for percentile position.
 */
function binarySearchPercentile(percentiles: number[], value: number): number {
  let left = 0;
  let right = percentiles.length - 1;
  
  while (left < right) {
    const mid = Math.floor((left + right) / 2);
    if (percentiles[mid] < value) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  
  return left;
}

/**
 * Compute percentiles for an array of values.
 */
export function computePercentiles(values: number[]): number[] {
  const sorted = [...values].sort((a, b) => a - b);
  const percentiles: number[] = [];
  
  for (let i = 0; i <= 100; i++) {
    const idx = Math.floor((i / 100) * (sorted.length - 1));
    percentiles.push(sorted[idx]);
  }
  
  return percentiles;
}

/**
 * Filter candidates based on constraints.
 */
function filterCandidates(
  candidates: { index: number; score: number }[],
  tracks: Track[],
  seedArtists: Set<string>,
  params: RecommendationParams,
): { index: number; score: number }[] {
  const { genres, excludeGenres, excludeArtists, artistDiversity = 0.5 } = params;
  
  const excludeArtistSet = new Set(excludeArtists || []);
  const excludeGenreSet = new Set(excludeGenres || []);
  const includeGenreSet = genres && genres.length > 0 ? new Set(genres) : null;
  
  return candidates
    .map(({ index, score }) => {
      const track = tracks[index];
      const artist = track.artist;
      const trackGenres = new Set(track.genres);
      
      // Exclude specific artists
      if (excludeArtistSet.has(artist)) {
        return null;
      }
      
      // Genre filtering
      const trackGenresArray = Array.from(trackGenres);
      if (excludeGenreSet.size > 0) {
        for (const g of trackGenresArray) {
          if (excludeGenreSet.has(g)) return null;
        }
      }
      
      if (includeGenreSet) {
        let hasMatchingGenre = false;
        for (const g of trackGenresArray) {
          if (includeGenreSet.has(g)) {
            hasMatchingGenre = true;
            break;
          }
        }
        if (!hasMatchingGenre) return null;
      }
      
      // Artist diversity: at high values, exclude seed artists entirely
      // At low values, just reduce their score
      let adjustedScore = score;
      if (seedArtists.has(artist)) {
        if (artistDiversity >= 0.8) {
          // High diversity: completely exclude seed artists
          return null;
        } else if (artistDiversity > 0) {
          // Medium diversity: penalize seed artists heavily
          // Scale from 0% penalty at 0 to 95% penalty at 0.8
          const penaltyFactor = (artistDiversity / 0.8) * 0.95;
          adjustedScore = score * (1 - penaltyFactor);
        }
      }
      
      return { index, score: adjustedScore };
    })
    .filter((c): c is { index: number; score: number } => c !== null);
}

/**
 * Apply popularity weighting to candidates.
 */
function applyPopularityWeighting(
  candidates: { index: number; score: number }[],
  tracks: Track[],
  popularity: number,
  playlistPercentiles: number[],
): { index: number; score: number }[] {
  if (popularity === 0.5) {
    return candidates;
  }
  
  return candidates.map(({ index, score }) => {
    const track = tracks[index];
    const popScore = computePopularityScore(track, playlistPercentiles);
    
    let weight: number;
    if (popularity > 0.5) {
      // Boost popular tracks
      weight = 1 + (popularity - 0.5) * 2 * popScore;
    } else {
      // Boost obscure tracks
      weight = 1 + (0.5 - popularity) * 2 * (1 - popScore);
    }
    
    return { index, score: score * weight };
  });
}

/**
 * Apply exploration factor to select final tracks.
 */
function applyExploration(
  candidates: { index: number; score: number }[],
  n: number,
  exploration: number,
): number[] {
  if (candidates.length === 0) return [];
  
  if (exploration === 0) {
    // Deterministic: top N by score
    const sorted = [...candidates].sort((a, b) => b.score - a.score);
    return sorted.slice(0, n).map(c => c.index);
  }
  
  // Probabilistic sampling with temperature-based softmax
  const temperature = 0.1 + exploration * 2.0;
  const scores = candidates.map(c => c.score);
  const maxScore = Math.max(...scores);
  
  // Compute exp(score / temperature) with numerical stability
  const expScores = scores.map(s => Math.exp((s - maxScore) / temperature));
  const sumExp = expScores.reduce((a, b) => a + b, 0);
  const probs = expScores.map(e => e / sumExp);
  
  // Weighted random sampling without replacement
  const selected: number[] = [];
  const available = candidates.map((c, i) => ({ ...c, prob: probs[i] }));
  
  for (let i = 0; i < Math.min(n, candidates.length); i++) {
    // Normalize remaining probabilities
    const totalProb = available.reduce((sum, c) => sum + c.prob, 0);
    const normalizedProbs = available.map(c => c.prob / totalProb);
    
    // Sample one
    const rand = Math.random();
    let cumulative = 0;
    let selectedIdx = 0;
    
    for (let j = 0; j < normalizedProbs.length; j++) {
      cumulative += normalizedProbs[j];
      if (rand <= cumulative) {
        selectedIdx = j;
        break;
      }
    }
    
    selected.push(available[selectedIdx].index);
    available.splice(selectedIdx, 1);
  }
  
  return selected;
}

/**
 * Generate recommendations with controllable parameters.
 */
export function recommend(
  embeddings: Float32Array,
  tracks: Track[],
  dimensions: number,
  playlistPercentiles: number[],
  params: RecommendationParams,
): RecommendationResult[] {
  const {
    seedIndices,
    n = 10,
    popularity = 0.5,
    artistDiversity = 0.5,
    exploration = 0,
    candidatePoolSize = 500,
  } = params;
  
  if (seedIndices.length === 0) {
    return [];
  }
  
  // Get seed artists
  const seedArtists = new Set(seedIndices.map(i => tracks[i].artist));
  const seedSet = new Set(seedIndices);
  
  // Compute centroid of seed tracks
  const centroid = computeCentroid(embeddings, seedIndices, dimensions);
  
  // Find similar tracks
  let candidates = findSimilar(
    centroid,
    embeddings,
    tracks.length,
    dimensions,
    candidatePoolSize,
    seedSet,
  );
  
  // Apply filters
  candidates = filterCandidates(candidates, tracks, seedArtists, params);
  
  // Apply popularity weighting
  candidates = applyPopularityWeighting(
    candidates,
    tracks,
    popularity,
    playlistPercentiles,
  );
  
  // Apply exploration and select final tracks
  const selectedIndices = applyExploration(candidates, n, exploration);
  
  // Build results
  return selectedIndices.map(index => {
    const candidateScore = candidates.find(c => c.index === index)?.score || 0;
    return {
      track: tracks[index],
      index,
      score: candidateScore,
    };
  });
}

/**
 * Find nearest neighbors for visualization.
 */
export function findNeighbors(
  embeddings: Float32Array,
  tracks: Track[],
  dimensions: number,
  targetIndex: number,
  n: number = 100,
): { track: Track; index: number; similarity: number }[] {
  const targetVec = getEmbedding(embeddings, targetIndex, dimensions);
  
  const neighbors: { index: number; similarity: number }[] = [];
  
  for (let i = 0; i < tracks.length; i++) {
    if (i === targetIndex) continue;
    
    const vec = getEmbedding(embeddings, i, dimensions);
    const similarity = cosineSimilarity(targetVec, vec);
    neighbors.push({ index: i, similarity });
  }
  
  neighbors.sort((a, b) => b.similarity - a.similarity);
  
  return neighbors.slice(0, n).map(({ index, similarity }) => ({
    track: tracks[index],
    index,
    similarity,
  }));
}
