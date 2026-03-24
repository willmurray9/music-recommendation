/**
 * Controllable recommender with optional exported reranker model support.
 */

import {
  Track,
  RerankerModel,
  computeCentroid,
  findSimilar,
} from './embeddings';
import {
  buildSeedNeighborHits,
  computeCandidateFeatureVector,
  mmrRescore,
  scoreRerankerModel,
} from './reranker';

export interface RecommendationParams {
  seedIndices: number[];
  n?: number;
  popularity?: number;
  artistDiversity?: number;
  exploration?: number;
  genres?: string[];
  excludeGenres?: string[];
  excludeArtists?: string[];
  candidatePoolSize?: number;
  rerankerModel?: RerankerModel | null;
  visualizationIndexMap?: Map<string, number>;
}

export interface RecommendationResult {
  track: Track;
  index: number;
  score: number;
}

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

export function computePercentiles(values: number[]): number[] {
  const sorted = [...values].sort((a, b) => a - b);
  const percentiles: number[] = [];

  for (let i = 0; i <= 100; i++) {
    const idx = Math.floor((i / 100) * (sorted.length - 1));
    percentiles.push(sorted[idx]);
  }

  return percentiles;
}

function computePopularityScore(track: Track, playlistPercentiles: number[]): number {
  const playlistPercentile =
    binarySearchPercentile(
      playlistPercentiles,
      track.support || track.playlistCount,
    ) / 100;
  const artistPercentile = track.popularity / 100;
  return 0.6 * playlistPercentile + 0.4 * artistPercentile;
}

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
      const trackGenres = new Set(track.genres || []);

      if (excludeArtistSet.has(artist)) {
        return null;
      }

      if (excludeGenreSet.size > 0) {
        for (const genre of trackGenres) {
          if (excludeGenreSet.has(genre)) return null;
        }
      }

      if (includeGenreSet) {
        let hasMatch = false;
        for (const genre of trackGenres) {
          if (includeGenreSet.has(genre)) {
            hasMatch = true;
            break;
          }
        }
        if (!hasMatch) return null;
      }

      let adjustedScore = score;
      if (seedArtists.has(track.artistId || track.artist)) {
        if (artistDiversity >= 0.85) {
          return null;
        }
        adjustedScore = score * (1 - artistDiversity * 0.85);
      }

      return { index, score: adjustedScore };
    })
    .filter((candidate): candidate is { index: number; score: number } => candidate !== null);
}

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
      weight = 1 + (popularity - 0.5) * 2 * popScore;
    } else {
      weight = 1 + (0.5 - popularity) * 2 * (1 - popScore);
    }

    return { index, score: score * weight };
  });
}

function applyExploration(
  candidates: { index: number; score: number }[],
  n: number,
  exploration: number,
): number[] {
  if (candidates.length === 0) return [];

  if (exploration === 0) {
    return candidates.slice(0, n).map(candidate => candidate.index);
  }

  const temperature = 0.1 + exploration * 2.0;
  const scores = candidates.map(candidate => candidate.score);
  const maxScore = Math.max(...scores);
  const expScores = scores.map(score => Math.exp((score - maxScore) / temperature));
  const sumExp = expScores.reduce((sum, value) => sum + value, 0);
  const probs = expScores.map(value => value / sumExp);

  const selected: number[] = [];
  const available = candidates.map((candidate, index) => ({
    ...candidate,
    prob: probs[index],
  }));

  for (let i = 0; i < Math.min(n, candidates.length); i++) {
    const totalProb = available.reduce((sum, candidate) => sum + candidate.prob, 0);
    const normalized = available.map(candidate => candidate.prob / totalProb);
    const rand = Math.random();
    let cumulative = 0;
    let selectedIdx = 0;

    for (let j = 0; j < normalized.length; j++) {
      cumulative += normalized[j];
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

function rerankCandidates(
  candidates: { index: number; score: number }[],
  tracks: Track[],
  embeddings: Float32Array,
  dimensions: number,
  playlistPercentiles: number[],
  params: RecommendationParams,
): { index: number; score: number }[] {
  const {
    seedIndices,
    rerankerModel,
    artistDiversity = 0.5,
  } = params;

  if (!rerankerModel) {
    return candidates.sort((a, b) => b.score - a.score);
  }

  const seedNeighborHits = buildSeedNeighborHits(
    seedIndices,
    embeddings,
    tracks,
    dimensions,
    100,
  );

  const scored = candidates.map(candidate => {
    const features = computeCandidateFeatureVector(
      candidate.index,
      seedIndices,
      tracks,
      embeddings,
      dimensions,
      playlistPercentiles,
      seedNeighborHits,
    );
    const rerankerScore = scoreRerankerModel(rerankerModel, features);
    return {
      index: candidate.index,
      score: rerankerScore + candidate.score * 0.15,
      baseScore: candidate.score,
    };
  });

  scored.sort((a, b) => b.score - a.score);

  const selected: { index: number; score: number }[] = [];
  const selectedIndices: number[] = [];
  const selectedArtists = new Set<string>();

  for (const candidate of scored) {
    const track = tracks[candidate.index];
    const artistKey = track.artistId || track.artist;
    if (artistDiversity >= 0.3 && selectedArtists.has(artistKey)) {
      continue;
    }

    const score = mmrRescore(
      candidate.index,
      candidate.score,
      selectedIndices,
      embeddings,
      dimensions,
      artistDiversity,
    );
    selected.push({ index: candidate.index, score });
    selectedIndices.push(candidate.index);
    selectedArtists.add(artistKey);
  }

  selected.sort((a, b) => b.score - a.score);
  return selected;
}

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
    exploration = 0,
    candidatePoolSize = 500,
    visualizationIndexMap,
  } = params;

  if (seedIndices.length === 0) {
    return [];
  }

  const seedArtists = new Set(seedIndices.map(index => tracks[index].artistId || tracks[index].artist));
  const seedSet = new Set(seedIndices);
  const centroid = computeCentroid(embeddings, seedIndices, dimensions);

  let candidates = findSimilar(
    centroid,
    embeddings,
    tracks.length,
    dimensions,
    candidatePoolSize,
    seedSet,
  );

  candidates = filterCandidates(candidates, tracks, seedArtists, params);
  candidates = applyPopularityWeighting(candidates, tracks, popularity, playlistPercentiles);
  candidates = rerankCandidates(candidates, tracks, embeddings, dimensions, playlistPercentiles, params);

  const selectedIndices = applyExploration(candidates, n, exploration);
  return selectedIndices.map(index => {
    const candidateScore = candidates.find(candidate => candidate.index === index)?.score || 0;
    return {
      track: tracks[index],
      index: visualizationIndexMap?.get(tracks[index].id) ?? -1,
      score: candidateScore,
    };
  });
}

export function findNeighbors(
  embeddings: Float32Array,
  tracks: Track[],
  dimensions: number,
  targetIndex: number,
  n: number = 100,
): { track: Track; index: number; similarity: number }[] {
  const targetVector = embeddings.subarray(
    targetIndex * dimensions,
    targetIndex * dimensions + dimensions,
  );
  const neighbors: { index: number; similarity: number }[] = [];

  for (let i = 0; i < tracks.length; i++) {
    if (i === targetIndex) continue;
    const vector = embeddings.subarray(i * dimensions, i * dimensions + dimensions);
    const similarity = (() => {
      let dotProduct = 0;
      let normA = 0;
      let normB = 0;
      for (let j = 0; j < dimensions; j++) {
        dotProduct += targetVector[j] * vector[j];
        normA += targetVector[j] * targetVector[j];
        normB += vector[j] * vector[j];
      }
      if (normA === 0 || normB === 0) return 0;
      return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    })();
    neighbors.push({ index: i, similarity });
  }

  neighbors.sort((a, b) => b.similarity - a.similarity);
  return neighbors.slice(0, n).map(({ index, similarity }) => ({
    track: tracks[index],
    index,
    similarity,
  }));
}
