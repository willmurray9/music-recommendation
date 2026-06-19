import {
  cosineSimilarity,
  findSimilar,
  getEmbedding,
  type RerankerModel,
  type Track,
} from './embeddings';

export interface CandidateFeatures {
  meanSeedCosine: number;
  maxSeedCosine: number;
  seedNeighborHits: number;
  sameArtist: number;
  genreOverlap: number;
  tagOverlap: number;
  releaseYearDistance: number;
  durationDelta: number;
  popularityPercentile: number;
  playlistSupport: number;
}

export function scoreRerankerModel(
  model: RerankerModel,
  features: number[],
): number {
  let score = model.baseline;
  for (const tree of model.trees) {
    let nodeIndex = 0;
    while (true) {
      const node = tree.nodes[nodeIndex];
      if (node.is_leaf) {
        score += node.value;
        break;
      }
      const featureValue = features[node.feature_idx] ?? 0;
      nodeIndex = featureValue <= node.num_threshold ? node.left : node.right;
    }
  }
  return score;
}

export function buildSeedNeighborHits(
  seedIndices: number[],
  embeddings: Float32Array,
  tracks: Track[],
  dimensions: number,
  probe: number,
): Map<number, number> {
  const hits = new Map<number, number>();
  for (const seedIndex of seedIndices) {
    const seedVector = getEmbedding(embeddings, seedIndex, dimensions);
    const neighbors = findSimilar(
      seedVector,
      embeddings,
      tracks.length,
      dimensions,
      probe,
      new Set([seedIndex]),
    );
    for (const neighbor of neighbors) {
      hits.set(neighbor.index, (hits.get(neighbor.index) || 0) + 1);
    }
  }
  return hits;
}

export function computeCandidateFeatureVector(
  candidateIndex: number,
  seedIndices: number[],
  tracks: Track[],
  embeddings: Float32Array,
  dimensions: number,
  playlistPercentiles: number[],
  seedNeighborHits: Map<number, number>,
): number[] {
  const candidateTrack = tracks[candidateIndex];
  const candidateVector = getEmbedding(embeddings, candidateIndex, dimensions);
  const seedVectors = seedIndices.map(index => getEmbedding(embeddings, index, dimensions));
  const seedTracks = seedIndices.map(index => tracks[index]);
  const seedArtists = new Set(seedTracks.map(track => track.artistId || track.artist));
  const seedGenres = new Set(seedTracks.flatMap(track => track.genres || []));
  const seedTags = new Set(seedTracks.flatMap(track => track.tags || []));
  const releaseYears = seedTracks.map(track => track.releaseYear).filter((value): value is number => typeof value === 'number');
  const meanReleaseYear = releaseYears.length > 0
    ? releaseYears.reduce((sum, year) => sum + year, 0) / releaseYears.length
    : null;
  const meanDuration = seedTracks.length > 0
    ? seedTracks.reduce((sum, track) => sum + (track.durationMs || 0), 0) / seedTracks.length
    : 0;
  const seedCosines = seedVectors.map(seedVector => cosineSimilarity(seedVector, candidateVector));
  const popularityPercentile = binarySearchPercentile(
    playlistPercentiles,
    candidateTrack.playlistCount || candidateTrack.support || 0,
  ) / 100;

  return [
    average(seedCosines),
    seedCosines.length > 0 ? Math.max(...seedCosines) : 0,
    seedNeighborHits.get(candidateIndex) || 0,
    seedArtists.has(candidateTrack.artistId || candidateTrack.artist) ? 1 : 0,
    overlapCount(seedGenres, new Set(candidateTrack.genres || [])),
    overlapCount(seedTags, new Set(candidateTrack.tags || [])),
    meanReleaseYear !== null && candidateTrack.releaseYear
      ? Math.abs(candidateTrack.releaseYear - meanReleaseYear)
      : 99,
    meanDuration > 0 ? Math.abs((candidateTrack.durationMs || 0) - meanDuration) : 0,
    popularityPercentile,
    candidateTrack.support || candidateTrack.playlistCount || 0,
  ];
}

export function mmrRescore(
  candidateIndex: number,
  currentScore: number,
  selectedIndices: number[],
  embeddings: Float32Array,
  dimensions: number,
  artistDiversity: number,
): number {
  if (selectedIndices.length === 0) return currentScore;

  const candidateVector = getEmbedding(embeddings, candidateIndex, dimensions);
  let redundancy = 0;
  for (const selectedIndex of selectedIndices) {
    redundancy = Math.max(
      redundancy,
      cosineSimilarity(candidateVector, getEmbedding(embeddings, selectedIndex, dimensions)),
    );
  }

  const lambdaWeight = 0.75 - artistDiversity * 0.25;
  return lambdaWeight * currentScore - (1 - lambdaWeight) * redundancy;
}

function average(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function overlapCount(left: Set<string>, right: Set<string>): number {
  let count = 0;
  for (const value of left) {
    if (right.has(value)) count += 1;
  }
  return count;
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
