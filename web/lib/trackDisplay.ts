export interface Track {
  id: string;
  name: string;
  artist: string;
  album: string;
  genres: string[];
  tags?: string[];
  popularity: number;
  playlistCount: number;
  support?: number;
  durationMs?: number;
  releaseYear?: number | null;
  modelVersion?: string;
}

interface TrackDescriptionOptions {
  action?: string;
  context?: string;
  score?: number;
}

const compactNumberFormatter = new Intl.NumberFormat('en-US', {
  notation: 'compact',
  maximumFractionDigits: 1,
});

const numberFormatter = new Intl.NumberFormat('en-US');

export function formatDuration(durationMs?: number): string | null {
  if (!durationMs || durationMs <= 0) return null;

  const totalSeconds = Math.round(durationMs / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }

  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

export function formatScore(score?: number): string | null {
  if (typeof score !== 'number') return null;
  return `${Math.round(score * 100)}% match`;
}

export function getTrackMetadataItems(track: Track, score?: number): { label: string; value: string }[] {
  const duration = formatDuration(track.durationMs);
  const items: { label: string; value: string }[] = [
    { label: 'Artist', value: track.artist },
    { label: 'Album', value: track.album },
  ];

  if (typeof track.releaseYear === 'number') {
    items.push({ label: 'Release year', value: String(track.releaseYear) });
  }

  items.push({ label: 'Popularity', value: `${track.popularity}/100` });
  items.push({ label: 'Playlist count', value: numberFormatter.format(track.playlistCount) });

  if (duration) {
    items.push({ label: 'Duration', value: duration });
  }

  if (track.genres.length > 0) {
    items.push({ label: 'Genres', value: track.genres.join(', ') });
  }

  const scoreLabel = formatScore(score);
  if (scoreLabel) {
    items.push({ label: 'Recommendation score', value: scoreLabel });
  }

  if (track.modelVersion) {
    items.push({ label: 'Model', value: track.modelVersion });
  }

  return items;
}

export function getInlineMetadata(track: Track, score?: number): string[] {
  const duration = formatDuration(track.durationMs);
  const scoreLabel = formatScore(score);
  const items = [
    typeof track.releaseYear === 'number' ? String(track.releaseYear) : null,
    `Pop ${track.popularity}`,
    `${compactNumberFormatter.format(track.playlistCount)} playlists`,
    duration,
    scoreLabel,
  ];

  return items.filter((item): item is string => Boolean(item));
}

export function getTrackDescription(track: Track, options: TrackDescriptionOptions = {}): string {
  const metadata = getTrackMetadataItems(track, options.score)
    .map(({ label, value }) => `${label}: ${value}`)
    .join('. ');
  const context = options.context ? `${options.context}. ` : '';
  const action = options.action ? `${options.action}. ` : '';

  return `${action}${context}${track.name} by ${track.artist}. ${metadata}.`;
}
