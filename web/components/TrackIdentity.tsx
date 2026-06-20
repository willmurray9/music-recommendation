'use client';

import {
  Track,
  getInlineMetadata,
  getTrackDescription,
  getTrackMetadataItems,
} from '@/lib/trackDisplay';

interface TrackIdentityProps {
  track: Track;
  score?: number;
  compact?: boolean;
  maxGenres?: number;
  className?: string;
}

interface TrackDetailsPanelProps {
  id: string;
  track: Track;
  score?: number;
  className?: string;
}

interface TrackDetailsButtonProps {
  controlsId: string;
  expanded: boolean;
  track: Track;
  score?: number;
  context?: string;
  onClick: () => void;
}

export function TrackIdentity({
  track,
  score,
  compact = false,
  maxGenres = 3,
  className = '',
}: TrackIdentityProps) {
  const metadata = getInlineMetadata(track, score);

  return (
    <div className={`min-w-0 ${className}`} title={getTrackDescription(track, { score })}>
      <div className={`track-title-2 text-white font-semibold ${compact ? 'text-sm' : 'text-[0.95rem]'}`}>
        {track.name}
      </div>

      <div className="mt-1 text-sm leading-5 text-spotify-light">
        <span className="text-white/80">{track.artist}</span>
        {track.album && (
          <span>
            <span aria-hidden="true"> · </span>
            {track.album}
          </span>
        )}
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        {metadata.map((item) => (
          <span
            key={item}
            className="rounded-full bg-white/5 px-2 py-0.5 text-[11px] font-medium leading-4 text-spotify-light"
          >
            {item}
          </span>
        ))}
        {track.genres.slice(0, maxGenres).map((genre) => (
          <span
            key={genre}
            className="rounded-full border border-spotify-green/30 bg-spotify-green/10 px-2 py-0.5 text-[11px] font-medium leading-4 text-spotify-green"
          >
            {genre}
          </span>
        ))}
      </div>
    </div>
  );
}

export function TrackDetailsPanel({ id, track, score, className = '' }: TrackDetailsPanelProps) {
  const details = getTrackMetadataItems(track, score);

  return (
    <div
      id={id}
      className={`rounded-lg border border-white/10 bg-black/30 px-3 py-3 ${className}`}
      title={getTrackDescription(track, { score })}
    >
      <div className="text-sm font-semibold leading-5 text-white">{track.name}</div>
      <dl className="mt-3 grid grid-cols-1 gap-2 text-xs sm:grid-cols-2">
        {details.map(({ label, value }) => (
          <div key={label} className="min-w-0">
            <dt className="text-[10px] font-semibold uppercase text-spotify-light/70">
              {label}
            </dt>
            <dd className="mt-0.5 break-words text-white/90">{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export function TrackDetailsButton({
  controlsId,
  expanded,
  track,
  score,
  context,
  onClick,
}: TrackDetailsButtonProps) {
  const action = expanded ? 'Hide details' : 'Show details';
  const description = getTrackDescription(track, { context, score });

  return (
    <button
      type="button"
      onClick={onClick}
      aria-expanded={expanded}
      aria-controls={controlsId}
      aria-label={`${action} for ${description}`}
      title={`${action}: ${description}`}
      className="track-icon-button text-spotify-light hover:text-white focus-visible:text-white"
    >
      <svg
        className={`h-5 w-5 transition-transform ${expanded ? 'rotate-180' : ''}`}
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        aria-hidden="true"
      >
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      </svg>
    </button>
  );
}
