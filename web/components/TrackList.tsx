'use client';

import { useState } from 'react';
import {
  TrackDetailsButton,
  TrackDetailsPanel,
  TrackIdentity,
} from '@/components/TrackIdentity';
import { Track, getTrackDescription } from '@/lib/trackDisplay';

interface TrackListProps {
  title: string;
  tracks: { track: Track; index: number }[];
  onRemove?: (trackId: string) => void;
  onSelect?: (track: Track, index: number) => void;
  emptyMessage?: string;
  showScore?: boolean;
  scores?: number[];
  variant?: 'seeds' | 'recommendations';
}

export default function TrackList({
  title,
  tracks,
  onRemove,
  onSelect,
  emptyMessage = 'No tracks selected',
  showScore = false,
  scores = [],
  variant = 'seeds',
}: TrackListProps) {
  const [expandedTrackId, setExpandedTrackId] = useState<string | null>(null);
  const bgColor = variant === 'seeds' ? 'bg-spotify-green/10' : 'bg-yellow-500/10';
  const accentColor = variant === 'seeds' ? 'text-spotify-green' : 'text-yellow-400';
  const iconBg = variant === 'seeds' ? 'bg-spotify-green' : 'bg-yellow-500';
  const actionLabel = variant === 'recommendations' ? 'Add as seed' : 'Select track';

  return (
    <div className="bg-spotify-gray/50 rounded-xl p-4">
      <h3 className={`font-bold mb-4 ${accentColor}`}>{title}</h3>

      {tracks.length === 0 ? (
        <p className="text-spotify-light text-sm py-4 text-center">{emptyMessage}</p>
      ) : (
        <div className="space-y-2">
          {tracks.map(({ track, index }, i) => {
            const score = showScore ? scores[i] : undefined;
            const detailsId = `track-details-${variant}-${track.id.replace(/[^a-zA-Z0-9_-]/g, '-')}`;
            const isExpanded = expandedTrackId === track.id;
            const description = getTrackDescription(track, { context: title, score });

            return (
              <div
                key={track.id}
                className={`${bgColor} rounded-lg p-3 transition hover:bg-opacity-80`}
                title={description}
                aria-label={description}
              >
                <div className="flex items-start gap-3">
                  <div className={`mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded ${iconBg} text-sm font-bold text-black`}>
                    {i + 1}
                  </div>

                  {onSelect ? (
                    <button
                      type="button"
                      onClick={() => onSelect(track, index)}
                      className="min-w-0 flex-1 rounded-md text-left transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-spotify-green focus-visible:ring-offset-2 focus-visible:ring-offset-spotify-black"
                      aria-label={`${actionLabel}. ${description}`}
                      title={`${actionLabel}: ${description}`}
                    >
                      <TrackIdentity track={track} score={score} />
                    </button>
                  ) : (
                    <TrackIdentity track={track} score={score} className="flex-1" />
                  )}

                  <div className="flex shrink-0 items-center gap-1">
                    <TrackDetailsButton
                      controlsId={detailsId}
                      expanded={isExpanded}
                      track={track}
                      score={score}
                      context={title}
                      onClick={() => setExpandedTrackId(isExpanded ? null : track.id)}
                    />

                    {onSelect && variant === 'recommendations' && (
                      <button
                        type="button"
                        onClick={() => onSelect(track, index)}
                        className="inline-flex items-center gap-1 rounded-full bg-yellow-500 px-3 py-1.5 text-xs font-bold text-black transition hover:bg-yellow-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-yellow-300 focus-visible:ring-offset-2 focus-visible:ring-offset-spotify-black"
                        aria-label={`Add as seed. ${description}`}
                        title={`Add as seed: ${description}`}
                      >
                        <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 5v14M5 12h14" />
                        </svg>
                        Add
                      </button>
                    )}

                    {onRemove && (
                      <button
                        type="button"
                        onClick={() => onRemove(track.id)}
                        className="track-icon-button text-spotify-light hover:text-white focus-visible:text-white"
                        aria-label={`Remove seed track. ${description}`}
                        title={`Remove seed track: ${description}`}
                      >
                        <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    )}
                  </div>
                </div>

                {isExpanded && (
                  <TrackDetailsPanel id={detailsId} track={track} score={score} className="mt-3" />
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
