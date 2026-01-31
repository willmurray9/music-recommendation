'use client';

interface Track {
  id: string;
  name: string;
  artist: string;
  album: string;
  genres: string[];
  popularity: number;
  playlistCount: number;
}

interface TrackListProps {
  title: string;
  tracks: { track: Track; index: number }[];
  onRemove?: (index: number) => void;
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
  const bgColor = variant === 'seeds' ? 'bg-spotify-green/10' : 'bg-yellow-500/10';
  const accentColor = variant === 'seeds' ? 'text-spotify-green' : 'text-yellow-400';
  const iconBg = variant === 'seeds' ? 'bg-spotify-green' : 'bg-yellow-500';

  return (
    <div className="bg-spotify-gray/50 rounded-xl p-4">
      <h3 className={`font-bold mb-4 ${accentColor}`}>{title}</h3>

      {tracks.length === 0 ? (
        <p className="text-spotify-light text-sm py-4 text-center">{emptyMessage}</p>
      ) : (
        <div className="space-y-2">
          {tracks.map(({ track, index }, i) => (
            <div
              key={track.id}
              className={`${bgColor} rounded-lg p-3 flex items-center gap-3 
                         ${onSelect ? 'cursor-pointer hover:bg-opacity-80 transition' : ''}`}
              onClick={() => onSelect?.(track, index)}
            >
              <div className={`w-8 h-8 ${iconBg} rounded flex items-center justify-center text-black font-bold text-sm`}>
                {i + 1}
              </div>
              
              <div className="flex-1 min-w-0">
                <div className="text-white font-medium truncate">{track.name}</div>
                <div className="text-spotify-light text-sm truncate">{track.artist}</div>
              </div>

              {showScore && scores[i] !== undefined && (
                <div className="text-spotify-light text-xs">
                  {(scores[i] * 100).toFixed(0)}%
                </div>
              )}

              <div className="text-right">
                <div className="text-spotify-light text-xs">
                  Pop: {track.popularity}
                </div>
                {track.genres.length > 0 && (
                  <div className="text-spotify-light text-xs truncate max-w-[100px]">
                    {track.genres[0]}
                  </div>
                )}
              </div>

              {onRemove && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemove(index);
                  }}
                  className="text-spotify-light hover:text-white transition p-1"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
