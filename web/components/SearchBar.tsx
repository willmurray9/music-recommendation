'use client';

import { useState, useMemo, useRef, useEffect } from 'react';
import {
  TrackDetailsButton,
  TrackDetailsPanel,
  TrackIdentity,
} from '@/components/TrackIdentity';
import { debounce } from '@/lib/utils';
import { Track, getTrackDescription } from '@/lib/trackDisplay';

interface SearchResult {
  track: Track;
  index: number;
}

interface SearchBarProps {
  onTrackSelect: (track: Track, index: number) => void;
  placeholder?: string;
}

export default function SearchBar({ onTrackSelect, placeholder = 'Search for songs or artists...' }: SearchBarProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [expandedTrackId, setExpandedTrackId] = useState<string | null>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const resultsId = 'track-search-results';

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const searchTracks = useMemo(
    () => debounce(async (q: string) => {
      if (q.length < 2) {
        setResults([]);
        setIsOpen(false);
        setExpandedTrackId(null);
        return;
      }

      setIsLoading(true);
      try {
        const response = await fetch(`/api/search?q=${encodeURIComponent(q)}&limit=10`);
        const data = await response.json();
        setResults(data.results || []);
        setExpandedTrackId(null);
        setIsOpen(true);
      } catch (error) {
        console.error('Search error:', error);
        setResults([]);
        setExpandedTrackId(null);
      } finally {
        setIsLoading(false);
      }
    }, 300),
    []
  );

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
    searchTracks(value);
  };

  const handleSelect = (result: SearchResult) => {
    onTrackSelect(result.track, result.index);
    setQuery('');
    setResults([]);
    setIsOpen(false);
    setExpandedTrackId(null);
  };

  return (
    <div ref={wrapperRef} className="relative w-full">
      <div className="relative">
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          onFocus={() => results.length > 0 && setIsOpen(true)}
          placeholder={placeholder}
          aria-controls={resultsId}
          aria-label="Search tracks by song title or artist"
          className="w-full px-4 py-3 bg-spotify-gray text-white rounded-full 
                     placeholder-spotify-light focus:outline-none focus:ring-2 
                     focus:ring-spotify-green transition-all"
        />
        {isLoading && (
          <div className="absolute right-4 top-1/2 -translate-y-1/2">
            <div className="w-5 h-5 border-2 border-spotify-green border-t-transparent rounded-full animate-spin" />
          </div>
        )}
      </div>

      {isOpen && results.length > 0 && (
        <div
          id={resultsId}
          className="absolute z-50 mt-2 max-h-[70vh] w-full overflow-y-auto rounded-lg bg-spotify-gray shadow-xl"
          role="list"
          aria-label="Track search results"
        >
          {results.map((result) => {
            const detailsId = `search-track-details-${result.track.id.replace(/[^a-zA-Z0-9_-]/g, '-')}`;
            const isExpanded = expandedTrackId === result.track.id;
            const description = getTrackDescription(result.track, { context: 'Search result' });

            return (
              <div
                key={result.track.id}
                className="border-b border-white/5 px-3 py-3 last:border-b-0 hover:bg-spotify-black/40"
                title={description}
                role="listitem"
                aria-label={description}
              >
                <div className="flex items-start gap-3">
                  <button
                    type="button"
                    onClick={() => handleSelect(result)}
                    className="flex min-w-0 flex-1 items-start gap-3 rounded-md text-left transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-spotify-green focus-visible:ring-offset-2 focus-visible:ring-offset-spotify-gray"
                    aria-label={`Add as seed from search. ${description}`}
                    title={`Add as seed: ${description}`}
                  >
                    <span className="mt-1 flex h-10 w-10 shrink-0 items-center justify-center rounded bg-spotify-black">
                      <svg className="h-5 w-5 text-spotify-light" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                        <path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/>
                      </svg>
                    </span>
                    <TrackIdentity track={result.track} compact maxGenres={2} />
                  </button>

                  <div className="flex shrink-0 items-center gap-1 pt-1">
                    <TrackDetailsButton
                      controlsId={detailsId}
                      expanded={isExpanded}
                      track={result.track}
                      context="Search result"
                      onClick={() => setExpandedTrackId(isExpanded ? null : result.track.id)}
                    />

                    <button
                      type="button"
                      onClick={() => handleSelect(result)}
                      className="inline-flex items-center gap-1 rounded-full bg-spotify-green px-3 py-1.5 text-xs font-bold text-black transition hover:bg-green-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-spotify-green focus-visible:ring-offset-2 focus-visible:ring-offset-spotify-gray"
                      aria-label={`Add as seed from search. ${description}`}
                      title={`Add as seed: ${description}`}
                    >
                      <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 5v14M5 12h14" />
                      </svg>
                      Add
                    </button>
                  </div>
                </div>

                {isExpanded && (
                  <TrackDetailsPanel id={detailsId} track={result.track} className="mt-3" />
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
