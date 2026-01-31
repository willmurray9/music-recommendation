'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { debounce } from '@/lib/utils';

interface Track {
  id: string;
  name: string;
  artist: string;
  album: string;
  genres: string[];
  popularity: number;
  playlistCount: number;
}

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
  const wrapperRef = useRef<HTMLDivElement>(null);

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

  const searchTracks = useCallback(
    debounce(async (q: string) => {
      if (q.length < 2) {
        setResults([]);
        setIsOpen(false);
        return;
      }

      setIsLoading(true);
      try {
        const response = await fetch(`/api/search?q=${encodeURIComponent(q)}&limit=10`);
        const data = await response.json();
        setResults(data.results || []);
        setIsOpen(true);
      } catch (error) {
        console.error('Search error:', error);
        setResults([]);
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
        <div className="absolute z-50 w-full mt-2 bg-spotify-gray rounded-lg shadow-xl overflow-hidden">
          {results.map((result) => (
            <button
              key={result.track.id}
              onClick={() => handleSelect(result)}
              className="w-full px-4 py-3 text-left hover:bg-spotify-black/50 
                         transition-colors flex items-center gap-3"
            >
              <div className="w-10 h-10 bg-spotify-black rounded flex items-center justify-center">
                <svg className="w-5 h-5 text-spotify-light" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/>
                </svg>
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-white font-medium truncate">{result.track.name}</div>
                <div className="text-spotify-light text-sm truncate">{result.track.artist}</div>
              </div>
              <div className="text-spotify-light text-xs">
                {result.track.genres.slice(0, 2).join(', ')}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
