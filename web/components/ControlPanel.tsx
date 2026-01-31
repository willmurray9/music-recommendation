'use client';

import { useState, useEffect } from 'react';

interface Genre {
  name: string;
  count: number;
}

interface ControlPanelProps {
  popularity: number;
  artistDiversity: number;
  exploration: number;
  selectedGenres: string[];
  onPopularityChange: (value: number) => void;
  onArtistDiversityChange: (value: number) => void;
  onExplorationChange: (value: number) => void;
  onGenresChange: (genres: string[]) => void;
}

function Slider({
  label,
  value,
  onChange,
  leftLabel,
  rightLabel,
  description,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  leftLabel: string;
  rightLabel: string;
  description?: string;
}) {
  return (
    <div className="mb-6">
      <div className="flex justify-between items-center mb-2">
        <label className="text-white font-medium">{label}</label>
        <span className="text-spotify-green font-mono">{Math.round(value * 100)}%</span>
      </div>
      <input
        type="range"
        min="0"
        max="100"
        value={value * 100}
        onChange={(e) => onChange(parseInt(e.target.value) / 100)}
        className="w-full h-2 bg-spotify-gray rounded-lg appearance-none cursor-pointer
                   [&::-webkit-slider-thumb]:appearance-none
                   [&::-webkit-slider-thumb]:w-4
                   [&::-webkit-slider-thumb]:h-4
                   [&::-webkit-slider-thumb]:bg-spotify-green
                   [&::-webkit-slider-thumb]:rounded-full
                   [&::-webkit-slider-thumb]:cursor-pointer
                   [&::-webkit-slider-thumb]:hover:scale-110
                   [&::-webkit-slider-thumb]:transition-transform"
      />
      <div className="flex justify-between text-xs text-spotify-light mt-1">
        <span>{leftLabel}</span>
        <span>{rightLabel}</span>
      </div>
      {description && (
        <p className="text-xs text-spotify-light mt-2">{description}</p>
      )}
    </div>
  );
}

export default function ControlPanel({
  popularity,
  artistDiversity,
  exploration,
  selectedGenres,
  onPopularityChange,
  onArtistDiversityChange,
  onExplorationChange,
  onGenresChange,
}: ControlPanelProps) {
  const [genres, setGenres] = useState<Genre[]>([]);
  const [isGenreDropdownOpen, setIsGenreDropdownOpen] = useState(false);

  useEffect(() => {
    fetch('/api/genres')
      .then((res) => res.json())
      .then((data) => setGenres(data.genres || []))
      .catch(console.error);
  }, []);

  const toggleGenre = (genre: string) => {
    if (selectedGenres.includes(genre)) {
      onGenresChange(selectedGenres.filter((g) => g !== genre));
    } else {
      onGenresChange([...selectedGenres, genre]);
    }
  };

  return (
    <div className="bg-spotify-gray/50 rounded-xl p-6">
      <h2 className="text-white text-lg font-bold mb-6">Control Knobs</h2>

      <Slider
        label="Popularity"
        value={popularity}
        onChange={onPopularityChange}
        leftLabel="Underground"
        rightLabel="Mainstream"
        description="Prefer obscure gems or popular hits"
      />

      <Slider
        label="Artist Diversity"
        value={artistDiversity}
        onChange={onArtistDiversityChange}
        leftLabel="Same Artists"
        rightLabel="New Artists"
        description="Discover new artists or stay familiar"
      />

      <Slider
        label="Exploration"
        value={exploration}
        onChange={onExplorationChange}
        leftLabel="Safe"
        rightLabel="Adventurous"
        description="Predictable picks or surprising discoveries"
      />

      {/* Genre Filter */}
      <div className="mt-6">
        <label className="text-white font-medium block mb-2">Genre Filter</label>
        <div className="relative">
          <button
            onClick={() => setIsGenreDropdownOpen(!isGenreDropdownOpen)}
            className="w-full px-4 py-2 bg-spotify-black text-left text-white rounded-lg
                       flex items-center justify-between hover:bg-spotify-black/80 transition"
          >
            <span className="truncate">
              {selectedGenres.length === 0
                ? 'All genres'
                : `${selectedGenres.length} genre${selectedGenres.length > 1 ? 's' : ''} selected`}
            </span>
            <svg
              className={`w-5 h-5 transition-transform ${isGenreDropdownOpen ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {isGenreDropdownOpen && (
            <div className="absolute z-50 w-full mt-2 bg-spotify-black rounded-lg shadow-xl 
                            max-h-60 overflow-y-auto border border-spotify-gray">
              {genres.slice(0, 30).map((genre) => (
                <button
                  key={genre.name}
                  onClick={() => toggleGenre(genre.name)}
                  className={`w-full px-4 py-2 text-left flex items-center justify-between
                             hover:bg-spotify-gray/50 transition
                             ${selectedGenres.includes(genre.name) ? 'bg-spotify-green/20' : ''}`}
                >
                  <span className="text-white">{genre.name}</span>
                  <span className="text-spotify-light text-xs">{genre.count.toLocaleString()}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Selected genre tags */}
        {selectedGenres.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-3">
            {selectedGenres.map((genre) => (
              <span
                key={genre}
                className="px-3 py-1 bg-spotify-green/20 text-spotify-green rounded-full text-sm
                           flex items-center gap-1"
              >
                {genre}
                <button
                  onClick={() => toggleGenre(genre)}
                  className="hover:text-white transition"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
