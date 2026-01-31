'use client';

import { useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import SearchBar from '@/components/SearchBar';
import ControlPanel from '@/components/ControlPanel';
import TrackList from '@/components/TrackList';

// Dynamically import 3D visualization (client-side only)
const Visualization3D = dynamic(() => import('@/components/Visualization3D'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full bg-spotify-dark flex items-center justify-center">
      <div className="text-spotify-light">Loading 3D visualization...</div>
    </div>
  ),
});

interface Track {
  id: string;
  name: string;
  artist: string;
  album: string;
  genres: string[];
  popularity: number;
  playlistCount: number;
}

interface RecommendationResult {
  track: Track;
  index: number;
  score: number;
}

export default function Home() {
  // Visualization data
  const [tsneCoords, setTsneCoords] = useState<number[][] | null>(null);
  const [allTracks, setAllTracks] = useState<Track[]>([]);
  const [isLoadingViz, setIsLoadingViz] = useState(true);
  const [vizError, setVizError] = useState<string | null>(null);

  // Seed tracks
  const [seedTracks, setSeedTracks] = useState<{ track: Track; index: number }[]>([]);

  // Control parameters
  const [popularity, setPopularity] = useState(0.5);
  const [artistDiversity, setArtistDiversity] = useState(0.5);
  const [exploration, setExploration] = useState(0);
  const [selectedGenres, setSelectedGenres] = useState<string[]>([]);

  // Recommendations
  const [recommendations, setRecommendations] = useState<RecommendationResult[]>([]);
  const [isLoadingRecs, setIsLoadingRecs] = useState(false);

  // Visualization settings
  const [colorBy, setColorBy] = useState<'popularity' | 'genre'>('popularity');

  // Load visualization data on mount
  useEffect(() => {
    async function loadVisualizationData() {
      try {
        setIsLoadingViz(true);
        setVizError(null);

        // Load tracks and t-SNE coordinates in parallel
        const [tracksRes, tsneRes] = await Promise.all([
          fetch('/data/tracks.json'),
          fetch('/data/tsne_coords.json'),
        ]);

        if (!tracksRes.ok) {
          throw new Error('Failed to load tracks data');
        }

        const tracksData = await tracksRes.json();
        setAllTracks(tracksData.tracks);

        if (tsneRes.ok) {
          const tsneData = await tsneRes.json();
          setTsneCoords(tsneData.coords);
        } else {
          // t-SNE might not be ready yet
          setVizError('3D visualization data is still being generated. The search and recommendations will work without it.');
        }
      } catch (error) {
        console.error('Error loading visualization data:', error);
        setVizError('Failed to load visualization data');
      } finally {
        setIsLoadingViz(false);
      }
    }

    loadVisualizationData();
  }, []);

  // Fetch recommendations when seeds or params change
  const fetchRecommendations = useCallback(async () => {
    if (seedTracks.length === 0) {
      setRecommendations([]);
      return;
    }

    setIsLoadingRecs(true);
    try {
      const response = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          seedTracks: seedTracks.map((s) => s.track.id),
          n: 10,
          popularity,
          artistDiversity,
          exploration,
          genres: selectedGenres.length > 0 ? selectedGenres : undefined,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setRecommendations(data.recommendations);
      }
    } catch (error) {
      console.error('Error fetching recommendations:', error);
    } finally {
      setIsLoadingRecs(false);
    }
  }, [seedTracks, popularity, artistDiversity, exploration, selectedGenres]);

  // Debounce recommendation fetching
  useEffect(() => {
    const timer = setTimeout(fetchRecommendations, 300);
    return () => clearTimeout(timer);
  }, [fetchRecommendations]);

  // Add seed track
  const handleAddSeed = (track: Track, index: number) => {
    if (seedTracks.length >= 5) {
      alert('Maximum 5 seed tracks allowed');
      return;
    }
    if (seedTracks.some((s) => s.track.id === track.id)) {
      return; // Already added
    }
    setSeedTracks([...seedTracks, { track, index }]);
  };

  // Remove seed track
  const handleRemoveSeed = (index: number) => {
    setSeedTracks(seedTracks.filter((s) => s.index !== index));
  };

  // Handle track selection from visualization
  const handleVizTrackSelect = (index: number) => {
    if (allTracks[index]) {
      handleAddSeed(allTracks[index], index);
    }
  };

  return (
    <main className="min-h-screen bg-spotify-black">
      {/* Header */}
      <header className="bg-gradient-to-b from-spotify-gray to-spotify-black px-6 py-4">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <svg className="w-8 h-8 text-spotify-green" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
            </svg>
            Music Recommender
          </h1>
          <p className="text-spotify-light mt-1">
            Explore 62,000+ tracks with controllable AI recommendations
          </p>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Panel - Search & Seeds */}
          <div className="lg:col-span-3 space-y-6">
            <SearchBar onTrackSelect={handleAddSeed} />

            <TrackList
              title="Seed Tracks"
              tracks={seedTracks}
              onRemove={handleRemoveSeed}
              emptyMessage="Search and add tracks to get recommendations"
              variant="seeds"
            />

            <ControlPanel
              popularity={popularity}
              artistDiversity={artistDiversity}
              exploration={exploration}
              selectedGenres={selectedGenres}
              onPopularityChange={setPopularity}
              onArtistDiversityChange={setArtistDiversity}
              onExplorationChange={setExploration}
              onGenresChange={setSelectedGenres}
            />
          </div>

          {/* Center - 3D Visualization */}
          <div className="lg:col-span-6">
            <div className="bg-spotify-gray/30 rounded-xl overflow-hidden">
              <div className="flex items-center justify-between px-4 py-3 border-b border-spotify-gray">
                <h2 className="text-white font-semibold">Track Embeddings</h2>
                <div className="flex items-center gap-4">
                  <label className="flex items-center gap-2 text-sm text-spotify-light">
                    Color by:
                    <select
                      value={colorBy}
                      onChange={(e) => setColorBy(e.target.value as 'popularity' | 'genre')}
                      className="bg-spotify-black text-white px-2 py-1 rounded"
                    >
                      <option value="popularity">Popularity</option>
                      <option value="genre">Genre</option>
                    </select>
                  </label>
                </div>
              </div>

              <div className="h-[500px]">
                {isLoadingViz ? (
                  <div className="w-full h-full flex items-center justify-center bg-spotify-dark">
                    <div className="text-center">
                      <div className="w-12 h-12 border-4 border-spotify-green border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                      <p className="text-spotify-light">Loading visualization...</p>
                    </div>
                  </div>
                ) : vizError ? (
                  <div className="w-full h-full flex items-center justify-center bg-spotify-dark p-8">
                    <div className="text-center">
                      <svg className="w-16 h-16 text-spotify-gray mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                      </svg>
                      <p className="text-spotify-light">{vizError}</p>
                      <p className="text-spotify-light text-sm mt-2">
                        Search and recommendations are still available below.
                      </p>
                    </div>
                  </div>
                ) : tsneCoords && allTracks.length > 0 ? (
                  <Visualization3D
                    coords={tsneCoords}
                    tracks={allTracks}
                    selectedIndices={seedTracks.map((s) => s.index)}
                    recommendedIndices={recommendations.map((r) => r.index)}
                    onTrackSelect={handleVizTrackSelect}
                    colorBy={colorBy}
                  />
                ) : null}
              </div>

              {/* Legend */}
              <div className="px-4 py-3 border-t border-spotify-gray flex items-center gap-6 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-spotify-green rounded-full" />
                  <span className="text-spotify-light">Seed tracks</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-yellow-500 rounded-full" />
                  <span className="text-spotify-light">Recommendations</span>
                </div>
                <div className="text-spotify-light ml-auto">
                  Click on points to add as seeds. Scroll to zoom, drag to rotate.
                </div>
              </div>
            </div>
          </div>

          {/* Right Panel - Recommendations */}
          <div className="lg:col-span-3">
            <div className="sticky top-6">
              {isLoadingRecs ? (
                <div className="bg-spotify-gray/50 rounded-xl p-6 flex items-center justify-center">
                  <div className="w-8 h-8 border-3 border-yellow-500 border-t-transparent rounded-full animate-spin" />
                </div>
              ) : (
                <TrackList
                  title="Recommendations"
                  tracks={recommendations.map((r) => ({ track: r.track, index: r.index }))}
                  onSelect={handleAddSeed}
                  emptyMessage={
                    seedTracks.length === 0
                      ? 'Add seed tracks to get recommendations'
                      : 'No recommendations found'
                  }
                  showScore
                  scores={recommendations.map((r) => r.score)}
                  variant="recommendations"
                />
              )}

              {recommendations.length > 0 && (
                <div className="mt-4 text-center">
                  <button
                    onClick={fetchRecommendations}
                    className="px-6 py-2 bg-spotify-green text-black font-semibold rounded-full
                               hover:bg-green-400 transition"
                  >
                    Refresh Recommendations
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
