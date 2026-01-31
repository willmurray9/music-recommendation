#!/usr/bin/env python3
"""
Export data for the music recommender web app.

This script generates:
1. tracks.json - Track metadata with all needed fields
2. embeddings.bin - Binary embeddings (Float32Array for JS)
3. tsne_coords.json - Pre-computed 3D t-SNE coordinates
4. search_index.json - Inverted index for fast search
5. genres.json - List of top genres for filtering
"""

import json
import struct
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from collections import defaultdict, Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "web" / "public" / "data"

def load_data():
    """Load embeddings, track metadata, and artist info."""
    print("Loading data...")
    
    # Load track embeddings
    wv = KeyedVectors.load(str(MODEL_DIR / "track2vec.wordvectors"))
    print(f"  Loaded {len(wv):,} track embeddings (dim={wv.vector_size})")
    
    # Load track metadata
    track_df = pd.read_parquet(MODEL_DIR / "track_metadata.parquet")
    print(f"  Loaded {len(track_df):,} tracks metadata")
    
    # Load artist info
    artist_df = pd.read_parquet(DATA_DIR / "artist_info.parquet")
    print(f"  Loaded {len(artist_df):,} artists info")
    
    return wv, track_df, artist_df


def build_enriched_data(wv, track_df, artist_df):
    """Merge track and artist data, filter to vocab."""
    print("\nBuilding enriched dataset...")
    
    # Extract IDs from URIs
    def extract_id(uri):
        return uri.split(":")[-1]
    
    track_df = track_df.copy()
    track_df['track_id'] = track_df['track_uri'].apply(extract_id)
    track_df['artist_id'] = track_df['artist_uri'].apply(extract_id)
    
    # Load playlist counts from raw data (if available)
    PLAYLIST_DATA_PATH = Path.home() / ".cache/kagglehub/datasets/himanshuwagh/spotify-million/versions/1/data"
    
    if PLAYLIST_DATA_PATH.exists():
        print("  Loading playlist counts from raw data...")
        slice_files = sorted(PLAYLIST_DATA_PATH.glob("mpd.slice.*.json"))
        track_playlist_count = Counter()
        for slice_file in slice_files:
            with open(slice_file) as f:
                data = json.load(f)
                for playlist in data["playlists"]:
                    for track in playlist["tracks"]:
                        track_playlist_count[track["track_uri"]] += 1
        track_df['playlist_count'] = track_df['track_uri'].map(track_playlist_count).fillna(0).astype(int)
    else:
        print("  Warning: Raw playlist data not found, using 0 for playlist counts")
        track_df['playlist_count'] = 0
    
    # Merge with artist info
    artist_df_slim = artist_df.drop(columns=['artist_name'], errors='ignore')
    enriched_df = track_df.merge(artist_df_slim, on='artist_id', how='left')
    
    # Normalize genres
    def normalize_genres(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return []
        if hasattr(x, 'tolist'):
            return x.tolist()
        if isinstance(x, list):
            return x
        return []
    
    enriched_df['genres'] = enriched_df['genres'].apply(normalize_genres)
    enriched_df['artist_popularity'] = enriched_df['artist_popularity'].fillna(50).astype(int)
    enriched_df['artist_followers'] = enriched_df['artist_followers'].fillna(0).astype(int)
    
    # Filter to only tracks in vocabulary
    vocab_set = set(wv.key_to_index.keys())
    enriched_df = enriched_df[enriched_df['track_uri'].isin(vocab_set)]
    
    # Drop duplicates
    enriched_df = enriched_df.drop_duplicates(subset='track_uri', keep='first')
    
    print(f"  Enriched dataset: {len(enriched_df):,} tracks in vocabulary")
    
    return enriched_df, vocab_set


def export_tracks_json(enriched_df, output_dir):
    """Export track metadata as JSON."""
    print("\nExporting tracks.json...")
    
    tracks = []
    for _, row in enriched_df.iterrows():
        tracks.append({
            "id": row['track_uri'],
            "name": row['track_name'],
            "artist": row['artist_name'],
            "album": row.get('album_name', ''),
            "genres": row['genres'][:5],  # Limit to 5 genres
            "popularity": int(row['artist_popularity']),
            "playlistCount": int(row['playlist_count']),
        })
    
    output_path = output_dir / "tracks.json"
    with open(output_path, 'w') as f:
        json.dump({"tracks": tracks}, f, separators=(',', ':'))
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {len(tracks):,} tracks ({size_mb:.1f} MB)")
    
    return tracks


def export_embeddings_binary(wv, enriched_df, output_dir):
    """Export embeddings as binary Float32Array."""
    print("\nExporting embeddings.bin...")
    
    # Get embeddings in the same order as tracks
    track_uris = enriched_df['track_uri'].tolist()
    embeddings = np.array([wv[uri] for uri in track_uris], dtype=np.float32)
    
    # Save as binary (can be loaded as Float32Array in JS)
    output_path = output_dir / "embeddings.bin"
    embeddings.tofile(output_path)
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {embeddings.shape[0]:,} embeddings ({embeddings.shape[1]} dims, {size_mb:.1f} MB)")
    
    # Also save dimensions metadata
    meta_path = output_dir / "embeddings_meta.json"
    with open(meta_path, 'w') as f:
        json.dump({
            "numTracks": embeddings.shape[0],
            "dimensions": embeddings.shape[1]
        }, f)
    
    return embeddings


def compute_tsne(embeddings, output_dir):
    """Compute 3D t-SNE coordinates."""
    print("\nComputing t-SNE (this may take 10-30 minutes)...")
    
    tsne = TSNE(
        n_components=3,
        perplexity=30,
        max_iter=1000,
        random_state=42,
        verbose=1
    )
    
    coords_3d = tsne.fit_transform(embeddings)
    
    # Normalize to [-1, 1] range for easier rendering
    coords_min = coords_3d.min(axis=0)
    coords_max = coords_3d.max(axis=0)
    coords_normalized = 2 * (coords_3d - coords_min) / (coords_max - coords_min) - 1
    
    # Save as JSON
    output_path = output_dir / "tsne_coords.json"
    with open(output_path, 'w') as f:
        json.dump({
            "coords": coords_normalized.tolist()
        }, f, separators=(',', ':'))
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved t-SNE coordinates ({size_mb:.1f} MB)")
    
    return coords_normalized


def build_search_index(tracks, output_dir):
    """Build inverted index for fast search."""
    print("\nBuilding search index...")
    
    # Tokenize function
    def tokenize(text):
        if not text:
            return []
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'[a-z0-9]+', text.lower())
        return [t for t in tokens if len(t) >= 2]
    
    # Build inverted index
    index = defaultdict(list)
    
    for i, track in enumerate(tracks):
        # Index track name tokens
        for token in tokenize(track['name']):
            if i not in index[token]:
                index[token].append(i)
        
        # Index artist name tokens
        for token in tokenize(track['artist']):
            if i not in index[token]:
                index[token].append(i)
    
    # Convert to regular dict and save
    output_path = output_dir / "search_index.json"
    with open(output_path, 'w') as f:
        json.dump(dict(index), f, separators=(',', ':'))
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved search index ({len(index):,} tokens, {size_mb:.1f} MB)")


def export_genres(enriched_df, output_dir):
    """Export list of top genres."""
    print("\nExporting genres.json...")
    
    # Count tracks per genre
    genre_counts = Counter()
    for genres in enriched_df['genres']:
        for genre in genres:
            genre_counts[genre] += 1
    
    # Get top 50 genres
    top_genres = [{"name": g, "count": c} for g, c in genre_counts.most_common(50)]
    
    output_path = output_dir / "genres.json"
    with open(output_path, 'w') as f:
        json.dump({"genres": top_genres}, f, separators=(',', ':'))
    
    print(f"  Saved {len(top_genres)} genres")


def main():
    """Main export function."""
    print("=" * 60)
    print("Music Recommender Data Export")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Load data
    wv, track_df, artist_df = load_data()
    
    # Build enriched dataset
    enriched_df, vocab_set = build_enriched_data(wv, track_df, artist_df)
    
    # Export tracks
    tracks = export_tracks_json(enriched_df, OUTPUT_DIR)
    
    # Export embeddings
    embeddings = export_embeddings_binary(wv, enriched_df, OUTPUT_DIR)
    
    # Compute and export t-SNE
    compute_tsne(embeddings, OUTPUT_DIR)
    
    # Build search index
    build_search_index(tracks, OUTPUT_DIR)
    
    # Export genres
    export_genres(enriched_df, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print(f"\nFiles created in {OUTPUT_DIR}:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        size = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size:.2f} MB")


if __name__ == "__main__":
    main()
