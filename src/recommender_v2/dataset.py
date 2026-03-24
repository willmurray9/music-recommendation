from __future__ import annotations

from collections import Counter
from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import json
import os

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .paths import RunLayout
from .utils import extract_spotify_id, normalize_text, read_jsonl, safe_year, write_json, write_jsonl


def build_corpus(config: PipelineConfig, layout: RunLayout) -> dict:
    tracks_df = pd.DataFrame(read_jsonl(layout.raw_dir / "tracks.jsonl"))
    artists_df = pd.DataFrame(read_jsonl(layout.raw_dir / "artists.jsonl"))
    albums_df = pd.DataFrame(read_jsonl(layout.raw_dir / "albums.jsonl"))
    playlists_df = pd.DataFrame(read_jsonl(layout.raw_dir / "playlists.jsonl"))
    playlist_tracks_df = pd.DataFrame(read_jsonl(layout.raw_dir / "playlist_tracks.jsonl"))

    if tracks_df.empty or playlists_df.empty or playlist_tracks_df.empty:
        raise RuntimeError("Raw collection artifacts are empty; run collect_spotify first")

    tracks_df = tracks_df.drop_duplicates(subset="track_uri", keep="first").copy()
    artists_df = artists_df.drop_duplicates(subset="artist_uri", keep="first").copy()
    albums_df = albums_df.drop_duplicates(subset="album_uri", keep="first").copy()
    playlists_df = playlists_df.drop_duplicates(subset="playlist_id", keep="first").copy()

    playlist_tracks_df = playlist_tracks_df.sort_values(["playlist_id", "position"]).copy()
    if config.corpus.drop_duplicate_tracks:
        playlist_tracks_df = playlist_tracks_df.drop_duplicates(subset=["playlist_id", "track_uri"], keep="first")

    playlist_sizes = playlist_tracks_df.groupby("playlist_id")["track_uri"].nunique()
    eligible = playlist_sizes[
        (playlist_sizes >= config.corpus.min_playlist_tracks)
        & (playlist_sizes <= config.corpus.max_playlist_tracks)
    ].index
    playlist_tracks_df = playlist_tracks_df[playlist_tracks_df["playlist_id"].isin(eligible)].copy()
    playlists_df = playlists_df[playlists_df["playlist_id"].isin(eligible)].copy()
    playlists_df["num_tracks"] = playlists_df["playlist_id"].map(playlist_sizes).astype(int)

    tracks_df["track_id"] = tracks_df["track_uri"].map(extract_spotify_id)
    tracks_df["artist_id"] = tracks_df["artist_uri"].map(extract_spotify_id)
    tracks_df["album_id"] = tracks_df["album_uri"].map(extract_spotify_id)
    tracks_df["release_year"] = tracks_df["release_year"].apply(safe_year)

    artist_info_path = config.project_root / "data" / "processed" / "artist_info.parquet"
    if artist_info_path.exists():
        artist_info_df = pd.read_parquet(artist_info_path)
        artist_info_df = artist_info_df.drop_duplicates(subset="artist_id", keep="first").copy()
        artist_info_df["artist_uri"] = "spotify:artist:" + artist_info_df["artist_id"].astype(str)
        artist_info_df["genres"] = artist_info_df["genres"].apply(
            lambda value: value.tolist() if hasattr(value, "tolist") else (value or [])
        )
        artists_df = artists_df.merge(
            artist_info_df[["artist_uri", "artist_name", "genres", "artist_popularity", "artist_followers"]],
            on="artist_uri",
            how="left",
            suffixes=("", "_artist_info"),
        )
        artists_df["artist_name"] = artists_df["artist_name"].combine_first(artists_df["artist_name_artist_info"])
        artists_df["artist_popularity"] = artists_df["artist_popularity"].combine_first(
            artists_df["artist_popularity_artist_info"]
        )
        artists_df["artist_followers"] = artists_df["artist_followers"].combine_first(
            artists_df["artist_followers_artist_info"]
        )
        artists_df["genres"] = artists_df["genres"].combine_first(artists_df["genres_artist_info"])
        artists_df = artists_df.drop(
            columns=[
                "artist_name_artist_info",
                "genres_artist_info",
                "artist_popularity_artist_info",
                "artist_followers_artist_info",
            ],
            errors="ignore",
        )

    artists_df = artists_df.drop_duplicates(subset="artist_uri", keep="first").copy()

    artists_df["artist_id"] = artists_df["artist_uri"].map(extract_spotify_id)
    artists_df["genres"] = artists_df["genres"].apply(_normalize_genres)
    artists_df["artist_popularity"] = pd.to_numeric(
        artists_df["artist_popularity"], errors="coerce"
    ).fillna(0).astype(int)
    artists_df["artist_followers"] = pd.to_numeric(
        artists_df["artist_followers"], errors="coerce"
    ).fillna(0).astype(int)

    support = playlist_tracks_df.groupby("track_uri")["playlist_id"].nunique().rename("playlist_support")
    tracks_df = tracks_df.merge(support, on="track_uri", how="left")
    tracks_df["playlist_support"] = tracks_df["playlist_support"].fillna(0).astype(int)

    tracks_df = tracks_df.merge(
        artists_df[["artist_uri", "artist_popularity", "artist_followers", "genres"]],
        on="artist_uri",
        how="left",
    )
    tracks_df = tracks_df.drop_duplicates(subset="track_uri", keep="first").copy()
    tracks_df["genres"] = tracks_df["genres"].apply(_normalize_genres)
    tracks_df["tag_count"] = tracks_df["genres"].apply(len)

    tracks_df.to_parquet(layout.normalized_dir / "tracks.parquet", index=False)
    artists_df.to_parquet(layout.normalized_dir / "artists.parquet", index=False)
    albums_df.to_parquet(layout.normalized_dir / "albums.parquet", index=False)
    playlists_df.to_parquet(layout.normalized_dir / "playlists.parquet", index=False)
    playlist_tracks_df.to_parquet(layout.normalized_dir / "playlist_tracks.parquet", index=False)

    summary = {
        "tracks": int(len(tracks_df)),
        "artists": int(len(artists_df)),
        "albums": int(len(albums_df)),
        "playlists": int(len(playlists_df)),
        "playlist_tracks": int(len(playlist_tracks_df)),
    }
    write_json(layout.manifests_dir / "build_corpus.json", summary)
    return summary


def enrich_metadata(config: PipelineConfig, layout: RunLayout) -> dict:
    tracks_df = pd.read_parquet(layout.normalized_dir / "tracks.parquet")
    artists_df = pd.read_parquet(layout.normalized_dir / "artists.parquet")

    track_tags: list[dict] = []
    for row in tracks_df.itertuples(index=False):
        for genre in _normalize_genres(row.genres):
            track_tags.append(
                {
                    "track_uri": row.track_uri,
                    "tag": genre,
                    "source": "spotify_artist_genres",
                    "confidence": 0.6,
                }
            )

    lastfm_key = os.getenv("LASTFM_API_KEY")
    if lastfm_key:
        for row in tracks_df.itertuples(index=False):
            external_tags = _fetch_lastfm_tags(lastfm_key, row.artist_name, row.track_name)
            for tag, confidence in external_tags:
                track_tags.append(
                    {
                        "track_uri": row.track_uri,
                        "tag": tag,
                        "source": "lastfm_track_tags",
                        "confidence": confidence,
                    }
                )

    mb_cache_rows: list[dict] = []
    if os.getenv("MUSICBRAINZ_ENABLE") == "1":
        for row in tracks_df.itertuples(index=False):
            if row.release_year:
                continue
            mb_match = _fetch_musicbrainz_match(row.artist_name, row.track_name, row.isrc)
            if mb_match is None:
                continue
            mb_cache_rows.append({"track_uri": row.track_uri, **mb_match})
    if mb_cache_rows:
        mb_df = pd.DataFrame(mb_cache_rows).drop_duplicates(subset="track_uri", keep="first")
        tracks_df = tracks_df.merge(mb_df, on="track_uri", how="left", suffixes=("", "_mb"))
        tracks_df["release_year"] = tracks_df["release_year"].fillna(tracks_df["release_year_mb"])
        tracks_df["release_year_source"] = tracks_df["release_year_source"].combine_first(tracks_df["source_mb"])
        tracks_df["release_year_confidence"] = tracks_df["release_year_confidence"].combine_first(
            tracks_df["confidence_mb"]
        )
        tracks_df = tracks_df.drop(columns=["release_year_mb", "source_mb", "confidence_mb"], errors="ignore")

    tag_df = pd.DataFrame(track_tags)
    if tag_df.empty:
        tag_df = pd.DataFrame(columns=["track_uri", "tag", "source", "confidence"])
    tag_df = tag_df.drop_duplicates(subset=["track_uri", "tag", "source"], keep="first")

    aggregated_tags = (
        tag_df.sort_values(["track_uri", "confidence"], ascending=[True, False])
        .groupby("track_uri")["tag"]
        .apply(list)
        .rename("tags")
    )
    tracks_df = tracks_df.merge(aggregated_tags, on="track_uri", how="left")
    tracks_df["tags"] = tracks_df["tags"].apply(lambda value: value if isinstance(value, list) else [])
    tracks_df["release_year_source"] = tracks_df.get("release_year_source", pd.Series(index=tracks_df.index)).fillna(
        tracks_df["release_date"].apply(lambda value: "album_release_date" if value else None)
    )
    tracks_df["release_year_confidence"] = tracks_df.get(
        "release_year_confidence", pd.Series(index=tracks_df.index)
    ).fillna(tracks_df["release_year"].apply(lambda value: 0.85 if value else None))

    tracks_df.to_parquet(layout.normalized_dir / "tracks.parquet", index=False)
    artists_df.to_parquet(layout.normalized_dir / "artists.parquet", index=False)
    tag_df.to_parquet(layout.normalized_dir / "track_tags.parquet", index=False)

    if mb_cache_rows:
        write_jsonl(layout.raw_dir / "musicbrainz_recordings.jsonl", mb_cache_rows)
    if lastfm_key:
        write_json(layout.manifests_dir / "lastfm_enrichment.json", {"enabled": True})

    summary = {
        "track_tags": int(len(tag_df)),
        "tracks_with_tags": int((tracks_df["tags"].apply(len) > 0).sum()),
        "tracks_with_release_year": int(tracks_df["release_year"].notna().sum()),
    }
    write_json(layout.manifests_dir / "enrich_metadata.json", summary)
    return summary


def _normalize_genres(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if hasattr(value, "tolist"):
        return [str(item) for item in value.tolist()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _fetch_lastfm_tags(api_key: str, artist_name: str, track_name: str) -> list[tuple[str, float]]:
    url = (
        "https://ws.audioscrobbler.com/2.0/?method=track.getTopTags"
        f"&artist={quote_plus(artist_name)}&track={quote_plus(track_name)}"
        f"&api_key={api_key}&format=json"
    )
    try:
        with urlopen(Request(url, headers={"User-Agent": "music-recommender-v2/1.0"}), timeout=10) as response:
            payload = json.load(response)
    except Exception:
        return []
    tags = payload.get("toptags", {}).get("tag", [])
    normalized: list[tuple[str, float]] = []
    for tag in tags[:10]:
        name = tag.get("name")
        count = tag.get("count")
        if not name:
            continue
        normalized.append((name, min(float(count or 0) / 100.0, 0.95)))
    return normalized


def _fetch_musicbrainz_match(artist_name: str, track_name: str, isrc: str | None) -> dict | None:
    query_parts: list[str] = []
    if isrc:
        query_parts.append(f'isrc:{isrc}')
    query_parts.append(f'recording:"{track_name}"')
    query_parts.append(f'artist:"{artist_name}"')
    query = " AND ".join(query_parts)
    url = f"https://musicbrainz.org/ws/2/recording/?fmt=json&limit=1&query={quote_plus(query)}"
    try:
        with urlopen(Request(url, headers={"User-Agent": "music-recommender-v2/1.0"}), timeout=10) as response:
            payload = json.load(response)
    except Exception:
        return None
    recordings = payload.get("recordings") or []
    if not recordings:
        return None
    recording = recordings[0]
    release_year = None
    releases = recording.get("releases") or []
    if releases:
        release_year = safe_year(releases[0].get("date"))
    confidence = 0.95 if isrc else 0.75
    source = "musicbrainz_isrc" if isrc else "musicbrainz_name_match"
    return {
        "release_year": release_year,
        "source": source,
        "confidence": confidence,
    }
