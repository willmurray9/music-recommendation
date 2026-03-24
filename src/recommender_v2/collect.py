from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

from .config import PipelineConfig
from .paths import RunLayout
from .utils import extract_spotify_id, write_json, write_jsonl


@dataclass(frozen=True)
class QuerySeed:
    query: str
    kind: str
    seed: str


def _local_mpd_path(config: PipelineConfig) -> Path:
    configured = config.collect.local_mpd_path.strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "kagglehub" / "datasets" / "himanshuwagh" / "spotify-million" / "versions" / "1" / "data"


def build_bootstrap_queries(config: PipelineConfig) -> list[QuerySeed]:
    track_metadata_path = config.project_root / "models" / "track_metadata.parquet"
    genres_path = config.project_root / "web" / "public" / "data" / "genres.json"

    track_df = pd.read_parquet(track_metadata_path, columns=["artist_name"])
    artist_counts = track_df["artist_name"].value_counts().head(config.collect.top_artists)

    with genres_path.open("r", encoding="utf-8") as handle:
        genre_payload = json.load(handle)
    top_genres = genre_payload["genres"][: config.collect.top_genres]

    queries: list[QuerySeed] = []
    for artist_name in artist_counts.index.tolist():
        queries.append(QuerySeed(query=f'"{artist_name}"', kind="artist", seed=artist_name))
    for genre_info in top_genres:
        queries.append(QuerySeed(query=f'"{genre_info["name"]}"', kind="genre", seed=genre_info["name"]))
    for keyword in config.collect.query_keywords:
        for year_bucket in config.collect.year_buckets:
            queries.append(
                QuerySeed(
                    query=f"{keyword} {year_bucket}s",
                    kind="keyword_year",
                    seed=f"{keyword}:{year_bucket}s",
                )
            )
    return queries


def _spotify_client() -> Spotify | None:
    load_dotenv()
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return Spotify(auth_manager=auth, requests_timeout=20, retries=3)


def _normalize_track_object(track_obj: dict, source: str) -> dict:
    album = track_obj.get("album") or {}
    external_ids = track_obj.get("external_ids") or {}
    artist = (track_obj.get("artists") or [{}])[0]
    release_date = album.get("release_date")
    return {
        "track_uri": track_obj.get("uri"),
        "track_id": track_obj.get("id"),
        "track_name": track_obj.get("name"),
        "artist_uri": artist.get("uri"),
        "artist_id": artist.get("id"),
        "artist_name": artist.get("name"),
        "album_uri": album.get("uri"),
        "album_id": album.get("id"),
        "album_name": album.get("name"),
        "duration_ms": track_obj.get("duration_ms"),
        "explicit": track_obj.get("explicit"),
        "popularity": track_obj.get("popularity"),
        "isrc": external_ids.get("isrc"),
        "release_date": release_date,
        "release_year": release_date[:4] if release_date else None,
        "source": source,
    }


def collect_spotify(config: PipelineConfig, layout: RunLayout, live: bool = False) -> dict:
    queries = build_bootstrap_queries(config)
    write_jsonl(layout.raw_dir / "queries.jsonl", [query.__dict__ for query in queries])

    summary = {
        "query_count": len(queries),
        "live": live,
        "collector": "spotify_api" if live else "local_mpd_fallback",
    }

    if live:
        client = _spotify_client()
        if client is None:
            raise RuntimeError("SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET are required for live collection")
        summary.update(_collect_spotify_api(config, layout, client, queries))
    else:
        summary.update(_collect_local_mpd(config, layout))

    write_json(layout.manifests_dir / "collect_spotify.json", summary)
    return summary


def _collect_local_mpd(config: PipelineConfig, layout: RunLayout) -> dict:
    data_path = _local_mpd_path(config)
    slice_files = sorted(data_path.glob("mpd.slice.*.json"))
    if not slice_files:
        raise FileNotFoundError(f"No local MPD slices found at {data_path}")

    playlists: list[dict] = []
    playlist_tracks: list[dict] = []
    tracks: dict[str, dict] = {}
    artists: dict[str, dict] = {}
    albums: dict[str, dict] = {}

    max_playlists = config.collect.max_playlists
    for slice_file in slice_files:
        with slice_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for playlist in payload["playlists"]:
            if len(playlists) >= max_playlists:
                break
            playlist_id = f"mpd:{playlist['pid']}"
            items = playlist.get("tracks") or []
            playlists.append(
                {
                    "playlist_id": playlist_id,
                    "playlist_name": playlist.get("name"),
                    "owner_id": "mpd_local",
                    "owner_name": "mpd_local",
                    "source": "local_mpd",
                    "market": config.collect.market,
                    "query": "local_mpd_fallback",
                    "num_tracks": len(items),
                    "snapshot_id": None,
                }
            )
            for position, track in enumerate(items):
                playlist_tracks.append(
                    {
                        "playlist_id": playlist_id,
                        "track_uri": track["track_uri"],
                        "position": position,
                    }
                )
                if track["track_uri"] not in tracks:
                    tracks[track["track_uri"]] = {
                        "track_uri": track["track_uri"],
                        "track_id": extract_spotify_id(track["track_uri"]),
                        "track_name": track["track_name"],
                        "artist_uri": track["artist_uri"],
                        "artist_id": extract_spotify_id(track["artist_uri"]),
                        "artist_name": track["artist_name"],
                        "album_uri": track["album_uri"],
                        "album_id": extract_spotify_id(track["album_uri"]),
                        "album_name": track["album_name"],
                        "duration_ms": track.get("duration_ms"),
                        "explicit": None,
                        "popularity": None,
                        "isrc": None,
                        "release_date": None,
                        "release_year": None,
                        "source": "local_mpd",
                    }
                artist_uri = track["artist_uri"]
                if artist_uri not in artists:
                    artists[artist_uri] = {
                        "artist_uri": artist_uri,
                        "artist_id": extract_spotify_id(artist_uri),
                        "artist_name": track["artist_name"],
                        "artist_popularity": None,
                        "artist_followers": None,
                        "genres": [],
                        "source": "local_mpd",
                    }
                album_uri = track["album_uri"]
                if album_uri not in albums:
                    albums[album_uri] = {
                        "album_uri": album_uri,
                        "album_id": extract_spotify_id(album_uri),
                        "album_name": track["album_name"],
                        "release_date": None,
                        "release_year": None,
                        "total_tracks": None,
                        "album_type": None,
                        "source": "local_mpd",
                    }
        if len(playlists) >= max_playlists:
            break

    write_jsonl(layout.raw_dir / "playlists.jsonl", playlists)
    write_jsonl(layout.raw_dir / "playlist_tracks.jsonl", playlist_tracks)
    write_jsonl(layout.raw_dir / "tracks.jsonl", tracks.values())
    write_jsonl(layout.raw_dir / "artists.jsonl", artists.values())
    write_jsonl(layout.raw_dir / "albums.jsonl", albums.values())
    return {
        "playlist_count": len(playlists),
        "track_count": len(tracks),
        "artist_count": len(artists),
        "album_count": len(albums),
        "source": "local_mpd",
    }


def _collect_spotify_api(
    config: PipelineConfig,
    layout: RunLayout,
    client: Spotify,
    queries: list[QuerySeed],
) -> dict:
    playlists: list[dict] = []
    playlist_tracks: list[dict] = []
    tracks: dict[str, dict] = {}
    artists: dict[str, dict] = {}
    albums: dict[str, dict] = {}
    seen_playlists: set[str] = set()
    retries = config.collect.retry_count

    for query in queries:
        if len(seen_playlists) >= config.collect.max_playlists:
            break
        for attempt in range(retries):
            try:
                search_response = client.search(
                    q=query.query,
                    type="playlist",
                    limit=config.collect.playlists_per_query,
                    market=config.collect.market,
                )
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(config.collect.retry_backoff_seconds * (attempt + 1))
        playlist_items = search_response.get("playlists", {}).get("items", [])
        for playlist_meta in playlist_items:
            playlist_id = playlist_meta["id"]
            if playlist_id in seen_playlists:
                continue
            seen_playlists.add(playlist_id)
            playlist_record = {
                "playlist_id": playlist_id,
                "playlist_name": playlist_meta.get("name"),
                "owner_id": (playlist_meta.get("owner") or {}).get("id"),
                "owner_name": (playlist_meta.get("owner") or {}).get("display_name"),
                "source": "spotify_api",
                "market": config.collect.market,
                "query": query.query,
                "num_tracks": (playlist_meta.get("tracks") or {}).get("total"),
                "snapshot_id": playlist_meta.get("snapshot_id"),
            }
            playlists.append(playlist_record)
            try:
                track_page = client.playlist_items(
                    playlist_id,
                    market=config.collect.market,
                    limit=config.collect.tracks_per_playlist,
                )
            except Exception:
                continue

            items = track_page.get("items", [])
            for position, item in enumerate(items):
                track_obj = item.get("track")
                if not track_obj or track_obj.get("type") != "track":
                    continue
                track_record = _normalize_track_object(track_obj, source="spotify_api")
                playlist_tracks.append(
                    {
                        "playlist_id": playlist_id,
                        "track_uri": track_record["track_uri"],
                        "position": position,
                    }
                )
                tracks.setdefault(track_record["track_uri"], track_record)

                artist = (track_obj.get("artists") or [{}])[0]
                artists.setdefault(
                    artist.get("uri"),
                    {
                        "artist_uri": artist.get("uri"),
                        "artist_id": artist.get("id"),
                        "artist_name": artist.get("name"),
                        "artist_popularity": None,
                        "artist_followers": None,
                        "genres": [],
                        "source": "spotify_api",
                    },
                )
                album = track_obj.get("album") or {}
                albums.setdefault(
                    album.get("uri"),
                    {
                        "album_uri": album.get("uri"),
                        "album_id": album.get("id"),
                        "album_name": album.get("name"),
                        "release_date": album.get("release_date"),
                        "release_year": (album.get("release_date") or "")[:4] or None,
                        "total_tracks": album.get("total_tracks"),
                        "album_type": album.get("album_type"),
                        "source": "spotify_api",
                    },
                )

    write_jsonl(layout.raw_dir / "playlists.jsonl", playlists)
    write_jsonl(layout.raw_dir / "playlist_tracks.jsonl", playlist_tracks)
    write_jsonl(layout.raw_dir / "tracks.jsonl", tracks.values())
    write_jsonl(layout.raw_dir / "artists.jsonl", artists.values())
    write_jsonl(layout.raw_dir / "albums.jsonl", albums.values())
    return {
        "playlist_count": len(playlists),
        "track_count": len(tracks),
        "artist_count": len(artists),
        "album_count": len(albums),
        "source": "spotify_api",
    }
