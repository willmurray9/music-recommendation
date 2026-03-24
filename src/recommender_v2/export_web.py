from __future__ import annotations

from collections import Counter, defaultdict
import json

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE

from .config import PipelineConfig
from .paths import RunLayout


def export_web(config: PipelineConfig, layout: RunLayout) -> dict:
    track_df = pd.read_parquet(layout.normalized_dir / "tracks.parquet").drop_duplicates(
        subset="track_uri", keep="first"
    ).copy()
    track_df["genres"] = track_df["genres"].apply(_normalize_list)
    track_df["tags"] = track_df.get("tags", pd.Series(index=track_df.index)).apply(_normalize_list)

    server_dir = config.export.server_dir
    public_dir = config.export.public_dir
    server_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)

    wv = KeyedVectors.load(str(layout.models_dir / "retrieval_best.wordvectors"))
    in_vocab = track_df["track_uri"].isin(set(wv.key_to_index.keys()))
    track_df = track_df[in_vocab].drop_duplicates(subset="track_uri", keep="first").copy()

    embeddings = np.array([wv[uri] for uri in track_df["track_uri"]], dtype=np.float32)
    _write_track_bundle(track_df, embeddings, server_dir, model_version=layout.root.name)

    reranker_path = layout.models_dir / "reranker_model.json"
    if reranker_path.exists():
        reranker_target = server_dir / "reranker_model.json"
        reranker_target.write_text(reranker_path.read_text(encoding="utf-8"), encoding="utf-8")

    viz_df = track_df.sort_values(["playlist_support", "artist_popularity"], ascending=[False, False]).head(
        config.export.public_viz_tracks
    )
    viz_embeddings = np.array([wv[uri] for uri in viz_df["track_uri"]], dtype=np.float32)
    _write_track_bundle(viz_df, viz_embeddings, public_dir, model_version=layout.root.name)
    _write_viz_index(track_df, viz_df, public_dir)
    _write_tsne(config, viz_embeddings, public_dir)

    summary = {
        "server_tracks": int(len(track_df)),
        "public_tracks": int(len(viz_df)),
        "model_version": layout.root.name,
    }
    (layout.manifests_dir / "export_web.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_track_bundle(track_df: pd.DataFrame, embeddings: np.ndarray, output_dir, model_version: str) -> None:
    tracks = []
    for row in track_df.itertuples(index=False):
        tracks.append(
            {
                "id": row.track_uri,
                "name": row.track_name,
                "artist": row.artist_name,
                "album": row.album_name,
                "artistId": row.artist_id,
                "genres": _normalize_list(row.genres)[:5],
                "tags": _normalize_list(row.tags)[:10],
                "popularity": int(row.artist_popularity or row.popularity or 0),
                "playlistCount": int(row.playlist_support or 0),
                "support": int(row.playlist_support or 0),
                "durationMs": int(row.duration_ms or 0),
                "releaseYear": int(row.release_year) if row.release_year else None,
                "modelVersion": model_version,
            }
        )
    with (output_dir / "tracks.json").open("w", encoding="utf-8") as handle:
        json.dump({"tracks": tracks}, handle, separators=(",", ":"))

    embeddings.astype(np.float32).tofile(output_dir / "embeddings.bin")
    with (output_dir / "embeddings_meta.json").open("w", encoding="utf-8") as handle:
        json.dump({"numTracks": int(embeddings.shape[0]), "dimensions": int(embeddings.shape[1])}, handle)

    search_index = defaultdict(list)
    for idx, track in enumerate(tracks):
        for token in _tokenize(track["name"]) + _tokenize(track["artist"]) + _tokenize(" ".join(track["tags"])):
            if idx not in search_index[token]:
                search_index[token].append(idx)
    with (output_dir / "search_index.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(search_index), handle, separators=(",", ":"))

    genre_counts = Counter()
    for track in tracks:
        for genre in track["genres"]:
            genre_counts[genre] += 1
    with (output_dir / "genres.json").open("w", encoding="utf-8") as handle:
        json.dump({"genres": [{"name": genre, "count": count} for genre, count in genre_counts.most_common(250)]}, handle)


def _write_viz_index(server_df: pd.DataFrame, viz_df: pd.DataFrame, public_dir) -> None:
    viz_index = {track_uri: idx for idx, track_uri in enumerate(viz_df["track_uri"].tolist())}
    with (public_dir / "viz_index.json").open("w", encoding="utf-8") as handle:
        json.dump(viz_index, handle, separators=(",", ":"))


def _write_tsne(config: PipelineConfig, embeddings: np.ndarray, public_dir) -> None:
    tsne = TSNE(
        n_components=3,
        perplexity=config.export.tsne_perplexity,
        max_iter=config.export.tsne_iter,
        random_state=config.export.tsne_random_state,
        verbose=0,
    )
    coords = tsne.fit_transform(embeddings)
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    normalized = 2 * (coords - coords_min) / (coords_max - coords_min) - 1
    with (public_dir / "tsne_coords.json").open("w", encoding="utf-8") as handle:
        json.dump({"coords": normalized.tolist()}, handle, separators=(",", ":"))


def _tokenize(text: str) -> list[str]:
    import re

    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 2]


def _normalize_list(value) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return [str(item) for item in value.tolist()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []
