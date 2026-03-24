from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class RunConfig:
    run_root: Path
    current_symlink: Path
    default_run_prefix: str


@dataclass(frozen=True)
class CollectConfig:
    market: str
    max_playlists: int
    playlists_per_query: int
    tracks_per_playlist: int
    top_artists: int
    top_genres: int
    year_buckets: list[int]
    query_keywords: list[str]
    retry_count: int
    retry_backoff_seconds: float
    local_mpd_path: str


@dataclass(frozen=True)
class CorpusConfig:
    min_playlist_tracks: int
    max_playlist_tracks: int
    drop_duplicate_tracks: bool


@dataclass(frozen=True)
class SplitConfig:
    eligible_min_tracks: int
    eligible_max_tracks: int
    train_fraction: float
    val_fraction: float
    random_seed: int
    min_seed_tracks: int
    max_seed_tracks: int
    min_positive_tracks: int
    max_positive_tracks: int


@dataclass(frozen=True)
class RetrievalConfig:
    vector_sizes: list[int]
    windows: list[int]
    min_counts: list[int]
    negatives: list[int]
    epochs: int
    sg: int
    seed: int
    workers: int
    candidate_pool_size: int


@dataclass(frozen=True)
class RerankerConfig:
    max_iter: int
    max_depth: int
    learning_rate: float
    l2_regularization: float
    random_seed: int
    max_seed_neighbor_probe: int


@dataclass(frozen=True)
class ExportConfig:
    server_dir: Path
    public_dir: Path
    public_viz_tracks: int
    tsne_perplexity: int
    tsne_iter: int
    tsne_random_state: int


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    run: RunConfig
    collect: CollectConfig
    corpus: CorpusConfig
    split: SplitConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    export: ExportConfig


def _resolve_path(project_root: Path, raw_value: str) -> Path:
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return project_root / path


def load_config(config_path: str | Path | None = None) -> PipelineConfig:
    project_root = Path(__file__).resolve().parents[2]
    path = Path(config_path) if config_path else project_root / "config" / "recommender_v2.toml"
    with path.open("rb") as handle:
        payload: dict[str, Any] = tomllib.load(handle)

    run = payload["run"]
    collect = payload["collect"]
    corpus = payload["corpus"]
    split = payload["split"]
    retrieval = payload["retrieval"]
    reranker = payload["reranker"]
    export = payload["export"]

    return PipelineConfig(
        project_root=project_root,
        run=RunConfig(
            run_root=_resolve_path(project_root, run["run_root"]),
            current_symlink=_resolve_path(project_root, run["current_symlink"]),
            default_run_prefix=run["default_run_prefix"],
        ),
        collect=CollectConfig(**collect),
        corpus=CorpusConfig(**corpus),
        split=SplitConfig(**split),
        retrieval=RetrievalConfig(**retrieval),
        reranker=RerankerConfig(**reranker),
        export=ExportConfig(
            server_dir=_resolve_path(project_root, export["server_dir"]),
            public_dir=_resolve_path(project_root, export["public_dir"]),
            public_viz_tracks=export["public_viz_tracks"],
            tsne_perplexity=export["tsne_perplexity"],
            tsne_iter=export["tsne_iter"],
            tsne_random_state=export["tsne_random_state"],
        ),
    )
