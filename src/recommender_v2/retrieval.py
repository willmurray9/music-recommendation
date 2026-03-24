from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
import json

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec

from .config import PipelineConfig
from .metrics import (
    catalog_coverage,
    mean_popularity_percentile,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    same_artist_rate,
    unique_artist_coverage,
    intra_list_diversity,
)
from .paths import RunLayout
from .utils import cosine_similarity, percentile_rank, read_jsonl, write_json


@dataclass(frozen=True)
class RetrievalExperiment:
    vector_size: int
    window: int
    min_count: int
    negative: int

    @property
    def name(self) -> str:
        return (
            f"vs{self.vector_size}-w{self.window}-mc{self.min_count}-"
            f"neg{self.negative}"
        )


def train_retrieval(config: PipelineConfig, layout: RunLayout) -> dict:
    playlist_tracks_df = pd.read_parquet(layout.normalized_dir / "playlist_tracks.parquet")
    track_df = pd.read_parquet(layout.normalized_dir / "tracks.parquet").drop_duplicates(
        subset="track_uri", keep="first"
    )
    split_rows = read_jsonl(layout.splits_dir / "val.jsonl")

    sequences = _load_train_sequences(layout)
    experiments = [
        RetrievalExperiment(*values)
        for values in product(
            config.retrieval.vector_sizes,
            config.retrieval.windows,
            config.retrieval.min_counts,
            config.retrieval.negatives,
        )
    ]

    track_lookup = track_df.set_index("track_uri").to_dict("index")
    popularity_values = np.sort(track_df["playlist_support"].fillna(0).astype(float).to_numpy())

    best_summary: dict | None = None
    for experiment in experiments:
        model = Word2Vec(
            sentences=sequences,
            vector_size=experiment.vector_size,
            window=experiment.window,
            min_count=experiment.min_count,
            workers=config.retrieval.workers,
            epochs=config.retrieval.epochs,
            sg=config.retrieval.sg,
            negative=experiment.negative,
            seed=config.retrieval.seed,
        )
        model_path = layout.models_dir / f"{experiment.name}.wordvectors"
        model.wv.save(str(model_path))

        metrics = evaluate_retrieval_model(
            model.wv,
            split_rows,
            track_lookup,
            popularity_values,
            config.retrieval.candidate_pool_size,
        )
        summary = {
            "experiment": experiment.name,
            "params": experiment.__dict__,
            "model_path": str(model_path.relative_to(layout.root)),
            "metrics": metrics,
        }
        write_json(layout.metrics_dir / f"{experiment.name}.json", summary)
        if best_summary is None or metrics["recall@50"] > best_summary["metrics"]["recall@50"]:
            best_summary = summary

    assert best_summary is not None
    best_vectors = KeyedVectors.load(str(layout.root / best_summary["model_path"]))
    best_vectors.save(str(layout.models_dir / "retrieval_best.wordvectors"))
    write_json(layout.manifests_dir / "train_retrieval.json", best_summary)
    return best_summary


def evaluate_retrieval_model(
    wv: KeyedVectors,
    split_rows: list[dict],
    track_lookup: dict[str, dict],
    popularity_values: np.ndarray,
    candidate_pool_size: int,
) -> dict[str, float]:
    recommendation_lists: list[list[str]] = []
    artist_lookup = {uri: row.get("artist_uri") for uri, row in track_lookup.items()}
    popularity_lookup = {
        uri: percentile_rank(popularity_values, float(row.get("playlist_support", 0))) for uri, row in track_lookup.items()
    }
    vector_lookup = {uri: wv[uri] for uri in wv.key_to_index.keys()}

    metrics = defaultdict(list)
    for row in split_rows:
        predictions = retrieve_candidates(
            wv,
            row["seed_tracks"],
            topn=candidate_pool_size,
        )
        positives = set(row["positive_tracks"])
        recommendation_lists.append(predictions[:50])
        metrics["recall@10"].append(recall_at_k(predictions, positives, 10))
        metrics["recall@50"].append(recall_at_k(predictions, positives, 50))
        metrics["ndcg@10"].append(ndcg_at_k(predictions, positives, 10))
        metrics["ndcg@50"].append(ndcg_at_k(predictions, positives, 50))
        metrics["mrr@10"].append(mrr_at_k(predictions, positives, 10))
        metrics["same_artist_rate"].append(same_artist_rate(predictions[:50], artist_lookup))

    summary = {name: float(np.mean(values)) if values else 0.0 for name, values in metrics.items()}
    summary["catalog_coverage@50"] = catalog_coverage(recommendation_lists, len(track_lookup))
    summary["unique_artist_coverage@50"] = unique_artist_coverage(recommendation_lists, artist_lookup)
    summary["mean_popularity_percentile"] = mean_popularity_percentile(recommendation_lists, popularity_lookup)
    summary["intra_list_diversity"] = intra_list_diversity(recommendation_lists, vector_lookup)
    return summary


def retrieve_candidates(
    wv: KeyedVectors,
    seed_tracks: list[str],
    topn: int,
) -> list[str]:
    valid = [seed for seed in seed_tracks if seed in wv]
    if not valid:
        return []
    centroid = np.mean([wv[seed] for seed in valid], axis=0)
    similar = wv.similar_by_vector(centroid, topn=topn + len(valid))
    return [uri for uri, _ in similar if uri not in valid][:topn]


def _load_train_sequences(layout: RunLayout) -> list[list[str]]:
    playlist_tracks_df = pd.read_parquet(layout.normalized_dir / "playlist_tracks.parquet")
    train_split = read_jsonl(layout.splits_dir / "train.jsonl")
    train_playlist_ids = {row["playlist_id"] for row in train_split}
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in playlist_tracks_df.sort_values(["playlist_id", "position"]).itertuples(index=False):
        if row.playlist_id in train_playlist_ids:
            grouped[row.playlist_id].append(row.track_uri)
    return list(grouped.values())
