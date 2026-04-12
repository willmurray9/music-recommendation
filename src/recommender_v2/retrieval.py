from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
import time

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
from .utils import format_duration, log_event, percentile_rank, read_json, read_jsonl, write_json


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
    progress_path = layout.manifests_dir / "train_retrieval_progress.json"
    total_experiments = len(experiments)
    total_sequences = len(sequences)
    total_interactions = sum(len(sequence) for sequence in sequences)

    log_event(
        "train_retrieval",
        "loaded retrieval inputs",
        experiments=total_experiments,
        sequences=total_sequences,
        interactions=total_interactions,
        tracks=len(track_df),
        validation_rows=len(split_rows),
    )

    best_summary: dict | None = None
    sweep_started = time.perf_counter()
    for idx, experiment in enumerate(experiments, start=1):
        model_path = layout.models_dir / f"{experiment.name}.wordvectors"
        metrics_path = layout.metrics_dir / f"{experiment.name}.json"
        experiment_started = time.perf_counter()

        if metrics_path.exists() and model_path.exists():
            summary = read_json(metrics_path)
            log_event(
                "train_retrieval",
                f"[{idx}/{total_experiments}] reusing cached experiment",
                experiment=experiment.name,
                recall50=f"{summary['metrics']['recall@50']:.4f}",
                ndcg10=f"{summary['metrics']['ndcg@10']:.4f}",
            )
        else:
            log_event(
                "train_retrieval",
                f"[{idx}/{total_experiments}] training experiment",
                experiment=experiment.name,
                vector_size=experiment.vector_size,
                window=experiment.window,
                min_count=experiment.min_count,
                negative=experiment.negative,
                epochs=config.retrieval.epochs,
            )
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
            model.wv.save(str(model_path))

            metrics = evaluate_retrieval_model(
                model.wv,
                split_rows,
                track_lookup,
                popularity_values,
                config.retrieval.candidate_pool_size,
                progress_label=experiment.name,
            )
            summary = {
                "experiment": experiment.name,
                "params": experiment.__dict__,
                "model_path": str(model_path.relative_to(layout.root)),
                "metrics": metrics,
            }
            write_json(metrics_path, summary)
            log_event(
                "train_retrieval",
                f"[{idx}/{total_experiments}] finished experiment",
                experiment=experiment.name,
                duration=format_duration(time.perf_counter() - experiment_started),
                recall50=f"{metrics['recall@50']:.4f}",
                ndcg10=f"{metrics['ndcg@10']:.4f}",
            )

        if best_summary is None or summary["metrics"]["recall@50"] > best_summary["metrics"]["recall@50"]:
            best_summary = summary

        elapsed = time.perf_counter() - sweep_started
        average_per_experiment = elapsed / idx
        eta = average_per_experiment * (total_experiments - idx)
        write_json(
            progress_path,
            {
                "status": "running",
                "completed_experiments": idx,
                "total_experiments": total_experiments,
                "current_experiment": experiment.name,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "best_experiment": best_summary["experiment"],
                "best_recall@50": best_summary["metrics"]["recall@50"],
                "best_ndcg@10": best_summary["metrics"]["ndcg@10"],
            },
        )
        log_event(
            "train_retrieval",
            f"[{idx}/{total_experiments}] sweep progress",
            best=best_summary["experiment"],
            best_recall50=f"{best_summary['metrics']['recall@50']:.4f}",
            elapsed=format_duration(elapsed),
            eta=format_duration(eta),
        )

    assert best_summary is not None
    best_vectors = KeyedVectors.load(str(layout.root / best_summary["model_path"]))
    best_vectors.save(str(layout.models_dir / "retrieval_best.wordvectors"))
    write_json(layout.manifests_dir / "train_retrieval.json", best_summary)

    total_elapsed = time.perf_counter() - sweep_started
    write_json(
        progress_path,
        {
            "status": "completed",
            "completed_experiments": total_experiments,
            "total_experiments": total_experiments,
            "elapsed_seconds": total_elapsed,
            "best_experiment": best_summary["experiment"],
            "best_recall@50": best_summary["metrics"]["recall@50"],
            "best_ndcg@10": best_summary["metrics"]["ndcg@10"],
        },
    )
    log_event(
        "train_retrieval",
        "selected best experiment",
        experiment=best_summary["experiment"],
        recall50=f"{best_summary['metrics']['recall@50']:.4f}",
        ndcg10=f"{best_summary['metrics']['ndcg@10']:.4f}",
        elapsed=format_duration(total_elapsed),
    )
    return best_summary


def evaluate_retrieval_model(
    wv: KeyedVectors,
    split_rows: list[dict],
    track_lookup: dict[str, dict],
    popularity_values: np.ndarray,
    candidate_pool_size: int,
    progress_label: str | None = None,
) -> dict[str, float]:
    recommendation_lists: list[list[str]] = []
    artist_lookup = {uri: row.get("artist_uri") for uri, row in track_lookup.items()}
    popularity_lookup = {
        uri: percentile_rank(popularity_values, float(row.get("playlist_support", 0))) for uri, row in track_lookup.items()
    }
    vector_lookup = {uri: wv[uri] for uri in wv.key_to_index.keys()}

    metrics = defaultdict(list)
    total_rows = len(split_rows)
    progress_step = max(1, total_rows // 5) if total_rows else 1
    if progress_label:
        log_event(
            "retrieval_eval",
            "starting validation sweep",
            experiment=progress_label,
            rows=total_rows,
            candidate_pool=candidate_pool_size,
        )

    for idx, row in enumerate(split_rows, start=1):
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
        if progress_label and (idx % progress_step == 0 or idx == total_rows):
            log_event(
                "retrieval_eval",
                "validation progress",
                experiment=progress_label,
                completed=f"{idx}/{total_rows}",
            )

    summary = {name: float(np.mean(values)) if values else 0.0 for name, values in metrics.items()}
    summary["catalog_coverage@50"] = catalog_coverage(recommendation_lists, len(track_lookup))
    summary["unique_artist_coverage@50"] = unique_artist_coverage(recommendation_lists, artist_lookup)
    summary["mean_popularity_percentile"] = mean_popularity_percentile(recommendation_lists, popularity_lookup)
    summary["intra_list_diversity"] = intra_list_diversity(recommendation_lists, vector_lookup)
    if progress_label:
        log_event(
            "retrieval_eval",
            "completed validation sweep",
            experiment=progress_label,
            recall50=f"{summary['recall@50']:.4f}",
            ndcg10=f"{summary['ndcg@10']:.4f}",
        )
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
