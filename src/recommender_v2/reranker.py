from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import time

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.ensemble import HistGradientBoostingClassifier

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
from .retrieval import evaluate_retrieval_model, retrieve_candidates
from .utils import (
    cosine_similarity,
    format_duration,
    log_event,
    percentile_rank,
    read_json,
    read_jsonl,
    write_json,
)


FEATURE_NAMES = [
    "mean_seed_cosine",
    "max_seed_cosine",
    "seed_neighbor_hits",
    "same_artist",
    "genre_overlap",
    "tag_overlap",
    "release_year_distance",
    "duration_delta",
    "popularity_percentile",
    "playlist_support",
]


@dataclass(frozen=True)
class CandidateRow:
    track_uri: str
    features: list[float]
    label: int


def train_reranker(config: PipelineConfig, layout: RunLayout) -> dict:
    total_started = time.perf_counter()
    progress_path = layout.manifests_dir / "train_reranker_progress.json"

    track_df = pd.read_parquet(layout.normalized_dir / "tracks.parquet").drop_duplicates(
        subset="track_uri", keep="first"
    )
    tags_df = pd.read_parquet(layout.normalized_dir / "track_tags.parquet")
    best_retrieval = read_json(layout.manifests_dir / "train_retrieval.json")
    wv = KeyedVectors.load(str(layout.root / best_retrieval["model_path"]))
    metadata = _build_metadata(track_df, tags_df)

    train_rows = read_jsonl(layout.splits_dir / "train.jsonl")
    val_rows = read_jsonl(layout.splits_dir / "val.jsonl")
    test_rows = read_jsonl(layout.splits_dir / "test.jsonl")
    popularity_values = np.sort(track_df["playlist_support"].fillna(0).astype(float).to_numpy())
    write_json(
        progress_path,
        {
            "status": "building_train_examples",
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "best_retrieval_experiment": best_retrieval["experiment"],
        },
    )
    log_event(
        "train_reranker",
        "building reranker training examples",
        train_rows=len(train_rows),
        val_rows=len(val_rows),
        test_rows=len(test_rows),
        candidate_pool=config.retrieval.candidate_pool_size,
        seed_neighbor_probe=config.reranker.max_seed_neighbor_probe,
    )

    example_started = time.perf_counter()
    train_examples = _build_candidate_rows(
        wv,
        train_rows,
        metadata,
        popularity_values,
        config.retrieval.candidate_pool_size,
        config.reranker.max_seed_neighbor_probe,
        progress_label="train",
    )
    X_train = np.array([row.features for row in train_examples], dtype=np.float32)
    y_train = np.array([row.label for row in train_examples], dtype=np.int8)
    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Reranker training data is degenerate; no positive/negative separation")
    positive_rows = int(y_train.sum())
    write_json(
        progress_path,
        {
            "status": "fitting_model",
            "training_rows": int(len(train_examples)),
            "positive_rows": positive_rows,
            "negative_rows": int(len(train_examples) - positive_rows),
            "example_build_seconds": time.perf_counter() - example_started,
        },
    )
    log_event(
        "train_reranker",
        "built reranker training examples",
        examples=len(train_examples),
        positive_rows=positive_rows,
        negative_rows=int(len(train_examples) - positive_rows),
        duration=format_duration(time.perf_counter() - example_started),
    )

    model = HistGradientBoostingClassifier(
        max_iter=config.reranker.max_iter,
        max_depth=config.reranker.max_depth,
        learning_rate=config.reranker.learning_rate,
        l2_regularization=config.reranker.l2_regularization,
        random_state=config.reranker.random_seed,
    )
    fit_started = time.perf_counter()
    log_event(
        "train_reranker",
        "fitting reranker model",
        max_iter=config.reranker.max_iter,
        max_depth=config.reranker.max_depth,
    )
    model.fit(X_train, y_train)
    write_json(
        progress_path,
        {
            "status": "evaluating_validation",
            "training_rows": int(len(train_examples)),
            "positive_rows": positive_rows,
            "fit_seconds": time.perf_counter() - fit_started,
        },
    )
    log_event(
        "train_reranker",
        "fitted reranker model",
        duration=format_duration(time.perf_counter() - fit_started),
    )

    retrieval_val_metrics = best_retrieval["metrics"]
    retrieval_test_metrics = evaluate_retrieval_model(
        wv,
        test_rows,
        metadata,
        popularity_values,
        config.retrieval.candidate_pool_size,
        progress_label="retrieval_test_baseline",
    )
    rerank_val_metrics = evaluate_reranker(
        model,
        wv,
        val_rows,
        metadata,
        popularity_values,
        config.retrieval.candidate_pool_size,
        config.reranker.max_seed_neighbor_probe,
        progress_label="val",
    )
    rerank_test_metrics = evaluate_reranker(
        model,
        wv,
        test_rows,
        metadata,
        popularity_values,
        config.retrieval.candidate_pool_size,
        config.reranker.max_seed_neighbor_probe,
        progress_label="test",
    )
    val_improvement = (
        (rerank_val_metrics["ndcg@10"] - retrieval_val_metrics["ndcg@10"]) / max(retrieval_val_metrics["ndcg@10"], 1e-9)
    )
    test_improvement = (
        (rerank_test_metrics["ndcg@10"] - retrieval_test_metrics["ndcg@10"])
        / max(retrieval_test_metrics["ndcg@10"], 1e-9)
    )
    val_coverage_ok = rerank_val_metrics["catalog_coverage@50"] >= retrieval_val_metrics["catalog_coverage@50"] * 0.95
    test_coverage_ok = rerank_test_metrics["catalog_coverage@50"] >= retrieval_test_metrics["catalog_coverage@50"] * 0.95
    val_diversity_ok = (
        rerank_val_metrics["intra_list_diversity"] >= retrieval_val_metrics["intra_list_diversity"] * 0.95
    )
    test_diversity_ok = (
        rerank_test_metrics["intra_list_diversity"] >= retrieval_test_metrics["intra_list_diversity"] * 0.95
    )
    val_promoted = val_improvement >= 0.10 and val_coverage_ok and val_diversity_ok
    test_promoted = test_improvement >= 0.10 and test_coverage_ok and test_diversity_ok
    promoted = val_promoted and test_promoted

    export_payload = export_hist_gradient_boosting(model)
    write_json(layout.models_dir / "reranker_model.json", export_payload)

    summary = {
        "feature_names": FEATURE_NAMES,
        "training_rows": int(len(train_examples)),
        "promoted": promoted,
        "val_promoted": val_promoted,
        "test_promoted": test_promoted,
        "retrieval_val_metrics": retrieval_val_metrics,
        "retrieval_test_metrics": retrieval_test_metrics,
        "reranker_val_metrics": rerank_val_metrics,
        "reranker_test_metrics": rerank_test_metrics,
        "ndcg10_improvement_ratio": val_improvement,
        "ndcg10_improvement_ratio_val": val_improvement,
        "ndcg10_improvement_ratio_test": test_improvement,
        "coverage_ok": val_coverage_ok and test_coverage_ok,
        "coverage_ok_val": val_coverage_ok,
        "coverage_ok_test": test_coverage_ok,
        "diversity_ok": val_diversity_ok and test_diversity_ok,
        "diversity_ok_val": val_diversity_ok,
        "diversity_ok_test": test_diversity_ok,
        "model_path": str((layout.models_dir / "reranker_model.json").relative_to(layout.root)),
    }
    write_json(layout.manifests_dir / "train_reranker.json", summary)
    write_json(
        progress_path,
        {
            "status": "completed",
            "training_rows": int(len(train_examples)),
            "positive_rows": positive_rows,
            "promoted": promoted,
            "val_promoted": val_promoted,
            "test_promoted": test_promoted,
            "ndcg10_improvement_ratio_val": val_improvement,
            "ndcg10_improvement_ratio_test": test_improvement,
            "coverage_ok_val": val_coverage_ok,
            "coverage_ok_test": test_coverage_ok,
            "diversity_ok_val": val_diversity_ok,
            "diversity_ok_test": test_diversity_ok,
            "elapsed_seconds": time.perf_counter() - total_started,
        },
    )
    log_event(
        "train_reranker",
        "completed reranker training",
        promoted=promoted,
        val_promoted=val_promoted,
        test_promoted=test_promoted,
        ndcg10_delta_val=f"{val_improvement:.4f}",
        ndcg10_delta_test=f"{test_improvement:.4f}",
        coverage_ok=val_coverage_ok and test_coverage_ok,
        diversity_ok=val_diversity_ok and test_diversity_ok,
        elapsed=format_duration(time.perf_counter() - total_started),
    )
    return summary


def evaluate_reranker(
    model: HistGradientBoostingClassifier,
    wv: KeyedVectors,
    split_rows: list[dict],
    metadata: dict[str, dict],
    popularity_values: np.ndarray,
    candidate_pool_size: int,
    seed_neighbor_probe: int,
    progress_label: str | None = None,
) -> dict[str, float]:
    recommendation_lists: list[list[str]] = []
    artist_lookup = {uri: row.get("artist_uri") for uri, row in metadata.items()}
    popularity_lookup = {
        uri: percentile_rank(popularity_values, float(row.get("playlist_support", 0))) for uri, row in metadata.items()
    }
    vector_lookup = {uri: wv[uri] for uri in wv.key_to_index.keys()}
    metrics = defaultdict(list)
    total_rows = len(split_rows)
    progress_step = max(1, total_rows // 5) if total_rows else 1
    if progress_label:
        log_event(
            "reranker_eval",
            "starting reranker evaluation",
            split=progress_label,
            rows=total_rows,
            candidate_pool=candidate_pool_size,
        )

    for idx, row in enumerate(split_rows, start=1):
        reranked = rerank_candidates(
            model,
            wv,
            row["seed_tracks"],
            metadata,
            popularity_values,
            topn=50,
            candidate_pool_size=candidate_pool_size,
            seed_neighbor_probe=seed_neighbor_probe,
        )
        positives = set(row["positive_tracks"])
        recommendation_lists.append(reranked)
        metrics["recall@10"].append(recall_at_k(reranked, positives, 10))
        metrics["recall@50"].append(recall_at_k(reranked, positives, 50))
        metrics["ndcg@10"].append(ndcg_at_k(reranked, positives, 10))
        metrics["ndcg@50"].append(ndcg_at_k(reranked, positives, 50))
        metrics["mrr@10"].append(mrr_at_k(reranked, positives, 10))
        metrics["same_artist_rate"].append(same_artist_rate(reranked, artist_lookup))
        if progress_label and (idx % progress_step == 0 or idx == total_rows):
            log_event(
                "reranker_eval",
                "evaluation progress",
                split=progress_label,
                completed=f"{idx}/{total_rows}",
            )

    summary = {name: float(np.mean(values)) if values else 0.0 for name, values in metrics.items()}
    summary["catalog_coverage@50"] = catalog_coverage(recommendation_lists, len(metadata))
    summary["unique_artist_coverage@50"] = unique_artist_coverage(recommendation_lists, artist_lookup)
    summary["mean_popularity_percentile"] = mean_popularity_percentile(recommendation_lists, popularity_lookup)
    summary["intra_list_diversity"] = intra_list_diversity(recommendation_lists, vector_lookup)
    if progress_label:
        log_event(
            "reranker_eval",
            "completed reranker evaluation",
            split=progress_label,
            recall50=f"{summary['recall@50']:.4f}",
            ndcg10=f"{summary['ndcg@10']:.4f}",
        )
    return summary


def rerank_candidates(
    model: HistGradientBoostingClassifier,
    wv: KeyedVectors,
    seed_tracks: list[str],
    metadata: dict[str, dict],
    popularity_values: np.ndarray,
    topn: int,
    candidate_pool_size: int,
    seed_neighbor_probe: int,
) -> list[str]:
    candidates = retrieve_candidates(wv, seed_tracks, topn=candidate_pool_size)
    candidate_rows = _candidate_features(
        wv,
        seed_tracks,
        candidates,
        metadata,
        popularity_values,
        seed_neighbor_probe,
    )
    if not candidate_rows:
        return []
    X = np.array([row.features for row in candidate_rows], dtype=np.float32)
    scores = model.decision_function(X)
    base_scores = {row.track_uri: float(score) for row, score in zip(candidate_rows, scores)}
    ordered = sorted(candidate_rows, key=lambda row: base_scores[row.track_uri], reverse=True)
    return _mmr_select(ordered, base_scores, metadata, wv, seed_tracks, topn)


def _build_candidate_rows(
    wv: KeyedVectors,
    split_rows: list[dict],
    metadata: dict[str, dict],
    popularity_values: np.ndarray,
    candidate_pool_size: int,
    seed_neighbor_probe: int,
    progress_label: str | None = None,
) -> list[CandidateRow]:
    examples: list[CandidateRow] = []
    total_rows = len(split_rows)
    progress_step = max(1, total_rows // 10) if total_rows else 1
    if progress_label:
        log_event(
            "reranker_data",
            "starting candidate generation",
            split=progress_label,
            rows=total_rows,
            candidate_pool=candidate_pool_size,
        )

    for idx, row in enumerate(split_rows, start=1):
        candidates = retrieve_candidates(wv, row["seed_tracks"], topn=candidate_pool_size)
        candidate_rows = _candidate_features(
            wv,
            row["seed_tracks"],
            candidates,
            metadata,
            popularity_values,
            seed_neighbor_probe,
            positives=set(row["positive_tracks"]),
        )
        examples.extend(candidate_rows)
        if progress_label and (idx % progress_step == 0 or idx == total_rows):
            log_event(
                "reranker_data",
                "candidate generation progress",
                split=progress_label,
                completed=f"{idx}/{total_rows}",
                accumulated_examples=len(examples),
            )
    return examples


def _candidate_features(
    wv: KeyedVectors,
    seed_tracks: list[str],
    candidates: list[str],
    metadata: dict[str, dict],
    popularity_values: np.ndarray,
    seed_neighbor_probe: int,
    positives: set[str] | None = None,
) -> list[CandidateRow]:
    positives = positives or set()
    valid_seeds = [seed for seed in seed_tracks if seed in wv]
    if not valid_seeds:
        return []
    seed_vectors = [wv[seed] for seed in valid_seeds]
    seed_neighbor_sets = []
    for seed in valid_seeds:
        neighbors = wv.most_similar(seed, topn=seed_neighbor_probe)
        seed_neighbor_sets.append({uri for uri, _ in neighbors})

    seed_years = [metadata[seed].get("release_year") for seed in valid_seeds if metadata.get(seed, {}).get("release_year")]
    seed_duration = np.mean(
        [metadata[seed].get("duration_ms", 0) for seed in valid_seeds if metadata.get(seed, {}).get("duration_ms")]
    )
    seed_genres = set().union(*(metadata.get(seed, {}).get("genres", []) for seed in valid_seeds))
    seed_tags = set().union(*(metadata.get(seed, {}).get("tags", []) for seed in valid_seeds))
    seed_artists = {metadata.get(seed, {}).get("artist_uri") for seed in valid_seeds}

    rows: list[CandidateRow] = []
    for candidate in candidates:
        if candidate not in wv or candidate not in metadata:
            continue
        candidate_vector = wv[candidate]
        seed_cosines = [cosine_similarity(candidate_vector, seed_vector) for seed_vector in seed_vectors]
        candidate_meta = metadata[candidate]
        release_year = candidate_meta.get("release_year")
        year_distance = abs(release_year - float(np.mean(seed_years))) if release_year and seed_years else 99.0
        duration_delta = abs(candidate_meta.get("duration_ms", 0) - seed_duration) if seed_duration else 0.0
        row = CandidateRow(
            track_uri=candidate,
            features=[
                float(np.mean(seed_cosines)),
                float(np.max(seed_cosines)),
                float(sum(candidate in seed_neighbors for seed_neighbors in seed_neighbor_sets)),
                1.0 if candidate_meta.get("artist_uri") in seed_artists else 0.0,
                float(len(seed_genres & set(candidate_meta.get("genres", [])))),
                float(len(seed_tags & set(candidate_meta.get("tags", [])))),
                float(year_distance),
                float(duration_delta),
                percentile_rank(popularity_values, float(candidate_meta.get("playlist_support", 0))),
                float(candidate_meta.get("playlist_support", 0)),
            ],
            label=1 if candidate in positives else 0,
        )
        rows.append(row)
    return rows


def _build_metadata(track_df: pd.DataFrame, tags_df: pd.DataFrame) -> dict[str, dict]:
    tags_map = (
        tags_df.groupby("track_uri")["tag"].apply(list).to_dict()
        if not tags_df.empty
        else {}
    )
    metadata = track_df.set_index("track_uri").to_dict("index")
    for track_uri, row in metadata.items():
        row["genres"] = _normalize_list(row.get("genres"))
        row["tags"] = tags_map.get(track_uri, _normalize_list(row.get("tags")))
    return metadata


def _normalize_list(value: object) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return [str(item) for item in value.tolist()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def export_hist_gradient_boosting(model: HistGradientBoostingClassifier) -> dict:
    trees = []
    for predictors in model._predictors:
        tree = predictors[0]
        nodes = []
        for node in tree.nodes:
            nodes.append(
                {
                    "value": float(node["value"]),
                    "feature_idx": int(node["feature_idx"]),
                    "num_threshold": float(node["num_threshold"]),
                    "left": int(node["left"]),
                    "right": int(node["right"]),
                    "is_leaf": bool(node["is_leaf"]),
                }
            )
        trees.append({"nodes": nodes})
    return {
        "feature_names": FEATURE_NAMES,
        "baseline": float(model._baseline_prediction[0][0]),
        "trees": trees,
    }


def _mmr_select(
    ordered_rows: list[CandidateRow],
    base_scores: dict[str, float],
    metadata: dict[str, dict],
    wv: KeyedVectors,
    seed_tracks: list[str],
    topn: int,
) -> list[str]:
    seed_artists = {metadata.get(seed, {}).get("artist_uri") for seed in seed_tracks}
    selected: list[str] = []
    selected_artists: set[str] = set()
    lambda_weight = 0.75

    while ordered_rows and len(selected) < topn:
        best_candidate = None
        best_score = -math.inf
        for row in ordered_rows:
            artist_uri = metadata.get(row.track_uri, {}).get("artist_uri")
            if artist_uri in seed_artists and artist_uri in selected_artists:
                continue
            redundancy = 0.0
            if selected:
                redundancy = max(
                    cosine_similarity(wv[row.track_uri], wv[chosen])
                    for chosen in selected
                    if chosen in wv
                )
            mmr_score = lambda_weight * base_scores[row.track_uri] - (1 - lambda_weight) * redundancy
            if best_candidate is None or mmr_score > best_score:
                best_candidate = row
                best_score = mmr_score
        if best_candidate is None:
            break
        selected.append(best_candidate.track_uri)
        artist_uri = metadata.get(best_candidate.track_uri, {}).get("artist_uri")
        if artist_uri:
            selected_artists.add(artist_uri)
        ordered_rows = [row for row in ordered_rows if row.track_uri != best_candidate.track_uri]
    return selected
