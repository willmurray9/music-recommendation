from __future__ import annotations

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from .config import PipelineConfig
from .paths import RunLayout
from .retrieval import evaluate_retrieval_model, retrieve_candidates
from .reranker import evaluate_reranker, rerank_candidates, _build_metadata
from .utils import read_json, read_jsonl, write_json


def evaluate_pipeline(config: PipelineConfig, layout: RunLayout) -> dict:
    track_df = pd.read_parquet(layout.normalized_dir / "tracks.parquet")
    tags_df_path = layout.normalized_dir / "track_tags.parquet"
    tags_df = pd.read_parquet(tags_df_path) if tags_df_path.exists() else pd.DataFrame()
    metadata = _build_metadata(track_df, tags_df)
    popularity_values = np.sort(track_df["playlist_support"].fillna(0).astype(float).to_numpy())
    val_rows = read_jsonl(layout.splits_dir / "val.jsonl")
    test_rows = read_jsonl(layout.splits_dir / "test.jsonl")

    results: dict[str, dict] = {}

    legacy_model_path = config.project_root / "models" / "track2vec.wordvectors"
    if legacy_model_path.exists():
        legacy = KeyedVectors.load(str(legacy_model_path))
        results["legacy_track2vec_val"] = evaluate_retrieval_model(
            legacy,
            val_rows,
            metadata,
            popularity_values,
            config.retrieval.candidate_pool_size,
        )

    best_retrieval = KeyedVectors.load(str(layout.models_dir / "retrieval_best.wordvectors"))
    results["retrieval_val"] = evaluate_retrieval_model(
        best_retrieval,
        val_rows,
        metadata,
        popularity_values,
        config.retrieval.candidate_pool_size,
    )
    results["retrieval_test"] = evaluate_retrieval_model(
        best_retrieval,
        test_rows,
        metadata,
        popularity_values,
        config.retrieval.candidate_pool_size,
    )

    reranker_manifest_path = layout.manifests_dir / "train_reranker.json"
    if reranker_manifest_path.exists():
        reranker_manifest = read_json(reranker_manifest_path)
        results["reranker"] = reranker_manifest

    summary = {
        "results": results,
        "promotion": _promotion_summary(results),
    }
    write_json(layout.metrics_dir / "evaluation.json", summary)
    write_json(layout.manifests_dir / "evaluate.json", summary)
    return summary


def _promotion_summary(results: dict[str, dict]) -> dict:
    retrieval_val = results.get("retrieval_val", {})
    reranker = results.get("reranker", {})
    return {
        "retrieval_recall@50": retrieval_val.get("recall@50"),
        "reranker_promoted": reranker.get("promoted", False),
        "reranker_ndcg@10": reranker.get("reranker_val_metrics", {}).get("ndcg@10"),
    }
