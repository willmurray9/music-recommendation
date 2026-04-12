from __future__ import annotations

import time

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from .config import PipelineConfig
from .paths import RunLayout
from .retrieval import evaluate_retrieval_model
from .reranker import _build_metadata
from .utils import format_duration, log_event, read_json, read_jsonl, write_json


def evaluate_pipeline(config: PipelineConfig, layout: RunLayout) -> dict:
    started = time.perf_counter()
    progress_path = layout.manifests_dir / "evaluate_progress.json"

    track_df = pd.read_parquet(layout.normalized_dir / "tracks.parquet").drop_duplicates(
        subset="track_uri", keep="first"
    )
    tags_df_path = layout.normalized_dir / "track_tags.parquet"
    tags_df = pd.read_parquet(tags_df_path) if tags_df_path.exists() else pd.DataFrame()
    metadata = _build_metadata(track_df, tags_df)
    popularity_values = np.sort(track_df["playlist_support"].fillna(0).astype(float).to_numpy())
    val_rows = read_jsonl(layout.splits_dir / "val.jsonl")
    test_rows = read_jsonl(layout.splits_dir / "test.jsonl")

    results: dict[str, dict] = {}
    write_json(
        progress_path,
        {
            "status": "running",
            "run_id": layout.root.name,
            "validation_rows": len(val_rows),
            "test_rows": len(test_rows),
        },
    )
    log_event(
        "evaluate",
        "starting evaluation",
        run_id=layout.root.name,
        validation_rows=len(val_rows),
        test_rows=len(test_rows),
    )

    legacy_model_path = config.project_root / "models" / "track2vec.wordvectors"
    if legacy_model_path.exists():
        log_event("evaluate", "evaluating legacy baseline", model="models/track2vec.wordvectors")
        legacy = KeyedVectors.load(str(legacy_model_path))
        results["legacy_track2vec_val"] = evaluate_retrieval_model(
            legacy,
            val_rows,
            metadata,
            popularity_values,
            config.retrieval.candidate_pool_size,
            progress_label="legacy_track2vec_val",
        )
        results["legacy_track2vec_test"] = evaluate_retrieval_model(
            legacy,
            test_rows,
            metadata,
            popularity_values,
            config.retrieval.candidate_pool_size,
            progress_label="legacy_track2vec_test",
        )

    log_event("evaluate", "evaluating retrieval winner", model="retrieval_best.wordvectors")
    best_retrieval = KeyedVectors.load(str(layout.models_dir / "retrieval_best.wordvectors"))
    results["retrieval_val"] = evaluate_retrieval_model(
        best_retrieval,
        val_rows,
        metadata,
        popularity_values,
        config.retrieval.candidate_pool_size,
        progress_label="retrieval_val",
    )
    results["retrieval_test"] = evaluate_retrieval_model(
        best_retrieval,
        test_rows,
        metadata,
        popularity_values,
        config.retrieval.candidate_pool_size,
        progress_label="retrieval_test",
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
    write_json(
        progress_path,
        {
            "status": "completed",
            "run_id": layout.root.name,
            "elapsed_seconds": time.perf_counter() - started,
            "promotion": summary["promotion"],
        },
    )
    log_event(
        "evaluate",
        "completed evaluation",
        retrieval_recall50=summary["promotion"].get("retrieval_recall@50"),
        reranker_promoted=summary["promotion"].get("reranker_promoted"),
        elapsed=format_duration(time.perf_counter() - started),
    )
    return summary


def _promotion_summary(results: dict[str, dict]) -> dict:
    retrieval_val = results.get("retrieval_val", {})
    retrieval_test = results.get("retrieval_test", {})
    reranker = results.get("reranker", {})
    return {
        "retrieval_recall@50": retrieval_val.get("recall@50"),
        "retrieval_recall@50_val": retrieval_val.get("recall@50"),
        "retrieval_recall@50_test": retrieval_test.get("recall@50"),
        "reranker_promoted": reranker.get("promoted", False),
        "reranker_ndcg@10": reranker.get("reranker_val_metrics", {}).get("ndcg@10"),
        "reranker_ndcg@10_val": reranker.get("reranker_val_metrics", {}).get("ndcg@10"),
        "reranker_ndcg@10_test": reranker.get("reranker_test_metrics", {}).get("ndcg@10"),
        "reranker_test_promoted": reranker.get("test_promoted", False),
        "reranker_val_promoted": reranker.get("val_promoted", False),
    }
