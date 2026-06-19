from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import read_json, write_json


METRIC_LABELS = {
    "ndcg@10": "NDCG@10",
    "recall@50": "Recall@50",
    "catalog_coverage@50": "Catalog coverage@50",
    "unique_artist_coverage@50": "Artist coverage@50",
    "mean_popularity_percentile": "Mean popularity percentile",
}

COVERAGE_GATE_MIN_RATIO = 0.95


def build_model_lab_snapshot(run_root: Path, output_path: Path | None = None) -> dict[str, Any]:
    manifests_dir = run_root / "manifests"
    metrics_dir = run_root / "metrics"

    collect = _read_optional_json(manifests_dir / "collect_spotify.json")
    corpus = _read_optional_json(manifests_dir / "build_corpus.json")
    split = _read_optional_json(manifests_dir / "split_eval.json")
    retrieval = _read_optional_json(manifests_dir / "train_retrieval.json")
    report = _read_optional_json(metrics_dir / "report.json")

    scorecard = _scorecard_rows(report)
    diagnostics = _diagnostics(report, scorecard)
    snapshot = {
        "schemaVersion": 1,
        "run": {
            "id": run_root.name,
            "status": "evaluated" if report else "incomplete",
            "source": collect.get("source") or collect.get("collector") or "unknown",
            "collector": collect.get("collector", "unknown"),
            "live": bool(collect.get("live", False)),
        },
        "data": {
            "playlists": int(corpus.get("playlists", collect.get("playlist_count", 0)) or 0),
            "tracks": int(corpus.get("tracks", collect.get("track_count", 0)) or 0),
            "artists": int(corpus.get("artists", collect.get("artist_count", 0)) or 0),
            "playlistTracks": int(corpus.get("playlist_tracks", 0) or 0),
            "splits": {
                "train": int(split.get("train", 0) or 0),
                "val": int(split.get("val", 0) or 0),
                "test": int(split.get("test", 0) or 0),
            },
        },
        "models": {
            "best_retrieval": {
                "experiment": retrieval.get("experiment"),
                "params": retrieval.get("params", {}),
                "metrics": retrieval.get("metrics", {}),
                "modelPath": retrieval.get("model_path"),
            }
        },
        "promotion": report.get("promotion", {}),
        "scorecard": scorecard,
        "diagnostics": diagnostics,
        "metricLabels": dict(METRIC_LABELS),
    }

    if output_path is not None:
        write_json(output_path, snapshot)
    return snapshot


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def _scorecard_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in report.get("rows", []):
        if row.get("split") != "test":
            continue
        metrics = row.get("metrics", {})
        rows.append(
            {
                "model": row.get("model", "unknown"),
                "split": "test",
                "metrics": {
                    key: float(metrics.get(key, 0.0) or 0.0)
                    for key in METRIC_LABELS
                },
            }
        )
    model_order = {"legacy_track2vec": 0, "retrieval": 1, "reranker": 2}
    rows.sort(key=lambda item: model_order.get(item["model"], 99))
    return rows


def _diagnostics(report: dict[str, Any], scorecard: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not report:
        return [
            {
                "severity": "warning",
                "title": "No evaluation report yet",
                "body": "Run `make report-model RUN_ID=<run-id>` after evaluation, then generate a model lab snapshot.",
            }
        ]

    raw_by_model = _raw_test_metrics_by_model(report)
    retrieval = raw_by_model.get("retrieval", {})
    reranker = raw_by_model.get("reranker", {})
    promotion = report.get("promotion", {})
    reranker_promoted = promotion.get("reranker_promoted")
    diagnostics: list[dict[str, str]] = []
    coverage_gate_warning_added = False

    if retrieval and reranker:
        has_ndcg_metrics = _has_raw_metrics(retrieval, reranker, "ndcg@10")
        has_coverage_metrics = "catalog_coverage@50" in retrieval and "catalog_coverage@50" in reranker
        ndcg_delta = _metric_delta(retrieval, reranker, "ndcg@10") if has_ndcg_metrics else 0.0
        metrics_coverage_failed = (
            has_coverage_metrics
            and float(reranker["catalog_coverage@50"] or 0.0)
            < float(retrieval["catalog_coverage@50"] or 0.0) * COVERAGE_GATE_MIN_RATIO
        )
        has_coverage_gate_evidence = has_ndcg_metrics and ndcg_delta > 0 and metrics_coverage_failed
        if reranker_promoted is False and has_coverage_gate_evidence:
            diagnostics.append(
                {
                    "severity": "warning",
                    "title": "Coverage gate blocked reranker promotion",
                    "body": "The reranker improved ranking metrics but reduced catalog coverage below the promotion threshold.",
                }
            )
            coverage_gate_warning_added = True
        tradeoff_body = _tradeoff_body(retrieval, reranker)
        if tradeoff_body:
            diagnostics.append(
                {
                    "severity": "info",
                    "title": "Reranker tradeoff",
                    "body": tradeoff_body,
                }
            )

    if reranker_promoted is not True and not coverage_gate_warning_added:
        diagnostics.append(
            {
                "severity": "warning",
                "title": "Reranker promotion not confirmed",
                "body": "The evaluation report does not include enough promotion or scorecard detail to explain why reranker promotion did not succeed.",
            }
        )

    diagnostics.append(
        {
            "severity": "info",
            "title": "Best next experiment",
            "body": "Tune reranker training or scoring to preserve coverage while keeping the ranking lift.",
        }
    )
    return diagnostics


def _raw_test_metrics_by_model(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in report.get("rows", []):
        if row.get("split") != "test":
            continue
        rows[row.get("model", "unknown")] = dict(row.get("metrics", {}))
    return rows


def _has_raw_metrics(retrieval: dict[str, Any], reranker: dict[str, Any], key: str) -> bool:
    return key in retrieval and key in reranker


def _metric_delta(retrieval: dict[str, Any], reranker: dict[str, Any], key: str) -> float:
    return float(reranker.get(key, 0.0) or 0.0) - float(retrieval.get(key, 0.0) or 0.0)


def _tradeoff_body(retrieval: dict[str, Any], reranker: dict[str, Any]) -> str | None:
    clauses: list[str] = []
    if _has_raw_metrics(retrieval, reranker, "ndcg@10"):
        clauses.append(f"test NDCG@10 by {_metric_delta(retrieval, reranker, 'ndcg@10'):+.4f}")
    if _has_raw_metrics(retrieval, reranker, "catalog_coverage@50"):
        clauses.append(
            f"catalog coverage@50 by {_metric_delta(retrieval, reranker, 'catalog_coverage@50'):+.4f}"
        )
    if _has_raw_metrics(retrieval, reranker, "mean_popularity_percentile"):
        clauses.append(
            "mean popularity percentile by "
            f"{_metric_delta(retrieval, reranker, 'mean_popularity_percentile'):+.4f}"
        )
    if not clauses:
        return None
    return f"Compared with retrieval, reranker changed {', '.join(clauses)}."
