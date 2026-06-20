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

NDCG_IMPROVEMENT_MIN_RATIO = 0.10


def build_model_lab_snapshot(run_root: Path, output_path: Path | None = None) -> dict[str, Any]:
    manifests_dir = run_root / "manifests"
    metrics_dir = run_root / "metrics"

    collect = _read_optional_json(manifests_dir / "collect_spotify.json")
    corpus = _read_optional_json(manifests_dir / "build_corpus.json")
    split = _read_optional_json(manifests_dir / "split_eval.json")
    retrieval = _read_optional_json(manifests_dir / "train_retrieval.json")
    reranker = _read_optional_json(manifests_dir / "train_reranker.json")
    report = _read_optional_json(metrics_dir / "report.json")

    scorecard = _scorecard_rows(report)
    diagnostics = _diagnostics(report, reranker)
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
                    key: _optional_float(metrics.get(key)) if key in metrics else None
                    for key in METRIC_LABELS
                },
            }
        )
    model_order = {"legacy_track2vec": 0, "retrieval": 1, "reranker": 2}
    rows.sort(key=lambda item: model_order.get(item["model"], 99))
    return rows


def _diagnostics(
    report: dict[str, Any],
    reranker_manifest: dict[str, Any],
) -> list[dict[str, str]]:
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
    reranker_promoted = promotion.get("reranker_promoted", reranker_manifest.get("promoted"))
    diagnostics: list[dict[str, str]] = []
    coverage_gate_warning_added = False

    if retrieval and reranker:
        if reranker_promoted is False and _explicit_coverage_gate_blocked(reranker_manifest):
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
    return _optional_float(retrieval.get(key)) is not None and _optional_float(reranker.get(key)) is not None


def _metric_delta(retrieval: dict[str, Any], reranker: dict[str, Any], key: str) -> float:
    reranker_value = _optional_float(reranker.get(key))
    retrieval_value = _optional_float(retrieval.get(key))
    if reranker_value is None or retrieval_value is None:
        raise ValueError(f"Cannot compute delta for missing metric {key}")
    return reranker_value - retrieval_value


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


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _explicit_coverage_gate_blocked(reranker_manifest: dict[str, Any]) -> bool:
    if not reranker_manifest:
        return False

    coverage_flags = _gate_flags(
        reranker_manifest,
        ("coverage_ok", "coverage_ok_val", "coverage_ok_test"),
    )
    diversity_flags = _gate_flags(
        reranker_manifest,
        ("diversity_ok", "diversity_ok_val", "diversity_ok_test"),
    )
    improvement_ratios = [
        _optional_float(reranker_manifest.get(key))
        for key in (
            "ndcg10_improvement_ratio",
            "ndcg10_improvement_ratio_val",
            "ndcg10_improvement_ratio_test",
        )
    ]

    if len(coverage_flags) != 3 or len(diversity_flags) != 3 or any(value is None for value in improvement_ratios):
        return False
    return (
        not all(coverage_flags)
        and all(diversity_flags)
        and all(value >= NDCG_IMPROVEMENT_MIN_RATIO for value in improvement_ratios if value is not None)
    )


def _gate_flags(reranker_manifest: dict[str, Any], keys: tuple[str, ...]) -> list[bool]:
    flags: list[bool] = []
    for key in keys:
        value = reranker_manifest.get(key)
        if isinstance(value, bool):
            flags.append(value)
    return flags
