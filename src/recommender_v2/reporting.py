from __future__ import annotations

from .paths import RunLayout
from .utils import log_event, read_json, write_json


SPLITS = ("val", "test")
METRIC_KEYS = ("ndcg@10", "recall@50", "catalog_coverage@50", "unique_artist_coverage@50", "mean_popularity_percentile")


def build_run_report(layout: RunLayout) -> dict:
    evaluation_path = layout.metrics_dir / "evaluation.json"
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Missing evaluation artifact: {evaluation_path}")

    evaluation = read_json(evaluation_path)
    results = evaluation.get("results", {})
    rows = _report_rows(results)
    winners = _split_winners(rows)
    deltas = _deltas(rows)
    report = {
        "run_id": layout.root.name,
        "promotion": evaluation.get("promotion", {}),
        "rows": rows,
        "winners": winners,
        "deltas": deltas,
    }

    report_path = layout.metrics_dir / "report.json"
    write_json(report_path, report)
    text = render_run_report(report)
    (layout.metrics_dir / "report.txt").write_text(text, encoding="utf-8")
    log_event(
        "report",
        "wrote run report",
        run_id=layout.root.name,
        val_winner=winners.get("val", {}).get("winner"),
        test_winner=winners.get("test", {}).get("winner"),
    )
    print(text, flush=True)
    return report


def render_run_report(report: dict) -> str:
    lines = [f"Run report: {report['run_id']}", ""]
    for split in SPLITS:
        winner = report["winners"].get(split)
        if winner:
            lines.append(
                f"{split.upper()} winner: {winner['winner']} "
                f"(ndcg@10={winner['metrics']['ndcg@10']:.4f}, recall@50={winner['metrics']['recall@50']:.4f})"
            )
        lines.append(f"{split.upper()} metrics:")
        lines.append("model                ndcg@10  recall@50  coverage@50  artist_cov  pop_pct")
        for row in report["rows"]:
            if row["split"] != split:
                continue
            metrics = row["metrics"]
            lines.append(
                f"{row['model']:<20} "
                f"{metrics.get('ndcg@10', 0.0):>7.4f}  "
                f"{metrics.get('recall@50', 0.0):>9.4f}  "
                f"{metrics.get('catalog_coverage@50', 0.0):>11.4f}  "
                f"{metrics.get('unique_artist_coverage@50', 0.0):>10.4f}  "
                f"{metrics.get('mean_popularity_percentile', 0.0):>7.4f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _report_rows(results: dict) -> list[dict]:
    rows: list[dict] = []
    for split in SPLITS:
        legacy_key = f"legacy_track2vec_{split}"
        if legacy_key in results:
            rows.append({"model": "legacy_track2vec", "split": split, "metrics": results[legacy_key]})

        retrieval_key = f"retrieval_{split}"
        if retrieval_key in results:
            rows.append({"model": "retrieval", "split": split, "metrics": results[retrieval_key]})

        reranker = results.get("reranker")
        if reranker:
            reranker_key = f"reranker_{split}_metrics"
            if reranker_key in reranker:
                rows.append({"model": "reranker", "split": split, "metrics": reranker[reranker_key]})
    return rows


def _split_winners(rows: list[dict]) -> dict[str, dict]:
    winners: dict[str, dict] = {}
    for split in SPLITS:
        candidates = [row for row in rows if row["split"] == split]
        if not candidates:
            continue
        winner = max(
            candidates,
            key=lambda row: (
                row["metrics"].get("ndcg@10", 0.0),
                row["metrics"].get("recall@50", 0.0),
            ),
        )
        winners[split] = {
            "winner": winner["model"],
            "metrics": winner["metrics"],
        }
    return winners


def _deltas(rows: list[dict]) -> dict[str, dict]:
    deltas: dict[str, dict] = {}
    for split in SPLITS:
        split_rows = {row["model"]: row["metrics"] for row in rows if row["split"] == split}
        retrieval = split_rows.get("retrieval")
        legacy = split_rows.get("legacy_track2vec")
        reranker = split_rows.get("reranker")
        split_delta: dict[str, dict] = {}
        if retrieval and legacy:
            split_delta["retrieval_vs_legacy"] = _metric_delta(retrieval, legacy)
        if reranker and retrieval:
            split_delta["reranker_vs_retrieval"] = _metric_delta(reranker, retrieval)
        if reranker and legacy:
            split_delta["reranker_vs_legacy"] = _metric_delta(reranker, legacy)
        deltas[split] = split_delta
    return deltas


def _metric_delta(left: dict, right: dict) -> dict[str, float]:
    delta = {}
    for key in METRIC_KEYS:
        delta[key] = float(left.get(key, 0.0) - right.get(key, 0.0))
    return delta
