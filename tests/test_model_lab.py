from __future__ import annotations

import tempfile
import unittest
from json import loads
from pathlib import Path

from src.recommender_v2.model_lab import build_model_lab_snapshot


class ModelLabSnapshotTests(unittest.TestCase):
    def test_snapshot_explains_reranker_tradeoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "local-v3"
            write_run_artifacts(run_root)
            output_path = run_root / "model_lab_snapshot.json"

            snapshot = build_model_lab_snapshot(run_root, output_path=output_path)

            self.assertEqual(snapshot["run"]["id"], "local-v3")
            self.assertEqual(snapshot["run"]["status"], "evaluated")
            self.assertEqual(snapshot["data"]["tracks"], 252999)
            self.assertEqual(snapshot["models"]["best_retrieval"]["experiment"], "vs256-w10-mc5-neg10")
            self.assertFalse(snapshot["promotion"]["reranker_promoted"])
            self.assertIn("coverage", snapshot["diagnostics"][0]["title"].lower())
            self.assertEqual(
                [row["model"] for row in snapshot["scorecard"]],
                ["legacy_track2vec", "retrieval", "reranker"],
            )
            self.assertEqual({row["split"] for row in snapshot["scorecard"]}, {"test"})
            self.assertEqual(
                list(snapshot["scorecard"][0]["metrics"].keys()),
                [
                    "ndcg@10",
                    "recall@50",
                    "catalog_coverage@50",
                    "unique_artist_coverage@50",
                    "mean_popularity_percentile",
                ],
            )
            self.assertEqual(snapshot["metricLabels"]["ndcg@10"], "NDCG@10")
            self.assertEqual(snapshot, loads(output_path.read_text(encoding="utf-8")))
            snapshot["metricLabels"]["ndcg@10"] = "Changed"
            self.assertEqual(build_model_lab_snapshot(run_root)["metricLabels"]["ndcg@10"], "NDCG@10")

    def test_snapshot_uses_generic_warning_for_partial_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "partial-report"
            write_run_artifacts(run_root, partial_report=True)

            snapshot = build_model_lab_snapshot(run_root)

            self.assertEqual(snapshot["run"]["status"], "evaluated")
            self.assertNotIn("coverage gate", snapshot["diagnostics"][0]["title"].lower())
            self.assertIn("not confirmed", snapshot["diagnostics"][0]["title"].lower())
            self.assertIn("does not include enough", snapshot["diagnostics"][0]["body"])

    def test_snapshot_does_not_claim_coverage_gate_for_allowed_drop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "allowed-coverage-drop"
            write_run_artifacts(run_root, allowed_coverage_drop=True)

            snapshot = build_model_lab_snapshot(run_root)
            diagnostic_text = "\n".join(
                f"{diagnostic['title']} {diagnostic['body']}" for diagnostic in snapshot["diagnostics"]
            ).lower()

            self.assertNotIn("coverage gate", diagnostic_text)
            self.assertNotIn("enough to fail promotion", diagnostic_text)
            self.assertIn("reranker promotion not confirmed", diagnostic_text)

    def test_snapshot_uses_generic_warning_when_reranker_coverage_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "missing-reranker-coverage"
            write_run_artifacts(run_root, partial_reranker_metrics=True)

            snapshot = build_model_lab_snapshot(run_root)
            diagnostic_text = "\n".join(
                f"{diagnostic['title']} {diagnostic['body']}" for diagnostic in snapshot["diagnostics"]
            ).lower()

            by_model = {row["model"]: row["metrics"] for row in snapshot["scorecard"]}
            self.assertEqual(by_model["retrieval"]["catalog_coverage@50"], 0.05501998031612773)
            self.assertEqual(by_model["reranker"]["catalog_coverage@50"], 0.0)
            self.assertIn("reranker promotion not confirmed", diagnostic_text)
            self.assertIn("test ndcg@10 by +0.0192", diagnostic_text)
            self.assertNotIn("coverage gate", diagnostic_text)
            self.assertNotIn("catalog coverage@50 by", diagnostic_text)
            self.assertNotIn("mean popularity percentile by", diagnostic_text)

    def test_snapshot_handles_missing_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "partial-run"
            write_run_artifacts(run_root, include_report=False)

            snapshot = build_model_lab_snapshot(run_root)

            self.assertEqual(snapshot["run"]["id"], "partial-run")
            self.assertEqual(snapshot["run"]["status"], "incomplete")
            self.assertEqual(snapshot["scorecard"], [])
            self.assertEqual(snapshot["diagnostics"][0]["severity"], "warning")


def write_run_artifacts(
    run_root: Path,
    include_report: bool = True,
    partial_report: bool = False,
    allowed_coverage_drop: bool = False,
    partial_reranker_metrics: bool = False,
) -> None:
    manifests = run_root / "manifests"
    metrics = run_root / "metrics"
    manifests.mkdir(parents=True)
    metrics.mkdir(parents=True)

    (manifests / "collect_spotify.json").write_text(
        """
{
  "album_count": 115697,
  "artist_count": 49124,
  "collector": "local_mpd_fallback",
  "live": false,
  "playlist_count": 19000,
  "query_count": 2078,
  "source": "local_mpd",
  "track_count": 252999
}
""".strip(),
        encoding="utf-8",
    )
    (manifests / "build_corpus.json").write_text(
        """
{
  "albums": 115697,
  "artists": 49124,
  "playlist_tracks": 1121638,
  "playlists": 18390,
  "tracks": 252999
}
""".strip(),
        encoding="utf-8",
    )
    (manifests / "split_eval.json").write_text(
        '{"train":11342,"val":1417,"test":1419}',
        encoding="utf-8",
    )
    (manifests / "train_retrieval.json").write_text(
        """
{
  "experiment": "vs256-w10-mc5-neg10",
  "params": {"vector_size": 256, "window": 10, "min_count": 5, "negative": 10},
  "metrics": {"ndcg@10": 0.02985884965621539, "recall@50": 0.10096447894613032},
  "model_path": "models/vs256-w10-mc5-neg10.wordvectors"
}
""".strip(),
        encoding="utf-8",
    )
    if not include_report:
        return

    if partial_report:
        (metrics / "report.json").write_text(
            """
{
  "run_id": "partial-report",
  "rows": [
    {"model": "retrieval", "split": "test", "metrics": {"ndcg@10": 0.030133422794937137, "recall@50": 0.09267089499647639}}
  ]
}
""".strip(),
            encoding="utf-8",
        )
        return

    if partial_reranker_metrics:
        (metrics / "report.json").write_text(
            """
{
  "run_id": "missing-reranker-coverage",
  "promotion": {
    "reranker_promoted": false
  },
  "rows": [
    {"model": "retrieval", "split": "test", "metrics": {"ndcg@10": 0.030133422794937137, "recall@50": 0.09267089499647639, "catalog_coverage@50": 0.05501998031612773}},
    {"model": "reranker", "split": "test", "metrics": {"ndcg@10": 0.04928801796691693, "recall@50": 0.1375264270613108}}
  ]
}
""".strip(),
            encoding="utf-8",
        )
        return

    reranker_catalog_coverage = "0.05300000000000000" if allowed_coverage_drop else "0.03335191048185961"
    (metrics / "report.json").write_text(
        """
{
  "run_id": "local-v3",
  "promotion": {
    "reranker_promoted": false,
    "reranker_val_promoted": false,
    "reranker_test_promoted": false,
    "retrieval_recall@50_test": 0.09267089499647639,
    "reranker_ndcg@10_test": 0.04928801796691693
  },
  "rows": [
    {"model": "reranker", "split": "val", "metrics": {"ndcg@10": 0.0703140046839557, "recall@50": 0.1503171247357294, "catalog_coverage@50": 0.04336218449683741, "unique_artist_coverage@50": 0.06292828565592181, "mean_popularity_percentile": 0.9841872541437601}},
    {"model": "legacy_track2vec", "split": "test", "metrics": {"ndcg@10": 0.04530150947119676, "recall@50": 0.11381254404510219, "catalog_coverage@50": 0.11172771433879185, "unique_artist_coverage@50": 0.17783568113345818, "mean_popularity_percentile": 0.8281706288859381}},
    {"model": "retrieval", "split": "test", "metrics": {"ndcg@10": 0.030133422794937137, "recall@50": 0.09267089499647639, "catalog_coverage@50": 0.05501998031612773, "unique_artist_coverage@50": 0.082403713052683, "mean_popularity_percentile": 0.9593971359891325}},
    {"model": "reranker", "split": "test", "metrics": {"ndcg@10": 0.04928801796691693, "recall@50": 0.1375264270613108, "catalog_coverage@50": 0.03335191048185961, "unique_artist_coverage@50": 0.05473902776646853, "mean_popularity_percentile": 0.9888624379527422}}
  ],
  "deltas": {
    "test": {
      "reranker_vs_retrieval": {"ndcg@10": 0.019154595171979795, "recall@50": 0.044855532064834405, "catalog_coverage@50": -0.021668069834268118, "unique_artist_coverage@50": -0.02766468528621447, "mean_popularity_percentile": 0.029465301963609747}
    }
  }
}
""".strip().replace("0.03335191048185961", reranker_catalog_coverage),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
