from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.recommender_v2.collect import collect_spotify
from src.recommender_v2.config import (
    CollectConfig,
    CorpusConfig,
    ExportConfig,
    PipelineConfig,
    RetrievalConfig,
    RerankerConfig,
    RunConfig,
    SplitConfig,
)
from src.recommender_v2.dataset import build_corpus, enrich_metadata
from src.recommender_v2.evaluate import evaluate_pipeline
from src.recommender_v2.export_web import export_web
from src.recommender_v2.metrics import ndcg_at_k, recall_at_k
from src.recommender_v2.paths import RunLayout
from src.recommender_v2.reporting import build_run_report
from src.recommender_v2.retrieval import train_retrieval
from src.recommender_v2.reranker import train_reranker
from src.recommender_v2.splits import build_eval_splits
from src.recommender_v2.utils import read_jsonl


class RecommenderV2Tests(unittest.TestCase):
    def test_metrics_are_reasonable(self) -> None:
        predictions = ["a", "b", "c", "d"]
        positives = {"b", "d"}
        self.assertAlmostEqual(recall_at_k(predictions, positives, 2), 0.5)
        self.assertGreater(ndcg_at_k(predictions, positives, 4), 0.0)

    def test_pipeline_smoke_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = build_test_config(root)
            run = RunLayout.create(config, run_id="smoke")

            collect_summary = collect_spotify(config, run, live=False)
            self.assertGreater(collect_summary["playlist_count"], 0)

            corpus_summary = build_corpus(config, run)
            self.assertGreater(corpus_summary["tracks"], 0)

            enrich_summary = enrich_metadata(config, run)
            self.assertGreaterEqual(enrich_summary["tracks_with_tags"], 0)

            split_summary = build_eval_splits(config, run)
            self.assertGreater(split_summary["train"], 0)
            self.assertGreater(split_summary["val"], 0)

            retrieval_summary = train_retrieval(config, run)
            self.assertIn("recall@50", retrieval_summary["metrics"])
            retrieval_summary_repeat = train_retrieval(config, run)
            self.assertEqual(retrieval_summary["experiment"], retrieval_summary_repeat["experiment"])
            retrieval_progress = json.loads((run.manifests_dir / "train_retrieval_progress.json").read_text())
            self.assertEqual(retrieval_progress["status"], "completed")

            reranker_summary = train_reranker(config, run)
            self.assertIn("promoted", reranker_summary)
            self.assertIn("reranker_test_metrics", reranker_summary)
            reranker_progress = json.loads((run.manifests_dir / "train_reranker_progress.json").read_text())
            self.assertEqual(reranker_progress["status"], "completed")

            evaluation_summary = evaluate_pipeline(config, run)
            self.assertIn("results", evaluation_summary)
            self.assertIn("retrieval_test", evaluation_summary["results"])
            evaluation_progress = json.loads((run.manifests_dir / "evaluate_progress.json").read_text())
            self.assertEqual(evaluation_progress["status"], "completed")
            report_summary = build_run_report(run)
            self.assertIn("winners", report_summary)
            self.assertTrue((run.metrics_dir / "report.txt").exists())

            export_summary = export_web(config, run)
            self.assertGreater(export_summary["server_tracks"], 0)
            self.assertTrue((config.export.server_dir / "tracks.json").exists())
            self.assertTrue((config.export.public_dir / "viz_index.json").exists())

    def test_split_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = build_test_config(root)
            run = RunLayout.create(config, run_id="repro")

            collect_spotify(config, run, live=False)
            build_corpus(config, run)
            enrich_metadata(config, run)

            build_eval_splits(config, run)
            first = read_jsonl(run.splits_dir / "train.jsonl")
            build_eval_splits(config, run)
            second = read_jsonl(run.splits_dir / "train.jsonl")
            self.assertEqual(first, second)


def build_test_config(root: Path) -> PipelineConfig:
    project_root = root / "project"
    run_root = root / "runs"
    server_dir = root / "web" / "data" / "server" / "current"
    public_dir = root / "web" / "public" / "data"
    mpd_dir = root / "mpd"

    (project_root / "models").mkdir(parents=True, exist_ok=True)
    (project_root / "web" / "public" / "data").mkdir(parents=True, exist_ok=True)
    mpd_dir.mkdir(parents=True, exist_ok=True)

    track_metadata = pd.DataFrame(
        {
            "track_uri": [f"spotify:track:track{i}" for i in range(20)],
            "artist_name": [f"Artist {i % 5}" for i in range(20)],
        }
    )
    track_metadata.to_parquet(project_root / "models" / "track_metadata.parquet", index=False)

    with (project_root / "web" / "public" / "data" / "genres.json").open("w", encoding="utf-8") as handle:
        json.dump({"genres": [{"name": "rock", "count": 10}, {"name": "pop", "count": 9}]}, handle)

    tracks = []
    for i in range(20):
        tracks.append(
            {
                "track_uri": f"spotify:track:track{i}",
                "track_name": f"Track {i}",
                "artist_name": f"Artist {i % 5}",
                "album_name": f"Album {i % 4}",
                "artist_uri": f"spotify:artist:artist{i % 5}",
                "album_uri": f"spotify:album:album{i % 4}",
                "duration_ms": 180000 + i * 1000,
            }
        )

    playlists = []
    for playlist_id in range(12):
        start = playlist_id % 10
        sequence = [tracks[(start + offset) % len(tracks)] for offset in range(10)]
        playlists.append({"pid": playlist_id, "name": f"Playlist {playlist_id}", "tracks": sequence})

    with (mpd_dir / "mpd.slice.0.json").open("w", encoding="utf-8") as handle:
        json.dump({"playlists": playlists}, handle)

    return PipelineConfig(
        project_root=project_root,
        run=RunConfig(
            run_root=run_root,
            current_symlink=run_root / "current",
            default_run_prefix="test-run",
        ),
        collect=CollectConfig(
            market="US",
            max_playlists=100,
            playlists_per_query=5,
            tracks_per_playlist=50,
            top_artists=5,
            top_genres=2,
            year_buckets=[2000],
            query_keywords=["focus"],
            retry_count=1,
            retry_backoff_seconds=0.1,
            local_mpd_path=str(mpd_dir),
        ),
        corpus=CorpusConfig(
            min_playlist_tracks=5,
            max_playlist_tracks=200,
            drop_duplicate_tracks=True,
        ),
        split=SplitConfig(
            eligible_min_tracks=10,
            eligible_max_tracks=100,
            train_fraction=0.7,
            val_fraction=0.15,
            random_seed=7,
            min_seed_tracks=3,
            max_seed_tracks=5,
            min_positive_tracks=2,
            max_positive_tracks=4,
        ),
        retrieval=RetrievalConfig(
            vector_sizes=[16],
            windows=[3],
            min_counts=[1],
            negatives=[2],
            epochs=5,
            sg=1,
            seed=7,
            workers=1,
            candidate_pool_size=20,
        ),
        reranker=RerankerConfig(
            max_iter=10,
            max_depth=3,
            learning_rate=0.1,
            l2_regularization=0.0,
            random_seed=7,
            max_seed_neighbor_probe=10,
        ),
        export=ExportConfig(
            server_dir=server_dir,
            public_dir=public_dir,
            public_viz_tracks=10,
            tsne_perplexity=3,
            tsne_iter=250,
            tsne_random_state=7,
        ),
    )


if __name__ == "__main__":
    unittest.main()
