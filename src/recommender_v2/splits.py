from __future__ import annotations

from collections import defaultdict
import random

import pandas as pd

from .config import PipelineConfig
from .paths import RunLayout
from .utils import write_json, write_jsonl


def build_eval_splits(config: PipelineConfig, layout: RunLayout) -> dict:
    playlists_df = pd.read_parquet(layout.normalized_dir / "playlists.parquet")
    playlist_tracks_df = pd.read_parquet(layout.normalized_dir / "playlist_tracks.parquet")

    grouped_tracks: dict[str, list[str]] = defaultdict(list)
    for row in playlist_tracks_df.sort_values(["playlist_id", "position"]).itertuples(index=False):
        grouped_tracks[row.playlist_id].append(row.track_uri)

    eligible_records: list[dict] = []
    for row in playlists_df.itertuples(index=False):
        sequence = grouped_tracks.get(row.playlist_id, [])
        length = len(sequence)
        if length < config.split.eligible_min_tracks or length > config.split.eligible_max_tracks:
            continue
        seed_count, positive_count = _seed_positive_counts(config, length)
        eligible_records.append(
            {
                "playlist_id": row.playlist_id,
                "playlist_name": row.playlist_name,
                "sequence_length": length,
                "seed_tracks": sequence[:seed_count],
                "positive_tracks": sequence[seed_count : seed_count + positive_count],
            }
        )

    rng = random.Random(config.split.random_seed)
    rng.shuffle(eligible_records)

    train_cut = int(len(eligible_records) * config.split.train_fraction)
    val_cut = train_cut + int(len(eligible_records) * config.split.val_fraction)
    splits = {
        "train": eligible_records[:train_cut],
        "val": eligible_records[train_cut:val_cut],
        "test": eligible_records[val_cut:],
    }

    for split_name, records in splits.items():
        write_jsonl(layout.splits_dir / f"{split_name}.jsonl", records)

    summary = {name: len(records) for name, records in splits.items()}
    write_json(layout.manifests_dir / "split_eval.json", summary)
    return summary


def _seed_positive_counts(config: PipelineConfig, length: int) -> tuple[int, int]:
    seed_count = min(
        config.split.max_seed_tracks,
        max(config.split.min_seed_tracks, int(length * 0.25)),
    )
    positive_count = min(
        config.split.max_positive_tracks,
        max(config.split.min_positive_tracks, int(length * 0.15)),
    )
    if seed_count + positive_count >= length:
        positive_count = max(config.split.min_positive_tracks, length - seed_count - 1)
    return seed_count, positive_count
