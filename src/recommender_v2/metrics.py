from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import math

import numpy as np

from .utils import mean_pairwise_distance


def recall_at_k(predictions: list[str], positives: set[str], k: int) -> float:
    if not positives:
        return 0.0
    hits = len(set(predictions[:k]) & positives)
    return hits / len(positives)


def ndcg_at_k(predictions: list[str], positives: set[str], k: int) -> float:
    dcg = 0.0
    for idx, track_uri in enumerate(predictions[:k], start=1):
        if track_uri in positives:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(positives), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def mrr_at_k(predictions: list[str], positives: set[str], k: int) -> float:
    for idx, track_uri in enumerate(predictions[:k], start=1):
        if track_uri in positives:
            return 1.0 / idx
    return 0.0


def same_artist_rate(track_uris: list[str], artist_lookup: dict[str, str]) -> float:
    if not track_uris:
        return 0.0
    artists = [artist_lookup.get(uri) for uri in track_uris if artist_lookup.get(uri)]
    if not artists:
        return 0.0
    counts = Counter(artists)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return duplicates / len(track_uris)


def catalog_coverage(recommendation_lists: Iterable[list[str]], catalog_size: int) -> float:
    seen = set()
    for recs in recommendation_lists:
        seen.update(recs)
    if catalog_size == 0:
        return 0.0
    return len(seen) / catalog_size


def unique_artist_coverage(recommendation_lists: Iterable[list[str]], artist_lookup: dict[str, str]) -> float:
    seen = set()
    for recs in recommendation_lists:
        for uri in recs:
            artist = artist_lookup.get(uri)
            if artist:
                seen.add(artist)
    catalog_artists = {artist for artist in artist_lookup.values() if artist}
    if not catalog_artists:
        return 0.0
    return len(seen) / len(catalog_artists)


def mean_popularity_percentile(recommendation_lists: Iterable[list[str]], popularity_lookup: dict[str, float]) -> float:
    values = [popularity_lookup[uri] for recs in recommendation_lists for uri in recs if uri in popularity_lookup]
    return float(np.mean(values)) if values else 0.0


def intra_list_diversity(recommendation_lists: Iterable[list[str]], vector_lookup: dict[str, np.ndarray]) -> float:
    values: list[float] = []
    for recs in recommendation_lists:
        vectors = [vector_lookup[uri] for uri in recs if uri in vector_lookup]
        values.append(mean_pairwise_distance(vectors))
    return float(np.mean(values)) if values else 0.0
