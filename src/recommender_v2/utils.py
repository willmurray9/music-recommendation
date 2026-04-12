from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def extract_spotify_id(uri: str | None) -> str | None:
    if not uri:
        return None
    return uri.split(":")[-1]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def percentile_rank(sorted_values: np.ndarray, value: float) -> float:
    if sorted_values.size == 0:
        return 0.0
    idx = np.searchsorted(sorted_values, value, side="left")
    return float(idx / max(len(sorted_values) - 1, 1))


def mean_pairwise_distance(vectors: list[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 0.0
    distances: list[float] = []
    for idx, left in enumerate(vectors[:-1]):
        for right in vectors[idx + 1 :]:
            distances.append(1.0 - cosine_similarity(left, right))
    return float(np.mean(distances)) if distances else 0.0


def safe_year(value: object) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value)
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return None


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    records = df.to_dict("records")
    fixed: list[dict] = []
    for record in records:
        normalized = {}
        for key, value in record.items():
            if hasattr(value, "tolist"):
                normalized[key] = value.tolist()
            elif isinstance(value, np.generic):
                normalized[key] = value.item()
            else:
                normalized[key] = value
        fixed.append(normalized)
    return fixed


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def log_event(stage: str, message: str, **fields: object) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    detail = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    suffix = f" | {detail}" if detail else ""
    print(f"[{timestamp}] {stage}: {message}{suffix}", flush=True)
