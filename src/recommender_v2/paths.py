from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os

from .config import PipelineConfig


@dataclass(frozen=True)
class RunLayout:
    root: Path
    raw_dir: Path
    normalized_dir: Path
    splits_dir: Path
    models_dir: Path
    metrics_dir: Path
    export_dir: Path
    manifests_dir: Path

    @classmethod
    def create(cls, config: PipelineConfig, run_id: str | None = None) -> "RunLayout":
        if not run_id:
            stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            run_id = f"{config.run.default_run_prefix}-{stamp}"

        root = config.run.run_root / run_id
        raw_dir = root / "raw"
        normalized_dir = root / "normalized"
        splits_dir = root / "splits"
        models_dir = root / "models"
        metrics_dir = root / "metrics"
        export_dir = root / "export"
        manifests_dir = root / "manifests"

        for path in [
            config.run.run_root,
            root,
            raw_dir,
            normalized_dir,
            splits_dir,
            models_dir,
            metrics_dir,
            export_dir,
            manifests_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        return cls(
            root=root,
            raw_dir=raw_dir,
            normalized_dir=normalized_dir,
            splits_dir=splits_dir,
            models_dir=models_dir,
            metrics_dir=metrics_dir,
            export_dir=export_dir,
            manifests_dir=manifests_dir,
        )

    def update_current_symlink(self, symlink_path: Path) -> None:
        symlink_path.parent.mkdir(parents=True, exist_ok=True)
        if symlink_path.is_symlink() or symlink_path.exists():
            if symlink_path.is_dir() and not symlink_path.is_symlink():
                raise ValueError(f"Refusing to replace non-symlink directory: {symlink_path}")
            symlink_path.unlink()
        target = os.path.relpath(self.root, symlink_path.parent)
        symlink_path.symlink_to(target)
