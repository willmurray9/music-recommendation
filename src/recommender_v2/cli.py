from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .collect import collect_spotify
from .dataset import build_corpus, enrich_metadata
from .evaluate import evaluate_pipeline
from .export_web import export_web
from .paths import RunLayout
from .reranker import train_reranker
from .retrieval import train_retrieval
from .splits import build_eval_splits
from .utils import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recommender V2 pipeline")
    parser.add_argument("--config", default=None, help="Path to recommender_v2.toml")
    parser.add_argument("--run-id", default=None, help="Explicit run identifier")

    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect_spotify")
    collect.add_argument("--live", action="store_true", help="Use Spotify Web API instead of local fallback")

    subparsers.add_parser("build_corpus")
    subparsers.add_parser("enrich_metadata")
    subparsers.add_parser("split_eval")
    subparsers.add_parser("train_retrieval")
    subparsers.add_parser("train_reranker")
    subparsers.add_parser("evaluate")
    subparsers.add_parser("export_web")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    layout = RunLayout.create(config, run_id=args.run_id)

    if args.command == "collect_spotify":
        result = collect_spotify(config, layout, live=args.live)
    elif args.command == "build_corpus":
        result = build_corpus(config, layout)
    elif args.command == "enrich_metadata":
        result = enrich_metadata(config, layout)
    elif args.command == "split_eval":
        result = build_eval_splits(config, layout)
    elif args.command == "train_retrieval":
        result = train_retrieval(config, layout)
    elif args.command == "train_reranker":
        result = train_reranker(config, layout)
    elif args.command == "evaluate":
        result = evaluate_pipeline(config, layout)
    elif args.command == "export_web":
        result = export_web(config, layout)
        layout.update_current_symlink(config.run.current_symlink)
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    write_json(layout.manifests_dir / "last_command.json", {"command": args.command, "result": result})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
