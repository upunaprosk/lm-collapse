from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from collapse.contamination.build_dataset import (
    BuildDatasetConfig,
    build_training_dataset,
)
from collapse.data.base import (
    load_examples_jsonl,
    save_examples_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a training corpus from aligned human and generated "
            "examples. Main-paper protocol uses --synthetic-fraction 1.0; "
            "smaller values are contamination-fraction ablations."
        )
    )

    parser.add_argument(
        "--human",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--generated",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--synthetic-fraction",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--stratify-key",
        type=str,
        default="profession_id",
    )

    parser.add_argument(
        "--no-stratify",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"Output already exists: {args.output}"
        )

    human = load_examples_jsonl(
        args.human
    )

    generated = load_examples_jsonl(
        args.generated
    )

    config = BuildDatasetConfig(
        synthetic_fraction=(
            args.synthetic_fraction
        ),
        seed=args.seed,
        stratify_by_metadata_key=(
            None
            if args.no_stratify
            else args.stratify_key
        ),
    )

    built = build_training_dataset(
        human_examples=human,
        generated_examples=generated,
        config=config,
    )

    save_examples_jsonl(
        built,
        args.output,
    )

    num_synthetic = sum(
        example.source == "synthetic"
        for example in built
    )

    manifest = {
        "human": str(
            args.human.resolve()
        ),
        "generated": str(
            args.generated.resolve()
        ),
        "output": str(
            args.output.resolve()
        ),
        "config": asdict(config),
        "num_examples": len(built),
        "num_synthetic_examples": int(
            num_synthetic
        ),
        "realized_synthetic_document_fraction": (
            num_synthetic
            / len(built)
            if built
            else 0.0
        ),
    }

    manifest_path = (
        args.output.parent
        / (
            args.output.stem
            + "_manifest.json"
        )
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Built {len(built):,} examples -> {args.output}"
    )
    print(
        f"Synthetic rows: {num_synthetic:,} "
        f"({manifest['realized_synthetic_document_fraction']:.2%})"
    )


if __name__ == "__main__":
    main()
