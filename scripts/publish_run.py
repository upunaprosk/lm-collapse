from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SMALL_SUFFIXES = {
    ".json",
    ".jsonl",
    ".csv",
    ".yaml",
    ".yml",
    ".txt",
    ".log",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a lightweight publication bundle containing configs, "
            "manifests, evaluation outputs, and aggregate tables while "
            "excluding model weights and large trainer data."
        )
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=25.0,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def should_skip(path: Path) -> bool:
    lowered = {
        part.lower()
        for part in path.parts
    }

    if "checkpoint" in lowered:
        return True

    if "trainer_data" in lowered:
        return True

    if path.suffix.lower() in {
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
    }:
        return True

    return False


def main() -> None:
    args = parse_args()

    if not args.run_root.exists():
        raise FileNotFoundError(
            args.run_root
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        any(
            args.output_dir.iterdir()
        )
        and not args.force
    ):
        raise FileExistsError(
            f"Output is non-empty: {args.output_dir}"
        )

    copied = []
    skipped = []

    max_bytes = int(
        args.max_file_mb
        * 1024
        * 1024
    )

    for source in args.run_root.rglob("*"):
        if not source.is_file():
            continue

        relative = source.relative_to(
            args.run_root
        )

        if should_skip(relative):
            skipped.append(
                str(relative)
            )
            continue

        if (
            source.stat().st_size
            > max_bytes
        ):
            skipped.append(
                str(relative)
            )
            continue

        if (
            source.suffix.lower()
            not in SMALL_SUFFIXES
        ):
            skipped.append(
                str(relative)
            )
            continue

        target = (
            args.output_dir
            / relative
        )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source,
            target,
        )

        copied.append(
            str(relative)
        )

    manifest = {
        "source_run_root": str(
            args.run_root.resolve()
        ),
        "num_files_copied": len(
            copied
        ),
        "num_files_skipped": len(
            skipped
        ),
        "copied": copied,
        "skipped": skipped,
    }

    with (
        args.output_dir
        / "publication_bundle_manifest.json"
    ).open(
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
        f"Publication bundle written to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
