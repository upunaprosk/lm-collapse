from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from collapse.records import Example


@dataclass
class PreparedDataset:
    name: str
    splits: dict[str, list[Example]]

    def available_splits(self) -> list[str]:
        return sorted(self.splits)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("PreparedDataset.name must be non-empty.")

        if not self.splits:
            raise ValueError("PreparedDataset has no splits.")

        for split_name, examples in self.splits.items():
            if not split_name.strip():
                raise ValueError("Split name must be non-empty.")

            ids = [str(example.id) for example in examples]

            if len(ids) != len(set(ids)):
                raise ValueError(
                    f"Duplicate example IDs inside split {split_name!r}."
                )

            empty = [
                example.id
                for example in examples
                if not example.text.strip()
            ]

            if empty:
                raise ValueError(
                    f"Split {split_name!r} contains empty text. "
                    f"Examples: {empty[:5]}"
                )


def save_examples_jsonl(
    examples: Sequence[Example],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(
                json.dumps(
                    example.to_dict(),
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_examples_jsonl(
    path: str | Path,
) -> list[Example]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")

    examples: list[Example] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from e

            if not isinstance(value, dict):
                raise TypeError(
                    f"Expected JSON object at {path}:{line_number}."
                )

            examples.append(
                Example.from_dict(value)
            )

    return examples


def save_prepared_dataset(
    dataset: PreparedDataset,
    output_dir: str | Path,
) -> None:
    dataset.validate()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_files = {}

    for split_name, examples in dataset.splits.items():
        filename = f"{split_name}.jsonl"

        save_examples_jsonl(
            examples,
            output_dir / filename,
        )

        split_files[split_name] = {
            "file": filename,
            "num_examples": len(examples),
        }

    metadata = {
        "name": dataset.name,
        "splits": split_files,
    }

    with (output_dir / "dataset.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )


def load_prepared_dataset(
    directory: str | Path,
) -> PreparedDataset:
    directory = Path(directory)
    metadata_path = directory / "dataset.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"dataset.json does not exist: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    splits = {}

    for split_name, info in metadata["splits"].items():
        splits[split_name] = load_examples_jsonl(
            directory / info["file"]
        )

    dataset = PreparedDataset(
        name=str(metadata["name"]),
        splits=splits,
    )

    dataset.validate()
    return dataset
