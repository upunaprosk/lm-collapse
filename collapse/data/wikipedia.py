from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

from collapse.data.base import PreparedDataset
from collapse.data.chunking import (
    ChunkingConfig,
    chunk_examples,
)
from collapse.records import Example


@dataclass(frozen=True)
class WikipediaConfig:
    dataset_name: str = "wikimedia/wikipedia"
    dataset_config_name: str | None = "20231101.en"
    revision: str | None = None

    text_column: str = "text"
    id_column: str | None = "id"

    splits: dict[str, str] = field(
        default_factory=lambda: {
            "train": "train",
        }
    )

    preserve_columns: list[str] = field(
        default_factory=lambda: [
            "title",
            "url",
        ]
    )

    min_document_chars: int = 200
    max_documents_per_split: int | None = None

    holdout_fraction: float = 0.01
    holdout_split_name: str = "validation"
    split_seed: int = 2026

    reference_tokenizer: str = "Qwen/Qwen2.5-0.5B"
    chunking: ChunkingConfig = field(
        default_factory=ChunkingConfig
    )


def _stable_fraction(
    value: str,
    *,
    seed: int,
) -> float:
    digest = hashlib.sha256(
        f"{seed}:{value}".encode(
            "utf-8"
        )
    ).digest()

    integer = int.from_bytes(
        digest[:8],
        "big",
        signed=False,
    )

    return integer / float(
        2 ** 64
    )


class WikipediaAdapter:
    def __init__(
        self,
        config: WikipediaConfig,
    ) -> None:
        self.config = config

    def _load(self):
        kwargs: dict[str, Any] = {}

        if self.config.revision is not None:
            kwargs["revision"] = (
                self.config.revision
            )

        if self.config.dataset_config_name:
            return load_dataset(
                self.config.dataset_name,
                self.config.dataset_config_name,
                **kwargs,
            )

        return load_dataset(
            self.config.dataset_name,
            **kwargs,
        )

    def _documents_from_source(
        self,
        source_rows,
        *,
        canonical_split: str,
    ) -> list[Example]:
        documents: list[Example] = []

        for row_index, row in enumerate(
            source_rows
        ):
            if (
                self.config.max_documents_per_split
                is not None
                and row_index
                >= self.config.max_documents_per_split
            ):
                break

            text = row.get(
                self.config.text_column
            )

            if (
                text is None
                or len(
                    str(text).strip()
                )
                < self.config.min_document_chars
            ):
                continue

            if (
                self.config.id_column
                and row.get(
                    self.config.id_column
                )
                is not None
            ):
                example_id = str(
                    row[
                        self.config.id_column
                    ]
                )
            else:
                example_id = (
                    f"{canonical_split}:{row_index:08d}"
                )

            metadata = {
                "source_dataset": (
                    self.config.dataset_name
                ),
            }

            for column in (
                self.config.preserve_columns
            ):
                if column in row:
                    metadata[column] = (
                        row[column]
                    )

            documents.append(
                Example(
                    id=example_id,
                    text=str(text),
                    metadata=metadata,
                    source="human",
                )
            )

        return documents

    def prepare(self) -> PreparedDataset:
        if not (
            0.0
            <= self.config.holdout_fraction
            < 1.0
        ):
            raise ValueError(
                "holdout_fraction must be in [0, 1)."
            )

        raw_dataset = self._load()

        tokenizer = (
            AutoTokenizer.from_pretrained(
                self.config.reference_tokenizer,
                use_fast=True,
            )
        )

        document_splits: dict[
            str,
            list[Example],
        ] = {}

        for canonical_split, source_split in (
            self.config.splits.items()
        ):
            if source_split not in raw_dataset:
                raise KeyError(
                    f"Configured split {source_split!r} does not exist. "
                    f"Available: {list(raw_dataset.keys())}"
                )

            documents = (
                self._documents_from_source(
                    raw_dataset[
                        source_split
                    ],
                    canonical_split=(
                        canonical_split
                    ),
                )
            )

            if (
                canonical_split == "train"
                and self.config.holdout_fraction
                > 0.0
                and self.config.holdout_split_name
                not in self.config.splits
            ):
                train_docs = []
                holdout_docs = []

                for example in documents:
                    value = _stable_fraction(
                        example.id,
                        seed=(
                            self.config.split_seed
                        ),
                    )

                    if (
                        value
                        < self.config.holdout_fraction
                    ):
                        holdout_docs.append(
                            example
                        )
                    else:
                        train_docs.append(
                            example
                        )

                document_splits[
                    "train"
                ] = train_docs

                document_splits[
                    self.config.holdout_split_name
                ] = holdout_docs
            else:
                document_splits[
                    canonical_split
                ] = documents

        splits = {
            split_name: chunk_examples(
                documents,
                tokenizer=tokenizer,
                config=(
                    self.config.chunking
                ),
            )
            for split_name, documents
            in document_splits.items()
        }

        dataset = PreparedDataset(
            name="wikipedia",
            splits=splits,
        )
        dataset.validate()
        return dataset
