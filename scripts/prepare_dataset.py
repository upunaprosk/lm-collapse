from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from collapse.contamination.seed_policy import (
    SeedPolicyConfig,
    apply_seed_policy,
    compute_token_lengths,
    fit_seed_policy,
    load_reference_tokenizer,
    save_seed_policy,
    save_seed_sweep,
    seed_length_sweep,
)
from collapse.data.base import (
    PreparedDataset,
    save_prepared_dataset,
)
from collapse.data.bias_in_bios import (
    BiasInBiosAdapter,
    BiasInBiosConfig,
)
from collapse.data.chunking import (
    ChunkingConfig,
)
from collapse.data.wikipedia import (
    WikipediaAdapter,
    WikipediaConfig,
)
from collapse.records import Example


DEFAULT_CONFIG = "configs/datasets/bias_in_bios.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a human dataset, fit one canonical seeded-continuation "
            "policy on TRAIN ONLY, and freeze raw-text prefixes."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f)

    if not isinstance(value, dict):
        raise TypeError(
            "Top-level dataset config must be a YAML mapping."
        )

    return value


def mapping(
    value: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    section = value.get(key, {})

    if section is None:
        return {}

    if not isinstance(section, dict):
        raise TypeError(
            f"`{key}` must be a YAML mapping."
        )

    return section


def build_adapter(
    raw: dict[str, Any],
):
    dataset = mapping(
        raw,
        "dataset",
    )

    kind = str(
        dataset.get(
            "kind",
            "bias_in_bios",
        )
    ).strip().lower()

    if kind == "bias_in_bios":
        return BiasInBiosAdapter(
            BiasInBiosConfig(
                dataset_name=str(
                    dataset[
                        "dataset_name"
                    ]
                ),
                dataset_config_name=(
                    dataset.get(
                        "dataset_config_name"
                    )
                ),
                revision=(
                    dataset.get(
                        "revision"
                    )
                ),
                text_column=str(
                    dataset.get(
                        "text_column",
                        "hard_text",
                    )
                ),
                profession_column=str(
                    dataset.get(
                        "profession_column",
                        "profession",
                    )
                ),
                gender_column=(
                    dataset.get(
                        "gender_column",
                        "gender",
                    )
                ),
                id_column=(
                    dataset.get(
                        "id_column"
                    )
                ),
                splits=dict(
                    dataset.get(
                        "splits",
                        {
                            "train": "train",
                        },
                    )
                ),
                preserve_columns=list(
                    dataset.get(
                        "preserve_columns",
                        [],
                    )
                ),
                profession_index_base=(
                    int(
                        dataset[
                            "profession_index_base"
                        ]
                    )
                    if dataset.get(
                        "profession_index_base"
                    )
                    is not None
                    else None
                ),
                gender_map=(
                    {
                        str(k): str(v)
                        for k, v
                        in dataset[
                            "gender_map"
                        ].items()
                    }
                    if isinstance(
                        dataset.get(
                            "gender_map"
                        ),
                        dict,
                    )
                    else None
                ),
            )
        )

    if kind == "wikipedia":
        chunk = mapping(
            dataset,
            "chunking",
        )

        return WikipediaAdapter(
            WikipediaConfig(
                dataset_name=str(
                    dataset.get(
                        "dataset_name",
                        "wikimedia/wikipedia",
                    )
                ),
                dataset_config_name=(
                    dataset.get(
                        "dataset_config_name",
                        "20231101.en",
                    )
                ),
                revision=(
                    dataset.get(
                        "revision"
                    )
                ),
                text_column=str(
                    dataset.get(
                        "text_column",
                        "text",
                    )
                ),
                id_column=(
                    dataset.get(
                        "id_column",
                        "id",
                    )
                ),
                splits=dict(
                    dataset.get(
                        "splits",
                        {
                            "train": "train",
                        },
                    )
                ),
                preserve_columns=list(
                    dataset.get(
                        "preserve_columns",
                        [
                            "title",
                            "url",
                        ],
                    )
                ),
                min_document_chars=int(
                    dataset.get(
                        "min_document_chars",
                        200,
                    )
                ),
                max_documents_per_split=(
                    int(
                        dataset[
                            "max_documents_per_split"
                        ]
                    )
                    if dataset.get(
                        "max_documents_per_split"
                    )
                    is not None
                    else None
                ),
                holdout_fraction=float(
                    dataset.get(
                        "holdout_fraction",
                        0.01,
                    )
                ),
                holdout_split_name=str(
                    dataset.get(
                        "holdout_split_name",
                        "validation",
                    )
                ),
                split_seed=int(
                    dataset.get(
                        "split_seed",
                        2026,
                    )
                ),
                reference_tokenizer=str(
                    dataset.get(
                        "reference_tokenizer",
                        mapping(
                            raw,
                            "seed_policy",
                        ).get(
                            "reference_tokenizer",
                            "Qwen/Qwen2.5-0.5B",
                        ),
                    )
                ),
                chunking=ChunkingConfig(
                    chunk_tokens=int(
                        chunk.get(
                            "chunk_tokens",
                            512,
                        )
                    ),
                    min_chunk_tokens=int(
                        chunk.get(
                            "min_chunk_tokens",
                            128,
                        )
                    ),
                    drop_short_final_chunk=bool(
                        chunk.get(
                            "drop_short_final_chunk",
                            True,
                        )
                    ),
                ),
            )
        )

    raise ValueError(
        f"Unknown dataset.kind={kind!r}."
    )


def build_seed_config(
    raw: dict[str, Any],
) -> SeedPolicyConfig:
    cfg = mapping(
        raw,
        "seed_policy",
    )

    return SeedPolicyConfig(
        reference_tokenizer=str(
            cfg.get(
                "reference_tokenizer",
                "Qwen/Qwen2.5-0.5B",
            )
        ),
        target_generated_fraction=float(
            cfg.get(
                "target_generated_fraction",
                0.50,
            )
        ),
        min_seed_tokens=int(
            cfg.get(
                "min_seed_tokens",
                5,
            )
        ),
        max_seed_tokens=int(
            cfg.get(
                "max_seed_tokens",
                60,
            )
        ),
        candidate_step=int(
            cfg.get(
                "candidate_step",
                1,
            )
        ),
        min_suffix_tokens=int(
            cfg.get(
                "min_suffix_tokens",
                5,
            )
        ),
        aggregation=str(
            cfg.get(
                "aggregation",
                "example_mean",
            )
        ),
    )


def ensure_output_dir(
    path: Path,
    force: bool,
) -> None:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    if any(path.iterdir()) and not force:
        raise FileExistsError(
            f"Output directory is not empty: {path}. "
            "Use --force intentionally."
        )


def validate_prefixes(
    examples: list[Example],
) -> None:
    errors = []

    for example in examples:
        if (
            example.prefix_text is None
            or example.human_suffix is None
        ):
            errors.append(
                f"{example.id}: missing prefix/suffix"
            )
            continue

        if (
            example.prefix_text
            + example.human_suffix
            != example.text
        ):
            errors.append(
                f"{example.id}: prefix+suffix != text"
            )

        if len(errors) >= 10:
            break

    if errors:
        raise RuntimeError(
            "Frozen prefix integrity failure:\n  - "
            + "\n  - ".join(
                errors
            )
        )


def save_preview(
    examples: list[Example],
    path: Path,
    n: int,
) -> None:
    if n <= 0:
        return

    eligible = [
        example
        for example in examples
        if example.metadata.get(
            "seed_eligible",
            False,
        )
    ][:n]

    with path.open("w", encoding="utf-8") as f:
        for example in eligible:
            f.write(
                json.dumps(
                    {
                        "id": example.id,
                        "metadata": (
                            example.metadata
                        ),
                        "prefix_text": (
                            example.prefix_text
                        ),
                        "human_suffix": (
                            example.human_suffix
                        ),
                        "original_text": (
                            example.text
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def numeric_summary(
    values: np.ndarray,
) -> dict[str, Any]:
    if len(values) == 0:
        return {
            "count": 0,
        }

    return {
        "count": int(
            len(values)
        ),
        "mean": float(
            values.mean()
        ),
        "std": float(
            values.std()
        ),
        "min": int(
            values.min()
        ),
        "median": float(
            np.median(
                values
            )
        ),
        "max": int(
            values.max()
        ),
    }


def main() -> None:
    args = parse_args()
    raw = load_yaml(
        args.config
    )

    adapter = build_adapter(
        raw
    )

    seed_cfg = build_seed_config(
        raw
    )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(
            raw.get(
                "output_dir",
                "data/processed/dataset",
            )
        )
    )

    ensure_output_dir(
        output_dir,
        args.force,
    )

    dataset = adapter.prepare()

    if "train" not in dataset.splits:
        raise ValueError(
            "Prepared dataset must contain canonical `train` split."
        )

    tokenizer = (
        load_reference_tokenizer(
            seed_cfg.reference_tokenizer
        )
    )

    human_train = dataset.splits[
        "train"
    ]

    policy = fit_seed_policy(
        examples=human_train,
        config=seed_cfg,
        tokenizer=tokenizer,
    )

    seeded_train = apply_seed_policy(
        examples=human_train,
        policy=policy,
        tokenizer=tokenizer,
    )

    validate_prefixes(
        seeded_train
    )

    processed = PreparedDataset(
        name=dataset.name,
        splits={
            **dataset.splits,
            "train": seeded_train,
        },
    )

    save_prepared_dataset(
        processed,
        output_dir,
    )

    save_seed_policy(
        policy,
        output_dir
        / "seed_policy.json",
    )

    sweep = seed_length_sweep(
        examples=human_train,
        config=seed_cfg,
        tokenizer=tokenizer,
    )

    save_seed_sweep(
        sweep,
        output_dir
        / "seed_sweep.json",
    )

    stats = {
        "dataset": (
            processed.name
        ),
        "seed_policy": (
            asdict(
                policy
            )
        ),
        "splits": {},
    }

    for split_name, examples in (
        processed.splits.items()
    ):
        lengths = (
            compute_token_lengths(
                examples,
                tokenizer,
            )
        )

        stats[
            "splits"
        ][
            split_name
        ] = {
            "num_examples": (
                len(
                    examples
                )
            ),
            "reference_token_lengths": (
                numeric_summary(
                    lengths
                )
            ),
        }

    with (
        output_dir
        / "stats.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=False,
        )

    save_preview(
        seeded_train,
        output_dir
        / "prefix_preview.jsonl",
        args.preview,
    )

    shutil.copy2(
        args.config,
        output_dir
        / "prepare_config.yaml",
    )

    print()
    print("=" * 72)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 72)
    print(
        f"Dataset:       {processed.name}"
    )
    print(
        f"Train rows:    {len(seeded_train):,}"
    )
    print(
        f"Seed length:   {policy.seed_length}"
    )
    print(
        f"Generated frac:{policy.realized_generated_fraction:.3f}"
    )
    print(
        f"Output:        {output_dir}"
    )


if __name__ == "__main__":
    main()
