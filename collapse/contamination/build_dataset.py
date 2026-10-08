from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

from collapse.records import Example


@dataclass(frozen=True)
class BuildDatasetConfig:
    """
    Construct the training corpus from aligned human/generated rows.
    """
    synthetic_fraction: float = 1.0
    seed: int = 42
    stratify_by_metadata_key: str | None = "profession_id"


def _index(
    examples: Sequence[Example],
    name: str,
) -> dict[str, Example]:
    result = {}

    for example in examples:
        if example.id in result:
            raise ValueError(
                f"Duplicate ID {example.id!r} in {name}."
            )
        result[example.id] = example

    return result


def build_training_dataset(
    *,
    human_examples: Sequence[Example],
    generated_examples: Sequence[Example],
    config: BuildDatasetConfig | None = None,
) -> list[Example]:
    if config is None:
        config = BuildDatasetConfig()

    if not (
        0.0
        <= config.synthetic_fraction
        <= 1.0
    ):
        raise ValueError(
            "synthetic_fraction must be in [0, 1]."
        )

    human = _index(
        human_examples,
        "human",
    )
    generated = _index(
        generated_examples,
        "generated",
    )

    if set(human) != set(generated):
        raise ValueError(
            "Human/generated corpora must contain identical IDs."
        )

    eligible_generated = [
        example_id
        for example_id, generated_example
        in generated.items()
        if generated_example.source == "synthetic"
        and generated_example.synthetic_suffix
        is not None
    ]

    if config.synthetic_fraction >= 1.0:
        selected = set(
            eligible_generated
        )
    elif config.synthetic_fraction <= 0.0:
        selected = set()
    else:
        rng = random.Random(
            config.seed
        )

        if (
            config.stratify_by_metadata_key
            is None
        ):
            candidates = list(
                eligible_generated
            )
            rng.shuffle(
                candidates
            )

            n = round(
                len(candidates)
                * config.synthetic_fraction
            )
            selected = set(
                candidates[:n]
            )
        else:
            groups = {}

            for example_id in (
                eligible_generated
            ):
                value = human[
                    example_id
                ].metadata.get(
                    config.stratify_by_metadata_key
                )

                groups.setdefault(
                    str(value),
                    [],
                ).append(
                    example_id
                )

            selected = set()

            for group_ids in groups.values():
                group_ids = list(
                    group_ids
                )
                rng.shuffle(
                    group_ids
                )

                n = round(
                    len(group_ids)
                    * config.synthetic_fraction
                )

                selected.update(
                    group_ids[:n]
                )

    output = []

    for human_example in human_examples:
        if human_example.id in selected:
            out = deepcopy(
                generated[
                    human_example.id
                ]
            )
            out.metadata = dict(
                out.metadata
            )
            out.metadata[
                "training_dataset_selected_synthetic"
            ] = True
        else:
            out = deepcopy(
                human_example
            )
            out.metadata = dict(
                out.metadata
            )
            out.metadata[
                "training_dataset_selected_synthetic"
            ] = False

        output.append(out)

    return output
