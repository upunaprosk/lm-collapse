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

    if not 0.0 <= config.synthetic_fraction <= 1.0:
        raise ValueError("synthetic_fraction must be in [0, 1].")

    human = _index(human_examples, "human")
    generated = _index(generated_examples, "generated")

    if set(human) != set(generated):
        raise ValueError(
            "Human/generated corpora must contain identical IDs."
        )

    eligible_generated = [
        example_id
        for example_id, generated_example in generated.items()
        if generated_example.source == "synthetic"
        and generated_example.synthetic_suffix is not None
    ]

    requested_n = round(
        len(human_examples) * config.synthetic_fraction
    )
    target_n = min(requested_n, len(eligible_generated))
    rng = random.Random(config.seed)

    if target_n <= 0:
        selected: set[str] = set()
    elif config.stratify_by_metadata_key is None:
        candidates = list(eligible_generated)
        rng.shuffle(candidates)
        selected = set(candidates[:target_n])
    else:
        key = config.stratify_by_metadata_key
        all_groups: dict[str, list[str]] = {}
        eligible_groups: dict[str, list[str]] = {}

        for example_id, human_example in human.items():
            group = str(human_example.metadata.get(key))
            all_groups.setdefault(group, []).append(example_id)

        for example_id in eligible_generated:
            group = str(human[example_id].metadata.get(key))
            eligible_groups.setdefault(group, []).append(example_id)

        # Desired quotas preserve the HUMAN corpus stratum proportions.
        raw_quota = {
            group: target_n * len(ids) / len(human_examples)
            for group, ids in all_groups.items()
        }
        quota = {
            group: min(
                int(raw_quota[group]),
                len(eligible_groups.get(group, [])),
            )
            for group in all_groups
        }

        # Allocate the remainder deterministically while respecting capacity.
        remaining = target_n - sum(quota.values())
        while remaining > 0:
            possible = [
                group
                for group in all_groups
                if quota[group] < len(eligible_groups.get(group, []))
            ]
            if not possible:
                break
            group = max(
                possible,
                key=lambda g: (
                    raw_quota[g] - quota[g],
                    -quota[g] / max(len(all_groups[g]), 1),
                    g,
                ),
            )
            quota[group] += 1
            remaining -= 1

        selected = set()
        for group in sorted(eligible_groups):
            ids = list(eligible_groups[group])
            rng.shuffle(ids)
            selected.update(ids[: quota.get(group, 0)])

    output: list[Example] = []
    for human_example in human_examples:
        if human_example.id in selected:
            out = deepcopy(generated[human_example.id])
            out.metadata = dict(out.metadata)
            out.metadata["training_dataset_selected_synthetic"] = True
        else:
            out = deepcopy(human_example)
            out.metadata = dict(out.metadata)
            out.metadata["training_dataset_selected_synthetic"] = False
        output.append(out)

    return output

