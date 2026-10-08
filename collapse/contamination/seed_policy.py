from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from collapse.records import Example


@dataclass(frozen=True)
class SeedPolicyConfig:
    reference_tokenizer: str = "Qwen/Qwen2.5-0.5B"

    target_generated_fraction: float = 0.50

    min_seed_tokens: int = 5
    max_seed_tokens: int = 100
    candidate_step: int = 1

    min_suffix_tokens: int = 5

    # "example_mean" = mean_i replaceable_fraction_i
    # "token_weighted" = total replaceable tokens / total tokens
    aggregation: str = "example_mean"


@dataclass(frozen=True)
class SeedPolicy:
    reference_tokenizer: str
    seed_length: int

    target_generated_fraction: float
    realized_generated_fraction: float
    realized_token_weighted_fraction: float

    min_suffix_tokens: int

    num_examples: int
    num_eligible: int
    eligible_fraction: float

    mean_reference_tokens: float
    median_reference_tokens: float


def load_reference_tokenizer(
    name_or_path: str,
) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        use_fast=True,
    )

    if not getattr(
        tokenizer,
        "is_fast",
        False,
    ):
        raise ValueError(
            "Seed splitting requires a fast tokenizer with offsets."
        )

    return tokenizer


def compute_token_lengths(
    examples: Sequence[Example],
    tokenizer: PreTrainedTokenizerBase,
) -> np.ndarray:
    lengths = []

    for example in examples:
        ids = tokenizer(
            example.text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        lengths.append(
            len(ids)
        )

    return np.asarray(
        lengths,
        dtype=np.int64,
    )


def validate_seed_config(
    config: SeedPolicyConfig,
) -> None:
    if not (
        0.0
        < config.target_generated_fraction
        < 1.0
    ):
        raise ValueError(
            "target_generated_fraction must be between 0 and 1."
        )

    if config.min_seed_tokens <= 0:
        raise ValueError(
            "min_seed_tokens must be > 0."
        )

    if (
        config.max_seed_tokens
        < config.min_seed_tokens
    ):
        raise ValueError(
            "max_seed_tokens must be >= min_seed_tokens."
        )

    if config.candidate_step <= 0:
        raise ValueError(
            "candidate_step must be > 0."
        )

    if config.min_suffix_tokens <= 0:
        raise ValueError(
            "min_suffix_tokens must be > 0."
        )

    if config.aggregation not in {
        "example_mean",
        "token_weighted",
    }:
        raise ValueError(
            "aggregation must be 'example_mean' or 'token_weighted'."
        )


def _candidate_stats(
    lengths: np.ndarray,
    *,
    seed_length: int,
    min_suffix_tokens: int,
) -> dict[str, Any]:
    eligible = (
        lengths
        >= (
            seed_length
            + min_suffix_tokens
        )
    )

    replaceable = np.where(
        eligible,
        lengths - seed_length,
        0,
    ).astype(
        np.float64
    )

    safe_lengths = np.maximum(
        lengths,
        1,
    ).astype(
        np.float64
    )

    example_fractions = (
        replaceable
        / safe_lengths
    )

    token_total = float(
        lengths.sum()
    )

    token_weighted = (
        float(
            replaceable.sum()
        )
        / token_total
        if token_total > 0
        else 0.0
    )

    return {
        "seed_length": int(
            seed_length
        ),
        "num_eligible": int(
            eligible.sum()
        ),
        "eligible_fraction": float(
            eligible.mean()
        )
        if len(eligible)
        else 0.0,
        "example_mean_generated_fraction": float(
            example_fractions.mean()
        )
        if len(example_fractions)
        else 0.0,
        "token_weighted_generated_fraction": float(
            token_weighted
        ),
    }


def seed_length_sweep(
    *,
    examples: Sequence[Example],
    config: SeedPolicyConfig,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> list[dict[str, Any]]:
    validate_seed_config(
        config
    )

    if tokenizer is None:
        tokenizer = (
            load_reference_tokenizer(
                config.reference_tokenizer
            )
        )

    lengths = compute_token_lengths(
        examples,
        tokenizer,
    )

    return [
        _candidate_stats(
            lengths,
            seed_length=k,
            min_suffix_tokens=(
                config.min_suffix_tokens
            ),
        )
        for k in range(
            config.min_seed_tokens,
            config.max_seed_tokens + 1,
            config.candidate_step,
        )
    ]


def fit_seed_policy(
    *,
    examples: Sequence[Example],
    config: SeedPolicyConfig,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> SeedPolicy:
    if not examples:
        raise ValueError(
            "Cannot fit seed policy on an empty dataset."
        )

    if tokenizer is None:
        tokenizer = (
            load_reference_tokenizer(
                config.reference_tokenizer
            )
        )

    lengths = compute_token_lengths(
        examples,
        tokenizer,
    )

    sweep = seed_length_sweep(
        examples=examples,
        config=config,
        tokenizer=tokenizer,
    )

    metric_key = (
        "example_mean_generated_fraction"
        if config.aggregation
        == "example_mean"
        else "token_weighted_generated_fraction"
    )

    best = min(
        sweep,
        key=lambda row: (
            abs(
                float(
                    row[
                        metric_key
                    ]
                )
                - config.target_generated_fraction
            ),
            # Deterministic tie-breaker: prefer the longer human prefix.
            -int(
                row[
                    "seed_length"
                ]
            ),
        ),
    )

    return SeedPolicy(
        reference_tokenizer=(
            config.reference_tokenizer
        ),
        seed_length=int(
            best[
                "seed_length"
            ]
        ),
        target_generated_fraction=(
            config.target_generated_fraction
        ),
        realized_generated_fraction=float(
            best[
                "example_mean_generated_fraction"
            ]
        ),
        realized_token_weighted_fraction=float(
            best[
                "token_weighted_generated_fraction"
            ]
        ),
        min_suffix_tokens=(
            config.min_suffix_tokens
        ),
        num_examples=len(
            examples
        ),
        num_eligible=int(
            best[
                "num_eligible"
            ]
        ),
        eligible_fraction=float(
            best[
                "eligible_fraction"
            ]
        ),
        mean_reference_tokens=float(
            lengths.mean()
        ),
        median_reference_tokens=float(
            np.median(
                lengths
            )
        ),
    )


def _split_at_token_boundary(
    text: str,
    *,
    seed_length: int,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[str, str, int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )

    offsets = encoded[
        "offset_mapping"
    ]

    n_tokens = len(
        offsets
    )

    if seed_length >= n_tokens:
        return (
            text,
            "",
            n_tokens,
        )

    char_end = int(
        offsets[
            seed_length - 1
        ][1]
    )

    return (
        text[:char_end],
        text[char_end:],
        n_tokens,
    )


def apply_seed_policy(
    *,
    examples: Sequence[Example],
    policy: SeedPolicy,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> list[Example]:
    if tokenizer is None:
        tokenizer = (
            load_reference_tokenizer(
                policy.reference_tokenizer
            )
        )

    output = []

    for example in examples:
        (
            prefix,
            suffix,
            n_tokens,
        ) = _split_at_token_boundary(
            example.text,
            seed_length=(
                policy.seed_length
            ),
            tokenizer=tokenizer,
        )

        eligible = (
            n_tokens
            >= (
                policy.seed_length
                + policy.min_suffix_tokens
            )
        )

        if eligible:
            suffix_tokens = (
                n_tokens
                - policy.seed_length
            )

            prefix_text = prefix
            human_suffix = suffix
        else:
            suffix_tokens = 0
            prefix_text = example.text
            human_suffix = ""

        metadata = dict(
            example.metadata
        )

        metadata.update(
            {
                "reference_tokenizer": (
                    policy.reference_tokenizer
                ),
                "reference_num_tokens": (
                    n_tokens
                ),
                "seed_length": (
                    policy.seed_length
                ),
                "seed_eligible": (
                    eligible
                ),
                "human_suffix_tokens": (
                    suffix_tokens
                ),
                "human_suffix_fraction": (
                    suffix_tokens
                    / n_tokens
                    if n_tokens
                    else 0.0
                ),
            }
        )

        output.append(
            Example(
                id=example.id,
                text=example.text,
                metadata=metadata,
                prefix_text=(
                    prefix_text
                ),
                human_suffix=(
                    human_suffix
                ),
                synthetic_suffix=None,
                source="human",
            )
        )

    return output


def save_seed_policy(
    policy: SeedPolicy,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            asdict(policy),
            f,
            indent=2,
            ensure_ascii=False,
        )


def save_seed_sweep(
    sweep: Sequence[dict[str, Any]],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            list(sweep),
            f,
            indent=2,
            ensure_ascii=False,
        )
