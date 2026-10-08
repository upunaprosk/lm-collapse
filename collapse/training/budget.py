from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from collapse.records import Example


@dataclass(frozen=True)
class TrainingBudgetConfig:

    cutoff_len: int = 512

    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    world_size: int = 1

    corpus_equivalents: float = 1.0

    add_special_tokens: bool = False

    add_eos_per_document: bool = True


@dataclass(frozen=True)
class TrainingBudget:
    tokenizer_name_or_path: str

    cutoff_len: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    world_size: int
    corpus_equivalents: float

    num_reference_examples: int

    reference_content_tokens: int
    reference_separator_tokens: int
    reference_training_tokens: int

    target_training_tokens: int

    tokens_per_microbatch: int
    microbatches_per_optimizer_step: int
    tokens_per_optimizer_step: int

    max_steps: int
    nominal_training_tokens: int

    token_overshoot: int
    token_overshoot_fraction: float


@dataclass(frozen=True)
class DatasetBudgetReport:
    """
    Compare a dataset to the human reference.
    """

    num_examples: int

    content_tokens: int
    separator_tokens: int
    available_training_tokens: int

    target_training_tokens: int
    nominal_training_tokens: int

    expected_dataset_passes: float | None

    available_vs_human_reference: float | None


def load_training_tokenizer(
    tokenizer_name_or_path: str,
) -> PreTrainedTokenizerBase:
    """
    Load the tokenizer.
    """
    return AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=True,
    )


def count_example_tokens(
    example: Example,
    tokenizer: PreTrainedTokenizerBase,
    *,
    add_special_tokens: bool = False,
) -> int:
    return len(
        tokenizer(
            example.text,
            add_special_tokens=add_special_tokens,
            truncation=False,
        )["input_ids"]
    )


def count_dataset_tokens(
    examples: Sequence[Example],
    tokenizer: PreTrainedTokenizerBase,
    *,
    add_special_tokens: bool = False,
    add_eos_per_document: bool = True,
) -> tuple[int, int, int]:

    content_tokens = 0

    for example in examples:
        content_tokens += count_example_tokens(
            example,
            tokenizer,
            add_special_tokens=add_special_tokens,
        )

    separator_tokens = 0

    if add_eos_per_document:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "add_eos_per_document=True, but the tokenizer has no "
                "eos_token_id. Disable EOS accounting or use a tokenizer "
                "with an EOS token."
            )

        separator_tokens = len(examples)

    total = content_tokens + separator_tokens

    return (
        int(content_tokens),
        int(separator_tokens),
        int(total),
    )


def validate_budget_config(
    config: TrainingBudgetConfig,
) -> None:
    if config.cutoff_len <= 0:
        raise ValueError("cutoff_len must be > 0.")

    if config.per_device_train_batch_size <= 0:
        raise ValueError(
            "per_device_train_batch_size must be > 0."
        )

    if config.gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be > 0."
        )

    if config.world_size <= 0:
        raise ValueError("world_size must be > 0.")

    if config.corpus_equivalents <= 0:
        raise ValueError(
            "corpus_equivalents must be > 0."
        )

def fit_training_budget(
    *,
    human_examples: Sequence[Example],
    tokenizer_name_or_path: str,
    config: TrainingBudgetConfig,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> TrainingBudget:

    validate_budget_config(
        config
    )

    if not human_examples:
        raise ValueError(
            "Cannot fit a training budget from an empty human dataset."
        )

    if tokenizer is None:
        tokenizer = load_training_tokenizer(
            tokenizer_name_or_path
        )

    (
        content_tokens,
        separator_tokens,
        reference_training_tokens,
    ) = count_dataset_tokens(
        human_examples,
        tokenizer,
        add_special_tokens=(
            config.add_special_tokens
        ),
        add_eos_per_document=(
            config.add_eos_per_document
        ),
    )

    target_training_tokens = int(
        math.ceil(
            reference_training_tokens
            * config.corpus_equivalents
        )
    )

    tokens_per_microbatch = (
        config.cutoff_len
        * config.per_device_train_batch_size
        * config.world_size
    )

    microbatches_per_optimizer_step = (
        config.gradient_accumulation_steps
    )

    tokens_per_optimizer_step = (
        tokens_per_microbatch
        * microbatches_per_optimizer_step
    )

    max_steps = int(
        math.ceil(
            target_training_tokens
            / tokens_per_optimizer_step
        )
    )

    nominal_training_tokens = (
        max_steps
        * tokens_per_optimizer_step
    )

    token_overshoot = (
        nominal_training_tokens
        - target_training_tokens
    )

    token_overshoot_fraction = (
        token_overshoot
        / target_training_tokens
        if target_training_tokens > 0
        else 0.0
    )

    return TrainingBudget(
        tokenizer_name_or_path=(
            tokenizer_name_or_path
        ),

        cutoff_len=config.cutoff_len,
        per_device_train_batch_size=(
            config.per_device_train_batch_size
        ),
        gradient_accumulation_steps=(
            config.gradient_accumulation_steps
        ),
        world_size=config.world_size,
        corpus_equivalents=(
            config.corpus_equivalents
        ),

        num_reference_examples=len(
            human_examples
        ),

        reference_content_tokens=(
            content_tokens
        ),
        reference_separator_tokens=(
            separator_tokens
        ),
        reference_training_tokens=(
            reference_training_tokens
        ),

        target_training_tokens=(
            target_training_tokens
        ),

        tokens_per_microbatch=(
            tokens_per_microbatch
        ),
        microbatches_per_optimizer_step=(
            microbatches_per_optimizer_step
        ),
        tokens_per_optimizer_step=(
            tokens_per_optimizer_step
        ),

        max_steps=max_steps,
        nominal_training_tokens=(
            nominal_training_tokens
        ),

        token_overshoot=token_overshoot,
        token_overshoot_fraction=(
            token_overshoot_fraction
        ),
    )

def report_dataset_against_budget(
    *,
    examples: Sequence[Example],
    budget: TrainingBudget,
    tokenizer: PreTrainedTokenizerBase | None = None,
    add_special_tokens: bool = False,
    add_eos_per_document: bool = True,
) -> DatasetBudgetReport:

    if tokenizer is None:
        tokenizer = load_training_tokenizer(
            budget.tokenizer_name_or_path
        )

    (
        content_tokens,
        separator_tokens,
        available_training_tokens,
    ) = count_dataset_tokens(
        examples,
        tokenizer,
        add_special_tokens=(
            add_special_tokens
        ),
        add_eos_per_document=(
            add_eos_per_document
        ),
    )

    expected_dataset_passes = None

    if available_training_tokens > 0:
        expected_dataset_passes = (
            budget.nominal_training_tokens
            / available_training_tokens
        )

    available_vs_human_reference = None

    if budget.reference_training_tokens > 0:
        available_vs_human_reference = (
            available_training_tokens
            / budget.reference_training_tokens
        )

    return DatasetBudgetReport(
        num_examples=len(examples),

        content_tokens=content_tokens,
        separator_tokens=separator_tokens,
        available_training_tokens=(
            available_training_tokens
        ),

        target_training_tokens=(
            budget.target_training_tokens
        ),
        nominal_training_tokens=(
            budget.nominal_training_tokens
        ),

        expected_dataset_passes=(
            float(expected_dataset_passes)
            if expected_dataset_passes
            is not None
            else None
        ),

        available_vs_human_reference=(
            float(
                available_vs_human_reference
            )
            if available_vs_human_reference
            is not None
            else None
        ),
    )


def llamafactory_budget_overrides(
    budget: TrainingBudget,
) -> dict[str, Any]:

    return {
        "cutoff_len": (
            budget.cutoff_len
        ),
        "per_device_train_batch_size": (
            budget.per_device_train_batch_size
        ),
        "gradient_accumulation_steps": (
            budget.gradient_accumulation_steps
        ),
        "max_steps": (
            budget.max_steps
        ),
        "packing": True,
    }


def save_training_budget(
    budget: TrainingBudget,
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
            asdict(budget),
            f,
            indent=2,
            ensure_ascii=False,
        )


def load_training_budget(
    path: str | Path,
) -> TrainingBudget:
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return TrainingBudget(
        **data
    )


def save_dataset_budget_report(
    report: DatasetBudgetReport,
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
            asdict(report),
            f,
            indent=2,
            ensure_ascii=False,
        )
