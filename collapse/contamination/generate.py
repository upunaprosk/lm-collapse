from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from collapse.records import Example

@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.9
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    max_new_tokens: int = 400
    min_new_tokens: int = 1

    batch_size: int = 32
    generation_prompt: str = ""

    match_human_suffix_length: bool = False
    length_tolerance: float = 0.25
    buffer_tokens: int = 15
    generation_seed: int = 42
    torch_dtype: str = "auto"
    device_map: str | None = "auto"
    add_special_tokens: bool = False
    max_prompt_tokens: int | None = None
    keep_ineligible_human: bool = True


@dataclass
class GenerationStats:
    model_name_or_path: str
    generation_seed: int

    num_examples: int
    num_eligible: int
    num_generated: int
    num_kept_human: int
    num_failed: int

    mean_prefix_tokens: float | None
    mean_generated_tokens: float | None
    median_generated_tokens: float | None

    min_generated_tokens: int | None
    max_generated_tokens: int | None

    mean_target_suffix_tokens: float | None
    median_target_suffix_tokens: float | None
    mean_generated_to_target_ratio: float | None

    empty_generation_rate: float
    length_window_failure_rate: float
    eos_terminated_rate: float
    sentence_boundary_rate: float
    max_length_stop_rate: float


@dataclass
class GenerationResult:
    examples: list[Example]
    stats: GenerationStats


def seed_generation(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(value: str):
    if value == "auto":
        return "auto"

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    if value not in mapping:
        raise ValueError(
            f"Unknown torch_dtype={value!r}. "
            f"Supported values: {sorted(mapping)} plus 'auto'."
        )

    return mapping[value]


def load_generation_tokenizer(
    model_name_or_path: str,
) -> PreTrainedTokenizerBase:
    """
    Load the tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                f"Tokenizer for {model_name_or_path!r} has neither a "
                "pad_token_id nor eos_token_id."
            )

        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_generation_model(
    model_name_or_path: str,
    config: GenerationConfig,
) -> PreTrainedModel:
    kwargs: dict[str, Any] = {
        "dtype": resolve_torch_dtype(
            config.torch_dtype
        ),
    }

    if config.device_map is not None:
        kwargs["device_map"] = config.device_map

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **kwargs,
    )

    model.eval()

    return model

def batched(
    items: Sequence[Example],
    batch_size: int,
) -> Iterator[Sequence[Example]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    for start in range(
        0,
        len(items),
        batch_size,
    ):
        yield items[
            start : start + batch_size
        ]


def get_model_device(
    model: PreTrainedModel,
) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as e:
        raise RuntimeError(
            "Could not determine model device."
        ) from e


def validate_generation_inputs(
    examples: Sequence[Example],
) -> None:
    missing = []

    for example in examples:
        eligible = example.metadata.get(
            "seed_eligible",
            False,
        )

        if eligible and example.prefix_text is None:
            missing.append(example.id)

        if len(missing) >= 10:
            break

    if missing:
        raise ValueError(
            "Eligible examples are missing prefix_text. "
            "Run scripts/prepare_dataset.py first.\n"
            "Examples: "
            + ", ".join(missing)
        )


def eligible_examples(
    examples: Sequence[Example],
) -> list[Example]:
    return [
        example
        for example in examples
        if example.metadata.get(
            "seed_eligible",
            False,
        )
    ]


def _prompt_token_lengths(
    prompts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    add_special_tokens: bool,
) -> list[int]:
    encoded = tokenizer(
        list(prompts),
        add_special_tokens=add_special_tokens,
        padding=False,
        truncation=False,
    )

    return [
        len(ids)
        for ids in encoded["input_ids"]
    ]

@torch.inference_mode()
def generate_batch(
    examples: Sequence[Example],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: GenerationConfig,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Generate one continuation for each canonical prefix.
    """

    if not examples:
        return [], []

    canonical_prefixes = [
        example.prefix_text
        for example in examples
    ]
    human_suffixes = [
        example.human_suffix
        for example in examples
    ]

    if any(prefix is None for prefix in canonical_prefixes):
        raise ValueError(
            "generate_batch received an example with prefix_text=None."
        )

    if (
        config.match_human_suffix_length
        and any(suffix is None for suffix in human_suffixes)
    ):
        raise ValueError(
            "Length-controlled generation requires human_suffix on every "
            "eligible prepared example."
        )

    canonical_prefixes = [str(x) for x in canonical_prefixes]
    human_suffixes = [str(x or "") for x in human_suffixes]

    model_prompts = [
        config.generation_prompt + prefix
        for prefix in canonical_prefixes
    ]

    native_seed_lengths = _prompt_token_lengths(
        canonical_prefixes,
        tokenizer,
        add_special_tokens=False,
    )
    native_prompt_lengths = _prompt_token_lengths(
        model_prompts,
        tokenizer,
        add_special_tokens=config.add_special_tokens,
    )

    suffix_encoded = tokenizer(
        human_suffixes,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]
    target_suffix_lengths = [len(ids) for ids in suffix_encoded]

    if config.match_human_suffix_length:
        if not (0.0 <= config.length_tolerance < 1.0):
            raise ValueError("length_tolerance must be in [0, 1).")
        if config.buffer_tokens < 0:
            raise ValueError("buffer_tokens must be >= 0.")

        min_lengths = [
            max(
                config.min_new_tokens,
                int(target * (1.0 - config.length_tolerance)),
            )
            for target in target_suffix_lengths
        ]
        max_lengths = [
            min(
                config.max_new_tokens,
                max(
                    min_len,
                    int(target * (1.0 + config.length_tolerance)),
                ),
            )
            for target, min_len in zip(target_suffix_lengths, min_lengths)
        ]
        batch_max_new_tokens = min(
            config.max_new_tokens,
            max(max_lengths) + config.buffer_tokens,
        )
    else:
        min_lengths = [config.min_new_tokens] * len(examples)
        max_lengths = [config.max_new_tokens] * len(examples)
        batch_max_new_tokens = config.max_new_tokens

    device = get_model_device(model)

    tokenizer_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "add_special_tokens": config.add_special_tokens,
    }

    if config.max_prompt_tokens is not None:
        tokenizer_kwargs.update(
            {
                "truncation": True,
                "max_length": config.max_prompt_tokens,
            }
        )
    else:
        tokenizer_kwargs["truncation"] = False

    inputs = tokenizer(
        model_prompts,
        **tokenizer_kwargs,
    )
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    input_width = inputs["input_ids"].shape[1]

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": batch_max_new_tokens,
        "min_new_tokens": config.min_new_tokens,
        "do_sample": True,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "repetition_penalty": config.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    outputs = model.generate(
        **inputs,
        **generation_kwargs,
    )

    continuations: list[str] = []
    metadata: list[dict[str, Any]] = []

    for (
        output_ids,
        seed_len,
        prompt_len,
        target_len,
        min_len,
        max_len,
    ) in zip(
        outputs,
        native_seed_lengths,
        native_prompt_lengths,
        target_suffix_lengths,
        min_lengths,
        max_lengths,
    ):
        raw_ids = output_ids[input_width:].tolist()
        raw_generated_tokens = len(raw_ids)

        eos_id = tokenizer.eos_token_id
        first_eos = None
        if eos_id is not None and eos_id in raw_ids:
            first_eos = raw_ids.index(eos_id)

        stop_reason = "natural_or_cap"
        length_window_failure = False

        if config.match_human_suffix_length:
            effective_raw_len = (
                first_eos
                if first_eos is not None
                else raw_generated_tokens
            )

            if effective_raw_len < min_len:
                chosen_ids: list[int] = []
                stop_reason = "too_short"
                length_window_failure = True
            else:
                upper = min(max_len, effective_raw_len)
                stop = None
                if (
                    first_eos is not None
                    and min_len <= first_eos <= upper
                ):
                    stop = first_eos
                    stop_reason = "eos"
                if stop is None:
                    for end_pos in range(min_len, upper + 1):
                        candidate = tokenizer.decode(
                            raw_ids[:end_pos],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        if candidate.rstrip().endswith((".", "!", "?")):
                            stop = end_pos
                            stop_reason = "sentence_boundary"
                            break

                if stop is None:
                    stop = upper
                    stop_reason = "max_length"

                chosen_ids = raw_ids[:stop]
        else:
            if first_eos is not None:
                chosen_ids = raw_ids[:first_eos]
                stop_reason = "eos"
            else:
                chosen_ids = raw_ids
                if raw_generated_tokens >= config.max_new_tokens:
                    stop_reason = "max_length"

        text = tokenizer.decode(
            chosen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        generation_empty = len(text.strip()) == 0

        continuations.append(text)
        metadata.append(
            {
                "native_prefix_tokens": int(seed_len),
                "native_generation_prompt_tokens": int(prompt_len),
                "native_target_suffix_tokens": int(target_len),
                "min_target_tokens": int(min_len),
                "max_target_tokens": int(max_len),
                "raw_generated_tokens": int(raw_generated_tokens),
                "generated_tokens": int(len(chosen_ids)),
                "stop_reason": stop_reason,
                "eos_terminated": stop_reason == "eos",
                "sentence_boundary_terminated": (
                    stop_reason == "sentence_boundary"
                ),
                "max_length_terminated": stop_reason == "max_length",
                "length_window_failure": length_window_failure,
                "generation_empty": generation_empty,
            }
        )

    return continuations, metadata

def generate_seeded_dataset(
    *,
    examples: Sequence[Example],
    model_name_or_path: str,
    iteration: int,
    config: GenerationConfig,
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> GenerationResult:
    """
    Generate synthetic suffixes for every eligible example.
    """

    validate_generation_inputs(
        examples
    )

    seed_generation(
        config.generation_seed
    )

    owns_model = model is None
    owns_tokenizer = tokenizer is None

    if tokenizer is None:
        tokenizer = load_generation_tokenizer(
            model_name_or_path
        )

    if model is None:
        model = load_generation_model(
            model_name_or_path,
            config,
        )

    eligible = eligible_examples(
        examples
    )

    if config.match_human_suffix_length:
        eligible = sorted(
            eligible,
            key=lambda example: int(
                example.metadata.get("human_suffix_tokens", 0)
            ),
        )
    generated_by_id: dict[
        str,
        tuple[str, dict[str, Any]]
    ] = {}

    total_batches = math.ceil(
        len(eligible) / config.batch_size
    ) if eligible else 0

    for batch in tqdm(
        batched(
            eligible,
            config.batch_size,
        ),
        total=total_batches,
        desc=(
            f"Generate iteration {iteration}"
        ),
    ):
        continuations, batch_meta = (
            generate_batch(
                examples=batch,
                model=model,
                tokenizer=tokenizer,
                config=config,
            )
        )

        for example, continuation, meta in zip(
            batch,
            continuations,
            batch_meta,
        ):
            generated_by_id[
                example.id
            ] = (
                continuation,
                meta,
            )

    result_examples: list[Example] = []

    generated_token_lengths: list[int] = []
    prefix_token_lengths: list[int] = []
    target_suffix_token_lengths: list[int] = []
    generated_to_target_ratios: list[float] = []

    num_generated = 0
    num_kept_human = 0
    num_failed = 0
    num_length_window_failed = 0
    num_eos_terminated = 0
    num_sentence_boundary_terminated = 0
    num_max_length_terminated = 0

    for example in examples:

        is_eligible = example.metadata.get(
            "seed_eligible",
            False,
        )

        if not is_eligible:
            if not config.keep_ineligible_human:
                continue

            out = deepcopy(example)

            out.source = "human"

            out.metadata = dict(
                out.metadata
            )

            out.metadata.update(
                {
                    "iteration": iteration,
                    "generation_model": (
                        model_name_or_path
                    ),
                    "generation_seed": (
                        config.generation_seed
                    ),
                    "generation_status": (
                        "ineligible_kept_human"
                    ),
                }
            )

            result_examples.append(out)
            num_kept_human += 1
            continue

        if example.id not in generated_by_id:
            raise RuntimeError(
                f"Eligible example {example.id!r} "
                "was not generated."
            )

        continuation, gen_meta = (
            generated_by_id[
                example.id
            ]
        )

        prefix_token_lengths.append(
            gen_meta[
                "native_prefix_tokens"
            ]
        )

        generated_token_lengths.append(
            gen_meta[
                "generated_tokens"
            ]
        )

        target_len = int(
            gen_meta.get("native_target_suffix_tokens", 0)
        )
        target_suffix_token_lengths.append(target_len)
        if target_len > 0 and gen_meta["generated_tokens"] > 0:
            generated_to_target_ratios.append(
                gen_meta["generated_tokens"] / target_len
            )

        if gen_meta.get("length_window_failure", False):
            num_length_window_failed += 1
        if gen_meta["eos_terminated"]:
            num_eos_terminated += 1
        if gen_meta.get("sentence_boundary_terminated", False):
            num_sentence_boundary_terminated += 1
        if gen_meta.get("max_length_terminated", False):
            num_max_length_terminated += 1

        if (
            gen_meta["generation_empty"]
            or gen_meta.get("length_window_failure", False)
        ):
            out = deepcopy(example)

            out.source = "human"

            out.metadata = dict(
                out.metadata
            )

            out.metadata.update(
                {
                    "iteration": iteration,
                    "generation_model": (
                        model_name_or_path
                    ),
                    "generation_seed": (
                        config.generation_seed
                    ),
                    "generation_status": (
                        "failed_generation_kept_human"
                    ),
                    **gen_meta,
                }
            )

            result_examples.append(out)

            num_failed += 1
            num_kept_human += 1
            continue

        out = deepcopy(example)

        out.synthetic_suffix = (
            continuation
        )

        out.text = (
            str(example.prefix_text)
            + continuation
        )

        out.source = "synthetic"

        out.metadata = dict(
            out.metadata
        )

        out.metadata.update(
            {
                "iteration": iteration,
                "generation_model": (
                    model_name_or_path
                ),
                "generation_seed": (
                    config.generation_seed
                ),
                "generation_status": (
                    "generated"
                ),
                "temperature": (
                    config.temperature
                ),
                "top_p": (
                    config.top_p
                ),
                "repetition_penalty": (
                    config.repetition_penalty
                ),
                "max_new_tokens": (
                    config.max_new_tokens
                ),
                "generation_prompt": config.generation_prompt,
                "match_human_suffix_length": (
                    config.match_human_suffix_length
                ),
                "length_tolerance": config.length_tolerance,
                "buffer_tokens": config.buffer_tokens,
                **gen_meta,
            }
        )

        result_examples.append(out)
        num_generated += 1

    num_eligible = len(
        eligible
    )

    stats = GenerationStats(
        model_name_or_path=(
            model_name_or_path
        ),
        generation_seed=(
            config.generation_seed
        ),

        num_examples=len(examples),
        num_eligible=num_eligible,
        num_generated=num_generated,
        num_kept_human=(
            num_kept_human
        ),
        num_failed=num_failed,

        mean_prefix_tokens=(
            float(
                np.mean(
                    prefix_token_lengths
                )
            )
            if prefix_token_lengths
            else None
        ),

        mean_generated_tokens=(
            float(
                np.mean(
                    generated_token_lengths
                )
            )
            if generated_token_lengths
            else None
        ),

        median_generated_tokens=(
            float(
                np.median(
                    generated_token_lengths
                )
            )
            if generated_token_lengths
            else None
        ),

        min_generated_tokens=(
            int(
                min(
                    generated_token_lengths
                )
            )
            if generated_token_lengths
            else None
        ),

        max_generated_tokens=(
            int(
                max(
                    generated_token_lengths
                )
            )
            if generated_token_lengths
            else None
        ),

        mean_target_suffix_tokens=(
            float(np.mean(target_suffix_token_lengths))
            if target_suffix_token_lengths
            else None
        ),
        median_target_suffix_tokens=(
            float(np.median(target_suffix_token_lengths))
            if target_suffix_token_lengths
            else None
        ),
        mean_generated_to_target_ratio=(
            float(np.mean(generated_to_target_ratios))
            if generated_to_target_ratios
            else None
        ),

        empty_generation_rate=(
            num_failed / num_eligible
            if num_eligible
            else 0.0
        ),
        length_window_failure_rate=(
            num_length_window_failed / num_eligible
            if num_eligible
            else 0.0
        ),

        eos_terminated_rate=(
            num_eos_terminated
            / num_eligible
            if num_eligible
            else 0.0
        ),
        sentence_boundary_rate=(
            num_sentence_boundary_terminated / num_eligible
            if num_eligible
            else 0.0
        ),
        max_length_stop_rate=(
            num_max_length_terminated / num_eligible
            if num_eligible
            else 0.0
        ),
    )

    if owns_model:
        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if owns_tokenizer:
        del tokenizer

    return GenerationResult(
        examples=result_examples,
        stats=stats,
    )

def example_to_dict(
    example: Example,
) -> dict[str, Any]:
    return asdict(example)


def save_generation_result(
    result: GenerationResult,
    output_dir: str | Path,
) -> None:

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    jsonl_path = (
        output_dir
        / "generated.jsonl"
    )

    with jsonl_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for example in result.examples:
            f.write(
                json.dumps(
                    example_to_dict(
                        example
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )

    stats_path = (
        output_dir
        / "generation_stats.json"
    )

    with stats_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            asdict(result.stats),
            f,
            indent=2,
            ensure_ascii=False,
        )
