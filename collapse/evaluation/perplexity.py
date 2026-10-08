from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from collapse.records import Example


@dataclass(frozen=True)
class PerplexityConfig:
    """
    Held-out human perplexity evaluation.
    """

    batch_size: int = 8

    max_length: int | None = None
    stride: int | None = None

    add_special_tokens: bool = False
    add_eos_per_document: bool = True

    torch_dtype: str = "auto"
    device_map: str | None = "auto"

    max_examples: int | None = None


@dataclass(frozen=True)
class ExamplePerplexity:
    id: str
    nll_sum: float
    num_scored_tokens: int
    mean_nll: float | None
    perplexity: float | None


@dataclass(frozen=True)
class PerplexityResult:
    model_name_or_path: str

    num_examples: int
    num_examples_scored: int
    num_scored_tokens: int

    nll_sum: float
    mean_nll: float
    perplexity: float

    max_length: int
    stride: int

    add_special_tokens: bool
    add_eos_per_document: bool

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
            f"Supported: {sorted(mapping)} plus 'auto'."
        )

    return mapping[value]


def load_tokenizer(
    model_name_or_path: str,
) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )

    tokenizer.padding_side = "right"

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                f"Tokenizer for {model_name_or_path!r} has neither "
                "pad_token_id nor eos_token_id."
            )
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(
    model_name_or_path: str,
    config: PerplexityConfig,
) -> PreTrainedModel:
    kwargs: dict[str, Any] = {
        "torch_dtype": resolve_torch_dtype(
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


def get_model_device(
    model: PreTrainedModel,
) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as e:
        raise RuntimeError(
            "Could not determine model device."
        ) from e

def infer_max_length(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    requested: int | None,
) -> int:
    if requested is not None:
        if requested < 2:
            raise ValueError(
                "Perplexity max_length must be >= 2."
            )
        return int(requested)

    candidates: list[int] = []

    model_value = getattr(
        model.config,
        "max_position_embeddings",
        None,
    )

    if (
        isinstance(model_value, int)
        and 2 <= model_value < 1_000_000
    ):
        candidates.append(
            model_value
        )

    tokenizer_value = getattr(
        tokenizer,
        "model_max_length",
        None,
    )

    if (
        isinstance(tokenizer_value, int)
        and 2 <= tokenizer_value < 1_000_000
    ):
        candidates.append(
            tokenizer_value
        )

    if not candidates:
        return 2048

    return min(candidates)


def resolve_stride(
    max_length: int,
    requested: int | None,
) -> int:
    if requested is None:
        return max(
            1,
            max_length // 2,
        )

    requested = int(
        requested
    )

    if requested <= 0:
        raise ValueError(
            "stride must be > 0."
        )

    if requested > max_length:
        raise ValueError(
            "stride must be <= max_length."
        )

    return requested


def encode_document(
    example: Example,
    tokenizer: PreTrainedTokenizerBase,
    config: PerplexityConfig,
) -> list[int]:
    ids = tokenizer(
        example.text,
        add_special_tokens=(
            config.add_special_tokens
        ),
        truncation=False,
    )["input_ids"]

    ids = list(ids)

    if config.add_eos_per_document:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "add_eos_per_document=True but tokenizer has no EOS token."
            )

        ids.append(
            tokenizer.eos_token_id
        )

    return ids


@dataclass
class Window:
    example_index: int
    input_ids: list[int]
    score_mask: list[bool]


def document_windows(
    *,
    example_index: int,
    token_ids: Sequence[int],
    max_length: int,
    stride: int,
) -> list[Window]:
    """
    Sliding-window causal-LM evaluation.
    """
    if len(token_ids) < 2:
        return []

    windows: list[Window] = []

    prev_end = 0

    for begin in range(
        0,
        len(token_ids),
        stride,
    ):
        end = min(
            begin + max_length,
            len(token_ids),
        )

        if end <= prev_end:
            continue

        ids = list(
            token_ids[begin:end]
        )

        target_len = (
            end - prev_end
        )

        target_start = max(
            0,
            len(ids) - target_len,
        )

        mask = [
            position >= target_start
            for position in range(
                len(ids)
            )
        ]

        windows.append(
            Window(
                example_index=example_index,
                input_ids=ids,
                score_mask=mask,
            )
        )

        prev_end = end

        if end == len(token_ids):
            break

    return windows

def batched(
    items: Sequence[Window],
    batch_size: int,
) -> Iterator[Sequence[Window]]:
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be > 0."
        )

    for start in range(
        0,
        len(items),
        batch_size,
    ):
        yield items[
            start : start + batch_size
        ]


def collate_windows(
    windows: Sequence[Window],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    max_len = max(
        len(window.input_ids)
        for window in windows
    )

    batch_size = len(
        windows
    )

    input_ids = torch.full(
        (batch_size, max_len),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
        device=device,
    )

    score_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.bool,
        device=device,
    )

    for row, window in enumerate(
        windows
    ):
        length = len(
            window.input_ids
        )

        input_ids[
            row,
            :length,
        ] = torch.tensor(
            window.input_ids,
            dtype=torch.long,
            device=device,
        )

        attention_mask[
            row,
            :length,
        ] = 1

        score_mask[
            row,
            :length,
        ] = torch.tensor(
            window.score_mask,
            dtype=torch.bool,
            device=device,
        )

    return (
        input_ids,
        attention_mask,
        score_mask,
    )


@torch.inference_mode()
def evaluate_fixed_human_perplexity(
    *,
    examples: Sequence[Example],
    model_name_or_path: str,
    config: PerplexityConfig,
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> tuple[
    PerplexityResult,
    list[ExamplePerplexity],
]:
    """
    Evaluate a model on a fixed human-written corpus.
    """

    if not examples:
        raise ValueError(
            "Perplexity evaluation received an empty dataset."
        )

    selected_examples = list(
        examples
    )

    if config.max_examples is not None:
        if config.max_examples <= 0:
            raise ValueError(
                "max_examples must be > 0."
            )

        selected_examples = (
            selected_examples[
                : config.max_examples
            ]
        )

    owns_model = model is None
    owns_tokenizer = tokenizer is None

    if tokenizer is None:
        tokenizer = load_tokenizer(
            model_name_or_path
        )

    if model is None:
        model = load_model(
            model_name_or_path,
            config,
        )

    max_length = infer_max_length(
        model,
        tokenizer,
        config.max_length,
    )

    stride = resolve_stride(
        max_length,
        config.stride,
    )

    windows: list[Window] = []

    for example_index, example in enumerate(
        selected_examples
    ):
        token_ids = encode_document(
            example,
            tokenizer,
            config,
        )

        windows.extend(
            document_windows(
                example_index=example_index,
                token_ids=token_ids,
                max_length=max_length,
                stride=stride,
            )
        )

    if not windows:
        raise ValueError(
            "No documents contain enough tokens to compute causal "
            "perplexity."
        )

    device = get_model_device(
        model
    )

    example_nll = [
        0.0
        for _ in selected_examples
    ]

    example_tokens = [
        0
        for _ in selected_examples
    ]

    total_nll = 0.0
    total_tokens = 0

    total_batches = math.ceil(
        len(windows)
        / config.batch_size
    )

    for batch in tqdm(
        batched(
            windows,
            config.batch_size,
        ),
        total=total_batches,
        desc="Held-out human PPL",
    ):
        (
            input_ids,
            attention_mask,
            score_mask,
        ) = collate_windows(
            batch,
            pad_token_id=(
                tokenizer.pad_token_id
            ),
            device=device,
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=(
                attention_mask
            ),
            use_cache=False,
        )

        logits = outputs.logits

        shift_logits = logits[
            :,
            :-1,
            :
        ].contiguous()

        shift_labels = input_ids[
            :,
            1:
        ].contiguous()

        valid_attention = (
            attention_mask[
                :,
                1:
            ].bool()
        )

        target_mask = (
            score_mask[
                :,
                1:
            ]
            & valid_attention
        )

        flat_loss = F.cross_entropy(
            shift_logits.float().view(
                -1,
                shift_logits.size(-1),
            ),
            shift_labels.view(-1),
            reduction="none",
        ).view_as(
            shift_labels
        )

        masked_loss = (
            flat_loss
            * target_mask
        )

        for row, window in enumerate(
            batch
        ):
            row_tokens = int(
                target_mask[
                    row
                ].sum().item()
            )

            if row_tokens == 0:
                continue

            row_nll = float(
                masked_loss[
                    row
                ].sum().item()
            )

            example_index = (
                window.example_index
            )

            example_nll[
                example_index
            ] += row_nll

            example_tokens[
                example_index
            ] += row_tokens

            total_nll += row_nll
            total_tokens += row_tokens

    if total_tokens == 0:
        raise RuntimeError(
            "No target tokens were scored."
        )

    mean_nll = (
        total_nll
        / total_tokens
    )

    perplexity = (
        math.exp(mean_nll)
        if mean_nll < 700
        else float("inf")
    )

    item_results: list[
        ExamplePerplexity
    ] = []

    for example, nll, num_tokens in zip(
        selected_examples,
        example_nll,
        example_tokens,
    ):
        if num_tokens > 0:
            item_mean = (
                nll / num_tokens
            )

            item_ppl = (
                math.exp(item_mean)
                if item_mean < 700
                else float("inf")
            )
        else:
            item_mean = None
            item_ppl = None

        item_results.append(
            ExamplePerplexity(
                id=example.id,
                nll_sum=float(nll),
                num_scored_tokens=int(
                    num_tokens
                ),
                mean_nll=(
                    float(item_mean)
                    if item_mean is not None
                    else None
                ),
                perplexity=(
                    float(item_ppl)
                    if item_ppl is not None
                    else None
                ),
            )
        )

    result = PerplexityResult(
        model_name_or_path=(
            model_name_or_path
        ),

        num_examples=len(
            selected_examples
        ),

        num_examples_scored=sum(
            item.num_scored_tokens > 0
            for item in item_results
        ),

        num_scored_tokens=(
            total_tokens
        ),

        nll_sum=float(
            total_nll
        ),

        mean_nll=float(
            mean_nll
        ),

        perplexity=float(
            perplexity
        ),

        max_length=max_length,
        stride=stride,

        add_special_tokens=(
            config.add_special_tokens
        ),

        add_eos_per_document=(
            config.add_eos_per_document
        ),
    )

    if owns_model:
        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if owns_tokenizer:
        del tokenizer

    return (
        result,
        item_results,
    )

def save_perplexity_result(
    *,
    result: PerplexityResult,
    items: Sequence[ExamplePerplexity],
    output_dir: str | Path,
) -> None:
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        output_dir
        / "perplexity.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            asdict(result),
            f,
            indent=2,
            ensure_ascii=False,
        )

    with (
        output_dir
        / "perplexity_items.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        for item in items:
            f.write(
                json.dumps(
                    asdict(item),
                    ensure_ascii=False,
                )
                + "\n"
            )
