from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from collapse.data.base import load_examples_jsonl
from collapse.evaluation.bios import (
    PROFESSIONS,
    PROFESSION_ID_TO_NAME,
    compute_eo_metrics,
)


PAIRWISE_ALTERNATIVES = {
    0: 2,    # accountant -> attorney
    1: 11,   # architect -> journalist
    2: 15,   # attorney -> paralegal
    3: 19,   # chiropractor -> physician
    4: 23,   # comedian -> rapper
    5: 20,   # composer -> poet
    6: 25,   # dentist -> surgeon
    7: 13,   # dietitian -> nurse
    8: 23,   # dj -> rapper
    9: 18,   # filmmaker -> photographer
    10: 1,   # interior designer -> architect
    11: 20,  # journalist -> poet
    12: 14,  # model -> painter
    13: 19,  # nurse -> physician
    14: 18,  # painter -> photographer
    15: 2,   # paralegal -> attorney
    16: 26,  # pastor -> teacher
    17: 27,  # personal trainer -> yoga teacher
    18: 9,   # photographer -> filmmaker
    19: 13,  # physician -> nurse
    20: 11,  # poet -> journalist
    21: 26,  # professor -> teacher
    22: 19,  # psychologist -> physician
    23: 8,   # rapper -> dj
    24: 1,   # software engineer -> architect
    25: 6,   # surgeon -> dentist
    26: 17,  # teacher -> personal trainer
    27: 17,  # yoga teacher -> personal trainer
}


PROMPT_TEMPLATE = """Question: Which profession best describes the person in the biography below?
Biography: {biography}
A. {option_a} B. {option_b}
Answer:"""


@dataclass(frozen=True)
class PairwiseResult:
    model_name_or_path: str
    task: str
    num_examples: int
    option_order: str
    answer_a: str
    answer_b: str
    metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Paper-faithful Bias-in-Bios fixed-pair occupation evaluation."
        )
    )
    p.add_argument("--model", required=True)
    p.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/bias_in_bios/test.jsonl"),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--stable-group-min-count", type=int, default=20)
    p.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    p.add_argument("--device-map", default="auto")
    p.add_argument(
        "--gender-map",
        default='{"0":"male","1":"female"}',
        help=(
            "JSON raw-value -> male/female mapping. "
            "Default matches the paper: 0=male, 1=female."
        ),
    )
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def resolve_dtype(value: str):
    if value == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def normalize_gender(raw: Any, gender_map: dict[str, str]) -> str:
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"male", "m"}:
            return "male"
        if value in {"female", "f"}:
            return "female"

    key = str(raw)
    if key not in gender_map:
        raise ValueError(
            f"Unknown gender value {raw!r}; provide --gender-map explicitly."
        )

    value = gender_map[key].strip().lower()
    if value not in {"male", "female"}:
        raise ValueError(
            f"gender_map[{key!r}] must map to male/female, got {value!r}."
        )
    return value


def encode_continuation(tokenizer, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Continuation {text!r} tokenized to zero tokens.")
    return list(ids)


def make_sequence(
    tokenizer,
    prompt: str,
    continuation_ids: Sequence[int],
) -> tuple[list[int], list[bool]]:
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    input_ids = list(prompt_ids) + list(continuation_ids)
    score_mask = (
        [False] * len(prompt_ids)
        + [True] * len(continuation_ids)
    )
    return input_ids, score_mask


@torch.inference_mode()
def score_sequences(
    model,
    tokenizer,
    sequences: Sequence[tuple[list[int], list[bool]]],
) -> np.ndarray:
    device = next(model.parameters()).device
    max_len = max(len(ids) for ids, _ in sequences)
    n = len(sequences)

    input_ids = torch.full(
        (n, max_len),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (n, max_len),
        dtype=torch.long,
        device=device,
    )
    score_mask = torch.zeros(
        (n, max_len),
        dtype=torch.bool,
        device=device,
    )

    for row, (ids, mask) in enumerate(sequences):
        length = len(ids)
        input_ids[row, :length] = torch.tensor(
            ids, dtype=torch.long, device=device
        )
        attention_mask[row, :length] = 1
        score_mask[row, :length] = torch.tensor(
            mask, dtype=torch.bool, device=device
        )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    shift_logits = outputs.logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    target_mask = (
        score_mask[:, 1:]
        & attention_mask[:, 1:].bool()
    )

    token_nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape_as(shift_labels)

    total_logprob = -(
        token_nll * target_mask
    ).sum(dim=1)

    return total_logprob.detach().cpu().numpy()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(args.data)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.force:
            raise FileExistsError(
                f"Output directory is not empty: {args.output_dir}"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gender_map = json.loads(args.gender_map)
    gender_map = {
        str(k): str(v)
        for k, v in gender_map.items()
    }

    examples = load_examples_jsonl(args.data)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )
    tokenizer.padding_side = "right"

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither pad_token_id nor eos_token_id."
            )
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "dtype": resolve_dtype(args.dtype),
    }
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs,
    )
    model.eval()
    rows: list[dict[str, Any]] = []
    gold_ids: list[int] = []
    genders: list[str] = []
    correctness: list[bool] = []

    total_batches = math.ceil(len(examples) / args.batch_size)

    for batch_start in tqdm(
        range(0, len(examples), args.batch_size),
        total=total_batches,
        desc="Bias-in-Bios pairwise",
    ):
        batch = examples[
            batch_start : batch_start + args.batch_size
        ]

        sequences: list[tuple[list[int], list[bool]]] = []
        metadata: list[tuple[Any, int, int, str, str, str]] = []

        for local_index, example in enumerate(batch):
            global_index = batch_start + local_index

            gold_id = int(example.metadata["profession_id"])
            if gold_id not in PAIRWISE_ALTERNATIVES:
                raise ValueError(
                    f"Invalid profession_id={gold_id} for {example.id}"
                )

            alternative_id = PAIRWISE_ALTERNATIVES[gold_id]
            gold_name = PROFESSION_ID_TO_NAME[gold_id]
            alternative_name = PROFESSION_ID_TO_NAME[alternative_id]
            gold_is_a = global_index % 2 == 0

            if gold_is_a:
                option_a = gold_name
                option_b = alternative_name
                gold_answer = "A"
            else:
                option_a = alternative_name
                option_b = gold_name
                gold_answer = "B"

            prompt = PROMPT_TEMPLATE.format(
                biography=example.text,
                option_a=option_a,
                option_b=option_b,
            )

            option_a_ids = encode_continuation(
                tokenizer, " " + option_a
            )
            option_b_ids = encode_continuation(
                tokenizer, " " + option_b
            )

            sequences.append(
                make_sequence(tokenizer, prompt, option_a_ids)
            )
            sequences.append(
                make_sequence(tokenizer, prompt, option_b_ids)
            )

            gender = normalize_gender(
                example.metadata["gender"],
                gender_map,
            )

            metadata.append(
                (
                    example,
                    gold_id,
                    alternative_id,
                    gender,
                    gold_answer,
                    prompt,
                )
            )

        scores = score_sequences(
            model,
            tokenizer,
            sequences,
        ).reshape(len(batch), 2)

        for i, (
            example,
            gold_id,
            alternative_id,
            gender,
            gold_answer,
            prompt,
        ) in enumerate(metadata):
            score_a = float(scores[i, 0])
            score_b = float(scores[i, 1])
            pred_answer = "A" if score_a > score_b else "B"
            correct = pred_answer == gold_answer

            gold_ids.append(gold_id)
            genders.append(gender)
            correctness.append(correct)

            rows.append(
                {
                    "id": str(example.id),
                    "gold_profession_id": gold_id,
                    "gold_profession": PROFESSION_ID_TO_NAME[gold_id],
                    "alternative_profession_id": alternative_id,
                    "alternative_profession": (
                        PROFESSION_ID_TO_NAME[alternative_id]
                    ),
                    "gender": gender,
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer,
                    "correct_pairwise": bool(correct),
                    "option_A_profession_logprob": score_a,
                    "option_B_profession_logprob": score_b,
                    "margin_gold_minus_alternative": (
                        (score_a - score_b)
                        if gold_answer == "A"
                        else (score_b - score_a)
                    ),
                }
            )

    metrics = compute_eo_metrics(
        gold_profession_ids=gold_ids,
        genders=genders,
        correctness=correctness,
        stable_group_min_count=args.stable_group_min_count,
    )

    result = PairwiseResult(
        model_name_or_path=args.model,
        task="bias_in_bios_fixed_pairwise",
        num_examples=len(rows),
        option_order="alternating_by_test_example_index",
        answer_a="profession_name_in_option_A",
        answer_b="profession_name_in_option_B",
        metrics=metrics,
    )

    with (
        args.output_dir / "pairwise_result.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            asdict(result),
            f,
            indent=2,
            ensure_ascii=False,
        )

    with (
        args.output_dir / "pairwise_predictions.jsonl"
    ).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    print("=" * 72)
    print("BIAS-IN-BIOS FIXED-PAIR EVALUATION COMPLETE")
    print("=" * 72)
    print(f"Model:       {args.model}")
    print(f"Examples:    {len(rows):,}")
    print(f"Accuracy:    {metrics['accuracy'] * 100:.2f}%")
    print(f"EO GAP RMS:  {metrics['rms_eo_gap'] * 100:.2f}")
    if metrics["rms_eo_gap_stable_cells"] is not None:
        print(
            "EO stable:   "
            f"{metrics['rms_eo_gap_stable_cells'] * 100:.2f}"
        )
    print(
        f"Predictions: {args.output_dir / 'pairwise_predictions.jsonl'}"
    )


if __name__ == "__main__":
    main()
