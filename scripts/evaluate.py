from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from collapse.data.base import load_examples_jsonl
from collapse.evaluation.bios import (
    BiasInBiosEvalConfig,
    evaluate_bias_in_bios_28way,
    save_bias_in_bios_result,
)
from collapse.evaluation.mmlu import (
    MMLUConfig,
    evaluate_mmlu,
)
from collapse.evaluation.perplexity import (
    PerplexityConfig,
    evaluate_fixed_human_perplexity,
    save_perplexity_result,
)


DEFAULT_CONFIG = "configs/experiments/qwen_bios_recursive.yaml"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one checkpoint with the fixed evaluation protocol: "
            "held-out human perplexity, Bias-in-Bios 28-way occupation "
            "prediction, and MMLU."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG),
        help=f"Experiment YAML. Default: {DEFAULT_CONFIG}",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Checkpoint/model to evaluate.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where evaluation outputs will be written.",
    )

    parser.add_argument(
        "--human-eval",
        type=Path,
        default=None,
        help=(
            "Fixed held-out human JSONL for PPL. "
            "Overrides evaluation.human_eval_path in config."
        ),
    )

    parser.add_argument(
        "--bios-eval",
        type=Path,
        default=None,
        help=(
            "Bias-in-Bios held-out test JSONL. "
            "Overrides evaluation.bios_eval_path in config."
        ),
    )

    parser.add_argument(
        "--skip-ppl",
        action="store_true",
        help="Skip fixed held-out human perplexity.",
    )

    parser.add_argument(
        "--skip-bios",
        action="store_true",
        help="Skip Bias-in-Bios 28-way evaluation.",
    )

    parser.add_argument(
        "--skip-mmlu",
        action="store_true",
        help="Skip MMLU.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help=(
            "Debugging only: cap PPL/Bias-in-Bios examples. "
            "Do not use for final paper results."
        ),
    )

    parser.add_argument(
        "--mmlu-limit",
        type=float,
        default=None,
        help=(
            "Debugging only: pass --limit to lm-eval. "
            "May be integer-like or fractional."
        ),
    )

    parser.add_argument(
        "--mmlu-dry-run",
        action="store_true",
        help="Write/show the MMLU command without executing lm-eval.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into a non-empty evaluation output directory.",
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment config does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(
            f"Experiment config is empty: {path}"
        )

    if not isinstance(raw, dict):
        raise TypeError(
            f"Top-level config must be a YAML mapping, got {type(raw)}."
        )

    return raw


def get_section(
    raw: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    value = raw.get(name, {})

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise TypeError(
            f"`{name}` must be a YAML mapping."
        )

    return value


def resolve_eval_path(
    *,
    cli_path: Path | None,
    evaluation: dict[str, Any],
    keys: tuple[str, ...],
    description: str,
) -> Path:
    if cli_path is not None:
        path = cli_path
    else:
        value = None

        for key in keys:
            if evaluation.get(key):
                value = evaluation[key]
                break

        if value is None:
            raise ValueError(
                f"Could not resolve {description}. "
                f"Set one of {keys} under `evaluation:` "
                "or pass the corresponding CLI argument."
            )

        path = Path(value)

    if not path.exists():
        raise FileNotFoundError(
            f"{description} does not exist: {path}"
        )

    return path


def ensure_output_dir(
    path: Path,
    *,
    force: bool,
) -> None:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = list(
        path.iterdir()
    )

    if existing and not force:
        raise FileExistsError(
            f"Evaluation output directory is not empty: {path}\n"
            "Use --force only if this is intentional."
        )

def build_perplexity_config(
    raw: dict[str, Any],
    *,
    max_examples: int | None,
) -> PerplexityConfig:
    evaluation = get_section(
        raw,
        "evaluation",
    )

    ppl = evaluation.get(
        "perplexity",
        {},
    )

    if ppl is None:
        ppl = {}

    if not isinstance(ppl, dict):
        raise TypeError(
            "`evaluation.perplexity` must be a YAML mapping."
        )

    return PerplexityConfig(
        batch_size=int(
            ppl.get(
                "batch_size",
                8,
            )
        ),
        max_length=(
            int(ppl["max_length"])
            if ppl.get("max_length") is not None
            else None
        ),
        stride=(
            int(ppl["stride"])
            if ppl.get("stride") is not None
            else None
        ),
        add_special_tokens=bool(
            ppl.get(
                "add_special_tokens",
                False,
            )
        ),
        add_eos_per_document=bool(
            ppl.get(
                "add_eos_per_document",
                True,
            )
        ),
        torch_dtype=str(
            ppl.get(
                "torch_dtype",
                "auto",
            )
        ),
        device_map=ppl.get(
            "device_map",
            "auto",
        ),
        max_examples=(
            max_examples
            if max_examples is not None
            else (
                int(ppl["max_examples"])
                if ppl.get("max_examples") is not None
                else None
            )
        ),
    )


def build_bios_config(
    raw: dict[str, Any],
    *,
    max_examples: int | None,
) -> BiasInBiosEvalConfig:
    evaluation = get_section(
        raw,
        "evaluation",
    )

    bios = evaluation.get(
        "bios",
        {},
    )

    if bios is None:
        bios = {}

    if not isinstance(bios, dict):
        raise TypeError(
            "`evaluation.bios` must be a YAML mapping."
        )

    gender_map = bios.get(
        "gender_map"
    )

    if gender_map is not None:
        if not isinstance(
            gender_map,
            dict,
        ):
            raise TypeError(
                "evaluation.bios.gender_map must be a mapping."
            )

        gender_map = {
            str(key): str(value)
            for key, value
            in gender_map.items()
        }

    return BiasInBiosEvalConfig(
        example_batch_size=int(
            bios.get(
                "example_batch_size",
                8,
            )
        ),
        candidate_chunk_size=int(
            bios.get(
                "candidate_chunk_size",
                4,
            )
        ),
        prompt_template=str(
            bios.get(
                "prompt_template",
                (
                    "Biography:\n"
                    "{biography}\n\n"
                    "Profession:"
                ),
            )
        ),
        label_prefix=str(
            bios.get(
                "label_prefix",
                " ",
            )
        ),
        primary_score=str(
            bios.get(
                "primary_score",
                "mean_logprob",
            )
        ),
        save_all_scores=bool(
            bios.get(
                "save_all_scores",
                True,
            )
        ),
        stable_group_min_count=int(
            bios.get(
                "stable_group_min_count",
                20,
            )
        ),
        gender_map=gender_map,
        torch_dtype=str(
            bios.get(
                "torch_dtype",
                "auto",
            )
        ),
        device_map=bios.get(
            "device_map",
            "auto",
        ),
        max_prompt_tokens=(
            int(
                bios[
                    "max_prompt_tokens"
                ]
            )
            if bios.get(
                "max_prompt_tokens"
            )
            is not None
            else None
        ),
        max_examples=(
            max_examples
            if max_examples is not None
            else (
                int(bios["max_examples"])
                if bios.get("max_examples") is not None
                else None
            )
        ),
    )


def build_mmlu_config(
    raw: dict[str, Any],
    *,
    cli_limit: float | None,
) -> MMLUConfig:
    evaluation = get_section(
        raw,
        "evaluation",
    )

    mmlu = evaluation.get(
        "mmlu",
        {},
    )

    if mmlu is None:
        mmlu = {}

    if not isinstance(mmlu, dict):
        raise TypeError(
            "`evaluation.mmlu` must be a YAML mapping."
        )

    model_args = mmlu.get(
        "model_args"
    )

    if model_args is not None and not isinstance(
        model_args,
        dict,
    ):
        raise TypeError(
            "evaluation.mmlu.model_args must be a mapping."
        )

    configured_limit = mmlu.get(
        "limit"
    )

    limit = (
        cli_limit
        if cli_limit is not None
        else configured_limit
    )

    return MMLUConfig(
        executable=str(
            mmlu.get(
                "executable",
                "lm-eval",
            )
        ),
        task=str(
            mmlu.get(
                "task",
                "mmlu",
            )
        ),
        num_fewshot=int(
            mmlu.get(
                "num_fewshot",
                5,
            )
        ),
        batch_size=str(
            mmlu.get(
                "batch_size",
                "auto",
            )
        ),
        device=str(
            mmlu.get(
                "device",
                "cuda:0",
            )
        ),
        dtype=str(
            mmlu.get(
                "dtype",
                "bfloat16",
            )
        ),
        seed=str(
            mmlu.get(
                "seed",
                "0,1234,1234,1234",
            )
        ),
        log_samples=bool(
            mmlu.get(
                "log_samples",
                True,
            )
        ),
        apply_chat_template=bool(
            mmlu.get(
                "apply_chat_template",
                False,
            )
        ),
        limit=limit,
        model_args=(
            dict(model_args)
            if model_args is not None
            else None
        ),
    )

def save_combined_summary(
    *,
    path: Path,
    model_name_or_path: str,
    ppl_result: Any | None,
    bios_result: Any | None,
    mmlu_result: Any | None,
) -> None:
    summary: dict[str, Any] = {
        "model_name_or_path": (
            model_name_or_path
        ),
        "heldout_human_perplexity": (
            asdict(ppl_result)
            if ppl_result is not None
            else None
        ),
        "bias_in_bios": (
            asdict(bios_result)
            if bios_result is not None
            else None
        ),
        "mmlu": (
            asdict(mmlu_result)
            if mmlu_result is not None
            else None
        ),
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

def main() -> None:
    args = parse_args()

    raw = load_yaml(
        args.config
    )

    evaluation = get_section(
        raw,
        "evaluation",
    )

    ensure_output_dir(
        args.output_dir,
        force=args.force,
    )

    print("=" * 72)
    print("FAIRNESS-COLLAPSE CHECKPOINT EVALUATION")
    print("=" * 72)
    print(f"Model:      {args.model}")
    print(f"Config:     {args.config}")
    print(f"Output:     {args.output_dir}")
    print()

    ppl_result = None
    bios_result = None
    mmlu_result = None

    if not args.skip_ppl:
        human_eval_path = resolve_eval_path(
            cli_path=args.human_eval,
            evaluation=evaluation,
            keys=(
                "human_eval_path",
                "ppl_eval_path",
                "heldout_human_path",
            ),
            description=(
                "held-out human evaluation JSONL"
            ),
        )

        human_examples = load_examples_jsonl(
            human_eval_path
        )

        ppl_config = build_perplexity_config(
            raw,
            max_examples=(
                args.max_examples
            ),
        )

        print("Running fixed held-out human PPL...")
        print(
            f"  dataset: {human_eval_path}"
        )

        (
            ppl_result,
            ppl_items,
        ) = evaluate_fixed_human_perplexity(
            examples=human_examples,
            model_name_or_path=(
                args.model
            ),
            config=ppl_config,
        )

        ppl_output = (
            args.output_dir
            / "human_ppl"
        )

        save_perplexity_result(
            result=ppl_result,
            items=ppl_items,
            output_dir=ppl_output,
        )

        print(
            f"  PPL: {ppl_result.perplexity:.4f}"
        )
        print()

    if not args.skip_bios:
        bios_eval_path = resolve_eval_path(
            cli_path=args.bios_eval,
            evaluation=evaluation,
            keys=(
                "bios_eval_path",
                "bias_in_bios_eval_path",
                "test_path",
            ),
            description=(
                "Bias-in-Bios evaluation JSONL"
            ),
        )

        bios_examples = load_examples_jsonl(
            bios_eval_path
        )

        bios_config = build_bios_config(
            raw,
            max_examples=(
                args.max_examples
            ),
        )

        print("Running Bias-in-Bios 28-way likelihood evaluation...")
        print(
            f"  dataset: {bios_eval_path}"
        )

        (
            bios_result,
            bios_predictions,
        ) = evaluate_bias_in_bios_28way(
            examples=bios_examples,
            model_name_or_path=(
                args.model
            ),
            config=bios_config,
        )

        bios_output = (
            args.output_dir
            / "bios"
        )

        save_bias_in_bios_result(
            result=bios_result,
            predictions=bios_predictions,
            output_dir=bios_output,
        )

        primary_metrics = (
            bios_result.metrics_mean_logprob
            if bios_result.primary_score
            == "mean_logprob"
            else bios_result.metrics_total_logprob
        )

        print(
            "  accuracy: "
            f"{primary_metrics['accuracy']:.4f}"
        )

        if (
            primary_metrics[
                "rms_eo_gap"
            ]
            is not None
        ):
            print(
                "  RMS EO gap: "
                f"{primary_metrics['rms_eo_gap']:.4f}"
            )

        print()

    if not args.skip_mmlu:
        mmlu_config = build_mmlu_config(
            raw,
            cli_limit=(
                args.mmlu_limit
            ),
        )

        print("Running MMLU...")

        mmlu_output = (
            args.output_dir
            / "mmlu"
        )

        mmlu_result = evaluate_mmlu(
            model_name_or_path=(
                args.model
            ),
            output_dir=mmlu_output,
            config=mmlu_config,
            dry_run=(
                args.mmlu_dry_run
            ),
        )

        print()

    save_combined_summary(
        path=(
            args.output_dir
            / "evaluation_summary.json"
        ),
        model_name_or_path=(
            args.model
        ),
        ppl_result=ppl_result,
        bios_result=bios_result,
        mmlu_result=mmlu_result,
    )

    print("=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)
    print(
        f"Summary: "
        f"{args.output_dir / 'evaluation_summary.json'}"
    )


if __name__ == "__main__":
    main()
