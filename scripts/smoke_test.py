from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import yaml

from collapse.contamination.generate import (
    GenerationConfig,
    generate_seeded_dataset,
    save_generation_result,
)
from collapse.data.base import (
    load_examples_jsonl,
    save_examples_jsonl,
)
from collapse.evaluation.bios import (
    BiasInBiosEvalConfig,
    evaluate_bias_in_bios_28way,
    save_bias_in_bios_result,
)
from collapse.evaluation.perplexity import (
    PerplexityConfig,
    evaluate_fixed_human_perplexity,
    save_perplexity_result,
)
from collapse.records import Example
from collapse.training.budget import (
    TrainingBudgetConfig,
    fit_training_budget,
    save_training_budget,
)
from collapse.training.llamafactory import (
    LlamaFactoryConfig,
    train_with_llamafactory,
)


DEFAULT_CONFIG = "configs/experiments/qwen_bios_recursive.yaml"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cheap end-to-end integration smoke test for the fairness-"
            "collapse pipeline. By default this performs preflight checks "
            "and creates a one-step LLaMA-Factory DRY-RUN config. Add "
            "--execute for actual generation + one optimizer step + "
            "checkpoint reload + tiny evaluation."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model/checkpoint used for generation and one-step training. "
            "If omitted, use an existing configured human_checkpoint; "
            "otherwise fall back to model.name/base_model."
        ),
    )

    parser.add_argument(
        "--train",
        type=Path,
        default=None,
        help=(
            "Prepared human train.jsonl override. "
            "Defaults to <dataset.prepared_dir>/train.jsonl."
        ),
    )

    parser.add_argument(
        "--eval",
        type=Path,
        default=None,
        help=(
            "Held-out human JSONL override. If omitted, use configured "
            "evaluation.human_eval_path when it exists; otherwise reuse "
            "the tiny human train subset strictly for integration testing."
        ),
    )

    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("runs/_smoke_test"),
    )

    parser.add_argument(
        "--num-train-examples",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num-eval-examples",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually load the model, generate, run ONE optimizer update "
            "through LLaMA-Factory, reload the resulting checkpoint, and "
            "run tiny evaluation. Without this flag no GPU training occurs."
        ),
    )

    parser.add_argument(
        "--skip-bios",
        action="store_true",
        help="Skip the tiny 28-way Bias-in-Bios evaluator.",
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

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        value = yaml.safe_load(f)

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            "Top-level experiment YAML must be a mapping."
        )

    return value


def section(
    raw: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    value = raw.get(
        name,
        {},
    )

    if value is None:
        return {}

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            f"`{name}` must be a YAML mapping."
        )

    return value


def resolve_train_path(
    raw: dict[str, Any],
    override: Path | None,
) -> Path:
    if override is not None:
        return override

    dataset = section(
        raw,
        "dataset",
    )

    direct = (
        dataset.get(
            "prepared_train"
        )
        or dataset.get(
            "prepared_train_path"
        )
        or dataset.get(
            "train_path"
        )
    )

    if direct:
        return Path(
            direct
        )

    prepared_dir = dataset.get(
        "prepared_dir"
    )

    if prepared_dir:
        return (
            Path(
                prepared_dir
            )
            / "train.jsonl"
        )

    raise ValueError(
        "Could not resolve prepared train.jsonl."
    )


def resolve_eval_path(
    raw: dict[str, Any],
    override: Path | None,
) -> Path | None:
    if override is not None:
        return override

    evaluation = section(
        raw,
        "evaluation",
    )

    for key in (
        "human_eval_path",
        "bios_eval_path",
        "ppl_eval_path",
    ):
        value = evaluation.get(
            key
        )

        if value:
            path = Path(
                value
            )

            if path.exists():
                return path

    return None


def resolve_model(
    raw: dict[str, Any],
    override: str | None,
) -> str:
    if override:
        return override

    model = section(
        raw,
        "model",
    )

    human_checkpoint = model.get(
        "human_checkpoint"
    )

    if human_checkpoint:
        human_path = Path(
            str(
                human_checkpoint
            )
        )
        if human_path.exists():
            return str(
                human_path
            )

    for key in (
        "base_model",
        "name_or_path",
        "name",
        "tokenizer_name_or_path",
    ):
        value = model.get(
            key
        )

        if value and "CHANGE_ME" not in str(
            value
        ):
            return str(
                value
            )

    raise ValueError(
        "Could not resolve smoke-test model. Pass --model explicitly."
    )


def resolve_tokenizer(
    raw: dict[str, Any],
    model_name: str,
) -> str:
    training = section(
        raw,
        "training",
    )
    model = section(
        raw,
        "model",
    )

    value = (
        training.get(
            "tokenizer_name_or_path"
        )
        or model.get(
            "tokenizer_name_or_path"
        )
        or model_name
    )

    if "CHANGE_ME" in str(
        value
    ):
        raise ValueError(
            "Tokenizer config still contains CHANGE_ME placeholder."
        )

    return str(
        value
    )


def resolve_lf_config(
    raw: dict[str, Any],
) -> tuple[
    LlamaFactoryConfig,
    dict[str, Any],
]:
    training = section(
        raw,
        "training",
    )

    lf = training.get(
        "llamafactory",
        {},
    )

    if not isinstance(
        lf,
        dict,
    ):
        raise TypeError(
            "training.llamafactory must be a mapping."
        )

    base_yaml = (
        lf.get(
            "base_yaml_path"
        )
        or training.get(
            "base_yaml_path"
        )
    )

    if not base_yaml:
        raise ValueError(
            "Missing LLaMA-Factory base_yaml_path."
        )

    base_yaml_path = Path(
        str(
            base_yaml
        )
    )

    if not base_yaml_path.exists():
        raise FileNotFoundError(
            f"LLaMA-Factory template does not exist: {base_yaml_path}"
        )

    backend = LlamaFactoryConfig(
        base_yaml_path=str(
            base_yaml_path
        ),
        executable=str(
            lf.get(
                "executable",
                "llamafactory-cli",
            )
        ),
        text_column=str(
            lf.get(
                "text_column",
                "text",
            )
        ),
        dataset_prompt_field=str(
            lf.get(
                "dataset_prompt_field",
                "prompt",
            )
        ),
        overwrite_output_dir=False,
        stream_logs=True,
    )

    extra = training.get(
        "extra_overrides",
        {},
    )

    if extra is None:
        extra = {}

    if not isinstance(
        extra,
        dict,
    ):
        raise TypeError(
            "training.extra_overrides must be a mapping."
        )

    return (
        backend,
        dict(
            extra
        ),
    )

def choose_generation_subset(
    examples: Sequence[Example],
    n: int,
) -> list[Example]:
    if n <= 0:
        raise ValueError(
            "num-train-examples must be > 0."
        )

    eligible = [
        example
        for example in examples
        if example.metadata.get(
            "seed_eligible",
            False,
        )
    ]

    if not eligible:
        raise ValueError(
            "Prepared training data contains no seed-eligible examples. "
            "Inspect seed_policy.json and train.jsonl."
        )

    return eligible[
        : min(
            n,
            len(
                eligible
            ),
        )
    ]


def validate_prepared_subset(
    examples: Sequence[Example],
) -> dict[str, Any]:
    errors = []

    for example in examples:
        if example.prefix_text is None:
            errors.append(
                f"{example.id}: prefix_text is None"
            )
            continue

        if example.human_suffix is None:
            errors.append(
                f"{example.id}: human_suffix is None"
            )
            continue

        if (
            example.prefix_text
            + example.human_suffix
            != example.text
        ):
            errors.append(
                f"{example.id}: prefix + human_suffix != original text"
            )

        if len(
            errors
        ) >= 10:
            break

    if errors:
        raise RuntimeError(
            "Prepared-data invariant failure:\n  - "
            + "\n  - ".join(
                errors
            )
        )

    return {
        "num_examples": len(
            examples
        ),
        "all_prefix_reconstructions_exact": True,
    }


def validate_generation_result(
    human: Sequence[Example],
    generated: Sequence[Example],
) -> dict[str, Any]:
    if len(
        human
    ) != len(
        generated
    ):
        raise RuntimeError(
            "Generation changed the number of smoke-test examples."
        )

    human_by_id = {
        example.id: example
        for example in human
    }

    generated_count = 0
    failures = []
    changed_count = 0

    for out in generated:
        source = human_by_id.get(
            out.id
        )

        if source is None:
            failures.append(
                f"{out.id}: unknown output ID"
            )
            continue

        if (
            out.prefix_text
            != source.prefix_text
        ):
            failures.append(
                f"{out.id}: frozen prefix_text changed"
            )

        if out.source == "synthetic":
            generated_count += 1

            if out.synthetic_suffix is None:
                failures.append(
                    f"{out.id}: synthetic row lacks synthetic_suffix"
                )

            if not out.text.startswith(
                str(
                    source.prefix_text
                )
            ):
                failures.append(
                    f"{out.id}: final text does not start with exact prefix"
                )

            if out.text != source.text:
                changed_count += 1

        if len(
            failures
        ) >= 10:
            break

    if failures:
        raise RuntimeError(
            "Generation invariant failure:\n  - "
            + "\n  - ".join(
                failures
            )
        )

    if generated_count == 0:
        raise RuntimeError(
            "Generation smoke test produced zero synthetic examples."
        )

    return {
        "num_examples": len(
            generated
        ),
        "num_generated": (
            generated_count
        ),
        "num_changed_from_human": (
            changed_count
        ),
        "all_frozen_prefixes_preserved": True,
    }


def infer_gender_map(
    raw: dict[str, Any],
) -> dict[str, str] | None:
    evaluation = section(
        raw,
        "evaluation",
    )

    bios = evaluation.get(
        "bios",
        {},
    )

    if not isinstance(
        bios,
        dict,
    ):
        return None

    value = bios.get(
        "gender_map"
    )

    if value is None:
        return None

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            "evaluation.bios.gender_map must be a mapping."
        )

    return {
        str(k): str(v)
        for k, v
        in value.items()
    }

def fit_one_step_budget(
    *,
    examples: Sequence[Example],
    tokenizer_name: str,
) -> Any:
    budget = fit_training_budget(
        human_examples=examples,
        tokenizer_name_or_path=(
            tokenizer_name
        ),
        config=TrainingBudgetConfig(
            cutoff_len=128,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            world_size=1,
            corpus_equivalents=1e-6,
            add_special_tokens=False,
            add_eos_per_document=True,
        ),
    )

    if budget.max_steps != 1:
        raise RuntimeError(
            "Smoke budget unexpectedly produced "
            f"max_steps={budget.max_steps}, expected 1."
        )

    return budget


def main() -> None:
    args = parse_args()

    if (
        args.work_dir.exists()
        and any(
            args.work_dir.iterdir()
        )
    ):
        if not args.force:
            raise FileExistsError(
                f"Smoke-test directory is not empty: {args.work_dir}\n"
                "Use --force to recreate it."
            )

        shutil.rmtree(
            args.work_dir
        )

    args.work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = load_yaml(
        args.config
    )

    train_path = resolve_train_path(
        raw,
        args.train,
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Prepared train data does not exist: {train_path}\n"
            "Run scripts/prepare_dataset.py first."
        )

    eval_path = resolve_eval_path(
        raw,
        args.eval,
    )

    model_name = resolve_model(
        raw,
        args.model,
    )

    tokenizer_name = resolve_tokenizer(
        raw,
        model_name,
    )

    (
        backend_config,
        extra_overrides,
    ) = resolve_lf_config(
        raw
    )

    all_train = load_examples_jsonl(
        train_path
    )

    human_subset = (
        choose_generation_subset(
            all_train,
            args.num_train_examples,
        )
    )

    prepared_check = (
        validate_prepared_subset(
            human_subset
        )
    )

    tiny_human_path = (
        args.work_dir
        / "tiny_human_train.jsonl"
    )

    save_examples_jsonl(
        human_subset,
        tiny_human_path,
    )

    if eval_path is not None:
        eval_examples = (
            load_examples_jsonl(
                eval_path
            )[
                : args.num_eval_examples
            ]
        )

        eval_source = str(
            eval_path.resolve()
        )
    else:
        eval_examples = list(
            human_subset[
                : args.num_eval_examples
            ]
        )

        eval_source = (
            "tiny training subset "
            "(integration-only fallback; not a scientific heldout metric)"
        )

    tiny_eval_path = (
        args.work_dir
        / "tiny_eval.jsonl"
    )

    save_examples_jsonl(
        eval_examples,
        tiny_eval_path,
    )

    budget = fit_one_step_budget(
        examples=human_subset,
        tokenizer_name=(
            tokenizer_name
        ),
    )

    save_training_budget(
        budget,
        args.work_dir
        / "smoke_training_budget.json",
    )

    dry_run_dir = (
        args.work_dir
        / "training_dry_run"
    )

    dry_backend = LlamaFactoryConfig(
        base_yaml_path=(
            backend_config.base_yaml_path
        ),
        executable=(
            backend_config.executable
        ),
        text_column=(
            backend_config.text_column
        ),
        dataset_prompt_field=(
            backend_config.dataset_prompt_field
        ),
        overwrite_output_dir=False,
        stream_logs=True,
    )

    dry_run = train_with_llamafactory(
        examples=human_subset,
        model_name_or_path=model_name,
        run_dir=dry_run_dir,
        run_name="smoke_train_dry_run",
        budget=budget,
        training_seed=991,
        backend_config=dry_backend,
        extra_overrides=(
            extra_overrides
        ),
        dry_run=True,
    )

    dry_runtime = Path(
        dry_run.runtime_yaml
    )

    if not dry_runtime.exists():
        raise RuntimeError(
            "Training dry-run did not create train_runtime.yaml."
        )

    dataset_info = (
        Path(
            dry_run.dataset_dir
        )
        / "dataset_info.json"
    )

    if not dataset_info.exists():
        raise RuntimeError(
            "Training dry-run did not create dataset_info.json."
        )

    report: dict[
        str,
        Any,
    ] = {
        "mode": (
            "execute"
            if args.execute
            else "preflight-only"
        ),
        "config": str(
            args.config.resolve()
        ),
        "model": (
            model_name
        ),
        "tokenizer": (
            tokenizer_name
        ),
        "train_source": str(
            train_path.resolve()
        ),
        "eval_source": (
            eval_source
        ),
        "prepared_data": (
            prepared_check
        ),
        "training_dry_run": {
            "runtime_yaml": (
                dry_run.runtime_yaml
            ),
            "dataset_info": str(
                dataset_info.resolve()
            ),
            "budget_max_steps": (
                budget.max_steps
            ),
            "passed": True,
        },
        "generation_iter1": None,
        "training_one_step": None,
        "recursive_reload_generation": None,
        "perplexity": None,
        "bios": None,
    }

    if not args.execute:
        report_path = (
            args.work_dir
            / "smoke_report.json"
        )

        with report_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                report,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print("=" * 72)
        print("SMOKE PREFLIGHT PASSED")
        print("=" * 72)
        print(
            "No model generation/training was executed."
        )
        print(
            f"Inspect runtime YAML: {dry_run.runtime_yaml}"
        )
        print(
            f"Inspect dataset info: {dataset_info}"
        )
        print()
        print(
            "When those look correct, rerun the same command with --execute."
        )
        return

    generation_cfg_raw = section(
        raw,
        "generation",
    )

    generation_config = (
        GenerationConfig(
            temperature=float(
                generation_cfg_raw.get(
                    "temperature",
                    0.9,
                )
            ),
            top_p=float(
                generation_cfg_raw.get(
                    "top_p",
                    0.9,
                )
            ),
            repetition_penalty=float(
                generation_cfg_raw.get(
                    "repetition_penalty",
                    1.1,
                )
            ),
            max_new_tokens=(
                args.max_new_tokens
            ),
            min_new_tokens=1,
            batch_size=(
                args.generation_batch_size
            ),
            generation_seed=9901,
            torch_dtype=str(
                generation_cfg_raw.get(
                    "torch_dtype",
                    "auto",
                )
            ),
            device_map=(
                generation_cfg_raw.get(
                    "device_map",
                    "auto",
                )
            ),
            add_special_tokens=bool(
                generation_cfg_raw.get(
                    "add_special_tokens",
                    False,
                )
            ),
            max_prompt_tokens=(
                generation_cfg_raw.get(
                    "max_prompt_tokens"
                )
            ),
            keep_ineligible_human=True,
        )
    )

    generation_1 = (
        generate_seeded_dataset(
            examples=human_subset,
            model_name_or_path=(
                model_name
            ),
            iteration=1,
            config=(
                generation_config
            ),
        )
    )

    generation_dir = (
        args.work_dir
        / "generation_iter1"
    )

    save_generation_result(
        generation_1,
        generation_dir,
    )

    generation_check = (
        validate_generation_result(
            human_subset,
            generation_1.examples,
        )
    )

    report[
        "generation_iter1"
    ] = {
        **generation_check,
        "stats": asdict(
            generation_1.stats
        ),
        "passed": True,
    }

    train_dir = (
        args.work_dir
        / "training_one_step"
    )

    training_run = (
        train_with_llamafactory(
            examples=(
                generation_1.examples
            ),
            model_name_or_path=(
                model_name
            ),
            run_dir=train_dir,
            run_name=(
                "smoke_recursive_iter1"
            ),
            budget=budget,
            training_seed=9902,
            backend_config=(
                backend_config
            ),
            extra_overrides=(
                extra_overrides
            ),
            dry_run=False,
        )
    )

    checkpoint = Path(
        training_run.output_dir
    )

    if not checkpoint.exists():
        raise RuntimeError(
            f"Training finished but checkpoint path is missing: {checkpoint}"
        )

    if not (
        checkpoint
        / "config.json"
    ).exists():
        raise RuntimeError(
            "One-step training directory exists but config.json is absent. "
            "The LLaMA-Factory output may not be a reloadable HF checkpoint."
        )

    report[
        "training_one_step"
    ] = {
        "checkpoint": str(
            checkpoint.resolve()
        ),
        "return_code": (
            training_run.return_code
        ),
        "passed": True,
    }

    reload_subset = list(
        human_subset[
            : min(
                4,
                len(
                    human_subset
                ),
            )
        ]
    )

    generation_2_cfg = (
        GenerationConfig(
            **{
                **asdict(
                    generation_config
                ),
                "generation_seed": 9903,
                "max_new_tokens": min(
                    args.max_new_tokens,
                    16,
                ),
            }
        )
    )

    generation_2 = (
        generate_seeded_dataset(
            examples=reload_subset,
            model_name_or_path=str(
                checkpoint
            ),
            iteration=2,
            config=(
                generation_2_cfg
            ),
        )
    )

    reload_check = (
        validate_generation_result(
            reload_subset,
            generation_2.examples,
        )
    )

    reload_dir = (
        args.work_dir
        / "generation_iter2_reload"
    )

    save_generation_result(
        generation_2,
        reload_dir,
    )

    report[
        "recursive_reload_generation"
    ] = {
        **reload_check,
        "checkpoint_reloaded": str(
            checkpoint.resolve()
        ),
        "passed": True,
    }

    ppl_result, ppl_items = (
        evaluate_fixed_human_perplexity(
            examples=eval_examples,
            model_name_or_path=str(
                checkpoint
            ),
            config=PerplexityConfig(
                batch_size=min(
                    4,
                    max(
                        1,
                        len(
                            eval_examples
                        ),
                    ),
                ),
                max_length=128,
                stride=64,
                add_special_tokens=False,
                add_eos_per_document=True,
                torch_dtype="auto",
                device_map="auto",
            ),
        )
    )

    ppl_dir = (
        args.work_dir
        / "evaluation"
        / "human_ppl"
    )

    save_perplexity_result(
        result=ppl_result,
        items=ppl_items,
        output_dir=ppl_dir,
    )

    if not (
        ppl_result.perplexity
        > 0.0
    ):
        raise RuntimeError(
            "Perplexity smoke test returned a non-positive value."
        )

    report[
        "perplexity"
    ] = {
        "value": (
            ppl_result.perplexity
        ),
        "num_examples": (
            ppl_result.num_examples
        ),
        "passed": True,
    }
    if not args.skip_bios:
        has_bios_metadata = all(
            (
                "profession_id"
                in example.metadata
                and "gender"
                in example.metadata
            )
            for example in eval_examples
        )

        if has_bios_metadata:
            bios_config = (
                BiasInBiosEvalConfig(
                    example_batch_size=min(
                        2,
                        max(
                            1,
                            len(
                                eval_examples
                            ),
                        ),
                    ),
                    candidate_chunk_size=4,
                    primary_score=(
                        "mean_logprob"
                    ),
                    save_all_scores=False,
                    stable_group_min_count=1,
                    gender_map=(
                        infer_gender_map(
                            raw
                        )
                    ),
                    torch_dtype="auto",
                    device_map="auto",
                    max_examples=(
                        len(
                            eval_examples
                        )
                    ),
                )
            )

            bios_result, bios_predictions = (
                evaluate_bias_in_bios_28way(
                    examples=eval_examples,
                    model_name_or_path=str(
                        checkpoint
                    ),
                    config=(
                        bios_config
                    ),
                )
            )

            bios_dir = (
                args.work_dir
                / "evaluation"
                / "bios"
            )

            save_bias_in_bios_result(
                result=bios_result,
                predictions=(
                    bios_predictions
                ),
                output_dir=bios_dir,
            )

            report[
                "bios"
            ] = {
                "num_examples": (
                    bios_result.num_examples
                ),
                "accuracy": (
                    bios_result.metrics_mean_logprob.get(
                        "accuracy"
                    )
                ),
                "rms_eo_gap": (
                    bios_result.metrics_mean_logprob.get(
                        "rms_eo_gap"
                    )
                ),
                "passed": True,
            }
        else:
            report[
                "bios"
            ] = {
                "skipped": True,
                "reason": (
                    "eval examples do not contain profession_id + gender"
                ),
            }

    report_path = (
        args.work_dir
        / "smoke_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 72)
    print("END-TO-END SMOKE TEST PASSED")
    print("=" * 72)
    print(
        f"One-step checkpoint: {checkpoint}"
    )
    print(
        f"Reload generation:   passed"
    )
    print(
        f"Tiny human PPL:      {ppl_result.perplexity:.4f}"
    )

    if report[
        "bios"
    ] is not None:
        print(
            f"Bias-in-Bios:        "
            f"{'passed' if report['bios'].get('passed') else 'skipped'}"
        )

    print(
        f"Report:              {report_path}"
    )
    print()
    print(
        "This validates plumbing only. Delete runs/_smoke_test afterward; "
        "none of these metrics belong in the paper."
    )


if __name__ == "__main__":
    main()
