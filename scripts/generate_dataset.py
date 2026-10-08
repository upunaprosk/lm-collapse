from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from collapse.contamination.generate import (
    GenerationConfig,
    generate_seeded_dataset,
    save_generation_result,
)
from collapse.data.base import load_examples_jsonl
from collapse.records import Example


DEFAULT_CONFIG = "configs/experiments/qwen_bios_recursive.yaml"
DEFAULT_INPUT = "data/processed/bias_in_bios/train.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one seeded synthetic dataset from a prepared "
            "fairness-collapse JSONL file."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG),
        help=f"Experiment YAML config. Default: {DEFAULT_CONFIG}",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Prepared train.jsonl. Overrides config. "
            f"Default fallback: {DEFAULT_INPUT}"
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Checkpoint used for generation. Overrides the model in config. "
            "For recursive iteration 1 this should usually be the "
            "human-trained checkpoint."
        ),
    )

    parser.add_argument(
        "--iteration",
        type=int,
        default=1,
        help="Synthetic-data iteration number. Default: 1",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory override.",
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=30,
        help="Number of generated examples to save for manual inspection.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional limit for a cheap sanity run. "
            "Useful before generating the full dataset."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )

    return parser.parse_args()

def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config is empty: {path}")

    if not isinstance(config, dict):
        raise TypeError(
            f"Top-level YAML object must be a mapping, got {type(config)}."
        )

    return config


def get_section(
    config: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    value = config.get(name, {})

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise TypeError(
            f"`{name}` must be a YAML mapping."
        )

    return value


def resolve_input_path(
    config: dict[str, Any],
    cli_input: Path | None,
) -> Path:
    if cli_input is not None:
        return cli_input

    dataset_cfg = get_section(config, "dataset")
    for key in (
        "prepared_train",
        "prepared_train_path",
        "train_path",
    ):
        if key in dataset_cfg:
            return Path(dataset_cfg[key])

    prepared_dir = dataset_cfg.get(
        "prepared_dir"
    )

    if prepared_dir is not None:
        return Path(prepared_dir) / "train.jsonl"

    return Path(DEFAULT_INPUT)


def resolve_model_name(
    config: dict[str, Any],
    cli_model: str | None,
) -> str:
    if cli_model is not None:
        return cli_model

    model_cfg = get_section(config, "model")

    for key in (
        "generation_checkpoint",
        "human_checkpoint",
        "name_or_path",
        "name",
    ):
        value = model_cfg.get(key)

        if value:
            return str(value)

    raise ValueError(
        "Could not determine generation model. "
        "Set --model or add one of "
        "`model.generation_checkpoint`, `model.human_checkpoint`, "
        "`model.name_or_path`, or `model.name` to the config."
    )


def build_generation_config(
    config: dict[str, Any],
) -> GenerationConfig:
    generation = get_section(
        config,
        "generation",
    )

    return GenerationConfig(
        temperature=float(
            generation.get("temperature", 0.9)
        ),
        top_p=float(
            generation.get("top_p", 0.9)
        ),
        repetition_penalty=float(
            generation.get(
                "repetition_penalty",
                1.1,
            )
        ),
        max_new_tokens=int(
            generation.get(
                "max_new_tokens",
                256,
            )
        ),
        min_new_tokens=int(
            generation.get(
                "min_new_tokens",
                1,
            )
        ),
        batch_size=int(
            generation.get(
                "batch_size",
                32,
            )
        ),
        generation_seed=int(
            generation.get(
                "generation_seed",
                42,
            )
        ),
        torch_dtype=str(
            generation.get(
                "torch_dtype",
                "auto",
            )
        ),
        device_map=generation.get(
            "device_map",
            "auto",
        ),
        add_special_tokens=bool(
            generation.get(
                "add_special_tokens",
                False,
            )
        ),
        max_prompt_tokens=(
            int(
                generation[
                    "max_prompt_tokens"
                ]
            )
            if generation.get(
                "max_prompt_tokens"
            )
            is not None
            else None
        ),
        keep_ineligible_human=bool(
            generation.get(
                "keep_ineligible_human",
                True,
            )
        ),
    )


def model_slug(
    model_name_or_path: str,
) -> str:
    return (
        model_name_or_path
        .strip("/")
        .replace("/", "__")
        .replace(" ", "_")
    )


def resolve_output_dir(
    config: dict[str, Any],
    cli_output_dir: Path | None,
    model_name_or_path: str,
    iteration: int,
) -> Path:
    if cli_output_dir is not None:
        return cli_output_dir

    experiment = get_section(
        config,
        "experiment",
    )

    name = experiment.get(
        "name",
        "generation_sanity",
    )

    return (
        Path("runs")
        / str(name)
        / "generation"
        / model_slug(
            model_name_or_path
        )
        / f"iter_{iteration:02d}"
    )


def ensure_output_dir(
    path: Path,
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
        names = ", ".join(
            sorted(
                p.name
                for p in existing[:8]
            )
        )

        suffix = (
            " ..."
            if len(existing) > 8
            else ""
        )

        raise FileExistsError(
            f"Output directory is not empty: {path}\n"
            f"Existing files: {names}{suffix}\n"
            "Use --force if this overwrite/update is intentional."
        )

def save_preview(
    examples: list[Example],
    path: Path,
    n: int,
) -> None:
    if n <= 0:
        return

    generated = [
        example
        for example in examples
        if example.metadata.get(
            "generation_status"
        )
        == "generated"
    ][:n]

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for example in generated:
            row = {
                "id": example.id,
                "profession": (
                    example.metadata.get(
                        "profession_name"
                    )
                ),
                "gender": (
                    example.metadata.get(
                        "gender"
                    )
                ),
                "iteration": (
                    example.metadata.get(
                        "iteration"
                    )
                ),
                "generation_model": (
                    example.metadata.get(
                        "generation_model"
                    )
                ),
                "native_prefix_tokens": (
                    example.metadata.get(
                        "native_prefix_tokens"
                    )
                ),
                "generated_tokens": (
                    example.metadata.get(
                        "generated_tokens"
                    )
                ),
                "eos_terminated": (
                    example.metadata.get(
                        "eos_terminated"
                    )
                ),
                "prefix_text": (
                    example.prefix_text
                ),
                "human_suffix": (
                    example.human_suffix
                ),
                "synthetic_suffix": (
                    example.synthetic_suffix
                ),
                "human_original": (
                    example.prefix_text or ""
                )
                + (
                    example.human_suffix or ""
                ),
                "synthetic_full": (
                    example.text
                ),
            }

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_manifest(
    *,
    path: Path,
    input_path: Path,
    model_name_or_path: str,
    iteration: int,
    generation_config: GenerationConfig,
    num_loaded_examples: int,
    limited_to: int | None,
) -> None:
    manifest = {
        "input_path": str(
            input_path
        ),
        "model_name_or_path": (
            model_name_or_path
        ),
        "iteration": iteration,
        "num_loaded_examples": (
            num_loaded_examples
        ),
        "limit": limited_to,
        "generation": {
            "temperature": (
                generation_config.temperature
            ),
            "top_p": (
                generation_config.top_p
            ),
            "repetition_penalty": (
                generation_config.repetition_penalty
            ),
            "max_new_tokens": (
                generation_config.max_new_tokens
            ),
            "min_new_tokens": (
                generation_config.min_new_tokens
            ),
            "batch_size": (
                generation_config.batch_size
            ),
            "generation_seed": (
                generation_config.generation_seed
            ),
            "torch_dtype": (
                generation_config.torch_dtype
            ),
            "device_map": (
                generation_config.device_map
            ),
            "add_special_tokens": (
                generation_config.add_special_tokens
            ),
            "max_prompt_tokens": (
                generation_config.max_prompt_tokens
            ),
            "keep_ineligible_human": (
                generation_config.keep_ineligible_human
            ),
        },
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

def main() -> None:
    args = parse_args()

    if args.iteration < 1:
        raise ValueError(
            "--iteration must be >= 1."
        )

    raw_config = load_yaml(
        args.config
    )

    input_path = resolve_input_path(
        raw_config,
        args.input,
    )

    model_name_or_path = (
        resolve_model_name(
            raw_config,
            args.model,
        )
    )

    generation_config = (
        build_generation_config(
            raw_config
        )
    )

    output_dir = resolve_output_dir(
        config=raw_config,
        cli_output_dir=args.output_dir,
        model_name_or_path=(
            model_name_or_path
        ),
        iteration=args.iteration,
    )

    ensure_output_dir(
        output_dir,
        force=args.force,
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset does not exist: {input_path}\n"
            "Run scripts/prepare_dataset.py first."
        )

    print("=" * 72)
    print("FAIRNESS-COLLAPSE SEEDED GENERATION")
    print("=" * 72)
    print(f"Config:       {args.config}")
    print(f"Input:        {input_path}")
    print(f"Model:        {model_name_or_path}")
    print(f"Iteration:    {args.iteration}")
    print(f"Output:       {output_dir}")
    print()
    print("Generation:")
    print(
        f"  temperature          "
        f"{generation_config.temperature}"
    )
    print(
        f"  top_p                "
        f"{generation_config.top_p}"
    )
    print(
        f"  max_new_tokens       "
        f"{generation_config.max_new_tokens}"
    )
    print(
        f"  generation_seed      "
        f"{generation_config.generation_seed}"
    )
    print(
        f"  batch_size           "
        f"{generation_config.batch_size}"
    )

    examples = load_examples_jsonl(
        input_path
    )

    num_loaded_examples = len(
        examples
    )

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(
                "--limit must be > 0."
            )

        examples = examples[
            : args.limit
        ]

        print()
        print(
            f"SANITY MODE: using first "
            f"{len(examples):,} of "
            f"{num_loaded_examples:,} examples."
        )

    num_eligible = sum(
        bool(
            example.metadata.get(
                "seed_eligible",
                False,
            )
        )
        for example in examples
    )

    print()
    print(
        f"Loaded:       "
        f"{len(examples):,} examples"
    )
    print(
        f"Eligible:     "
        f"{num_eligible:,} examples"
    )
    print()

    result = generate_seeded_dataset(
        examples=examples,
        model_name_or_path=(
            model_name_or_path
        ),
        iteration=args.iteration,
        config=generation_config,
    )

    save_generation_result(
        result,
        output_dir,
    )

    save_preview(
        result.examples,
        output_dir
        / "generation_preview.jsonl",
        n=args.preview,
    )

    save_manifest(
        path=(
            output_dir
            / "generation_manifest.json"
        ),
        input_path=input_path,
        model_name_or_path=(
            model_name_or_path
        ),
        iteration=args.iteration,
        generation_config=(
            generation_config
        ),
        num_loaded_examples=(
            num_loaded_examples
        ),
        limited_to=args.limit,
    )

    shutil.copy2(
        args.config,
        output_dir
        / "experiment_config.yaml",
    )

    stats = result.stats

    print()
    print("Generation summary:")
    print(
        f"  eligible              "
        f"{stats.num_eligible:,}"
    )
    print(
        f"  generated             "
        f"{stats.num_generated:,}"
    )
    print(
        f"  failed                "
        f"{stats.num_failed:,}"
    )
    print(
        f"  kept human            "
        f"{stats.num_kept_human:,}"
    )

    if (
        stats.mean_generated_tokens
        is not None
    ):
        print(
            f"  mean generated tokens "
            f"{stats.mean_generated_tokens:.2f}"
        )
        print(
            f"  median generated      "
            f"{stats.median_generated_tokens:.2f}"
        )

    print(
        f"  empty rate            "
        f"{stats.empty_generation_rate:.4%}"
    )
    print(
        f"  EOS termination rate  "
        f"{stats.eos_terminated_rate:.4%}"
    )

    print()
    print("Saved:")
    print(
        f"  {output_dir / 'generated.jsonl'}"
    )
    print(
        f"  {output_dir / 'generation_stats.json'}"
    )
    print(
        f"  {output_dir / 'generation_manifest.json'}"
    )

    if args.preview > 0:
        print(
            f"  {output_dir / 'generation_preview.jsonl'}"
        )

    print()
    print(
        "Inspect generation_preview.jsonl and generation_stats.json "
        "before training on the generated corpus."
    )


if __name__ == "__main__":
    main()
