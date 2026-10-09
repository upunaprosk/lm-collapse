from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from collapse.contamination.build_dataset import BuildDatasetConfig
from collapse.contamination.generate import GenerationConfig
from collapse.experiment import (
    RecursiveExperimentConfig,
    run_recursive_experiment,
)
from collapse.training.budget import TrainingBudgetConfig
from collapse.training.llamafactory import LlamaFactoryConfig


DEFAULT_CONFIG = "configs/experiments/qwen_bios_recursive.yaml"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one complete fairness-collapse experiment for one model "
            "and one pipeline seed: matched human-only trajectory plus "
            "recursive seeded synthetic-data trajectory."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG),
        help=f"Experiment YAML. Default: {DEFAULT_CONFIG}",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override the experiment pipeline seed. "
            "Useful for repeated runs such as 42, 43, 44."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs"),
        help="Root directory for experiment outputs. Default: runs",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override the number of recursive/human iterations.",
    )

    parser.add_argument(
        "--human-checkpoint",
        type=str,
        default=None,
        help="Override the human-trained t=0 checkpoint.",
    )

    parser.add_argument(
        "--prepared-train",
        type=Path,
        default=None,
        help="Override the prepared train.jsonl path.",
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


def first_nonempty(
    *values: Any,
) -> Any:
    for value in values:
        if value is not None and value != "":
            return value

    return None

def resolve_experiment_name(
    raw: dict[str, Any],
) -> str:
    experiment = get_section(
        raw,
        "experiment",
    )

    value = first_nonempty(
        experiment.get("name"),
        raw.get("name"),
    )

    if not value:
        raise ValueError(
            "Missing experiment name. Set `experiment.name`."
        )

    return str(value)


def resolve_num_iterations(
    raw: dict[str, Any],
    cli_iterations: int | None,
) -> int:
    if cli_iterations is not None:
        value = cli_iterations
    else:
        experiment = get_section(
            raw,
            "experiment",
        )

        value = first_nonempty(
            experiment.get("iterations"),
            experiment.get("num_iterations"),
            4,
        )

    value = int(value)

    if value <= 0:
        raise ValueError(
            "Number of iterations must be > 0."
        )

    return value


def resolve_seed(
    raw: dict[str, Any],
    cli_seed: int | None,
) -> int:
    if cli_seed is not None:
        return int(cli_seed)

    experiment = get_section(
        raw,
        "experiment",
    )

    return int(
        first_nonempty(
            experiment.get("seed"),
            42,
        )
    )


def resolve_human_checkpoint(
    raw: dict[str, Any],
    cli_value: str | None,
) -> str:
    if cli_value:
        return cli_value

    model = get_section(
        raw,
        "model",
    )

    value = first_nonempty(
        model.get("human_checkpoint"),
        model.get("generation_checkpoint"),
    )

    if not value:
        raise ValueError(
            "Missing human-trained t=0 checkpoint. "
            "Set `model.human_checkpoint` or pass "
            "--human-checkpoint."
        )

    return str(value)


def resolve_training_tokenizer(
    raw: dict[str, Any],
) -> str:
    model = get_section(
        raw,
        "model",
    )

    training = get_section(
        raw,
        "training",
    )

    value = first_nonempty(
        training.get("tokenizer_name_or_path"),
        model.get("tokenizer_name_or_path"),
        model.get("name"),
        model.get("name_or_path"),
    )

    if not value:
        raise ValueError(
            "Could not resolve the model-native tokenizer used for "
            "training-budget accounting. Set `model.name`, "
            "`model.tokenizer_name_or_path`, or "
            "`training.tokenizer_name_or_path`."
        )

    return str(value)


def resolve_prepared_train(
    raw: dict[str, Any],
    cli_value: Path | None,
) -> str:
    if cli_value is not None:
        return str(cli_value)

    dataset = get_section(
        raw,
        "dataset",
    )

    direct = first_nonempty(
        dataset.get("prepared_train"),
        dataset.get("prepared_train_path"),
        dataset.get("train_path"),
    )

    if direct:
        return str(direct)

    prepared_dir = dataset.get(
        "prepared_dir"
    )

    if prepared_dir:
        return str(
            Path(prepared_dir)
            / "train.jsonl"
        )

    raise ValueError(
        "Could not resolve prepared training data. "
        "Set `dataset.prepared_dir`, `dataset.prepared_train`, "
        "or pass --prepared-train."
    )


def build_generation_config(
    raw: dict[str, Any],
    *,
    base_seed: int,
) -> GenerationConfig:
    generation = get_section(
        raw,
        "generation",
    )

    generation_seed = int(
        first_nonempty(
            generation.get("generation_seed"),
            base_seed,
        )
    )

    return GenerationConfig(
        temperature=float(
            generation.get(
                "temperature",
                0.9,
            )
        ),
        top_p=float(
            generation.get(
                "top_p",
                0.9,
            )
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
        generation_prompt=str(
            generation.get(
                "generation_prompt",
                "",
            )
        ),
        match_human_suffix_length=bool(
            generation.get(
                "match_human_suffix_length",
                False,
            )
        ),
        length_tolerance=float(
            generation.get(
                "length_tolerance",
                0.25,
            )
        ),
        buffer_tokens=int(
            generation.get(
                "buffer_tokens",
                15,
            )
        ),
        generation_seed=(
            generation_seed
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


def build_dataset_mix_config(
    raw: dict[str, Any],
    *,
    base_seed: int,
) -> BuildDatasetConfig:
    contamination = get_section(
        raw,
        "contamination",
    )

    fraction = float(
        contamination.get(
            "synthetic_document_fraction",
            1.0,
        )
    )
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            "contamination.synthetic_document_fraction must be in [0, 1]."
        )

    stratify = contamination.get(
        "stratify_by",
        "profession_id",
    )
    if stratify is not None:
        stratify = str(stratify)

    return BuildDatasetConfig(
        synthetic_fraction=fraction,
        seed=int(
            contamination.get(
                "sampling_seed",
                base_seed + 40_000,
            )
        ),
        stratify_by_metadata_key=stratify,
    )



def build_budget_config(
    raw: dict[str, Any],
) -> TrainingBudgetConfig:
    training = get_section(
        raw,
        "training",
    )

    budget = training.get(
        "budget",
        {},
    )

    if budget is None:
        budget = {}

    if not isinstance(
        budget,
        dict,
    ):
        raise TypeError(
            "`training.budget` must be a YAML mapping."
        )

    def pick(
        key: str,
        default: Any,
    ) -> Any:
        if key in budget:
            return budget[key]

        if key in training:
            return training[key]

        return default

    return TrainingBudgetConfig(
        cutoff_len=int(
            pick(
                "cutoff_len",
                512,
            )
        ),
        per_device_train_batch_size=int(
            pick(
                "per_device_train_batch_size",
                2,
            )
        ),
        gradient_accumulation_steps=int(
            pick(
                "gradient_accumulation_steps",
                16,
            )
        ),
        world_size=int(
            pick(
                "world_size",
                1,
            )
        ),
        corpus_equivalents=float(
            pick(
                "corpus_equivalents",
                1.0,
            )
        ),
        add_special_tokens=bool(
            pick(
                "add_special_tokens",
                False,
            )
        ),
        add_eos_per_document=bool(
            pick(
                "add_eos_per_document",
                True,
            )
        ),
    )


def build_llamafactory_config(
    raw: dict[str, Any],
) -> LlamaFactoryConfig:
    training = get_section(
        raw,
        "training",
    )

    lf = training.get(
        "llamafactory",
        {},
    )

    if lf is None:
        lf = {}

    if not isinstance(lf, dict):
        raise TypeError(
            "`training.llamafactory` must be a YAML mapping."
        )

    base_yaml_path = first_nonempty(
        lf.get("base_yaml_path"),
        training.get("base_yaml_path"),
    )

    if not base_yaml_path:
        raise ValueError(
            "Missing LLaMA-Factory template. Set "
            "`training.llamafactory.base_yaml_path`."
        )

    return LlamaFactoryConfig(
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
        overwrite_output_dir=bool(
            lf.get(
                "overwrite_output_dir",
                False,
            )
        ),
        stream_logs=bool(
            lf.get(
                "stream_logs",
                True,
            )
        ),
    )


def build_training_overrides(
    raw: dict[str, Any],
) -> dict[str, Any]:
    training = get_section(
        raw,
        "training",
    )

    value = training.get(
        "extra_overrides",
        {},
    )

    if value is None:
        return {}

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            "`training.extra_overrides` must be a YAML mapping."
        )

    forbidden = {
        "model_name_or_path",
        "dataset",
        "dataset_dir",
        "output_dir",
        "seed",
        "data_seed",
        "max_steps",
        "num_train_epochs",
        "cutoff_len",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "packing",
    }

    conflicts = (
        forbidden
        & set(value)
    )

    if conflicts:
        raise ValueError(
            "training.extra_overrides contains fields controlled by the "
            "experiment/budget and therefore cannot be overridden:\n  "
            + ", ".join(
                sorted(conflicts)
            )
        )

    return dict(value)


def build_experiment_config(
    *,
    raw: dict[str, Any],
    args: argparse.Namespace,
) -> RecursiveExperimentConfig:
    seed = resolve_seed(
        raw,
        args.seed,
    )

    return RecursiveExperimentConfig(
        experiment_name=(
            resolve_experiment_name(
                raw
            )
        ),
        human_checkpoint=(
            resolve_human_checkpoint(
                raw,
                args.human_checkpoint,
            )
        ),
        prepared_train_path=(
            resolve_prepared_train(
                raw,
                args.prepared_train,
            )
        ),
        training_tokenizer=(
            resolve_training_tokenizer(
                raw
            )
        ),
        num_iterations=(
            resolve_num_iterations(
                raw,
                args.iterations,
            )
        ),
        seed=seed,
        generation=(
            build_generation_config(
                raw,
                base_seed=seed,
            )
        ),
        dataset_mix=(
            build_dataset_mix_config(
                raw,
                base_seed=seed,
            )
        ),
        budget=(
            build_budget_config(
                raw
            )
        ),
        trainer=(
            build_llamafactory_config(
                raw
            )
        ),
        training_overrides=(
            build_training_overrides(
                raw
            )
        ),
    )

def print_resolved_config(
    config: RecursiveExperimentConfig,
    *,
    output_root: Path,
) -> None:
    print("=" * 72)
    print("FAIRNESS-COLLAPSE EXPERIMENT")
    print("=" * 72)
    print(
        f"Experiment:       "
        f"{config.experiment_name}"
    )
    print(
        f"Seed:             "
        f"{config.seed}"
    )
    print(
        f"Iterations:       "
        f"{config.num_iterations}"
    )
    print(
        f"Human checkpoint: "
        f"{config.human_checkpoint}"
    )
    print(
        f"Prepared train:   "
        f"{config.prepared_train_path}"
    )
    print(
        f"Train tokenizer:  "
        f"{config.training_tokenizer}"
    )
    print(
        f"Output root:      "
        f"{output_root}"
    )

    print()
    print("Generation:")
    print(
        f"  temperature        "
        f"{config.generation.temperature}"
    )
    print(
        f"  top_p              "
        f"{config.generation.top_p}"
    )
    print(
        f"  max_new_tokens     "
        f"{config.generation.max_new_tokens}"
    )
    print(
        f"  batch_size         "
        f"{config.generation.batch_size}"
    )
    print(
        f"  suffix length match "
        f"{config.generation.match_human_suffix_length}"
    )
    print(
        f"  length tolerance    "
        f"{config.generation.length_tolerance}"
    )

    print()
    print("Contamination:")
    print(
        f"  synthetic docs      "
        f"{config.dataset_mix.synthetic_fraction:.2%}"
    )
    print(
        f"  stratify by         "
        f"{config.dataset_mix.stratify_by_metadata_key}"
    )

    print()
    print("Training budget:")
    print(
        f"  corpus equivalents "
        f"{config.budget.corpus_equivalents}"
    )
    print(
        f"  cutoff_len         "
        f"{config.budget.cutoff_len}"
    )
    print(
        f"  batch size         "
        f"{config.budget.per_device_train_batch_size}"
    )
    print(
        f"  grad accumulation  "
        f"{config.budget.gradient_accumulation_steps}"
    )

    print()
    print("Trainer:")
    print(
        f"  base YAML          "
        f"{config.trainer.base_yaml_path}"
    )
    print(
        f"  executable         "
        f"{config.trainer.executable}"
    )

    print()
    print(
        "Protocol: matched human-only + recursive seeded trajectory."
    )
    print(
        "The human prefixes and model-specific HCE budget are frozen "
        "before recursive training begins."
    )
    print()

def main() -> None:
    args = parse_args()

    raw = load_yaml(
        args.config
    )

    config = build_experiment_config(
        raw=raw,
        args=args,
    )

    print_resolved_config(
        config,
        output_root=args.output_root,
    )

    run_recursive_experiment(
        config=config,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
