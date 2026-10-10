from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from collapse.data.base import load_examples_jsonl
from collapse.training.budget import (
    TrainingBudget,
    TrainingBudgetConfig,
    fit_training_budget,
    load_training_budget,
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
            "Train one fairness-collapse checkpoint with a frozen "
            "human-corpus-equivalent token budget."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG),
        help=f"Experiment YAML. Default: {DEFAULT_CONFIG}",
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Training JSONL. Use prepared human train.jsonl for the "
            "human-only branch or generated.jsonl for the synthetic branch."
        ),
    )

    parser.add_argument(
        "--init-model",
        type=str,
        required=True,
        help=(
            "EXPLICIT checkpoint used to initialize this training run. "
            "Recursive: previous recursive checkpoint. "
            "Human-only: previous human-only checkpoint. "
            "Iterative ablation: original human checkpoint."
        ),
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory for this individual training run.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Stable descriptive run identifier.",
    )

    parser.add_argument(
        "--budget",
        type=Path,
        required=True,
        help=(
            "Path to the frozen per-model training_budget.json. "
            "If it does not yet exist, provide --human-reference and this "
            "script will fit it from HUMAN data only."
        ),
    )

    parser.add_argument(
        "--human-reference",
        type=Path,
        default=None,
        help=(
            "Prepared HUMAN train.jsonl used ONLY to fit a missing budget. "
            "Never point this at generated data."
        ),
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help=(
            "Tokenizer/model name used for token-budget accounting. "
            "Overrides config. Required when fitting a new budget if it "
            "cannot be resolved from config."
        ),
    )

    parser.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help="Training seed override.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Write trainer_data, manifests and runtime YAML, but do not "
            "execute LLaMA-Factory."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Allow an existing run/checkpoint directory to be reused. "
            "Use carefully."
        ),
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
            f"Top-level YAML must be a mapping, got {type(config)}."
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
        raise TypeError(f"`{name}` must be a YAML mapping.")

    return value


def resolve_tokenizer_name(
    config: dict[str, Any],
    cli_tokenizer: str | None,
) -> str:
    if cli_tokenizer:
        return cli_tokenizer

    training = get_section(config, "training")
    model = get_section(config, "model")

    candidates = [
        training.get("tokenizer_name_or_path"),
        model.get("tokenizer_name_or_path"),
        model.get("name"),
        model.get("name_or_path"),
    ]

    for value in candidates:
        if value:
            return str(value)

    raise ValueError(
        "Could not determine tokenizer for budget accounting. "
        "Pass --tokenizer or define model.name / "
        "model.tokenizer_name_or_path in the experiment config."
    )


def resolve_training_seed(
    config: dict[str, Any],
    cli_seed: int | None,
) -> int:
    if cli_seed is not None:
        return int(cli_seed)

    training = get_section(config, "training")
    experiment = get_section(config, "experiment")

    if "training_seed" in training:
        return int(training["training_seed"])

    if "seed" in training:
        return int(training["seed"])

    if "seed" in experiment:
        return int(experiment["seed"])

    return 42


def build_budget_config(
    config: dict[str, Any],
) -> TrainingBudgetConfig:
    training = get_section(config, "training")

    budget_cfg = training.get("budget", {})

    if budget_cfg is None:
        budget_cfg = {}

    if not isinstance(budget_cfg, dict):
        raise TypeError("training.budget must be a YAML mapping.")

    def pick(key: str, default: Any) -> Any:
        if key in budget_cfg:
            return budget_cfg[key]
        if key in training:
            return training[key]
        return default

    return TrainingBudgetConfig(
        cutoff_len=int(
            pick("cutoff_len", 512)
        ),
        per_device_train_batch_size=int(
            pick("per_device_train_batch_size", 2)
        ),
        gradient_accumulation_steps=int(
            pick("gradient_accumulation_steps", 16)
        ),
        world_size=int(
            pick("world_size", 1)
        ),
        corpus_equivalents=float(
            pick("corpus_equivalents", 1.0)
        ),
        add_special_tokens=bool(
            pick("add_special_tokens", False)
        ),
        add_eos_per_document=bool(
            pick("add_eos_per_document", True)
        ),
    )


def build_backend_config(
    config: dict[str, Any],
    *,
    force: bool,
) -> LlamaFactoryConfig:
    training = get_section(config, "training")

    backend = training.get("llamafactory", {})

    if backend is None:
        backend = {}

    if not isinstance(backend, dict):
        raise TypeError(
            "training.llamafactory must be a YAML mapping."
        )

    base_yaml_path = (
        backend.get("base_yaml_path")
        or training.get("base_yaml_path")
    )

    if not base_yaml_path:
        raise ValueError(
            "Missing LLaMA-Factory base YAML. Set "
            "training.llamafactory.base_yaml_path or "
            "training.base_yaml_path."
        )

    return LlamaFactoryConfig(
        base_yaml_path=str(base_yaml_path),
        executable=str(
            backend.get(
                "executable",
                "llamafactory-cli",
            )
        ),
        text_column=str(
            backend.get(
                "text_column",
                "text",
            )
        ),
        dataset_prompt_field=str(
            backend.get(
                "dataset_prompt_field",
                "prompt",
            )
        ),
        overwrite_output_dir=bool(
            force
            or backend.get(
                "overwrite_output_dir",
                False,
            )
        ),
        stream_logs=bool(
            backend.get(
                "stream_logs",
                True,
            )
        ),
    )


def resolve_extra_overrides(
    config: dict[str, Any],
) -> dict[str, Any]:
    training = get_section(config, "training")

    value = training.get(
        "extra_overrides",
        {},
    )

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise TypeError(
            "training.extra_overrides must be a YAML mapping."
        )

    return dict(value)


def get_or_fit_budget(
    *,
    budget_path: Path,
    human_reference_path: Path | None,
    tokenizer_name_or_path: str,
    config: TrainingBudgetConfig,
) -> tuple[TrainingBudget, str]:

    if budget_path.exists():
        budget = load_training_budget(
            budget_path
        )

        if (
            budget.tokenizer_name_or_path
            != tokenizer_name_or_path
        ):
            raise ValueError(
                "Existing budget tokenizer does not match requested "
                "tokenizer:\n"
                f"  budget:    {budget.tokenizer_name_or_path}\n"
                f"  requested: {tokenizer_name_or_path}\n"
                "Use a separate budget file for each model/tokenizer."
            )

        return budget, "loaded"

    if human_reference_path is None:
        raise FileNotFoundError(
            f"Budget does not exist: {budget_path}\n"
            "To create it safely, provide --human-reference pointing to "
            "the prepared HUMAN train.jsonl."
        )

    if not human_reference_path.exists():
        raise FileNotFoundError(
            f"Human reference does not exist: {human_reference_path}"
        )

    human_examples = load_examples_jsonl(
        human_reference_path
    )
    synthetic_ids = [
        example.id
        for example in human_examples
        if example.source == "synthetic"
    ]

    if synthetic_ids:
        preview = ", ".join(
            synthetic_ids[:5]
        )

        raise ValueError(
            "--human-reference contains examples marked synthetic. "
            "Refusing to fit the human-corpus-equivalent budget.\n"
            f"Examples: {preview}"
        )

    budget = fit_training_budget(
        human_examples=human_examples,
        tokenizer_name_or_path=(
            tokenizer_name_or_path
        ),
        config=config,
    )

    save_training_budget(
        budget,
        budget_path,
    )

    return budget, "fitted"


def save_launcher_manifest(
    *,
    path: Path,
    config_path: Path,
    input_path: Path,
    init_model: str,
    run_name: str,
    training_seed: int,
    budget_path: Path,
    budget_status: str,
    tokenizer_name_or_path: str,
    dry_run: bool,
) -> None:
    data = {
        "config": str(config_path),
        "input": str(input_path),
        "init_model": init_model,
        "run_name": run_name,
        "training_seed": training_seed,
        "budget_path": str(
            budget_path
        ),
        "budget_status": (
            budget_status
        ),
        "tokenizer_name_or_path": (
            tokenizer_name_or_path
        ),
        "dry_run": dry_run,
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

def main() -> None:
    args = parse_args()

    raw_config = load_yaml(
        args.config
    )

    if not args.input.exists():
        raise FileNotFoundError(
            f"Training dataset does not exist: {args.input}"
        )

    tokenizer_name = resolve_tokenizer_name(
        raw_config,
        args.tokenizer,
    )

    training_seed = resolve_training_seed(
        raw_config,
        args.training_seed,
    )

    budget_config = build_budget_config(
        raw_config
    )

    backend_config = build_backend_config(
        raw_config,
        force=args.force,
    )

    extra_overrides = resolve_extra_overrides(
        raw_config
    )

    args.run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    budget, budget_status = get_or_fit_budget(
        budget_path=args.budget,
        human_reference_path=(
            args.human_reference
        ),
        tokenizer_name_or_path=(
            tokenizer_name
        ),
        config=budget_config,
    )

    training_examples = load_examples_jsonl(
        args.input
    )

    print("=" * 72)
    print("FAIRNESS-COLLAPSE TRAIN ONE CHECKPOINT")
    print("=" * 72)
    print(f"Config:          {args.config}")
    print(f"Input:           {args.input}")
    print(f"Examples:        {len(training_examples):,}")
    print(f"Init model:      {args.init_model}")
    print(f"Run name:        {args.run_name}")
    print(f"Run dir:         {args.run_dir}")
    print(f"Training seed:   {training_seed}")
    print(f"Tokenizer:       {tokenizer_name}")
    print(f"Budget:          {args.budget} ({budget_status})")
    print()
    print("Frozen training budget:")
    print(
        f"  human reference tokens   "
        f"{budget.reference_training_tokens:,}"
    )
    print(
        f"  target tokens            "
        f"{budget.target_training_tokens:,}"
    )
    print(
        f"  tokens / optimizer step  "
        f"{budget.tokens_per_optimizer_step:,}"
    )
    print(
        f"  max optimizer steps      "
        f"{budget.max_steps:,}"
    )
    print(
        f"  nominal exposure         "
        f"{budget.nominal_training_tokens:,}"
    )
    print(
        f"  step-rounding overshoot  "
        f"{budget.token_overshoot_fraction:.3%}"
    )
    save_training_budget(
        budget,
        args.run_dir / "training_budget.json",
    )

    shutil.copy2(
        args.config,
        args.run_dir / "experiment_config.yaml",
    )

    save_launcher_manifest(
        path=args.run_dir / "launcher_manifest.json",
        config_path=args.config,
        input_path=args.input,
        init_model=args.init_model,
        run_name=args.run_name,
        training_seed=training_seed,
        budget_path=args.budget,
        budget_status=budget_status,
        tokenizer_name_or_path=tokenizer_name,
        dry_run=args.dry_run,
    )

    run = train_with_llamafactory(
        examples=training_examples,
        model_name_or_path=args.init_model,
        run_dir=args.run_dir,
        run_name=args.run_name,
        budget=budget,
        training_seed=training_seed,
        backend_config=backend_config,
        extra_overrides=(
            extra_overrides
        ),
        dry_run=args.dry_run,
    )

    print()
    print("Training launcher finished.")
    print(f"  checkpoint:     {run.output_dir}")
    print(f"  runtime YAML:   {run.runtime_yaml}")
    print(f"  trainer data:   {run.dataset_dir}")
    print(
        f"  budget report:  "
        f"{run.dataset_budget_report}"
    )

    if args.dry_run:
        print()
        print(
            "DRY RUN ONLY. Inspect train_runtime.yaml, "
            "trainer_data/dataset_info.json, and "
            "dataset_budget_report.json before running training."
        )


if __name__ == "__main__":
    main()
