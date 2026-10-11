from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from collapse.contamination.build_dataset import (
    BuildDatasetConfig,
    build_training_dataset,
)
from collapse.contamination.generate import (
    GenerationConfig,
    generate_seeded_dataset,
    save_generation_result,
)
from collapse.data.base import (
    load_examples_jsonl,
    save_examples_jsonl,
)
from collapse.records import Example
from collapse.training.budget import (
    TrainingBudget,
    TrainingBudgetConfig,
    fit_training_budget,
    save_training_budget,
)
from collapse.training.base import TrainingBackend
from collapse.training.llamafactory import (
    LlamaFactoryConfig,
    TrainingRun,
)


@dataclass(frozen=True)
class RecursiveExperimentConfig:

    experiment_name: str
    human_checkpoint: str
    prepared_train_path: str
    training_tokenizer: str

    num_iterations: int = 4
    seed: int = 42

    generation: GenerationConfig = field(
        default_factory=GenerationConfig
    )
    dataset_mix: BuildDatasetConfig = field(
        default_factory=BuildDatasetConfig
    )
    budget: TrainingBudgetConfig = field(
        default_factory=TrainingBudgetConfig
    )

    trainer: LlamaFactoryConfig | None = None
    training_overrides: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class IterationRecord:
    branch: str
    iteration: int

    input_checkpoint: str
    output_checkpoint: str

    training_data: str

    generation_dir: str | None
    train_dir: str

    generation_seed: int | None
    training_seed: int


@dataclass(frozen=True)
class ExperimentResult:
    experiment_name: str
    seed: int
    run_root: str
    budget_path: str

    human_iterations: list[IterationRecord]
    recursive_iterations: list[IterationRecord]

def human_training_seed(
    base_seed: int,
    iteration: int,
) -> int:
    return int(
        base_seed + 10_000 + iteration
    )


def recursive_generation_seed(
    base_seed: int,
    iteration: int,
) -> int:
    return int(
        base_seed + 20_000 + iteration
    )


def recursive_training_seed(
    base_seed: int,
    iteration: int,
) -> int:
    return int(
        base_seed + 30_000 + iteration
    )

def validate_config(
    config: RecursiveExperimentConfig,
) -> None:
    if config.num_iterations <= 0:
        raise ValueError(
            "num_iterations must be > 0."
        )

    if not config.experiment_name.strip():
        raise ValueError(
            "experiment_name must be non-empty."
        )

    if config.trainer is None:
        raise ValueError(
            "RecursiveExperimentConfig.trainer must be provided."
        )

    train_path = Path(
        config.prepared_train_path
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Prepared train JSONL does not exist: {train_path}"
        )


def validate_human_examples(
    examples: Sequence[Example],
) -> None:
    if not examples:
        raise ValueError(
            "Prepared human training corpus is empty."
        )

    bad_source = [
        example.id
        for example in examples
        if example.source != "human"
    ]

    if bad_source:
        raise ValueError(
            "Prepared human corpus contains examples not marked `human`.\n"
            f"Examples: {bad_source[:5]}"
        )

    missing_prefixes = [
        example.id
        for example in examples
        if (
            example.metadata.get(
                "seed_eligible",
                False,
            )
            and example.prefix_text is None
        )
    ]

    if missing_prefixes:
        raise ValueError(
            "Eligible human examples are missing frozen prefixes. "
            "Run scripts/prepare_dataset.py first.\n"
            f"Examples: {missing_prefixes[:5]}"
        )

def write_json(
    value: Any,
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
            value,
            f,
            indent=2,
            ensure_ascii=False,
        )


def save_experiment_manifest(
    *,
    config: RecursiveExperimentConfig,
    run_root: Path,
    budget: TrainingBudget,
) -> None:
    manifest = {
        "experiment_name": (
            config.experiment_name
        ),
        "seed": config.seed,
        "num_iterations": (
            config.num_iterations
        ),
        "human_checkpoint": (
            config.human_checkpoint
        ),
        "prepared_train_path": str(
            Path(
                config.prepared_train_path
            ).resolve()
        ),
        "training_tokenizer": (
            config.training_tokenizer
        ),
        "generation": asdict(
            config.generation
        ),
        "dataset_mix": asdict(
            config.dataset_mix
        ),
        "budget": asdict(
            budget
        ),
        "trainer": asdict(
            config.trainer
        ),
        "training_overrides": (
            config.training_overrides
        ),
        "protocol": {
            "human_branch": (
                "H = Train(H_0, D_human) once; fixed comparator for all t"
            ),
            "recursive_branch": (
                "D_gen,t = Generate(R_{t-1}, frozen human prefixes), with R_0=H_0; "
                "D_mix,t = Mix(D_human, D_gen,t, fixed synthetic fraction); "
                "R_t = Train(H_0, D_mix,t) with a matched fixed training seed"
            ),
            "prefix_policy": (
                "Fixed once from human training data; identical raw "
                "prefix strings reused across recursive iterations."
            ),
            "training_budget": (
                "Fixed model-specific human-corpus-equivalent budget "
                "reused across all human and recursive iterations."
            ),
        },
    }

    write_json(
        manifest,
        run_root
        / "experiment_manifest.json",
    )


def save_progress(
    *,
    run_root: Path,
    human_records: Sequence[IterationRecord],
    recursive_records: Sequence[IterationRecord],
) -> None:
    write_json(
        {
            "human": [
                asdict(record)
                for record in human_records
            ],
            "recursive": [
                asdict(record)
                for record in recursive_records
            ],
        },
        run_root / "progress.json",
    )

def fit_and_save_budget(
    *,
    human_examples: Sequence[Example],
    config: RecursiveExperimentConfig,
    run_root: Path,
) -> tuple[TrainingBudget, Path]:
    budget = fit_training_budget(
        human_examples=human_examples,
        tokenizer_name_or_path=(
            config.training_tokenizer
        ),
        config=config.budget,
    )

    budget_dir = (
        run_root / "budget"
    )

    budget_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    budget_path = (
        budget_dir
        / "training_budget.json"
    )

    save_training_budget(
        budget,
        budget_path,
    )

    return budget, budget_path

def run_human_trajectory(
    *,
    human_examples: Sequence[Example],
    config: RecursiveExperimentConfig,
    budget: TrainingBudget,
    run_root: Path,
) -> list[IterationRecord]:
    """Train one fixed human comparator H from the frozen H_0 checkpoint."""
    if config.trainer is None:
        raise ValueError(
            "RecursiveExperimentConfig.trainer must be provided."
        )

    backend = TrainingBackend(
        config=config.trainer,
        extra_overrides=config.training_overrides,
    )

    iteration = 1
    iteration_dir = run_root / "human" / "iter_01"
    checkpoint = iteration_dir / "checkpoint"
    train_seed = human_training_seed(config.seed, 1)

    if (checkpoint / "config.json").exists():
        print()
        print("=" * 72)
        print("HUMAN CONTROL — FIXED H [EXISTING CHECKPOINT — SKIP]")
        print("=" * 72)
        record = IterationRecord(
            branch="human",
            iteration=1,
            input_checkpoint=config.human_checkpoint,
            output_checkpoint=str(checkpoint.resolve()),
            training_data=str(Path(config.prepared_train_path).resolve()),
            generation_dir=None,
            train_dir=str(iteration_dir.resolve()),
            generation_seed=None,
            training_seed=train_seed,
        )
        return [record]

    if iteration_dir.exists() and any(iteration_dir.iterdir()):
        raise FileExistsError(
            "Found a partial fixed-human run without a completed checkpoint: "
            f"{iteration_dir}\nRefusing to overwrite it automatically."
        )

    print()
    print("=" * 72)
    print("HUMAN CONTROL — FIXED H")
    print("=" * 72)

    run_name = (
        f"{config.experiment_name}__seed{config.seed}__human__fixed"
    )
    train_run = backend.train(
        examples=human_examples,
        model_name_or_path=config.human_checkpoint,
        run_dir=iteration_dir,
        run_name=run_name,
        budget=budget,
        training_seed=train_seed,
    )

    record = IterationRecord(
        branch="human",
        iteration=1,
        input_checkpoint=config.human_checkpoint,
        output_checkpoint=train_run.output_dir,
        training_data=str(Path(config.prepared_train_path).resolve()),
        generation_dir=None,
        train_dir=str(iteration_dir.resolve()),
        generation_seed=None,
        training_seed=train_seed,
    )
    save_progress(
        run_root=run_root,
        human_records=[record],
        recursive_records=[],
    )
    return [record]

def generation_config_for_iteration(
    base: GenerationConfig,
    *,
    generation_seed: int,
) -> GenerationConfig:
    return GenerationConfig(
        temperature=base.temperature,
        top_p=base.top_p,
        repetition_penalty=(
            base.repetition_penalty
        ),
        max_new_tokens=(
            base.max_new_tokens
        ),
        min_new_tokens=(
            base.min_new_tokens
        ),
        batch_size=base.batch_size,
        generation_prompt=base.generation_prompt,
        match_human_suffix_length=(
            base.match_human_suffix_length
        ),
        length_tolerance=base.length_tolerance,
        buffer_tokens=base.buffer_tokens,
        generation_seed=(
            generation_seed
        ),
        torch_dtype=base.torch_dtype,
        device_map=base.device_map,
        add_special_tokens=(
            base.add_special_tokens
        ),
        max_prompt_tokens=(
            base.max_prompt_tokens
        ),
        keep_ineligible_human=(
            base.keep_ineligible_human
        ),
    )


def run_recursive_trajectory(
    *,
    human_examples: Sequence[Example],
    config: RecursiveExperimentConfig,
    budget: TrainingBudget,
    run_root: Path,
    existing_human_records: Sequence[IterationRecord],
) -> list[IterationRecord]:
    """Recursive data contamination with fixed model initialization.

    R_0 is config.human_checkpoint.  R_{t-1} generates the synthetic
    continuations for iteration t, but every R_t is trained fresh from R_0.
    The same optimizer/data seed as the fixed human comparator is reused so
    the H-vs-R_t difference is attributable to the training corpus.
    """
    records: list[IterationRecord] = []

    if config.trainer is None:
        raise ValueError(
            "RecursiveExperimentConfig.trainer must be provided."
        )

    backend = TrainingBackend(
        config=config.trainer,
        extra_overrides=config.training_overrides,
    )

    generation_checkpoint = config.human_checkpoint
    branch_root = run_root / "recursive"
    matched_train_seed = human_training_seed(config.seed, 1)

    for iteration in range(1, config.num_iterations + 1):
        iteration_dir = branch_root / f"iter_{iteration:02d}"
        generation_dir = iteration_dir / "generation"
        train_dir = iteration_dir / "train"
        checkpoint = train_dir / "checkpoint"
        training_data_path = generation_dir / "training_mix.jsonl"

        generation_seed = recursive_generation_seed(
            config.seed,
            iteration,
        )

        # Resume completed iterations without regenerating or retraining.
        if (checkpoint / "config.json").exists():
            if not training_data_path.exists():
                raise FileNotFoundError(
                    "Recursive checkpoint exists but its frozen training mix "
                    f"is missing: {training_data_path}"
                )
            print()
            print("=" * 72)
            print(
                f"RECURSIVE — ITERATION {iteration} "
                "[EXISTING CHECKPOINT — SKIP]"
            )
            print("=" * 72)
            record = IterationRecord(
                branch="recursive",
                iteration=iteration,
                input_checkpoint=config.human_checkpoint,
                output_checkpoint=str(checkpoint.resolve()),
                training_data=str(training_data_path.resolve()),
                generation_dir=str(generation_dir.resolve()),
                train_dir=str(train_dir.resolve()),
                generation_seed=generation_seed,
                training_seed=matched_train_seed,
            )
            records.append(record)
            generation_checkpoint = str(checkpoint.resolve())
            continue

        if iteration_dir.exists() and any(iteration_dir.iterdir()):
            raise FileExistsError(
                "Found a partial recursive iteration without a completed "
                f"checkpoint: {iteration_dir}\n"
                "Refusing to overwrite it automatically."
            )

        print()
        print("=" * 72)
        print(f"RECURSIVE — ITERATION {iteration}")
        print("=" * 72)
        print(f"Generation model: {generation_checkpoint}")
        print(f"Training init:    {config.human_checkpoint}")

        generation_cfg = generation_config_for_iteration(
            config.generation,
            generation_seed=generation_seed,
        )
        generation_result = generate_seeded_dataset(
            examples=human_examples,
            model_name_or_path=generation_checkpoint,
            iteration=iteration,
            config=generation_cfg,
        )
        save_generation_result(
            generation_result,
            generation_dir,
        )

        mix_config = BuildDatasetConfig(
            synthetic_fraction=config.dataset_mix.synthetic_fraction,
            seed=config.dataset_mix.seed + iteration,
            stratify_by_metadata_key=(
                config.dataset_mix.stratify_by_metadata_key
            ),
        )
        training_examples = build_training_dataset(
            human_examples=human_examples,
            generated_examples=generation_result.examples,
            config=mix_config,
        )
        save_examples_jsonl(
            training_examples,
            training_data_path,
        )

        num_synthetic_training = sum(
            example.source == "synthetic"
            for example in training_examples
        )
        write_json(
            {
                "config": asdict(mix_config),
                "num_examples": len(training_examples),
                "num_synthetic_examples": num_synthetic_training,
                "realized_synthetic_document_fraction": (
                    num_synthetic_training / len(training_examples)
                    if training_examples else 0.0
                ),
            },
            generation_dir / "training_mix_manifest.json",
        )

        run_name = (
            f"{config.experiment_name}__seed{config.seed}"
            f"__recursive__iter{iteration:02d}"
        )
        train_run = backend.train(
            examples=training_examples,
            model_name_or_path=config.human_checkpoint,
            run_dir=train_dir,
            run_name=run_name,
            budget=budget,
            training_seed=matched_train_seed,
        )

        record = IterationRecord(
            branch="recursive",
            iteration=iteration,
            input_checkpoint=config.human_checkpoint,
            output_checkpoint=train_run.output_dir,
            training_data=str(training_data_path.resolve()),
            generation_dir=str(generation_dir.resolve()),
            train_dir=str(train_dir.resolve()),
            generation_seed=generation_seed,
            training_seed=matched_train_seed,
        )
        records.append(record)
        generation_checkpoint = train_run.output_dir

        save_progress(
            run_root=run_root,
            human_records=existing_human_records,
            recursive_records=records,
        )

    return records

def run_recursive_experiment(
    *,
    config: RecursiveExperimentConfig,
    output_root: str | Path = "runs",
) -> ExperimentResult:

    validate_config(
        config
    )

    human_examples = (
        load_examples_jsonl(
            config.prepared_train_path
        )
    )

    validate_human_examples(
        human_examples
    )

    run_root = (
        Path(output_root)
        / config.experiment_name
        / f"seed_{config.seed}"
    )

    already_exists = run_root.exists() and any(run_root.iterdir())

    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if already_exists:
        print()
        print("=" * 72)
        print("RESUMING EXISTING EXPERIMENT")
        print("=" * 72)
        print(f"Run root: {run_root}")
        print("Completed checkpoints will be reused; partial iterations are not overwritten.")
    write_json(
        {
            "prepared_train_path": str(
                Path(
                    config.prepared_train_path
                ).resolve()
            ),
            "num_human_examples": len(
                human_examples
            ),
        },
        run_root / "input_manifest.json",
    )
    budget, budget_path = (
        fit_and_save_budget(
            human_examples=human_examples,
            config=config,
            run_root=run_root,
        )
    )

    save_experiment_manifest(
        config=config,
        run_root=run_root,
        budget=budget,
    )

    human_records = (
        run_human_trajectory(
            human_examples=human_examples,
            config=config,
            budget=budget,
            run_root=run_root,
        )
    )

    recursive_records = (
        run_recursive_trajectory(
            human_examples=human_examples,
            config=config,
            budget=budget,
            run_root=run_root,
            existing_human_records=(
                human_records
            ),
        )
    )

    result = ExperimentResult(
        experiment_name=(
            config.experiment_name
        ),
        seed=config.seed,
        run_root=str(
            run_root.resolve()
        ),
        budget_path=str(
            budget_path.resolve()
        ),
        human_iterations=(
            human_records
        ),
        recursive_iterations=(
            recursive_records
        ),
    )

    write_json(
        {
            "experiment_name": (
                result.experiment_name
            ),
            "seed": result.seed,
            "run_root": (
                result.run_root
            ),
            "budget_path": (
                result.budget_path
            ),
            "human_iterations": [
                asdict(record)
                for record
                in result.human_iterations
            ],
            "recursive_iterations": [
                asdict(record)
                for record
                in result.recursive_iterations
            ],
        },
        run_root / "result.json",
    )

    print()
    print("=" * 72)
    print("EXPERIMENT COMPLETE")
    print("=" * 72)
    print(
        f"Run root: {run_root}"
    )
    print(
        f"Human checkpoints: "
        f"{len(human_records)}"
    )
    print(
        f"Recursive checkpoints: "
        f"{len(recursive_records)}"
    )

    return result
