from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from collapse.records import Example
from collapse.training.budget import (
    DatasetBudgetReport,
    TrainingBudget,
    llamafactory_budget_overrides,
    report_dataset_against_budget,
    save_dataset_budget_report,
)


@dataclass(frozen=True)
class LlamaFactoryConfig:

    base_yaml_path: str
    executable: str = "llamafactory-cli"

    text_column: str = "text"
    dataset_prompt_field: str = "prompt"

    overwrite_output_dir: bool = False
    stream_logs: bool = True


@dataclass(frozen=True)
class TrainingRun:
    init_model: str
    output_dir: str
    runtime_yaml: str
    dataset_dir: str
    dataset_name: str
    dataset_budget_report: str
    return_code: int

def safe_dataset_name(
    run_name: str,
) -> str:
    cleaned = []

    for ch in run_name:
        if ch.isalnum() or ch in {"_", "-"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")

    value = "".join(cleaned).strip("_")

    if not value:
        value = "collapse_train"

    return value


def write_training_jsonl(
    examples: Sequence[Example],
    path: str | Path,
    *,
    text_column: str = "text",
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
        for example in examples:
            if not example.text.strip():
                raise ValueError(
                    f"Example {example.id!r} has empty training text."
                )

            f.write(
                json.dumps(
                    {
                        text_column: example.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_dataset_info(
    *,
    dataset_name: str,
    dataset_filename: str,
    path: str | Path,
    text_column: str = "text",
    prompt_field: str = "prompt",
) -> None:
    path = Path(path)

    data = {
        dataset_name: {
            "file_name": dataset_filename,
            "columns": {
                prompt_field: text_column,
            },
        }
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


def prepare_private_dataset(
    *,
    examples: Sequence[Example],
    run_dir: str | Path,
    run_name: str,
    config: LlamaFactoryConfig,
) -> tuple[str, Path]:

    run_dir = Path(run_dir)

    dataset_dir = (
        run_dir
        / "trainer_data"
    )

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_name = safe_dataset_name(
        run_name
    )

    train_filename = "train.jsonl"

    write_training_jsonl(
        examples,
        dataset_dir / train_filename,
        text_column=config.text_column,
    )

    write_dataset_info(
        dataset_name=dataset_name,
        dataset_filename=train_filename,
        path=dataset_dir / "dataset_info.json",
        text_column=config.text_column,
        prompt_field=config.dataset_prompt_field,
    )

    return (
        dataset_name,
        dataset_dir,
    )

def load_base_yaml(
    path: str | Path,
) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"LLaMA-Factory base YAML does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise TypeError(
            f"Expected YAML mapping in {path}, got {type(cfg)}."
        )

    return cfg


def build_runtime_config(
    *,
    init_model: str,
    dataset_name: str,
    dataset_dir: str | Path,
    output_dir: str | Path,
    budget: TrainingBudget,
    training_seed: int,
    backend_config: LlamaFactoryConfig,
    extra_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:

    cfg = load_base_yaml(
        backend_config.base_yaml_path
    )

    cfg["model_name_or_path"] = (
        init_model
    )

    cfg["dataset"] = dataset_name
    cfg["dataset_dir"] = str(
        Path(dataset_dir).resolve()
    )

    cfg["output_dir"] = str(
        Path(output_dir).resolve()
    )

    cfg["seed"] = int(
        training_seed
    )

    cfg["data_seed"] = int(
        training_seed
    )

    cfg.update(
        llamafactory_budget_overrides(
            budget
        )
    )

    cfg.pop(
        "num_train_epochs",
        None,
    )

    cfg["overwrite_output_dir"] = (
        backend_config.overwrite_output_dir
    )

    if extra_overrides:
        cfg.update(
            extra_overrides
        )

    return cfg


def save_runtime_yaml(
    cfg: dict[str, Any],
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
        yaml.safe_dump(
            cfg,
            f,
            sort_keys=False,
        )

def validate_output_dir(
    output_dir: str | Path,
    *,
    overwrite: bool,
) -> Path:
    output_dir = Path(
        output_dir
    )

    if output_dir.exists():
        existing = list(
            output_dir.iterdir()
        )

        if existing and not overwrite:
            raise FileExistsError(
                f"Checkpoint output directory is not empty: {output_dir}\n"
                "Use a fresh run directory or explicitly enable "
                "overwrite_output_dir."
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def validate_budget_matches_runtime(
    runtime_cfg: dict[str, Any],
    budget: TrainingBudget,
) -> None:

    checks = {
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
    }

    mismatches = []

    for key, expected in checks.items():
        actual = runtime_cfg.get(
            key
        )

        if actual != expected:
            mismatches.append(
                f"{key}: expected {expected!r}, got {actual!r}"
            )

    if mismatches:
        raise ValueError(
            "Runtime YAML no longer matches frozen training budget:\n  - "
            + "\n  - ".join(mismatches)
        )


def run_llamafactory(
    *,
    runtime_yaml: str | Path,
    backend_config: LlamaFactoryConfig,
    dry_run: bool = False,
) -> int:
    command = [
        backend_config.executable,
        "train",
        str(
            Path(runtime_yaml).resolve()
        ),
    ]

    print()
    print("LLaMA-Factory command:")
    print(
        "  "
        + " ".join(command)
    )

    if dry_run:
        print(
            "DRY RUN: training command was not executed."
        )
        return 0

    if backend_config.stream_logs:
        completed = subprocess.run(
            command,
            check=False,
        )
    else:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )

        if completed.stdout:
            print(
                completed.stdout
            )

        if completed.stderr:
            print(
                completed.stderr
            )

    if completed.returncode != 0:
        raise RuntimeError(
            "LLaMA-Factory training failed with "
            f"return code {completed.returncode}."
        )

    return int(
        completed.returncode
    )

def save_training_manifest(
    *,
    path: str | Path,
    run_name: str,
    init_model: str,
    training_seed: int,
    dataset_name: str,
    dataset_dir: str | Path,
    output_dir: str | Path,
    runtime_yaml: str | Path,
    budget: TrainingBudget,
    dataset_report: DatasetBudgetReport,
    backend_config: LlamaFactoryConfig,
) -> None:
    path = Path(path)

    manifest = {
        "run_name": run_name,
        "init_model": init_model,
        "training_seed": training_seed,
        "dataset_name": dataset_name,
        "dataset_dir": str(
            Path(dataset_dir).resolve()
        ),
        "output_dir": str(
            Path(output_dir).resolve()
        ),
        "runtime_yaml": str(
            Path(runtime_yaml).resolve()
        ),
        "budget": asdict(
            budget
        ),
        "dataset_budget_report": asdict(
            dataset_report
        ),
        "llamafactory": asdict(
            backend_config
        ),
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


def train_with_llamafactory(
    *,
    examples: Sequence[Example],
    init_model: str,
    run_dir: str | Path,
    run_name: str,
    budget: TrainingBudget,
    training_seed: int,
    backend_config: LlamaFactoryConfig,
    extra_overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> TrainingRun:

    if not examples:
        raise ValueError(
            "Cannot train on an empty dataset."
        )

    run_dir = Path(
        run_dir
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir = (
        run_dir
        / "checkpoint"
    )

    validate_output_dir(
        checkpoint_dir,
        overwrite=(
            backend_config.overwrite_output_dir
        ),
    )

    (
        dataset_name,
        dataset_dir,
    ) = prepare_private_dataset(
        examples=examples,
        run_dir=run_dir,
        run_name=run_name,
        config=backend_config,
    )

    dataset_report = (
        report_dataset_against_budget(
            examples=examples,
            budget=budget,
        )
    )

    report_path = (
        run_dir
        / "dataset_budget_report.json"
    )

    save_dataset_budget_report(
        dataset_report,
        report_path,
    )
    runtime_cfg = (
        build_runtime_config(
            init_model=init_model,
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            output_dir=checkpoint_dir,
            budget=budget,
            training_seed=(
                training_seed
            ),
            backend_config=(
                backend_config
            ),
            extra_overrides=(
                extra_overrides
            ),
        )
    )

    validate_budget_matches_runtime(
        runtime_cfg,
        budget,
    )

    runtime_yaml = (
        run_dir
        / "train_runtime.yaml"
    )

    save_runtime_yaml(
        runtime_cfg,
        runtime_yaml,
    )

    shutil.copy2(
        backend_config.base_yaml_path,
        run_dir
        / "train_template.yaml",
    )

    manifest_path = (
        run_dir
        / "training_manifest.json"
    )

    save_training_manifest(
        path=manifest_path,
        run_name=run_name,
        init_model=init_model,
        training_seed=training_seed,
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        output_dir=checkpoint_dir,
        runtime_yaml=runtime_yaml,
        budget=budget,
        dataset_report=(
            dataset_report
        ),
        backend_config=(
            backend_config
        ),
    )

    return_code = run_llamafactory(
        runtime_yaml=runtime_yaml,
        backend_config=(
            backend_config
        ),
        dry_run=dry_run,
    )

    return TrainingRun(
        init_model=init_model,
        output_dir=str(
            checkpoint_dir.resolve()
        ),
        runtime_yaml=str(
            runtime_yaml.resolve()
        ),
        dataset_dir=str(
            dataset_dir.resolve()
        ),
        dataset_name=dataset_name,
        dataset_budget_report=str(
            report_path.resolve()
        ),
        return_code=return_code,
    )
