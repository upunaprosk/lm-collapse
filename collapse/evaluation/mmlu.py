from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MMLUConfig:
    """
    MMLU evaluation through EleutherAI's lm-evaluation-harness.
    """

    executable: str = "lm-eval"

    task: str = "mmlu"
    num_fewshot: int = 5

    batch_size: str = "auto"
    device: str = "cuda:0"

    dtype: str = "bfloat16"

    seed: str = "0,1234,1234,1234"

    log_samples: bool = True

    apply_chat_template: bool = False

    limit: int | float | None = None

    model_args: dict[str, Any] | None = None


@dataclass(frozen=True)
class MMLUSummary:
    model_name_or_path: str

    task: str
    num_fewshot: int

    accuracy: float | None
    accuracy_stderr: float | None

    subtask_accuracies: dict[str, float]

    raw_results_path: str
    command_path: str

    lm_eval_version: str | None


def require_lm_eval(
    executable: str,
) -> str:
    path = shutil.which(
        executable
    )

    if path is None:
        raise FileNotFoundError(
            f"Could not find {executable!r} on PATH.\n"
            "Install the Hugging Face-enabled lm-evaluation-harness, e.g.\n"
            '  pip install "lm_eval[hf]"\n'
            "and verify with:\n"
            "  lm-eval -h"
        )

    return path


def get_lm_eval_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    for package_name in ("lm_eval", "lm-eval"):
        try:
            return version(package_name)
        except PackageNotFoundError:
            continue

    return None

def encode_model_args(
    *,
    model_name_or_path: str,
    config: MMLUConfig,
) -> str:
    """
    Build lm-eval Hugging Face model_args.
    """

    args: dict[str, Any] = {
        "pretrained": (
            model_name_or_path
        ),
        "dtype": (
            config.dtype
        ),
    }

    if config.model_args:
        args.update(
            config.model_args
        )

    return ",".join(
        f"{key}={value}"
        for key, value
        in args.items()
    )


def build_command(
    *,
    model_name_or_path: str,
    output_dir: Path,
    config: MMLUConfig,
) -> list[str]:

    command = [
        config.executable,
        "run",

        "--model",
        "hf",

        "--model_args",
        encode_model_args(
            model_name_or_path=(
                model_name_or_path
            ),
            config=config,
        ),

        "--tasks",
        config.task,

        "--num_fewshot",
        str(
            config.num_fewshot
        ),

        "--batch_size",
        str(
            config.batch_size
        ),

        "--device",
        config.device,

        "--seed",
        config.seed,

        "--output_path",
        str(
            output_dir.resolve()
        ),
    ]

    if config.log_samples:
        command.append(
            "--log_samples"
        )

    if config.apply_chat_template:
        command.append(
            "--apply_chat_template"
        )

    if config.limit is not None:
        command.extend(
            [
                "--limit",
                str(
                    config.limit
                ),
            ]
        )

    return command

def _is_result_json(
    path: Path,
) -> bool:
    if not path.is_file():
        return False

    if path.suffix.lower() != ".json":
        return False

    if path.name in {
        "mmlu_summary.json",
        "mmlu_command.json",
    }:
        return False

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)
    except Exception:
        return False

    return (
        isinstance(
            data,
            dict,
        )
        and "results" in data
    )


def find_raw_results_json(
    output_dir: Path,
) -> Path:

    candidates = [
        path
        for path
        in output_dir.rglob("*.json")
        if _is_result_json(
            path
        )
    ]

    if not candidates:
        raise FileNotFoundError(
            "lm-eval finished but no results JSON containing a top-level "
            f"`results` field was found under {output_dir}."
        )

    candidates.sort(
        key=lambda path: (
            path.stat().st_mtime,
            str(path),
        ),
        reverse=True,
    )

    return candidates[0]


def _metric_value(
    entry: dict[str, Any],
    base_name: str,
) -> float | None:

    preferred = (
        f"{base_name},none"
    )

    if preferred in entry:
        value = entry[
            preferred
        ]

        if isinstance(
            value,
            (
                int,
                float,
            ),
        ):
            return float(
                value
            )

    if base_name in entry:
        value = entry[
            base_name
        ]

        if isinstance(
            value,
            (
                int,
                float,
            ),
        ):
            return float(
                value
            )

    for key, value in entry.items():
        if (
            key.startswith(
                base_name + ","
            )
            and isinstance(
                value,
                (
                    int,
                    float,
                ),
            )
        ):
            return float(
                value
            )

    return None


def extract_mmlu_summary(
    raw: dict[str, Any],
) -> tuple[
    float | None,
    float | None,
    dict[str, float],
]:

    results = raw.get(
        "results",
        {},
    )

    groups = raw.get(
        "groups",
        {},
    )

    aggregate_entry = None

    if (
        isinstance(groups, dict)
        and isinstance(
            groups.get("mmlu"),
            dict,
        )
    ):
        aggregate_entry = groups[
            "mmlu"
        ]

    elif (
        isinstance(results, dict)
        and isinstance(
            results.get("mmlu"),
            dict,
        )
    ):
        aggregate_entry = results[
            "mmlu"
        ]

    accuracy = None
    stderr = None

    if aggregate_entry is not None:
        accuracy = _metric_value(
            aggregate_entry,
            "acc",
        )

        stderr = _metric_value(
            aggregate_entry,
            "acc_stderr",
        )

    subtasks: dict[
        str,
        float
    ] = {}

    if isinstance(
        results,
        dict,
    ):
        for name, entry in results.items():
            if (
                not isinstance(
                    name,
                    str,
                )
                or not name.startswith(
                    "mmlu"
                )
                or not isinstance(
                    entry,
                    dict,
                )
            ):
                continue

            acc = _metric_value(
                entry,
                "acc",
            )

            if acc is not None:
                subtasks[
                    name
                ] = acc

    return (
        accuracy,
        stderr,
        subtasks,
    )


def save_command_manifest(
    *,
    command: list[str],
    model_name_or_path: str,
    config: MMLUConfig,
    output_dir: Path,
) -> Path:

    path = (
        output_dir
        / "mmlu_command.json"
    )

    data = {
        "command": command,
        "model_name_or_path": (
            model_name_or_path
        ),
        "config": asdict(
            config
        ),
        "lm_eval_version": (
            get_lm_eval_version()
        ),
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

    return path


def save_summary(
    summary: MMLUSummary,
    output_dir: Path,
) -> Path:

    path = (
        output_dir
        / "mmlu_summary.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            asdict(summary),
            f,
            indent=2,
            ensure_ascii=False,
        )

    return path


def evaluate_mmlu(
    *,
    model_name_or_path: str,
    output_dir: str | Path,
    config: MMLUConfig | None = None,
    dry_run: bool = False,
) -> MMLUSummary | None:
    """
    Run MMLU through lm-evaluation-harness.
    """

    if config is None:
        config = (
            MMLUConfig()
        )

    if config.num_fewshot < 0:
        raise ValueError(
            "num_fewshot must be >= 0."
        )

    require_lm_eval(
        config.executable
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_result_files = [
        path
        for path
        in output_dir.rglob(
            "*.json"
        )
        if _is_result_json(
            path
        )
    ]

    if existing_result_files:
        raise FileExistsError(
            f"MMLU output directory already contains lm-eval results: "
            f"{output_dir}"
        )

    command = build_command(
        model_name_or_path=(
            model_name_or_path
        ),
        output_dir=output_dir,
        config=config,
    )

    command_path = (
        save_command_manifest(
            command=command,
            model_name_or_path=(
                model_name_or_path
            ),
            config=config,
            output_dir=output_dir,
        )
    )

    print()
    print("MMLU command:")
    print(
        "  "
        + " ".join(
            command
        )
    )

    if dry_run:
        print(
            "DRY RUN: lm-eval was not executed."
        )

        return None

    environment = dict(
        os.environ
    )

    completed = subprocess.run(
        command,
        check=False,
        env=environment,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "lm-eval MMLU evaluation failed with "
            f"return code {completed.returncode}."
        )

    raw_results_path = (
        find_raw_results_json(
            output_dir
        )
    )

    with raw_results_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        raw = json.load(
            f
        )

    (
        accuracy,
        stderr,
        subtasks,
    ) = extract_mmlu_summary(
        raw
    )

    summary = MMLUSummary(
        model_name_or_path=(
            model_name_or_path
        ),
        task=config.task,
        num_fewshot=(
            config.num_fewshot
        ),
        accuracy=accuracy,
        accuracy_stderr=stderr,
        subtask_accuracies=(
            subtasks
        ),
        raw_results_path=str(
            raw_results_path.resolve()
        ),
        command_path=str(
            command_path.resolve()
        ),
        lm_eval_version=(
            get_lm_eval_version()
        ),
    )

    save_summary(
        summary,
        output_dir,
    )

    print()
    print("MMLU:")
    print(
        f"  model:       "
        f"{model_name_or_path}"
    )
    print(
        f"  shots:       "
        f"{config.num_fewshot}"
    )

    if accuracy is not None:
        print(
            f"  accuracy:    "
            f"{accuracy:.4f} "
            f"({100.0 * accuracy:.2f}%)"
        )
    else:
        print(
            "  accuracy:    could not extract aggregate; "
            "raw output was preserved"
        )

    print(
        f"  raw results: "
        f"{raw_results_path}"
    )

    return summary
