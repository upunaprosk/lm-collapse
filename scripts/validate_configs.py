from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


PLACEHOLDER_PATTERNS = (
    "CHANGE_ME",
    "TODO_MODEL",
    "<checkpoint>",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate fairness-collapse YAMLs before smoke/final runs."
        )
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
    )

    parser.add_argument(
        "--require-data",
        action="store_true",
        help=(
            "Also require prepared dataset paths referenced by experiment "
            "configs to exist."
        ),
    )

    parser.add_argument(
        "--require-human-checkpoints",
        action="store_true",
        help=(
            "Also require configured human_checkpoint directories to exist."
        ),
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f)

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise TypeError(
            f"{path}: top-level YAML must be a mapping."
        )

    return value


def nested(
    value: dict[str, Any],
    *keys: str,
):
    current: Any = value

    for key in keys:
        if not isinstance(current, dict):
            return None

        current = current.get(key)

    return current


def find_placeholders(
    value: Any,
    *,
    prefix: str = "",
) -> list[str]:
    found = []

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            found.extend(
                find_placeholders(
                    child,
                    prefix=child_prefix,
                )
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(
                find_placeholders(
                    child,
                    prefix=(
                        f"{prefix}[{index}]"
                    ),
                )
            )

    elif isinstance(value, str):
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern in value:
                found.append(
                    f"{prefix}={value!r}"
                )
                break

    return found


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()

    yaml_files = sorted(
        (root / "configs").rglob(
            "*.yaml"
        )
    )

    errors = []
    warnings = []

    for path in yaml_files:
        try:
            raw = load_yaml(path)
        except Exception as exc:
            errors.append(
                f"{path.relative_to(root)}: {exc}"
            )
            continue

        placeholders = find_placeholders(
            raw
        )

        for placeholder in placeholders:
            errors.append(
                f"{path.relative_to(root)}: unresolved placeholder "
                f"{placeholder}"
            )
        if (
            "configs/experiments"
            in str(
                path.relative_to(root)
            )
            .replace("\\", "/")
        ):
            exp_name = nested(
                raw,
                "experiment",
                "name",
            )

            if not exp_name:
                errors.append(
                    f"{path.relative_to(root)}: missing experiment.name"
                )

            iterations = nested(
                raw,
                "experiment",
                "iterations",
            )

            if iterations != 4:
                warnings.append(
                    f"{path.relative_to(root)}: iterations={iterations}; "
                    "main protocol expects 4"
                )

            temperature = nested(
                raw,
                "generation",
                "temperature",
            )

            if temperature != 0.9:
                warnings.append(
                    f"{path.relative_to(root)}: temperature={temperature}; "
                    "main protocol expects 0.9"
                )

            base_yaml = nested(
                raw,
                "training",
                "llamafactory",
                "base_yaml_path",
            )

            if base_yaml:
                candidate = root / str(
                    base_yaml
                )

                if not candidate.exists():
                    errors.append(
                        f"{path.relative_to(root)}: missing base YAML "
                        f"{candidate}"
                    )

            prepared_dir = nested(
                raw,
                "dataset",
                "prepared_dir",
            )

            if (
                args.require_data
                and prepared_dir
            ):
                train = (
                    root
                    / str(
                        prepared_dir
                    )
                    / "train.jsonl"
                )

                if not train.exists():
                    errors.append(
                        f"{path.relative_to(root)}: prepared train missing "
                        f"{train}"
                    )

            human_checkpoint = nested(
                raw,
                "model",
                "human_checkpoint",
            )

            if (
                args.require_human_checkpoints
                and human_checkpoint
            ):
                checkpoint = (
                    root
                    / str(
                        human_checkpoint
                    )
                )

                if not checkpoint.exists():
                    errors.append(
                        f"{path.relative_to(root)}: human checkpoint missing "
                        f"{checkpoint}"
                    )

    report = {
        "repo_root": str(root),
        "num_yaml_files": len(
            yaml_files
        ),
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )

    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
