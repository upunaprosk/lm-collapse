from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f)

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise TypeError(
            f"Top-level YAML object must be a mapping, got {type(value)}."
        )

    return value


def save_yaml(value: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(value, f, sort_keys=False)


def deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    result = deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(
                result[key],
                value,
            )
        else:
            result[key] = deepcopy(value)

    return result


def require_mapping(
    value: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    section = value.get(key, {})

    if section is None:
        return {}

    if not isinstance(section, dict):
        raise TypeError(f"`{key}` must be a YAML mapping.")

    return section
