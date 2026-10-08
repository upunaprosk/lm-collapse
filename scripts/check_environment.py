from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED_MODULES = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "numpy",
    "scipy",
    "yaml",
]


def module_version(module_name: str) -> str | None:
    package_candidates = {
        "yaml": ["PyYAML"],
        "torch": ["torch"],
        "transformers": ["transformers"],
        "datasets": ["datasets"],
        "accelerate": ["accelerate"],
        "numpy": ["numpy"],
        "scipy": ["scipy"],
    }.get(module_name, [module_name])

    for candidate in package_candidates:
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue

    return None


def command_version(command: str, args: list[str]) -> dict:
    path = shutil.which(command)

    if path is None:
        return {
            "available": False,
            "path": None,
            "version_output": None,
        }

    try:
        completed = subprocess.run(
            [path, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        output = (completed.stdout or completed.stderr).strip()
    except Exception as exc:
        output = f"ERROR: {exc}"

    return {
        "available": True,
        "path": path,
        "version_output": output[:2000],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check runtime dependencies for fairness-collapse experiments."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    args = parser.parse_args()

    modules = {}
    missing_modules = []

    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
            modules[name] = {
                "available": True,
                "version": module_version(name),
            }
        except Exception as exc:
            modules[name] = {
                "available": False,
                "version": None,
                "error": repr(exc),
            }
            missing_modules.append(name)

    torch_info = {}
    if modules.get("torch", {}).get("available"):
        import torch

        torch_info = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_version": torch.version.cuda,
            "bf16_supported": (
                bool(torch.cuda.is_bf16_supported())
                if torch.cuda.is_available()
                else False
            ),
            "devices": [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ],
        }

    commands = {
        "llamafactory-cli": command_version(
            "llamafactory-cli",
            ["version"],
        ),
        "lm-eval": command_version(
            "lm-eval",
            ["--version"],
        ),
    }

    try:
        import collapse
        collapse_import = {
            "available": True,
            "path": str(Path(collapse.__file__).resolve()),
        }
    except Exception as exc:
        collapse_import = {
            "available": False,
            "error": repr(exc),
        }

    blockers = []

    if missing_modules:
        blockers.append(
            "Missing Python modules: " + ", ".join(missing_modules)
        )

    if not collapse_import["available"]:
        blockers.append(
            "The local `collapse` package is not importable. Run `pip install -e .`."
        )

    if not commands["llamafactory-cli"]["available"]:
        blockers.append(
            "`llamafactory-cli` is not on PATH. Install LLaMA-Factory before training."
        )

    if not commands["lm-eval"]["available"]:
        blockers.append(
            "`lm-eval` is not on PATH. Install `lm_eval[hf]` before MMLU evaluation."
        )

    if torch_info and not torch_info.get("cuda_available"):
        blockers.append(
            "PyTorch cannot see CUDA; GPU generation/training cannot run."
        )

    report = {
        "python": sys.version,
        "modules": modules,
        "torch": torch_info,
        "commands": commands,
        "collapse_import": collapse_import,
        "blockers": blockers,
        "ready_for_gpu_smoke_test": not blockers,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
