from __future__ import annotations

import argparse
from pathlib import Path

from collapse.evaluation.aggregate import aggregate_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate completed fairness-collapse evaluations and paired "
            "statistics into tidy CSV/JSON tables."
        )
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help=(
            "Experiment root, e.g. runs/qwen_bios_recursive "
            "or a higher directory containing multiple experiments."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: <run-root>/aggregate"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    aggregate_results(
        run_root=args.run_root,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
