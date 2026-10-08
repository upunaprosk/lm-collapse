from __future__ import annotations

import argparse
import json
from pathlib import Path

from collapse.evaluation.bootstrap import (
    BootstrapConfig,
    paired_stratified_bootstrap,
    save_bootstrap_result,
)
from collapse.evaluation.mmlu_stats import (
    MMLUStatsConfig,
    paired_mmlu_stats,
    save_mmlu_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired statistical comparisons between one matched "
            "human-only checkpoint H_t and recursive checkpoint R_t."
        )
    )

    parser.add_argument(
        "--human-eval-dir",
        type=Path,
        required=True,
        help=(
            "Evaluation directory for matched human-only checkpoint, "
            "containing bios/ and mmlu/."
        ),
    )

    parser.add_argument(
        "--recursive-eval-dir",
        type=Path,
        required=True,
        help=(
            "Evaluation directory for recursive checkpoint, "
            "containing bios/ and mmlu/."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for paired statistical outputs.",
    )

    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )

    parser.add_argument(
        "--mmlu-permutations",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--skip-bios",
        action="store_true",
    )

    parser.add_argument(
        "--skip-mmlu",
        action="store_true",
    )

    parser.add_argument(
        "--save-distributions",
        action="store_true",
    )

    return parser.parse_args()


def require_path(
    path: Path,
    description: str,
) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} does not exist: {path}"
        )

    return path


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "human_eval_dir": str(
            args.human_eval_dir.resolve()
        ),
        "recursive_eval_dir": str(
            args.recursive_eval_dir.resolve()
        ),
        "confidence_level": (
            args.confidence
        ),
        "bootstrap_replicates": (
            args.bootstrap_replicates
        ),
        "bios": None,
        "mmlu": None,
    }

    if not args.skip_bios:
        human_bios = require_path(
            args.human_eval_dir
            / "bios"
            / "bios_predictions.jsonl",
            "human Bias-in-Bios predictions",
        )

        recursive_bios = require_path(
            args.recursive_eval_dir
            / "bios"
            / "bios_predictions.jsonl",
            "recursive Bias-in-Bios predictions",
        )

        bios_result, bios_distribution = (
            paired_stratified_bootstrap(
                human_predictions_path=(
                    human_bios
                ),
                comparison_predictions_path=(
                    recursive_bios
                ),
                config=BootstrapConfig(
                    n_bootstrap=(
                        args.bootstrap_replicates
                    ),
                    confidence_level=(
                        args.confidence
                    ),
                    seed=args.seed,
                    correctness_field=(
                        "correct_mean"
                    ),
                    save_distributions=(
                        args.save_distributions
                    ),
                ),
            )
        )

        bios_output = (
            args.output_dir
            / "bios"
        )

        save_bootstrap_result(
            result=bios_result,
            distributions=(
                bios_distribution
            ),
            output_dir=bios_output,
            save_distributions=(
                args.save_distributions
            ),
        )

        summary["bios"] = {
            "human_rms_eo_gap": (
                bios_result.human_rms_eo_gap
            ),
            "recursive_rms_eo_gap": (
                bios_result.comparison_rms_eo_gap
            ),
            "delta_eo": (
                bios_result.rms_eo_gap_difference.estimate
            ),
            "ci_low": (
                bios_result.rms_eo_gap_difference.ci_low
            ),
            "ci_high": (
                bios_result.rms_eo_gap_difference.ci_high
            ),
            "significant_increase": (
                bios_result.rms_eo_gap_difference.significant_increase
            ),
        }

    if not args.skip_mmlu:
        human_mmlu_dir = require_path(
            args.human_eval_dir
            / "mmlu",
            "human MMLU directory",
        )

        recursive_mmlu_dir = require_path(
            args.recursive_eval_dir
            / "mmlu",
            "recursive MMLU directory",
        )

        (
            mmlu_result,
            mmlu_distribution,
        ) = paired_mmlu_stats(
            human_samples_path=(
                human_mmlu_dir
            ),
            comparison_samples_path=(
                recursive_mmlu_dir
            ),
            config=MMLUStatsConfig(
                n_bootstrap=(
                    args.bootstrap_replicates
                ),
                confidence_level=(
                    args.confidence
                ),
                seed=(
                    args.seed + 1
                ),
                n_permutations=(
                    args.mmlu_permutations
                ),
                save_distribution=(
                    args.save_distributions
                ),
            ),
        )

        mmlu_output = (
            args.output_dir
            / "mmlu"
        )

        save_mmlu_stats(
            result=mmlu_result,
            bootstrap_distribution=(
                mmlu_distribution
            ),
            output_dir=mmlu_output,
            save_distribution=(
                args.save_distributions
            ),
        )

        summary["mmlu"] = {
            "human_accuracy": (
                mmlu_result.human_accuracy
            ),
            "recursive_accuracy": (
                mmlu_result.comparison_accuracy
            ),
            "delta_accuracy": (
                mmlu_result.delta_accuracy
            ),
            "ci_low": (
                mmlu_result.bootstrap.ci_low
            ),
            "ci_high": (
                mmlu_result.bootstrap.ci_high
            ),
            "significant_degradation": (
                mmlu_result.significant_performance_degradation
            ),
            "permutation_pvalue_two_sided": (
                mmlu_result.permutation_pvalue_two_sided
            ),
            "mcnemar_pvalue_two_sided": (
                mmlu_result.mcnemar_pvalue_two_sided
            ),
        }

    fairness = (
        summary["bios"][
            "significant_increase"
        ]
        if summary["bios"]
        is not None
        else None
    )

    performance = (
        summary["mmlu"][
            "significant_degradation"
        ]
        if summary["mmlu"]
        is not None
        else None
    )

    if fairness is True and performance is True:
        state = (
            "fairness + performance significant"
        )
    elif fairness is True:
        state = "fairness significant"
    elif performance is True:
        state = "performance significant"
    elif fairness is None or performance is None:
        state = "incomplete statistics"
    else:
        state = "neither significant"

    summary[
        "significance_state"
    ] = state

    summary_path = (
        args.output_dir
        / "paired_comparison_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 72)
    print("PAIRED CHECKPOINT COMPARISON COMPLETE")
    print("=" * 72)
    print(
        f"State:   {state}"
    )
    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
