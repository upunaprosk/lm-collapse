from __future__ import annotations

import argparse
import json
from pathlib import Path

from collapse.evaluation.bootstrap import (
    BootstrapConfig,
    paired_stratified_bootstrap,
    save_bootstrap_result,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Paired H_t vs R_t bootstrap for fixed-pair Bias-in-Bios."
        )
    )
    p.add_argument(
        "--human-predictions",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--recursive-predictions",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=10_000,
    )
    p.add_argument(
        "--confidence",
        type=float,
        default=0.99,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=12345,
    )
    p.add_argument(
        "--save-distributions",
        action="store_true",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result, distributions = paired_stratified_bootstrap(
        human_predictions_path=args.human_predictions,
        comparison_predictions_path=args.recursive_predictions,
        config=BootstrapConfig(
            n_bootstrap=args.bootstrap_replicates,
            confidence_level=args.confidence,
            seed=args.seed,
            correctness_field="correct_pairwise",
            save_distributions=args.save_distributions,
        ),
    )

    save_bootstrap_result(
        result=result,
        distributions=distributions,
        output_dir=args.output_dir,
        save_distributions=args.save_distributions,
    )

    summary = {
        "human_accuracy": result.human_accuracy,
        "recursive_accuracy": result.comparison_accuracy,
        "delta_accuracy": result.accuracy_difference.estimate,
        "delta_accuracy_ci_low": result.accuracy_difference.ci_low,
        "delta_accuracy_ci_high": result.accuracy_difference.ci_high,
        "human_rms_eo_gap": result.human_rms_eo_gap,
        "recursive_rms_eo_gap": result.comparison_rms_eo_gap,
        "delta_eo": result.rms_eo_gap_difference.estimate,
        "delta_eo_ci_low": result.rms_eo_gap_difference.ci_low,
        "delta_eo_ci_high": result.rms_eo_gap_difference.ci_high,
        "significant_fairness_increase": (
            result.rms_eo_gap_difference.significant_increase
        ),
        "confidence_level": args.confidence,
    }

    with (
        args.output_dir / "pairwise_comparison_summary.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 72)
    print("PAIRWISE H_t vs R_t COMPARISON")
    print("=" * 72)
    print(
        "Accuracy: "
        f"H={100*result.human_accuracy:.2f}%  "
        f"R={100*result.comparison_accuracy:.2f}%  "
        f"Δ={100*result.accuracy_difference.estimate:+.2f} pp"
    )
    print(
        "EO GAP:   "
        f"H={100*result.human_rms_eo_gap:.2f}  "
        f"R={100*result.comparison_rms_eo_gap:.2f}  "
        f"Δ={100*result.rms_eo_gap_difference.estimate:+.2f}"
    )
    print(
        f"{100*args.confidence:.1f}% CI for ΔEO: "
        f"[{100*result.rms_eo_gap_difference.ci_low:+.2f}, "
        f"{100*result.rms_eo_gap_difference.ci_high:+.2f}]"
    )
    print(
        "Significant fairness increase:",
        result.rms_eo_gap_difference.significant_increase,
    )


if __name__ == "__main__":
    main()
