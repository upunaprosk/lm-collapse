from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from collapse.data.base import load_examples_jsonl
from collapse.evaluation.synthetic_text import (
    SyntheticTextConfig,
    analyze_synthetic_text,
    save_synthetic_text_result,
)


DEFAULT_CONFIG = "configs/experiments/qwen_bios_recursive.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze paired human vs. generated biographies for lexical, "
            "pronoun, length, and style drift."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG),
        help=f"Experiment YAML. Default: {DEFAULT_CONFIG}",
    )

    parser.add_argument(
        "--human",
        type=Path,
        required=True,
        help=(
            "Prepared human JSONL containing the original text and, for "
            "seeded continuation, frozen human_suffix."
        ),
    )

    parser.add_argument(
        "--synthetic",
        type=Path,
        required=True,
        help=(
            "Generated JSONL for one iteration, typically "
            ".../generation/iter_XX/generated.jsonl."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for synthetic-text analysis outputs.",
    )

    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=None,
        help="Override paired bootstrap replicate count.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Override confidence level.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override analysis bootstrap seed.",
    )

    parser.add_argument(
        "--min-group-size",
        type=int,
        default=None,
        help="Override minimum profession×gender cell size for tests.",
    )

    parser.add_argument(
        "--no-full-text",
        action="store_true",
        help="Skip full-biography analysis.",
    )

    parser.add_argument(
        "--no-suffix",
        action="store_true",
        help="Skip human-suffix vs. synthetic-suffix analysis.",
    )

    parser.add_argument(
        "--include-human-fallbacks",
        action="store_true",
        help=(
            "Include rows that were not actually regenerated in the "
            "full-text analysis. By default, rows without synthetic_suffix "
            "are excluded so unchanged fallback examples do not dilute "
            "the generated-text analysis."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )

    return parser.parse_args()

def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        value = yaml.safe_load(f)

    if value is None:
        return {}

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            "Top-level YAML config must be a mapping."
        )

    return value


def get_mapping(
    value: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    section = value.get(
        key,
        {},
    )

    if section is None:
        return {}

    if not isinstance(
        section,
        dict,
    ):
        raise TypeError(
            f"`{key}` must be a YAML mapping."
        )

    return section


def get_synthetic_text_section(
    raw: dict[str, Any],
) -> dict[str, Any]:

    analysis = get_mapping(
        raw,
        "analysis",
    )

    section = analysis.get(
        "synthetic_text"
    )

    if section is None:
        evaluation = get_mapping(
            raw,
            "evaluation",
        )

        section = evaluation.get(
            "synthetic_text",
            {},
        )

    if section is None:
        return {}

    if not isinstance(
        section,
        dict,
    ):
        raise TypeError(
            "`analysis.synthetic_text` must be a YAML mapping."
        )

    return section


def build_config(
    *,
    raw: dict[str, Any],
    args: argparse.Namespace,
) -> SyntheticTextConfig:
    section = get_synthetic_text_section(
        raw
    )

    lexical_categories = section.get(
        "lexical_categories"
    )

    if lexical_categories is not None:
        if not isinstance(
            lexical_categories,
            dict,
        ):
            raise TypeError(
                "synthetic_text.lexical_categories must be a mapping."
            )

        lexical_categories = {
            str(category): [
                str(value)
                for value in values
            ]
            for category, values
            in lexical_categories.items()
        }

    bootstrap_replicates = (
        args.bootstrap_replicates
        if args.bootstrap_replicates
        is not None
        else int(
            section.get(
                "bootstrap_replicates",
                5_000,
            )
        )
    )

    confidence = (
        args.confidence
        if args.confidence
        is not None
        else float(
            section.get(
                "confidence_level",
                0.95,
            )
        )
    )

    seed = (
        args.seed
        if args.seed is not None
        else int(
            section.get(
                "bootstrap_seed",
                34567,
            )
        )
    )

    min_group_size = (
        args.min_group_size
        if args.min_group_size
        is not None
        else int(
            section.get(
                "min_group_size",
                10,
            )
        )
    )

    analyze_full_text = (
        False
        if args.no_full_text
        else bool(
            section.get(
                "analyze_full_text",
                True,
            )
        )
    )

    analyze_suffix = (
        False
        if args.no_suffix
        else bool(
            section.get(
                "analyze_suffix",
                True,
            )
        )
    )

    return SyntheticTextConfig(
        bootstrap_replicates=(
            bootstrap_replicates
        ),
        confidence_level=(
            confidence
        ),
        bootstrap_seed=seed,
        min_group_size=(
            min_group_size
        ),
        use_default_lexicons=bool(
            section.get(
                "use_default_lexicons",
                True,
            )
        ),
        lexical_categories=(
            lexical_categories
        ),
        analyze_full_text=(
            analyze_full_text
        ),
        analyze_suffix=(
            analyze_suffix
        ),
        save_per_example=bool(
            section.get(
                "save_per_example",
                True,
            )
        ),
    )

def filter_to_generated_pairs(
    *,
    human_examples,
    synthetic_examples,
    include_human_fallbacks: bool,
):

    if include_human_fallbacks:
        return (
            list(human_examples),
            list(synthetic_examples),
            {
                "filter": (
                    "all aligned examples, including unchanged fallbacks"
                ),
                "num_human_input": len(
                    human_examples
                ),
                "num_synthetic_input": len(
                    synthetic_examples
                ),
                "num_retained": len(
                    synthetic_examples
                ),
            },
        )

    human_by_id = {
        str(example.id): example
        for example in human_examples
    }

    synthetic_retained = [
        example
        for example in synthetic_examples
        if (
            example.synthetic_suffix
            is not None
            and bool(
                example.synthetic_suffix.strip()
            )
        )
    ]

    missing_human = [
        str(example.id)
        for example in synthetic_retained
        if str(example.id)
        not in human_by_id
    ]

    if missing_human:
        raise ValueError(
            "Synthetic examples have IDs absent from the human reference. "
            f"First IDs: {missing_human[:10]}"
        )

    human_retained = [
        human_by_id[
            str(example.id)
        ]
        for example in synthetic_retained
    ]

    return (
        human_retained,
        synthetic_retained,
        {
            "filter": (
                "actual generated rows only: non-empty synthetic_suffix"
            ),
            "num_human_input": len(
                human_examples
            ),
            "num_synthetic_input": len(
                synthetic_examples
            ),
            "num_retained": len(
                synthetic_retained
            ),
            "num_excluded_fallbacks": (
                len(
                    synthetic_examples
                )
                - len(
                    synthetic_retained
                )
            ),
        },
    )

def ensure_output_dir(
    path: Path,
    *,
    force: bool,
) -> None:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        any(
            path.iterdir()
        )
        and not force
    ):
        raise FileExistsError(
            f"Output directory is not empty: {path}\n"
            "Use --force only if overwriting/adding files is intentional."
        )


def print_top_changes(
    result,
    *,
    n: int = 15,
) -> None:

    rows = [
        stat
        for stat in result.profession_gender_stats
        if stat.feature.startswith(
            "lex_"
        )
        and stat.n > 0
    ]

    rows.sort(
        key=lambda stat: abs(
            stat.mean_difference
        ),
        reverse=True,
    )

    print()
    print(
        f"Top {min(n, len(rows))} profession×gender lexical changes "
        "(synthetic - human):"
    )

    for stat in rows[:n]:
        profession = (
            stat.profession
            if stat.profession is not None
            else (
                f"id={stat.profession_id}"
                if stat.profession_id
                is not None
                else "unknown"
            )
        )

        q = (
            f"{stat.bh_qvalue:.4g}"
            if stat.bh_qvalue
            is not None
            else "n/a"
        )

        print(
            f"  {stat.scope:6s} | "
            f"{stat.feature:32s} | "
            f"{profession:20s} | "
            f"{str(stat.gender):8s} | "
            f"Δ={stat.mean_difference:+.4f} | "
            f"n={stat.n:4d} | q={q}"
        )


def main() -> None:
    args = parse_args()

    ensure_output_dir(
        args.output_dir,
        force=args.force,
    )

    raw = load_yaml(
        args.config
    )

    config = build_config(
        raw=raw,
        args=args,
    )

    human_examples = (
        load_examples_jsonl(
            args.human
        )
    )

    synthetic_examples = (
        load_examples_jsonl(
            args.synthetic
        )
    )

    (
        human_examples,
        synthetic_examples,
        filter_info,
    ) = filter_to_generated_pairs(
        human_examples=(
            human_examples
        ),
        synthetic_examples=(
            synthetic_examples
        ),
        include_human_fallbacks=(
            args.include_human_fallbacks
        ),
    )

    if not synthetic_examples:
        raise ValueError(
            "No generated examples remain after filtering."
        )

    print("=" * 72)
    print("SYNTHETIC BIOGRAPHY ANALYSIS")
    print("=" * 72)
    print(
        f"Human:      {args.human}"
    )
    print(
        f"Synthetic:  {args.synthetic}"
    )
    print(
        f"Output:     {args.output_dir}"
    )
    print(
        f"Pairs:      {len(synthetic_examples)}"
    )
    print(
        f"Scopes:     "
        f"{'full ' if config.analyze_full_text else ''}"
        f"{'suffix' if config.analyze_suffix else ''}"
    )
    print()

    (
        result,
        paired_rows,
        per_example,
    ) = analyze_synthetic_text(
        human_examples=(
            human_examples
        ),
        synthetic_examples=(
            synthetic_examples
        ),
        config=config,
    )

    save_synthetic_text_result(
        result=result,
        paired_rows=paired_rows,
        per_example=per_example,
        output_dir=args.output_dir,
        save_per_example=(
            config.save_per_example
        ),
    )

    manifest = {
        "human_path": str(
            args.human.resolve()
        ),
        "synthetic_path": str(
            args.synthetic.resolve()
        ),
        "output_dir": str(
            args.output_dir.resolve()
        ),
        "config": asdict(
            config
        ),
        "filtering": (
            filter_info
        ),
        "num_analyzed_pairs": (
            result.num_pairs
        ),
    }

    with (
        args.output_dir
        / "synthetic_text_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print_top_changes(
        result
    )

    print()
    print("=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)
    print(
        f"Summary: "
        f"{args.output_dir / 'synthetic_text_summary.json'}"
    )


if __name__ == "__main__":
    main()
