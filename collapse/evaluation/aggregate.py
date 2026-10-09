from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

@dataclass(frozen=True)
class CheckpointRow:
    experiment: str
    model_key: str | None

    seed: int | None
    branch: str | None
    iteration: int | None

    checkpoint: str
    evaluation_summary_path: str

    heldout_human_ppl: float | None

    mmlu_accuracy: float | None
    mmlu_accuracy_percent: float | None

    bios_accuracy: float | None
    bios_rms_eo_gap: float | None
    bios_rms_eo_gap_stable: float | None

    pairwise_bios_accuracy: float | None
    pairwise_bios_rms_eo_gap: float | None


@dataclass
class MatchedRow:
    experiment: str
    model_key: str | None

    seed: int
    iteration: int

    human_checkpoint: str
    recursive_checkpoint: str

    human_ppl: float | None
    recursive_ppl: float | None
    delta_ppl: float | None
    ppl_increase: float | None

    human_mmlu: float | None
    recursive_mmlu: float | None
    delta_mmlu: float | None
    mmlu_loss: float | None

    human_bios_accuracy: float | None
    recursive_bios_accuracy: float | None
    delta_bios_accuracy: float | None
    bios_accuracy_loss: float | None

    human_eo: float | None
    recursive_eo: float | None
    delta_eo: float | None
    eo_increase: float | None

    human_stable_eo: float | None
    recursive_stable_eo: float | None
    delta_stable_eo: float | None

    eo_bootstrap_estimate: float | None = None
    eo_bootstrap_ci_low: float | None = None
    eo_bootstrap_ci_high: float | None = None
    eo_significant_increase: bool | None = None

    accuracy_bootstrap_estimate: float | None = None
    accuracy_bootstrap_ci_low: float | None = None
    accuracy_bootstrap_ci_high: float | None = None
    accuracy_significant_decrease: bool | None = None

    stable_eo_bootstrap_estimate: float | None = None
    stable_eo_bootstrap_ci_low: float | None = None
    stable_eo_bootstrap_ci_high: float | None = None
    stable_eo_significant_increase: bool | None = None

    bootstrap_path: str | None = None
    mmlu_paired_delta: float | None = None
    mmlu_paired_ci_low: float | None = None
    mmlu_paired_ci_high: float | None = None
    mmlu_permutation_pvalue: float | None = None
    mmlu_mcnemar_pvalue: float | None = None
    mmlu_significant_degradation: bool | None = None

    mmlu_stats_path: str | None = None
    significance_state: str | None = None
    trajectory_stage: str | None = None
    fairness_onset_iteration: int | None = None
    performance_onset_iteration: int | None = None

def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected JSON object in {path}, got {type(value)}."
        )

    return value


def write_json(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            value,
            f,
            indent=2,
            ensure_ascii=False,
        )


def write_jsonl(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_csv(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def _parse_seed_component(
    component: str,
) -> int | None:
    for prefix in ("seed_", "seed"):
        if component.startswith(prefix):
            suffix = component[len(prefix):]

            if suffix.lstrip("-").isdigit():
                return int(suffix)

    return None


def _parse_iteration_component(
    component: str,
) -> int | None:
    for prefix in (
        "iter_",
        "iteration_",
        "iter",
    ):
        if component.startswith(prefix):
            suffix = component[len(prefix):]

            if suffix.isdigit():
                return int(suffix)

    return None


def infer_path_metadata(
    path: str | Path,
) -> dict[str, Any]:
    path = Path(path)

    seed = None
    iteration = None
    branch = None

    for component in path.parts:
        maybe_seed = _parse_seed_component(
            component
        )

        if maybe_seed is not None:
            seed = maybe_seed

        maybe_iteration = (
            _parse_iteration_component(
                component
            )
        )

        if maybe_iteration is not None:
            iteration = maybe_iteration

        if component in {
            "human",
            "recursive",
            "iterative",
        }:
            branch = component

    return {
        "seed": seed,
        "iteration": iteration,
        "branch": branch,
    }


def find_nearest_file(
    start: str | Path,
    filename: str,
) -> Path | None:
    current = Path(start)

    if current.is_file():
        current = current.parent

    for directory in (
        current,
        *current.parents,
    ):
        candidate = directory / filename

        if candidate.exists():
            return candidate

    return None


def read_experiment_metadata(
    path: str | Path,
) -> tuple[str, str | None]:
    manifest_path = find_nearest_file(
        path,
        "experiment_manifest.json",
    )

    if manifest_path is None:
        path = Path(path)
        parts = list(path.parts)

        for i, part in enumerate(parts):
            if _parse_seed_component(part) is not None and i > 0:
                return parts[i - 1], None

        return (
            path.parent.name
            if path.parent.name
            else "unknown",
            None,
        )

    manifest = load_json(
        manifest_path
    )

    experiment = str(
        manifest.get(
            "experiment_name",
            manifest_path.parent.name,
        )
    )

    model_key = (
        manifest.get("training_tokenizer")
        or manifest.get("human_checkpoint")
    )

    return (
        experiment,
        str(model_key)
        if model_key is not None
        else None,
    )

def _nested(
    value: dict[str, Any],
    *keys: str,
) -> Any:
    current: Any = value

    for key in keys:
        if not isinstance(current, dict):
            return None

        current = current.get(key)

        if current is None:
            return None

    return current


def _float_or_none(
    value: Any,
) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        result = float(value)

        if math.isnan(result):
            return None

        return result

    return None


def _bool_or_none(
    value: Any,
) -> bool | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    return None


def _difference(
    comparison: float | None,
    human: float | None,
) -> float | None:
    if comparison is None or human is None:
        return None

    return float(
        comparison - human
    )

def discover_evaluation_summaries(
    run_root: str | Path,
) -> list[Path]:
    return sorted(
        Path(run_root).rglob(
            "evaluation_summary.json"
        )
    )


def checkpoint_row_from_summary(
    path: str | Path,
) -> CheckpointRow:
    path = Path(path)
    summary = load_json(path)

    inferred = infer_path_metadata(
        path
    )

    experiment, model_key = (
        read_experiment_metadata(
            path
        )
    )

    checkpoint = str(
        summary.get(
            "model_name_or_path",
            "",
        )
    )

    bios = summary.get(
        "bias_in_bios"
    )

    primary_bios = None
    pairwise_bios = None

    if isinstance(bios, dict):
        primary_score = bios.get(
            "primary_score",
            "mean_logprob",
        )

        if primary_score == "total_logprob":
            primary_bios = bios.get(
                "metrics_total_logprob"
            )
            pairwise_bios = bios.get(
                "pairwise_metrics_total_logprob"
            )
        else:
            primary_bios = bios.get(
                "metrics_mean_logprob"
            )
            pairwise_bios = bios.get(
                "pairwise_metrics_mean_logprob"
            )

    mmlu_accuracy = _float_or_none(
        _nested(
            summary,
            "mmlu",
            "accuracy",
        )
    )

    return CheckpointRow(
        experiment=experiment,
        model_key=model_key,
        seed=inferred["seed"],
        branch=inferred["branch"],
        iteration=inferred["iteration"],
        checkpoint=checkpoint,
        evaluation_summary_path=str(
            path.resolve()
        ),
        heldout_human_ppl=_float_or_none(
            _nested(
                summary,
                "heldout_human_perplexity",
                "perplexity",
            )
        ),
        mmlu_accuracy=mmlu_accuracy,
        mmlu_accuracy_percent=(
            100.0 * mmlu_accuracy
            if mmlu_accuracy is not None
            else None
        ),
        bios_accuracy=_float_or_none(
            (primary_bios or {}).get(
                "accuracy"
            )
        ),
        bios_rms_eo_gap=_float_or_none(
            (primary_bios or {}).get(
                "rms_eo_gap"
            )
        ),
        bios_rms_eo_gap_stable=_float_or_none(
            (primary_bios or {}).get(
                "rms_eo_gap_stable_cells"
            )
        ),
        pairwise_bios_accuracy=_float_or_none(
            (pairwise_bios or {}).get(
                "accuracy"
            )
        ),
        pairwise_bios_rms_eo_gap=_float_or_none(
            (pairwise_bios or {}).get(
                "rms_eo_gap"
            )
        ),
    )


def collect_checkpoint_rows(
    run_root: str | Path,
) -> list[CheckpointRow]:
    return [
        checkpoint_row_from_summary(path)
        for path in discover_evaluation_summaries(
            run_root
        )
    ]


def _stats_key_from_comparison_path(
    *,
    stats_path: Path,
    payload: dict[str, Any],
    comparison_key: str,
) -> tuple[str, int, int] | None:
    comparison_path = payload.get(
        comparison_key
    )

    source = (
        comparison_path
        if comparison_path
        else stats_path
    )

    inferred = infer_path_metadata(
        source
    )

    if (
        inferred["seed"] is None
        or inferred["iteration"] is None
    ):
        inferred = infer_path_metadata(
            stats_path
        )

    if (
        inferred["seed"] is None
        or inferred["iteration"] is None
    ):
        return None

    experiment, _ = (
        read_experiment_metadata(
            stats_path
        )
    )

    return (
        experiment,
        int(inferred["seed"]),
        int(inferred["iteration"]),
    )


def collect_stats_index(
    *,
    run_root: str | Path,
    filename: str,
    comparison_key: str,
) -> dict[
    tuple[str, int, int],
    tuple[Path, dict[str, Any]],
]:
    index = {}

    for path in sorted(
        Path(run_root).rglob(
            filename
        )
    ):
        payload = load_json(
            path
        )

        key = _stats_key_from_comparison_path(
            stats_path=path,
            payload=payload,
            comparison_key=(
                comparison_key
            ),
        )

        if key is None:
            continue

        if key in index:
            raise ValueError(
                f"Multiple {filename} files found for {key}:\n"
                f"  {index[key][0]}\n"
                f"  {path}"
            )

        index[key] = (
            path,
            payload,
        )

    return index

def pair_checkpoint_rows(
    checkpoint_rows: Sequence[
        CheckpointRow
    ],
    *,
    bootstrap_index: dict[
        tuple[str, int, int],
        tuple[Path, dict[str, Any]],
    ] | None = None,
    mmlu_stats_index: dict[
        tuple[str, int, int],
        tuple[Path, dict[str, Any]],
    ] | None = None,
) -> list[MatchedRow]:

    human = {}
    recursive = {}

    for row in checkpoint_rows:
        if (
            row.seed is None
            or row.iteration is None
        ):
            continue

        key = (
            row.experiment,
            row.seed,
            row.iteration,
        )

        if row.branch == "human":
            if key in human:
                raise ValueError(
                    f"Duplicate human evaluation for {key}."
                )
            human[key] = row

        elif row.branch == "recursive":
            if key in recursive:
                raise ValueError(
                    f"Duplicate recursive evaluation for {key}."
                )
            recursive[key] = row

    shared = sorted(
        set(human)
        & set(recursive)
    )

    matched: list[
        MatchedRow
    ] = []

    for key in shared:
        (
            experiment,
            seed,
            iteration,
        ) = key

        h = human[key]
        r = recursive[key]

        bootstrap_path = None
        bootstrap = None

        if (
            bootstrap_index is not None
            and key in bootstrap_index
        ):
            bp, bootstrap = (
                bootstrap_index[key]
            )
            bootstrap_path = str(
                bp.resolve()
            )

        mmlu_stats_path = None
        mmlu_stats = None

        if (
            mmlu_stats_index is not None
            and key in mmlu_stats_index
        ):
            mp, mmlu_stats = (
                mmlu_stats_index[key]
            )
            mmlu_stats_path = str(
                mp.resolve()
            )

        eo_ci = (
            bootstrap.get(
                "rms_eo_gap_difference"
            )
            if isinstance(
                bootstrap,
                dict,
            )
            else None
        )

        acc_ci = (
            bootstrap.get(
                "accuracy_difference"
            )
            if isinstance(
                bootstrap,
                dict,
            )
            else None
        )

        stable_ci = (
            bootstrap.get(
                "rms_eo_gap_stable_difference"
            )
            if isinstance(
                bootstrap,
                dict,
            )
            else None
        )

        mmlu_bootstrap = (
            mmlu_stats.get(
                "bootstrap"
            )
            if isinstance(
                mmlu_stats,
                dict,
            )
            else None
        )

        delta_ppl = _difference(
            r.heldout_human_ppl,
            h.heldout_human_ppl,
        )

        delta_mmlu = _difference(
            r.mmlu_accuracy,
            h.mmlu_accuracy,
        )

        delta_acc = _difference(
            r.bios_accuracy,
            h.bios_accuracy,
        )

        delta_eo = _difference(
            r.bios_rms_eo_gap,
            h.bios_rms_eo_gap,
        )

        delta_stable_eo = _difference(
            r.bios_rms_eo_gap_stable,
            h.bios_rms_eo_gap_stable,
        )

        matched.append(
            MatchedRow(
                experiment=experiment,
                model_key=(
                    r.model_key
                    or h.model_key
                ),
                seed=int(seed),
                iteration=int(
                    iteration
                ),
                human_checkpoint=(
                    h.checkpoint
                ),
                recursive_checkpoint=(
                    r.checkpoint
                ),
                human_ppl=(
                    h.heldout_human_ppl
                ),
                recursive_ppl=(
                    r.heldout_human_ppl
                ),
                delta_ppl=delta_ppl,
                ppl_increase=delta_ppl,
                human_mmlu=(
                    h.mmlu_accuracy
                ),
                recursive_mmlu=(
                    r.mmlu_accuracy
                ),
                delta_mmlu=delta_mmlu,
                mmlu_loss=(
                    -delta_mmlu
                    if delta_mmlu is not None
                    else None
                ),
                human_bios_accuracy=(
                    h.bios_accuracy
                ),
                recursive_bios_accuracy=(
                    r.bios_accuracy
                ),
                delta_bios_accuracy=(
                    delta_acc
                ),
                bios_accuracy_loss=(
                    -delta_acc
                    if delta_acc is not None
                    else None
                ),
                human_eo=(
                    h.bios_rms_eo_gap
                ),
                recursive_eo=(
                    r.bios_rms_eo_gap
                ),
                delta_eo=delta_eo,
                eo_increase=delta_eo,
                human_stable_eo=(
                    h.bios_rms_eo_gap_stable
                ),
                recursive_stable_eo=(
                    r.bios_rms_eo_gap_stable
                ),
                delta_stable_eo=(
                    delta_stable_eo
                ),

                eo_bootstrap_estimate=_float_or_none(
                    (eo_ci or {}).get(
                        "estimate"
                    )
                ),
                eo_bootstrap_ci_low=_float_or_none(
                    (eo_ci or {}).get(
                        "ci_low"
                    )
                ),
                eo_bootstrap_ci_high=_float_or_none(
                    (eo_ci or {}).get(
                        "ci_high"
                    )
                ),
                eo_significant_increase=_bool_or_none(
                    (eo_ci or {}).get(
                        "significant_increase"
                    )
                ),

                accuracy_bootstrap_estimate=_float_or_none(
                    (acc_ci or {}).get(
                        "estimate"
                    )
                ),
                accuracy_bootstrap_ci_low=_float_or_none(
                    (acc_ci or {}).get(
                        "ci_low"
                    )
                ),
                accuracy_bootstrap_ci_high=_float_or_none(
                    (acc_ci or {}).get(
                        "ci_high"
                    )
                ),
                accuracy_significant_decrease=_bool_or_none(
                    (acc_ci or {}).get(
                        "significant_decrease"
                    )
                ),

                stable_eo_bootstrap_estimate=_float_or_none(
                    (stable_ci or {}).get(
                        "estimate"
                    )
                ),
                stable_eo_bootstrap_ci_low=_float_or_none(
                    (stable_ci or {}).get(
                        "ci_low"
                    )
                ),
                stable_eo_bootstrap_ci_high=_float_or_none(
                    (stable_ci or {}).get(
                        "ci_high"
                    )
                ),
                stable_eo_significant_increase=_bool_or_none(
                    (stable_ci or {}).get(
                        "significant_increase"
                    )
                ),

                bootstrap_path=(
                    bootstrap_path
                ),

                mmlu_paired_delta=_float_or_none(
                    (
                        mmlu_stats
                        or {}
                    ).get(
                        "delta_accuracy"
                    )
                ),
                mmlu_paired_ci_low=_float_or_none(
                    (
                        mmlu_bootstrap
                        or {}
                    ).get(
                        "ci_low"
                    )
                ),
                mmlu_paired_ci_high=_float_or_none(
                    (
                        mmlu_bootstrap
                        or {}
                    ).get(
                        "ci_high"
                    )
                ),
                mmlu_permutation_pvalue=_float_or_none(
                    (
                        mmlu_stats
                        or {}
                    ).get(
                        "permutation_pvalue_two_sided"
                    )
                ),
                mmlu_mcnemar_pvalue=_float_or_none(
                    (
                        mmlu_stats
                        or {}
                    ).get(
                        "mcnemar_pvalue_two_sided"
                    )
                ),
                mmlu_significant_degradation=_bool_or_none(
                    (
                        mmlu_stats
                        or {}
                    ).get(
                        "significant_performance_degradation"
                    )
                ),
                mmlu_stats_path=(
                    mmlu_stats_path
                ),
            )
        )

    assign_stage_labels(
        matched
    )

    return matched


def significance_state(
    *,
    fairness: bool | None,
    performance: bool | None,
) -> str:

    if fairness is True and performance is True:
        return (
            "fairness + performance significant"
        )

    if fairness is True:
        return "fairness significant"

    if performance is True:
        return "performance significant"

    if fairness is None or performance is None:
        return "incomplete statistics"

    return "neither significant"


def assign_stage_labels(
    rows: Sequence[MatchedRow],
) -> None:

    grouped: dict[
        tuple[str, int],
        list[MatchedRow],
    ] = {}

    for row in rows:
        grouped.setdefault(
            (
                row.experiment,
                row.seed,
            ),
            [],
        ).append(
            row
        )

    for _, group in grouped.items():
        ordered = sorted(
            group,
            key=lambda x: (
                x.iteration
            ),
        )

        fairness_iterations = [
            row.iteration
            for row in ordered
            if row.eo_significant_increase
            is True
        ]

        performance_iterations = [
            row.iteration
            for row in ordered
            if row.mmlu_significant_degradation
            is True
        ]

        fairness_onset = (
            min(fairness_iterations)
            if fairness_iterations
            else None
        )

        performance_onset = (
            min(performance_iterations)
            if performance_iterations
            else None
        )

        for row in ordered:
            row.fairness_onset_iteration = (
                fairness_onset
            )
            row.performance_onset_iteration = (
                performance_onset
            )

            row.significance_state = (
                significance_state(
                    fairness=(
                        row.eo_significant_increase
                    ),
                    performance=(
                        row.mmlu_significant_degradation
                    ),
                )
            )

            if (
                row.eo_significant_increase
                is None
                or row.mmlu_significant_degradation
                is None
            ):
                row.trajectory_stage = (
                    "incomplete statistics"
                )
                continue

            t = row.iteration

            if (
                fairness_onset is None
                and performance_onset is None
            ):
                row.trajectory_stage = (
                    "pre-collapse"
                )

            elif (
                fairness_onset is not None
                and performance_onset is not None
                and fairness_onset
                == performance_onset
            ):
                if t < fairness_onset:
                    row.trajectory_stage = (
                        "pre-collapse"
                    )
                elif t == fairness_onset:
                    row.trajectory_stage = (
                        "fairness + performance collapse"
                    )
                else:
                    row.trajectory_stage = (
                        "post-collapse"
                    )

            elif (
                fairness_onset is not None
                and (
                    performance_onset is None
                    or fairness_onset
                    < performance_onset
                )
            ):
                if t < fairness_onset:
                    row.trajectory_stage = (
                        "pre-collapse"
                    )
                elif t == fairness_onset:
                    row.trajectory_stage = (
                        "fairness collapse"
                    )
                elif (
                    performance_onset is not None
                    and t
                    == performance_onset
                ):
                    row.trajectory_stage = (
                        "performance collapse"
                    )
                elif (
                    performance_onset is not None
                    and t
                    > performance_onset
                ):
                    row.trajectory_stage = (
                        "post-collapse"
                    )
                else:
                    row.trajectory_stage = (
                        "fairness collapse"
                    )

            else:
                if (
                    performance_onset is not None
                    and t
                    < performance_onset
                ):
                    row.trajectory_stage = (
                        "pre-collapse"
                    )
                elif (
                    performance_onset is not None
                    and t
                    == performance_onset
                ):
                    row.trajectory_stage = (
                        "performance collapse"
                    )
                else:
                    row.trajectory_stage = (
                        "performance-before-fairness"
                    )


def _mean(
    values: Sequence[float],
) -> float | None:
    if not values:
        return None

    return float(
        statistics.fmean(values)
    )


def _sample_std(
    values: Sequence[float],
) -> float | None:
    if len(values) < 2:
        return None

    return float(
        statistics.stdev(values)
    )


def _numeric_values(
    rows: Sequence[dict[str, Any]],
    key: str,
) -> list[float]:
    values = []

    for row in rows:
        value = _float_or_none(
            row.get(key)
        )

        if value is not None:
            values.append(value)

    return values


def summarize_checkpoint_rows(
    rows: Sequence[CheckpointRow],
) -> list[dict[str, Any]]:
    grouped = {}

    for row in rows:
        key = (
            row.experiment,
            row.branch,
            row.iteration,
        )

        grouped.setdefault(
            key,
            [],
        ).append(
            row
        )

    metrics = [
        "heldout_human_ppl",
        "mmlu_accuracy",
        "bios_accuracy",
        "bios_rms_eo_gap",
        "bios_rms_eo_gap_stable",
        "pairwise_bios_accuracy",
        "pairwise_bios_rms_eo_gap",
    ]

    output = []

    for (
        experiment,
        branch,
        iteration,
    ), group in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            str(item[0][1]),
            item[0][2]
            if item[0][2] is not None
            else -1,
        ),
    ):
        row = {
            "experiment": experiment,
            "model_key": next(
                (
                    item.model_key
                    for item in group
                    if item.model_key
                    is not None
                ),
                None,
            ),
            "branch": branch,
            "iteration": iteration,
            "n_seed_rows": len(group),
            "seeds": ",".join(
                str(seed)
                for seed in sorted(
                    {
                        item.seed
                        for item in group
                        if item.seed
                        is not None
                    }
                )
            ),
        }

        group_dicts = [
            asdict(item)
            for item in group
        ]

        for metric in metrics:
            values = _numeric_values(
                group_dicts,
                metric,
            )

            row[
                f"{metric}_mean"
            ] = _mean(values)

            row[
                f"{metric}_std"
            ] = _sample_std(values)

            row[
                f"{metric}_n"
            ] = len(values)

        output.append(row)

    return output


def summarize_matched_rows(
    rows: Sequence[MatchedRow],
) -> list[dict[str, Any]]:
    grouped = {}

    for row in rows:
        key = (
            row.experiment,
            row.iteration,
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)

    metrics = [
        "delta_ppl",
        "ppl_increase",
        "delta_mmlu",
        "mmlu_loss",
        "delta_bios_accuracy",
        "bios_accuracy_loss",
        "delta_eo",
        "eo_increase",
        "delta_stable_eo",
    ]

    output = []

    for (
        experiment,
        iteration,
    ), group in sorted(
        grouped.items()
    ):
        row = {
            "experiment": experiment,
            "model_key": next(
                (
                    item.model_key
                    for item in group
                    if item.model_key
                    is not None
                ),
                None,
            ),
            "iteration": iteration,
            "n_seeds": len(group),
            "seeds": ",".join(
                str(item.seed)
                for item in sorted(
                    group,
                    key=lambda x: (
                        x.seed
                    ),
                )
            ),
        }

        group_dicts = [
            asdict(item)
            for item in group
        ]

        for metric in metrics:
            values = _numeric_values(
                group_dicts,
                metric,
            )

            row[
                f"{metric}_mean"
            ] = _mean(values)

            row[
                f"{metric}_std"
            ] = _sample_std(values)

            row[
                f"{metric}_n"
            ] = len(values)

        fairness_flags = [
            item.eo_significant_increase
            for item in group
            if item.eo_significant_increase
            is not None
        ]

        performance_flags = [
            item.mmlu_significant_degradation
            for item in group
            if item.mmlu_significant_degradation
            is not None
        ]

        row[
            "fairness_significant_seeds"
        ] = sum(
            bool(x)
            for x in fairness_flags
        )

        row[
            "fairness_stats_seeds_available"
        ] = len(
            fairness_flags
        )

        row[
            "performance_significant_seeds"
        ] = sum(
            bool(x)
            for x in performance_flags
        )

        row[
            "performance_stats_seeds_available"
        ] = len(
            performance_flags
        )

        stage_counts = {}

        for item in group:
            stage = (
                item.trajectory_stage
                or "unknown"
            )

            stage_counts[
                stage
            ] = (
                stage_counts.get(
                    stage,
                    0,
                )
                + 1
            )

        row[
            "trajectory_stage_counts"
        ] = json.dumps(
            stage_counts,
            ensure_ascii=False,
            sort_keys=True,
        )

        output.append(row)

    return output


def onset_summary(
    rows: Sequence[MatchedRow],
) -> list[dict[str, Any]]:
    grouped = {}

    for row in rows:
        grouped.setdefault(
            (
                row.experiment,
                row.seed,
            ),
            [],
        ).append(row)

    output = []

    for (
        experiment,
        seed,
    ), group in sorted(
        grouped.items()
    ):
        first = sorted(
            group,
            key=lambda x: x.iteration,
        )[0]

        fairness_onset = (
            first.fairness_onset_iteration
        )

        performance_onset = (
            first.performance_onset_iteration
        )

        if (
            fairness_onset is not None
            and performance_onset is not None
        ):
            lead = (
                performance_onset
                - fairness_onset
            )
        else:
            lead = None

        output.append(
            {
                "experiment": experiment,
                "model_key": (
                    first.model_key
                ),
                "seed": seed,
                "fairness_onset_iteration": (
                    fairness_onset
                ),
                "performance_onset_iteration": (
                    performance_onset
                ),
                "fairness_lead_iterations": (
                    lead
                ),
                "fairness_before_performance": (
                    bool(
                        fairness_onset
                        is not None
                        and (
                            performance_onset
                            is None
                            or fairness_onset
                            < performance_onset
                        )
                    )
                ),
                "concurrent_onset": (
                    bool(
                        fairness_onset
                        is not None
                        and performance_onset
                        is not None
                        and fairness_onset
                        == performance_onset
                    )
                ),
            }
        )

    return output

def aggregate_results(
    *,
    run_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:

    run_root = Path(run_root)

    if not run_root.exists():
        raise FileNotFoundError(
            f"Run root does not exist: {run_root}"
        )

    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else run_root / "aggregate"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_rows = (
        collect_checkpoint_rows(
            run_root
        )
    )

    if not checkpoint_rows:
        raise FileNotFoundError(
            f"No evaluation_summary.json files found beneath {run_root}."
        )

    bootstrap_index = (
        collect_stats_index(
            run_root=run_root,
            filename=(
                "paired_bootstrap.json"
            ),
            comparison_key=(
                "comparison_predictions_path"
            ),
        )
    )

    mmlu_stats_index = (
        collect_stats_index(
            run_root=run_root,
            filename=(
                "mmlu_paired_stats.json"
            ),
            comparison_key=(
                "comparison_samples_path"
            ),
        )
    )

    matched_rows = (
        pair_checkpoint_rows(
            checkpoint_rows,
            bootstrap_index=(
                bootstrap_index
            ),
            mmlu_stats_index=(
                mmlu_stats_index
            ),
        )
    )

    checkpoint_dicts = [
        asdict(row)
        for row in checkpoint_rows
    ]

    matched_dicts = [
        asdict(row)
        for row in matched_rows
    ]

    checkpoint_summary = (
        summarize_checkpoint_rows(
            checkpoint_rows
        )
    )

    matched_summary = (
        summarize_matched_rows(
            matched_rows
        )
    )

    onsets = onset_summary(
        matched_rows
    )

    write_csv(
        checkpoint_dicts,
        output_dir
        / "checkpoint_results.csv",
    )

    write_jsonl(
        checkpoint_dicts,
        output_dir
        / "checkpoint_results.jsonl",
    )

    write_csv(
        matched_dicts,
        output_dir
        / "matched_results.csv",
    )

    write_jsonl(
        matched_dicts,
        output_dir
        / "matched_results.jsonl",
    )

    write_csv(
        checkpoint_summary,
        output_dir
        / "checkpoint_seed_summary.csv",
    )

    write_json(
        checkpoint_summary,
        output_dir
        / "checkpoint_seed_summary.json",
    )

    write_csv(
        matched_summary,
        output_dir
        / "matched_seed_summary.csv",
    )

    write_json(
        matched_summary,
        output_dir
        / "matched_seed_summary.json",
    )

    write_csv(
        onsets,
        output_dir
        / "collapse_onsets.csv",
    )

    write_json(
        onsets,
        output_dir
        / "collapse_onsets.json",
    )

    manifest = {
        "run_root": str(
            run_root.resolve()
        ),
        "output_dir": str(
            output_dir.resolve()
        ),
        "num_checkpoint_evaluations": (
            len(checkpoint_rows)
        ),
        "num_matched_human_recursive_rows": (
            len(matched_rows)
        ),
        "num_bias_bootstrap_results": (
            len(bootstrap_index)
        ),
        "num_mmlu_paired_stats_results": (
            len(mmlu_stats_index)
        ),
        "operational_definition": {
            "fairness_onset": (
                "first iteration where the 99% paired stratified "
                "bootstrap CI for EO(R_t)-EO(H_t) is entirely > 0"
            ),
            "performance_onset": (
                "first iteration where the 99% paired MMLU bootstrap "
                "CI for Acc(R_t)-Acc(H_t) is entirely < 0"
            ),
        },
        "notes": {
            "matched_delta_definition": (
                "raw delta = recursive - matched human-only"
            ),
            "degradation_oriented_fields": (
                "ppl_increase, mmlu_loss, bios_accuracy_loss, "
                "eo_increase are positive when recursive is worse"
            ),
            "seed_summary": (
                "mean/std are descriptive statistics across pipeline seeds"
            ),
        },
    }

    write_json(
        manifest,
        output_dir
        / "aggregation_manifest.json",
    )

    print()
    print("=" * 72)
    print("AGGREGATION COMPLETE")
    print("=" * 72)
    print(
        f"Checkpoint evaluations: "
        f"{len(checkpoint_rows)}"
    )
    print(
        f"Matched H_t/R_t rows:    "
        f"{len(matched_rows)}"
    )
    print(
        f"Bias bootstrap files:     "
        f"{len(bootstrap_index)}"
    )
    print(
        f"MMLU stats files:         "
        f"{len(mmlu_stats_index)}"
    )
    print(
        f"Output:                   "
        f"{output_dir}"
    )

    return {
        "checkpoint_rows": (
            checkpoint_dicts
        ),
        "matched_rows": (
            matched_dicts
        ),
        "checkpoint_seed_summary": (
            checkpoint_summary
        ),
        "matched_seed_summary": (
            matched_summary
        ),
        "collapse_onsets": (
            onsets
        ),
        "manifest": manifest,
    }
