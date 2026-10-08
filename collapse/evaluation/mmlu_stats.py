from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class MMLUStatsConfig:
    """
    Paired significance analysis for matched MMLU evaluations.
    """

    n_bootstrap: int = 10_000
    confidence_level: float = 0.99
    seed: int = 23456
    n_permutations: int = 50_000

    metric_name: str = "acc"

    require_task_match: bool = True

    require_hash_match: bool = True

    save_distribution: bool = False


@dataclass(frozen=True)
class MMLUAccuracyCI:
    estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float

    prob_le_zero: float
    prob_ge_zero: float

    significant_increase: bool
    significant_decrease: bool


@dataclass(frozen=True)
class MMLUPairedStatsResult:
    human_samples_path: str
    comparison_samples_path: str

    metric_name: str

    num_items: int
    num_tasks: int

    human_accuracy: float
    comparison_accuracy: float

    delta_accuracy: float
    accuracy_loss: float

    bootstrap: MMLUAccuracyCI

    permutation_pvalue_two_sided: float

    mcnemar_pvalue_two_sided: float

    human_correct_comparison_wrong: int
    human_wrong_comparison_correct: int

    significant_performance_degradation: bool

    task_accuracies_human: dict[str, float]
    task_accuracies_comparison: dict[str, float]
    task_accuracy_differences: dict[str, float]

def _looks_like_sample_filename(path: Path) -> bool:
    name = path.name.lower()

    return (
        "sample" in name
        and path.suffix.lower() in {
            ".jsonl",
            ".json",
        }
    )


def discover_lm_eval_sample_files(
    root: str | Path,
) -> list[Path]:
    root = Path(root)

    if root.is_file():
        if _looks_like_sample_filename(
            root
        ):
            return [root]

        raise ValueError(
            f"File does not look like an lm-eval sample log: {root}"
        )

    if not root.exists():
        raise FileNotFoundError(
            f"lm-eval output path does not exist: {root}"
        )

    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and _looks_like_sample_filename(
            path
        )
    ]

    return sorted(
        files
    )

def _read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    rows: list[
        dict[str, Any]
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(
                    line
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from e

            if not isinstance(
                item,
                dict,
            ):
                raise TypeError(
                    f"Expected JSON object at {path}:{line_number}."
                )

            rows.append(
                item
            )

    return rows


def _read_json(
    path: Path,
) -> list[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        value = json.load(
            f
        )

    if isinstance(
        value,
        list,
    ):
        rows = value

    elif isinstance(
        value,
        dict,
    ):
        rows = None

        for key in (
            "samples",
            "logs",
            "items",
        ):
            candidate = value.get(
                key
            )

            if isinstance(
                candidate,
                list,
            ):
                rows = candidate
                break

        if rows is None:
            raise ValueError(
                f"Could not find a sample list in {path}."
            )
    else:
        raise TypeError(
            f"Unsupported JSON root type in {path}: {type(value)}."
        )

    result = []

    for index, row in enumerate(
        rows
    ):
        if not isinstance(
            row,
            dict,
        ):
            raise TypeError(
                f"Expected object at {path} sample index {index}."
            )

        result.append(
            row
        )

    return result


def load_lm_eval_samples(
    root: str | Path,
) -> list[dict[str, Any]]:

    files = (
        discover_lm_eval_sample_files(
            root
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No lm-eval sample JSON/JSONL files found under {root}."
        )

    rows: list[
        dict[str, Any]
    ] = []

    for path in files:
        if path.suffix.lower() == ".jsonl":
            loaded = _read_jsonl(
                path
            )
        else:
            loaded = _read_json(
                path
            )

        for row in loaded:
            copied = dict(
                row
            )

            copied[
                "_source_file"
            ] = str(
                path.resolve()
            )

            rows.append(
                copied
            )

    if not rows:
        raise ValueError(
            f"No sample rows loaded from {root}."
        )

    return rows

def _task_name(
    row: dict[str, Any],
) -> str:
    for key in (
        "task_name",
        "task",
        "task_alias",
    ):
        value = row.get(
            key
        )

        if value is not None:
            return str(
                value
            )

    source = row.get(
        "_source_file"
    )

    if source:
        return Path(
            str(source)
        ).stem

    return "unknown"


def _doc_id(
    row: dict[str, Any],
) -> str:
    for key in (
        "doc_id",
        "id",
        "sample_id",
        "example_id",
    ):
        value = row.get(
            key
        )

        if value is not None:
            return str(
                value
            )

    raise ValueError(
        "lm-eval sample row has no recognizable document ID field. "
        f"Available keys: {sorted(row.keys())}"
    )


def _hash_value(
    row: dict[str, Any],
) -> str | None:
    values = []

    for key in (
        "doc_hash",
        "target_hash",
        "prompt_hash",
    ):
        value = row.get(
            key
        )

        if value is not None:
            values.append(
                f"{key}={value}"
            )

    if not values:
        return None

    return "|".join(
        values
    )


def _extract_metric(
    row: dict[str, Any],
    metric_name: str,
) -> float:

    metrics = row.get(
        "metrics"
    )

    if isinstance(
        metrics,
        dict,
    ):
        if metric_name in metrics:
            value = metrics[
                metric_name
            ]

            if isinstance(
                value,
                (
                    bool,
                    int,
                    float,
                ),
            ):
                return float(
                    value
                )

        for key, value in metrics.items():
            if (
                isinstance(
                    key,
                    str,
                )
                and key.startswith(
                    metric_name + ","
                )
                and isinstance(
                    value,
                    (
                        bool,
                        int,
                        float,
                    ),
                )
            ):
                return float(
                    value
                )

    direct = row.get(
        metric_name
    )

    if isinstance(
        direct,
        (
            bool,
            int,
            float,
        ),
    ):
        return float(
            direct
        )

    if (
        row.get(
            "metric"
        )
        == metric_name
        and isinstance(
            row.get(
                "value"
            ),
            (
                bool,
                int,
                float,
            ),
        )
    ):
        return float(
            row[
                "value"
            ]
        )

    raise ValueError(
        f"Could not extract metric {metric_name!r} from lm-eval row. "
        f"Available keys: {sorted(row.keys())}"
    )


@dataclass(frozen=True)
class NormalizedSample:
    key: str
    task: str
    doc_id: str
    hash_value: str | None
    correct: float
    source_file: str


def normalize_samples(
    rows: Sequence[
        dict[str, Any]
    ],
    *,
    metric_name: str,
) -> list[
    NormalizedSample
]:
    normalized: list[
        NormalizedSample
    ] = []

    seen = set()

    for row in rows:
        task = _task_name(
            row
        )

        doc_id = _doc_id(
            row
        )

        key = (
            f"{task}::{doc_id}"
        )

        if key in seen:
            raise ValueError(
                f"Duplicate lm-eval sample key {key!r}. "
                "This usually means multiple sample logs from the same "
                "evaluation were mixed in one directory."
            )

        seen.add(
            key
        )

        correct = _extract_metric(
            row,
            metric_name,
        )
        if not (
            math.isclose(
                correct,
                0.0,
            )
            or math.isclose(
                correct,
                1.0,
            )
        ):
            raise ValueError(
                f"Expected binary {metric_name!r} for MMLU item {key}, "
                f"got {correct!r}."
            )

        normalized.append(
            NormalizedSample(
                key=key,
                task=task,
                doc_id=doc_id,
                hash_value=(
                    _hash_value(
                        row
                    )
                ),
                correct=float(
                    round(
                        correct
                    )
                ),
                source_file=str(
                    row.get(
                        "_source_file",
                        "",
                    )
                ),
            )
        )

    return normalized

def align_samples(
    *,
    human: Sequence[
        NormalizedSample
    ],
    comparison: Sequence[
        NormalizedSample
    ],
    config: MMLUStatsConfig,
) -> list[
    tuple[
        NormalizedSample,
        NormalizedSample,
    ]
]:

    h_index = {
        item.key: item
        for item in human
    }

    c_index = {
        item.key: item
        for item in comparison
    }

    h_keys = set(
        h_index
    )
    c_keys = set(
        c_index
    )

    if h_keys != c_keys:
        only_h = sorted(
            h_keys
            - c_keys
        )[:20]

        only_c = sorted(
            c_keys
            - h_keys
        )[:20]

        raise ValueError(
            "MMLU sample logs do not contain exactly the same items.\n"
            f"Only human (first 20): {only_h}\n"
            f"Only comparison (first 20): {only_c}"
        )

    aligned = []

    for key in sorted(
        h_keys
    ):
        h = h_index[
            key
        ]

        c = c_index[
            key
        ]

        if (
            config.require_task_match
            and h.task != c.task
        ):
            raise ValueError(
                f"Task mismatch for {key!r}: "
                f"human={h.task!r}, comparison={c.task!r}"
            )

        if (
            config.require_hash_match
            and h.hash_value is not None
            and c.hash_value is not None
            and h.hash_value != c.hash_value
        ):
            raise ValueError(
                f"Document/prompt hash mismatch for {key!r}.\n"
                f"human:      {h.hash_value}\n"
                f"comparison: {c.hash_value}"
            )

        aligned.append(
            (
                h,
                c,
            )
        )

    return aligned

def _validate_config(
    config: MMLUStatsConfig,
) -> None:
    if config.n_bootstrap < 100:
        raise ValueError(
            "n_bootstrap should be at least 100."
        )

    if not (
        0.0
        < config.confidence_level
        < 1.0
    ):
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    if config.n_permutations < 100:
        raise ValueError(
            "n_permutations should be at least 100."
        )


def _percentile_ci(
    values: np.ndarray,
    confidence_level: float,
) -> tuple[
    float,
    float,
]:
    alpha = (
        1.0
        - confidence_level
    )

    return (
        float(
            np.quantile(
                values,
                alpha / 2.0,
            )
        ),
        float(
            np.quantile(
                values,
                1.0 - alpha / 2.0,
            )
        ),
    )


def paired_bootstrap_accuracy_difference(
    *,
    human: np.ndarray,
    comparison: np.ndarray,
    n_bootstrap: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[
    MMLUAccuracyCI,
    np.ndarray,
]:

    if (
        human.ndim != 1
        or comparison.ndim != 1
        or len(human)
        != len(comparison)
    ):
        raise ValueError(
            "human/comparison correctness arrays must be aligned 1D arrays."
        )

    n = len(
        human
    )

    if n == 0:
        raise ValueError(
            "No MMLU items available."
        )

    distribution = np.empty(
        n_bootstrap,
        dtype=np.float64,
    )

    chunk_size = 250
    offset = 0

    while offset < n_bootstrap:
        chunk_n = min(
            chunk_size,
            n_bootstrap
            - offset,
        )

        indices = rng.integers(
            0,
            n,
            size=(
                chunk_n,
                n,
            ),
        )

        h_acc = (
            human[
                indices
            ].mean(
                axis=1
            )
        )

        c_acc = (
            comparison[
                indices
            ].mean(
                axis=1
            )
        )

        distribution[
            offset:
            offset + chunk_n
        ] = (
            c_acc
            - h_acc
        )

        offset += chunk_n

    estimate = float(
        comparison.mean()
        - human.mean()
    )

    low, high = (
        _percentile_ci(
            distribution,
            confidence_level,
        )
    )

    result = (
        MMLUAccuracyCI(
            estimate=estimate,
            ci_low=low,
            ci_high=high,
            confidence_level=(
                confidence_level
            ),
            prob_le_zero=float(
                np.mean(
                    distribution
                    <= 0.0
                )
            ),
            prob_ge_zero=float(
                np.mean(
                    distribution
                    >= 0.0
                )
            ),
            significant_increase=bool(
                low > 0.0
            ),
            significant_decrease=bool(
                high < 0.0
            ),
        )
    )

    return (
        result,
        distribution,
    )


def paired_sign_flip_pvalue(
    *,
    differences: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:

    nonzero = differences[
        differences != 0.0
    ]

    if len(
        nonzero
    ) == 0:
        return 1.0

    observed = abs(
        float(
            nonzero.mean()
        )
    )

    exceed = 0
    done = 0
    chunk_size = 1000

    while done < n_permutations:
        chunk_n = min(
            chunk_size,
            n_permutations
            - done,
        )

        signs = rng.choice(
            np.array(
                [-1.0, 1.0],
                dtype=np.float64,
            ),
            size=(
                chunk_n,
                len(nonzero),
            ),
            replace=True,
        )

        permuted = (
            signs
            * nonzero[
                None,
                :
            ]
        ).mean(
            axis=1
        )

        exceed += int(
            np.sum(
                np.abs(
                    permuted
                )
                >= observed
            )
        )

        done += chunk_n
    return float(
        (
            exceed
            + 1
        )
        /
        (
            n_permutations
            + 1
        )
    )


def _binomial_tail_probability(
    k: int,
    n: int,
) -> float:

    if n == 0:
        return 1.0

    total = 0

    for i in range(
        0,
        k + 1,
    ):
        total += math.comb(
            n,
            i,
        )

    return float(
        total
        / (2 ** n)
    )


def exact_mcnemar_pvalue(
    *,
    human_correct_comparison_wrong: int,
    human_wrong_comparison_correct: int,
) -> float:
    """
    Two-sided McNemar/binomial test.
    """

    b = int(
        human_correct_comparison_wrong
    )

    c = int(
        human_wrong_comparison_correct
    )

    n = (
        b + c
    )

    if n == 0:
        return 1.0

    smaller = min(
        b,
        c,
    )

    p = (
        2.0
        * _binomial_tail_probability(
            smaller,
            n,
        )
    )

    return float(
        min(
            1.0,
            p,
        )
    )

def task_accuracy_breakdown(
    aligned: Sequence[
        tuple[
            NormalizedSample,
            NormalizedSample,
        ]
    ],
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:

    human: dict[
        str,
        list[float],
    ] = {}

    comparison: dict[
        str,
        list[float],
    ] = {}

    for h, c in aligned:
        human.setdefault(
            h.task,
            [],
        ).append(
            h.correct
        )

        comparison.setdefault(
            c.task,
            [],
        ).append(
            c.correct
        )

    h_acc = {
        task: float(
            np.mean(
                values
            )
        )
        for task, values
        in sorted(
            human.items()
        )
    }

    c_acc = {
        task: float(
            np.mean(
                values
            )
        )
        for task, values
        in sorted(
            comparison.items()
        )
    }

    delta = {
        task: float(
            c_acc[
                task
            ]
            - h_acc[
                task
            ]
        )
        for task in sorted(
            set(
                h_acc
            )
            & set(
                c_acc
            )
        )
    }

    return (
        h_acc,
        c_acc,
        delta,
    )

def paired_mmlu_stats(
    *,
    human_samples_path: str | Path,
    comparison_samples_path: str | Path,
    config: MMLUStatsConfig | None = None,
) -> tuple[
    MMLUPairedStatsResult,
    np.ndarray,
]:

    if config is None:
        config = (
            MMLUStatsConfig()
        )

    _validate_config(
        config
    )

    human_rows = (
        load_lm_eval_samples(
            human_samples_path
        )
    )

    comparison_rows = (
        load_lm_eval_samples(
            comparison_samples_path
        )
    )

    human_samples = (
        normalize_samples(
            human_rows,
            metric_name=(
                config.metric_name
            ),
        )
    )

    comparison_samples = (
        normalize_samples(
            comparison_rows,
            metric_name=(
                config.metric_name
            ),
        )
    )

    aligned = (
        align_samples(
            human=human_samples,
            comparison=(
                comparison_samples
            ),
            config=config,
        )
    )

    human = np.asarray(
        [
            h.correct
            for h, _
            in aligned
        ],
        dtype=np.float64,
    )

    comparison = np.asarray(
        [
            c.correct
            for _, c
            in aligned
        ],
        dtype=np.float64,
    )

    rng = (
        np.random.default_rng(
            config.seed
        )
    )

    (
        bootstrap,
        bootstrap_distribution,
    ) = (
        paired_bootstrap_accuracy_difference(
            human=human,
            comparison=comparison,
            n_bootstrap=(
                config.n_bootstrap
            ),
            confidence_level=(
                config.confidence_level
            ),
            rng=rng,
        )
    )

    differences = (
        comparison
        - human
    )

    permutation_p = (
        paired_sign_flip_pvalue(
            differences=differences,
            n_permutations=(
                config.n_permutations
            ),
            rng=rng,
        )
    )

    h_correct_c_wrong = int(
        np.sum(
            (human == 1.0)
            & (comparison == 0.0)
        )
    )

    h_wrong_c_correct = int(
        np.sum(
            (human == 0.0)
            & (comparison == 1.0)
        )
    )

    mcnemar_p = (
        exact_mcnemar_pvalue(
            human_correct_comparison_wrong=(
                h_correct_c_wrong
            ),
            human_wrong_comparison_correct=(
                h_wrong_c_correct
            ),
        )
    )

    (
        task_human,
        task_comparison,
        task_delta,
    ) = task_accuracy_breakdown(
        aligned
    )

    human_accuracy = float(
        human.mean()
    )

    comparison_accuracy = float(
        comparison.mean()
    )

    delta_accuracy = float(
        comparison_accuracy
        - human_accuracy
    )

    result = (
        MMLUPairedStatsResult(
            human_samples_path=str(
                Path(
                    human_samples_path
                ).resolve()
            ),
            comparison_samples_path=str(
                Path(
                    comparison_samples_path
                ).resolve()
            ),
            metric_name=(
                config.metric_name
            ),
            num_items=len(
                aligned
            ),
            num_tasks=len(
                task_human
            ),
            human_accuracy=(
                human_accuracy
            ),
            comparison_accuracy=(
                comparison_accuracy
            ),
            delta_accuracy=(
                delta_accuracy
            ),
            accuracy_loss=float(
                -delta_accuracy
            ),
            bootstrap=bootstrap,
            permutation_pvalue_two_sided=(
                permutation_p
            ),
            mcnemar_pvalue_two_sided=(
                mcnemar_p
            ),
            human_correct_comparison_wrong=(
                h_correct_c_wrong
            ),
            human_wrong_comparison_correct=(
                h_wrong_c_correct
            ),
            significant_performance_degradation=bool(
                bootstrap.ci_high
                < 0.0
            ),
            task_accuracies_human=(
                task_human
            ),
            task_accuracies_comparison=(
                task_comparison
            ),
            task_accuracy_differences=(
                task_delta
            ),
        )
    )

    return (
        result,
        bootstrap_distribution,
    )


def save_mmlu_stats(
    *,
    result: MMLUPairedStatsResult,
    output_dir: str | Path,
    bootstrap_distribution: np.ndarray | None = None,
    save_distribution: bool = False,
) -> None:

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        output_dir
        / "mmlu_paired_stats.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            asdict(
                result
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )

    if (
        save_distribution
        and bootstrap_distribution
        is not None
    ):
        np.save(
            output_dir
            / "mmlu_bootstrap_delta.npy",
            bootstrap_distribution,
        )