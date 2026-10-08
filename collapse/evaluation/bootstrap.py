from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapConfig:
    """
    Paired stratified bootstrap for Bias-in-Bios checkpoint comparisons.
    Intended comparison:
        recursive R_t  vs. matched human-only H_t
    Stratification:
        gold profession x gender
    Pairing:
        the same sampled example IDs are used for H_t and R_t inside
        every bootstrap replicate.
    fairness collapse at iteration t if the 99% CI for
        EO(R_t) - EO(H_t)
    lies entirely above zero.
    """

    n_bootstrap: int = 10_000
    confidence_level: float = 0.99
    seed: int = 12345

    correctness_field: str = "correct_mean"

    stable_group_min_count: int = 20

    bootstrap_chunk_size: int = 250
    save_distributions: bool = False


@dataclass(frozen=True)
class MetricCI:
    estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float
    prob_le_zero: float

    significant_increase: bool
    significant_decrease: bool


@dataclass(frozen=True)
class PairedBootstrapResult:
    human_predictions_path: str
    comparison_predictions_path: str

    correctness_field: str

    num_examples: int
    num_professions: int
    num_strata: int

    n_bootstrap: int
    confidence_level: float
    seed: int

    human_accuracy: float
    comparison_accuracy: float
    accuracy_difference: MetricCI

    human_rms_eo_gap: float
    comparison_rms_eo_gap: float
    rms_eo_gap_difference: MetricCI

    human_rms_eo_gap_stable_cells: float | None
    comparison_rms_eo_gap_stable_cells: float | None
    rms_eo_gap_stable_difference: MetricCI | None

    stable_group_min_count: int
    num_stable_professions: int

    profession_gaps_human: dict[str, float]
    profession_gaps_comparison: dict[str, float]
    profession_gap_differences: dict[str, float]


def load_prediction_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file does not exist: {path}"
        )

    rows: list[dict[str, Any]] = []

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
                row = json.loads(
                    line
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from e

            if not isinstance(
                row,
                dict,
            ):
                raise TypeError(
                    f"Expected object at {path}:{line_number}."
                )

            rows.append(
                row
            )

    if not rows:
        raise ValueError(
            f"No prediction rows found in {path}."
        )

    return rows


def _index_by_id(
    rows: Sequence[dict[str, Any]],
    *,
    source_name: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[
        str,
        dict[str, Any],
    ] = {}

    for row in rows:
        example_id = row.get(
            "id"
        )

        if example_id is None:
            raise ValueError(
                f"{source_name} prediction row is missing `id`."
            )

        example_id = str(
            example_id
        )

        if example_id in indexed:
            raise ValueError(
                f"Duplicate prediction id {example_id!r} in {source_name}."
            )

        indexed[
            example_id
        ] = row

    return indexed


def validate_and_align_predictions(
    *,
    human_rows: Sequence[dict[str, Any]],
    comparison_rows: Sequence[dict[str, Any]],
    correctness_field: str,
) -> list[
    tuple[
        str,
        dict[str, Any],
        dict[str, Any],
    ]
]:

    human = _index_by_id(
        human_rows,
        source_name="human",
    )

    comparison = _index_by_id(
        comparison_rows,
        source_name="comparison",
    )

    human_ids = set(
        human
    )

    comparison_ids = set(
        comparison
    )

    if human_ids != comparison_ids:
        only_human = sorted(
            human_ids
            - comparison_ids
        )[:10]

        only_comparison = sorted(
            comparison_ids
            - human_ids
        )[:10]

        raise ValueError(
            "Prediction files do not contain exactly the same example IDs.\n"
            f"Only human (first 10): {only_human}\n"
            f"Only comparison (first 10): {only_comparison}"
        )

    aligned = []

    required = {
        "gold_profession_id",
        "gold_profession",
        "gender",
        correctness_field,
    }

    for example_id in sorted(
        human_ids
    ):
        h = human[
            example_id
        ]

        c = comparison[
            example_id
        ]

        for source_name, row in (
            ("human", h),
            ("comparison", c),
        ):
            missing = (
                required
                - set(row)
            )

            if missing:
                raise ValueError(
                    f"{source_name} row {example_id!r} is missing "
                    f"fields: {sorted(missing)}"
                )

        for key in (
            "gold_profession_id",
            "gold_profession",
            "gender",
        ):
            if h[key] != c[key]:
                raise ValueError(
                    f"Metadata mismatch for id={example_id!r}, field={key!r}: "
                    f"human={h[key]!r}, comparison={c[key]!r}"
                )

        for source_name, row in (
            ("human", h),
            ("comparison", c),
        ):
            value = row[
                correctness_field
            ]

            if not isinstance(
                value,
                (
                    bool,
                    int,
                    float,
                ),
            ):
                raise TypeError(
                    f"{source_name} {correctness_field} for id={example_id!r} "
                    f"is not boolean/numeric: {value!r}"
                )

        aligned.append(
            (
                example_id,
                h,
                c,
            )
        )

    return aligned


@dataclass
class Stratum:
    profession_id: int
    profession_name: str
    gender: str

    human_correct: np.ndarray
    comparison_correct: np.ndarray

    @property
    def n(self) -> int:
        return int(
            len(
                self.human_correct
            )
        )


def build_strata(
    aligned: Sequence[
        tuple[
            str,
            dict[str, Any],
            dict[str, Any],
        ]
    ],
    *,
    correctness_field: str,
) -> list[Stratum]:
    grouped: dict[
        tuple[int, str, str],
        dict[str, list[float]],
    ] = {}

    for _, human, comparison in aligned:
        profession_id = int(
            human[
                "gold_profession_id"
            ]
        )

        profession_name = str(
            human[
                "gold_profession"
            ]
        )

        gender = str(
            human[
                "gender"
            ]
        ).strip().lower()

        if gender not in {
            "male",
            "female",
        }:
            raise ValueError(
                f"Expected canonical gender 'male'/'female', got {gender!r}."
            )

        key = (
            profession_id,
            profession_name,
            gender,
        )

        if key not in grouped:
            grouped[
                key
            ] = {
                "human": [],
                "comparison": [],
            }

        grouped[
            key
        ]["human"].append(
            float(
                bool(
                    human[
                        correctness_field
                    ]
                )
            )
        )

        grouped[
            key
        ]["comparison"].append(
            float(
                bool(
                    comparison[
                        correctness_field
                    ]
                )
            )
        )

    strata: list[
        Stratum
    ] = []

    for (
        profession_id,
        profession_name,
        gender,
    ), values in sorted(
        grouped.items()
    ):
        human_arr = np.asarray(
            values[
                "human"
            ],
            dtype=np.float64,
        )

        comparison_arr = np.asarray(
            values[
                "comparison"
            ],
            dtype=np.float64,
        )

        if (
            len(human_arr)
            != len(
                comparison_arr
            )
        ):
            raise RuntimeError(
                "Paired stratum lengths unexpectedly differ."
            )

        strata.append(
            Stratum(
                profession_id=(
                    profession_id
                ),
                profession_name=(
                    profession_name
                ),
                gender=gender,
                human_correct=(
                    human_arr
                ),
                comparison_correct=(
                    comparison_arr
                ),
            )
        )

    return strata

def _stratum_point_means(
    strata: Sequence[Stratum],
) -> tuple[
    dict[tuple[int, str], float],
    dict[tuple[int, str], float],
]:
    human: dict[
        tuple[int, str],
        float,
    ] = {}

    comparison: dict[
        tuple[int, str],
        float,
    ] = {}

    for stratum in strata:
        key = (
            stratum.profession_id,
            stratum.gender,
        )

        human[
            key
        ] = float(
            stratum.human_correct.mean()
        )

        comparison[
            key
        ] = float(
            stratum.comparison_correct.mean()
        )

    return (
        human,
        comparison,
    )


def _profession_names(
    strata: Sequence[Stratum],
) -> dict[int, str]:
    result = {}

    for stratum in strata:
        result[
            stratum.profession_id
        ] = stratum.profession_name

    return result


def _point_eo(
    *,
    means: dict[
        tuple[int, str],
        float,
    ],
    profession_ids: Sequence[int],
) -> tuple[
    float,
    dict[int, float],
]:
    gaps: dict[
        int,
        float,
    ] = {}

    for profession_id in profession_ids:
        male_key = (
            profession_id,
            "male",
        )

        female_key = (
            profession_id,
            "female",
        )

        if (
            male_key not in means
            or female_key not in means
        ):
            continue

        gaps[
            profession_id
        ] = (
            means[
                male_key
            ]
            - means[
                female_key
            ]
        )

    if not gaps:
        raise ValueError(
            "No professions contain both male and female examples."
        )

    values = np.asarray(
        list(
            gaps.values()
        ),
        dtype=np.float64,
    )

    rms = float(
        np.sqrt(
            np.mean(
                values ** 2
            )
        )
    )

    return (
        rms,
        gaps,
    )


def _weighted_accuracy(
    *,
    means: dict[
        tuple[int, str],
        float,
    ],
    counts: dict[
        tuple[int, str],
        int,
    ],
) -> float:
    numerator = 0.0
    denominator = 0

    for key, mean in means.items():
        n = counts[
            key
        ]

        numerator += (
            mean
            * n
        )

        denominator += n

    if denominator == 0:
        raise ValueError(
            "Cannot compute accuracy from zero examples."
        )

    return float(
        numerator
        / denominator
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

    low = float(
        np.quantile(
            values,
            alpha / 2.0,
        )
    )

    high = float(
        np.quantile(
            values,
            1.0 - alpha / 2.0,
        )
    )

    return (
        low,
        high,
    )


def _make_metric_ci(
    *,
    estimate: float,
    distribution: np.ndarray,
    confidence_level: float,
) -> MetricCI:
    low, high = _percentile_ci(
        distribution,
        confidence_level,
    )

    prob_le_zero = float(
        np.mean(
            distribution <= 0.0
        )
    )

    return MetricCI(
        estimate=float(
            estimate
        ),
        ci_low=low,
        ci_high=high,
        confidence_level=(
            confidence_level
        ),
        prob_le_zero=(
            prob_le_zero
        ),
        significant_increase=bool(
            low > 0.0
        ),
        significant_decrease=bool(
            high < 0.0
        ),
    )


def _validate_bootstrap_config(
    config: BootstrapConfig,
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

    if config.bootstrap_chunk_size <= 0:
        raise ValueError(
            "bootstrap_chunk_size must be > 0."
        )

    if config.stable_group_min_count <= 0:
        raise ValueError(
            "stable_group_min_count must be > 0."
        )


def paired_stratified_bootstrap(
    *,
    human_predictions_path: str | Path,
    comparison_predictions_path: str | Path,
    config: BootstrapConfig | None = None,
) -> tuple[
    PairedBootstrapResult,
    dict[str, np.ndarray],
]:
    """
    Compare a checkpoint against a matched human-only checkpoint, i.e.:
        H_2 predictions vs. R_2 predictions.
    """

    if config is None:
        config = (
            BootstrapConfig()
        )

    _validate_bootstrap_config(
        config
    )

    human_rows = (
        load_prediction_jsonl(
            human_predictions_path
        )
    )

    comparison_rows = (
        load_prediction_jsonl(
            comparison_predictions_path
        )
    )

    aligned = (
        validate_and_align_predictions(
            human_rows=human_rows,
            comparison_rows=(
                comparison_rows
            ),
            correctness_field=(
                config.correctness_field
            ),
        )
    )

    strata = build_strata(
        aligned,
        correctness_field=(
            config.correctness_field
        ),
    )

    counts = {
        (
            stratum.profession_id,
            stratum.gender,
        ): stratum.n
        for stratum in strata
    }

    names = _profession_names(
        strata
    )

    profession_ids = sorted(
        names
    )

    (
        human_point_means,
        comparison_point_means,
    ) = _stratum_point_means(
        strata
    )

    human_accuracy = (
        _weighted_accuracy(
            means=human_point_means,
            counts=counts,
        )
    )

    comparison_accuracy = (
        _weighted_accuracy(
            means=(
                comparison_point_means
            ),
            counts=counts,
        )
    )

    (
        human_eo,
        human_profession_gaps,
    ) = _point_eo(
        means=human_point_means,
        profession_ids=(
            profession_ids
        ),
    )

    (
        comparison_eo,
        comparison_profession_gaps,
    ) = _point_eo(
        means=(
            comparison_point_means
        ),
        profession_ids=(
            profession_ids
        ),
    )

    stable_profession_ids = []

    for profession_id in profession_ids:
        male_n = counts.get(
            (
                profession_id,
                "male",
            ),
            0,
        )

        female_n = counts.get(
            (
                profession_id,
                "female",
            ),
            0,
        )

        if (
            male_n
            >= config.stable_group_min_count
            and female_n
            >= config.stable_group_min_count
        ):
            stable_profession_ids.append(
                profession_id
            )

    if stable_profession_ids:
        (
            human_stable_eo,
            _,
        ) = _point_eo(
            means=(
                human_point_means
            ),
            profession_ids=(
                stable_profession_ids
            ),
        )

        (
            comparison_stable_eo,
            _,
        ) = _point_eo(
            means=(
                comparison_point_means
            ),
            profession_ids=(
                stable_profession_ids
            ),
        )
    else:
        human_stable_eo = None
        comparison_stable_eo = None

    rng = np.random.default_rng(
        config.seed
    )

    accuracy_difference = np.empty(
        config.n_bootstrap,
        dtype=np.float64,
    )

    eo_difference = np.empty(
        config.n_bootstrap,
        dtype=np.float64,
    )

    stable_eo_difference = (
        np.empty(
            config.n_bootstrap,
            dtype=np.float64,
        )
        if stable_profession_ids
        else None
    )

    stratum_lookup = {
        (
            stratum.profession_id,
            stratum.gender,
        ): index
        for index, stratum
        in enumerate(
            strata
        )
    }

    total_n = sum(
        stratum.n
        for stratum in strata
    )

    offset = 0

    while offset < config.n_bootstrap:
        chunk_n = min(
            config.bootstrap_chunk_size,
            config.n_bootstrap
            - offset,
        )

        human_means = np.empty(
            (
                chunk_n,
                len(strata),
            ),
            dtype=np.float64,
        )

        comparison_means = np.empty_like(
            human_means
        )

        for s_idx, stratum in enumerate(
            strata
        ):
            sample_indices = rng.integers(
                0,
                stratum.n,
                size=(
                    chunk_n,
                    stratum.n,
                ),
            )

            human_means[
                :,
                s_idx,
            ] = (
                stratum.human_correct[
                    sample_indices
                ].mean(
                    axis=1
                )
            )

            comparison_means[
                :,
                s_idx,
            ] = (
                stratum.comparison_correct[
                    sample_indices
                ].mean(
                    axis=1
                )
            )

        weights = np.asarray(
            [
                stratum.n
                for stratum in strata
            ],
            dtype=np.float64,
        )

        human_acc_chunk = (
            human_means
            @ weights
        ) / total_n

        comparison_acc_chunk = (
            comparison_means
            @ weights
        ) / total_n

        accuracy_difference[
            offset:
            offset + chunk_n
        ] = (
            comparison_acc_chunk
            - human_acc_chunk
        )

        human_gap_columns = []
        comparison_gap_columns = []

        for profession_id in profession_ids:
            male_key = (
                profession_id,
                "male",
            )

            female_key = (
                profession_id,
                "female",
            )

            if (
                male_key
                not in stratum_lookup
                or female_key
                not in stratum_lookup
            ):
                continue

            male_idx = (
                stratum_lookup[
                    male_key
                ]
            )

            female_idx = (
                stratum_lookup[
                    female_key
                ]
            )

            human_gap_columns.append(
                human_means[
                    :,
                    male_idx,
                ]
                - human_means[
                    :,
                    female_idx,
                ]
            )

            comparison_gap_columns.append(
                comparison_means[
                    :,
                    male_idx,
                ]
                - comparison_means[
                    :,
                    female_idx,
                ]
            )

        human_gap_matrix = np.stack(
            human_gap_columns,
            axis=1,
        )

        comparison_gap_matrix = np.stack(
            comparison_gap_columns,
            axis=1,
        )

        human_eo_chunk = np.sqrt(
            np.mean(
                human_gap_matrix ** 2,
                axis=1,
            )
        )

        comparison_eo_chunk = np.sqrt(
            np.mean(
                comparison_gap_matrix ** 2,
                axis=1,
            )
        )

        eo_difference[
            offset:
            offset + chunk_n
        ] = (
            comparison_eo_chunk
            - human_eo_chunk
        )

        if (
            stable_eo_difference
            is not None
        ):
            human_stable_columns = []
            comparison_stable_columns = []

            for profession_id in (
                stable_profession_ids
            ):
                male_idx = (
                    stratum_lookup[
                        (
                            profession_id,
                            "male",
                        )
                    ]
                )

                female_idx = (
                    stratum_lookup[
                        (
                            profession_id,
                            "female",
                        )
                    ]
                )

                human_stable_columns.append(
                    human_means[
                        :,
                        male_idx,
                    ]
                    - human_means[
                        :,
                        female_idx,
                    ]
                )

                comparison_stable_columns.append(
                    comparison_means[
                        :,
                        male_idx,
                    ]
                    - comparison_means[
                        :,
                        female_idx,
                    ]
                )

            human_stable_matrix = np.stack(
                human_stable_columns,
                axis=1,
            )

            comparison_stable_matrix = np.stack(
                comparison_stable_columns,
                axis=1,
            )

            human_stable_chunk = np.sqrt(
                np.mean(
                    human_stable_matrix ** 2,
                    axis=1,
                )
            )

            comparison_stable_chunk = np.sqrt(
                np.mean(
                    comparison_stable_matrix ** 2,
                    axis=1,
                )
            )

            stable_eo_difference[
                offset:
                offset + chunk_n
            ] = (
                comparison_stable_chunk
                - human_stable_chunk
            )

        offset += chunk_n

    accuracy_ci = _make_metric_ci(
        estimate=(
            comparison_accuracy
            - human_accuracy
        ),
        distribution=(
            accuracy_difference
        ),
        confidence_level=(
            config.confidence_level
        ),
    )

    eo_ci = _make_metric_ci(
        estimate=(
            comparison_eo
            - human_eo
        ),
        distribution=(
            eo_difference
        ),
        confidence_level=(
            config.confidence_level
        ),
    )

    stable_ci = None

    if (
        stable_eo_difference
        is not None
        and human_stable_eo
        is not None
        and comparison_stable_eo
        is not None
    ):
        stable_ci = _make_metric_ci(
            estimate=(
                comparison_stable_eo
                - human_stable_eo
            ),
            distribution=(
                stable_eo_difference
            ),
            confidence_level=(
                config.confidence_level
            ),
        )

    human_gap_named = {
        names[
            profession_id
        ]: float(gap)
        for profession_id, gap
        in human_profession_gaps.items()
    }

    comparison_gap_named = {
        names[
            profession_id
        ]: float(gap)
        for profession_id, gap
        in comparison_profession_gaps.items()
    }

    difference_gap_named = {
        names[
            profession_id
        ]: float(
            comparison_profession_gaps[
                profession_id
            ]
            - human_profession_gaps[
                profession_id
            ]
        )
        for profession_id
        in sorted(
            set(
                human_profession_gaps
            )
            & set(
                comparison_profession_gaps
            )
        )
    }

    result = (
        PairedBootstrapResult(
            human_predictions_path=str(
                Path(
                    human_predictions_path
                ).resolve()
            ),
            comparison_predictions_path=str(
                Path(
                    comparison_predictions_path
                ).resolve()
            ),
            correctness_field=(
                config.correctness_field
            ),
            num_examples=len(
                aligned
            ),
            num_professions=len(
                profession_ids
            ),
            num_strata=len(
                strata
            ),
            n_bootstrap=(
                config.n_bootstrap
            ),
            confidence_level=(
                config.confidence_level
            ),
            seed=config.seed,
            human_accuracy=(
                human_accuracy
            ),
            comparison_accuracy=(
                comparison_accuracy
            ),
            accuracy_difference=(
                accuracy_ci
            ),
            human_rms_eo_gap=(
                human_eo
            ),
            comparison_rms_eo_gap=(
                comparison_eo
            ),
            rms_eo_gap_difference=(
                eo_ci
            ),
            human_rms_eo_gap_stable_cells=(
                human_stable_eo
            ),
            comparison_rms_eo_gap_stable_cells=(
                comparison_stable_eo
            ),
            rms_eo_gap_stable_difference=(
                stable_ci
            ),
            stable_group_min_count=(
                config.stable_group_min_count
            ),
            num_stable_professions=len(
                stable_profession_ids
            ),
            profession_gaps_human=(
                human_gap_named
            ),
            profession_gaps_comparison=(
                comparison_gap_named
            ),
            profession_gap_differences=(
                difference_gap_named
            ),
        )
    )

    distributions = {
        "accuracy_difference": (
            accuracy_difference
        ),
        "rms_eo_gap_difference": (
            eo_difference
        ),
    }

    if (
        stable_eo_difference
        is not None
    ):
        distributions[
            "rms_eo_gap_stable_difference"
        ] = stable_eo_difference

    return (
        result,
        distributions,
    )

def save_bootstrap_result(
    *,
    result: PairedBootstrapResult,
    output_dir: str | Path,
    distributions: dict[
        str,
        np.ndarray,
    ] | None = None,
    save_distributions: bool = False,
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
        / "paired_bootstrap.json"
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
        save_distributions
        and distributions is not None
    ):
        np.savez_compressed(
            output_dir
            / "paired_bootstrap_distributions.npz",
            **distributions,
        )
