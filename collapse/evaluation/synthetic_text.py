from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from collapse.records import Example

DEFAULT_LEXICONS: dict[str, list[str]] = {
    "status_achievement": [
        "accomplished",
        "achievement",
        "award",
        "awarded",
        "award-winning",
        "distinguished",
        "expert",
        "leading",
        "prestigious",
        "prominent",
        "renowned",
        "respected",
        "successful",
        "top",
        "recognized",
    ],
    "care_helping": [
        "care",
        "caring",
        "compassion",
        "compassionate",
        "help",
        "helped",
        "helping",
        "support",
        "supported",
        "supporting",
        "nurture",
        "nurturing",
        "empathetic",
        "empathy",
        "community",
    ],
    "leadership_agency": [
        "lead",
        "leader",
        "leadership",
        "led",
        "direct",
        "directed",
        "director",
        "manage",
        "managed",
        "manager",
        "founded",
        "founder",
        "established",
        "headed",
        "oversaw",
    ],
    "family_relationship": [
        "family",
        "husband",
        "wife",
        "married",
        "mother",
        "father",
        "parent",
        "parents",
        "children",
        "child",
        "daughter",
        "son",
        "partner",
    ],
}


PRONOUN_LEXICONS: dict[str, set[str]] = {
    "male_pronoun": {
        "he",
        "him",
        "his",
        "himself",
    },
    "female_pronoun": {
        "she",
        "her",
        "hers",
        "herself",
    },
    "neutral_pronoun": {
        "they",
        "them",
        "their",
        "theirs",
        "themself",
        "themselves",
    },
}

@dataclass(frozen=True)
class SyntheticTextConfig:
    bootstrap_replicates: int = 5_000
    confidence_level: float = 0.95
    bootstrap_seed: int = 34567

    min_group_size: int = 10

    use_default_lexicons: bool = True
    lexical_categories: dict[str, list[str]] | None = None

    analyze_full_text: bool = True
    analyze_suffix: bool = True
    save_per_example: bool = True


@dataclass(frozen=True)
class TextFeatures:
    word_count: int
    char_count: int
    sentence_count: int
    avg_sentence_words: float
    type_token_ratio: float

    male_pronoun_per_100: float
    female_pronoun_per_100: float
    neutral_pronoun_per_100: float

    lexical_per_100: dict[str, float]


@dataclass(frozen=True)
class PairedFeatureRow:
    id: str
    profession_id: int | None
    profession: str | None
    gender: str | None
    scope: str

    feature: str

    human_value: float
    synthetic_value: float
    difference: float


@dataclass(frozen=True)
class GroupFeatureStats:
    scope: str
    feature: str

    profession_id: int | None
    profession: str | None
    gender: str | None

    n: int

    human_mean: float
    synthetic_mean: float

    mean_difference: float
    median_difference: float

    ci_low: float | None
    ci_high: float | None
    confidence_level: float

    wilcoxon_pvalue: float | None
    bh_qvalue: float | None

    significant_after_bh: bool | None


@dataclass(frozen=True)
class SyntheticTextResult:
    num_pairs: int

    analyzed_scopes: list[str]
    lexical_categories: list[str]

    global_stats: list[GroupFeatureStats]
    profession_gender_stats: list[GroupFeatureStats]

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

_WORD_RE = re.compile(
    r"[A-Za-z]+(?:['’-][A-Za-z]+)?"
)

_SENTENCE_RE = re.compile(
    r"[.!?]+"
)


def word_tokens(
    text: str,
) -> list[str]:
    return [
        match.group(0).lower()
        for match in _WORD_RE.finditer(
            text
        )
    ]


def sentence_count(
    text: str,
) -> int:
    stripped = text.strip()

    if not stripped:
        return 0

    matches = _SENTENCE_RE.findall(
        stripped
    )

    return max(
        1,
        len(matches),
    )


def _normalize_phrase(
    value: str,
) -> tuple[str, ...]:
    return tuple(
        word_tokens(
            value
        )
    )


def build_lexicons(
    config: SyntheticTextConfig,
) -> dict[
    str,
    list[tuple[str, ...]],
]:
    raw: dict[
        str,
        list[str],
    ] = {}

    if config.use_default_lexicons:
        for key, values in DEFAULT_LEXICONS.items():
            raw[key] = list(
                values
            )

    if config.lexical_categories:
        for key, values in config.lexical_categories.items():
            raw[key] = list(
                values
            )

    result: dict[
        str,
        list[tuple[str, ...]],
    ] = {}

    for category, phrases in raw.items():
        normalized = []

        for phrase in phrases:
            tokens = _normalize_phrase(
                phrase
            )

            if tokens:
                normalized.append(
                    tokens
                )

        result[
            category
        ] = normalized

    return result


def _count_phrase(
    tokens: Sequence[str],
    phrase: Sequence[str],
) -> int:
    if not phrase:
        return 0

    length = len(
        phrase
    )

    if length > len(tokens):
        return 0

    count = 0

    for i in range(
        0,
        len(tokens) - length + 1,
    ):
        if tuple(
            tokens[
                i:
                i + length
            ]
        ) == tuple(phrase):
            count += 1

    return count


def extract_text_features(
    text: str,
    *,
    lexicons: dict[
        str,
        list[tuple[str, ...]],
    ],
) -> TextFeatures:
    tokens = word_tokens(
        text
    )

    n_words = len(
        tokens
    )

    n_sentences = sentence_count(
        text
    )

    if n_words:
        ttr = (
            len(
                set(tokens)
            )
            / n_words
        )

        avg_sentence_words = (
            n_words
            / max(
                1,
                n_sentences,
            )
        )
    else:
        ttr = 0.0
        avg_sentence_words = 0.0

    scale = (
        100.0
        / n_words
        if n_words
        else 0.0
    )

    pronoun_counts = {}

    token_set_counts = {}

    for token in tokens:
        token_set_counts[
            token
        ] = (
            token_set_counts.get(
                token,
                0,
            )
            + 1
        )

    for category, vocabulary in PRONOUN_LEXICONS.items():
        count = sum(
            token_set_counts.get(
                token,
                0,
            )
            for token in vocabulary
        )

        pronoun_counts[
            category
        ] = (
            count
            * scale
        )

    lexical_per_100 = {}

    for category, phrases in lexicons.items():
        count = sum(
            _count_phrase(
                tokens,
                phrase,
            )
            for phrase in phrases
        )

        lexical_per_100[
            category
        ] = (
            count
            * scale
        )

    return TextFeatures(
        word_count=n_words,
        char_count=len(
            text
        ),
        sentence_count=(
            n_sentences
        ),
        avg_sentence_words=float(
            avg_sentence_words
        ),
        type_token_ratio=float(
            ttr
        ),
        male_pronoun_per_100=float(
            pronoun_counts[
                "male_pronoun"
            ]
        ),
        female_pronoun_per_100=float(
            pronoun_counts[
                "female_pronoun"
            ]
        ),
        neutral_pronoun_per_100=float(
            pronoun_counts[
                "neutral_pronoun"
            ]
        ),
        lexical_per_100={
            key: float(value)
            for key, value
            in lexical_per_100.items()
        },
    )


def flatten_features(
    features: TextFeatures,
) -> dict[str, float]:
    output = {
        "word_count": float(
            features.word_count
        ),
        "char_count": float(
            features.char_count
        ),
        "sentence_count": float(
            features.sentence_count
        ),
        "avg_sentence_words": float(
            features.avg_sentence_words
        ),
        "type_token_ratio": float(
            features.type_token_ratio
        ),
        "male_pronoun_per_100": float(
            features.male_pronoun_per_100
        ),
        "female_pronoun_per_100": float(
            features.female_pronoun_per_100
        ),
        "neutral_pronoun_per_100": float(
            features.neutral_pronoun_per_100
        ),
    }

    for category, value in (
        features.lexical_per_100.items()
    ):
        output[
            f"lex_{category}_per_100"
        ] = float(
            value
        )

    return output

def _index_examples(
    examples: Sequence[Example],
    *,
    name: str,
) -> dict[str, Example]:
    result = {}

    for example in examples:
        example_id = str(
            example.id
        )

        if example_id in result:
            raise ValueError(
                f"Duplicate example id {example_id!r} in {name} examples."
            )

        result[
            example_id
        ] = example

    return result


def align_human_and_synthetic(
    *,
    human_examples: Sequence[Example],
    synthetic_examples: Sequence[Example],
) -> list[
    tuple[
        Example,
        Example,
    ]
]:
    human = _index_examples(
        human_examples,
        name="human",
    )

    synthetic = _index_examples(
        synthetic_examples,
        name="synthetic",
    )

    h_ids = set(
        human
    )

    s_ids = set(
        synthetic
    )

    if h_ids != s_ids:
        only_h = sorted(
            h_ids
            - s_ids
        )[:10]

        only_s = sorted(
            s_ids
            - h_ids
        )[:10]

        raise ValueError(
            "Human and synthetic datasets do not contain exactly the same "
            "example IDs.\n"
            f"Only human (first 10): {only_h}\n"
            f"Only synthetic (first 10): {only_s}"
        )

    return [
        (
            human[
                example_id
            ],
            synthetic[
                example_id
            ],
        )
        for example_id in sorted(
            h_ids
        )
    ]


def _metadata_value(
    human: Example,
    synthetic: Example,
    *keys: str,
) -> Any:
    for source in (
        synthetic.metadata,
        human.metadata,
    ):
        for key in keys:
            if key in source:
                return source[
                    key
                ]

    return None


def example_group_metadata(
    human: Example,
    synthetic: Example,
) -> tuple[
    int | None,
    str | None,
    str | None,
]:
    profession_id_raw = (
        _metadata_value(
            human,
            synthetic,
            "profession_id",
            "profession",
        )
    )

    profession_id = None
    profession = None

    if profession_id_raw is not None:
        try:
            profession_id = int(
                profession_id_raw
            )
        except (
            TypeError,
            ValueError,
        ):
            profession = str(
                profession_id_raw
            )

    profession_name_raw = (
        _metadata_value(
            human,
            synthetic,
            "profession_name",
            "profession_label",
        )
    )

    if profession_name_raw is not None:
        profession = str(
            profession_name_raw
        )

    gender_raw = _metadata_value(
        human,
        synthetic,
        "gender",
    )

    gender = (
        str(
            gender_raw
        ).strip().lower()
        if gender_raw is not None
        else None
    )

    return (
        profession_id,
        profession,
        gender,
    )

def _scope_texts(
    *,
    human: Example,
    synthetic: Example,
    scope: str,
) -> tuple[
    str,
    str,
] | None:
    if scope == "full":
        return (
            human.text,
            synthetic.text,
        )

    if scope == "suffix":
        human_suffix = (
            human.human_suffix
            if human.human_suffix
            is not None
            else synthetic.human_suffix
        )

        synthetic_suffix = (
            synthetic.synthetic_suffix
        )

        if (
            human_suffix is None
            or synthetic_suffix is None
        ):
            return None

        return (
            human_suffix,
            synthetic_suffix,
        )

    raise ValueError(
        f"Unknown scope: {scope}"
    )


def build_paired_feature_rows(
    *,
    human_examples: Sequence[Example],
    synthetic_examples: Sequence[Example],
    config: SyntheticTextConfig,
) -> tuple[
    list[PairedFeatureRow],
    dict[str, list[dict[str, Any]]],
]:
    aligned = (
        align_human_and_synthetic(
            human_examples=(
                human_examples
            ),
            synthetic_examples=(
                synthetic_examples
            ),
        )
    )

    lexicons = build_lexicons(
        config
    )

    scopes = []

    if config.analyze_full_text:
        scopes.append(
            "full"
        )

    if config.analyze_suffix:
        scopes.append(
            "suffix"
        )

    if not scopes:
        raise ValueError(
            "At least one analysis scope must be enabled."
        )

    paired_rows: list[
        PairedFeatureRow
    ] = []

    per_example: dict[
        str,
        list[dict[str, Any]],
    ] = {
        scope: []
        for scope in scopes
    }

    for human, synthetic in aligned:
        (
            profession_id,
            profession,
            gender,
        ) = example_group_metadata(
            human,
            synthetic,
        )

        for scope in scopes:
            pair = _scope_texts(
                human=human,
                synthetic=synthetic,
                scope=scope,
            )

            if pair is None:
                continue

            human_text, synthetic_text = (
                pair
            )

            human_features = (
                flatten_features(
                    extract_text_features(
                        human_text,
                        lexicons=lexicons,
                    )
                )
            )

            synthetic_features = (
                flatten_features(
                    extract_text_features(
                        synthetic_text,
                        lexicons=lexicons,
                    )
                )
            )

            record = {
                "id": str(
                    human.id
                ),
                "profession_id": (
                    profession_id
                ),
                "profession": profession,
                "gender": gender,
                "scope": scope,
                "human": (
                    human_features
                ),
                "synthetic": (
                    synthetic_features
                ),
            }

            per_example[
                scope
            ].append(
                record
            )

            for feature in sorted(
                human_features
            ):
                human_value = float(
                    human_features[
                        feature
                    ]
                )

                synthetic_value = float(
                    synthetic_features[
                        feature
                    ]
                )

                paired_rows.append(
                    PairedFeatureRow(
                        id=str(
                            human.id
                        ),
                        profession_id=(
                            profession_id
                        ),
                        profession=(
                            profession
                        ),
                        gender=gender,
                        scope=scope,
                        feature=feature,
                        human_value=(
                            human_value
                        ),
                        synthetic_value=(
                            synthetic_value
                        ),
                        difference=(
                            synthetic_value
                            - human_value
                        ),
                    )
                )

    return (
        paired_rows,
        per_example,
    )

def _bootstrap_mean_difference_ci(
    differences: np.ndarray,
    *,
    n_bootstrap: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[
    float,
    float,
]:
    if len(
        differences
    ) == 0:
        raise ValueError(
            "Cannot bootstrap an empty array."
        )

    distribution = np.empty(
        n_bootstrap,
        dtype=np.float64,
    )

    n = len(
        differences
    )

    chunk_size = 500
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

        distribution[
            offset:
            offset + chunk_n
        ] = differences[
            indices
        ].mean(
            axis=1
        )

        offset += chunk_n

    alpha = (
        1.0
        - confidence_level
    )

    return (
        float(
            np.quantile(
                distribution,
                alpha / 2.0,
            )
        ),
        float(
            np.quantile(
                distribution,
                1.0 - alpha / 2.0,
            )
        ),
    )


def _wilcoxon_pvalue(
    human: np.ndarray,
    synthetic: np.ndarray,
) -> float | None:
    differences = (
        synthetic
        - human
    )

    if np.allclose(
        differences,
        0.0,
    ):
        return 1.0

    try:
        from scipy.stats import wilcoxon
    except Exception:
        return None

    try:
        result = wilcoxon(
            synthetic,
            human,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="auto",
        )

        return float(
            result.pvalue
        )
    except ValueError:
        return None


def benjamini_hochberg(
    pvalues: Sequence[
        float | None
    ],
) -> list[
    float | None
]:
    indexed = [
        (
            index,
            float(p)
        )
        for index, p in enumerate(
            pvalues
        )
        if p is not None
        and not math.isnan(
            float(p)
        )
    ]

    if not indexed:
        return [
            None
            for _ in pvalues
        ]

    indexed.sort(
        key=lambda item: (
            item[1]
        )
    )

    m = len(
        indexed
    )

    adjusted_sorted = [
        0.0
        for _ in range(
            m
        )
    ]

    running = 1.0

    for reverse_rank in range(
        m - 1,
        -1,
        -1,
    ):
        original_index, p = (
            indexed[
                reverse_rank
            ]
        )

        rank = (
            reverse_rank
            + 1
        )

        adjusted = min(
            running,
            p
            * m
            / rank,
            1.0,
        )

        running = adjusted

        adjusted_sorted[
            reverse_rank
        ] = adjusted

    result: list[
        float | None
    ] = [
        None
        for _ in pvalues
    ]

    for (
        adjusted,
        (
            original_index,
            _,
        ),
    ) in zip(
        adjusted_sorted,
        indexed,
    ):
        result[
            original_index
        ] = float(
            adjusted
        )

    return result


def _group_stats(
    *,
    rows: Sequence[
        PairedFeatureRow
    ],
    config: SyntheticTextConfig,
    rng: np.random.Generator,
    profession_id: int | None,
    profession: str | None,
    gender: str | None,
) -> GroupFeatureStats:
    if not rows:
        raise ValueError(
            "Cannot summarize empty group."
        )

    human = np.asarray(
        [
            row.human_value
            for row in rows
        ],
        dtype=np.float64,
    )

    synthetic = np.asarray(
        [
            row.synthetic_value
            for row in rows
        ],
        dtype=np.float64,
    )

    differences = (
        synthetic
        - human
    )

    if len(
        rows
    ) >= config.min_group_size:
        ci_low, ci_high = (
            _bootstrap_mean_difference_ci(
                differences,
                n_bootstrap=(
                    config.bootstrap_replicates
                ),
                confidence_level=(
                    config.confidence_level
                ),
                rng=rng,
            )
        )

        wilcoxon_p = (
            _wilcoxon_pvalue(
                human,
                synthetic,
            )
        )
    else:
        ci_low = None
        ci_high = None
        wilcoxon_p = None

    first = rows[
        0
    ]

    return GroupFeatureStats(
        scope=first.scope,
        feature=first.feature,
        profession_id=(
            profession_id
        ),
        profession=(
            profession
        ),
        gender=gender,
        n=len(
            rows
        ),
        human_mean=float(
            human.mean()
        ),
        synthetic_mean=float(
            synthetic.mean()
        ),
        mean_difference=float(
            differences.mean()
        ),
        median_difference=float(
            np.median(
                differences
            )
        ),
        ci_low=ci_low,
        ci_high=ci_high,
        confidence_level=(
            config.confidence_level
        ),
        wilcoxon_pvalue=(
            wilcoxon_p
        ),
        bh_qvalue=None,
        significant_after_bh=None,
    )


def summarize_global(
    *,
    rows: Sequence[
        PairedFeatureRow
    ],
    config: SyntheticTextConfig,
    rng: np.random.Generator,
) -> list[
    GroupFeatureStats
]:
    grouped: dict[
        tuple[
            str,
            str,
            str | None,
        ],
        list[
            PairedFeatureRow
        ],
    ] = {}

    for row in rows:
        for gender in (
            None,
            row.gender,
        ):
            key = (
                row.scope,
                row.feature,
                gender,
            )

            grouped.setdefault(
                key,
                [],
            ).append(
                row
            )

    output = []

    for (
        _,
        _,
        gender,
    ), group in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            str(
                item[0][2]
            ),
        ),
    ):
        output.append(
            _group_stats(
                rows=group,
                config=config,
                rng=rng,
                profession_id=None,
                profession=None,
                gender=gender,
            )
        )

    return output


def summarize_profession_gender(
    *,
    rows: Sequence[
        PairedFeatureRow
    ],
    config: SyntheticTextConfig,
    rng: np.random.Generator,
) -> list[
    GroupFeatureStats
]:
    grouped: dict[
        tuple[
            str,
            str,
            int | None,
            str | None,
            str | None,
        ],
        list[
            PairedFeatureRow
        ],
    ] = {}

    for row in rows:
        key = (
            row.scope,
            row.feature,
            row.profession_id,
            row.profession,
            row.gender,
        )

        grouped.setdefault(
            key,
            [],
        ).append(
            row
        )

    provisional = []

    for (
        _,
        _,
        profession_id,
        profession,
        gender,
    ), group in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            (
                item[0][2]
                if item[0][2]
                is not None
                else 10_000
            ),
            str(
                item[0][3]
            ),
            str(
                item[0][4]
            ),
        ),
    ):
        provisional.append(
            _group_stats(
                rows=group,
                config=config,
                rng=rng,
                profession_id=(
                    profession_id
                ),
                profession=(
                    profession
                ),
                gender=gender,
            )
        )

    family_indices: dict[
        tuple[
            str,
            str,
            str | None,
        ],
        list[int],
    ] = {}

    for index, stat in enumerate(
        provisional
    ):
        family = (
            stat.scope,
            stat.feature,
            stat.gender,
        )

        family_indices.setdefault(
            family,
            [],
        ).append(
            index
        )

    corrected = list(
        provisional
    )

    for indices in family_indices.values():
        pvalues = [
            provisional[
                i
            ].wilcoxon_pvalue
            for i in indices
        ]

        qvalues = (
            benjamini_hochberg(
                pvalues
            )
        )

        for i, q in zip(
            indices,
            qvalues,
        ):
            old = provisional[
                i
            ]

            corrected[
                i
            ] = (
                GroupFeatureStats(
                    scope=old.scope,
                    feature=old.feature,
                    profession_id=(
                        old.profession_id
                    ),
                    profession=(
                        old.profession
                    ),
                    gender=old.gender,
                    n=old.n,
                    human_mean=(
                        old.human_mean
                    ),
                    synthetic_mean=(
                        old.synthetic_mean
                    ),
                    mean_difference=(
                        old.mean_difference
                    ),
                    median_difference=(
                        old.median_difference
                    ),
                    ci_low=old.ci_low,
                    ci_high=old.ci_high,
                    confidence_level=(
                        old.confidence_level
                    ),
                    wilcoxon_pvalue=(
                        old.wilcoxon_pvalue
                    ),
                    bh_qvalue=q,
                    significant_after_bh=(
                        bool(
                            q < 0.05
                        )
                        if q
                        is not None
                        else None
                    ),
                )
            )

    return corrected

def analyze_synthetic_text(
    *,
    human_examples: Sequence[Example],
    synthetic_examples: Sequence[Example],
    config: SyntheticTextConfig | None = None,
) -> tuple[
    SyntheticTextResult,
    list[PairedFeatureRow],
    dict[
        str,
        list[dict[str, Any]],
    ],
]:
    if config is None:
        config = (
            SyntheticTextConfig()
        )

    if config.bootstrap_replicates < 100:
        raise ValueError(
            "bootstrap_replicates should be at least 100."
        )

    if not (
        0.0
        < config.confidence_level
        < 1.0
    ):
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    (
        paired_rows,
        per_example,
    ) = build_paired_feature_rows(
        human_examples=(
            human_examples
        ),
        synthetic_examples=(
            synthetic_examples
        ),
        config=config,
    )

    if not paired_rows:
        raise ValueError(
            "No paired text features were produced."
        )

    rng = np.random.default_rng(
        config.bootstrap_seed
    )

    global_stats = summarize_global(
        rows=paired_rows,
        config=config,
        rng=rng,
    )

    profession_gender_stats = (
        summarize_profession_gender(
            rows=paired_rows,
            config=config,
            rng=rng,
        )
    )

    lexical_categories = sorted(
        build_lexicons(
            config
        )
    )

    scopes = sorted(
        {
            row.scope
            for row in paired_rows
        }
    )

    num_pairs = len(
        {
            row.id
            for row in paired_rows
        }
    )

    result = SyntheticTextResult(
        num_pairs=num_pairs,
        analyzed_scopes=scopes,
        lexical_categories=(
            lexical_categories
        ),
        global_stats=(
            global_stats
        ),
        profession_gender_stats=(
            profession_gender_stats
        ),
        metadata={
            "difference_definition": (
                "synthetic - human"
            ),
            "lexical_rate_unit": (
                "occurrences per 100 words"
            ),
            "profession_test_correction": (
                "Benjamini-Hochberg within each "
                "scope × feature × gender family"
            ),
            "suffix_note": (
                "For seeded continuation, suffix analysis compares "
                "synthetic_suffix against the matched human_suffix while "
                "holding the human prefix fixed."
            ),
        },
    )

    return (
        result,
        paired_rows,
        per_example,
    )


def save_synthetic_text_result(
    *,
    result: SyntheticTextResult,
    paired_rows: Sequence[
        PairedFeatureRow
    ],
    per_example: dict[
        str,
        list[dict[str, Any]],
    ],
    output_dir: str | Path,
    save_per_example: bool = True,
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
        / "synthetic_text_summary.json"
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

    with (
        output_dir
        / "paired_feature_rows.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in paired_rows:
            f.write(
                json.dumps(
                    asdict(
                        row
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )

    if save_per_example:
        with (
            output_dir
            / "per_example_features.jsonl"
        ).open(
            "w",
            encoding="utf-8",
        ) as f:
            for scope in sorted(
                per_example
            ):
                for row in per_example[
                    scope
                ]:
                    f.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
