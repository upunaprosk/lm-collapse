from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset

from collapse.data.base import PreparedDataset
from collapse.records import Example


PROFESSIONS_CANON = [
    "accountant",
    "architect",
    "attorney",
    "chiropractor",
    "comedian",
    "composer",
    "dentist",
    "dietitian",
    "dj",
    "filmmaker",
    "interior_designer",
    "journalist",
    "model",
    "nurse",
    "painter",
    "paralegal",
    "pastor",
    "personal_trainer",
    "photographer",
    "physician",
    "poet",
    "professor",
    "psychologist",
    "rapper",
    "software_engineer",
    "surgeon",
    "teacher",
    "yoga_teacher",
]

PROFESSION_NAME_TO_ID = {
    name: index
    for index, name in enumerate(PROFESSIONS_CANON)
}


@dataclass(frozen=True)
class BiasInBiosConfig:
    dataset_name: str

    dataset_config_name: str | None = None
    revision: str | None = None

    text_column: str = "hard_text"
    profession_column: str = "profession"
    gender_column: str | None = "gender"
    id_column: str | None = None

    # canonical split name -> HF split name
    splits: dict[str, str] = field(
        default_factory=lambda: {
            "train": "train",
        }
    )

    preserve_columns: list[str] = field(
        default_factory=list
    )

    # Set explicitly when integer labels are not safely inferable.
    # 0 => 0..27, 1 => 1..28.
    profession_index_base: int | None = None

    # Example: {"0": "female", "1": "male"}
    gender_map: dict[str, str] | None = None


class BiasInBiosAdapter:
    def __init__(
        self,
        config: BiasInBiosConfig,
    ) -> None:
        self.config = config

    def _load(self):
        kwargs: dict[str, Any] = {}

        if self.config.revision is not None:
            kwargs["revision"] = (
                self.config.revision
            )

        if self.config.dataset_config_name:
            return load_dataset(
                self.config.dataset_name,
                self.config.dataset_config_name,
                **kwargs,
            )

        return load_dataset(
            self.config.dataset_name,
            **kwargs,
        )

    def _profession_id(
        self,
        raw: Any,
    ) -> tuple[int, str]:
        if isinstance(raw, str):
            normalized = (
                raw.strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            if normalized in PROFESSION_NAME_TO_ID:
                pid = PROFESSION_NAME_TO_ID[
                    normalized
                ]
                return (
                    pid,
                    profession_display(pid),
                )
            try:
                raw = int(raw)
            except ValueError as e:
                raise ValueError(
                    f"Unknown profession label {raw!r}."
                ) from e

        if not isinstance(raw, int):
            try:
                raw = int(raw)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Profession value is not integer/string: {raw!r}"
                ) from e

        base = self.config.profession_index_base

        if base is None:
            if 0 <= raw <= 27:
                base = 0
            elif raw == 28:
                base = 1
            else:
                raise ValueError(
                    f"Profession label {raw!r} is outside 0..27/1..28. "
                    "Set profession_index_base explicitly if needed."
                )

        if base not in {0, 1}:
            raise ValueError(
                "profession_index_base must be 0, 1, or null."
            )

        pid = raw - base

        if pid not in range(28):
            raise ValueError(
                f"Profession label {raw!r} with base={base} "
                "does not map to 0..27."
            )

        return (
            pid,
            profession_display(pid),
        )

    def _gender(
        self,
        raw: Any,
    ) -> Any:
        if raw is None:
            return None

        if self.config.gender_map is None:
            return raw

        key = str(raw)

        if key not in self.config.gender_map:
            raise ValueError(
                f"Gender label {raw!r} missing from configured gender_map."
            )

        value = (
            self.config.gender_map[key]
            .strip()
            .lower()
        )

        if value not in {
            "male",
            "female",
        }:
            raise ValueError(
                "gender_map values must be 'male' or 'female'."
            )

        return value

    def prepare(self) -> PreparedDataset:
        raw_dataset = self._load()
        splits: dict[
            str,
            list[Example],
        ] = {}

        for canonical_split, source_split in (
            self.config.splits.items()
        ):
            if source_split not in raw_dataset:
                raise KeyError(
                    f"Configured split {source_split!r} does not exist in "
                    f"{self.config.dataset_name}. Available: "
                    f"{list(raw_dataset.keys())}"
                )

            examples: list[Example] = []

            for row_index, row in enumerate(
                raw_dataset[
                    source_split
                ]
            ):
                text = row.get(
                    self.config.text_column
                )

                if text is None or not str(text).strip():
                    continue

                pid, profession_name = (
                    self._profession_id(
                        row[
                            self.config.profession_column
                        ]
                    )
                )

                if self.config.id_column:
                    example_id = str(
                        row[
                            self.config.id_column
                        ]
                    )
                else:
                    example_id = (
                        f"{canonical_split}:{row_index:08d}"
                    )

                metadata = {
                    "profession_id": pid,
                    "profession_name": (
                        profession_name
                    ),
                }

                if self.config.gender_column:
                    metadata["gender"] = (
                        self._gender(
                            row.get(
                                self.config.gender_column
                            )
                        )
                    )

                for column in (
                    self.config.preserve_columns
                ):
                    if column in row:
                        metadata[column] = (
                            row[column]
                        )

                examples.append(
                    Example(
                        id=example_id,
                        text=str(text),
                        metadata=metadata,
                        source="human",
                    )
                )

            splits[
                canonical_split
            ] = examples

        dataset = PreparedDataset(
            name="bias_in_bios",
            splits=splits,
        )
        dataset.validate()
        return dataset


def profession_display(
    profession_id: int,
) -> str:
    return professions_display()[profession_id]


def professions_display() -> list[str]:
    return [
        value.replace("_", " ")
        for value in PROFESSIONS_CANON
    ]
