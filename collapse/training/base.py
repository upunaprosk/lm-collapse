from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from collapse.records import Example
from collapse.training.budget import TrainingBudget
from collapse.training.llamafactory import (
    LlamaFactoryConfig,
    TrainingRun,
    train_with_llamafactory,
)


@dataclass(frozen=True)
class TrainingBackend:
    config: LlamaFactoryConfig
    extra_overrides: Mapping[str, Any] = field(default_factory=dict)

    def train(
        self,
        *,
        examples: Sequence[Example],
        model_name_or_path: str,
        run_dir: str | Path,
        run_name: str,
        budget: TrainingBudget,
        training_seed: int,
        dry_run: bool = False,
    ) -> TrainingRun:
        if not examples:
            raise ValueError(
                "TrainingBackend.train() received an empty dataset."
            )

        if not str(model_name_or_path).strip():
            raise ValueError(
                "model_name_or_path must be non-empty."
            )

        if not str(run_name).strip():
            raise ValueError(
                "run_name must be non-empty."
            )

        return train_with_llamafactory(
            examples=examples,
            model_name_or_path=str(model_name_or_path),
            run_dir=run_dir,
            run_name=run_name,
            budget=budget,
            training_seed=int(training_seed),
            backend_config=self.config,
            extra_overrides=dict(self.extra_overrides),
            dry_run=bool(dry_run),
        )

    def dry_run(
        self,
        *,
        examples: Sequence[Example],
        model_name_or_path: str,
        run_dir: str | Path,
        run_name: str,
        budget: TrainingBudget,
        training_seed: int,
    ) -> TrainingRun:
        return self.train(
            examples=examples,
            model_name_or_path=model_name_or_path,
            run_dir=run_dir,
            run_name=run_name,
            budget=budget,
            training_seed=training_seed,
            dry_run=True,
        )
