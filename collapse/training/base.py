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
    """
    Training backend used by the experiment runner.
    """

    config: LlamaFactoryConfig

    extra_overrides: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.config.base_yaml_path:
            raise ValueError(
                "TrainingBackend requires a non-empty "
                "LlamaFactoryConfig.base_yaml_path."
            )

        if not self.config.executable:
            raise ValueError(
                "TrainingBackend requires a non-empty LLaMA-Factory executable."
            )

    def train(
        self,
        *,
        examples: Sequence[Example],
        init_model: str,
        run_dir: str | Path,
        run_name: str,
        budget: TrainingBudget,
        training_seed: int,
        dry_run: bool = False,
    ) -> TrainingRun:
        """
        Train one model checkpoint.
        """

        if not examples:
            raise ValueError(
                "TrainingBackend.train() received an empty dataset."
            )

        if not str(init_model).strip():
            raise ValueError(
                "TrainingBackend.train() requires an explicit init_model."
            )

        if not str(run_name).strip():
            raise ValueError(
                "TrainingBackend.train() requires a non-empty run_name."
            )

        return train_with_llamafactory(
            examples=examples,
            init_model=str(init_model),
            run_dir=run_dir,
            run_name=run_name,
            budget=budget,
            training_seed=int(training_seed),
            backend_config=self.config,
            extra_overrides=dict(
                self.extra_overrides
            ),
            dry_run=bool(dry_run),
        )

    def dry_run(
        self,
        *,
        examples: Sequence[Example],
        init_model: str,
        run_dir: str | Path,
        run_name: str,
        budget: TrainingBudget,
        training_seed: int,
    ) -> TrainingRun:
        return self.train(
            examples=examples,
            init_model=init_model,
            run_dir=run_dir,
            run_name=run_name,
            budget=budget,
            training_seed=training_seed,
            dry_run=True,
        )
