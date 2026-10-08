from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Example:
    id: str
    text: str

    metadata: dict[str, Any] = field(default_factory=dict)

    prefix_text: str | None = None
    human_suffix: str | None = None
    synthetic_suffix: str | None = None

    source: str = "human"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Example":
        return cls(
            id=str(value["id"]),
            text=str(value["text"]),
            metadata=dict(value.get("metadata") or {}),
            prefix_text=value.get("prefix_text"),
            human_suffix=value.get("human_suffix"),
            synthetic_suffix=value.get("synthetic_suffix"),
            source=str(value.get("source", "human")),
        )
