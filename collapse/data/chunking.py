from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from transformers import PreTrainedTokenizerBase

from collapse.records import Example


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_tokens: int = 512
    min_chunk_tokens: int = 128
    drop_short_final_chunk: bool = True


def validate_chunking_config(
    config: ChunkingConfig,
) -> None:
    if config.chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be > 0.")

    if config.min_chunk_tokens <= 0:
        raise ValueError("min_chunk_tokens must be > 0.")

    if config.min_chunk_tokens > config.chunk_tokens:
        raise ValueError(
            "min_chunk_tokens must be <= chunk_tokens."
        )


def chunk_text_by_reference_tokens(
    text: str,
    *,
    tokenizer: PreTrainedTokenizerBase,
    config: ChunkingConfig,
) -> list[str]:
    """
    Split one document into raw-text chunks.
    """
    validate_chunking_config(config)

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "Chunking requires a fast tokenizer with offset mappings."
        )

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )

    offsets = encoded["offset_mapping"]

    if not offsets:
        return []

    chunks: list[str] = []
    token_start = 0
    char_start = 0
    n_tokens = len(offsets)

    while token_start < n_tokens:
        token_end = min(
            token_start + config.chunk_tokens,
            n_tokens,
        )

        n_this = token_end - token_start

        if (
            token_end == n_tokens
            and n_this < config.min_chunk_tokens
            and config.drop_short_final_chunk
        ):
            break

        if token_end == n_tokens:
            char_end = len(text)
        else:
            char_end = int(
                offsets[token_end - 1][1]
            )

        chunk = text[
            char_start:char_end
        ]

        if chunk.strip():
            chunks.append(chunk)

        char_start = char_end
        token_start = token_end

    return chunks


def chunk_examples(
    examples: Sequence[Example],
    *,
    tokenizer: PreTrainedTokenizerBase,
    config: ChunkingConfig,
) -> list[Example]:
    output: list[Example] = []

    for example in examples:
        chunks = chunk_text_by_reference_tokens(
            example.text,
            tokenizer=tokenizer,
            config=config,
        )

        for chunk_index, chunk in enumerate(chunks):
            metadata = dict(example.metadata)

            metadata.update(
                {
                    "parent_id": example.id,
                    "chunk_index": chunk_index,
                    "chunk_tokens_target": (
                        config.chunk_tokens
                    ),
                }
            )

            output.append(
                Example(
                    id=(
                        f"{example.id}::chunk_{chunk_index:05d}"
                    ),
                    text=chunk,
                    metadata=metadata,
                    source="human",
                )
            )

    return output
