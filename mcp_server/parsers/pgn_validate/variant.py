"""Variant validation.

:func:\`validate_variant\` rejects variants beyond standard chess.
:data:\`SUPPORTED_VARIANTS\` lists the accepted variant tags.
"""

from __future__ import annotations

from typing import Final


SUPPORTED_VARIANTS: Final[frozenset[str | None]] = frozenset(
    {None, "", "standard", "from position"}
)


def validate_variant(variant: str | None) -> None:
    if variant is not None and variant.strip().lower() not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"UNSUPPORTED_VARIANT: Variant '{variant.strip()}' is not supported. Chess MCP currently analyzes standard chess only."
        )


# Back-compat shim.
_validate_variant = validate_variant
