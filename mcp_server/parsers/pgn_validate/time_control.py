"""PGN TimeControl tag grammar validation.

:func:`is_valid_pgn_time_control` validates the PGN TimeControl tag
grammar (sudden-death ``300`, moves/seconds ``40/7200`, Fischer
``300+5`, hourglass ``*60`). :func:`stage_has_positive_number`
rejects semantically impossible values flagged by the 2026-09-02 audit
(``"0+0"`, ``"0+1"`, ``"0/600"`, etc.) while keeping the legitimate
ones intact.
"""

from __future__ import annotations

import re
from typing import Final


TIME_CONTROL_STAGE_RE: Final[re.Pattern[str]] = re.compile(r"^(?:\d+|\d+/\d+|\d+\+\d+|\*\d+)$")
_TIME_CONTROL_STAGE_RE = TIME_CONTROL_STAGE_RE


def stage_has_positive_number(stage: str) -> bool:
    """Return True iff every numeric half of ``stage` is valid.

    For Fischer increment (``+``), the base time must contain a non-zero
    digit, while the increment is allowed to be zero (e.g. ``600+0``).
    For moves/seconds (``/``) and hourglass (``*``), every numeric
    component must contain a non-zero digit.
    """
    body = stage[1:] if stage.startswith("*") else stage
    if "+" in body:
        pieces = body.split("+")
        if len(pieces) != 2:
            return False
        base, inc = pieces
        if not base or not any(c in "123456789" for c in base):
            return False
        if not inc or not inc.isdigit():
            return False
        return True
    elif "/" in body:
        pieces = body.split("/")
        if len(pieces) != 2:
            return False
        return all(bool(p) and any(c in "123456789" for c in p) for p in pieces)
    else:
        return bool(body) and any(c in "123456789" for c in body)



def is_valid_pgn_time_control(value: str) -> bool:
    """Validate the PGN TimeControl tag grammar.

    PGN permits a single stage or colon-separated stages. A stage is one of:
    sudden-death seconds (``300`), moves/seconds (``40/7200`), Fischer
    seconds+increment (``300+5`), or hourglass (``*60`). ``?` and ``-`
    are the standard unknown/unspecified markers.

    Every numeric component must contain at least one non-zero digit.
    """
    text = value.strip()
    if text in {"?", "-"}:
        return True
    if not text:
        return False
    stages = text.split(":")
    return all(
        bool(stage)
        and _TIME_CONTROL_STAGE_RE.fullmatch(stage) is not None
        and stage_has_positive_number(stage)
        for stage in stages
    )


# Back-compat shims.
_is_valid_pgn_time_control = is_valid_pgn_time_control
_stage_has_positive_number = stage_has_positive_number
