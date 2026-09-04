"""PGN Date tag validation.

:func:`validate_pgn_date` returns an error message or ``None`.
PGN §7.1 allows ``????` for unknown year and ``??` for unknown month
/ day. Calendar semantics (Apr 31, etc.) only run when all three
components are concrete.
"""

from __future__ import annotations

import datetime as _dt
import re


def validate_pgn_date(date_val: str) -> str | None:
    """Validate a PGN Date tag value. Returns an error message or None.

    Caller is responsible for normalizing sentinel values (empty, ``?`,
    ``????.??.??`) to ``None` before calling.
    """
    parts = date_val.split(".")
    if len(parts) != 3:
        return (
            f"Invalid Date tag '{date_val}': must match YYYY.MM.DD "
            f"(with ? wildcards allowed per component)."
        )

    year_str, month_str, day_str = parts

    if year_str == "????":
        year: int | None = None
    elif re.fullmatch(r"(?:19|20)\d{2}", year_str):
        year = int(year_str)
    else:
        return f"Invalid Date tag '{date_val}': year must be 19xx/20xx or '????' for unknown."

    if month_str == "??":
        month: int | None = None
    elif re.fullmatch(r"(?:0[1-9]|1[0-2])", month_str):
        month = int(month_str)
    else:
        return f"Invalid Date tag '{date_val}': month must be 01-12 or '??' for unknown."

    if day_str == "??":
        day: int | None = None
    elif re.fullmatch(r"(?:0[1-9]|[12]\d|3[01])", day_str):
        day = int(day_str)
    else:
        return f"Invalid Date tag '{date_val}': day must be 01-31 or '??' for unknown."

    if year is not None and month is not None and day is not None:
        try:
            _dt.date(year, month, day)
        except ValueError as exc:
            return f"Impossible Date tag '{date_val}': {exc}."

    return None


# Back-compat shim.
_validate_pgn_date = validate_pgn_date
