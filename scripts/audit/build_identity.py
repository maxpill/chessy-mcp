from __future__ import annotations


def validate_call_build_identity(
    build_sha: str,
    pinned_sha: str,
    is_success: bool,
) -> list[str]:
    """R2-006, R2-007: Every success call must expose build_sha matching pinned_sha."""
    errors: list[str] = []
    if is_success:
        if not build_sha:
            errors.append("missing_build_identity: success response does not expose build_sha")
        elif pinned_sha and not build_sha.startswith(pinned_sha):
            errors.append(f"build_sha drift: expected {pinned_sha}, got {build_sha}")
    return errors


def validate_observed_build_shas(
    observed_shas: set[str],
    pinned_sha: str,
) -> list[str]:
    """Final certification gate: exactly one build SHA observed matching pinned SHA."""
    errors: list[str] = []
    if not observed_shas:
        errors.append("certification failure: zero build SHAs observed during stress run")
    elif len(observed_shas) > 1:
        errors.append(
            f"certification failure: multiple build SHAs observed during stress run: {sorted(observed_shas)}"
        )
    elif pinned_sha and not next(iter(observed_shas)).startswith(pinned_sha):
        errors.append(
            f"certification failure: observed build SHA '{next(iter(observed_shas))}' does not match pinned SHA '{pinned_sha}'"
        )
    return errors
