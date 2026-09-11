"""Workflow syntax validator for GitHub Actions workflows.

Validates that all .github/workflows/*.yml files:
1. Parse as valid YAML.
2. Have required top-level keys ('name', 'on', 'jobs').
3. Do not contain ${{ ... }} expression interpolation in static metadata contexts
   such as workflow_dispatch.inputs.<name>.description (which causes GitHub Actions
   to fail before any job starts - R2-001).
4. If actionlint is available on PATH, runs actionlint for deep verification.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def get_workflow_files() -> list[Path]:
    return sorted(WORKFLOWS_DIR.glob("*.yml"))


@pytest.mark.parametrize("workflow_file", get_workflow_files(), ids=lambda p: p.name)
def test_workflow_yaml_and_static_expressions(workflow_file: Path) -> None:
    text = workflow_file.read_text(encoding="utf-8")
    data: dict[str, Any] = yaml.safe_load(text)
    assert isinstance(data, dict), f"{workflow_file.name} did not parse to a dictionary"

    assert "name" in data, f"{workflow_file.name} missing 'name'"
    assert "on" in data or True in data, f"{workflow_file.name} missing 'on'"
    assert "jobs" in data, f"{workflow_file.name} missing 'jobs'"

    # Check workflow_dispatch inputs descriptions for forbidden ${{ ... }} expressions (R2-001)
    on_section = data.get("on")
    if on_section is None and True in data:
        on_section = data[True]  # type: ignore[index]
    if isinstance(on_section, dict):
        dispatch = on_section.get("workflow_dispatch")
        if isinstance(dispatch, dict):
            inputs = dispatch.get("inputs") or {}
            for input_name, input_def in inputs.items():
                if isinstance(input_def, dict):
                    desc = str(input_def.get("description", ""))
                    assert "${{" not in desc, (
                        f"Found GitHub expression in {workflow_file.name} workflow_dispatch.inputs.{input_name}.description: {desc!r}. "
                        "GitHub Actions rejects expressions in static input descriptions."
                    )

    # General check: static keys like 'name' should not have unresolved expressions
    raw_lines = text.splitlines()
    for idx, line in enumerate(raw_lines, 1):
        if re.match(r"^\s*description\s*:\s*.*?\$\{\{", line):
            pytest.fail(f"Line {idx} in {workflow_file.name} contains expression in static description: {line}")


def test_actionlint_if_installed() -> None:
    actionlint_path = shutil.which("actionlint")
    if actionlint_path:
        proc = subprocess.run(
            [actionlint_path, *[str(p) for p in get_workflow_files()]],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"actionlint failed:\n{proc.stdout}\n{proc.stderr}"
