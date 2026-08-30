#!/usr/bin/env bash
# Stamps the current HEAD and pyproject version into the stack .env so the
# compose `up` step picks them up via `${BUILD_SHA:-unknown}` /
# `${CHESSY_PACKAGE_VERSION:-0.1.0}`. Idempotent: only rewrites the file when
# the values actually change, to avoid touching unrelated lines and keep the
# existing CHESS_MCP_AUTH_TOKEN untouched.
#
# Used by Komodo as the chessy-mcp stack `pre_deploy` step. Also safe to run
# manually before `docker compose up -d --build`.

set -euo pipefail

ROOT="${STACK_ROOT:-/etc/komodo/stacks/chessy-mcp}"
PYPROJECT="${ROOT}/pyproject.toml"
ENVFILE="${ROOT}/.env"

if [[ ! -d "${ROOT}/.git" ]]; then
    echo "deploy-helper: ${ROOT} is not a git checkout; skipping" >&2
    exit 0
fi

sha="$(git -C "${ROOT}" rev-parse HEAD)"
ver="$(awk -F'"' '/^version = / {print $2; exit}' "${PYPROJECT}")"

touch "${ENVFILE}"
chmod 600 "${ENVFILE}" || true

stamp() {
    local key="$1" value="$2" file="$3"
    if grep -qE "^${key}=" "${file}"; then
        # Update in place — preserve order, only mutate the value.
        python3 - "$key" "$value" "${file}" <<'PY'
import sys, pathlib
key, value, path = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path(path)
lines = p.read_text().splitlines()
out = []
seen = False
for line in lines:
    if line.startswith(f"{key}="):
        out.append(f"{key}={value}")
        seen = True
    else:
        out.append(line)
if not seen:
    out.append(f"{key}={value}")
p.write_text("\n".join(out) + ("\n" if out else ""))
PY
    else
        printf "%s=%s\n" "${key}" "${value}" >> "${ENVFILE}"
    fi
}

stamp BUILD_SHA "${sha}" "${ENVFILE}"
stamp CHESSY_PACKAGE_VERSION "${ver}" "${ENVFILE}"

echo "deploy-helper: BUILD_SHA=${sha} CHESSY_PACKAGE_VERSION=${ver}"
