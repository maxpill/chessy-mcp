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
        # Pure awk so this runs inside the Komodo periphery container
        # which has bash + awk but no python3.
        local tmp
        tmp="$(mktemp "${file}.XXXXXX")"
        awk -v k="${key}" -v v="${value}" '
            BEGIN { seen = 0 }
            index($0, k "=") == 1 { print k "=" v; seen = 1; next }
            { print }
            END { if (!seen) print k "=" v }
        ' "${file}" > "${tmp}"
        mv "${tmp}" "${file}"
    else
        printf "%s=%s\n" "${key}" "${value}" >> "${ENVFILE}"
    fi
}

stamp BUILD_SHA "${sha}" "${ENVFILE}"
stamp CHESSY_PACKAGE_VERSION "${ver}" "${ENVFILE}"

echo "deploy-helper: BUILD_SHA=${sha} CHESSY_PACKAGE_VERSION=${ver}"
