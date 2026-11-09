#!/usr/bin/env bash
# scripts/install_hooks.sh
# Installs BenoStreamDB git hooks into .git/hooks/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOKS_DIR="${REPO_ROOT}/.git/hooks"

if [[ ! -d "${HOOKS_DIR}" ]]; then
    echo "Error: ${HOOKS_DIR} not found (not a git repository checkout)."
    exit 1
fi

echo "Installing git hooks into ${HOOKS_DIR}..."
cp "${REPO_ROOT}/scripts/git-hooks/pre-push" "${HOOKS_DIR}/pre-push"
chmod +x "${HOOKS_DIR}/pre-push"

echo "✅ Git pre-push hook successfully installed."
