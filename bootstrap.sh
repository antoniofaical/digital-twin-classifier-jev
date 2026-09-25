#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python_bin="${BOOTSTRAP_PYTHON:-python3}"
if ! command -v git >/dev/null 2>&1; then
    printf 'Git is required.\n' >&2
    exit 1
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'Python is required: %s\n' "$python_bin" >&2
    exit 1
fi
if ! "$python_bin" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    printf 'Python 3.11 or newer is required; set BOOTSTRAP_PYTHON if necessary.\n' >&2
    exit 1
fi

printf '==> Switching to main\n'
git fetch origin main
git switch main
printf '==> Pulling main\n'
git pull --ff-only origin main

if [[ ! -d src/startup_adherence ]]; then
    git status --short -- src
    printf 'src/startup_adherence is missing. Inspect local changes before continuing.\n' >&2
    exit 1
fi

if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    printf '==> Creating or repairing virtual environment\n'
    if ! "$python_bin" -m venv .venv; then
        python_version="$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        printf 'Could not create .venv. On Ubuntu, run: apt update && apt install python%s-venv\n' "$python_version" >&2
        printf 'Then rerun this script; it will repair the incomplete .venv.\n' >&2
        exit 1
    fi
fi
if ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    printf 'Existing .venv uses Python older than 3.11; recreate it with a supported version.\n' >&2
    exit 1
fi

printf '==> Installing project and development dependencies\n'
.venv/bin/python -m pip install -e '.[dev]'
printf '==> Running pytest\n'
.venv/bin/python -m pytest -q
printf '==> Running Ruff checks\n'
.venv/bin/python -m ruff check src tests orchestrator.py fit.py
printf '==> Checking Ruff formatting\n'
.venv/bin/python -m ruff format --check src tests orchestrator.py fit.py

printf '\nBootstrap complete. Activate the environment with: source .venv/bin/activate\n'
