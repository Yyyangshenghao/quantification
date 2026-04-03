#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
./.venv/bin/python scripts/run_demo.py "$@"
