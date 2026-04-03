#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/run_backtest.sh --bucket combined --start-date YYYY-MM-DD --end-date YYYY-MM-DD

Run the configured point-in-time backtest.
EOF
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" scripts/backtest.py "$@"
