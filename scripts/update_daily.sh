#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/update_daily.sh --start-date YYYY-MM-DD --end-date YYYY-MM-DD [--symbols 600000.sh,601398.sh]

Update daily raw data and rebuild feature files.
EOF
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" scripts/update_market_data.py "$@"
"$PYTHON_BIN" scripts/build_features.py
