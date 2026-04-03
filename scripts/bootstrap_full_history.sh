#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/bootstrap_full_history.sh --start-date YYYY-MM-DD [--end-date YYYY-MM-DD] [--max-symbols N]

Bootstrap raw history and rebuild features.
EOF
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
END_DATE="${END_DATE:-$(date +%F)}"

"$PYTHON_BIN" scripts/update_market_data.py "$@" --end-date "${END_DATE}"
"$PYTHON_BIN" scripts/build_features.py
