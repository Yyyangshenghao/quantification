#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/run_daily.sh --start-date YYYY-MM-DD --end-date YYYY-MM-DD [--symbols 600000.sh,601398.sh]

Sequence:
1. 更新日线与财务
2. 生成特征
3. 生成 snapshot
4. 可选调用 Codex CLI 日常决策 prompt
5. 写出 reports/daily/latest.md 与 latest.json
EOF
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
CODEX_BIN="${CODEX_CLI_BIN:-codex}"
CODEX_ENABLED="${CODEX_DECISION_ENABLED:-0}"
SNAPSHOT_JSON="${SNAPSHOT_JSON:-data/snapshots/latest.json}"
CODEX_NOTE_FILE="${CODEX_NOTE_FILE:-reports/daily/codex_decision.md}"

"$PYTHON_BIN" scripts/update_market_data.py "$@"
"$PYTHON_BIN" scripts/build_features.py
"$PYTHON_BIN" scripts/prepare_snapshot.py

if [[ "$CODEX_ENABLED" == "1" ]]; then
  if ! command -v "$CODEX_BIN" >/dev/null 2>&1; then
    echo "Codex CLI not found: $CODEX_BIN" >&2
    exit 1
  fi
  "$CODEX_BIN" exec "阅读 prompts/daily_decision.md 与 ${SNAPSHOT_JSON}，生成今日研究结论。" > "$CODEX_NOTE_FILE"
fi

"$PYTHON_BIN" scripts/render_report.py --snapshot-json "$SNAPSHOT_JSON" --codex-note-file "$CODEX_NOTE_FILE"
