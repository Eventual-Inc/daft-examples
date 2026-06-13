#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "python" ]]; then
  shift
fi

exec python "$@"
