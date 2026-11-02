#!/usr/bin/env bash
set -euo pipefail

# Run inside the provider's configured PyTorch image. Do not replace DTK with
# generic CUDA wheels. This script never provisions resources or incurs a charge.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd -- "$SCRIPT_DIR/.."
PYTHON_BIN=${RPT_PYTHON:-python3}
exec "$PYTHON_BIN" -m validation.run_validation "$@"
