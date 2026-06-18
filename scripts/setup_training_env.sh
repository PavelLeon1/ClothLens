#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-t4}"

if [[ ! -d "venv" ]]; then
  python3 -m venv venv
fi

PYTHON="venv/bin/python"
"$PYTHON" -m pip install --upgrade pip

case "$PROFILE" in
  rtx5070)
    TORCH_INDEX="https://download.pytorch.org/whl/cu130"
    ;;
  t4)
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    ;;
  cpu)
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    ;;
  *)
    echo "Unknown profile: $PROFILE"
    echo "Use one of: rtx5070, t4, cpu"
    exit 2
    ;;
esac

echo "Installing PyTorch from $TORCH_INDEX"
"$PYTHON" -m pip install torch torchvision --index-url "$TORCH_INDEX"
"$PYTHON" -m pip install -e ".[ml,dev]"

"$PYTHON" - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
