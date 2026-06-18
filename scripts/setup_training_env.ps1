param(
    [ValidateSet("rtx5070", "t4", "cpu")]
    [string]$Profile = "rtx5070"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path "venv")) {
    python -m venv venv
}

$Python = Join-Path $PWD "venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip

switch ($Profile) {
    "rtx5070" { $TorchIndex = "https://download.pytorch.org/whl/cu130" }
    "t4" { $TorchIndex = "https://download.pytorch.org/whl/cu128" }
    "cpu" { $TorchIndex = "https://download.pytorch.org/whl/cpu" }
}

Write-Host "Installing PyTorch from $TorchIndex"
& $Python -m pip install torch torchvision --index-url $TorchIndex
& $Python -m pip install -e ".[ml,dev]"

& $Python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda:', torch.version.cuda); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
