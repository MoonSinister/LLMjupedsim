param(
    [string]$Path = "",
    [int]$MaxAgents = 300,
    [int]$FrameStride = 10,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Resolve-JuPedSimPython {
    $candidates = @()
    if ($env:JUPEDSIM_PYTHON) {
        $candidates += $env:JUPEDSIM_PYTHON
    }
    $candidates += Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    $candidates += "D:\anaconda\envs\python3.12\python.exe"
    $candidates += "python"

    foreach ($candidate in $candidates) {
        if ($candidate -eq "python" -or (Test-Path $candidate)) {
            try {
                & $candidate --version *> $null
                if ($LASTEXITCODE -eq 0) {
                    return $candidate
                }
            } catch {
                continue
            }
        }
    }

    throw "No working Python interpreter found. Set JUPEDSIM_PYTHON to a Python 3.12 executable."
}

$PythonExe = Resolve-JuPedSimPython
$env:JUPEDSIM_PYTHON = $PythonExe
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

$arguments = @(
    "src\demo_3d_visualize.py",
    "--max-agents", $MaxAgents,
    "--frame-stride", $FrameStride
)
if ($Path) {
    $arguments += $Path
}
if (-not $NoOpen) {
    $arguments += "--open"
}

Push-Location $ProjectRoot
try {
    & $PythonExe @arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
