param(
    [string]$Path = "",
    [switch]$PrintOnly
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

function Find-LatestTrajectory {
    $patterns = @(
        "outputs\runs\*\trajectory.sqlite",
        "outputs\trajectories\*.sqlite",
        "outputs\smoke\*.sqlite"
    )
    $files = foreach ($pattern in $patterns) {
        Get-ChildItem -Path (Join-Path $ProjectRoot $pattern) -File -ErrorAction SilentlyContinue
    }
    $latest = $files | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) {
        throw "No trajectory SQLite file found. Run start.bat first to generate one."
    }
    return $latest.FullName
}

$PythonExe = Resolve-JuPedSimPython
$env:JUPEDSIM_PYTHON = $PythonExe
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

$TrajectoryPath = if ($Path) { $Path } else { Find-LatestTrajectory }
if (-not [System.IO.Path]::IsPathRooted($TrajectoryPath)) {
    $TrajectoryPath = Join-Path $ProjectRoot $TrajectoryPath
}
if (-not (Test-Path $TrajectoryPath)) {
    throw "Trajectory file not found: $TrajectoryPath"
}

Write-Host "Trajectory: $TrajectoryPath"
if ($PrintOnly) {
    exit 0
}

Push-Location $ProjectRoot
try {
    & $PythonExe "src\demo_visualize.py" $TrajectoryPath
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
