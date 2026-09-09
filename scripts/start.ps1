param(
    [string]$Scenario = "baseline_random",
    [int]$Agents = 0,
    [switch]$DryRun,
    [switch]$SkipDoctor,
    [switch]$List,
    [switch]$Pause
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

Push-Location $ProjectRoot
try {
    Write-Host "JuPedSim Mall launcher"
    Write-Host "Project: $ProjectRoot"
    Write-Host "Python : $PythonExe"
    & $PythonExe --version

    if ($List) {
        & $PythonExe -m jupedsim_mall scenarios list
        exit $LASTEXITCODE
    }

    if (-not $SkipDoctor) {
        Write-Host ""
        Write-Host "Running environment check..."
        & $PythonExe -m jupedsim_mall doctor
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }

    $arguments = @("-m", "jupedsim_mall", "run")
    if ($DryRun) {
        $arguments += "--dry-run"
    }
    if ($Agents -gt 0) {
        $arguments += @("--num-agents", $Agents)
    }
    $arguments += $Scenario

    Write-Host ""
    Write-Host "Starting scenario: $Scenario"
    if ($Agents -gt 0) {
        Write-Host "Agent override: $Agents"
    }
    & $PythonExe @arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
    if ($Pause) {
        Write-Host ""
        Read-Host "Press Enter to close"
    }
}
