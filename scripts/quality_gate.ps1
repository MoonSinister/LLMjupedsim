param(
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
$python = if ($env:JUPEDSIM_PYTHON) { $env:JUPEDSIM_PYTHON } else { "python" }
$arguments = @("-m", "jupedsim_mall", "quality-gate")
if ($Quick) {
    $arguments += "--quick"
}
$env:PYTHONPATH = "src"
& $python @arguments
exit $LASTEXITCODE
