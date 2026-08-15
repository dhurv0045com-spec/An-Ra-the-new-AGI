param(
    [string]$Checkpoint = "",
    [int]$Port = 8010,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv-cuda\Scripts\python.exe"
$launcher = Join-Path $repoRoot "scripts\launch_sft_prototype.py"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "CUDA Python environment is missing: $python"
}

$arguments = @($launcher, "--port", "$Port")
if ($Checkpoint) {
    $arguments += @("--checkpoint", $Checkpoint)
}
if ($NoBrowser) {
    $arguments += "--no-browser"
}

& $python @arguments
exit $LASTEXITCODE
