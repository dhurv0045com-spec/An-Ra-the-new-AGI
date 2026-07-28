param(
    [string]$Checkpoint = "",
    [int]$Port = 8000,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv-cuda\Scripts\python.exe"
$stateDir = Join-Path $repoRoot "state"
$pidPath = Join-Path $stateDir "local_chat.pid"
$stdoutPath = Join-Path $stateDir "local_chat.stdout.log"
$stderrPath = Join-Path $stateDir "local_chat.stderr.log"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "CUDA Python environment is missing: $python"
}

if (-not $Checkpoint) {
    $candidates = @(
        (Join-Path $repoRoot "anra_v4_180m.pt"),
        (Join-Path $repoRoot "output\v2\checkpoints\anra_v4_same_commit_interrupt_part2.pt"),
        (Join-Path $repoRoot "output\v2\checkpoints\anra_v4_180m_rehearsal_corrected_part2.pt")
    )
    $Checkpoint = $candidates | Where-Object {
        Test-Path -LiteralPath $_ -PathType Leaf
    } | Select-Object -First 1
}

if (-not $Checkpoint) {
    throw "No compatible V4 checkpoint was found. Pass -Checkpoint with an explicit path."
}
$Checkpoint = (Resolve-Path -LiteralPath $Checkpoint).Path

New-Item -ItemType Directory -Path $stateDir -Force | Out-Null

if (Test-Path -LiteralPath $pidPath) {
    $existingPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $existing = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "An-Ra local chat is already running (PID $existingPid)."
        Write-Host "Open http://127.0.0.1:$Port/developer"
        if (-not $NoBrowser) {
            Start-Process "http://127.0.0.1:$Port/developer"
        }
        exit 0
    }
    Remove-Item -LiteralPath $pidPath -Force
}

$cuda = & $python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
if ($LASTEXITCODE -ne 0) {
    throw "The local Python environment could not import PyTorch."
}

$env:ANRA_CHECKPOINT_PATH = $Checkpoint
$env:ANRA_MODEL_PROFILE = "frontier"

Write-Host "Starting An-Ra local chat..."
Write-Host "Checkpoint: $Checkpoint"
Write-Host "Compute: $cuda"
Write-Host "URL: http://127.0.0.1:$Port/developer"

$arguments = @(
    "-m", "uvicorn", "app:app",
    "--host", "127.0.0.1",
    "--port", "$Port"
)
$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru
Set-Content -LiteralPath $pidPath -Value $process.Id -Encoding ascii

$healthUrl = "http://127.0.0.1:$Port/health"
$deadline = (Get-Date).AddSeconds(120)
$health = $null
while ((Get-Date) -lt $deadline) {
    if ($process.HasExited) {
        $errorText = if (Test-Path $stderrPath) {
            Get-Content -LiteralPath $stderrPath -Raw
        } else {
            "No server error log was produced."
        }
        Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
        throw "An-Ra server exited during startup.`n$errorText"
    }
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 5
        if ($health.service_status -ne "loading") {
            break
        }
    } catch {
        Start-Sleep -Milliseconds 750
    }
}

if (-not $health) {
    throw "An-Ra did not become reachable within 120 seconds. See $stderrPath"
}
if ($health.service_status -eq "failed") {
    throw "An-Ra model loading failed: $($health.model_error)"
}

Write-Host "Ready: $($health.profile), $($health.param_count) parameters, $($health.device)"
Write-Host "Note: this local fallback checkpoint has only three optimizer steps."
if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:$Port/developer"
}

