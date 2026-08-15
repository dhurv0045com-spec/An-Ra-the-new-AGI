$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pidPath = Join-Path $repoRoot "state\local_chat.pid"

if (-not (Test-Path -LiteralPath $pidPath -PathType Leaf)) {
    Write-Host "An-Ra local chat is not running."
    exit 0
}

$serverPid = [int](Get-Content -LiteralPath $pidPath -Raw)
$process = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
if ($process) {
    function Stop-ProcessTree([int]$ParentPid) {
        $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $ParentPid" `
            -ErrorAction SilentlyContinue
        foreach ($child in $children) {
            Stop-ProcessTree -ParentPid ([int]$child.ProcessId)
        }
        Stop-Process -Id $ParentPid -Force -ErrorAction SilentlyContinue
    }
    Stop-ProcessTree -ParentPid $serverPid
    Write-Host "Stopped An-Ra local chat (PID $serverPid)."
} else {
    Write-Host "Removed stale local-chat PID $serverPid."
}
Remove-Item -LiteralPath $pidPath -Force
