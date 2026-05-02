param(
    [int]$Port = 8001,
    [string]$BindHost = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

function Get-PortOwningProcessId {
    param([int]$Port)

    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $conn) {
        return $null
    }

    return $conn.OwningProcess
}

$owningPid = Get-PortOwningProcessId -Port $Port
if ($null -ne $owningPid -and $owningPid -ne 0) {
    try {
        $p = Get-Process -Id $owningPid -ErrorAction Stop
        Write-Host "Port $Port is in use by PID $owningPid ($($p.ProcessName)). Stopping it..."
        Stop-Process -Id $owningPid -Force
        Start-Sleep -Milliseconds 400
    } catch {
        Write-Host "Port $Port is in use by PID $owningPid. Attempting to stop it..."
        Stop-Process -Id $owningPid -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 400
    }
}

Write-Host "Starting uvicorn on http://$BindHost`:$Port ..."
& .\venv\Scripts\python.exe -m uvicorn dashboard:app --host $BindHost --port $Port
