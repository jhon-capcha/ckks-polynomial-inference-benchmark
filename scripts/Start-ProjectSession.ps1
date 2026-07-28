# =========================================================
# Inicia una sesión de PowerShell registrada para el
# proyecto CKKS.
# =========================================================

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = "D:\Proyectos\CKKS"
$logDirectory = Join-Path $projectRoot "logs\powershell"

# Crear la carpeta de logs si no existe.
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

# Nombre único para el transcript.
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$computerName = $env:COMPUTERNAME
$userName = $env:USERNAME -replace "\s+", "_"

$logFile = Join-Path `
    $logDirectory `
    "CKKS_${computerName}_${userName}_$timestamp.txt"

# Situarse siempre en la raíz del proyecto.
Set-Location $projectRoot

# Mostrar fecha y hora en cada prompt.
function global:prompt {
    $dateTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $currentPath = (Get-Location).Path

    "[$dateTime] PS $currentPath> "
}

# Iniciar el transcript.
Start-Transcript `
    -Path $logFile `
    -Force

Write-Host ""
Write-Host "Sesión CKKS registrada." -ForegroundColor Green
Write-Host "Archivo de log: $logFile" -ForegroundColor Cyan
Write-Host ""

# Activar el entorno virtual si existe.
$activateScript = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"

if (Test-Path -LiteralPath $activateScript) {
    & $activateScript
    Write-Host "Entorno virtual activado: .venv" -ForegroundColor Green
}
else {
    Write-Warning "No se encontró el entorno virtual .venv."
}

Write-Host ""
Write-Host "Para finalizar el registro, ejecuta: Stop-Transcript" -ForegroundColor Yellow
Write-Host ""
