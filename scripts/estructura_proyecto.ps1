# =========================================================
# Estructura base del proyecto CKKS
#
# Ejecutar desde:
# D:\Proyectos\CKKS
#
# Características:
# - Idempotente: puede ejecutarse más de una vez.
# - No sobrescribe archivos existentes.
# - No crea todavía el entorno virtual.
# - No crea todavía GitHub Actions.
# =========================================================

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function New-FileIfMissing {
    param(
        [Parameter(Mandatory)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType File -Path $Path | Out-Null
        Write-Host "Archivo creado: $Path"
    }
    else {
        Write-Host "Archivo existente, se conserva: $Path" -ForegroundColor DarkGray
    }
}

# ---------------------------------------------------------
# Validar directorio de trabajo
# ---------------------------------------------------------

$expectedRoot = "D:\Proyectos\CKKS"
$currentRoot = (Get-Location).Path.TrimEnd("\")

if ($currentRoot -ne $expectedRoot) {
    throw "Ejecuta este script desde $expectedRoot. Ruta actual: $currentRoot"
}

# ---------------------------------------------------------
# Carpetas
# ---------------------------------------------------------

$directories = @(
    ".vscode",
    "configs",

    "src/ckks_benchmark",
    "src/ckks_benchmark/model",
    "src/ckks_benchmark/approximation",
    "src/ckks_benchmark/homomorphic",
    "src/ckks_benchmark/experiment",

    "notebooks",
    "data",
    "models",

    "results/raw",
    "results/processed",
    "results/figures",
    "results/manifests",

    "tests",

    "docs/articulo",
    "docs/figuras",
    "docs/metodologia",
    "docs/instalacion"
)

foreach ($directory in $directories) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}

# ---------------------------------------------------------
# Archivos __init__.py de los paquetes Python
# ---------------------------------------------------------

$packageInitializers = @(
    "src/ckks_benchmark/__init__.py",
    "src/ckks_benchmark/model/__init__.py",
    "src/ckks_benchmark/approximation/__init__.py",
    "src/ckks_benchmark/homomorphic/__init__.py",
    "src/ckks_benchmark/experiment/__init__.py"
)

foreach ($file in $packageInitializers) {
    New-FileIfMissing -Path $file
}

# ---------------------------------------------------------
# Configuración interna del paquete
#
# Los parámetros científicos estarán en configs/*.yaml.
# config.py contendrá únicamente rutas, tipos y constantes
# internas necesarias para el código.
# ---------------------------------------------------------

New-FileIfMissing -Path "src/ckks_benchmark/config.py"

# ---------------------------------------------------------
# Módulos del modelo
# ---------------------------------------------------------

$modelModules = @(
    "src/ckks_benchmark/model/architecture.py",
    "src/ckks_benchmark/model/train.py",
    "src/ckks_benchmark/model/preactivations.py"
)

foreach ($file in $modelModules) {
    New-FileIfMissing -Path $file
}

# ---------------------------------------------------------
# Módulos de aproximación polinómica
# ---------------------------------------------------------

$approximationModules = @(
    "src/ckks_benchmark/approximation/base.py",
    "src/ckks_benchmark/approximation/taylor.py",
    "src/ckks_benchmark/approximation/chebyshev.py",
    "src/ckks_benchmark/approximation/least_squares.py"
)

foreach ($file in $approximationModules) {
    New-FileIfMissing -Path $file
}

# ---------------------------------------------------------
# Módulos CKKS
# ---------------------------------------------------------

$homomorphicModules = @(
    "src/ckks_benchmark/homomorphic/context.py",
    "src/ckks_benchmark/homomorphic/evaluator.py",
    "src/ckks_benchmark/homomorphic/instrumentation.py",
    "src/ckks_benchmark/homomorphic/depth.py"
)

foreach ($file in $homomorphicModules) {
    New-FileIfMissing -Path $file
}

# ---------------------------------------------------------
# Módulos de experimentación
# ---------------------------------------------------------

$experimentModules = @(
    "src/ckks_benchmark/experiment/runner.py",
    "src/ckks_benchmark/experiment/metrics.py",
    "src/ckks_benchmark/experiment/pareto.py"
)

foreach ($file in $experimentModules) {
    New-FileIfMissing -Path $file
}

# ---------------------------------------------------------
# Pruebas
# ---------------------------------------------------------

$testFiles = @(
    "tests/test_approximation.py",
    "tests/test_homomorphic.py",
    "tests/test_depth.py"
)

foreach ($file in $testFiles) {
    New-FileIfMissing -Path $file
}

# ---------------------------------------------------------
# Archivos .gitkeep
#
# Se utilizan únicamente en directorios que estarán vacíos
# durante las primeras fases.
# ---------------------------------------------------------

$gitkeepFiles = @(
    "notebooks/.gitkeep",
    "data/.gitkeep",
    "models/.gitkeep",
    "results/raw/.gitkeep",
    "results/processed/.gitkeep",
    "results/figures/.gitkeep",
    "results/manifests/.gitkeep",
    "docs/articulo/.gitkeep",
    "docs/figuras/.gitkeep"
)

foreach ($file in $gitkeepFiles) {
    New-FileIfMissing -Path $file
}

# ---------------------------------------------------------
# Archivos raíz
# ---------------------------------------------------------

$rootFiles = @(
    "README.md",
    "README-BRIEFING.md",
    "LICENSE",
    ".gitignore",
    "requirements.txt",
    "pyproject.toml"
)

foreach ($file in $rootFiles) {
    New-FileIfMissing -Path $file
}

# ---------------------------------------------------------
# Configuraciones YAML
#
# Se completarán después de validar el entorno y definir
# formalmente los parámetros experimentales y CKKS.
# ---------------------------------------------------------

$configFiles = @(
    "configs/experiment.yaml",
    "configs/ckks.yaml"
)

foreach ($file in $configFiles) {
    New-FileIfMissing -Path $file
}

# ---------------------------------------------------------
# Documentación metodológica
# ---------------------------------------------------------

$documentationFiles = @(
    "docs/metodologia/definicion_taylor.md",
    "docs/instalacion/decision_libreria.md"
)

foreach ($file in $documentationFiles) {
    New-FileIfMissing -Path $file
}

Write-Host ""
Write-Host "Estructura base creada correctamente." -ForegroundColor Green
Write-Host "No se modificaron archivos que ya existían." -ForegroundColor Green
