<#
.SYNOPSIS
    Launches TerraSpectra components locally (FastAPI server or Streamlit 3D GIS dashboard).

.DESCRIPTION
    Activates the local virtual environment (.venv), optionally executes the test suite,
    and runs either the Streamlit dashboard or the FastAPI API server based on -Mode.

.PARAMETER Mode
    Component to run: "dashboard" (default) launches the Streamlit 3D GIS dashboard;
    "api" launches the FastAPI server via uvicorn.

.PARAMETER RunTests
    Switch to run unittest discovery before launching.
#>

[CmdletBinding()]
param(
    [ValidateSet("dashboard", "api")]
    [string]$Mode = "dashboard",

    [switch]$RunTests
)

$ErrorActionPreference = "Stop"

# Navigate to project root directory
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# Activate virtual environment
$VenvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    Write-Host "[TerraSpectra] Activating virtual environment..." -ForegroundColor Cyan
    & $VenvActivate
} else {
    Write-Warning "[TerraSpectra] Virtual environment not found at $VenvActivate. Running with system python."
}

# Optionally run tests
if ($RunTests) {
    Write-Host "[TerraSpectra] Executing test suite..." -ForegroundColor Cyan
    python -m unittest discover tests -v
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[TerraSpectra] Tests failed. Aborting launch."
        exit $LASTEXITCODE
    }
}

# Launch requested mode
if ($Mode -eq "api") {
    Write-Host "[TerraSpectra] Starting FastAPI server on http://127.0.0.1:8000..." -ForegroundColor Green
    uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
} else {
    Write-Host "[TerraSpectra] Starting Streamlit 3D GIS dashboard on http://localhost:8501..." -ForegroundColor Green
    python -m streamlit run app.py
}
