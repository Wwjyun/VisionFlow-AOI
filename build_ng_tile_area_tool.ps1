$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "env\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment python not found: $python"
}

$spec = Join-Path $PSScriptRoot "NG Tile Area Tool.spec"
if (-not (Test-Path $spec)) {
    throw "PyInstaller spec not found: $spec"
}

$distRoot = Join-Path $PSScriptRoot "dist\NG-Tile-Area-Tool"
$workRoot = Join-Path $PSScriptRoot "build\ng_tile_area_tool"
$readme = Join-Path $PSScriptRoot "docs\packaging\NG_TILE_AREA_TOOL_README.txt"

Push-Location $PSScriptRoot
try {
    & $python -m PyInstaller `
        --noconfirm `
        --clean `
        --distpath $distRoot `
        --workpath $workRoot `
        $spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
    Copy-Item -Force $readme $distRoot
} finally {
    Pop-Location
}

Write-Host "Built standalone utility in dist\NG-Tile-Area-Tool"
